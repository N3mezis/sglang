"""On-disk cache of the repacked (gptq-marlin) host store: the repack is deterministic, so persisting
it turns every later boot's read-checkpoint-and-repack into a straight sequential read. Only the marlin
store is cached (bf16/fp8 fills are already direct byte copies). Extracted verbatim from pager.py.
"""

import functools
import json
import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_STORE_CACHE_VERSION = (
    2  # v2: gptq store may include paged g_idx/sort_indices (desc_act)
)
_STORE_CACHE_LOGGED = False


@functools.lru_cache(maxsize=8)
def _store_cache_dir(model_path: str) -> Optional[str]:
    """Cache directory for the REPACKED (gptq-marlin) host store, keyed by checkpoint identity +
    layout version — the repack is deterministic, so persisting it turns every later boot's
    read-checkpoint-and-repack into a straight sequential read. Only the marlin store is cached: the
    bf16/fp8 fills are already direct copies of checkpoint bytes, so a cache would just duplicate
    them on disk. Lives under the HF cache (mounted wherever the checkpoint cache is). Delete the
    cache directory to force a fresh repack."""
    import glob
    import hashlib

    folder = model_path
    if not os.path.isdir(folder):
        try:
            from huggingface_hub import snapshot_download

            folder = snapshot_download(model_path, local_files_only=True)
        except Exception:
            return None
    h = hashlib.sha256(f"paged-experts-store-v{_STORE_CACHE_VERSION}".encode())
    try:
        for fn in (
            "config.json",
            "model.safetensors.index.json",
            "quantize_config.json",
        ):
            fp = os.path.join(folder, fn)
            if os.path.exists(fp):
                with open(fp, "rb") as f:
                    h.update(f.read())
        for fp in sorted(glob.glob(os.path.join(folder, "*.safetensors"))):
            st = os.stat(fp)
            # size AND mtime: an updated checkpoint (e.g. an RL loop rewriting shards in place) keeps
            # names/shapes/sizes — without the mtime the digest would collide and serve stale experts
            h.update(os.path.basename(fp).encode())
            h.update(f"{st.st_size}:{st.st_mtime_ns}".encode())
    except Exception:
        return None
    root = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    return os.path.join(root, "paged_experts_cache", f"store-{h.hexdigest()[:16]}")


def _store_v2_layer_dir(cache_dir: Optional[str], layer_idx: int) -> Optional[str]:
    """Per-layer directory of the v2 RAW store cache (one page-aligned .bin per paged tensor +
    manifest.json) — the mmap+register (zero-copy) store's format. v1 (safetensors) stays readable for
    the copy path; v2 exists because safetensors tensor offsets are only 8-byte aligned, while the UVA
    gather needs a 16-byte-aligned base — a per-tensor file's mmap base is page-aligned by construction."""
    return os.path.join(cache_dir, "v2", f"layer_{layer_idx}") if cache_dir else None

def _v2_cache_complete(layer_dir: Optional[str], names) -> bool:
    if not layer_dir or not os.path.exists(os.path.join(layer_dir, "manifest.json")):
        return False
    return all(os.path.exists(os.path.join(layer_dir, f"{n}.bin")) for n in names)

def _save_store_v2(store, cache_dir: Optional[str], layer_idx: int) -> None:
    """Persist the filled host store as v2 raw per-tensor files (atomic per file), so the NEXT boot can
    mmap+register it in place (--paged-experts-store mmap) instead of re-filling."""
    layer_dir = _store_v2_layer_dir(cache_dir, layer_idx)
    if not layer_dir:
        return
    try:
        os.makedirs(layer_dir, exist_ok=True)
        manifest = {}
        for name, p in store.gpu.items():
            host = store.host.get(name) if getattr(store, "host", None) is not None else None
            if host is None:  # windowed store: reconstruct expert order via the fill accessors
                host = torch.empty((store.E, *p.shape[1:]), dtype=p.dtype)
                for e in range(store.E):
                    host[e].copy_(store.row(name, e))
            raw = host.contiguous().view(torch.uint8).numpy()  # byte reinterpretation (bf16/fp8-safe)
            path = os.path.join(layer_dir, f"{name}.bin")
            with open(path + ".tmp", "wb") as f:
                f.write(raw.tobytes())
            os.replace(path + ".tmp", path)
            manifest[name] = {
                "shape": list(host.shape),
                "dtype": str(host.dtype),
                "nbytes": host.numel() * host.element_size(),
            }
        mpath = os.path.join(layer_dir, "manifest.json")
        with open(mpath + ".tmp", "w") as f:
            json.dump(manifest, f)
        os.replace(mpath + ".tmp", mpath)
    except Exception as e:
        logger.warning(
            "[paged-experts] v2 store cache write failed for layer %d (%s) — boot unaffected",
            layer_idx,
            e,
        )


def _fill_store_from_cache(store, cache_dir: Optional[str], layer_idx: int) -> bool:
    """Fill the host store for one layer from the repack cache. Returns False (caller refills from
    the checkpoint) on any mismatch — missing file, different tensor set, shape/dtype drift, torn
    write."""
    if not cache_dir:
        return False
    path = os.path.join(cache_dir, f"layer_{layer_idx}.safetensors")
    if not os.path.exists(path):
        return False
    try:
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as f:
            if set(f.keys()) != set(store.gpu):
                return False
            for name, p in store.gpu.items():
                t = f.get_tensor(name)
                if tuple(t.shape) != (store.E, *p.shape[1:]) or t.dtype != p.dtype:
                    return False
                store.fill_tensor(name, t)
        return True
    except Exception as e:
        logger.warning(
            "[paged-experts] store cache read failed for layer %d (%s) — refilling from checkpoint",
            layer_idx,
            e,
        )
        return False


def _save_store_to_cache(store, cache_dir: Optional[str], layer_idx: int) -> None:
    if not cache_dir:
        return
    try:
        from safetensors.torch import save_file

        os.makedirs(cache_dir, exist_ok=True)
        tensors = {}
        host = getattr(store, "host", None)
        for name, p in store.gpu.items():
            if host is not None and name in host:
                tensors[name] = host[name]
            else:  # windowed store: reconstruct expert order via the fill accessors
                full = torch.empty((store.E, *p.shape[1:]), dtype=p.dtype)
                for e in range(store.E):
                    full[e].copy_(store.row(name, e))
                tensors[name] = full
        path = os.path.join(cache_dir, f"layer_{layer_idx}.safetensors")
        save_file(tensors, path + ".tmp")
        os.replace(path + ".tmp", path)
    except Exception as e:
        logger.warning(
            "[paged-experts] store cache write failed for layer %d (%s) — boot unaffected",
            layer_idx,
            e,
        )
