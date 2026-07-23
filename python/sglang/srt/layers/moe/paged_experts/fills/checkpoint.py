"""Checkpoint-reading helpers shared by every ExpertFill: locate the snapshot, map tensor->shard,
find the per-layer expert prefix and the (gate,up,down) proj stems. Extracted verbatim from pager.py
so the fills (fills/*.py) share one copy. Also the config quant_method reader used to disambiguate
fills that produce the same store keys (gptq vs awq marlin)."""

import json
import os
from typing import Dict


def _drop_file_cache(path: str) -> None:
    """Best-effort ``POSIX_FADV_DONTNEED`` on a checkpoint shard AFTER its experts have been copied into
    the store. ``safe_open`` mmaps each shard, so without this the read pages accumulate in the OS page
    cache across all shards/layers — up to the FULL model size — alongside the (separate) host store,
    doubling peak RAM during load. Dropping each shard as it's consumed keeps the source-side cache to
    ~one shard. Each layer reads DISJOINT byte ranges of a shared shard, so this drops almost nothing
    another layer reuses (only bounded read-ahead). Linux-only; a no-op where posix_fadvise is
    unavailable."""
    fadvise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    if fadvise is None or dontneed is None:
        return
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            fadvise(fd, 0, 0, dontneed)  # (offset=0, len=0) => whole file
        finally:
            os.close(fd)
    except OSError:
        pass


def _snapshot_dir(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    from huggingface_hub import snapshot_download

    return snapshot_download(model_path, local_files_only=True)


def _weight_map(snap: str) -> Dict[str, str]:
    """{tensor_name: shard_file}; falls back to the single .safetensors when there's no index.json
    (small/quantized checkpoints are often one file)."""
    import glob

    for idx_name in (
        "model.safetensors.index.json",
        "consolidated.safetensors.index.json",
    ):
        idx = os.path.join(snap, idx_name)
        if os.path.exists(idx):
            return json.load(open(idx))["weight_map"]
    from safetensors import safe_open

    files = glob.glob(os.path.join(snap, "*.safetensors"))
    assert len(files) == 1, f"no index.json and != 1 safetensors shard: {files}"
    with safe_open(files[0], framework="pt") as f:
        return {k: os.path.basename(files[0]) for k in f.keys()}


def _experts_prefix(wmap: Dict[str, str], layer_idx: int) -> str:
    """The checkpoint name prefix of this layer's routed experts. Text-only checkpoints use
    ``model.layers.N.mlp.experts.``; VL checkpoints (e.g. Qwen3.5/3.6 MoE) nest the text model under
    ``model.language_model.``. Probed against the weight map so new nestings fail loudly.
    """
    for pre in (
        f"model.layers.{layer_idx}.mlp.experts.",
        f"model.language_model.layers.{layer_idx}.mlp.experts.",
        # Gemma-4 VL MoE: text tower nested under model.language_model, experts directly under the
        # layer (no .mlp.) as stacked fused tensors (experts.gate_up_proj / experts.down_proj).
        f"model.language_model.layers.{layer_idx}.experts.",
        # Mixtral / Mistral MoE: experts under block_sparse_moe (proj stems w1/w3/w2).
        f"model.layers.{layer_idx}.block_sparse_moe.experts.",
        # Mistral consolidated native layout (Mistral-Small-4 nvfp4): no model./mlp. nesting.
        f"layers.{layer_idx}.experts.",
        # DeepSeek-V4: routed experts under .ffn.experts. (proj stems w1/w3/w2 = gate/up/down),
        # weights + .scale (mxfp4 int8-packed + e8m0). Raw checkpoint keys carry no model. prefix.
        f"model.layers.{layer_idx}.ffn.experts.",
        f"layers.{layer_idx}.ffn.experts.",
    ):
        if any(
            k.startswith(pre) for k in (wmap.keys() if hasattr(wmap, "keys") else wmap)
        ):
            return pre
    raise RuntimeError(
        f"[paged-experts] no expert tensors found for layer {layer_idx} under known prefixes "
        "(model.layers. / model.language_model.layers. / layers.) — unsupported checkpoint layout."
    )


# proj naming: HF layouts use gate/up/down_proj; Mistral consolidated uses w1/w3/w2.
def _proj_names(wmap, pre: str) -> tuple:
    """Return (gate, up, down) proj tensor-name stems present under ``pre`` for expert 0."""
    keys = wmap.keys() if hasattr(wmap, "keys") else wmap
    have = {k[len(pre) + 2 :].split(".")[0] for k in keys if k.startswith(pre + "0.")}
    if {"w1", "w2", "w3"} <= have:
        return ("w1", "w3", "w2")  # gate, up, down (Mistral)
    return ("gate_proj", "up_proj", "down_proj")


def checkpoint_quant_method(model_path: str) -> str:
    """Lowercased ``quantization_config.quant_method`` from the checkpoint config (``""`` if
    unquantized). Used by the fill registry to split fills that register the SAME store keys — gptq
    and awq marlin both produce w13_qweight/scales/qzeros, but need different repacks.
    """
    snap = _snapshot_dir(model_path)
    cfg = json.load(open(os.path.join(snap, "config.json")))
    qc = cfg.get("quantization_config") or {}
    return (qc.get("quant_method") or "").lower()
