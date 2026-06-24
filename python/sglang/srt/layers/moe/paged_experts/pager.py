"""Paged expert store + page-in, built on sglang's existing host<->device transfer kernel.

The K-slot GPU pool *is* the layer's expert params (the native loader filled slots 0..K-1). We allocate a
pinned host store holding ALL E experts per paged tensor and fill it from the checkpoint (repacked to the
marlin layout for gptq-int4, copied directly for bf16) — no offline artifact. On a miss, ``page_in`` copies
expert rows into their slots with
``transfer_kv_per_layer_mla`` (device-indexed, dynamic count, capture-safe) — no custom CUDA. ``item_size``
is the per-expert row in bytes, so one call per tensor moves the chosen experts.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict

import torch

logger = logging.getLogger(__name__)

# packed-quant scaffolding the fused-MoE kernel never reads on the paged path
_NONPAGED_SUFFIXES = ("_g_idx", "_g_idx_sort_indices", "_weight_shape")


def discover_paged_params(layer, num_slots: int) -> Dict[str, torch.Tensor]:
    """Per-expert params on ``layer``: leading dim == num_slots (the K-slot pool) and non-empty per-slot."""
    out = {}
    for name, p in list(layer.named_parameters(recurse=False)) + list(
        layer.named_buffers(recurse=False)
    ):
        if any(name.endswith(s) for s in _NONPAGED_SUFFIXES):
            continue
        if p.dim() >= 1 and p.shape[0] == num_slots and p[0].numel() > 0:
            out[name] = p
    return out


class PagedExpertStore:
    def __init__(
        self,
        layer,
        num_experts_E: int,
        num_resident_K: int,
        device,
        pin_host: bool = True,
    ):
        self.E = num_experts_E
        self.K = num_resident_K
        self.device = device
        # Pinned store -> fast transfer_kv page-in. Pageable store -> a plain indexed copy (see page_in):
        # correct but slower, for when the pinned store would exceed the host page-locked limit.
        self.pin_host = pin_host
        self.gpu = discover_paged_params(
            layer, num_resident_K
        )  # the K-slot GPU pool (layer params)
        assert self.gpu, "no per-expert params found on layer"
        self.host: Dict[str, torch.Tensor] = {}
        self.item_bytes: Dict[str, int] = {}
        for name, p in self.gpu.items():
            self.host[name] = torch.empty(
                (self.E, *p.shape[1:]), dtype=p.dtype, device="cpu", pin_memory=pin_host
            )
            self.item_bytes[name] = p[0].numel() * p.element_size()
            # transfer_kv_per_layer_mla requires the per-expert block to be 8-byte aligned. Real
            # weight rows (bf16 / marlin qweight+scales+qzeros) satisfy this; a 1-D per-expert scalar
            # scale (e.g. fp8, 4 B) does not -> that needs the deferred scalar-gather path. The pageable
            # copy path has no such requirement.
            if pin_host and self.item_bytes[name] % 8 != 0:
                raise RuntimeError(
                    f"[paged-experts] paged tensor {name!r} per-expert size {self.item_bytes[name]} B is "
                    "not 8-byte aligned (transfer_kv requirement); unsupported on the reuse gather path."
                )

        # Eager residency state (host-side decide; page_in does the device transfer). Slots 0..K-1
        # start holding experts 0..K-1 (what the native loader put there). logical_to_gpu_index[e] is
        # the slot of expert e (-1 if not resident); its device mirror drives the remap each step.
        self.slot_expert = list(range(self.K))  # slot -> expert id (-1 == empty)
        self.slot_lastuse = [0] * self.K
        self.logical_to_gpu_index = torch.full((self.E,), -1, dtype=torch.int32)
        self.logical_to_gpu_index[: self.K] = torch.arange(self.K, dtype=torch.int32)
        self.logical_to_gpu_index_cuda = self.logical_to_gpu_index.to(device)
        self._step = 0

    def distinct_active(self, topk_ids: torch.Tensor):
        """Sorted distinct active (>=0) expert ids this step, as a host list (one host sync)."""
        return [int(e) for e in torch.unique(topk_ids).tolist() if e >= 0]

    def decide_keep_warm(self, topk_ids: torch.Tensor, distinct=None):
        """Host-side residency decision (eager keep-warm + LRU): for each distinct active expert not
        resident, evict the LRU non-needed slot and assign it. Updates the maps in place and returns
        ``(src_experts, dst_slots)`` (device int64) for ``page_in``. **Requires ``len(distinct) <= K``**
        — the caller routes steps with more distinct experts to the wave path (see forward.py). Data-
        dependent -> not capturable (the eager path).
        """
        self._step += 1
        step = self._step
        if distinct is None:
            distinct = self.distinct_active(topk_ids)
        l2g = self.logical_to_gpu_index
        needed = set(distinct)
        for e in distinct:  # touch recency of resident hits
            s = int(l2g[e])
            if s >= 0:
                self.slot_lastuse[s] = step
        src, dst = [], []
        for e in distinct:
            if int(l2g[e]) >= 0:
                continue  # already resident (or just assigned)
            victim, best_lu = -1, None  # LRU non-needed slot
            for s in range(self.K):
                if self.slot_expert[s] in needed:
                    continue
                if best_lu is None or self.slot_lastuse[s] < best_lu:
                    best_lu, victim = self.slot_lastuse[s], s
            if victim < 0:
                continue  # pool too small (shouldn't happen: distinct <= K)
            old = self.slot_expert[victim]
            if old >= 0:
                l2g[old] = -1
            self.slot_expert[victim] = e
            l2g[e] = victim
            self.slot_lastuse[victim] = step
            src.append(e)
            dst.append(victim)
        self.logical_to_gpu_index_cuda.copy_(l2g)
        return (
            torch.tensor(src, dtype=torch.int64, device=self.device),
            torch.tensor(dst, dtype=torch.int64, device=self.device),
        )

    def page_in(self, src_experts: torch.Tensor, dst_slots: torch.Tensor) -> None:
        """Copy ``host[src_experts[i]] -> gpu[dst_slots[i]]`` for every paged tensor.

        Pinned store: reuse sglang's ``transfer_kv_per_layer_mla`` (pinned-host -> device, indices read
        on-device, dynamic count). Pageable store: a plain indexed copy (gather rows on the host, one H2D,
        scatter into the slots) — ``transfer_kv`` would read stale data from non-page-locked memory.
        """
        if src_experts.numel() == 0:
            return
        if self.pin_host:
            from sgl_kernel import transfer_kv_per_layer_mla

            for name, gpu_param in self.gpu.items():
                transfer_kv_per_layer_mla(
                    src=self.host[name],
                    dst=gpu_param.data,
                    src_indices=src_experts,
                    dst_indices=dst_slots,
                    item_size=self.item_bytes[name],
                )
        else:
            src_cpu = src_experts.to("cpu")
            for name, gpu_param in self.gpu.items():
                rows = self.host[name].index_select(0, src_cpu).to(gpu_param.device)
                gpu_param.data.index_copy_(0, dst_slots, rows)

    def set_residency(self, experts) -> None:
        """Force slot ``i`` to hold ``experts[i]`` and rebuild the maps. Called after the wave path so
        the next keep-warm step's residency state matches what is physically in the slots.
        """
        experts = list(experts)
        self.slot_expert = experts + [-1] * (self.K - len(experts))
        self.logical_to_gpu_index.fill_(-1)
        for i, e in enumerate(experts):
            self.logical_to_gpu_index[e] = i
        self.logical_to_gpu_index_cuda.copy_(self.logical_to_gpu_index)


def _snapshot_dir(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    from huggingface_hub import snapshot_download

    return snapshot_download(model_path, local_files_only=True)


def _weight_map(snap: str) -> Dict[str, str]:
    """{tensor_name: shard_file}; falls back to the single .safetensors when there's no index.json
    (small/quantized checkpoints are often one file)."""
    import glob

    idx = os.path.join(snap, "model.safetensors.index.json")
    if os.path.exists(idx):
        return json.load(open(idx))["weight_map"]
    from safetensors import safe_open

    files = glob.glob(os.path.join(snap, "*.safetensors"))
    assert len(files) == 1, f"no index.json and != 1 safetensors shard: {files}"
    with safe_open(files[0], framework="pt") as f:
        return {k: os.path.basename(files[0]) for k in f.keys()}


def _fill_gptq_marlin_from_checkpoint(
    store: PagedExpertStore, model_path: str, layer_idx: int
) -> None:
    """gptq-int4: repack the GPTQ checkpoint into the on-GPU marlin layout for ALL E experts, using
    sglang's own ops, straight into the host store. sglang's loader repacks only the K resident slots
    (num_local_experts=K); we repack all E so the paged experts match. This is the per-layer repack the
    offline builder did, moved to load time -> no offline store artifact needed. (At runtime the
    quantization package is already imported, so the gptq_kernels/wNa16 circular import doesn't apply.)
    """
    from safetensors import safe_open

    # Load the quantization package fully before importing gptq_kernels directly — gptq_kernels and
    # compressed_tensors_wNa16_moe form an import cycle that only fails when gptq_kernels is the entry
    # point. At server runtime it is already imported; this makes the order-independent too.
    import sglang.srt.layers.quantization  # noqa: F401
    from sglang.srt.hardware_backend.gpu.quantization.gptq_kernels import (
        gptq_marlin_moe_repack,
    )
    from sglang.srt.layers.quantization.marlin_utils import marlin_moe_permute_scales

    snap = _snapshot_dir(model_path)
    cfg = json.load(open(os.path.join(snap, "config.json")))
    tcfg = cfg.get("text_config", cfg)
    inter = tcfg["moe_intermediate_size"]
    qc = cfg["quantization_config"]
    bits, group = qc["bits"], qc["group_size"]
    pack = 32 // bits
    assert not qc.get(
        "desc_act", False
    ), "desc_act=True needs g_idx paging (unsupported)"
    wmap = _weight_map(snap)
    pre = f"model.layers.{layer_idx}.mlp.experts."
    dev = store.device

    open_shards: Dict[str, object] = {}

    def get(name: str) -> torch.Tensor:
        sh = wmap[name]
        if sh not in open_shards:
            open_shards[sh] = safe_open(os.path.join(snap, sh), framework="pt")
        return open_shards[sh].get_tensor(name)

    w13_qw, w2_qw, w13_s, w2_s, w13_qz, w2_qz = [], [], [], [], [], []
    for e in range(store.E):
        p = f"{pre}{e}."
        w13_qw.append(
            torch.cat([get(p + "gate_proj.qweight"), get(p + "up_proj.qweight")], dim=1)
        )
        w2_qw.append(get(p + "down_proj.qweight"))
        w13_s.append(
            torch.cat([get(p + "gate_proj.scales"), get(p + "up_proj.scales")], dim=1)
        )
        w2_s.append(get(p + "down_proj.scales"))
        w13_qz.append(
            torch.cat([get(p + "gate_proj.qzeros"), get(p + "up_proj.qzeros")], dim=1)
        )
        w2_qz.append(get(p + "down_proj.qzeros"))
    w13_qw, w2_qw = torch.stack(w13_qw).to(dev), torch.stack(w2_qw).to(dev)
    w13_s, w2_s = torch.stack(w13_s).to(dev), torch.stack(w2_s).to(dev)
    sort = torch.empty((store.E, 0), dtype=torch.int32, device=dev)
    marlin = {
        "w13_qweight": gptq_marlin_moe_repack(
            w13_qw, sort, w13_qw.shape[1] * pack, w13_qw.shape[2], bits
        ),
        "w2_qweight": gptq_marlin_moe_repack(
            w2_qw, sort, w2_qw.shape[1] * pack, w2_qw.shape[2], bits
        ),
        "w13_scales": marlin_moe_permute_scales(
            s=w13_s, size_k=inter, size_n=w13_s.shape[2], group_size=group
        ),
        "w2_scales": marlin_moe_permute_scales(
            s=w2_s, size_k=w2_s.shape[1] * group, size_n=w2_s.shape[2], group_size=group
        ),
        "w13_qzeros": torch.stack(w13_qz),  # carried unrepacked (sym); kernel ignores
        "w2_qzeros": torch.stack(w2_qz),
    }
    for name in store.gpu:
        t = marlin[name].contiguous().cpu()
        assert tuple(t.shape) == tuple(store.host[name].shape), (
            name,
            t.shape,
            store.host[name].shape,
        )
        store.host[name].copy_(t)


def _fill_bf16_from_checkpoint(
    store: PagedExpertStore, model_path: str, layer_idx: int
) -> None:
    """bf16: host w13_weight=[E,2*inter,hidden]=concat(gate,up), w2_weight=[E,hidden,inter]."""
    from safetensors import safe_open

    snap = _snapshot_dir(model_path)
    wmap = _weight_map(snap)
    pre = f"model.layers.{layer_idx}.mlp.experts."
    dt = store.host["w13_weight"].dtype
    by_shard: Dict[str, list] = {}
    for e in range(store.E):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            by_shard.setdefault(wmap[f"{pre}{e}.{proj}.weight"], []).append((e, proj))
    for shard, items in by_shard.items():
        with safe_open(os.path.join(snap, shard), framework="pt") as f:
            for e, proj in items:
                t = f.get_tensor(f"{pre}{e}.{proj}.weight").to(dt)
                if proj == "down_proj":
                    store.host["w2_weight"][e].copy_(t)
                    continue
                # w13 packs gate (first half of dim 0) then up (second half)
                row = store.host["w13_weight"][e]
                half = row.shape[0] // 2
                if proj == "gate_proj":
                    row[:half].copy_(t)
                else:  # up_proj
                    row[half:].copy_(t)


def setup_pager(method, layer) -> PagedExpertStore:
    """Build the host store and fill it from the checkpoint (all E experts), then return the pager.
    ``method`` carries E, K, and the resident map. gptq-int4 is repacked to marlin at load time; bf16 is
    copied directly. No offline artifact."""
    from sglang.srt.server_args import get_global_server_args

    dev = next(layer.parameters()).device
    store = PagedExpertStore(
        layer,
        method.E,
        method.num_resident,
        dev,
        pin_host=getattr(method, "pin_host", True),
    )

    layer_idx = getattr(layer, "layer_id", getattr(layer, "layer_idx", 0))
    model_path = get_global_server_args().model_path
    if any(n.endswith("qweight") for n in store.gpu):  # gptq-marlin int4
        _fill_gptq_marlin_from_checkpoint(store, model_path, layer_idx)
    elif "w13_weight" in store.gpu:  # bf16
        _fill_bf16_from_checkpoint(store, model_path, layer_idx)
    else:
        raise RuntimeError(
            f"[paged-experts] no fill for params {list(store.gpu)} "
            "(supported: gptq-marlin int4, unquantized bf16)"
        )
    logger.debug(
        "[paged-experts] L%d host store filled: E=%d, %d tensors %s",
        layer_idx,
        store.E,
        len(store.gpu),
        list(store.gpu),
    )
    return store
