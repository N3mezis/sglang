"""Paged expert pager: the per-step residency decision over the K-slot GPU pool.

The pager owns *which* expert lives in *which* slot and when — a host-side keep-warm + LRU decision each
decode step — and hands the resulting ``(src_experts, dst_slots)`` plan to its ``ExpertStore``
(``store.py``), which owns the host backing and the actual byte movement (pinned ``transfer_kv`` or a
pageable copy). Slots 0..K-1 start holding experts 0..K-1 (what sglang's native loader put there);
``logical_to_gpu_index[e]`` is the slot of expert e (-1 if not resident) and its device mirror drives the
forward remap. Store fill from the checkpoint (marlin repack for gptq-int4, direct copy for bf16 — no
offline artifact) lives in ``setup_pager`` below.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, Optional

import torch

from sglang.srt.layers.moe.paged_experts.policy import (
    ResidencyPolicy,
    make_residency_policy,
)
from sglang.srt.layers.moe.paged_experts.store import ExpertStore, make_expert_store

logger = logging.getLogger(__name__)


# --- Replay-twice registry (captured pinned-window fallback) -------------------------------------------
# Each windowed layer registers its pager here and gets a slot in a shared device miss-vector. After a
# captured decode replay, the post-replay hook polls the whole vector in ONE D2H: if every layer hit its
# window (count 0) the token is correct and we stop; otherwise each missed layer stages its deferred cold
# experts into their GPU slots out-of-graph and we replay the SAME graph again (the residency maps it reads
# are fixed-address, so the next replay sees them resident). Converges in ~1 extra replay.
_REPLAY_PAGERS: list = []
_MISS_VEC: Optional[torch.Tensor] = None  # [N] int32; slot i = layer i's window-miss count this replay
_MISS_VEC_N: int = 0
_REPLAY_HOOK_INSTALLED = False


def _alloc_miss_slot(device) -> tuple:
    """Reserve this layer's slot in the shared miss-vector; returns (index, the [1] view decide writes)."""
    global _MISS_VEC, _MISS_VEC_N
    if _MISS_VEC is None:
        _MISS_VEC = torch.zeros(512, dtype=torch.int32, device=device)
    idx = _MISS_VEC_N
    _MISS_VEC_N += 1
    return idx, _MISS_VEC[idx : idx + 1]


def _post_replay_refill_all() -> bool:
    """Post-replay hook (registered with the cuda-graph backend). One scalar D2H over the shared miss-vector
    short-circuits the no-miss case; only an actual miss-step pays the per-layer count read + staging."""
    if _MISS_VEC is None or _MISS_VEC_N == 0:
        return False
    total = int(_MISS_VEC[:_MISS_VEC_N].sum().item())  # one sync; no-miss steps stop here
    if total == 0:
        return False
    counts = _MISS_VEC[:_MISS_VEC_N].tolist()
    staged = False
    for p in _REPLAY_PAGERS:
        cn = counts[p._miss_idx]
        if cn > 0 and p._refill_after_replay(cn):
            staged = True
    if staged:
        torch.cuda.synchronize()  # one device sync for all layers' staging, then replay again
        return True
    return False


def _register_replay_pager(p: "PagedExpertStore") -> None:
    global _REPLAY_HOOK_INSTALLED
    if p not in _REPLAY_PAGERS:
        _REPLAY_PAGERS.append(p)
    if not _REPLAY_HOOK_INSTALLED:
        from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
            set_post_replay_hook,
        )

        set_post_replay_hook(_post_replay_refill_all)
        _REPLAY_HOOK_INSTALLED = True


class PagedExpertStore:
    """Per-step residency decision over the K-slot pool; delegates backing + page-in to an ``ExpertStore``
    and the eviction choice to a ``ResidencyPolicy``.

    Name + positional constructor kept for back-compat: ``(layer, E, K, device, pin_host=...)`` builds the
    matching store, or pass a prebuilt ``store=`` to compose one directly (what ``setup_pager`` does).
    ``eviction`` selects the residency policy (``lru`` default | ``lfu``).
    """

    def __init__(
        self,
        layer=None,
        num_experts_E: int = 0,
        num_resident_K: int = 0,
        device=None,
        pin_host: bool = True,
        *,
        store: Optional[ExpertStore] = None,
        eviction: str = "lru",
    ):
        self.store = store or make_expert_store(
            layer, num_experts_E, num_resident_K, device, pin_host=pin_host
        )
        self.E = self.store.E
        self.K = self.store.K
        self.device = self.store.device

        # Residency state (host-side decide; the store does the device transfer). Slots 0..K-1 start
        # holding experts 0..K-1 (what the native loader put there). logical_to_gpu_index[e] is the slot
        # of expert e (-1 if not resident); its device mirror drives the remap each step. The policy owns
        # the eviction choice + its recency/frequency bookkeeping (see policy.py).
        self.policy: ResidencyPolicy = make_residency_policy(eviction, self.K, self.E)
        self.slot_expert = list(range(self.K))  # slot -> expert id (-1 == empty)
        self.logical_to_gpu_index = torch.full((self.E,), -1, dtype=torch.int32)
        self.logical_to_gpu_index[: self.K] = torch.arange(self.K, dtype=torch.int32)
        self.logical_to_gpu_index_cuda = self.logical_to_gpu_index.to(self.device)

        # On-device residency state (the captured path; allocated lazily by setup_ondevice). The decision
        # then runs in the decide kernel with NO host sync, so sglang's decode CUDA graph can capture it.
        self.ondevice = False
        self.store_devptr: Dict[str, int] = {}
        self._slot_expert_d = self._slot_lastuse_d = self._freq_d = None
        self._step_ctr_d = self._src_d = self._dst_d = self._n_out_d = None
        self._topk_i32 = None
        # Windowed (bounded) captured path — set up by setup_ondevice when the store is a WindowedExpertStore:
        # static hot/cold membership maps the decide_bounded kernel reads, the cold (deferred-miss) plan
        # buffers, the needed[] mask, and this layer's slot in the shared replay-twice miss-vector.
        self._windowed = False
        self._log2hot_d = self._log2cold_d = None
        self._cold_log_d = self._cold_dst_d = self._needed_d = self._cold_n_d = None
        self._miss_idx = -1

    # --- backing delegated to the store (exposed on the pager for the fill code + back-compat) ---
    @property
    def gpu(self) -> Dict[str, torch.Tensor]:
        return self.store.gpu

    @property
    def host(self) -> Dict[str, torch.Tensor]:
        return self.store.host

    @property
    def item_bytes(self) -> Dict[str, int]:
        return self.store.item_bytes

    @property
    def pin_host(self) -> bool:
        return self.store.pinned

    def page_in(self, src_experts: torch.Tensor, dst_slots: torch.Tensor) -> None:
        """Page the chosen experts into their slots via the store (transport-specific; a no-op if empty)."""
        self.store.page_in(src_experts, dst_slots)

    def distinct_active(self, topk_ids: torch.Tensor):
        """Sorted distinct active (>=0) expert ids this step, as a host list (one host sync)."""
        return [int(e) for e in torch.unique(topk_ids).tolist() if e >= 0]

    def decide_keep_warm(self, topk_ids: torch.Tensor, distinct=None):
        """Host-side residency decision (eager keep-warm): for each distinct active expert not resident,
        evict a non-needed slot (chosen by ``self.policy`` — LRU/LFU) and assign it. Updates the maps in
        place and returns ``(src_experts, dst_slots)`` (device int64) for ``page_in``. **Requires
        ``len(distinct) <= K``** — the caller routes steps with more distinct experts to the wave path
        (see forward.py). Data-dependent -> not capturable (the eager path).
        """
        self.policy.begin_step()
        if distinct is None:
            distinct = self.distinct_active(topk_ids)
        l2g = self.logical_to_gpu_index
        needed = set(distinct)
        for e in distinct:  # touch recency/frequency of resident hits
            s = int(l2g[e])
            if s >= 0:
                self.policy.record_use(e, s)
        src, dst = [], []
        for e in distinct:
            if int(l2g[e]) >= 0:
                continue  # already resident (or just assigned)
            victim = self.policy.pick_victim(self.slot_expert, needed)
            if victim < 0:
                continue  # pool too small (shouldn't happen: distinct <= K)
            old = self.slot_expert[victim]
            if old >= 0:
                l2g[old] = -1
            self.slot_expert[victim] = e
            l2g[e] = victim
            self.policy.record_use(e, victim)  # the fresh assignment counts as a use
            src.append(e)
            dst.append(victim)
        self.logical_to_gpu_index_cuda.copy_(l2g)
        return (
            torch.tensor(src, dtype=torch.int64, device=self.device),
            torch.tensor(dst, dtype=torch.int64, device=self.device),
        )

    def setup_ondevice(self) -> None:
        """Allocate the device-resident residency state for the captured path and resolve the pinned
        store's UVA device pointer (once, outside any graph). Requires a pinned store with 16-byte-aligned
        per-expert blocks (the gather is float4). Slots 0..K-1 start holding experts 0..K-1, matching the
        eager seeding."""
        from sglang.jit_kernel.paged_experts_decide import paged_experts_host_devptr

        assert self.pin_host, "on-device gather needs a pinned store (UVA)"
        # Windowed store (>pin-ceiling fallback): only the hot window is pinned/UVA-gatherable — the gather
        # reads host_hot; cold experts are staged out-of-graph by the replay-twice refill.
        windowed = hasattr(self.store, "host_hot")
        for name, sz in self.item_bytes.items():
            if sz % 16 != 0:
                raise RuntimeError(
                    f"[paged-experts] on-device gather needs 16-byte-aligned per-expert blocks; "
                    f"{name!r} is {sz} B. Use --disable-cuda-graph (eager transfer_kv path)."
                )
            src = self.store.host_hot[name] if windowed else self.host[name]
            self.store_devptr[name] = paged_experts_host_devptr(src)

        dev = self.device
        i32 = torch.int32
        self._slot_expert_d = torch.arange(self.K, dtype=i32, device=dev)
        self._slot_lastuse_d = torch.zeros(self.K, dtype=i32, device=dev)
        self._freq_d = torch.zeros(self.E, dtype=i32, device=dev)
        self._step_ctr_d = torch.zeros(1, dtype=i32, device=dev)
        self._src_d = torch.zeros(self.K, dtype=i32, device=dev)
        self._dst_d = torch.zeros(self.K, dtype=i32, device=dev)
        self._n_out_d = torch.zeros(1, dtype=i32, device=dev)
        # expert_slot and idx are the same buffer the forward remap reads (logical_to_gpu_index_cuda).
        self.ondevice = True
        if windowed:
            self._setup_window_ondevice()

    def _setup_window_ondevice(self) -> None:
        """Device state for the captured windowed (bounded) path: the static hot/cold membership maps the
        decide_bounded kernel reads, the deferred-cold plan buffers + needed[] mask, and this layer's slot
        in the shared replay-twice miss-vector (registers the post-replay hook on first call)."""
        dev, i32 = self.device, torch.int32
        self._windowed = True
        # log2hot[e] = hot-block index (or -1 if cold); log2cold[e] = cold-block index (or -1 if hot).
        self._log2hot_d = self.store.hot_pos.to(dtype=i32, device=dev)
        self._log2cold_d = self.store.cold_pos.to(dtype=i32, device=dev)
        self._cold_log_d = torch.zeros(self.K, dtype=i32, device=dev)
        self._cold_dst_d = torch.zeros(self.K, dtype=i32, device=dev)
        self._needed_d = torch.zeros(self.K, dtype=i32, device=dev)
        self._miss_idx, self._cold_n_d = _alloc_miss_slot(dev)
        _register_replay_pager(self)

    def _prep_topk_ondevice(self, topk_ids: torch.Tensor) -> None:
        """Copy the router's topk ids into the persistent int32 buffer the kernels read (casts int64 ->
        int32; capture-safe). Allocated once at the captured shape, reused across replays.
        """
        flat = topk_ids.reshape(-1)
        if self._topk_i32 is None or self._topk_i32.numel() != flat.numel():
            self._topk_i32 = torch.empty(
                flat.numel(), dtype=torch.int32, device=self.device
            )
        self._topk_i32.copy_(flat)

    def _gather_planned_ondevice(self) -> None:
        """Gather the experts the last decide chose (``_src_d`` -> ``_dst_d``, count ``_n_out_d``) from the
        pinned store into the GPU pool, for every paged tensor. Count read on-device -> capture-safe.
        """
        from sglang.jit_kernel.paged_experts_decide import paged_experts_gather

        for name, gpu_param in self.gpu.items():
            paged_experts_gather(
                self.store_devptr[name],
                gpu_param.data,
                self._src_d,
                self._dst_d,
                self._n_out_d,
                self.item_bytes[name],
            )

    def decide_and_page_ondevice(self, topk_ids: torch.Tensor) -> None:
        """Capture-safe keep-warm: decide residency + page the misses entirely on-device (no host sync).
        Mutates the persistent state buffers and ``logical_to_gpu_index_cuda`` (the remap table) in place;
        gathers exactly the chosen experts. Requires distinct active experts <= K (the caller guarantees it
        via the shape guard ``num_tokens * top_k <= K``)."""
        from sglang.jit_kernel.paged_experts_decide import paged_experts_decide

        self._prep_topk_ondevice(topk_ids)
        l2g = self.logical_to_gpu_index_cuda  # serves as both expert_slot and idx
        paged_experts_decide(
            self._topk_i32,
            self._step_ctr_d,
            self._slot_expert_d,
            l2g,
            self._slot_lastuse_d,
            self._freq_d,
            False,  # LRU (matches the host keep-warm path)
            self._src_d,
            self._dst_d,
            self._n_out_d,
            l2g,
        )
        self._gather_planned_ondevice()

    def decide_and_page_bounded_ondevice(self, topk_ids: torch.Tensor) -> None:
        """Capture-safe windowed keep-warm (the >pin-ceiling fallback). ``decide_bounded`` splits the plan by
        window membership: window hits gather in-graph from the pinned ``host_hot`` (capture-safe); cold
        (window-missing) experts are deferred — their logical ids land in ``_cold_log_d`` and they stay
        masked this replay — for the post-replay refill to stage and converge. Requires distinct active <= K.
        """
        from sglang.jit_kernel.paged_experts_decide import paged_experts_decide_bounded

        self._prep_topk_ondevice(topk_ids)
        l2g = self.logical_to_gpu_index_cuda  # serves as both expert_slot and idx
        paged_experts_decide_bounded(
            self._topk_i32,
            self._step_ctr_d,
            self._slot_expert_d,
            l2g,
            self._slot_lastuse_d,
            self._freq_d,
            False,  # LRU
            True,  # defer_cold: replay-twice (pageable-RAM cold tier, not gatherable in-graph)
            self._log2hot_d,
            self._log2cold_d,
            self._src_d,
            self._dst_d,
            self._n_out_d,
            self._cold_log_d,
            self._cold_dst_d,
            self._cold_n_d,
            l2g,
            self._needed_d,
        )
        self._gather_planned_ondevice()  # window hits only; cold misses deferred to _refill_after_replay

    def _refill_after_replay(self, cn: int) -> bool:
        """Post-replay (out-of-graph): stage the ``cn`` deferred cold experts ``host_cold`` -> their GPU
        slots and mark them resident, so the next replay's ``decide_bounded`` sees them as hits and the loop
        converges (typically one extra replay). Evicts only slots NOT needed this step (the ``needed[]`` mask
        decide emits), so a still-needed expert is never displaced. Eager copies; the hook syncs once for all
        layers before replaying again."""
        missed = self._cold_log_d[:cn].tolist()  # logical ids (defer mode emits logical ids)
        needed = self._needed_d.tolist()
        evictable = [s for s in range(self.K) if not needed[s]]
        if len(evictable) < cn:
            logger.error(
                "[paged-experts] replay-twice: %d cold misses but only %d evictable slots (K=%d) — "
                "keep-warm invariant (distinct <= K) broken",
                cn,
                len(evictable),
                self.K,
            )
            return False
        l2g = self.logical_to_gpu_index_cuda
        se = self._slot_expert_d.tolist()
        for e, slot in zip(missed, evictable):
            old = se[slot]
            for name, gpu_param in self.gpu.items():  # eager stage host_cold -> GPU slot (out of graph)
                gpu_param.data[slot].copy_(self.store.row(name, e), non_blocking=True)
            if old >= 0:  # update device residency so replay-2's decide sees e resident in `slot`
                l2g[old] = -1
            self._slot_expert_d[slot] = e
            l2g[e] = slot
        return True

    def decide_and_page_wave_ondevice(self, topk_ids: torch.Tensor, wave: int) -> None:
        """One static wave (distinct > K, e.g. prefill): plan + gather the in-wave experts on-device. The
        caller runs ceil(E/K) waves and sums the per-wave GEMM partials, then calls
        ``resync_residency_ondevice`` so the keep-warm state matches the slots."""
        from sglang.jit_kernel.paged_experts_decide import paged_experts_decide_wave

        if wave == 0:
            self._prep_topk_ondevice(topk_ids)
        paged_experts_decide_wave(
            self._topk_i32,
            self.E,
            self.K,
            wave,
            self._src_d,
            self._dst_d,
            self._n_out_d,
            self.logical_to_gpu_index_cuda,
        )
        self._gather_planned_ondevice()

    def resync_residency_ondevice(self, last_wave: int) -> None:
        """After the wave loop the slots physically hold wave ``last_wave``'s experts. Point the device
        keep-warm state at that so the next decode step is consistent (``logical_to_gpu_index_cuda`` was
        already set to this wave by the last ``decide_wave``)."""
        lo = last_wave * self.K
        ngrp = min(self.K, self.E - lo)
        idx = torch.arange(lo, lo + ngrp, dtype=torch.int32, device=self.device)
        self._slot_expert_d[:ngrp] = idx
        if ngrp < self.K:
            self._slot_expert_d[ngrp:] = -1
        self._slot_lastuse_d.zero_()

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
    store: ExpertStore, model_path: str, layer_idx: int
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
        expected = (store.E, *store.gpu[name].shape[1:])
        assert tuple(t.shape) == expected, (name, t.shape, expected)
        store.fill_tensor(name, t)


def _fill_bf16_from_checkpoint(
    store: ExpertStore, model_path: str, layer_idx: int
) -> None:
    """bf16: host w13_weight=[E,2*inter,hidden]=concat(gate,up), w2_weight=[E,hidden,inter]."""
    from safetensors import safe_open

    snap = _snapshot_dir(model_path)
    wmap = _weight_map(snap)
    pre = f"model.layers.{layer_idx}.mlp.experts."
    dt = store.gpu["w13_weight"].dtype
    by_shard: Dict[str, list] = {}
    for e in range(store.E):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            by_shard.setdefault(wmap[f"{pre}{e}.{proj}.weight"], []).append((e, proj))
    for shard, items in by_shard.items():
        with safe_open(os.path.join(snap, shard), framework="pt") as f:
            for e, proj in items:
                t = f.get_tensor(f"{pre}{e}.{proj}.weight").to(dt)
                if proj == "down_proj":
                    store.row("w2_weight", e).copy_(t)
                    continue
                # w13 packs gate (first half of dim 0) then up (second half)
                row = store.row("w13_weight", e)
                half = row.shape[0] // 2
                if proj == "gate_proj":
                    row[:half].copy_(t)
                else:  # up_proj
                    row[half:].copy_(t)


def setup_pager(method, layer) -> PagedExpertStore:
    """Build the host store and fill it from the checkpoint (all E experts), then return the pager wrapping
    it. ``method`` carries E, K, and the resident map. gptq-int4 is repacked to marlin at load time; bf16 is
    copied directly. No offline artifact."""
    from sglang.srt.server_args import get_global_server_args

    dev = next(layer.parameters()).device
    store = make_expert_store(
        layer,
        method.E,
        method.num_resident,
        dev,
        pin_host=getattr(method, "pin_host", True),
        window_W=getattr(method, "window", 0),
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
    pager = PagedExpertStore(store=store, eviction=getattr(method, "eviction", "lru"))
    if method._placement.needs_ondevice_store:
        pager.setup_ondevice()  # captured path: device-resident decide + UVA gather
    return pager
