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

import functools
import json
import logging
import os
import time
from typing import Dict, Optional

import torch

# Per-quant-format expert-store fills (registry + config quant_method reader). The gptq repack cache
# lives under fills too; re-exported here because method.py imports _store_cache_dir from pager.
from sglang.srt.layers.moe.paged_experts.fills import select_fill  # noqa: E402
from sglang.srt.layers.moe.paged_experts.fills.cache import (  # noqa: E402,F401
    _store_cache_dir,
)
from sglang.srt.layers.moe.paged_experts.fills.checkpoint import (  # noqa: E402
    checkpoint_quant_method,
)
from sglang.srt.layers.moe.paged_experts.policy import (
    ResidencyPolicy,
    make_residency_policy,
)
from sglang.srt.layers.moe.paged_experts.store import ExpertStore, make_expert_store

logger = logging.getLogger(__name__)

# ALL pagers in model-layer order (appended by setup_pager) — the wave path's next-layer prefetch.
_ALL_PAGERS: list = []


# --- Replay-twice registry (captured pinned-window fallback) -------------------------------------------
# Each windowed layer registers its pager here and gets a slot in a shared device miss-vector. After a
# captured decode replay, the post-replay hook polls the whole vector in ONE D2H: if every layer hit its
# window (count 0) the token is correct and we stop; otherwise each missed layer stages its deferred cold
# experts into their GPU slots out-of-graph and we replay the SAME graph again (the residency maps it reads
# are fixed-address, so the next replay sees them resident). Converges in ~1 extra replay.
_REPLAY_PAGERS: list = []
# Shared double-buffered wave context per device: (transfer_stream, ev_h2d[2], ev_gemm[2], idx_banks[2]).
# Global because the staging pin buffers it sequences are shared across layers (see ``wave_ctx``).
_WAVE_CTX: Dict = {}
# Streaming-prefill scratch pools per device: two full-E per-layer expert buffers ping-ponged across
# layers (see ``scratch_ctx``). ``False`` = allocation failed once; stay on the wave path.
_SCRATCH_CTX: Dict = {}
_MISS_VEC: Optional[torch.Tensor] = (
    None  # [N] int32; slot i = layer i's window-miss count this replay
)
_MISS_VEC_N: int = 0
_REPLAY_HOOK_INSTALLED = False

# Freq-ranked window: profile the first _PROFILE_TOKENS decode tokens (decide_bounded accumulates per-expert
# use counts on-device), then re-pin each windowed layer's hottest W experts once, out-of-graph, so the cold
# tail becomes the least-routed experts (rare window-misses). Always on with a fixed horizon: it is a no-op
# unless the store is windowed (nothing is registered in _REPLAY_PAGERS), and a few hundred tokens is plenty.
_PROFILE_TOKENS: int = 128
_profile_count: int = 0
_profile_done: bool = False


def _maybe_profile_refresh() -> None:
    """Called once per decode token (at replay convergence, out-of-graph). After the profiling horizon,
    re-pin every windowed layer's hottest W experts once."""
    global _profile_count, _profile_done
    if _profile_done or not _REPLAY_PAGERS:
        return
    _profile_count += 1
    if _profile_count < _PROFILE_TOKENS:
        return
    # The re-pin rewrites the pinned host_hot store in place, which the captured UVA gathers read —
    # any still-in-flight gather must finish first. Paid once (this is the horizon step).
    torch.cuda.synchronize()
    for p in _REPLAY_PAGERS:
        try:
            p.refresh_window_freq()
        except Exception as e:
            logger.error("[paged-experts] freq-window refresh failed: %s", e)
    _profile_done = True
    logger.info(
        "[paged-experts] freq-ranked window: re-pinned hottest W on %d layer(s) after %d tokens",
        len(_REPLAY_PAGERS),
        _profile_count,
    )


_BCG_HOOK_INSTALLED = False


# The pinned staging buffers are the store's shared machinery (store.py) — the replay-twice/BCG
# refill below stages through the same buffers the wave page-in uses.
from sglang.srt.layers.moe.paged_experts.store import (  # noqa: E402
    _STAGE_PIN,
    _stage_pin_buf,
)

# Shared DEVICE staging rows for the fused refill scatter, reused across ALL layers (refills are
# sequential and each ends with a stream sync before the buffers can be reused). Allocated at a FIXED
# row cap so the per-pager descriptor tensors (which bake the base pointers) never go stale.
_SC_STAGE: Dict = {}
_SC_STAGE_CAP = 16


def _sc_stage_buf(name: str, row_shape, dtype, device) -> torch.Tensor:
    key = (name, tuple(row_shape), dtype, str(device))
    buf = _SC_STAGE.get(key)
    if buf is None:
        buf = torch.empty((_SC_STAGE_CAP, *row_shape), dtype=dtype, device=device)
        _SC_STAGE[key] = buf
    return buf


def _prefetch_next_step() -> None:
    """Temporal cold-tier prefetch, fired at the step boundary (from the BCG post-step hook AND the
    replay-twice hook's convergence branches). Each windowed layer recorded the cold experts it missed
    THIS step (``_last_cold_ids``); decode routing is temporally stable, so the NEXT step will miss
    ≈ the same set. Issue one deep MADV_WILLNEED batch across ALL layers' misses now, so the kernel reads the
    whole token's cold working set in parallel (high queue depth -> near disk bandwidth) overlapping the next
    step's compute — instead of the ~0.5 thinly-spread faults per layer at warm steady state, which can't
    build a queue and run at random-read latency. No-op for the RAM cold tier / non-disk stores.
    Consume-once: a prediction is prefetched a single time, so a long run of clean tokens doesn't re-issue
    madvise for the same stale set every step.
    """
    for p in _REPLAY_PAGERS:
        ids = getattr(p, "_last_cold_ids", None)
        if not ids:
            continue
        p._last_cold_ids = []
        store = getattr(p, "store", None)
        if store is not None and hasattr(store, "prefetch_cold"):
            # force: the decode refill reads these rows via the mmap, not O_DIRECT
            store.prefetch_cold(ids, force=True)


# --- temporal routing-stability counter (SGLANG_PE_PRED_STATS) -------------------------------------
# The make-or-break measurement for temporal-prefetch H2D staging (#1) and the stage-0 signal for
# pre-gating: of the experts routed THIS step, how many were also routed LAST step
# (|routed_t ∩ routed_{t-1}| / |routed_t|), aggregated over all layers. High => next-step routing is
# predictable from this step => proactively staging the prediction into LFU victims between replays would
# convert predicted misses to hits (kill the replay-twice tax). Measures ROUTING (not the miss set, which
# the reactive stager confounds), so it is capture-mode-independent. Tapped on the EAGER decide, where the
# distinct set is already a host list — the captured path is sync-free by design (a host read there would
# break graph capture), so run the measurement eager (routing stability does not depend on the backend).
_PRED_STATS_ON = os.environ.get("SGLANG_PE_PRED_STATS") not in (None, "", "0")
_PRED_HIT = 0  # routed_t ∩ routed_{t-1}  (routing stability)
_PRED_TOTAL = 0  # routed_t
_PRED_MISS_HIT = 0  # missed_t ∩ routed_{t-1}  (predictable misses)
_PRED_MISS_TOTAL = 0  # missed_t  (experts routed this step but NOT resident = actual page-ins)
_PRED_LAYER_STEPS = 0
_PRED_LOG_EVERY = 2000  # layer-steps between readouts


def _record_pred_stats(pager, routed, missed=None) -> None:
    """Accumulate two temporal overlaps for ``pager`` (one MoE layer) against the PREVIOUS step's routed
    set, and log every ``_PRED_LOG_EVERY`` layer-steps. No-op unless ``SGLANG_PE_PRED_STATS`` is set.

    * routing hit-rate  |routed_t ∩ routed_{t-1}| / |routed_t|  — fundamental routing stability; dominated
      by the resident core LFU already keeps warm.
    * MISS hit-rate     |missed_t ∩ routed_{t-1}| / |missed_t|  — of the experts that actually MISS (routed
      but not resident, i.e. the page-ins), the fraction that were routed last step. THIS is the #1
      incremental signal: how much of the churning tail a "stage last step's set" predictor would catch.
    """
    if not _PRED_STATS_ON:
        return
    global _PRED_HIT, _PRED_TOTAL, _PRED_MISS_HIT, _PRED_MISS_TOTAL, _PRED_LAYER_STEPS
    cur = routed if isinstance(routed, (set, frozenset)) else set(routed)
    prev = getattr(pager, "_prev_routed", None)
    if prev is not None:
        if cur:
            _PRED_HIT += len(prev & cur)
            _PRED_TOTAL += len(cur)
        if missed:
            miss = missed if isinstance(missed, (set, frozenset)) else set(missed)
            _PRED_MISS_HIT += len(prev & miss)
            _PRED_MISS_TOTAL += len(miss)
    pager._prev_routed = cur
    _PRED_LAYER_STEPS += 1
    if _PRED_LAYER_STEPS % _PRED_LOG_EVERY == 0 and _PRED_TOTAL:
        logger.info(
            "[pe-pred] routing hit-rate=%.3f | MISS hit-rate=%.3f (%d misses) | over %d layer-steps",
            _PRED_HIT / _PRED_TOTAL,
            (_PRED_MISS_HIT / _PRED_MISS_TOTAL) if _PRED_MISS_TOTAL else 0.0,
            _PRED_MISS_TOTAL,
            _PRED_LAYER_STEPS,
        )


def _bcg_post_step() -> None:
    """BCG per-step boundary callback, fired once after the full forward + all eager breaks complete and
    before the next step. Does two things, in this order:

    1. Temporal prefetch (every step): kick async read-ahead of the next step's likely cold working set.
    2. Freq-window re-pin (until the horizon): this is the ONLY safe place to re-pin — ``refresh_window_freq``
       -> ``set_window_membership`` rewrites the pinned ``host_hot`` store in place, which the captured UVA
       gather reads. Mid-step (at a layer's break) it would race the later layers' in-flight gathers and
       corrupt output. Here no captured gather is in flight; we synchronize first so the step's async gathers
       finish before host_hot is permuted."""
    _prefetch_next_step()  # every step, async — overlaps the next step's compute
    _maybe_profile_refresh()  # no-op after the horizon; syncs internally only on the re-pin step


def _ensure_bcg_post_step_hook() -> None:
    """Install the per-step boundary hook on the breakable backend the first time the BCG break path runs
    (we only know decode is breakable once a break actually executes)."""
    global _BCG_HOOK_INSTALLED
    if _BCG_HOOK_INSTALLED:
        return
    from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
        set_post_replay_hook as _set_bcg_hook,
    )

    _set_bcg_hook(_bcg_post_step)
    _BCG_HOOK_INSTALLED = True


_MISS_VEC_CAP = (
    512  # max windowed layers; far above any current model's MoE layer count
)


_MISS_VEC_HOST = (
    None  # mapped PINNED mirror: decide_bounded also writes each layer's count here,
)
_MISS_VEC_HOST_BASE = (
    0  # host-visibly, so the BCG break spins on memory instead of a stream sync
)


def _alloc_miss_slot(device) -> tuple:
    """Reserve this layer's slot in the shared miss-vector; returns ``(index, device [1] view,
    pinned host [1] doorbell view, doorbell device address)``."""
    global _MISS_VEC, _MISS_VEC_N, _MISS_VEC_HOST, _MISS_VEC_HOST_BASE
    if _MISS_VEC is None:
        from sglang.kernels.jit.paged_experts_decide import paged_experts_host_devptr

        _MISS_VEC = torch.zeros(_MISS_VEC_CAP, dtype=torch.int32, device=device)
        _MISS_VEC_HOST = torch.full(
            (_MISS_VEC_CAP,), -1, dtype=torch.int32, pin_memory=True
        )
        _MISS_VEC_HOST_BASE = paged_experts_host_devptr(_MISS_VEC_HOST)
    if _MISS_VEC_N >= _MISS_VEC_CAP:
        # Past the cap the [idx:idx+1] view would be EMPTY and the layer's misses invisible (silent
        # non-convergence). Fail loudly instead.
        raise RuntimeError(
            f"paged-experts: more than {_MISS_VEC_CAP} windowed layers registered; raise _MISS_VEC_CAP"
        )
    idx = _MISS_VEC_N
    _MISS_VEC_N += 1
    return (
        idx,
        _MISS_VEC[idx : idx + 1],
        _MISS_VEC_HOST[idx : idx + 1],
        _MISS_VEC_HOST_BASE + 4 * idx,
    )


def _post_replay_refill_all() -> bool:
    """Post-replay hook (registered with the cuda-graph backend). One scalar D2H over the shared miss-vector
    short-circuits the no-miss case; only an actual miss-step pays the per-layer count read + staging.
    """
    if _MISS_VEC is None or _MISS_VEC_N == 0:
        return False
    total = int(
        _MISS_VEC[:_MISS_VEC_N].sum().item()
    )  # one sync; no-miss steps stop here
    if total == 0:
        _prefetch_next_step()  # step boundary: temporal read-ahead of the predicted cold set
        _maybe_profile_refresh()  # token converged (no misses) -> count it toward the profiling horizon
        return False
    counts = _MISS_VEC[:_MISS_VEC_N].tolist()
    missed = [p for p in _REPLAY_PAGERS if counts[p._miss_idx] > 0]
    staged = False
    if missed:
        # Batch the per-layer refill reads into 4 D2H copies total (stack across all missed layers) instead
        # of one .tolist() sync per tensor per layer.
        cold_logs = torch.stack([p._cold_log_d for p in missed]).cpu().tolist()
        neededs = torch.stack([p._needed_d for p in missed]).cpu().tolist()
        slot_exps = torch.stack([p._slot_expert_d for p in missed]).cpu().tolist()
        lastuses = torch.stack([p._slot_lastuse_d for p in missed]).cpu().tolist()
        for i, p in enumerate(missed):
            if p._refill_after_replay(
                counts[p._miss_idx], cold_logs[i], neededs[i], slot_exps[i], lastuses[i]
            ):
                staged = True
    if staged:
        torch.cuda.synchronize()  # one device sync for all layers' staging, then replay again
        return True
    _prefetch_next_step()  # step boundary after staging: same temporal read-ahead as the clean case
    _maybe_profile_refresh()  # token converged after staging -> count it toward the profiling horizon
    return False


def _register_replay_pager(p: ExpertPager) -> None:
    global _REPLAY_HOOK_INSTALLED
    if p not in _REPLAY_PAGERS:
        _REPLAY_PAGERS.append(p)
    if not _REPLAY_HOOK_INSTALLED:
        from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
            set_post_replay_hook,
        )

        set_post_replay_hook(_post_replay_refill_all)
        _REPLAY_HOOK_INSTALLED = True


def reset_paged_experts_state() -> None:
    """Drop all module-global replay/profile/staging state. Called when a NEW model build starts in this
    process (see ``method.make_for_layer``): a re-created engine must not inherit the previous model's
    registered pagers (freed GPU tensors), a spent profiling horizon, or the old staging buffers.
    The installed backend hooks are process-global and idempotent, so they stay."""
    global _MISS_VEC, _MISS_VEC_N, _profile_count, _profile_done
    _REPLAY_PAGERS.clear()
    _ALL_PAGERS.clear()
    _MISS_VEC = None
    global _MISS_VEC_HOST, _MISS_VEC_HOST_BASE
    _MISS_VEC_HOST = None
    _MISS_VEC_HOST_BASE = 0
    _MISS_VEC_N = 0
    _profile_count = 0
    _profile_done = False
    _STAGE_PIN.clear()
    _SC_STAGE.clear()
    # Drop the previous model's staging/wave device buffers + streams (sized to the old geometry);
    # they re-validate shape per call but would otherwise leak across an in-process rebuild.
    _WAVE_CTX.clear()
    _SCRATCH_CTX.clear()


class ExpertPager:
    """Per-step residency decision over the K-slot pool; delegates backing + page-in to an ``ExpertStore``
    and the eviction choice to a ``ResidencyPolicy``.

    The positional constructor ``(layer, E, K, device, pin_host=...)`` builds the matching store, or
    pass a prebuilt ``store=`` to compose one directly (what ``setup_pager`` does). ``eviction`` selects
    the residency policy (``lru`` default | ``lfu``).
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
        # Fused-launch state: gather_multi descriptors (per-tensor base pointers, built once at setup)
        # and the remap_mask output buffers (allocated at the captured shape).
        self._gm_stores = self._gm_slots = self._gm_e16s = None
        self._safe_ids_d = self._masked_tw_d = None
        # True after decide_and_page_ondevice fused the forward remap into the decide launch (#4), so
        # remap_mask_ondevice returns the already-computed buffers instead of a second launch. Set per step
        # by the plain captured keep-warm path only; the windowed/BCG paths leave it False (they remap after
        # their out-of-graph cold-staging break).
        self._remap_prefused = False
        # Windowed (bounded) captured path — set up by setup_ondevice when the store is a WindowedExpertStore:
        # static hot/cold membership maps the decide_bounded kernel reads, the cold (deferred-miss) plan
        # buffers, the needed[] mask, and this layer's slot in the shared replay-twice miss-vector.
        self._windowed = False
        self._log2hot_d = None
        self._cold_log_d = self._needed_d = self._cold_n_d = None
        self._miss_idx = -1

    # --- backing delegated to the store (exposed on the pager for the fill code) ---
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

    def page_in(
        self,
        src_experts: torch.Tensor,
        dst_slots: torch.Tensor,
        *,
        stage_bank: int = 0,
        async_h2d: bool = False,
        src_host: Optional[list] = None,
    ) -> None:
        """Page the chosen experts into their slots via the store (transport-specific; a no-op if empty).
        ``stage_bank``/``async_h2d``/``src_host`` are the double-buffered wave path's knobs: separate
        staging buffers per bank, no trailing stream sync (the caller sequences buffer reuse with
        events), and a host-side copy of the plan (no D2H read-back)."""
        self.store.page_in(
            src_experts,
            dst_slots,
            stage_bank=stage_bank,
            async_h2d=async_h2d,
            src_host=src_host,
        )

    def scratch_ctx(self):
        """Streaming-prefill scratch pools: TWO full-``[E, *slot]`` buffer sets (one per paged tensor),
        ping-ponged across layers — while layer i's vanilla E-wide GEMM computes out of one set, the
        transfer stream fills the other with layer i+1's whole expert set. GLOBAL per device (every MoE
        layer has identical per-expert shapes in the supported models; verified per call). Returns
        ``(bufs[2], ev_ready[2], ev_gemm[2])`` or ``None`` (allocation failed / shape mismatch — the
        caller falls back to the wave path)."""
        global _SCRATCH_CTX
        key = torch.device(self.device).index or 0
        ctx = _SCRATCH_CTX.get(key)
        if ctx is None:
            try:
                bufs = tuple(
                    {
                        name: torch.empty(
                            (self.E, *p.shape[1:]), dtype=p.dtype, device=self.device
                        )
                        for name, p in self.store.gpu.items()
                    }
                    for _ in range(2)
                )
                ctx = (
                    bufs,
                    (torch.cuda.Event(), torch.cuda.Event()),
                    (torch.cuda.Event(), torch.cuda.Event()),
                )
            except torch.cuda.OutOfMemoryError:
                logger.warning(
                    "[paged-experts] streaming-prefill scratch (2x full-E layer set) does not fit "
                    "free VRAM — prefill stays on the wave path"
                )
                ctx = False
            _SCRATCH_CTX[key] = ctx
        if ctx is False:
            return None
        # per-call shape guard: a layer whose paged tensors differ (names or shapes) can't use the pool
        ref = ctx[0][0]
        if len(ref) != len(self.store.gpu):
            return None
        for name, p in self.store.gpu.items():
            t = ref.get(name)
            if t is None or t.shape != (self.E, *p.shape[1:]) or t.dtype != p.dtype:
                return None
        return ctx

    def scratch_fill_resident_aware(self, bufs: Dict[str, torch.Tensor]) -> bool:
        """Fill the scratch pool for the streaming prefill from TWO sources: experts resident in this
        layer's K-slot pool are copied device-to-device (~15x the PCIe rate; pool rows are exact copies
        of their store rows — page-in never mutates), and only the complement streams from the pinned
        host store. The split is planned ON DEVICE from the live residency map (``scratch_split``): the
        host mirror is stale after captured decode replays, and reading the device map back would stall
        the CPU on the transfer stream. Both plans execute via ``gather_multi`` (address-agnostic copy;
        the D2D pass gets plain device pool pointers instead of UVA host pointers). Returns ``False``
        when the on-device machinery isn't set up (caller falls back to ``store.read_full``).
        """
        if not self.ondevice or self._gm_stores is None:
            return False
        from sglang.kernels.jit.paged_experts_decide import (
            paged_experts_gather_multi,
            paged_experts_scratch_split,
        )

        plans = getattr(self, "_scratch_plans", None)
        if plans is None:
            mk = lambda n: torch.zeros(n, dtype=torch.int32, device=self.device)
            plans = (mk(self.E), mk(self.E), mk(1), mk(self.E), mk(self.E), mk(1))
            self._scratch_plans = plans
        base_cache = getattr(self, "_scratch_dst_bases", None)
        if base_cache is None:
            base_cache = {}
            self._scratch_dst_bases = base_cache
        # keyed per bank buffer-set (two banks, distinct pointers); the buffers are persistent, so
        # their base pointers are stable. Order must match the pool/store descriptor tensors (built
        # from the same store.gpu iteration order).
        dst_bases = base_cache.get(id(bufs))
        if dst_bases is None:
            dst_bases = torch.tensor(
                [bufs[name].data_ptr() for name in self.store.gpu],
                dtype=torch.int64,
                device=self.device,
            )
            base_cache[id(bufs)] = dst_bases
        res_src, res_dst, res_n, h2d_src, h2d_dst, h2d_n = plans
        paged_experts_scratch_split(
            self.logical_to_gpu_index_cuda,
            res_src,
            res_dst,
            res_n,
            h2d_src,
            h2d_dst,
            h2d_n,
        )
        # complement from the pinned host store (UVA bases), residents from the pool (device bases)
        paged_experts_gather_multi(
            self._gm_stores, dst_bases, self._gm_e16s, h2d_src, h2d_dst, h2d_n
        )
        paged_experts_gather_multi(
            self._gm_slots, dst_bases, self._gm_e16s, res_src, res_dst, res_n
        )
        return True

    def wave_ctx(self):
        """Shared state for the double-buffered (banked) wave path: ONE transfer stream, two event pairs
        (h2d-done / gemm-done per bank), and two per-bank idx buffers so wave w+1's decide cannot race
        wave w's remap. GLOBAL (per device), not per-pager: the staging pin buffers it guards are shared
        across layers, so bank reuse must be sequenced across layer boundaries by the same events.
        """
        global _WAVE_CTX
        dev = torch.device(self.device)
        key = dev.index or 0
        ctx = _WAVE_CTX.get(key)
        if ctx is None or ctx[3][0].numel() < self.E:
            ctx = (
                torch.cuda.Stream(device=dev),
                (torch.cuda.Event(), torch.cuda.Event()),
                (torch.cuda.Event(), torch.cuda.Event()),
                (
                    torch.full((self.E,), -1, dtype=torch.int32, device=dev),
                    torch.full((self.E,), -1, dtype=torch.int32, device=dev),
                ),
            )
            _WAVE_CTX[key] = ctx
        return ctx

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
        l2g_t = self.logical_to_gpu_index  # [E] int32 CPU host mirror
        l2g = l2g_t.tolist()  # one bulk read -> python list; avoids ~2*len(distinct) 0-d tensor builds
        se = self.slot_expert  # already a python list; mutations below persist (same object)
        needed = set(distinct)
        if _PRED_STATS_ON:  # SGLANG_PE_PRED_STATS: routing + miss temporal hit-rate (l2g is pre-step here)
            _record_pred_stats(self, needed, {e for e in distinct if l2g[e] < 0})
        for e in distinct:  # touch recency/frequency of resident hits
            s = l2g[e]
            if s >= 0:
                self.policy.record_use(e, s)
        src, dst = [], []
        for e in distinct:
            if l2g[e] >= 0:
                continue  # already resident (or just assigned)
            victim = self.policy.pick_victim(se, needed)
            if victim < 0:
                continue  # pool too small (shouldn't happen: distinct <= K)
            old = se[victim]
            if old >= 0:
                l2g[old] = -1
            se[victim] = e
            l2g[e] = victim
            self.policy.record_use(e, victim)  # the fresh assignment counts as a use
            src.append(e)
            dst.append(victim)
        if not src:
            # pure-hit step: residency (and thus logical_to_gpu_index_cuda) is unchanged from last step,
            # so skip the [E] H2D map copy AND both torch.tensor(list, device=) builds entirely.
            if not hasattr(self, "_kw_empty"):
                self._kw_empty = torch.empty(0, dtype=torch.int64, device=self.device)
            return self._kw_empty, self._kw_empty
        l2g_t.copy_(torch.tensor(l2g, dtype=l2g_t.dtype))  # one bulk write-back (miss steps only)
        self.logical_to_gpu_index_cuda.copy_(l2g_t, non_blocking=True)
        return (
            torch.tensor(src, dtype=torch.int64, device=self.device),
            torch.tensor(dst, dtype=torch.int64, device=self.device),
        )

    def setup_ondevice(self) -> None:
        """Allocate the device-resident residency state for the captured path and resolve the pinned
        store's UVA device pointer (once, outside any graph). Requires a pinned store with 16-byte-aligned
        per-expert blocks (the gather is float4). Slots 0..K-1 start holding experts 0..K-1, matching the
        eager seeding."""
        from sglang.kernels.jit.paged_experts_decide import paged_experts_host_devptr

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
        # gather_multi descriptors: one fused launch pages ALL paged tensors (6 launches -> 1 for marlin
        # int4). Base pointers are capture-stable: the UVA store pointer and the GPU pool allocations are
        # resolved after weight processing and never move.
        names = list(self.gpu)
        self._gm_stores = torch.tensor(
            [self.store_devptr[n] for n in names], dtype=torch.int64, device=dev
        )
        self._gm_slots = torch.tensor(
            [self.gpu[n].data.data_ptr() for n in names], dtype=torch.int64, device=dev
        )
        self._gm_e16s = torch.tensor(
            [self.item_bytes[n] // 16 for n in names], dtype=torch.int64, device=dev
        )
        # expert_slot and idx are the same buffer the forward remap reads (logical_to_gpu_index_cuda).
        self.ondevice = True
        if windowed:
            self._setup_window_ondevice()

    # LFU frequency sentinel for pinned experts: large enough to never be the min-freq victim under any
    # realistic organic use count, with headroom below int32 max so the per-use atomicAdd can't overflow.
    def seed_from_recording(self, counts: torch.Tensor) -> None:
        """Warm-start + optional PINNED HOT TIER from an expert-distribution recording (this layer's
        per-expert selection counts):
        (1) physically page the hottest-K experts into the slot pool at init so the FIRST request hits them
        instead of paging the hot set in as it goes (freq priors alone can't do this — the pool boots
        holding experts 0..K-1, so the hot set would still page in on first use);
        (2) seed the LFU frequency priors so eviction protects them from step 1;
        (3) if ``pin_count`` > 0, set the hottest ``pin_count`` experts' freq to a sentinel so LFU never
        evicts them while any non-pinned non-needed slot exists — a static hot tier + dynamic remainder that
        removes the churn keeping LFU below the top-K coverage ceiling. SOFT: under pressure (a step needing
        more non-pinned slots than the dynamic pool has) ``pick_victim`` still evicts the least-hot pinned
        slot, so an expert is never masked. Runs EAGERLY at setup (before capture); reuses the capture-safe
        UVA gather. No-op unless the on-device state exists."""
        if self._freq_d is None or self._gm_stores is None or self._src_d is None:
            return
        from sglang.kernels.jit.paged_experts_decide import paged_experts_gather_multi

        dev, i32 = self.device, torch.int32
        counts = counts.to(device=dev, dtype=torch.float32)
        self._freq_d.copy_(counts.to(i32))  # LFU priors
        # DELTA reseed: keep hot experts already resident where they sit; page only the missing hot
        # experts into slots whose occupant is NOT hot. Eager-only (host sync OK) — reads the LIVE device
        # maps (the wave resync updates only the device side). A full-K unconditional reseed cost ~75MB x
        # layers per prefill and sank short generations; the delta is usually a small fraction.
        hot_l = counts.topk(self.K).indices.cpu().tolist()
        hotset = set(hot_l)
        cur_slot_exp = self._slot_expert_d.cpu().tolist()  # slot -> expert (live)
        resident = set(x for x in cur_slot_exp if x >= 0)
        free = [s for s, e in enumerate(cur_slot_exp) if e not in hotset]
        missing = [e for e in hot_l if e not in resident]  # hottest-first order
        src, dst = missing[: len(free)], free[: len(missing)]
        n = len(src)
        if n == 0:
            # pool already holds the hot set (repeat prompt / later chunk): just sync the host mirrors
            self.slot_expert = cur_slot_exp
            self.logical_to_gpu_index.fill_(-1)
            for s_, e in enumerate(cur_slot_exp):
                if e >= 0:
                    self.logical_to_gpu_index[e] = s_
            return
        if n:
            self._src_d[:n].copy_(torch.tensor(src, dtype=i32, device=dev))
            self._dst_d[:n].copy_(torch.tensor(dst, dtype=i32, device=dev))
            self._n_out_d.fill_(n)
            paged_experts_gather_multi(
                self._gm_stores, self._gm_slots, self._gm_e16s, self._src_d, self._dst_d, self._n_out_d
            )
        # rebuild maps host-side from the reconciled layout, push once
        for e, s_ in zip(src, dst):
            cur_slot_exp[s_] = e
        self.slot_expert = cur_slot_exp
        self.logical_to_gpu_index.fill_(-1)
        for s_, e in enumerate(cur_slot_exp):
            if e >= 0:
                self.logical_to_gpu_index[e] = s_
        self.logical_to_gpu_index_cuda.copy_(self.logical_to_gpu_index)
        self._slot_expert_d.copy_(torch.tensor(cur_slot_exp, dtype=i32, device=dev))
        # reset recency uniformly: after prefill the kept residents' lastuse is ancient, and mixed-age
        # state made LRU evict the kept hot residents first (measured: killed the warm-up win).
        self._slot_lastuse_d.zero_()
        hot = torch.tensor(hot_l, dtype=i32, device=dev)

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
        pinned store into the GPU pool — ONE fused launch for all paged tensors (count read on-device ->
        capture-safe)."""
        from sglang.kernels.jit.paged_experts_decide import paged_experts_gather_multi

        paged_experts_gather_multi(
            self._gm_stores,
            self._gm_slots,
            self._gm_e16s,
            self._src_d,
            self._dst_d,
            self._n_out_d,
        )

    def remap_mask_ondevice(self, topk_ids: torch.Tensor, topk_weights: torch.Tensor):
        """Fused remap + routing-weight mask (one launch instead of the 5-node python chain). Reads the
        LIVE ``logical_to_gpu_index_cuda``, so on the BCG path it runs AFTER the staging break and sees
        just-staged experts. Returns ``(safe_ids, masked_tw)`` shaped like the inputs, or ``None`` when
        the weights layout is unsupported (caller falls back to the python chain)."""
        from sglang.kernels.jit.paged_experts_decide import paged_experts_remap_mask

        if topk_weights.dtype != torch.float32 or not topk_weights.is_contiguous():
            if not getattr(self, "_remap_fallback_warned", False):
                self._remap_fallback_warned = True
                logger.warning(
                    "[paged-experts] fused remap_mask unavailable (topk_weights dtype=%s, contiguous=%s)"
                    " — falling back to the elementwise remap chain for this layer's graphs.",
                    topk_weights.dtype,
                    topk_weights.is_contiguous(),
                )
            return None
        if self._remap_prefused:
            # #4: decide_and_page_ondevice already emitted the remap in the decide launch this step.
            return (
                self._safe_ids_d.view(topk_ids.shape),
                self._masked_tw_d.view(topk_weights.shape),
            )
        t = topk_ids.numel()
        if self._safe_ids_d is None or self._safe_ids_d.numel() != t:
            self._safe_ids_d = torch.empty(t, dtype=torch.int32, device=self.device)
            self._masked_tw_d = torch.empty(t, dtype=torch.float32, device=self.device)
        paged_experts_remap_mask(
            self._topk_i32,  # already holds this step's flattened ids (_prep_topk_ondevice in decide)
            self.logical_to_gpu_index_cuda,
            topk_weights.reshape(-1),
            self._safe_ids_d,
            self._masked_tw_d,
        )
        return (
            self._safe_ids_d.view(topk_ids.shape),
            self._masked_tw_d.view(topk_weights.shape),
        )

    def decide_and_page_ondevice(
        self, topk_ids: torch.Tensor, topk_weights: torch.Tensor = None
    ) -> None:
        """Capture-safe keep-warm: decide residency + page the misses entirely on-device (no host sync).
        Mutates the persistent state buffers and ``logical_to_gpu_index_cuda`` (the remap table) in place;
        gathers exactly the chosen experts. Requires distinct active experts <= K (the caller guarantees it
        via the shape guard ``num_tokens * top_k <= K``).

        When ``topk_weights`` is passed and remap-fusable (float32, contiguous), the decide launch ALSO
        emits the forward remap (``paged_experts_decide_remap``, #4) — one launch replacing decide + a
        separate ``remap_mask`` — and ``remap_mask_ondevice`` returns the buffers without relaunching. Safe
        only here (the plain path): idx aliases ``logical_to_gpu_index_cuda`` and nothing runs between the
        decide and the remap. The windowed/BCG paths call the non-fusing variants (they must remap after
        their out-of-graph cold-staging break)."""
        self._prep_topk_ondevice(topk_ids)
        l2g = self.logical_to_gpu_index_cuda  # serves as both expert_slot and idx
        lfu = self.policy.name == "lfu"  # --paged-experts-eviction, same as the host path
        fusable = (
            topk_weights is not None
            and topk_weights.dtype == torch.float32
            and topk_weights.is_contiguous()
        )
        if fusable:
            from sglang.kernels.jit.paged_experts_decide import (
                paged_experts_decide_remap,
            )

            t = topk_ids.numel()
            if self._safe_ids_d is None or self._safe_ids_d.numel() != t:
                self._safe_ids_d = torch.empty(t, dtype=torch.int32, device=self.device)
                self._masked_tw_d = torch.empty(
                    t, dtype=torch.float32, device=self.device
                )
            paged_experts_decide_remap(
                self._topk_i32,
                self._step_ctr_d,
                self._slot_expert_d,
                l2g,
                self._slot_lastuse_d,
                self._freq_d,
                lfu,
                self._src_d,
                self._dst_d,
                self._n_out_d,
                l2g,
                topk_weights.reshape(-1),
                self._safe_ids_d,
                self._masked_tw_d,
            )
            self._remap_prefused = True
        else:
            from sglang.kernels.jit.paged_experts_decide import paged_experts_decide

            paged_experts_decide(
                self._topk_i32,
                self._step_ctr_d,
                self._slot_expert_d,
                l2g,
                self._slot_lastuse_d,
                self._freq_d,
                lfu,
                self._src_d,
                self._dst_d,
                self._n_out_d,
                l2g,
            )
            self._remap_prefused = False
        self._gather_planned_ondevice()

    def decide_and_page_bounded_ondevice(self, topk_ids: torch.Tensor) -> None:
        """Capture-safe windowed keep-warm (the >pin-ceiling fallback). ``decide_bounded`` splits the plan by
        window membership: window hits gather in-graph from the pinned ``host_hot`` (capture-safe); cold
        (window-missing) experts are deferred — their logical ids land in ``_cold_log_d`` and they stay
        masked this replay — for the post-replay refill to stage and converge. Requires distinct active <= K.
        """
        from sglang.kernels.jit.paged_experts_decide import paged_experts_decide_bounded

        self._prep_topk_ondevice(topk_ids)
        l2g = self.logical_to_gpu_index_cuda  # serves as both expert_slot and idx
        paged_experts_decide_bounded(
            self._topk_i32,
            self._step_ctr_d,
            self._slot_expert_d,
            l2g,
            self._slot_lastuse_d,
            self._freq_d,
            self.policy.name
            == "lfu",  # --paged-experts-eviction, same as the host path
            self._log2hot_d,
            self._src_d,
            self._dst_d,
            self._n_out_d,
            self._cold_log_d,
            self._cold_n_d,
            l2g,
            self._needed_d,
            doorbell=self._doorbell_ptr,
        )
        self._gather_planned_ondevice()  # window hits only; cold misses deferred to _refill_after_replay

    def stage_cold_at_break(self) -> None:
        """BCG break-and-page-in: stage THIS step's deferred cold experts at an in-layer eager break
        (between decide and the expert GEMM), instead of deferring to a post-replay refill. Runs eager
        every replay; reads this layer's cold-miss count + plan and refills directly — so the GEMM segment
        sees the cold experts resident with NO second full-graph replay. Reuses the replay-twice refill,
        but inline (one D2H for the count, plus the staging copies)."""
        # Doorbell read: decide_bounded wrote the cold count to mapped pinned memory, so a HIT layer
        # costs a plain memory read here instead of a per-layer stream sync — the host can run ahead
        # launching segments. Bounded spin with an .item() fallback (capture-time state is stale by
        # design; the fallback also covers a lost write).
        bell = self._doorbell_np
        cn = int(bell[0])
        if cn < 0:
            deadline = time.perf_counter() + 0.2
            while cn < 0 and time.perf_counter() < deadline:
                cn = int(bell[0])
            if cn < 0:
                cn = int(self._cold_n_d.item())
        bell[0] = -1  # re-arm for the next replay
        if cn > 0:
            # ONE batched D2H for plan + residency + recency (vs one .tolist() sync per tensor),
            # through preallocated device/pinned snapshot buffers (no per-call allocations).
            k = self.K
            torch.cat(
                [
                    self._cold_log_d,
                    self._needed_d,
                    self._slot_expert_d,
                    self._slot_lastuse_d,
                ],
                out=self._snap_dev,
            )
            self._snap_pin.copy_(self._snap_dev)  # one D2H sync into pinned host
            snap = self._snap_pin.tolist()
            staged = self._refill_after_replay(
                cn, snap[:k], snap[k : 2 * k], snap[2 * k : 3 * k], snap[3 * k :]
            )
            if staged < cn:
                # Impossible under the keep-warm bound (bs*top_k <= K guarantees eviction headroom);
                # if it fires, silently masking the unstaged experts would corrupt this token's output.
                raise RuntimeError(
                    f"[paged-experts] BCG staging placed only {staged}/{cn} cold experts (K={self.K}) "
                    "— the keep-warm bound was violated; raise --paged-experts-num-resident."
                )
        else:
            self._last_cold_ids = []
        # Freq-window driver: install the per-step boundary hook on the breakable backend on first use. The
        # actual re-pin + the temporal prefetch run there (between steps), NOT here mid-step — see _bcg_post_step.
        _ensure_bcg_post_step_hook()

    def _refill_after_replay(self, cn: int, cold_log, needed, se, lastuse) -> int:
        """Post-replay (out-of-graph): stage the deferred cold experts ``host_cold`` -> their GPU slots and
        mark them resident, so the next replay's ``decide_bounded`` sees them as hits and the loop converges.
        Evicts only slots NOT needed this step (the ``needed[]`` mask), choosing victims TIER- and
        POLICY-aware: hot-tier residents (cheap in-graph re-gather) before cold-tier residents (whose
        re-fetch costs another deferred round), oldest-recency first — instead of arbitrary slot order,
        which could evict the hottest resident. ``cold_log``/``needed``/``se``/``lastuse`` are host lists
        the caller read in batched D2H copies (no per-layer sync here).

        Robust to a momentary shortage: if there are fewer evictable slots than cold misses this round (no
        eviction headroom — e.g. K too close to top_k), it stages as many as fit and returns True so the loop
        retries the rest, rather than giving up and masking experts. (With K > top_k the headroom exists and
        it converges in one extra replay; K <= top_k is the misconfig this guards — raise
        --paged-experts-num-resident above top_k.) Returns the number staged."""
        missed = cold_log[:cn]  # logical ids (defer mode emits logical ids)
        # Record the prediction for the next-step temporal prefetch BEFORE any early return — the
        # no-headroom case is exactly the set guaranteed to miss again next round.
        self._last_cold_ids = list(missed)
        hot_pos = getattr(self.store, "hot_pos", None)
        hot_list = hot_pos.tolist() if hot_pos is not None else None

        def _victim_key(s):
            e_res = se[s]
            cold_res = (
                1
                if (hot_list is not None and e_res >= 0 and hot_list[e_res] < 0)
                else 0
            )
            return (cold_res, lastuse[s])

        evictable = sorted((s for s in range(self.K) if not needed[s]), key=_victim_key)
        n = min(len(evictable), cn)
        if n < cn:
            logger.warning(
                "[paged-experts] replay-twice: %d cold misses but only %d evictable slots (K=%d) — staging "
                "%d this round, retrying the rest. Raise --paged-experts-num-resident above top_k.",
                cn,
                len(evictable),
                self.K,
                n,
            )
        if n == 0:
            return (
                0  # no headroom this round; nothing staged (avoid a no-progress replay)
            )
        l2g = self.logical_to_gpu_index_cuda
        slots = evictable[:n]
        ids = list(missed[:n])
        # Disk cold tier: kick parallel read-ahead for all of this layer's cold rows up front (MADV_WILLNEED),
        # so the gather below doesn't serialize on one page fault at a time. No-op for the RAM tier.
        store = getattr(self, "store", None)
        if store is not None and hasattr(store, "prefetch_cold"):
            store.prefetch_cold(ids)
        # Gather the cold rows into PINNED buffers (vs torch.stack -> pageable), so the H2D below is a
        # fast pinned transfer instead of a slow synchronous pageable one. Serial on purpose: with the
        # WILLNEED read-ahead above the faults are already serviced concurrently, and threading the warm
        # copies was measured net-negative (see store.py's staging notes).
        bufs = {
            name: _stage_pin_buf(name, self.K, p.shape[1:], p.dtype)
            for name, p in self.gpu.items()
        }
        # Cold rows read through the O_DIRECT queue-depth pool (QD ~6) when the store backs cold on
        # disk — the serial store.row() mmap-fault loop is single-threaded and left the disk at ~40% of
        # its concurrent ceiling on decode. Per-tensor fallback to the mmap copy (RAM-windowed stores,
        # unaligned rows, or IO failure), mirroring store.page_in.
        cp = getattr(self.store, "cold_pos", None)
        direct = getattr(self.store, "_read_cold_rows_direct", None)
        cold_rows = [int(cp[e]) for e in ids] if cp is not None else None
        for name in self.gpu:
            buf = bufs[name]
            if not (
                cold_rows is not None
                and direct is not None
                and direct(name, cold_rows, buf)
            ):
                for i, e in enumerate(ids):
                    buf[i].copy_(self.store.row(name, e))
        # Fused scatter: ONE contiguous async H2D per tensor into the device staging rows, then one
        # scatter_multi launch places every tensor's rows into the victim slots — replacing 4*n
        # micro-copies (two of which move <1 KB fp8 scale rows). Falls back to per-row copies past the
        # staging cap or on non-windowed stores.
        if getattr(self, "_sc_stage", None) is not None and n <= self._sc_cap:
            from sglang.kernels.jit.paged_experts_decide import (
                paged_experts_scatter_multi,
            )

            for name in self.gpu:
                self._sc_stage[name][:n].copy_(bufs[name][:n], non_blocking=True)
            self._sc_dst_np[:n] = slots
            self._sc_dst_d[:n].copy_(self._sc_dst_pin[:n], non_blocking=True)
            paged_experts_scatter_multi(
                self._sc_stage_ptrs, self._gm_slots, self._gm_e16s, self._sc_dst_d, n
            )
        else:
            for name, gpu_param in self.gpu.items():
                buf = bufs[name]
                for i, slot in enumerate(slots):
                    gpu_param.data[slot].copy_(buf[i], non_blocking=True)
        # Batched residency + recency update (3 indexed writes total, vs 2-3 scalar device writes per
        # staged expert): unmap the evicted occupants, map the staged experts, and stamp the staged slots
        # with the CURRENT step — a staged cold expert must not inherit its victim's stale recency, or the
        # next decide evicts it first and it thrashes through another deferred round.
        slots_dev = torch.tensor(slots, dtype=torch.int64, device=self.device)
        olds = [se[s] for s in slots if se[s] >= 0]
        if olds:
            l2g[torch.tensor(olds, dtype=torch.int64, device=self.device)] = -1
        self._slot_expert_d[slots_dev] = torch.tensor(
            ids, dtype=torch.int32, device=self.device
        )
        l2g[torch.tensor(ids, dtype=torch.int64, device=self.device)] = slots_dev.to(
            torch.int32
        )
        self._slot_lastuse_d[slots_dev] = self._step_ctr_d
        # One sync for all tensors' async H2D: the shared pinned bufs must not be overwritten (by the next
        # layer's refill) while copies are in flight.
        torch.cuda.current_stream().synchronize()
        return n

    def refresh_window_freq(self) -> None:
        """P3 freq-ranked window: re-pin the hottest W experts (by the on-device use counts ``decide_bounded``
        accumulated during profiling) so the cold tail is the *least*-routed experts -> window-misses become
        rare -> few replay-twice rounds. Runs once, out-of-graph, between tokens; refreshes the membership
        maps in place (fixed address -> the captured graph reads the new split on its next replay). The GPU
        slots keep their expert-indexed data, so only the page-in source tier moves (no residency reset).
        """
        if not self._windowed:
            return
        freq = (
            self._freq_d.tolist()
        )  # per-expert routing count over the profiling window (one-time D2H)
        hot = sorted(range(self.E), key=lambda e: freq[e], reverse=True)[: self.store.W]
        self.store.set_window_membership(hot)
        # refresh the device membership map IN PLACE (same buffer the captured decide_bounded reads)
        self._log2hot_d.copy_(
            self.store.hot_pos.to(dtype=torch.int32, device=self.device)
        )
        # Age the counts (halve) instead of zeroing: ``_freq_d`` doubles as the LFU eviction key, and a
        # hard zero would erase the very frequency history LFU keys on right after the re-pin.
        self._freq_d.copy_(torch.div(self._freq_d, 2, rounding_mode="floor"))

    def decide_and_page_wave_ondevice(
        self,
        topk_ids: torch.Tensor,
        wave: int,
        *,
        wave_k: Optional[int] = None,
        slot_base: int = 0,
        idx_out: Optional[torch.Tensor] = None,
    ) -> None:
        """One static wave (distinct > K, e.g. prefill): plan + gather the in-wave experts on-device. The
        caller runs ceil(E/wave_k) waves and sums the per-wave GEMM partials, then calls
        ``resync_residency_ondevice`` so the keep-warm state matches the slots (zero-copy mode instead
        restores its static hot tier afterwards via ``reseed_hot``). The banked
        (double-buffered) caller passes ``wave_k = K//2`` with alternating ``slot_base`` and a per-bank
        ``idx_out`` (so the next wave's decide cannot race this wave's remap); the captured caller uses
        the defaults (full-K waves into ``logical_to_gpu_index_cuda``)."""
        from sglang.kernels.jit.paged_experts_decide import paged_experts_decide_wave

        if wave == 0:
            self._prep_topk_ondevice(topk_ids)
        paged_experts_decide_wave(
            self._topk_i32,
            self.E,
            wave_k if wave_k is not None else self.K,
            wave,
            self._src_d,
            self._dst_d,
            self._n_out_d,
            idx_out if idx_out is not None else self.logical_to_gpu_index_cuda,
            slot_base=slot_base,
        )
        self._gather_planned_ondevice()

    def resync_residency_ondevice(self, lo: int, ngrp: int, slot_base: int = 0) -> None:
        """After the wave loop, slots ``[slot_base, slot_base+ngrp)`` physically hold experts
        ``[lo, lo+ngrp)``. Point the device keep-warm state (and the live remap) at that so the next
        decode step is consistent."""
        self._slot_expert_d.fill_(-1)
        idx = torch.arange(lo, lo + ngrp, dtype=torch.int32, device=self.device)
        self._slot_expert_d[slot_base : slot_base + ngrp] = idx
        l2g = self.logical_to_gpu_index_cuda
        l2g.fill_(-1)
        l2g[lo : lo + ngrp] = torch.arange(
            slot_base, slot_base + ngrp, dtype=torch.int32, device=self.device
        )
        self._slot_lastuse_d.zero_()

    def set_residency(self, experts, base: int = 0) -> None:
        """Force slot ``base + i`` to hold ``experts[i]`` and rebuild the maps. Called after the wave
        path so the next keep-warm step's residency state matches what is physically in the slots
        (``base`` is the last wave's bank offset on the double-buffered path).
        """
        experts = list(experts)
        self.slot_expert = [-1] * base + experts + [-1] * (self.K - base - len(experts))
        self.logical_to_gpu_index.fill_(-1)
        for i, e in enumerate(experts):
            self.logical_to_gpu_index[e] = base + i
        self.logical_to_gpu_index_cuda.copy_(self.logical_to_gpu_index)
        if self.ondevice:
            # Keep the device keep-warm state coherent too: decide/decide_bounded read _slot_expert_d to
            # pick eviction victims, and a stale slot->expert map makes them clear the WRONG expert_slot
            # entries — a token then runs with another expert's weights (silent corruption).
            self._slot_expert_d.copy_(
                torch.tensor(self.slot_expert, dtype=torch.int32, device=self.device)
            )
            self._slot_lastuse_d.zero_()


def setup_pager(method, layer) -> ExpertPager:
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
        cold_backing=getattr(method, "cold_backing", "ram"),
        cold_dir=getattr(method, "cold_dir", None),
    )

    layer_idx = getattr(layer, "layer_id", getattr(layer, "layer_idx", 0))
    model_path = get_global_server_args().model_path
    try:
        # One ExpertFill per quant format, selected by predicate (fills/). The registry keys on the
        # store's paged params + the checkpoint quant_method (gptq vs awq register identical keys).
        # A fill may return a resident full-E scalar table (nvfp4) to stash on the method.
        quant_method = checkpoint_quant_method(model_path)
        resident = select_fill(store, quant_method).fill(
            store, model_path, layer_idx, dev
        )
        if resident is not None:
            method._nvfp4_full_e = resident
    except KeyError as e:
        raise RuntimeError(
            f"[paged-experts] checkpoint tensor {e} not found for layer {layer_idx}: the fill expects "
            "per-expert {gate,up,down}_proj tensors; fused-expert checkpoint layouts (e.g. a packed "
            "experts.gate_up_proj) are unsupported."
        ) from e
    logger.debug(
        "[paged-experts] L%d host store filled: E=%d, %d tensors %s",
        layer_idx,
        store.E,
        len(store.gpu),
        list(store.gpu),
    )
    pager = ExpertPager(store=store, eviction=getattr(method, "eviction", "lru"))
    if method._placement.needs_ondevice_store:
        pager.setup_ondevice()  # captured path: device-resident decide + UVA gather
    # Layer-ordered registry (setup runs per layer in model order): the wave path uses it to prefetch
    # the NEXT layer's cold tier while the current layer transfers/computes. A weight reload
    # (update_weights_from_disk) re-runs setup on the same method — REPLACE the old pager in place so
    # the registry doesn't grow a stale duplicate holding a second full host store.
    old_pager = getattr(method, "_pager", None)
    if old_pager is not None and old_pager in _REPLAY_PAGERS:
        # weight reload: stop the post-replay hook from polling the stale pager (its miss-vec slot
        # stays allocated — bounded by _MISS_VEC_CAP — but is never read again)
        _REPLAY_PAGERS.remove(old_pager)
    if old_pager is not None and 0 <= getattr(old_pager, "_layer_ord", -1) < len(
        _ALL_PAGERS
    ):
        pager._layer_ord = old_pager._layer_ord
        _ALL_PAGERS[pager._layer_ord] = pager
    else:
        pager._layer_ord = len(_ALL_PAGERS)
        _ALL_PAGERS.append(pager)
    return pager


def next_layer_pager(pager) -> Optional[ExpertPager]:
    """The pager of the next MoE layer in model order (None at the last layer / unknown order)."""
    i = getattr(pager, "_layer_ord", -1)
    if 0 <= i < len(_ALL_PAGERS) - 1:
        return _ALL_PAGERS[i + 1]
    return None
