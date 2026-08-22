"""Paged-experts forward.

Per step the active experts are paged into the K-slot pool and the real fused-MoE GEMM runs over it, in
two regimes:

* ``distinct active experts <= K``: keep-warm. Page only the misses (resident experts are reused across
  steps), remap, one GEMM.
* ``distinct active experts > K`` (e.g. prefill, or batched decode): the pool can't hold them at once, so
  serve them in ``ceil(distinct / K)`` **waves** — each wave pages <=K experts, masks the routing to that
  wave, runs the GEMM, and the per-wave partials are **summed**. Each active expert is in exactly one wave
  and out-of-wave experts are masked to weight 0, so the sum equals the full MoE output (lossless).

*Where* the decision + page-in run — host (eager) or on the GPU inside the captured decode graph — is a
``Placement`` strategy (``placement.py``); ``paged_apply`` just dispatches to it. The shared building
blocks (``mask_and_remap_expert_ids``, ``_gemm_hidden``, and the two wave helpers) live here.

Routing stays E-wide; only the table is K.
"""

from __future__ import annotations

import contextlib
import os
import threading

import torch

# Opt-in (SGLANG_PE_SHARED_OVERLAP): default OFF runs the registered work inline (serial, current
# behavior); ON runs it on the compute stream overlapped with the routed page-in. Gated so the feature
# ships dark and gives a clean A/B.
_SHARED_OVERLAP_ON = os.environ.get("SGLANG_PE_SHARED_OVERLAP") not in (None, "", "0")


# SGLANG_PE_WAVE_SKIP=1: in the wave path, leave out-of-wave pairs at id -1 instead of clamping them to
# slot 0, so the aligner drops them and marlin never GEMMs them (see the flag's note in fused_marlin_moe).
# Each wave otherwise GEMMs the WHOLE batch and keeps ~1/nwaves of it: measured at bs=1024, marlin's
# us/CALL is constant (500 at 6 waves, 513 at 4) while total GEMM ms/step tracks nwaves exactly (288 ->
# 197 ms for 6 -> 4 waves), so the GEMM -- 60% of the step -- does ~nwaves x the necessary work.
# Marlin-only: the flag makes fused_marlin_moe pass ignore_invalid_expert; other runners still need the
# slot-0 clamp, so keep-warm callers never set wave_skip and the flag stays opt-in.
_WAVE_SKIP = os.environ.get("SGLANG_PE_WAVE_SKIP", "0") != "0"

# Router-independent-work overlap (paged-experts #4): a model with a SHARED expert wraps its paged-experts
# call in ``shared_expert_overlap(fn)``; the EAGER keep-warm placement then runs ``fn`` (the shared-expert
# GEMM) on the compute stream while the routed page-in transfers on a dedicated stream, hiding
# min(shared_gemm, transfer) per missing layer. Transparent no-op otherwise: on the captured path, when
# there are no misses, or when the model isn't paged, ``handle.result`` stays None and the caller runs
# ``fn`` itself — so correctness never depends on the overlap firing.
_OVERLAP = threading.local()
_OVERLAP_STREAMS: dict = {}


class OverlapHandle:
    __slots__ = ("fn", "result")

    def __init__(self, fn):
        self.fn = fn
        self.result = (
            None  # the placement sets this iff it ran fn; else None -> caller runs fn
        )


@contextlib.contextmanager
def shared_expert_overlap(fn):
    """Register ``fn`` (router-independent work) for the paged keep-warm path to overlap with the routed
    page-in. Yields a handle whose ``.result`` is ``fn()``'s output when the overlap ran it, else None.
    """
    h = OverlapHandle(fn)
    prev = getattr(_OVERLAP, "handle", None)
    _OVERLAP.handle = h
    try:
        yield h
    finally:
        _OVERLAP.handle = prev


def _current_overlap():
    """The overlap handle registered for the in-flight paged call, or None."""
    return getattr(_OVERLAP, "handle", None)


def _overlap_stream(device):
    """Lazy per-device transfer stream for the shared-expert overlap."""
    key = torch.device(device).index or 0
    s = _OVERLAP_STREAMS.get(key)
    if s is None:
        s = torch.cuda.Stream(device=device)
        _OVERLAP_STREAMS[key] = s
    return s


def mask_and_remap_expert_ids(
    topk_ids: torch.Tensor, logical_to_gpu_index: torch.Tensor
) -> torch.Tensor:
    """Logical expert ids -> GPU slot ids; non-resident experts map to -1 (masked below).
    ``logical_to_gpu_index[e]`` is the slot of expert e (-1 if absent)."""
    return logical_to_gpu_index[topk_ids]


def _refresh_nvfp4_scalars(method, layer, logical_to_slot=None):
    """nvfp4: scatter the resident full-E per-expert scalars (g*_alphas, w*_input_scale_quant) into the
    K slots by the live residency map, so slot s carries the scalar of whatever logical expert is paged
    there. The big weights + swizzled block scales page normally; only these sub-8-byte scalars need the
    per-step refresh (they can't ride the pinned gather). No-op for every other quant method.

    ``logical_to_slot`` overrides the map used by the eager path. The wave path (``_wave_apply``) positions
    each wave's weights via a LOCAL logical->slot map and leaves ``pager.logical_to_gpu_index_cuda`` stale
    until ``set_residency`` at the end — so the scalars MUST be scattered by that same local map, else slot
    s gets the right weight but a DIFFERENT expert's alpha (wrong output magnitude, confident-wrong logits;
    it is the multi-token verify batch, not single-token decode, that trips this).
    """
    fe = getattr(method, "_nvfp4_full_e", None)
    if fe is None:
        return
    pager = method._pager
    s2l = getattr(pager, "_slot_expert_d", None)
    if logical_to_slot is None and s2l is not None:
        # Captured / on-device path: a FIXED-[K] gather by the slot->logical map (updated in-graph by
        # the decide kernel) — capture-safe, re-read each replay. Only valid when the ON-DEVICE decide
        # positioned the weights (captured keep-warm / on-device waves), i.e. no explicit wave map was
        # passed. Empty slots (-1) clamp to 0; their GEMM output is masked out downstream, so the
        # borrowed scalar is inert.
        # Fused refresh: ONE index_select gathers all nvfp4 scalars into a persistent buffer, then copy
        # each row into its K-slot param — replaces the per-scalar clamp/long/gather/copy chain
        # (~10 launches + ~6 allocs -> ~7 launches, 0 per-step allocs). Buffers built once (shapes are
        # graph-stable: K slots, fixed scalar set). Bit-identical to the old ``full[s2l.clamp(0)]`` gather.
        if not all(full.dim() > 0 for full in fe.values()):
            # Some resident scalars are 0-dim per-tensor (input_scale_quant): they cannot be stacked or
            # gathered per-expert, and torch.stack below would raise. Per-tensor loop handles both.
            idx0 = s2l.clamp(min=0).long()
            for nm, full in fe.items():
                tgt = getattr(layer, nm).data
                tgt.copy_(full) if full.dim() == 0 else tgt.copy_(full[idx0])
            return
        idx = getattr(method, "_nvfp4_idx", None)
        if idx is None:
            method._nvfp4_names = list(fe.keys())
            method._nvfp4_stacked = torch.stack(
                [fe[n] for n in method._nvfp4_names]
            )  # [ntens, E]
            method._nvfp4_idx = idx = torch.empty_like(s2l, dtype=torch.long)
            method._nvfp4_gathered = torch.empty(
                (len(method._nvfp4_names), s2l.numel()),
                dtype=method._nvfp4_stacked.dtype,
                device=s2l.device,
            )
        idx.copy_(s2l)  # int32 slot->logical -> int64 index buffer (no alloc)
        idx.clamp_(
            min=0
        )  # empty slots (-1) -> 0; their GEMM output is masked out downstream
        torch.index_select(method._nvfp4_stacked, 1, idx, out=method._nvfp4_gathered)
        for i, nm in enumerate(method._nvfp4_names):
            getattr(layer, nm).data.copy_(method._nvfp4_gathered[i])
    else:
        # Eager scatter by the logical->slot map. The HOST wave path (_wave_apply) positions weights by
        # its LOCAL l2g (host page_in, not the on-device decide) and passes it as logical_to_slot — that
        # map, NOT the on-device _slot_expert_d (which the host wave leaves stale, and which can even
        # hold an out-of-range logical id from a prior spec-verify step -> a scalar-gather OOB), is what
        # matches where the weights landed. Falls back to the live pager map for the eager keep-warm path
        # (logical_to_slot=None, s2l=None). Boolean-mask indexing is data-dependent-shape (NOT
        # capturable), but _wave_apply is only ever reached OFF the capture path, so this is safe.
        l2g = (
            logical_to_slot
            if logical_to_slot is not None
            else pager.logical_to_gpu_index_cuda
        )  # [E] int32: slot of each logical expert, -1 if not
        resident = l2g >= 0
        slots = l2g[resident].long()
        for nm, full in fe.items():
            tgt = getattr(layer, nm).data
            if full.dim() == 0:
                tgt.copy_(
                    full
                )  # per-tensor scalar (input_scale_quant is 0-dim) — no per-slot scatter
            else:
                tgt[slots] = full[resident]


# SGLANG_PE_PARANOIA=1: per-step residency audit for the nvfp4 debug hunt. After the scalar refresh,
# verify for every resident logical expert that (a) the GPU slot's weight bytes equal the authoritative
# cold-store row and (b) the slot's g1_alpha equals the full-E table entry. Whichever fires first, and at
# which step, localises the corruption (fill vs page_in vs refresh-map skew). Eager path only; heavy.
_PARANOIA = os.environ.get("SGLANG_PE_PARANOIA", "0") != "0"
_PARANOIA_STEP = [0]


def _paranoia_audit(method, layer):
    pager = getattr(method, "_pager", None)
    fe = getattr(method, "_nvfp4_full_e", None)
    if pager is None or fe is None or torch.cuda.is_current_stream_capturing():
        return
    _PARANOIA_STEP[0] += 1
    step = _PARANOIA_STEP[0]
    store = pager.store
    cold = getattr(store, "_cold", None)
    l2g = pager.logical_to_gpu_index  # host [E] logical -> slot
    import logging

    log = logging.getLogger(__name__)
    checked = 0
    for e in range(store.E):
        s_ = int(l2g[e])
        if s_ < 0:
            continue
        checked += 1
        # (a) weight bytes: GPU slot vs authoritative cold row
        if cold is not None and "w13_weight" in cold:
            gslot = store.gpu["w13_weight"].data[s_].detach().cpu().view(torch.uint8)
            cref = cold["w13_weight"][e].view(torch.uint8)
            if not torch.equal(gslot, cref):
                nz = (gslot != cref).nonzero().flatten()
                log.error(
                    "[paranoia] step %d L%s: WEIGHT bytes diverge: logical %d in slot %d, "
                    "%d/%d bytes differ, first at offset %d",
                    step,
                    getattr(layer, "layer_id", "?"),
                    e,
                    s_,
                    nz.numel(),
                    gslot.numel(),
                    int(nz[0]),
                )
                return
        # (b) scalar: slot alpha vs full-E table
        ga = fe.get("g1_alphas")
        if ga is not None and ga.dim() > 0:
            got = float(layer.g1_alphas.data[s_])
            want = float(ga[e])
            if got != want:
                log.error(
                    "[paranoia] step %d L%s: SCALAR diverges: logical %d in slot %d, "
                    "g1_alpha got %.6g want %.6g",
                    step,
                    getattr(layer, "layer_id", "?"),
                    e,
                    s_,
                    got,
                    want,
                )
                return
    if step % 200 == 0:
        log.warning(
            "[paranoia] step %d: %d resident slots verified clean", step, checked
        )


def _gemm_hidden(
    method,
    layer,
    dispatch_output,
    remap: torch.Tensor,
    *,
    clone_hidden: bool,
    logical_to_slot: torch.Tensor = None,
    wave_skip: bool = False,
):
    """Run the base fused-MoE over the K-slot pool for one (wave's) remap, returning the hidden output.

    Zero the routing weight where the expert is masked out (remap == -1) so its contribution is provably
    0, and clamp masked ids -1 -> 0 (slot-0 output x 0 = exact 0; required for marlin's moe_align binning,
    bit-identical for triton). ``clone_hidden`` is set on the wave path, where the same input is reused
    across waves and the base method may consume it in place. ``logical_to_slot`` is the wave's local
    logical->slot map, threaded to the nvfp4 scalar refresh (see ``_refresh_nvfp4_scalars``).
    """
    topk_output = dispatch_output.topk_output
    tw = topk_output.topk_weights
    mask = (
        remap >= 0
    )  # compute once; scalar `0` in where avoids two throwaway zeros_like allocs
    masked_tw = torch.where(mask, tw, 0.0)
    if wave_skip and _WAVE_SKIP:
        # Leave masked ids NEGATIVE: the aligner drops them (ignore_invalid_expert), so no GEMM block
        # covers them, their pre-zeroed cache rows stay zero, and masked_tw already zeroes the sum.
        safe_ids = remap
    else:
        safe_ids = torch.where(mask, remap, 0)
    hidden = dispatch_output.hidden_states
    md = dispatch_output._replace(
        hidden_states=hidden.clone() if clone_hidden else hidden,
        topk_output=topk_output._replace(topk_ids=safe_ids, topk_weights=masked_tw),
    )
    _refresh_nvfp4_scalars(method, layer, logical_to_slot=logical_to_slot)
    if _PARANOIA:
        _paranoia_audit(method, layer)
    out = method.base_method.apply(layer, md)
    return out.hidden_states if hasattr(out, "hidden_states") else out


def _gemm_hidden_fused(
    method, layer, dispatch_output, safe_ids, masked_tw, *, clone_hidden: bool
):
    """Like :func:`_gemm_hidden`, but the masking/remap chain was already computed by the fused
    ``remap_mask`` kernel (``pager.remap_mask_ondevice``) — just swap the buffers in and run the GEMM.
    """
    topk_output = dispatch_output.topk_output
    hidden = dispatch_output.hidden_states
    md = dispatch_output._replace(
        hidden_states=hidden.clone() if clone_hidden else hidden,
        topk_output=topk_output._replace(topk_ids=safe_ids, topk_weights=masked_tw),
    )
    _refresh_nvfp4_scalars(method, layer)
    out = method.base_method.apply(layer, md)
    return out.hidden_states if hasattr(out, "hidden_states") else out


def _wave_apply(method, layer, dispatch_output, topk_ids: torch.Tensor, distinct):
    """Serve > K distinct experts in waves; sum the per-wave partials (lossless).

    DOUBLE-BUFFERED: the K slots are split into two banks of K//2 and the waves ping-pong between them —
    wave w+1's page-in (hot ``transfer_kv`` + cold staged H2D, on a dedicated transfer stream) overlaps
    wave w's GEMM (compute stream), and the CPU-side cold gather (which faults the disk tier in) runs
    while the GPU is busy. Events sequence the two hazards: a bank's slots are rewritten only after the
    GEMM that read them (gemm-done), and a bank's staging buffer is refilled only after its H2D drained
    (h2d-done). Each active expert is still served in exactly one wave, so the partial-sum stays
    lossless.
    """
    pager = method._pager
    K, E, dev = pager.K, pager.E, pager.device
    store = pager.store
    half = K // 2
    # Bank (double-buffer) the host wave path only where the overlap pays: the DISK cold tier, whose
    # CPU-side gather (page faults) hides under the previous wave's GEMM. On RAM-windowed stores the
    # waves are transfer-bound with little to hide, and halving the wave size nearly doubles the
    # per-wave fixed costs (masked-GEMM pass + transfer launches) — measured net-negative there, so
    # those keep serial full-K waves.
    banked = (
        half > 0
        and bool(getattr(store, "_cold_mm", {}))
        and not torch.cuda.is_current_stream_capturing()
    )
    wave_k = half if banked else K
    groups = [distinct[w : w + wave_k] for w in range(0, len(distinct), wave_k)]
    rolling = False
    if hasattr(store, "prefetch_cold"):
        # Disk cold tier read-ahead. When the cold tier fits comfortably in the page cache, queue the
        # WHOLE step's reads up front (one deep madvise batch) and the NEXT layer's cold tier behind it.
        # When it does NOT (a true >RAM store), a whole-step WILLNEED evicts its own tail before the
        # later waves arrive — roll the read-ahead one wave ahead of the gather instead.
        from sglang.srt.layers.moe.paged_experts.store import _host_available_bytes

        # #10: mm_total is a store invariant (the cold mmaps are fixed at setup) — cache it instead of
        # re-summing every wave-layer; and sample /proc/meminfo (avail) at most once per few dozen calls
        # rather than every layer — the rolling threshold (mm_total > avail//2) moves slowly, and the
        # open+parse was the latency-sensitive part. Both were re-derived on every _wave_apply call.
        mm_total = getattr(store, "_mm_total_bytes", None)
        if mm_total is None:
            mm_total = sum(len(m) for m in getattr(store, "_cold_mm", {}).values())
            store._mm_total_bytes = mm_total
        ctr = getattr(store, "_avail_ctr", 0)
        if ctr % 48 == 0 or not hasattr(store, "_rolling"):
            avail = _host_available_bytes()
            store._rolling = bool(mm_total) and bool(avail) and mm_total > avail // 2
        store._avail_ctr = ctr + 1
        rolling = store._rolling
        if rolling:
            store.prefetch_cold(groups[0])
        else:
            store.prefetch_cold(distinct)
            from sglang.srt.layers.moe.paged_experts.pager import next_layer_pager

            nxt = next_layer_pager(pager)
            if nxt is not None and hasattr(nxt.store, "prefetch_cold_all"):
                nxt.store.prefetch_cold_all()
        store._step_prefetched = (
            True  # page_in skips its per-wave re-prefetch of the same ranges
        )
    l2g = torch.full((E,), -1, dtype=torch.int32, device=dev)
    cs = torch.cuda.current_stream()
    if banked:
        ts, ev_h2d, ev_gemm, _ = pager.wave_ctx()
        ts.wait_stream(cs)
    out = None
    group, base = [], 0
    for i, group in enumerate(groups):
        b = i & 1
        base = b * half if banked else 0
        if rolling and i + 1 < len(groups):
            store.prefetch_cold(groups[i + 1])  # keep the disk one wave ahead
        if banked:
            # staging buffers for bank b are free once wave i-2's H2D drained (CPU wait: the gather
            # below writes them from the CPU side)
            ev_h2d[b].synchronize()
            with torch.cuda.stream(ts):
                # the plan tensors MUST be created on ts: an arange enqueued on the compute stream is
                # not synchronized with ts, and the transfer kernels reading a not-yet-materialized
                # dst was a real (intermittent, load-dependent) illegal-memory-access
                src = torch.tensor(group, dtype=torch.int64, device=dev)
                dst = torch.arange(
                    base, base + len(group), dtype=torch.int64, device=dev
                )
                # bank b's slots are free once wave i-2's GEMM finished reading them
                ts.wait_event(ev_gemm[b])
                pager.page_in(
                    src,
                    dst,
                    stage_bank=b,
                    async_h2d=True,
                    src_host=group,
                    dst_host=list(range(base, base + len(group))),
                )
                ev_h2d[b].record(ts)
            cs.wait_event(ev_h2d[b])
        else:
            src = torch.tensor(group, dtype=torch.int64, device=dev)
            dst = torch.arange(base, base + len(group), dtype=torch.int64, device=dev)
            pager.page_in(
                src,
                dst,
                src_host=group,
                dst_host=list(range(base, base + len(group))),
            )
        l2g.fill_(-1)
        l2g[src] = dst.to(torch.int32)
        partial = _gemm_hidden(
            method,
            layer,
            dispatch_output,
            l2g[topk_ids],
            clone_hidden=True,
            logical_to_slot=l2g,  # nvfp4: scatter scalars by THIS wave's map, not the stale pager map
        )
        if banked:
            ev_gemm[b].record(cs)
        out = partial if out is None else out + partial
    if banked:
        # the NEXT step's page_in gathers into the shared staging buffers from the CPU side, which no
        # stream ordering protects — drain this step's H2D before returning (GEMMs stay async)
        ev_h2d[0].synchronize()
        ev_h2d[1].synchronize()
    pager.set_residency(
        group, base=base
    )  # leave the maps consistent for the next keep-warm step
    if hasattr(store, "_step_prefetched"):
        store._step_prefetched = False
    return out


def _ondevice_wave_apply(method, layer, dispatch_output, topk_ids):
    """On-device static-wave path (distinct > K, e.g. prefill): waves planned+gathered on-device,
    GEMM'd and summed. No host sync. Resyncs the keep-warm state to the last wave so a following decode
    step is consistent. Lossless (each active expert is served in exactly one wave).

    Outside graph capture the waves are DOUBLE-BUFFERED (banked K//2 slots, decide+gather on a transfer
    stream overlapping the previous wave's GEMM, per-bank idx buffers so the next decide can't race the
    current remap). Under capture the serial full-K wave path runs unchanged — the cross-stream event
    choreography is not worth capturing.
    """
    pager = method._pager
    E, K = pager.E, pager.K
    half = K // 2
    banked = half > 0 and not torch.cuda.is_current_stream_capturing()
    if not banked:
        nwaves = (E + K - 1) // K
        out = None
        tw = dispatch_output.topk_output.topk_weights
        for w in range(nwaves):
            pager.decide_and_page_wave_ondevice(topk_ids, w)
            # Fuse the remap/mask into one kernel (as the captured keep-warm path does) instead of the
            # ~6-node python chain per wave: decide_and_page_wave_ondevice sets _topk_i32 (wave 0) and
            # updates logical_to_gpu_index_cuda for THIS wave — exactly what remap_mask_ondevice reads.
            # nvfp4 stays on the s2l gather (no logical_to_slot), identical to the unfused call below.
            fused = pager.remap_mask_ondevice(topk_ids, tw)
            if fused is not None:
                safe_ids, masked_tw = fused
                partial = _gemm_hidden_fused(
                    method,
                    layer,
                    dispatch_output,
                    safe_ids,
                    masked_tw,
                    clone_hidden=True,
                )
            else:
                remap = mask_and_remap_expert_ids(
                    topk_ids, pager.logical_to_gpu_index_cuda
                )
                partial = _gemm_hidden(
                    method,
                    layer,
                    dispatch_output,
                    remap,
                    clone_hidden=True,
                    wave_skip=True,
                )
            out = partial if out is None else out + partial
        lo = (nwaves - 1) * K
        pager.resync_residency_ondevice(lo, min(K, E - lo))
        return out

    ts, ev_h2d, ev_gemm, idx_banks = pager.wave_ctx()
    cs = torch.cuda.current_stream()
    ts.wait_stream(
        cs
    )  # topk_ids (and wave 0's _prep copy) depend on compute-stream work
    nwaves = (E + half - 1) // half
    out = None
    for w in range(nwaves):
        b = w & 1
        with torch.cuda.stream(ts):
            ts.wait_event(ev_gemm[b])  # bank b free once wave w-2's GEMM read it
            pager.decide_and_page_wave_ondevice(
                topk_ids, w, wave_k=half, slot_base=b * half, idx_out=idx_banks[b]
            )
            ev_h2d[b].record(ts)
        cs.wait_event(ev_h2d[b])
        remap = mask_and_remap_expert_ids(topk_ids, idx_banks[b])
        partial = _gemm_hidden(
            method, layer, dispatch_output, remap, clone_hidden=True, wave_skip=True
        )
        ev_gemm[b].record(cs)
        out = partial if out is None else out + partial
    lo = (nwaves - 1) * half
    pager.resync_residency_ondevice(lo, E - lo, slot_base=((nwaves - 1) & 1) * half)
    return out


def paged_apply(method, layer, dispatch_output):
    """Dispatch the step to the method's decode placement (eager host vs captured on-device).

    The placement (``method._placement``) owns the decide + page-in flow; both end in ``_gemm_hidden``
    over the K-slot pool. See ``placement.py``.
    """
    return method._placement.apply(method, layer, dispatch_output)
