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

import torch


def mask_and_remap_expert_ids(
    topk_ids: torch.Tensor, logical_to_gpu_index: torch.Tensor
) -> torch.Tensor:
    """Logical expert ids -> GPU slot ids; non-resident experts map to -1 (masked below).
    ``logical_to_gpu_index[e]`` is the slot of expert e (-1 if absent)."""
    return logical_to_gpu_index[topk_ids]


def _gemm_hidden(
    method, layer, dispatch_output, remap: torch.Tensor, *, clone_hidden: bool
):
    """Run the base fused-MoE over the K-slot pool for one (wave's) remap, returning the hidden output.

    Zero the routing weight where the expert is masked out (remap == -1) so its contribution is provably
    0, and clamp masked ids -1 -> 0 (slot-0 output x 0 = exact 0; required for marlin's moe_align binning,
    bit-identical for triton). ``clone_hidden`` is set on the wave path, where the same input is reused
    across waves and the base method may consume it in place.
    """
    topk_output = dispatch_output.topk_output
    tw = topk_output.topk_weights
    masked_tw = torch.where(remap >= 0, tw, torch.zeros_like(tw))
    safe_ids = torch.where(remap >= 0, remap, torch.zeros_like(remap))
    hidden = dispatch_output.hidden_states
    md = dispatch_output._replace(
        hidden_states=hidden.clone() if clone_hidden else hidden,
        topk_output=topk_output._replace(topk_ids=safe_ids, topk_weights=masked_tw),
    )
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
    # per-wave fixed costs (masked-GEMM pass + transfer launches) — measured net-negative (fp8-30B:
    # 934 -> 603 tok/s prefill), so those keep serial full-K waves.
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

        mm_total = sum(len(m) for m in getattr(store, "_cold_mm", {}).values())
        avail = _host_available_bytes()
        rolling = bool(mm_total) and bool(avail) and mm_total > avail // 2
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
        src = torch.tensor(group, dtype=torch.int64, device=dev)
        dst = torch.arange(base, base + len(group), dtype=torch.int64, device=dev)
        if banked:
            # staging buffers for bank b are free once wave i-2's H2D drained (CPU wait: the gather
            # below writes them from the CPU side)
            ev_h2d[b].synchronize()
            with torch.cuda.stream(ts):
                # bank b's slots are free once wave i-2's GEMM finished reading them
                ts.wait_event(ev_gemm[b])
                pager.page_in(src, dst, stage_bank=b, async_h2d=True, src_host=group)
                ev_h2d[b].record(ts)
            cs.wait_event(ev_h2d[b])
        else:
            pager.page_in(src, dst, src_host=group)
        l2g.fill_(-1)
        l2g[src] = dst.to(torch.int32)
        partial = _gemm_hidden(
            method, layer, dispatch_output, l2g[topk_ids], clone_hidden=True
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
        for w in range(nwaves):
            pager.decide_and_page_wave_ondevice(topk_ids, w)
            remap = mask_and_remap_expert_ids(topk_ids, pager.logical_to_gpu_index_cuda)
            partial = _gemm_hidden(
                method, layer, dispatch_output, remap, clone_hidden=True
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
        partial = _gemm_hidden(method, layer, dispatch_output, remap, clone_hidden=True)
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
