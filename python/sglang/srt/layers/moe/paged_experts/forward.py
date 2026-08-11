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

import os

import torch

# Adaptive top-k: stream only the N highest-weight of the router's top_k experts (env-gated). The decode
# floor is bytes-moved-per-token = top_k x layers x expert_bytes; cutting top_k -> N cuts the disk traffic
# ~N/top_k. Dropped experts get weight 0 (no GEMM contribution) and their id duplicates the row's #1 expert
# (adds no distinct expert to page), and the kept weights are renormalized to preserve the row magnitude.
# Approximation (drops low-weight experts) — off by default; set e.g. SGLANG_PAGED_STREAM_TOPK=4.
_STREAM_TOPK = int(os.environ.get("SGLANG_PAGED_STREAM_TOPK", "0"))
_STREAM_TOPK_LOGGED = False
# VARIABLE-k: keep the smallest prefix of routed experts whose cumulative NORMALIZED gate weight crosses
# this threshold, per token (0 = off). A better dial than fixed-k: an easy token may satisfy the mass with
# 2 experts, a hard one with 6 — average-k floats with difficulty. Takes precedence over fixed _STREAM_TOPK.
_STREAM_GATE = float(os.environ.get("SGLANG_PAGED_STREAM_GATE", "0"))
_STREAM_GATE_LOGGED = False
_STREAM_GATE_KSUM = [0.0, 0]  # running (sum_kept, n_tokens) for an observed-average-k log

# Stage A gate for per-layer MoE mini-graph capture (docs/design/paged-experts-perlayer-capture.md):
# probe whether a MANUAL torch.cuda.CUDAGraph can capture base_method.apply standalone (outside sglang's
# backend) and replay correctly. Fires ONCE on the first eligible decode _gemm_hidden, logs PASS/FAIL, then
# falls through to the normal eager apply. Default off — pure diagnostic, never on the serving path.
_PERLAYER_CAPTURE_PROBE = os.environ.get("SGLANG_PERLAYER_CAPTURE_PROBE", "0") == "1"
_PERLAYER_PROBE_DONE = False

# Per-layer MoE mini-graph capture (docs/design/paged-experts-perlayer-capture.md). Captures the CAPTURABLE
# per-wave compute (s2l scalar refresh + remap + grouped GEMM) into a manual torch.cuda.CUDAGraph per layer,
# replayed per wave — killing the ~20% compute-orchestration CPU without the full-forward VRAM wall. Pure
# function (fixed in-buffers -> fixed `partial` out-buffer); eager accumulate + eager page-in outside.
# Default OFF; K=1 + bs=1 + nvfp4 only. Self-gated: falls back to eager on any captured-vs-eager mismatch.
_PERLAYER_CAPTURE = os.environ.get("SGLANG_PERLAYER_CAPTURE", "0") == "1"


def _maybe_truncate_topk(dispatch_output) -> None:
    """In-place truncate the routing per token. Gate-threshold mode (variable k) if _STREAM_GATE>0,
    else fixed top-``_STREAM_TOPK``. No-op if both unset."""
    if _STREAM_GATE > 0:
        _truncate_by_gate(dispatch_output)
        return
    if _STREAM_TOPK <= 0:
        return
    to = dispatch_output.topk_output
    ids, w = to.topk_ids, to.topk_weights
    tk = ids.shape[-1]
    if _STREAM_TOPK >= tk:
        return
    orig_sum = w.sum(dim=-1, keepdim=True)
    topw, topi = torch.topk(w, _STREAM_TOPK, dim=-1)  # [T,N] weights + their positions
    kept_ids = torch.gather(ids, -1, topi)
    top1 = kept_ids[:, :1].expand_as(ids)  # row's #1 expert, to duplicate into dropped slots
    keep_mask = torch.zeros_like(w, dtype=torch.bool).scatter_(-1, topi, True)
    keep_w = topw / topw.sum(dim=-1, keepdim=True) * orig_sum  # renormalize -> preserve row magnitude
    new_w = torch.zeros_like(w).scatter_(-1, topi, keep_w)
    new_ids = torch.where(keep_mask, ids, top1)
    w.copy_(new_w)
    ids.copy_(new_ids)
    global _STREAM_TOPK_LOGGED
    if not _STREAM_TOPK_LOGGED:
        _STREAM_TOPK_LOGGED = True
        import logging

        logging.getLogger(__name__).info(
            "[paged-experts] adaptive top-k ACTIVE: streaming top-%d of %d routed experts "
            "(~%.0f%% of per-token disk traffic)",
            _STREAM_TOPK,
            tk,
            100.0 * _STREAM_TOPK / tk,
        )


def _truncate_by_gate(dispatch_output) -> None:
    """Variable-k: per token, keep the smallest prefix of routed experts whose cumulative normalized
    gate weight reaches _STREAM_GATE (always keep the #1 expert). Dropped experts get weight 0 and their
    id duplicates the row's #1 expert (adds no distinct expert to page); kept weights renormalized to
    preserve row magnitude. Works for the sigmoid router (weights need not sum to 1 — we normalize)."""
    to = dispatch_output.topk_output
    ids, w = to.topk_ids, to.topk_weights
    orig_sum = w.sum(dim=-1, keepdim=True)
    sw, si = torch.sort(w, dim=-1, descending=True)  # weights desc + their original positions
    norm = sw / sw.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    cum_before = torch.cumsum(norm, dim=-1) - norm  # mass strictly BEFORE this expert
    keep_sorted = cum_before < _STREAM_GATE  # include the expert that crosses the threshold
    keep_sorted[:, 0] = True  # never drop the top expert
    keep_mask = torch.zeros_like(w, dtype=torch.bool).scatter_(-1, si, keep_sorted)
    kept_w = torch.where(keep_mask, w, torch.zeros_like(w))
    kept_w = kept_w / kept_w.sum(dim=-1, keepdim=True).clamp_min(1e-9) * orig_sum
    argmax = w.argmax(dim=-1, keepdim=True)
    top1 = ids.gather(-1, argmax).expand_as(ids)
    new_ids = torch.where(keep_mask, ids, top1)
    w.copy_(kept_w)
    ids.copy_(new_ids)
    _STREAM_GATE_KSUM[0] += float(keep_sorted.sum().item())
    _STREAM_GATE_KSUM[1] += ids.shape[0]
    global _STREAM_GATE_LOGGED
    if not _STREAM_GATE_LOGGED or _STREAM_GATE_KSUM[1] % 6000 == 0:
        _STREAM_GATE_LOGGED = True
        import logging

        avg = _STREAM_GATE_KSUM[0] / max(1, _STREAM_GATE_KSUM[1])
        logging.getLogger(__name__).info(
            "[paged-experts] gate-threshold k ACTIVE: Sigma-weight>=%.2f -> observed avg-k=%.2f of %d",
            _STREAM_GATE,
            avg,
            ids.shape[-1],
        )


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
        idx = s2l.clamp(min=0).long()
        for nm, full in fe.items():
            tgt = getattr(layer, nm).data
            if full.dim() == 0:
                tgt.copy_(full)  # per-tensor scalar (input_scale_quant is 0-dim) — no per-slot gather
            else:
                tgt.copy_(full[idx])
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
                tgt.copy_(full)  # per-tensor scalar (input_scale_quant is 0-dim) — no per-slot scatter
            else:
                tgt[slots] = full[resident]


def _perlayer_capture_probe(method, layer, md) -> None:
    """Stage A gate (docs/design/paged-experts-perlayer-capture.md): verify a MANUAL torch.cuda.CUDAGraph
    can capture base_method.apply standalone (outside sglang's backend) and replay it correctly with the
    same fixed input buffers. If this passes, the per-layer MoE mini-graph capture is viable (the win that
    sidesteps the full-forward VRAM wall). If it raises, the idea is dead cheaply. Logs PASS/FAIL; never
    propagates (pure diagnostic on a one-shot decode call)."""
    import logging

    log = logging.getLogger(__name__)

    def _apply():
        o = method.base_method.apply(layer, md)
        return o.hidden_states if hasattr(o, "hidden_states") else o

    try:
        # A captured op sequence must be warmed on a side stream first (allocate workspaces, JIT, etc.).
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _ = _apply()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            cap_out = _apply()  # captured; writes a fixed graph-pool output buffer
        g.replay()
        torch.cuda.synchronize()
        eager_out = _apply()  # fresh eager result on the same (unmutated) md
        diff = (cap_out.float() - eager_out.float()).abs().max().item()
        denom = eager_out.float().abs().max().item() + 1e-9
        log.info(
            "[perlayer-probe] PASS: manual CUDAGraph capture+replay of base_method.apply works "
            "(max|Δ|=%.3e rel=%.3e shape=%s) -> per-layer MoE capture is VIABLE",
            diff,
            diff / denom,
            tuple(cap_out.shape),
        )
    except Exception as e:  # noqa: BLE001 — diagnostic must never crash serving
        log.error(
            "[perlayer-probe] FAIL: manual CUDAGraph capture of base_method.apply raised %r -> per-layer "
            "MoE capture is likely DEAD (host sync / dynamic alloc under manual capture; note "
            "expandable_segments can conflict with CUDA graphs)",
            e,
        )


def _gemm_hidden(
    method,
    layer,
    dispatch_output,
    remap: torch.Tensor,
    *,
    clone_hidden: bool,
    logical_to_slot: torch.Tensor = None,
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
    # mask masked-out experts (remap<0) to weight 0 + slot 0; clamp/mul avoid the where+zeros_like
    # temporaries (2 fewer allocs/launches per wave x waves x layers). Same result.
    keep = (remap >= 0).to(tw.dtype)
    masked_tw = tw * keep
    safe_ids = remap.clamp_(min=0)
    hidden = dispatch_output.hidden_states
    _refresh_nvfp4_scalars(method, layer, logical_to_slot=logical_to_slot)
    # LEAN single-expert path: at K=1 with few tokens (decode), skip the grouped-MoE GEMM's per-call
    # binning/offset/workspace setup (the ~0.33 s/expert per-wave overhead that starves the GPU) and run the
    # one resident expert (slot 0) as 2 plain non-grouped nvfp4 GEMMs + silu. Scalars for THIS wave's expert
    # are already in slot 0 (_refresh_nvfp4_scalars above).
    if (
        _LEAN_MOE
        and getattr(method, "_nvfp4_full_e", None) is not None
        and getattr(method._pager, "K", 0) == 1
        and hidden.shape[0] <= _LEAN_TOK_MAX
        and hasattr(layer, "g1_alphas")
    ):
        lean = _lean_single_expert(layer, hidden, masked_tw)
        if lean is not None:
            return lean
    md = dispatch_output._replace(
        hidden_states=hidden.clone() if clone_hidden else hidden,
        topk_output=topk_output._replace(topk_ids=safe_ids, topk_weights=masked_tw),
    )
    global _PERLAYER_PROBE_DONE
    if (
        _PERLAYER_CAPTURE_PROBE
        and not _PERLAYER_PROBE_DONE
        and not torch.cuda.is_current_stream_capturing()
        and hidden.shape[0] <= 4  # decode-shaped only
    ):
        _PERLAYER_PROBE_DONE = True
        _perlayer_capture_probe(method, layer, md)
    if _sT is not None:
        import time as _time

        _g0 = _time.perf_counter_ns()
        out = method.base_method.apply(layer, md)
        _sT("gemm_launch", _time.perf_counter_ns() - _g0)
    else:
        out = method.base_method.apply(layer, md)
    return out.hidden_states if hasattr(out, "hidden_states") else out


try:
    from sglang.srt.layers.moe.paged_experts.inplace_nvfp4_store import (
        _PE_TIMING as _PET,
    )
    from sglang.srt.layers.moe.paged_experts.inplace_nvfp4_store import _stage as _sT_

    _sT = _sT_ if _PET else None
except Exception:
    _sT = None

import time as _time_mod

_perf_ns = _time_mod.perf_counter_ns


# Default OFF: the lean single-expert path is CORRECT but MEASURED SLOWER (2.38 vs 2.02 s/tok) — the grouped
# cutlass_moe_fp4 is already well-fused (silu+mul+quant in one kernel), so the manual 2-GEMM decomposition
# adds more launches than the binning it removes. Kept for reference / larger-K experiments.
_LEAN_MOE = os.environ.get("SGLANG_PAGED_LEAN_MOE", "0") != "0"
_LEAN_TOK_MAX = int(os.environ.get("SGLANG_PAGED_LEAN_TOK_MAX", "8"))


def _lean_single_expert(layer, hidden, masked_tw):
    """Non-grouped nvfp4 MoE for the K=1 single-resident-expert wave: 2 plain fp4 GEMMs + silu, bypassing
    the grouped-MoE binning/offsets/workspace. Slot 0 = the resident expert. Scale/alpha wiring is identical
    to cutlass_moe_fp4: fp4_quantize global_scale = *_input_scale_quant (1/input_scale); fp4_gemm alpha =
    g*_alphas (input_scale*weight_scale_2). Returns None to fall back if the fp4 primitives are unavailable.
    """
    try:
        from sglang.srt.layers.quantization.fp4_utils import fp4_quantize
        # call the cutlass non-grouped fp4 mm DIRECTLY (weight layout [N,K/2]); fp4_gemm()'s auto-dispatch
        # can pick flashinfer mm_fp4 which wants the transposed [K/2,N] layout we don't have.
        from sglang.kernels.jit.nvfp4 import cutlass_scaled_fp4_mm
    except Exception:
        return None
    import torch.nn.functional as F

    sc13 = "w13_blockscale_swizzled" if hasattr(layer, "w13_blockscale_swizzled") else "w13_weight_scale"
    sc2 = "w2_blockscale_swizzled" if hasattr(layer, "w2_blockscale_swizzled") else "w2_weight_scale"
    w13, w2 = layer.w13_weight[0], layer.w2_weight[0]  # [2I,H/2], [H,I/2]
    s13, s2 = getattr(layer, sc13)[0], getattr(layer, sc2)[0]
    two_i, hsz = w13.shape[0], w2.shape[0]
    i = two_i // 2
    dt = hidden.dtype
    f8 = torch.float8_e4m3fn

    def _mm(x, w, wsf, alpha):
        xf, xsf = fp4_quantize(x, alpha[1])  # global_scale = *_input_scale_quant (1/input_scale)
        if xsf.dtype != f8:
            xsf = xsf.view(f8)
        if wsf.dtype != f8:
            wsf = wsf.view(f8)
        return cutlass_scaled_fp4_mm(xf, w, xsf, wsf, alpha[0], dt)  # alpha[0] = g*_alphas (dequant)

    # GEMM1 gate|up, then SwiGLU
    gate_up = _mm(hidden, w13, s13, (layer.g1_alphas[0], layer.w13_input_scale_quant[0]))  # [T,2I]
    act = F.silu(gate_up[:, :i]) * gate_up[:, i:]  # [T,I]
    # GEMM2 down
    down = _mm(act, w2, s2, (layer.g2_alphas[0], layer.w2_input_scale_quant[0]))  # [T,H]
    # router weight for the resident expert (masked_tw nonzero only where the token routes to slot 0)
    return down * masked_tw.sum(dim=-1, keepdim=True).to(dt)


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


def _scratch_fill(pager, bufs, bank, ts, ev_ready, ev_gemm) -> None:
    """Enqueue a full-store read into scratch bank ``bank`` on the transfer stream. The bank is free
    once the GEMM that last read it (two layers ago) completed; the staging pin buffers (windowed cold
    rows) are free once this bank's previous fill drained (CPU wait — the gather writes them host-side).
    """
    ev_ready.synchronize()
    with torch.cuda.stream(ts):
        ts.wait_event(ev_gemm)
        # resident-aware: D2D the K residents out of the layer's own pool, stream only the complement
        if not pager.scratch_fill_resident_aware(bufs):
            pager.store.read_full(bufs, stage_key=bank)
        ev_ready.record(ts)


def _scratch_prefill_apply(method, layer, dispatch_output, topk_ids, distinct=None):
    """Streaming prefill: run this layer's MoE as ONE vanilla E-wide fused-MoE pass out of a full-E
    scratch pool instead of ``ceil(E/K)`` masked waves through the K-slot pool.

    Two global scratch banks ping-pong across layers: while this layer's GEMM computes out of bank
    ``layer_ord & 1``, the NEXT layer's whole expert set streams into the other bank on the transfer
    stream — cross-layer overlap with none of the residency hazards of pre-paging the K-slot pool
    (scratch is not serving state; the K slots and every residency map stay untouched, so decode
    resumes warm after prefill). The layer's paged params are swapped to the scratch views only for
    the duration of the base-method call (every supported backend derives its expert count from the
    weight shapes), and ``topk_ids`` pass through UNREMAPPED — bit-identical math to a fully-resident
    serve. Returns the hidden output, or ``None`` when the scratch pool is unavailable (caller falls
    back to the wave path).
    """
    import os

    pager = method._pager
    if torch.cuda.is_current_stream_capturing():
        return None
    if os.environ.get("SGLANG_PAGED_EXPERTS_SCRATCH", "1") == "0":
        return None  # kill switch (debug / A-B)
    if getattr(method, "_nvfp4_full_e", None) is not None:
        # nvfp4 scalar params (g*_alphas, w*_input_scale_quant) are K-sized and refreshed by the
        # K-slot residency map; the E-wide unremapped scratch pass would index them out of range.
        # Route nvfp4 through the wave path, where the per-wave residency refresh is correct.
        return None
    if distinct is not None and len(distinct) < (6 * pager.E) // 10:
        return None  # sparse big batch: waves move fewer bytes
    store = pager.store
    if getattr(store, "host", None) is None or not store.pinned:
        # full-pin stores only: any cold tier (windowed RAM or disk) needs a per-layer CPU-side
        # staging gather in read_full, which stalls the pipeline on the CPU — measured net-negative
        # (fp8-30B windowed: 1019 -> ~600 tok/s prefill). Windowed stores keep the wave path.
        return None
    ctx = pager.scratch_ctx()
    if ctx is None:
        return None
    bufs, ev_ready, ev_gemm = ctx
    bank = getattr(pager, "_layer_ord", 0) & 1
    ts = pager.wave_ctx()[0]
    cs = torch.cuda.current_stream()
    if not getattr(pager, "_scratch_prefilled", False):
        ts.wait_stream(cs)  # first fill of the pass: order behind enqueued compute
        _scratch_fill(pager, bufs[bank], bank, ts, ev_ready[bank], ev_gemm[bank])
    pager._scratch_prefilled = False
    cs.wait_event(ev_ready[bank])
    gpu = pager.store.gpu
    saved = {name: p.data for name, p in gpu.items()}
    try:
        for name, p in gpu.items():
            p.data = bufs[bank][name]
        hidden = _gemm_hidden(
            method, layer, dispatch_output, topk_ids, clone_hidden=False
        )
    finally:
        for name, p in gpu.items():
            p.data = saved[name]
    ev_gemm[bank].record(cs)
    # stream the NEXT layer's experts into the other bank while its attention runs
    from sglang.srt.layers.moe.paged_experts.pager import next_layer_pager

    nxt = next_layer_pager(pager)
    if nxt is not None and nxt.scratch_ctx() is not None:
        nbank = getattr(nxt, "_layer_ord", 0) & 1
        _scratch_fill(nxt, bufs[nbank], nbank, ts, ev_ready[nbank], ev_gemm[nbank])
        nxt._scratch_prefilled = True
    return hidden


def _pl_capture_body(method, layer, cap):
    """The CAPTURED per-wave compute (pure fn of the fixed buffers): s2l scalar refresh + remap/mask +
    grouped GEMM -> writes cap['partial']. Reads cap['l2g']/cap['topk_ids']/cap['topk_weights']/cap['hidden']
    + slot-0 weights (all fixed addresses, mutated eagerly between replays). No accumulate here — the caller
    sums cap['partial'] into out eagerly, so the graph stays a stateless function (clean warmup/capture)."""
    _refresh_nvfp4_scalars(method, layer, logical_to_slot=None)  # s2l fixed-[K] gather (capturable)
    remap = cap["l2g"][cap["topk_ids"]]
    keep = (remap >= 0).to(cap["topk_weights"].dtype)
    mtw = cap["topk_weights"] * keep
    sids = remap.clamp(min=0)
    md = cap["dispatch"]._replace(
        hidden_states=cap["hidden"],
        topk_output=cap["topk_out"]._replace(topk_ids=sids, topk_weights=mtw),
    )
    o = method.base_method.apply(layer, md)
    cap["partial"].copy_(o.hidden_states if hasattr(o, "hidden_states") else o)


def _wave_apply_perlayer_captured(method, layer, dispatch_output, topk_ids, distinct):
    """Per-layer MoE mini-graph capture path (K=1, bs=1, nvfp4). Replays a captured per-wave compute graph
    for each of the ``distinct`` experts (paged serially into slot-0 EAGERLY between replays), summing the
    per-wave partials. Returns the MoE output, or None to signal "fall back to eager" (unsupported shape,
    capture failure, or a correctness-check mismatch — never silently wrong)."""
    import logging

    pager = method._pager
    K, E, dev = pager.K, pager.E, pager.device
    store = pager.store
    hidden = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    topk_weights = topk_output.topk_weights
    T = hidden.shape[0]
    if T != 1 or K != 1 or getattr(method, "_pl_disabled", False):
        return None  # this path targets bs=1/K=1; _pl_disabled latches after a mismatch

    groups = [distinct[w : w + K] for w in range(0, len(distinct), K)]
    # The captured s2l scalar refresh reads pager._slot_expert_d (slot->logical), which the ON-DEVICE init
    # allocates but the EAGER InPlace store leaves None. Allocate it here (K slots) so the in-graph fixed-[K]
    # gather branch of _refresh_nvfp4_scalars is available. Safe for eager: that path passes logical_to_slot
    # explicitly, so it never reads _slot_expert_d regardless.
    if getattr(pager, "_slot_expert_d", None) is None:
        pager._slot_expert_d = torch.full((K,), -1, dtype=torch.int32, device=dev)
    cap = getattr(method, "_pl_cap", None)
    if cap is None or cap["T"] != T or cap["tk"] != topk_ids.shape[-1]:
        cap = {
            "T": T,
            "tk": topk_ids.shape[-1],
            "hidden": torch.empty_like(hidden),
            "topk_ids": torch.empty_like(topk_ids),
            "topk_weights": torch.empty_like(topk_weights),
            "l2g": torch.full((E,), -1, dtype=torch.int32, device=dev),
            "partial": torch.empty_like(hidden),
            "dispatch": dispatch_output,
            "topk_out": topk_output,
            "graph": None,
        }
        method._pl_cap = cap

    # Fill the fixed input buffers for this token (eager).
    cap["hidden"].copy_(hidden)
    cap["topk_ids"].copy_(topk_ids)
    cap["topk_weights"].copy_(topk_weights)
    l2g = cap["l2g"]

    # Whole-step cold read-ahead up front (same QD prefetch the eager path uses).
    if hasattr(store, "prefetch_cold"):
        store.prefetch_cold(distinct)
        store._step_prefetched = True

    out = None
    for w, group in enumerate(groups):
        src = torch.tensor(group, dtype=torch.int64, device=dev)
        dst = torch.arange(0, len(group), dtype=torch.int64, device=dev)  # slots 0..K-1
        pager.page_in(
            src, dst, src_host=group, dst_host=list(range(len(group))), async_h2d=True
        )
        l2g.fill_(-1)
        l2g[src] = dst.to(torch.int32)
        pager._slot_expert_d[dst] = src.to(torch.int32)  # for the in-graph s2l scalar refresh
        if cap["graph"] is None:
            # First wave of the first decode: capture (slot-0 is now populated), correctness-gate vs eager,
            # then replay to actually produce this wave's partial.
            eager_ref = _gemm_hidden(
                method, layer, dispatch_output, l2g[topk_ids], clone_hidden=True, logical_to_slot=l2g
            )
            try:
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        _pl_capture_body(method, layer, cap)
                torch.cuda.current_stream().wait_stream(s)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    _pl_capture_body(method, layer, cap)
                g.replay()
                torch.cuda.synchronize()
            except Exception as e:  # noqa: BLE001
                logging.getLogger(__name__).error(
                    "[perlayer] capture failed on layer %s: %r -> disabling per-layer capture (eager fallback)",
                    getattr(layer, "layer_id", "?"),
                    e,
                )
                method._pl_disabled = True
                return None
            diff = (cap["partial"].float() - eager_ref.float()).abs().max().item()
            denom = eager_ref.float().abs().max().item() + 1e-9
            if diff / denom > 1e-2:
                logging.getLogger(__name__).error(
                    "[perlayer] correctness MISMATCH layer %s (rel=%.3e) -> disabling per-layer capture",
                    getattr(layer, "layer_id", "?"),
                    diff / denom,
                )
                method._pl_disabled = True
                return None
            cap["graph"] = g
            logging.getLogger(__name__).info(
                "[perlayer] layer %s captured + correctness OK (rel=%.3e); replaying per wave",
                getattr(layer, "layer_id", "?"),
                diff / denom,
            )
        else:
            cap["graph"].replay()
        out = cap["partial"].clone() if out is None else out + cap["partial"]
    store._step_prefetched = False
    return out


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
    hidden = _scratch_prefill_apply(
        method, layer, dispatch_output, topk_ids, distinct=distinct
    )
    if hidden is not None:
        return hidden
    if (
        _PERLAYER_CAPTURE
        and getattr(method, "_nvfp4_full_e", None) is not None
        and getattr(method._pager, "K", 0) == 1
        and not torch.cuda.is_current_stream_capturing()
    ):
        r = _wave_apply_perlayer_captured(method, layer, dispatch_output, topk_ids, distinct)
        if r is not None:
            return r  # else fall through to the eager wave path (unsupported shape / disabled)
    pager = method._pager
    K, E, dev = pager.K, pager.E, pager.device
    store = pager.store
    half = K // 2
    # Bank (double-buffer) the host wave path only where the overlap pays: the DISK cold tier, whose
    # CPU-side gather (page faults) hides under the previous wave's GEMM. On RAM-windowed stores the
    # waves are transfer-bound with little to hide, and halving the wave size nearly doubles the
    # per-wave fixed costs (masked-GEMM pass + transfer launches) — measured net-negative (fp8-30B:
    # 934 -> 603 tok/s prefill), so those keep serial full-K waves.
    import os

    banked = (
        half > 0
        and bool(getattr(store, "_cold_mm", {}))
        and os.environ.get("SGLANG_PAGED_EXPERTS_BANKED", "1") != "0"
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
    _cap = _diff_wave_active(layer, dispatch_output)  # Hole-A per-wave capture (decode only)
    _capw = [] if _cap else None
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
                pager.page_in(src, dst, stage_bank=b, async_h2d=True, src_host=group)
                ev_h2d[b].record(ts)
            cs.wait_event(ev_h2d[b])
        else:
            src = torch.tensor(group, dtype=torch.int64, device=dev)
            dst = torch.arange(base, base + len(group), dtype=torch.int64, device=dev)
            # async_h2d=True: the H2D and the GEMM below are on the SAME (current) stream, so stream order
            # already guarantees the copy lands before the GEMM reads the slot — the per-wave
            # stream.synchronize() in page_in is redundant here and just stalls the CPU (~1 sync/wave x
            # waves x layers). dst_host avoids the dst_slots.to("cpu") sync too.
            # wl_pagein bracket: sizes the UN-capturable per-wave page-in orchestration (disk+H2D+plan
            # Python) vs the capturable compute below — gates the per-layer-capture build (perlayer doc).
            _tp = _perf_ns() if _sT is not None else 0
            pager.page_in(
                src, dst, src_host=group, dst_host=list(range(base, base + len(group))), async_h2d=True
            )
            if _sT is not None:
                _sT("wl_pagein", _perf_ns() - _tp)
        l2g.fill_(-1)
        l2g[src] = dst.to(torch.int32)
        _tc = _perf_ns() if _sT is not None else 0
        partial = _gemm_hidden(
            method,
            layer,
            dispatch_output,
            l2g[topk_ids],
            clone_hidden=True,
            logical_to_slot=l2g,  # nvfp4: scatter scalars by THIS wave's map, not the stale pager map
        )
        if _sT is not None:
            _sT("wl_compute", _perf_ns() - _tc)  # CAPTURABLE ceiling (refresh+remap+clone+gemm)
        if banked:
            ev_gemm[b].record(cs)
        out = partial if out is None else out + partial
        if _cap:
            _capw.append((int(group[0]), partial))  # (expert_id, weighted cutlass output w_e*f_e(h))
    if _cap:
        _diff_record_wave(layer, dispatch_output, _capw)
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
    hidden = _scratch_prefill_apply(method, layer, dispatch_output, topk_ids)
    if hidden is not None:
        return hidden
    import os

    pager = method._pager
    E, K = pager.E, pager.K
    half = K // 2
    banked = (
        half > 0
        and os.environ.get("SGLANG_PAGED_EXPERTS_BANKED", "1") != "0"
        and not torch.cuda.is_current_stream_capturing()
    )
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


def _ondevice_bounded_wave_apply(method, layer, dispatch_output, topk_ids):
    """Captured WINDOWED wave path (distinct > K on a windowed store; breakable backend only). Serve in
    ``ceil(E/K)`` static waves, each planned on-device (window hits gathered in-graph from ``host_hot``,
    cold deferred) with its cold experts staged at an in-layer eager break before the wave's GEMM — so a
    ``> K`` windowed decode batch stays CAPTURED instead of falling to the sync-heavy eager host wave
    (``_wave_apply``). Serial full-K (no banking) under capture. Sums the per-wave partials (lossless);
    resets the keep-warm residency maps at the end so the following keep-warm decode step re-pages clean.
    Windowed analog of ``_ondevice_wave_apply``."""
    from sglang.srt.layers.moe.paged_experts.placement import _bcg_cold_wave_break

    pager = method._pager
    E, K = pager.E, pager.K
    topk_weights = dispatch_output.topk_output.topk_weights
    # COMPACTED waves: the distinct active experts number at most num_tokens*top_k (capped at E), and the
    # decide kernel packs them by appearance order, so we run only ceil(min(num_tokens*top_k, E)/K) waves —
    # not ceil(E/K). num_tokens/top_k are static under capture, so the wave count is graph-stable.
    max_distinct = min(topk_ids.shape[0] * topk_ids.shape[-1], E)
    nwaves = (max_distinct + K - 1) // K
    out = None
    for w in range(nwaves):
        # segment 1: decide (compact + window-split) + window-hit gather; cold experts deferred to the break
        pager.decide_and_page_bounded_wave_ondevice(topk_ids, w)
        # eager break: stage THIS wave's cold experts into their assigned slots, then the GEMM segment runs
        # with them resident. Remap AFTER the break reads the live map (hits from decide + cold staged).
        _bcg_cold_wave_break()(pager, dispatch_output.hidden_states)
        # Fused remap+mask (ONE launch, reading the live post-break map) — the same kernel the keep-warm
        # path uses; saves the gather + 2x where + 2x zeros_like chain PER WAVE. Python-chain fallback for
        # weight layouts the kernel doesn't handle.
        fused = pager.remap_mask_ondevice(topk_ids, topk_weights)
        if fused is not None:
            partial = _gemm_hidden_fused(
                method, layer, dispatch_output, fused[0], fused[1], clone_hidden=True
            )
        else:
            remap = mask_and_remap_expert_ids(topk_ids, pager.logical_to_gpu_index_cuda)
            partial = _gemm_hidden(method, layer, dispatch_output, remap, clone_hidden=True)
        out = partial if out is None else out + partial
    pager.reset_residency_ondevice()
    return out


_DIFF_CAP = os.environ.get("SGLANG_DIFF_CAPTURE", "0") == "1"
_DIFF_LAYERS = {
    int(x) for x in os.environ.get("SGLANG_DIFF_LAYERS", "10,40,70").split(",") if x.strip()
}
_DIFF_OUT = os.environ.get("SGLANG_DIFF_OUT", "/root/.cache/huggingface/_diff")  # HF mount -> persists to host
_DIFF_NTOK = int(os.environ.get("SGLANG_DIFF_NTOK", "40"))
_diff_done: dict = {}
_diff_tokens: dict = {}


def _diff_record_wave(layer, dispatch_output, partials) -> None:
    """Hole-A PER-WAVE capture (decode; the wave loop runs per decode token, one expert per wave at K=1).
    Accumulates, per target layer, up to _DIFF_NTOK decode tokens of {h, native top-8 ids+weights, per-expert
    cutlass WEIGHTED partial (w_e·f_e(h)) per wave}. Offline divides by w_e -> cutlass f_e(h), compares to a
    bf16-dequant reference PER RANK (rank-1 = positive control, must match; rank-5-8 = the Hole-A test)."""
    import logging

    lid = getattr(layer, "layer_id", None)
    lst = _diff_tokens.setdefault(lid, [])
    try:
        to = dispatch_output.topk_output
        lst.append({
            "h": dispatch_output.hidden_states.detach().float().cpu(),
            "topk_ids": to.topk_ids.detach().cpu(),
            "topk_weights": to.topk_weights.detach().float().cpu(),
            "partials": [(int(e), p.detach().float().cpu()) for e, p in partials],
        })
        if len(lst) >= _DIFF_NTOK:
            import os as _os
            _os.makedirs(_DIFF_OUT, exist_ok=True)
            torch.save({"layer_id": int(lid), "tokens": lst}, _os.path.join(_DIFF_OUT, f"wave_{int(lid)}.pt"))
            _diff_done[lid] = True
            logging.getLogger(__name__).info(
                "[diff-capture] layer %s: %d decode tokens x %d waves -> %s [%d/%d]",
                lid, len(lst), len(partials), _DIFF_OUT, len(_diff_done), len(_DIFF_LAYERS))
    except Exception as e:  # noqa: BLE001
        logging.getLogger(__name__).error("[diff-capture] layer %s failed: %r", lid, e)


def _diff_wave_active(layer, dispatch_output) -> bool:
    lid = getattr(layer, "layer_id", None)
    return (
        _DIFF_CAP and lid in _DIFF_LAYERS and lid not in _diff_done
        and dispatch_output.hidden_states.shape[0] == 1  # decode (per-wave loop runs here, not prefill)
        and not torch.cuda.is_current_stream_capturing()
    )


def paged_apply(method, layer, dispatch_output):
    """Dispatch the step to the method's decode placement (eager host vs captured on-device).

    The placement (``method._placement``) owns the decide + page-in flow; both end in ``_gemm_hidden``
    over the K-slot pool. See ``placement.py``.
    """
    _maybe_truncate_topk(dispatch_output)
    if _sT is not None:
        import time as _time

        _m0 = _time.perf_counter_ns()
        r = method._placement.apply(method, layer, dispatch_output)
        _sT("moe_call", _time.perf_counter_ns() - _m0)
        return r
    return method._placement.apply(method, layer, dispatch_output)
