"""Decode-placement strategy: *where* the per-step residency decision + page-in run.

Both placements end in the same K-slot fused-MoE GEMM (``forward._gemm_hidden``); they differ only in
where the per-step decide + page-in happen — and therefore whether sglang's decode CUDA graph can capture
the step:

* ``EagerPlacement`` — a host-side keep-warm/LRU decision + ``transfer_kv`` page-in. Data-dependent, so it
  runs outside any graph (requires ``--disable-cuda-graph``). Kernel-free.
* ``CapturedPlacement`` — the decide + UVA gather run on the GPU with no host sync, so the decode step is
  captured. The keep-warm vs static-wave regime is chosen from shapes alone (``num_tokens*top_k <= K``),
  which is static under capture; it needs the pager's on-device state (``setup_ondevice``), flagged by
  ``needs_ondevice_store``.

Selected once per layer (from ``--disable-cuda-graph``; see ``method.make_for_layer``). A third placement
is a new subclass — no ``use_ondevice`` bool threaded through method / pager / forward.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod

import torch

# Hoisted from EagerPlacement.apply (was re-imported per layer per step — pure _handle_fromlist overhead
# on the host-bound eager path). forward.py imports no paged_experts sibling, so this is circular-safe.
from sglang.srt.layers.moe.paged_experts.forward import (
    _SHARED_OVERLAP_ON,
    _current_overlap,
    _gemm_hidden,
    _overlap_stream,
    _wave_apply,
    mask_and_remap_expert_ids,
)

from sglang.srt.server_args import get_global_server_args

logger = logging.getLogger(__name__)


class Placement(ABC):
    """Strategy for where a paged-experts decode step decides residency + pages experts in."""

    #: whether the pager must allocate on-device residency state (``setup_ondevice``) for this placement
    needs_ondevice_store: bool = False

    @abstractmethod
    def apply(self, method, layer, dispatch_output):
        """Decide + page-in + run the K-slot GEMM for one step; return a ``StandardCombineInput``."""


# --- tau-dial: weight-gated cold-miss skipping (SGLANG_PE_TAU, default 0 = off) -----------------
# On a disk-backed cold tier the step time is ~proportional to fetched bytes, so skipping the routed
# pairs that are BOTH cold (a fetch away) AND low-weight (weight < TAU * the token's max weight) trades
# a bounded output approximation for a linear byte saving. APPROXIMATE-MoE: quality must be measured
# alongside any throughput number (two-axis price sheet). Mechanism: the skipped pair is redirected to
# the token's own top-1 expert (never skipped, by construction) with weight zeroed — no fetch is
# planned, no new kernel path runs, the redundant row contributes exactly nothing through the weighted
# reduce. Wasted compute on one resident row; saved disk bytes. Skips are LOGGED per expert
# (SGLANG_PE_TAU_LOG_EVERY layer-calls) so the skip distribution can be cross-referenced against
# frequency tails from other modalities. Runs before decide AND before the keep-warm/wave branch, so
# it also shrinks the distinct set. Side effect: LFU freq counts see the redirected top-1 id twice.
_TAU = float(os.environ.get("SGLANG_PE_TAU", "0"))
_TAU_LOG_EVERY = int(os.environ.get("SGLANG_PE_TAU_LOG_EVERY", "2000"))
_tau_calls = 0
_tau_pairs = 0
_tau_skipped = 0
_tau_mass_skipped = 0.0
_tau_hist = None


def _tau_skip(pager, topk_output):
    global _tau_calls, _tau_pairs, _tau_skipped, _tau_mass_skipped, _tau_hist
    ids, w = topk_output.topk_ids, topk_output.topk_weights
    if ids.numel() == 0 or w is None:
        return
    m = pager.logical_to_gpu_index_cuda
    valid = (ids >= 0) & (ids < m.numel())
    safe = torch.where(valid, ids, torch.zeros_like(ids))
    miss = (m[safe.long()] < 0) & valid
    wmax = w.max(dim=-1, keepdim=True).values
    skip = miss & (w < _TAU * wmax)
    _tau_calls += 1
    _tau_pairs += int(ids.numel())
    if bool(skip.any()):
        if _tau_hist is None:
            _tau_hist = torch.zeros(m.numel(), dtype=torch.int64, device=ids.device)
        _tau_hist.index_add_(
            0, ids[skip].long(), torch.ones(int(skip.sum()), dtype=torch.int64, device=ids.device)
        )
        _tau_mass_skipped += float(w[skip].sum())
        _tau_skipped += int(skip.sum())
        top1 = w.argmax(dim=-1, keepdim=True)
        top1_ids = ids.gather(-1, top1)
        ids.copy_(torch.where(skip, top1_ids.expand_as(ids), ids))
        w.masked_fill_(skip, 0.0)
    if _TAU_LOG_EVERY and _tau_calls % _TAU_LOG_EVERY == 0:
        top = torch.topk(_tau_hist, 10)
        logger.info(
            "[tau-dial] tau=%.2f calls=%d skipped=%d/%d pairs (%.1f%%) mass_skipped=%.1f | top skipped experts: %s",
            _TAU, _tau_calls, _tau_skipped, _tau_pairs, 100 * _tau_skipped / max(1, _tau_pairs),
            _tau_mass_skipped,
            " ".join(f"e{int(i)}:{int(c)}" for i, c in zip(top.indices.tolist(), top.values.tolist()) if c > 0),
        )


class EagerPlacement(Placement):
    """Host decide (keep-warm + LRU/LFU) + ``transfer_kv`` page-in. Kernel-free; requires
    ``--disable-cuda-graph`` (the host decision is data-dependent, so the step is not capturable).
    """

    needs_ondevice_store = False

    def apply(self, method, layer, dispatch_output):
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        pager = method._pager
        if _TAU > 0.0:
            _tau_skip(pager, dispatch_output.topk_output)
        topk_ids = dispatch_output.topk_output.topk_ids
        distinct = pager.distinct_active(topk_ids)
        if len(distinct) <= pager.K:  # keep-warm: page only the misses
            src, dst = pager.decide_keep_warm(topk_ids, distinct=distinct)
            # #11: the host plan decide_keep_warm just built — lets the windowed cold tier skip a
            # per-layer dst_slots D2H (a no-op for non-windowed stores, which ignore it).
            sh, dh = pager._kw_src_host, pager._kw_dst_host
            ov = _current_overlap()
            if ov is not None and _SHARED_OVERLAP_ON and int(src.numel()) > 0:
                # #4: run the routed page-in on a transfer stream and the caller's router-independent work
                # (e.g. the shared expert) on the compute stream, so they overlap; sync before the GEMM.
                cs = torch.cuda.current_stream()
                ts = _overlap_stream(pager.device)
                ts.wait_stream(cs)  # transfer must see src/dst (created on cs)
                with torch.cuda.stream(ts):
                    pager.page_in(src, dst, src_host=sh, dst_host=dh)
                ov.result = ov.fn()  # compute stream — overlaps the H2D
                cs.wait_stream(ts)  # routed GEMM must see the paged-in experts
            else:
                pager.page_in(src, dst, src_host=sh, dst_host=dh)
                if ov is not None:  # no misses to hide behind: run it inline (still correct)
                    ov.result = ov.fn()
            remap = mask_and_remap_expert_ids(topk_ids, pager.logical_to_gpu_index_cuda)
            hidden = _gemm_hidden(
                method, layer, dispatch_output, remap, clone_hidden=False
            )
        else:  # distinct > K: serve in waves, sum the partials (lossless)
            hidden = _wave_apply(method, layer, dispatch_output, topk_ids, distinct)
        return StandardCombineInput(hidden_states=hidden)


def _keep_warm_gemm(method, layer, dispatch_output, pager):
    """The keep-warm GEMM tail shared by the captured placements: fused remap+mask (ONE launch replacing
    the gather + 2x where + 2x zeros_like chain) with the python chain as fallback for weight layouts the
    kernel doesn't handle."""
    from sglang.srt.layers.moe.paged_experts.forward import (
        _gemm_hidden,
        _gemm_hidden_fused,
        mask_and_remap_expert_ids,
    )

    topk_output = dispatch_output.topk_output
    fused = pager.remap_mask_ondevice(topk_output.topk_ids, topk_output.topk_weights)
    if fused is not None:
        return _gemm_hidden_fused(
            method, layer, dispatch_output, fused[0], fused[1], clone_hidden=False
        )
    remap = mask_and_remap_expert_ids(
        topk_output.topk_ids, pager.logical_to_gpu_index_cuda
    )
    return _gemm_hidden(method, layer, dispatch_output, remap, clone_hidden=False)


_WAVE_CAPTURE_WARNED = False


def _warn_wave_capture_once(pager, topk_ids):
    """Full-pin path: a capture batch with ``bs*top_k > K`` is servable (on-device waves) but pays
    ``ceil(E/K)`` GEMMs per layer per step — a silent multi-x cliff. Say so once, with the fix.
    """
    global _WAVE_CAPTURE_WARNED
    import torch

    if _WAVE_CAPTURE_WARNED or not torch.cuda.is_current_stream_capturing():
        return
    _WAVE_CAPTURE_WARNED = True
    bs, top_k = topk_ids.shape[0], topk_ids.shape[-1]
    nwaves = (pager.E + pager.K - 1) // pager.K
    logger.warning(
        "[paged-experts] capture batch bs=%d exceeds the keep-warm bound (bs*top_k=%d > K=%d): decode at "
        "this batch size serves every MoE layer in %d waves (~%dx the expert GEMM cost). Cap "
        "--cuda-graph-max-bs at %d (K//top_k) to keep captured decode in the keep-warm regime.",
        bs,
        bs * top_k,
        pager.K,
        nwaves,
        nwaves,
        max(1, pager.K // top_k),
    )


class CapturedPlacement(Placement):
    """On-device decide + UVA gather, run inside sglang's captured decode graph (no host sync). The
    keep-warm vs static-wave regime is chosen from shapes alone (``num_tokens*top_k <= K``).
    """

    needs_ondevice_store = True

    def apply(self, method, layer, dispatch_output):
        import torch

        from sglang.srt.layers.moe.paged_experts.forward import _ondevice_wave_apply
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        pager = method._pager
        topk_ids = dispatch_output.topk_output.topk_ids
        keep_warm = topk_ids.shape[0] * topk_ids.shape[-1] <= pager.K
        if keep_warm:
            # #4: pass topk_weights so the decide launch also emits the forward remap (fused), and
            # _keep_warm_gemm's remap_mask_ondevice returns it without a second launch.
            pager.decide_and_page_ondevice(
                topk_ids, dispatch_output.topk_output.topk_weights
            )
            hidden = _keep_warm_gemm(method, layer, dispatch_output, pager)
        else:  # distinct can exceed K (prefill / big batch): static waves, summed
            _warn_wave_capture_once(pager, topk_ids)
            hidden = _ondevice_wave_apply(method, layer, dispatch_output, topk_ids)
            if not torch.cuda.is_current_stream_capturing() and getattr(
                get_global_server_args(), "paged_experts_prompt_warmup", False
            ):
                # Prompt-aware warm-up: seed the pool from THIS chunk's routing counts (the prompt tail —
                # the most recency-relevant prior for the coming generation) instead of the last wave's
                # arbitrary leftovers. Delta-aware: pages only the missing hot experts (one small gather).
                flat = topk_ids.reshape(-1).long()
                counts = torch.bincount(flat[flat >= 0], minlength=pager.E).float()
                pager.seed_from_recording(counts)
        return StandardCombineInput(hidden_states=hidden)


def _reject_wave_under_capture(pager, topk_ids):
    """The windowed placements' distinct>K fallback is a HOST wave (syncs) — fine for prefill, fatal
    inside a decode graph capture. Fires when a capture batch needs more distinct experts than the K-slot
    pool holds; fail with the fix instead of letting the raise abort capture mid-region (which resurfaces
    as a cryptic cudaErrorStreamCaptureUnjoined).

    ``topk_ids.shape[0]`` is the captured TOKEN count: for a plain decode graph that equals the batch
    size, but for a speculative TARGET_VERIFY graph it is ``batch_size * num_draft_tokens`` — so capping
    ``--cuda-graph-max-bs`` alone does NOT shrink it under spec; the tree width must shrink too.
    """
    import torch

    if torch.cuda.is_current_stream_capturing():
        num_tokens, top_k = topk_ids.shape[0], topk_ids.shape[-1]
        raise RuntimeError(
            f"Paged Experts (windowed): the captured batch needs up to num_tokens*top_k="
            f"{num_tokens * top_k} distinct-expert slots but the pool has only K={pager.K}, and the "
            f"windowed wave fallback cannot be captured. Reduce the captured token count so "
            f"num_tokens*top_k <= K: cap --cuda-graph-max-bs (and, for speculative decoding, also "
            f"--speculative-num-draft-tokens — the verify graph captures batch_size*num_draft_tokens "
            f"tokens, so with top_k={top_k} and K={pager.K} keep num_draft_tokens <= {max(1, pager.K // top_k)} "
            f"at batch size 1), or run with --disable-cuda-graph."
        )


class CapturedWindowedPlacement(Placement):
    """Captured decode for the pinned-WINDOW store (the >pin-ceiling fallback). Keep-warm decode runs the
    on-device ``decide_bounded`` + windowed gather: window hits gather in-graph from ``host_hot``, while cold
    (window-missing) experts are deferred and staged out-of-graph by the replay-twice post-replay hook
    (registered when the pager set up its window state). The rare ``distinct > K`` step (prefill / big batch)
    falls back to the eager host wave path — the window store pages hot via ``transfer_kv`` and cold via an
    indexed copy — since prefill is one-shot and not on the captured decode path."""

    needs_ondevice_store = True

    def apply(self, method, layer, dispatch_output):
        from sglang.srt.layers.moe.paged_experts.forward import _wave_apply
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        pager = method._pager
        topk_ids = dispatch_output.topk_output.topk_ids
        keep_warm = topk_ids.shape[0] * topk_ids.shape[-1] <= pager.K
        if keep_warm:
            pager.decide_and_page_bounded_ondevice(topk_ids)
            hidden = _keep_warm_gemm(method, layer, dispatch_output, pager)
        else:  # prefill / big batch: eager host wave (window store pages hot+cold); not captured
            _reject_wave_under_capture(pager, topk_ids)
            distinct = pager.distinct_active(topk_ids)
            hidden = _wave_apply(method, layer, dispatch_output, topk_ids, distinct)
        return StandardCombineInput(hidden_states=hidden)


_bcg_break = None


def _bcg_cold_break():
    """The eager break that stages a windowed layer's cold experts (BCG break-and-page-in). Wrapped with
    ``eager_on_graph`` so, under breakable-decode capture, calling it ends the decide+gather segment, runs
    the staging eager (host_cold -> slots), and starts the GEMM segment — eliminating the replay-twice
    second full-graph replay. Built lazily (eager_on_graph hard-raises off CUDA)."""
    global _bcg_break
    if _bcg_break is None:
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
            eager_on_graph,
        )

        def _stage(pager, hidden_states):
            pager.stage_cold_at_break()  # side effect: refill cold into slots + update residency maps
            # Return None on purpose: a pass-through tensor made the backend's output-copy launch a
            # redundant D2D self-copy every break (48/token); None falls through with no copy.

        _bcg_break = eager_on_graph(True)(_stage)
    return _bcg_break


class CapturedWindowedBCGPlacement(Placement):
    """Captured windowed decode under the *breakable* backend (BCG break-and-page-in). Same on-device
    decide_bounded + windowed gather as the replay-twice variant, but the deferred cold experts are staged
    at an in-layer eager break (between decide and the expert GEMM) — so a cold miss is paged inline in the
    same forward pass, with NO second full-graph replay. Requires --cuda-graph-backend-decode breakable.
    """

    needs_ondevice_store = True

    def apply(self, method, layer, dispatch_output):
        from sglang.srt.layers.moe.paged_experts.forward import _wave_apply
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        pager = method._pager
        topk_ids = dispatch_output.topk_output.topk_ids
        keep_warm = topk_ids.shape[0] * topk_ids.shape[-1] <= pager.K
        if keep_warm:
            pager.decide_and_page_bounded_ondevice(
                topk_ids
            )  # segment 1: decide + window-hit gather
            # eager break: stage this step's cold experts into their slots, then the GEMM segment runs with
            # them resident (no replay-twice). Called for its side effect + the segment boundary; it passes
            # hidden_states through unchanged (the GEMM below reads the same fixed-address buffer), so the
            # return is ignored (dispatch_output.hidden_states is a read-only property).
            _bcg_cold_break()(pager, dispatch_output.hidden_states)
            # Remap AFTER the break: the fused remap_mask reads the LIVE map, so it sees the experts the
            # break just staged (segment 2 of the broken graph).
            hidden = _keep_warm_gemm(method, layer, dispatch_output, pager)
        else:  # prefill / big batch: eager host wave (not on the captured decode path)
            _reject_wave_under_capture(pager, topk_ids)
            distinct = pager.distinct_active(topk_ids)
            hidden = _wave_apply(method, layer, dispatch_output, topk_ids, distinct)
        return StandardCombineInput(hidden_states=hidden)


def make_placement(
    use_ondevice: bool, windowed: bool = False, breakable_decode: bool = False
) -> Placement:
    """Captured when CUDA graphs are on (and a pinned store is available), else eager host. A windowed
    (>pin-ceiling) store uses the captured replay-twice variant when on-device — or the BCG break-and-page-in
    variant when decode runs under the breakable backend (no second full-graph replay).
    """
    if not use_ondevice:
        return EagerPlacement()
    if windowed:
        return (
            CapturedWindowedBCGPlacement()
            if breakable_decode
            else CapturedWindowedPlacement()
        )
    return CapturedPlacement()
