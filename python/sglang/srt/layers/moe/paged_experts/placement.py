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

from abc import ABC, abstractmethod


class Placement(ABC):
    """Strategy for where a paged-experts decode step decides residency + pages experts in."""

    #: whether the pager must allocate on-device residency state (``setup_ondevice``) for this placement
    needs_ondevice_store: bool = False

    @abstractmethod
    def apply(self, method, layer, dispatch_output):
        """Decide + page-in + run the K-slot GEMM for one step; return a ``StandardCombineInput``."""


class EagerPlacement(Placement):
    """Host decide (keep-warm + LRU/LFU) + ``transfer_kv`` page-in. Kernel-free; requires
    ``--disable-cuda-graph`` (the host decision is data-dependent, so the step is not capturable)."""

    needs_ondevice_store = False

    def apply(self, method, layer, dispatch_output):
        from sglang.srt.layers.moe.paged_experts.forward import (
            _gemm_hidden,
            _wave_apply,
            mask_and_remap_expert_ids,
        )
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        pager = method._pager
        topk_ids = dispatch_output.topk_output.topk_ids
        distinct = pager.distinct_active(topk_ids)
        if len(distinct) <= pager.K:  # keep-warm: page only the misses
            src, dst = pager.decide_keep_warm(topk_ids, distinct=distinct)
            pager.page_in(src, dst)
            remap = mask_and_remap_expert_ids(topk_ids, pager.logical_to_gpu_index_cuda)
            hidden = _gemm_hidden(
                method, layer, dispatch_output, remap, clone_hidden=False
            )
        else:  # distinct > K: serve in waves, sum the partials (lossless)
            hidden = _wave_apply(method, layer, dispatch_output, topk_ids, distinct)
        return StandardCombineInput(hidden_states=hidden)


class CapturedPlacement(Placement):
    """On-device decide + UVA gather, run inside sglang's captured decode graph (no host sync). The
    keep-warm vs static-wave regime is chosen from shapes alone (``num_tokens*top_k <= K``)."""

    needs_ondevice_store = True

    def apply(self, method, layer, dispatch_output):
        from sglang.srt.layers.moe.paged_experts.forward import (
            _gemm_hidden,
            _ondevice_wave_apply,
            mask_and_remap_expert_ids,
        )
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        pager = method._pager
        topk_ids = dispatch_output.topk_output.topk_ids
        keep_warm = topk_ids.shape[0] * topk_ids.shape[-1] <= pager.K
        if keep_warm:
            pager.decide_and_page_ondevice(topk_ids)
            remap = mask_and_remap_expert_ids(topk_ids, pager.logical_to_gpu_index_cuda)
            hidden = _gemm_hidden(
                method, layer, dispatch_output, remap, clone_hidden=False
            )
        else:  # distinct can exceed K (prefill / big batch): static waves, summed
            hidden = _ondevice_wave_apply(method, layer, dispatch_output, topk_ids)
        return StandardCombineInput(hidden_states=hidden)


class CapturedWindowedPlacement(Placement):
    """Captured decode for the pinned-WINDOW store (the >pin-ceiling fallback). Keep-warm decode runs the
    on-device ``decide_bounded`` + windowed gather: window hits gather in-graph from ``host_hot``, while cold
    (window-missing) experts are deferred and staged out-of-graph by the replay-twice post-replay hook
    (registered when the pager set up its window state). The rare ``distinct > K`` step (prefill / big batch)
    falls back to the eager host wave path — the window store pages hot via ``transfer_kv`` and cold via an
    indexed copy — since prefill is one-shot and not on the captured decode path."""

    needs_ondevice_store = True

    def apply(self, method, layer, dispatch_output):
        from sglang.srt.layers.moe.paged_experts.forward import (
            _gemm_hidden,
            _wave_apply,
            mask_and_remap_expert_ids,
        )
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        pager = method._pager
        topk_ids = dispatch_output.topk_output.topk_ids
        keep_warm = topk_ids.shape[0] * topk_ids.shape[-1] <= pager.K
        if keep_warm:
            pager.decide_and_page_bounded_ondevice(topk_ids)
            remap = mask_and_remap_expert_ids(topk_ids, pager.logical_to_gpu_index_cuda)
            hidden = _gemm_hidden(
                method, layer, dispatch_output, remap, clone_hidden=False
            )
        else:  # prefill / big batch: eager host wave (window store pages hot+cold); not captured
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
            return hidden_states  # pass-through so the break has a (copyable) output

        _bcg_break = eager_on_graph(True)(_stage)
    return _bcg_break


class CapturedWindowedBCGPlacement(Placement):
    """Captured windowed decode under the *breakable* backend (BCG break-and-page-in). Same on-device
    decide_bounded + windowed gather as the replay-twice variant, but the deferred cold experts are staged
    at an in-layer eager break (between decide and the expert GEMM) — so a cold miss is paged inline in the
    same forward pass, with NO second full-graph replay. Requires --cuda-graph-backend-decode breakable."""

    needs_ondevice_store = True

    def apply(self, method, layer, dispatch_output):
        from sglang.srt.layers.moe.paged_experts.forward import (
            _gemm_hidden,
            _wave_apply,
            mask_and_remap_expert_ids,
        )
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        pager = method._pager
        topk_ids = dispatch_output.topk_output.topk_ids
        keep_warm = topk_ids.shape[0] * topk_ids.shape[-1] <= pager.K
        if keep_warm:
            pager.decide_and_page_bounded_ondevice(topk_ids)  # segment 1: decide + window-hit gather
            # eager break: stage this step's cold experts into their slots, then the GEMM segment runs with
            # them resident (no replay-twice). Called for its side effect + the segment boundary; it passes
            # hidden_states through unchanged (the GEMM below reads the same fixed-address buffer), so the
            # return is ignored (dispatch_output.hidden_states is a read-only property).
            _bcg_cold_break()(pager, dispatch_output.hidden_states)
            remap = mask_and_remap_expert_ids(topk_ids, pager.logical_to_gpu_index_cuda)
            hidden = _gemm_hidden(
                method, layer, dispatch_output, remap, clone_hidden=False
            )
        else:  # prefill / big batch: eager host wave (not on the captured decode path)
            distinct = pager.distinct_active(topk_ids)
            hidden = _wave_apply(method, layer, dispatch_output, topk_ids, distinct)
        return StandardCombineInput(hidden_states=hidden)


def make_placement(
    use_ondevice: bool, windowed: bool = False, breakable_decode: bool = False
) -> Placement:
    """Captured when CUDA graphs are on (and a pinned store is available), else eager host. A windowed
    (>pin-ceiling) store uses the captured replay-twice variant when on-device — or the BCG break-and-page-in
    variant when decode runs under the breakable backend (no second full-graph replay)."""
    if not use_ondevice:
        return EagerPlacement()
    if windowed:
        return CapturedWindowedBCGPlacement() if breakable_decode else CapturedWindowedPlacement()
    return CapturedPlacement()
