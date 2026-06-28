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


def make_placement(use_ondevice: bool) -> Placement:
    """Captured when CUDA graphs are on (and a pinned store is available), else eager host."""
    return CapturedPlacement() if use_ondevice else EagerPlacement()
