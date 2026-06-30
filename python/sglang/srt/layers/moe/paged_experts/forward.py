"""Paged-experts forward (eager).

Per decode step, the active experts are paged into the K-slot pool and the real fused-MoE GEMM runs over
it. Two regimes:

* ``distinct active experts <= K``: keep-warm. Page only the misses (resident experts are reused across
  steps), remap, one GEMM.
* ``distinct active experts > K`` (e.g. prefill, or batched decode): the pool can't hold them at once, so
  serve them in ``ceil(distinct / K)`` **waves** — each wave pages <=K experts, masks the routing to that
  wave, runs the GEMM, and the per-wave partials are **summed**. Each active expert is in exactly one wave
  and out-of-wave experts are masked to weight 0, so the sum equals the full MoE output (lossless).

Routing stays E-wide; only the table is K. (The captured fast path will replace the host-side decision
with static-wave masking; same GEMM.)
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


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


def _wave_apply(method, layer, dispatch_output, topk_ids: torch.Tensor, distinct):
    """Serve > K distinct experts in ceil(len(distinct)/K) waves; sum the per-wave partials (lossless)."""
    pager = method._pager
    K, E, dev = pager.K, pager.E, pager.device
    l2g = torch.full((E,), -1, dtype=torch.int32, device=dev)
    out = None
    group = []
    for w in range(0, len(distinct), K):
        group = distinct[w : w + K]
        src = torch.tensor(group, dtype=torch.int64, device=dev)
        dst = torch.arange(len(group), dtype=torch.int64, device=dev)
        pager.page_in(src, dst)
        l2g.fill_(-1)
        l2g[src] = dst.to(torch.int32)
        partial = _gemm_hidden(
            method, layer, dispatch_output, l2g[topk_ids], clone_hidden=True
        )
        out = partial if out is None else out + partial
    pager.set_residency(group)  # leave the maps consistent for the next keep-warm step
    return out


def paged_apply(method, layer, dispatch_output):
    """Orchestrate the page-in + GEMM and wrap the result for the combiner."""
    from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

    pager = method._pager
    topk_ids = dispatch_output.topk_output.topk_ids
    distinct = pager.distinct_active(topk_ids)

    if len(distinct) <= pager.K:
        src, dst = pager.decide_keep_warm(topk_ids, distinct=distinct)
        pager.page_in(src, dst)
        remap = mask_and_remap_expert_ids(topk_ids, pager.logical_to_gpu_index_cuda)
        hidden = _gemm_hidden(method, layer, dispatch_output, remap, clone_hidden=False)
    else:
        hidden = _wave_apply(method, layer, dispatch_output, topk_ids, distinct)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "paged_apply L%s: topk=%s distinct=%d K=%d (%s)",
            getattr(layer, "layer_id", "?"),
            tuple(topk_ids.shape),
            len(distinct),
            pager.K,
            "wave" if len(distinct) > pager.K else "keep-warm",
        )
    return StandardCombineInput(hidden_states=hidden)
