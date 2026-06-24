from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_paged_experts_decide_module() -> Module:
    """Compile and cache the on-device Paged Experts residency-decision module."""
    return load_jit(
        "paged_experts_decide",
        cuda_files=["moe/paged_experts_decide.cuh"],
        cuda_wrappers=[("decide", "decide"), ("decide_wave", "decide_wave")],
    )


def paged_experts_decide(
    topk: torch.Tensor,
    step_ctr: torch.Tensor,
    slot_expert: torch.Tensor,
    expert_slot: torch.Tensor,
    slot_lastuse: torch.Tensor,
    freq: torch.Tensor,
    lfu: bool,
    src: torch.Tensor,
    dst: torch.Tensor,
    n_out: torch.Tensor,
    idx: torch.Tensor,
) -> None:
    """On-device keep-warm + LRU/LFU residency decision for Paged Experts (distinct active experts <= K).

    Computes the per-step paging plan entirely on the GPU — no host sync — so the decode step is
    CUDA-graph-capturable. Mutates the residency state (``step_ctr`` / ``slot_expert`` / ``expert_slot`` /
    ``slot_lastuse`` / ``freq``) in place and writes the page-in plan into the preallocated output buffers,
    which the existing ``transfer_kv_per_layer_mla`` gather then consumes (it reads the indices on-device).

    All tensors are ``int32`` and CUDA-resident. ``topk`` is ``[topk_n]`` (flattened active expert ids,
    negative = padding); ``step_ctr`` is ``[1]`` (a monotonic counter the kernel increments on-device, so a
    captured graph advances LRU recency every replay); ``slot_expert``/``slot_lastuse`` are ``[K]``;
    ``expert_slot``/``freq``/``idx`` are ``[E]``; ``src``/``dst`` are ``[>=K]`` (filled ``0..n``); ``n_out``
    is ``[1]`` (the page-in count). ``lfu`` selects LFU eviction (use-count, LRU tiebreak) over plain LRU.
    ``idx`` receives the updated logical->slot map (-1 == not resident) for the forward remap.
    """
    module = _jit_paged_experts_decide_module()
    module.decide(
        topk,
        step_ctr,
        slot_expert,
        expert_slot,
        slot_lastuse,
        freq,
        int(lfu),
        src,
        dst,
        n_out,
        idx,
    )


def paged_experts_decide_wave(
    topk: torch.Tensor,
    num_experts: int,
    num_slots: int,
    wave: int,
    src: torch.Tensor,
    dst: torch.Tensor,
    n_out: torch.Tensor,
    idx: torch.Tensor,
) -> None:
    """On-device static fixed-wave decision for Paged Experts (distinct active experts > K).

    Expert ``e`` has a static home — wave ``floor(e/K)``, slot ``e % K``. For ``wave`` this emits the
    page-in plan for the distinct in-wave experts present in ``topk`` and writes ``idx`` so out-of-wave
    experts map to -1 (masked to weight 0). The caller runs ``ceil(num_experts/num_slots)`` waves and sums
    the per-wave GEMM partials — lossless. No eviction, no state mutation, no host sync (capturable).

    All tensors are ``int32`` CUDA: ``topk`` ``[topk_n]``, ``src``/``dst`` ``[>=K]``, ``n_out`` ``[1]``,
    ``idx`` ``[num_experts]``.
    """
    module = _jit_paged_experts_decide_module()
    module.decide_wave(
        topk, int(num_experts), int(num_slots), int(wave), src, dst, n_out, idx
    )
