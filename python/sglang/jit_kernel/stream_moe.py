from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_stream_moe_module() -> Module:
    """Compile and cache the compute-through streaming expert-FFN kernel for Paged Experts."""
    return load_jit(
        "stream_moe",
        cuda_files=["moe/stream_moe.cuh"],
        cuda_wrappers=[
            ("stream_expert_ffn", "stream_expert_ffn"),
        ],
    )


def stream_expert_ffn(
    w13_qw: int,
    w13_s: int,
    w2_qw: int,
    w2_s: int,
    X: torch.Tensor,
    O: torch.Tensor,
    gu: torch.Tensor,
    h: torch.Tensor,
    inter: int,
    hidden: int,
    group: int,
) -> None:
    """Stream one expert's int4 gated FFN over ``T`` tokens, weights read from the transposed-plain store.

    Computes ``O = (SiLU(x @ w13_gate^T) * (x @ w13_up^T)) @ w2^T`` — the single expert's pre-scatter
    output — streaming ``w13``/``w2`` from their pinned-store UVA device pointers (``w13_qw`` etc., from
    :func:`paged_experts_host_devptr`, or a device ``data_ptr()`` for the reference test). The caller
    applies the routing weight and accumulates into the MoE output (explicit scatter — this kernel does
    NOT fold ``topk_weights`` the way marlin's ``mul_topk_weights`` gemm2 does).

    ``w13_qw``/``w2_qw`` point at ``[OUT, IN/8]`` int32 (packed low-nibble-first along IN, sym => zero=8);
    ``w13_s``/``w2_s`` at ``[OUT, IN/group]`` fp16 scales. ``X`` is ``[T, H]`` fp16 CUDA; ``O`` ``[T, H]``
    fp32 CUDA; ``gu`` ``[T, 2*inter]`` fp32 and ``h`` ``[T, inter]`` fp16 preallocated scratch (capture-safe).
    """
    module = _jit_stream_moe_module()
    module.stream_expert_ffn(
        int(w13_qw),
        int(w13_s),
        int(w2_qw),
        int(w2_s),
        X,
        O,
        gu,
        h,
        int(inter),
        int(hidden),
        int(group),
    )
