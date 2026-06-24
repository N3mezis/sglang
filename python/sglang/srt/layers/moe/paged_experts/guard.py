"""Compatibility guard for Paged Experts.

Hard-fail at model init if the server is configured with a parallelism / placement mode the paging path
cannot honor yet, instead of silently paging the WRONG experts. Mirrors the style of sglang's own
``ServerArgs`` checks (assert/raise with a what / why / how-to-fix message) and fires before any weight
touches the GPU.

States (see the contribution plan, "TP/EP vs paging"):
  * not-supported-yet: ``tp_size`` / ``ep_size`` / ``pp_size`` / ``dp_size`` (single-GPU first cut; the
    rank-aware per-rank store is future work)
  * gate-now-subsume-later: ``enable_eplb`` (overlaps keep-warm; no-op at ``ep_size == 1`` anyway)
  * validate-before-allow: ``moe_a2a_backend`` (the dispatch/combine kernels must survive the K-slot remap)
  * hard: ``load_format == "dummy"`` (the host store reads REAL expert weights)
"""

from __future__ import annotations

from typing import Any


def check_paged_experts_compat(server_args: Any) -> None:
    """Raise ``RuntimeError`` if ``server_args`` is incompatible with Paged Experts.

    Call once, before wrapping any MoE layer. Paged Experts is single-GPU for now: any multi-device
    parallelism (tp/ep/pp/dp) is rejected.
    """
    tp = getattr(server_args, "tp_size", 1) or 1
    ep = getattr(server_args, "ep_size", 1) or 1
    pp = getattr(server_args, "pp_size", 1) or 1
    dp = getattr(server_args, "dp_size", 1) or 1
    a2a = getattr(server_args, "moe_a2a_backend", None)
    load_format = str(getattr(server_args, "load_format", "") or "")

    problems = []
    if tp > 1:
        problems.append(
            f"tensor parallelism (tp_size={tp}) is not supported yet: the host expert store is not "
            "rank-aware (single-GPU only for now). Use --tp-size 1."
        )
    if ep > 1:
        problems.append(
            f"expert parallelism (ep_size={ep}) is not supported yet: the store is built for all E "
            "experts, not this rank's E/ep_size local experts. Use --ep-size 1."
        )
    if pp > 1:
        problems.append(
            f"pipeline parallelism (pp_size={pp}) is not supported: the per-layer pool + pinned store "
            "assume all layers on one device. Use --pp-size 1."
        )
    if dp > 1:
        problems.append(
            f"data parallelism (dp_size={dp}) is untested: each replica needs its own pool + pinned "
            "store. Use --dp-size 1."
        )
    if getattr(server_args, "enable_eplb", False):
        problems.append(
            "EPLB (--enable-eplb) is gated: it relocates experts across ranks at runtime, but the "
            "resident map is built once (static, 1:1). It overlaps keep-warm and is a no-op at "
            "ep_size==1. Drop --enable-eplb."
        )
    if a2a not in (None, "none", ""):
        problems.append(
            f"MoE all-to-all backend (moe_a2a_backend={a2a!r}) is unvalidated: its dispatch/combine "
            "kernels may assume all local experts are GPU-resident & contiguously indexed, which the "
            "K-slot indirection breaks. Use --moe-a2a-backend none."
        )
    if load_format == "dummy":
        problems.append(
            "--load-format dummy is incompatible: the host expert store reads REAL weights. Use a real "
            "checkpoint."
        )

    if problems:
        raise RuntimeError(
            "Paged Experts is incompatible with the current parallelism / placement config:\n  - "
            + "\n  - ".join(problems)
        )
