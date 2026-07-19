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


def check_paged_experts_quant(hf_text_config: Any) -> None:
    """Raise ``RuntimeError`` if the checkpoint's quantization is not one the paging path supports.

    The host store's fill + gather understand unquantized (bf16/fp16), gptq-marlin int4, and fp8
    BLOCK-quant tensor layouts. Anything else (AWQ, per-tensor fp8, compressed-tensors, ...) would be
    routed through the wrong fill and load WRONG weights — reject it up front instead.
    """
    qc = getattr(hf_text_config, "quantization_config", None)
    if qc is None:
        return  # unquantized (bf16/fp16) — supported
    quant_method = (
        (qc.get("quant_method") or "").lower() if isinstance(qc, dict) else ""
    )
    if quant_method == "gptq":
        return
    if quant_method == "fp8":
        # Only BLOCK quantization: its per-expert weight + block-scale rows copy straight from the
        # checkpoint. Per-tensor fp8 has [E]/[E,2] scalar scales the pinned transfer can't page
        # (sub-8-byte rows) and a different post-load path.
        if isinstance(qc, dict) and qc.get("weight_block_size"):
            return
        raise RuntimeError(
            "Paged Experts supports fp8 only with block quantization (weight_block_size, e.g. "
            "[128, 128]); this checkpoint uses per-tensor fp8 scales. Use a block-quant fp8, GPTQ "
            "int4, or unquantized checkpoint, or run without --enable-paged-experts."
        )
    if quant_method == "compressed-tensors":
        # NVFP4 (nvfp4-pack-quantized): packed fp4 weights + per-group-of-16 fp8 block scales page
        # as ordinary per-expert rows; the small per-expert global/input scale scalars ride the
        # deferred resident-scalar path (they are too small for the pinned gather). Only the nvfp4
        # packing is wired; other compressed-tensors packings (int-quant, W8A8, ...) are not.
        fmt = (qc.get("format") or "").lower() if isinstance(qc, dict) else ""
        if "nvfp4" in fmt:
            return
        # pack-quantized int4/int8 (the checkpoints the Hub labels "AWQ-4bit"/"w4a16" — actually
        # compressed-tensors, not classic AWQ). Bit-layout-compatible with GPTQ marlin, so it fills
        # via the gptq-marlin repack. See _fill_ct_wna16_from_checkpoint. v1 supports symmetric,
        # group-wise, no act-order only (asym zero-points / g_idx paging are unwired).
        if "pack-quantized" in fmt:
            groups = qc.get("config_groups") or {} if isinstance(qc, dict) else {}
            w = (next(iter(groups.values()), {}) or {}).get("weights", {}) or {}
            wtype = (w.get("type") or "").lower()
            bits = w.get("num_bits")
            actorder = w.get("actorder")
            if wtype != "int":
                raise RuntimeError(
                    f"Paged Experts supports compressed-tensors pack-quantized only with int "
                    f"weights; this checkpoint has weights.type={wtype or 'unknown'!r}."
                )
            if bits not in (4, 8):
                raise RuntimeError(
                    f"Paged Experts supports compressed-tensors int pack-quantized at 4 or 8 bits; "
                    f"this checkpoint uses num_bits={bits!r}."
                )
            if not w.get("symmetric", True):
                raise RuntimeError(
                    "Paged Experts supports compressed-tensors pack-quantized only with symmetric "
                    "weights (no zero-points); this checkpoint is asymmetric. Use a symmetric quant "
                    "or run without --enable-paged-experts."
                )
            # actorder "weight"/"static" bake the channel permutation into the weights at quant time
            # (no g_idx stored) — the fill handles them like plain grouped quant. Only "group" stores a
            # runtime g_idx (would need per-expert g_idx paging) and "dynamic" recomputes it online.
            if actorder not in (None, "", "static", "weight"):
                raise RuntimeError(
                    f"Paged Experts does not support runtime act-order (g_idx paging) for "
                    f"compressed-tensors pack-quantized; this checkpoint uses actorder={actorder!r} "
                    f"(only baked-in 'weight'/'static' act-order or none is supported)."
                )
            return
        # float-quantized: per-channel (or per-tensor) fp8 weights with DYNAMIC activations. The fp8
        # weights + per-output-channel [out,1] scales page as ordinary per-expert rows (no repack, no
        # resident-scalar table needed). Static input scales would need input-scale paging — reject.
        # See _fill_ct_fp8_channel_from_checkpoint.
        if "float-quantized" in fmt:
            groups = qc.get("config_groups") or {} if isinstance(qc, dict) else {}
            g = next(iter(groups.values()), {}) or {}
            w = g.get("weights", {}) or {}
            act = g.get("input_activations") or {}
            strat = (w.get("strategy") or "").lower()
            if (w.get("num_bits") or 8) != 8 or (w.get("type") or "float").lower() != "float":
                raise RuntimeError(
                    f"Paged Experts supports compressed-tensors float-quantized only as 8-bit fp8; "
                    f"this checkpoint uses num_bits={w.get('num_bits')!r} type={w.get('type')!r}."
                )
            # Only per-CHANNEL weight scales ([out,1] rows that page) are wired + tested. Per-tensor
            # fp8 has [E]/[E,2] scalar scales that need the resident-scalar path (as nvfp4 does) and a
            # different fill; no per-tensor+dynamic checkpoint exists in the wild to validate, so reject.
            if strat not in ("channel", ""):
                raise RuntimeError(
                    f"Paged Experts supports compressed-tensors fp8 float-quantized only with "
                    f"per-channel weight scales; this checkpoint uses strategy={strat!r} (per-tensor "
                    "fp8 scalar scales are not wired)."
                )
            if act and not act.get("dynamic", False):
                raise RuntimeError(
                    "Paged Experts supports compressed-tensors fp8 float-quantized only with dynamic "
                    "activation quantization; this checkpoint has static input scales."
                )
            return
        if "int-quantized" in fmt:
            # int8 W8A8 (per-channel int8 weights + dynamic int8 activations). sglang's compressed-
            # tensors int8 fused-MoE scheme is NPU-only (CUDA raises NotImplementedError in
            # get_moe_scheme) — there is no GPU method for paging to wrap. Blocked upstream, not by us.
            raise RuntimeError(
                "Paged Experts cannot serve compressed-tensors int-quantized (int8 W8A8) MoE: sglang "
                "has no CUDA fused-MoE for it (the W8A8Int8 MoE scheme is NPU-only). Use an nvfp4, int "
                "pack-quantized, fp8 float-quantized, block-quant fp8, or GPTQ int4 checkpoint instead."
            )
        raise RuntimeError(
            f"Paged Experts supports compressed-tensors with nvfp4, int pack-quantized, or fp8 "
            f"float-quantized packings; this checkpoint uses format={fmt or 'unknown'!r}. Use one of "
            "those, block-quant fp8, GPTQ int4, or unquantized, or run without --enable-paged-experts."
        )
    if quant_method == "mxfp4":
        # MXFP4 (gpt-oss): packed fp4 weights + per-group-of-32 e8m0 block scales + per-expert bf16
        # biases, filled via the Marlin mxfp4 repack (SM90/SM120). See _fill_mxfp4_from_checkpoint.
        return
    if quant_method == "awq":
        # classic AWQ (asymmetric, per-group zero-points). sglang converts it to the awq-marlin MoE
        # layout; the paged fill mirrors that repack (awq_marlin_moe_repack + zero-point conversion).
        # See _fill_awq_marlin_from_checkpoint. Serve with --dtype float16 (marlin scales are fp16).
        bits = qc.get("bits") if isinstance(qc, dict) else None
        if bits in (4, 8):
            return
        raise RuntimeError(
            f"Paged Experts supports AWQ only at 4 or 8 bits; this checkpoint uses bits={bits!r}."
        )
    if quant_method == "auto-round":
        # Intel AutoRound. Despite auto_gptq packing, sglang's AutoRound MoE gates marlin off via
        # check_moe_marlin_supports_layer, so it lands on the moe_wna16 uint8 triton method — filled
        # by _fill_moe_wna16_from_checkpoint (dispatched on the uint8 qweight dtype). Symmetric
        # auto_gptq only for now; asymmetric / auto_awq (wna16 zero-point path) is unwired.
        fmt = (qc.get("packing_format") or "").lower() if isinstance(qc, dict) else ""
        sym = qc.get("sym", True) if isinstance(qc, dict) else True
        bits = qc.get("bits") if isinstance(qc, dict) else None
        if "auto_gptq" in fmt and sym and bits in (4, 8):
            return
        raise RuntimeError(
            f"Paged Experts supports AutoRound only as symmetric auto_gptq (4/8-bit); this checkpoint "
            f"uses packing_format={fmt or 'unknown'!r} sym={sym} bits={bits!r}. Asymmetric / auto_awq "
            "AutoRound is not supported yet."
        )
    raise RuntimeError(
        f"Paged Experts does not support quant_method={quant_method or 'unknown'!r}: the host "
        "store handles unquantized (bf16/fp16), gptq-marlin int4, fp8 block-quant, and mxfp4 "
        "checkpoints only. Other packings (e.g. AWQ) would be routed through the wrong fill and load "
        "wrong weights. Use a supported checkpoint, or run without --enable-paged-experts."
    )
