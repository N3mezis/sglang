"""Hybrid INT4-backbone + NVFP4-experts quantization.

For a checkpoint whose dense/attention backbone is RTN-INT4 (produced offline by the bench repack) and
whose routed MoE experts are NVIDIA ModelOpt NVFP4 (as shipped). No existing sglang config serves this
combination: the ModelOpt mixed config is fp8+nvfp4 only (its linear dispatch has no int4 branch), and the
compressed-tensors config can't hold ModelOpt-named experts. This config holds two sub-configs and routes
per layer type — Linear/embedding/LM-head -> INT4, FusedMoE -> ModelOptNvFp4FusedMoEMethod (which the
paged-experts fill reads in place, see paged_experts/pager._fill_nvfp4_from_checkpoint).

Motivating case: nvidia/GLM-5.2-NVFP4 on a 16 GB card — the 37.8 GB BF16 backbone must be int4 (~9.5 GB) to
fit resident alongside a K=1 streamed expert pool. INT4 is RTN (round-to-nearest, calibration-free): MLA
attention is quant-sensitive, so quality is measured via ppl, not assumed.

INT4 layout (fully self-contained — no gptq/awq/marlin packing to match):
  * ``qweight``      uint8 ``[out, in // 2]``  — two signed int4 nibbles per byte, offset-binary (stored
                     value = q + 8, q in [-8, 7]); low nibble = even input index, high nibble = odd.
  * ``weight_scale`` bf16  ``[out, num_groups]`` — symmetric per-group scale; group_size = in // num_groups.
Dequant (torch, per forward — perf is irrelevant at the < 1 tok/s streamed-expert regime): unpack -> (q *
scale) -> F.linear / F.embedding. Only one layer's weight is materialized at a time (bounded transient).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.nn import Parameter

from sglang.srt.layers.parameter import (
    GroupQuantScaleParameter,
    PackedvLLMParameter,
)
from sglang.srt.layers.quantization.base_config import (
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
import os

from sglang.srt.utils import set_weight_attrs

# Keep embed_tokens + LM-head int4 weights on the host (each ~0.5 GB) to free VRAM for a larger paged
# K-slot pool. They run once per token (not per layer), so a host gather / host matmul is a cheap trade.
_OFFLOAD_VOCAB = os.environ.get("SGLANG_INT4_OFFLOAD_VOCAB", "0") != "0"
_OFFLOAD_EMBED = os.environ.get("SGLANG_INT4_OFFLOAD_EMBED", "0") != "0"  # eager-only (see create_weights)
# Backbone int4 linears above this element count (out*in) chunk their dequant over the output dim to bound
# the bf16 transient. Default is above every GLM backbone linear -> INERT at the roomy K=1 budget (chunking
# added measurable per-token overhead). Lower it (e.g. 16000000 -> ~128 MB) only for a tight budget (K>=2).
_INT4_CHUNK_ELEMS = int(os.environ.get("SGLANG_INT4_CHUNK_ELEMS", "200000000"))
# LM-head dequant chunk (vocab rows/chunk). It is IN the captured decode graph, so on a tight VRAM budget
# (K=1 + 13 GB backbone leaves <1 GB) shrink it to keep the per-chunk bf16 transient small enough to fit.
_LMHEAD_CHUNK = int(os.environ.get("SGLANG_INT4_LMHEAD_CHUNK", "16384"))

# ---------------------------------------------------------------------------------------------------
# RTN int4 pack/quant helpers (shared with the offline repack in sglang-bench). Pure torch, CPU-safe.
# ---------------------------------------------------------------------------------------------------


def rtn_quantize_int4(w: torch.Tensor, group_size: int = 128):
    """Symmetric per-group RTN quantize a 2-D weight ``[out, in]`` to (packed uint8 ``[out, in//2]``,
    bf16 scales ``[out, num_groups]``). ``group_size`` divides ``in`` when possible, else falls back to a
    single per-output-row group (num_groups = 1). ``in`` must be even (2 nibbles per byte)."""
    assert w.dim() == 2, f"expected 2-D weight, got {tuple(w.shape)}"
    out, cin = w.shape
    assert cin % 2 == 0, f"input dim {cin} must be even to pack 2 int4/byte"
    g = group_size if (group_size > 0 and cin % group_size == 0) else cin
    ng = cin // g
    wf = w.float().reshape(out, ng, g)
    # symmetric: scale = max|w| / 7 (int4 range [-8,7]; use 7 so +max maps to +7, -max to -7)
    amax = wf.abs().amax(dim=2, keepdim=True).clamp_min(1e-8)
    scale = amax / 7.0
    q = torch.clamp(torch.round(wf / scale), -8, 7).to(torch.int8).reshape(out, cin)
    # offset-binary nibbles, pack even->low, odd->high
    nib = (q + 8).to(torch.uint8)
    packed = (nib[:, 0::2] | (nib[:, 1::2] << 4)).contiguous()
    return packed, scale.reshape(out, ng).to(torch.bfloat16)


def dequantize_int4(
    packed: torch.Tensor, scale: torch.Tensor, out_dtype=torch.bfloat16
) -> torch.Tensor:
    """Inverse of :func:`rtn_quantize_int4`: ``[out, in//2]`` uint8 + ``[out, ng]`` scale -> ``[out, in]``."""
    out, halfin = packed.shape
    cin = halfin * 2
    lo = (packed & 0x0F).to(torch.int16) - 8
    hi = (packed >> 4).to(torch.int16) - 8
    q = torch.empty(out, cin, dtype=torch.int16, device=packed.device)
    q[:, 0::2] = lo
    q[:, 1::2] = hi
    ng = scale.shape[1]
    g = cin // ng
    # multiply directly in out_dtype (bf16): int4 values (-8..7) are exact in bf16, and this avoids the
    # float32 intermediate (4 B/elem) that spikes ~1.15 GB transient VRAM and OOMs the VRAM-edge forward.
    w = q.reshape(out, ng, g).to(out_dtype) * scale.reshape(out, ng, 1).to(out_dtype)
    return w.reshape(out, cin)


def _register_int4_weights(
    layer, out_features, in_features, group_size, extra_weight_attrs, device=None
):
    """Register the ``qweight`` + ``weight_scale`` params (shared by linear + embedding methods).
    ``device="cpu"`` keeps the weight off the GPU (embed/LM-head offload — each runs once per token, not
    per layer, so a host gather / host matmul is cheap and frees ~0.5 GB VRAM each toward a larger K)."""
    g = group_size if (group_size > 0 and in_features % group_size == 0) else in_features
    ng = in_features // g
    # device="cpu" -> PINNED host (pin_memory) so the LM-head apply can async-stream chunks H2D even under
    # CUDA-graph capture (a pinned async copy is capturable; a pageable one is not).
    pin = device == "cpu"
    qweight = Parameter(
        torch.empty(out_features, in_features // 2, dtype=torch.uint8, device=device, pin_memory=pin),
        requires_grad=False,
    )
    weight_scale = Parameter(
        torch.empty(out_features, ng, dtype=torch.bfloat16, device=device, pin_memory=pin),
        requires_grad=False,
    )
    set_weight_attrs(qweight, {"input_dim": 1, "output_dim": 0})
    set_weight_attrs(weight_scale, {"input_dim": 1, "output_dim": 0})
    layer.register_parameter("qweight", qweight)
    layer.register_parameter("weight_scale", weight_scale)
    # weight_loader (+ any harmless extras like skip_block_quant_check) — default v1 loader copies, which
    # is correct at TP=1 (paged-experts is single-GPU; no shard math).
    set_weight_attrs(qweight, extra_weight_attrs)
    set_weight_attrs(weight_scale, extra_weight_attrs)


# ---------------------------------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------------------------------


class Int4LinearMethod(LinearMethodBase):
    """RTN-int4 weight-only linear. Dequant-then-matmul (one weight materialized per call)."""

    def __init__(self, quant_config: "HybridInt4NvFp4Config"):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype,
        **extra_weight_attrs,
    ):
        # Typed params so sglang's merged/qkv weight loaders (weight_loader_v2) fuse the checkpoint's
        # per-proj int4 tensors into the model's fused params (gate_up_proj, fused_qkv_a_proj_with_mqa).
        # Packing is along the INPUT dim (packed_dim=1, packed_factor=2); fusion is along the OUTPUT dim
        # (output_dim=0, un-packed) so the output-slice merge is clean. Requires "Int4LinearMethod" in
        # linear.WEIGHT_LOADER_V2_SUPPORTED.
        weight_loader = extra_weight_attrs.get("weight_loader")
        out = sum(output_partition_sizes)
        cin = input_size_per_partition
        g = self.quant_config.group_size
        g = g if (g > 0 and cin % g == 0) else cin
        qweight = PackedvLLMParameter(
            data=torch.empty(out, cin // 2, dtype=torch.uint8),
            input_dim=1,
            output_dim=0,
            packed_dim=1,
            packed_factor=2,
            weight_loader=weight_loader,
        )
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(out, cin // g, dtype=torch.bfloat16),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("qweight", qweight)
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer) -> None:
        # Typed params carry loader-only attrs; drop the wrapper so forward sees plain tensors. Keep packed
        # (pre-dequant would materialize the whole backbone in bf16 and defeat the VRAM saving). Guard with
        # hasattr: an MLA-absorbed proj (e.g. kv_b_proj) has had these freed during post_load_weights.
        if hasattr(layer, "qweight"):
            layer.qweight = Parameter(layer.qweight.data, requires_grad=False)
        if hasattr(layer, "weight_scale"):
            layer.weight_scale = Parameter(layer.weight_scale.data, requires_grad=False)

    def apply(self, layer, x: torch.Tensor, bias: Optional[torch.Tensor] = None):
        qw, sc = layer.qweight, layer.weight_scale
        out_f, halfin = qw.shape
        in_f = halfin * 2
        # Dequant materializes an [out,in] bf16 weight (+ int16 q intermediate). For the big MLA projs that
        # transient OOMs a tight VRAM budget (e.g. K=2 leaves <1 GB). Size-gate: small linears take the
        # fast full path; big ones chunk over the output dim so the transient stays ~bounded. Cheap check.
        if out_f * in_f <= _INT4_CHUNK_ELEMS:
            w = dequantize_int4(qw, sc, out_dtype=x.dtype)
            return F.linear(x, w, bias)
        chunk = max(1, _INT4_CHUNK_ELEMS // in_f)
        outs = []
        for lo in range(0, out_f, chunk):
            hi = min(lo + chunk, out_f)
            w = dequantize_int4(qw[lo:hi], sc[lo:hi], out_dtype=x.dtype)
            outs.append(F.linear(x, w, None if bias is None else bias[lo:hi]))
        return torch.cat(outs, dim=-1)


class Int4EmbeddingMethod(QuantizeMethodBase):
    """RTN-int4 for VocabParallelEmbedding / ParallelLMHead. ``embedding`` dequantizes only the gathered
    rows (cheap at decode); ``apply`` (LM-head logits) dequantizes the full weight for the vocab matmul."""

    def __init__(self, quant_config: "HybridInt4NvFp4Config"):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype,
        **extra_weight_attrs,
    ):
        # Offload the LM-head weight to host (env-gated) to free VRAM. ONLY the LM head, NOT the input
        # embedding: the embedding runs INSIDE the captured decode graph, and a CPU gather there is illegal
        # under CUDA-graph capture ("cannot copy CPU<->CUDA"). The LM head runs at the end (post-graph in
        # sglang), so a host matmul is capture-safe. type-name check avoids importing ParallelLMHead.
        is_lm_head = type(layer).__name__ == "ParallelLMHead"
        # SGLANG_INT4_OFFLOAD_EMBED additionally offloads the input embedding — ONLY safe in EAGER mode
        # (no CUDA-graph capture; a captured embed gather can't copy CPU<->GPU). Frees another ~0.5 GB
        # (e.g. to fit the MTP spec-decode draft worker).
        is_embed = type(layer).__name__ == "VocabParallelEmbedding"
        offload = (_OFFLOAD_VOCAB and is_lm_head) or (_OFFLOAD_EMBED and is_embed)
        dev = "cpu" if offload else None
        _register_int4_weights(
            layer,
            sum(output_partition_sizes),
            input_size_per_partition,
            self.quant_config.group_size,
            extra_weight_attrs,
            device=dev,
        )

    def process_weights_after_loading(self, layer) -> None:
        return

    def embedding(self, layer, input_: torch.Tensor) -> torch.Tensor:
        # gather the packed rows for the requested ids, then dequant only those (bounded work). When the
        # weight is host-offloaded, gather on CPU (ids -> cpu) and move only the small result to the GPU.
        dst = input_.device
        ids = input_.reshape(-1).to(layer.qweight.device)
        rows = dequantize_int4(
            layer.qweight[ids], layer.weight_scale[ids], out_dtype=layer.weight_scale.dtype
        )
        return rows.reshape(*input_.shape, -1).to(dst)

    def apply(self, layer, x: torch.Tensor, bias: Optional[torch.Tensor] = None):
        # LM head is the largest weight (vocab x hidden ~ 150K x 5120): a full dequant materializes ~3 GB
        # transient and OOMs the VRAM-edge forward. Chunk the dequant+matmul over the vocab dim so peak
        # transient is bounded to CHUNK rows (~few hundred MB), streaming logits directly into the output.
        qw, sc = layer.qweight, layer.weight_scale
        vocab = qw.shape[0]
        dst = x.device
        CHUNK = _LMHEAD_CHUNK
        if qw.device.type == "cpu":
            # HOST-OFFLOADED: weights stay pinned on host (freeing VRAM); stream each vocab chunk to the GPU
            # via a PINNED async copy (capturable), dequant + matmul on GPU, so nothing runs on the CPU and
            # the full vocab weight never lives in VRAM. The per-chunk GPU tensors free between chunks.
            out = torch.empty(x.shape[0], vocab, dtype=x.dtype, device=dst)
            for lo in range(0, vocab, CHUNK):
                hi = min(lo + CHUNK, vocab)
                qc = qw[lo:hi].to(dst, non_blocking=True)
                scc = sc[lo:hi].to(dst, non_blocking=True)
                w = dequantize_int4(qc, scc, out_dtype=x.dtype)
                out[:, lo:hi] = F.linear(x, w, None if bias is None else bias[lo:hi].to(dst))
            return out
        # resident on GPU: chunk the dequant only to bound the transient (LM head is the largest weight).
        if vocab <= CHUNK:
            w = dequantize_int4(qw, sc, out_dtype=x.dtype)
            return F.linear(x, w, bias)
        out = torch.empty(x.shape[0], vocab, dtype=x.dtype, device=dst)
        for lo in range(0, vocab, CHUNK):
            hi = min(lo + CHUNK, vocab)
            w = dequantize_int4(qw[lo:hi], sc[lo:hi], out_dtype=x.dtype)
            out[:, lo:hi] = F.linear(x, w, None if bias is None else bias[lo:hi])
        return out


# ---------------------------------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------------------------------


class HybridInt4NvFp4Config(QuantizationConfig):
    """INT4 (backbone linear/embedding/LM-head) + NVFP4 (routed FusedMoE experts)."""

    def __init__(
        self,
        nvfp4_config,
        group_size: int = 128,
        exclude_modules: Optional[List[str]] = None,
        packed_modules_mapping: Optional[Dict] = None,
    ):
        super().__init__()
        self.nvfp4_config = nvfp4_config
        self.group_size = group_size
        self.exclude_modules = list(exclude_modules or [])
        self.packed_modules_mapping = packed_modules_mapping or {}

    @classmethod
    def get_name(cls) -> str:
        return "hybrid_int4_nvfp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @staticmethod
    def get_config_filenames() -> List[str]:
        return []

    def get_scaled_act_names(self) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HybridInt4NvFp4Config":
        from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config

        group_size = int(config.get("int4_group_size", 128))
        # The NVFP4 expert sub-config: reuse the ModelOpt fields verbatim, only forcing a quant_method the
        # ModelOptFp4Config parser accepts. The bench repack copies the original modelopt quantization_config
        # into ``nvfp4`` (nested) or leaves the modelopt keys at top level.
        nvfp4_src = dict(config.get("nvfp4", config))
        nvfp4_src["quant_method"] = "modelopt_fp4"
        nvfp4_config = ModelOptFp4Config.from_config(nvfp4_src)
        return cls(
            nvfp4_config=nvfp4_config,
            group_size=group_size,
            exclude_modules=config.get("exclude_modules") or config.get("ignore"),
            packed_modules_mapping=config.get("packed_modules_mapping"),
        )

    def is_layer_excluded(self, prefix: str) -> bool:
        # Match a WHOLE module path segment, not a raw substring — else "mlp.gate" (the MoE router) would
        # wrongly match "mlp.gate_up_proj" / "mlp.gate_proj" and drop them from int4, diverging from the
        # repack (which int4'd them) and causing a .qweight-vs-.weight load mismatch.
        for m in self.exclude_modules:
            if prefix == m or prefix.endswith("." + m) or ("." + m + ".") in prefix:
                return True
        return False

    def get_quant_method(
        self, layer, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.quantization.modelopt_quant import (
            ModelOptNvFp4FusedMoEMethod,
        )
        from sglang.srt.layers.quantization.unquant import (
            UnquantizedEmbeddingMethod,
            UnquantizedLinearMethod,
        )
        from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding

        if isinstance(layer, FusedMoE):
            if self.nvfp4_config.is_layer_excluded(prefix):
                return None
            return ModelOptNvFp4FusedMoEMethod(self.nvfp4_config)
        if isinstance(layer, LinearBase):
            if self.is_layer_excluded(prefix):
                return UnquantizedLinearMethod()
            return Int4LinearMethod(self)
        if isinstance(layer, VocabParallelEmbedding):
            # covers ParallelLMHead (subclass). Skip if excluded -> unquantized embedding.
            if self.is_layer_excluded(prefix):
                return UnquantizedEmbeddingMethod()
            return Int4EmbeddingMethod(self)
        return None
