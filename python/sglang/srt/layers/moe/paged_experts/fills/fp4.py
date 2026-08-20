"""FP4 expert fills: mxfp4 (gpt-oss, marlin repack + per-expert biases) and nvfp4 (compressed-tensors
nvfp4-pack, swizzled fp8 block scales + a resident full-E scalar table returned to the method).
"""

import json
import logging
import os
from typing import Dict

import torch

from ..store import ExpertStore
from .base import ExpertFill

logger = logging.getLogger(__name__)
from .checkpoint import (
    _drop_file_cache,
    _experts_prefix,
    _proj_names,
    _snapshot_dir,
    _weight_map,
)


def _fill_mxfp4_from_checkpoint(
    store: ExpertStore, model_path: str, layer_idx: int
) -> None:
    """MXFP4 (gpt-oss): reproduce the Mxfp4MarlinMoEMethod load+repack for ALL E experts into the host
    store. The checkpoint stores STACKED FUSED experts under ``...experts.``:
    ``gate_up_proj_blocks`` u8 [E, 2*inter, hidden//32, 16] (packed fp4, 16 B = 32 vals), ``_scales``
    u8 [E, 2*inter, hidden//32] (e8m0), ``_bias`` bf16 [E, 2*inter]; ``down_proj_*`` likewise on
    [E, hidden, ...]. Load them into the base method's PADDED pre-repack buffers (hidden->round_up(256),
    inter->round_up(128); gate/up occupy the two intermediate halves with per-half padding), run
    ``prepare_moe_mxfp4_layer_for_marlin``, then fill every store tensor from the repacked result. All
    tensors (marlin weights + scales + permuted biases) page per-expert.
    """
    from safetensors import safe_open

    from sglang.srt.layers.quantization.marlin_utils_fp4 import (
        deinterleave_moe_mxfp4_w13_for_marlin,
        prepare_moe_mxfp4_layer_for_marlin,
    )
    from sglang.srt.utils import round_up

    snap = _snapshot_dir(model_path)
    cfg = json.load(open(os.path.join(snap, "config.json")))
    tcfg = cfg.get("text_config", cfg)
    hidden = tcfg["hidden_size"]
    inter = tcfg.get("moe_intermediate_size") or tcfg["intermediate_size"]
    wmap = _weight_map(snap)
    pre = _experts_prefix(wmap, layer_idx)
    E = store.E
    p_inter, p_hidden = round_up(inter, 128), round_up(hidden, 256)

    _loaded_shards: set = set()

    def load(stem: str) -> torch.Tensor:
        name = f"{pre}{stem}"
        path = os.path.join(snap, wmap[name])
        _loaded_shards.add(path)
        with safe_open(path, framework="pt") as f:
            return f.get_tensor(name)

    gu_b = (
        load("gate_up_proj_blocks").reshape(E, 2 * inter, hidden // 2).view(torch.int8)
    )
    dn_b = load("down_proj_blocks").reshape(E, hidden, inter // 2).view(torch.int8)
    gu_s = load("gate_up_proj_scales").view(torch.uint8)  # [E, 2*inter, hidden//32]
    dn_s = load("down_proj_scales").view(torch.uint8)  # [E, hidden, inter//32]
    gu_bias = load("gate_up_proj_bias")  # [E, 2*inter] bf16
    dn_bias = load("down_proj_bias")  # [E, hidden]
    for _p in _loaded_shards:  # release the consumed shards' page cache (bound peak load RAM)
        _drop_file_cache(_p)

    e8 = torch.float8_e8m0fnu
    # Match the base method's create_weights buffers (intermediate padded to p_inter, hidden to p_hidden).
    # gpt-oss's gate/up rows are INTERLEAVED [g0,u0,g1,u1,...]; the loader places the 2*inter real rows
    # contiguously in [:2*inter] (rest is pad), then deinterleave_moe_mxfp4_w13_for_marlin (below) views
    # [.., p_inter, 2, ..] to split into [gate; up]. w2 (down) has no gate/up split.
    w13 = torch.zeros(E, 2 * p_inter, p_hidden // 2, dtype=torch.int8)
    w2 = torch.zeros(E, p_hidden, p_inter // 2, dtype=torch.int8)
    w13_s = torch.full((E, 2 * p_inter, p_hidden // 32), 127, dtype=torch.uint8)
    w2_s = torch.full((E, p_hidden, p_inter // 32), 127, dtype=torch.uint8)
    w13_bias = torch.zeros(E, 2 * p_inter, dtype=gu_bias.dtype)
    w2_bias = torch.zeros(E, p_hidden, dtype=dn_bias.dtype)
    w13[:, : 2 * inter, : hidden // 2] = gu_b
    w13_s[:, : 2 * inter, : hidden // 32] = gu_s
    w13_bias[:, : 2 * inter] = gu_bias
    w2[:, :hidden, : inter // 2] = dn_b
    w2_s[:, :hidden, : inter // 32] = dn_s
    w2_bias[:, :hidden] = dn_bias

    dev = store.device
    mock = torch.nn.Module()
    mock.hidden_size = hidden
    mock.orig_dtype = gu_bias.dtype
    for n, t in (
        ("w13_weight", w13),
        ("w2_weight", w2),
        ("w13_weight_scale_inv", w13_s.view(e8)),
        ("w2_weight_scale_inv", w2_s.view(e8)),
        ("w13_weight_bias", w13_bias),
        ("w2_weight_bias", w2_bias),
    ):
        setattr(mock, n, torch.nn.Parameter(t.to(dev), requires_grad=False))
    # gpt-oss stores gate/up interleaved (swiglu with gemm1_alpha); match the base method's
    # deinterleave-then-repack order so the paged weights equal the native load.
    deinterleave_moe_mxfp4_w13_for_marlin(mock)
    prepare_moe_mxfp4_layer_for_marlin(mock)

    for name in store.gpu:
        t = getattr(mock, name).data.contiguous().cpu()
        expected = (E, *store.gpu[name].shape[1:])
        assert tuple(t.shape) == expected, (name, tuple(t.shape), expected)
        store.fill_tensor(name, t)


def _fill_nvfp4_from_checkpoint(store, model_path, layer_idx, device):
    """NVFP4 (compressed-tensors nvfp4-pack). Packed uint8 weights copy straight into the host store;
    per-group-of-16 fp8 block scales are swizzled to the cutlass 128x4 layout (matching
    CompressedTensorsW4A4Nvfp4MoE.process_weights_after_loading, non-trtllm path) and stored as paged
    tensors. The tiny per-expert global/input scalars can't page (sub-8-byte rows), so this returns the
    four runtime-relevant ones as a resident full-E table {name: [E] f32}; forward._gemm_hidden scatters
    them into the K slots each step by the live residency map. w1=gate, w3=up, w2=down.
    """
    from safetensors import safe_open

    from sglang.srt.layers.quantization.utils import swizzle_blockscale

    E = store.E
    assert (
        store.gpu["w13_weight"].dtype == torch.uint8
    ), "nvfp4 fill expects uint8 packed weights"
    snap = _snapshot_dir(model_path)
    wmap = _weight_map(snap)
    pre = _experts_prefix(wmap, layer_idx)
    gate, up, down = _proj_names(wmap, pre)
    # SWAP gate<->up for the fused w13. The MoE GEMM silus w13's FIRST half, and this checkpoint family
    # needs gate in the SECOND half to yield silu(gate)*up -- the in-place nvfp4 store (the nvfp4 path
    # known to serve correctly) does exactly this, and _fill_bf16 carries the same fix. Swapping the NAMES
    # here rather than the individual destinations flips ALL of it together: packed weights, swizzled
    # block scales, AND the per-half global-scale bookkeeping (w1_* must describe whichever projection
    # occupies the first half). Flipping only the weights leaves the alphas describing the wrong half.
    gate, up = up, gate

    # raw (pre-swizzle) block-scale + global-scale collectors, filled per expert then transformed en masse
    w13_sc_raw = w2_sc_raw = None
    z = lambda: torch.empty(E, dtype=torch.float32)
    w1_wgs, w2_wgs, w1_igs, w3_igs, w2_igs = z(), z(), z(), z(), z()

    by_shard: Dict[str, list] = {}
    for e in range(E):
        for proj in (gate, up, down):
            for suf in (
                "weight_packed",
                "weight_scale",
                "weight_global_scale",
                "input_global_scale",
            ):
                by_shard.setdefault(wmap[f"{pre}{e}.{proj}.{suf}"], []).append(
                    (e, proj, suf)
                )
    for shard, items in by_shard.items():
        _shard_path = os.path.join(snap, shard)
        with safe_open(_shard_path, framework="pt") as f:
            for e, proj, suf in items:
                t = f.get_tensor(f"{pre}{e}.{proj}.{suf}")
                is_down = proj == down
                if suf == "weight_packed":
                    if is_down:
                        store.row("w2_weight", e).copy_(t)
                    else:
                        row = store.row("w13_weight", e)
                        half = row.shape[0] // 2
                        (row[:half] if proj == gate else row[half:]).copy_(t)
                elif suf == "weight_scale":
                    if is_down:
                        if w2_sc_raw is None:
                            w2_sc_raw = torch.empty((E, *t.shape), dtype=t.dtype)
                        w2_sc_raw[e].copy_(t)
                    else:
                        if w13_sc_raw is None:
                            w13_sc_raw = torch.empty(
                                (E, t.shape[0] * 2, t.shape[1]), dtype=t.dtype
                            )
                        half = w13_sc_raw.shape[1] // 2
                        (
                            w13_sc_raw[e][:half]
                            if proj == gate
                            else w13_sc_raw[e][half:]
                        ).copy_(t)
                elif suf == "weight_global_scale":
                    if proj == gate:
                        w1_wgs[e] = t.flatten()[0]
                    elif is_down:
                        w2_wgs[e] = t.flatten()[0]
                else:  # input_global_scale
                    dst = w1_igs if proj == gate else (w2_igs if is_down else w3_igs)
                    dst[e] = t.flatten()[0]
        _drop_file_cache(_shard_path)  # release this shard's page cache before the next

    # swizzle block scales to the cutlass 128x4 layout (same transform the method's PWAL applies)
    store.fill_tensor(
        "w13_weight_scale", swizzle_blockscale(w13_sc_raw.to(device)).cpu()
    )
    store.fill_tensor("w2_weight_scale", swizzle_blockscale(w2_sc_raw.to(device)).cpu())

    # derived scalars (cutlass/trtllm path): weight_scale_2 = 1/weight_global_scale.
    w1_wgs, w2_wgs = w1_wgs.to(device), w2_wgs.to(device)
    w1_igs, w3_igs, w2_igs = w1_igs.to(device), w3_igs.to(device), w2_igs.to(device)
    w13_ws2 = 1.0 / w1_wgs  # per-expert weight_scale_2 [E]
    w2_ws2 = 1.0 / w2_wgs
    w13_iq = torch.minimum(w1_igs, w3_igs)  # per-expert input_scale_quant (1/input_scale) [E]
    # Upstream's NVFP4 fused-MoE (ModelOptNvFp4FusedMoEMethod, flashinfer_cutlass/trtllm) collapses the
    # per-expert input scale to a PER-TENSOR max: input_scale = 1/iq, max_input_scale = 1/min(iq). The
    # kernel quantizes activations with 1/max, so the alphas MUST use max_input_scale too — with a
    # per-expert input scale the activation-quant and dequant scales disagree and every expert's output is
    # mis-scaled by its own factor. That is fluent-but-wrong text whose flavour tracks which experts happen
    # to be resident (it looks non-deterministic at temperature 0). g*_alphas stay per-expert
    # (max_input_scale * per-expert weight_scale_2); *_input_scale_quant become 0-dim per-tensor scalars.
    w13_is_max = (1.0 / w13_iq).max()
    w2_is_max = (1.0 / w2_igs).max()
    return {
        "g1_alphas": (w13_is_max * w13_ws2).float(),
        "g2_alphas": (w2_is_max * w2_ws2).float(),
        "w13_input_scale_quant": (1.0 / w13_is_max).float().reshape(()),
        "w2_input_scale_quant": (1.0 / w2_is_max).float().reshape(()),
    }


class Mxfp4Fill(ExpertFill):
    name = "mxfp4"

    def matches(self, store, quant_method: str) -> bool:
        return "w13_weight_bias" in store.gpu

    def fill(self, store, model_path, layer_idx, device):
        _fill_mxfp4_from_checkpoint(store, model_path, layer_idx)
        return None


class Nvfp4Fill(ExpertFill):
    name = "nvfp4"

    def matches(self, store, quant_method: str) -> bool:
        g = store.gpu
        return (
            "w13_weight_scale" in g
            and "w13_weight" in g
            and g["w13_weight"].dtype == torch.uint8
            and "w13_weight_bias" not in g  # not mxfp4 (also uint8 weight + scale)
        )

    def fill(self, store, model_path, layer_idx, device):
        # returns the resident full-E scalar table -> method._nvfp4_full_e
        return _fill_nvfp4_from_checkpoint(store, model_path, layer_idx, device)


def _fill_dsv4_fp4_from_checkpoint(
    store: ExpertStore, model_path: str, layer_idx: int
) -> None:
    """DeepSeek-V4 MXFP4 routed experts: int8-packed fp4 weights + e8m0 block scales (block 32),
    filled by a direct per-expert copy — mirrors the Fp8MoEMethod ``is_fp4_experts`` branch on the
    non-DeepGEMM (Hopper) path, which keeps weights int8-packed and casts the e8m0 scale to float32 at
    load with no repack. Host ``w13_weight=[E,2*I,H//2]`` int8 (concat gate=w1 then up=w3 on dim 0),
    ``w2_weight=[E,H,I//2]`` int8; ``w13_weight_scale_inv=[E,2*I,H//32]`` and
    ``w2_weight_scale_inv=[E,H,I//32]`` float32. No biases, no per-expert scalars.

    Checkpoint keys are the RAW DSV4 layout ``layers.N.ffn.experts.{e}.{w1|w3|w2}.{weight|scale}`` (w1
    gate, w3 up, w2 down) — NOT the model-side remapped ``mlp...gate_proj.weight_scale_inv`` names.

    The resident scale is float32 only off the DeepGEMM ue8m0 path (Hopper / non-Blackwell). On a
    Blackwell + DeepGEMM build the resident scale is a packed ue8m0 layout this direct copy would not
    reproduce, so assert against it rather than silently mis-fill.
    """
    from safetensors import safe_open

    assert store.gpu["w13_weight"].dtype == torch.int8, (
        "dsv4-fp4 fill expects int8-packed fp4 weights",
        store.gpu["w13_weight"].dtype,
    )
    assert store.gpu["w13_weight_scale_inv"].dtype == torch.float32, (
        "dsv4-fp4 fill expects float32 block scales (packed-ue8m0/DeepGEMM layout unsupported)",
        store.gpu["w13_weight_scale_inv"].dtype,
    )
    snap = _snapshot_dir(model_path)
    wmap = _weight_map(snap)
    pre = _experts_prefix(wmap, layer_idx)
    gate, up, down = _proj_names(wmap, pre)  # DSV4 -> (w1, w3, w2)
    by_shard: Dict[str, list] = {}
    for e in range(store.E):
        for proj in (gate, up, down):
            for suffix in ("weight", "scale"):
                by_shard.setdefault(wmap[f"{pre}{e}.{proj}.{suffix}"], []).append(
                    (e, proj, suffix)
                )
    for shard, items in by_shard.items():
        _shard_path = os.path.join(snap, shard)
        with safe_open(_shard_path, framework="pt") as f:
            for e, proj, suffix in items:
                t = f.get_tensor(f"{pre}{e}.{proj}.{suffix}")
                base = "w2_weight" if proj == down else "w13_weight"
                # checkpoint '.scale' (e8m0) -> store '..._scale_inv' (float32; cast on copy, exactly
                # as the resident weight_loader does)
                name = base if suffix == "weight" else base + "_scale_inv"
                row = store.row(name, e)
                if proj == down:
                    row.copy_(t)
                    continue
                # w13 packs gate (first half of dim 0) then up (second half); the block scales follow
                # the same row split, so the same halving works for both weight and scale.
                half = row.shape[0] // 2
                if proj == gate:
                    row[:half].copy_(t)
                else:  # up
                    row[half:].copy_(t)
        _drop_file_cache(_shard_path)  # release this shard's page cache before the next


class Dsv4Fp4Fill(ExpertFill):
    name = "dsv4-fp4"

    def matches(self, store, quant_method: str) -> bool:
        # DSV4 mxfp4 (Fp8MoEMethod is_fp4_experts): int8-packed weights + block scales under the
        # _scale_inv name. Distinct from fp8-block (e4m3 weights), mxfp4 (has w13_weight_bias), and
        # nvfp4 (uint8 weights + w13_weight_scale, not _scale_inv).
        return (
            "w13_weight_scale_inv" in store.gpu
            and store.gpu["w13_weight"].dtype == torch.int8
        )

    def fill(self, store, model_path, layer_idx, device):
        _fill_dsv4_fp4_from_checkpoint(store, model_path, layer_idx)
        return None


def _fill_dsv4_mxfp4_marlin_from_checkpoint(
    store: ExpertStore, model_path: str, layer_idx: int
) -> None:
    """DeepSeek-V4 MXFP4 routed experts served through ``Mxfp4MarlinMoEMethod`` (``--moe-runner-backend
    marlin`` — the working SM90/SM120 path; ``auto`` falls to triton, which can't consume packed fp4).
    Reproduces that method's load+repack for ALL E experts into the host store, whose paged params are the
    POST-repack marlin layout: ``w13_weight``/``w2_weight`` int8 + ``w13_weight_scale``/``w2_weight_scale``
    ``float8_e8m0fnu`` (no ``_scale_inv``, no bias).

    Unlike gpt-oss (``_fill_mxfp4_from_checkpoint``), DSV4 stores SEPARATE, PER-EXPERT gate/up tensors
    (``w1``=gate, ``w3``=up, ``w2``=down; ``.weight`` int8-packed fp4, ``.scale`` e8m0) and has no bias, so
    we place w1/w3 straight into the two padded intermediate halves (gate ``[0:I]``, up ``[p_inter:p_inter+I]``
    — exactly the deinterleaved layout the loader produces) and SKIP ``deinterleave_moe_mxfp4_w13_for_marlin``
    (DSV4 is silu, ``gemm1_alpha is None``). ``prepare_moe_mxfp4_layer_for_marlin`` then repacks + renames the
    scales, matching the resident K-slot layout exactly. For DSV4 the padding is a no-op (p_inter=I=2048,
    p_hidden=H=4096).
    """
    from safetensors import safe_open

    from sglang.srt.layers.quantization.marlin_utils_fp4 import (
        prepare_moe_mxfp4_layer_for_marlin,
    )
    from sglang.srt.utils import round_up

    # Post-repack the marlin qweight is int32; the stable signature is the e8m0 scale.
    assert store.gpu["w13_weight_scale"].dtype == torch.float8_e8m0fnu, (
        "dsv4-mxfp4-marlin fill expects e8m0 marlin scales",
        store.gpu["w13_weight_scale"].dtype,
    )

    snap = _snapshot_dir(model_path)
    cfg = json.load(open(os.path.join(snap, "config.json")))
    tcfg = cfg.get("text_config", cfg)
    hidden = tcfg["hidden_size"]
    inter = tcfg.get("moe_intermediate_size") or tcfg["intermediate_size"]
    wmap = _weight_map(snap)
    pre = _experts_prefix(wmap, layer_idx)
    gate, up, down = _proj_names(wmap, pre)  # DSV4 -> (w1, w3, w2)
    E = store.E
    p_inter, p_hidden = round_up(inter, 128), round_up(hidden, 256)
    e8 = torch.float8_e8m0fnu

    # Pre-repack buffers matching Mxfp4MarlinMoEMethod.create_weights (scales default 127 == e8m0 1.0).
    w13 = torch.zeros(E, 2 * p_inter, p_hidden // 2, dtype=torch.int8)
    w2 = torch.zeros(E, p_hidden, p_inter // 2, dtype=torch.int8)
    w13_s = torch.full((E, 2 * p_inter, p_hidden // 32), 127, dtype=torch.uint8)
    w2_s = torch.full((E, p_hidden, p_inter // 32), 127, dtype=torch.uint8)

    by_shard: Dict[str, list] = {}
    for e in range(E):
        for proj in (gate, up, down):
            for suffix in ("weight", "scale"):
                by_shard.setdefault(wmap[f"{pre}{e}.{proj}.{suffix}"], []).append(
                    (e, proj, suffix)
                )
    for shard, items in by_shard.items():
        _shard_path = os.path.join(snap, shard)
        with safe_open(_shard_path, framework="pt") as f:
            for e, proj, suffix in items:
                t = f.get_tensor(f"{pre}{e}.{proj}.{suffix}")
                if proj == down:
                    if suffix == "weight":
                        w2[e, :hidden, : inter // 2] = t
                    else:
                        w2_s[e, :hidden, : inter // 32] = t.view(torch.uint8)
                else:
                    # gate (w1) -> first padded half [0:I]; up (w3) -> second half [p_inter:p_inter+I]
                    start = 0 if proj == gate else p_inter
                    if suffix == "weight":
                        w13[e, start : start + inter, : hidden // 2] = t
                    else:
                        w13_s[e, start : start + inter, : hidden // 32] = t.view(
                            torch.uint8
                        )
        _drop_file_cache(_shard_path)  # release this shard's page cache before the next

    dev = store.device
    mock = torch.nn.Module()
    mock.hidden_size = hidden
    mock.orig_dtype = torch.bfloat16  # == params_dtype; drives the scale repack path
    for n, t in (
        ("w13_weight", w13),
        ("w2_weight", w2),
        ("w13_weight_scale_inv", w13_s.view(e8)),  # _inv name -> prepare renames + deletes it
        ("w2_weight_scale_inv", w2_s.view(e8)),
    ):
        setattr(mock, n, torch.nn.Parameter(t.to(dev), requires_grad=False))
    # DSV4 is silu (gemm1_alpha is None) -> no deinterleave; no biases.
    prepare_moe_mxfp4_layer_for_marlin(mock)

    for name in store.gpu:
        t = getattr(mock, name).data.contiguous().cpu()
        expected = (E, *store.gpu[name].shape[1:])
        assert tuple(t.shape) == expected, (name, tuple(t.shape), expected)
        store.fill_tensor(name, t)


class Dsv4Mxfp4MarlinFill(ExpertFill):
    name = "dsv4-mxfp4-marlin"

    def matches(self, store, quant_method: str) -> bool:
        # DSV4 mxfp4 via Mxfp4MarlinMoEMethod: marlin-repacked weights (int32) + e8m0 w*_weight_scale,
        # no _scale_inv, no bias. The e8m0 scale dtype is the discriminator: nvfp4 uses e4m3/uint8
        # scales, ct-fp8-channel float32; gpt-oss mxfp4 has w13_weight_bias; Dsv4Fp4Fill has _scale_inv.
        g = store.gpu
        return (
            "w13_weight_scale" in g
            and "w13_weight_scale_inv" not in g
            and "w13_weight_bias" not in g
            and g["w13_weight_scale"].dtype == torch.float8_e8m0fnu
        )

    def fill(self, store, model_path, layer_idx, device):
        # Cache the repacked store like gptq-marlin: the DSV4 marlin repack is expensive (~seconds/layer),
        # deterministic, and independent of K, so persisting it turns later boots' read+repack into a
        # straight sequential read.
        from . import cache

        cache_dir = cache._store_cache_dir(model_path)
        if cache._fill_store_from_cache(store, cache_dir, layer_idx):
            if not cache._STORE_CACHE_LOGGED:
                cache._STORE_CACHE_LOGGED = True
                logger.info(
                    "[paged-experts] host store loading from the repack cache (%s)", cache_dir
                )
        else:
            _fill_dsv4_mxfp4_marlin_from_checkpoint(store, model_path, layer_idx)
            cache._save_store_to_cache(store, cache_dir, layer_idx)
        return None
