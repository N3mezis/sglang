"""FP8 expert fills: block-quant (weight_block_size) and compressed-tensors float-quantized
per-channel (dynamic act). Both page fp8 weights + scales as ordinary per-expert rows (no repack).
"""

import os
from typing import Dict

import torch

from ..store import ExpertStore
from .base import ExpertFill
from .checkpoint import (
    _drop_file_cache,
    _experts_prefix,
    _proj_names,
    _snapshot_dir,
    _weight_map,
)


def _fill_fp8_block_from_checkpoint(
    store: ExpertStore, model_path: str, layer_idx: int
) -> None:
    """fp8 block-quant: direct copy, like bf16 but with the per-block scales as paged tensors too.
    Host ``w13_weight=[E,2*inter,hidden]`` e4m3 (concat gate,up), ``w2_weight=[E,hidden,inter]``;
    ``w13_weight_scale_inv``/``w2_weight_scale_inv`` are the [E, ceil(rows/block), ceil(cols/block)]
    float32 block scales, concatenated the same way. The CUDA triton path applies no post-load
    transform (no repack); layouts that DO transform (deepgemm ue8m0, mxfp8) are rejected by the
    dtype assertions below.
    """
    from safetensors import safe_open

    assert store.gpu["w13_weight"].dtype == torch.float8_e4m3fn, (
        "fp8 fill expects e4m3fn weights",
        store.gpu["w13_weight"].dtype,
    )
    assert store.gpu["w13_weight_scale_inv"].dtype == torch.float32, (
        "fp8 fill expects float32 block scales (ue8m0/mxfp8 layouts unsupported)",
        store.gpu["w13_weight_scale_inv"].dtype,
    )
    snap = _snapshot_dir(model_path)
    wmap = _weight_map(snap)
    pre = _experts_prefix(wmap, layer_idx)
    gate, up, down = _proj_names(wmap, pre)  # (gate,up,down); Mixtral -> (w1,w3,w2)
    by_shard: Dict[str, list] = {}
    for e in range(store.E):
        for proj in (gate, up, down):
            for suffix in ("weight", "weight_scale_inv"):
                by_shard.setdefault(wmap[f"{pre}{e}.{proj}.{suffix}"], []).append(
                    (e, proj, suffix)
                )
    for shard, items in by_shard.items():
        _shard_path = os.path.join(snap, shard)
        with safe_open(_shard_path, framework="pt") as f:
            for e, proj, suffix in items:
                t = f.get_tensor(f"{pre}{e}.{proj}.{suffix}")
                base = "w2_weight" if proj == down else "w13_weight"
                name = base if suffix == "weight" else base + "_scale_inv"
                row = store.row(name, e)
                if proj == down:
                    row.copy_(t)
                    continue
                # w13 packs gate (first half of dim 0) then up (second half); the block scales
                # follow the same row split, so the same halving works for both suffixes.
                half = row.shape[0] // 2
                if proj == gate:
                    row[:half].copy_(t)
                else:  # up
                    row[half:].copy_(t)
        _drop_file_cache(_shard_path)  # release this shard's page cache before the next


def _fill_ct_fp8_channel_from_checkpoint(
    store: ExpertStore, model_path: str, layer_idx: int
) -> None:
    """compressed-tensors fp8 float-quantized, per-channel (or per-tensor) weights with DYNAMIC
    activations (BCCard/…-FP8-Dynamic). Like the fp8-block fill but the scales are per-output-channel
    ``[out,1]`` (not block ``_scale_inv``): host ``w13_weight=[E,2*inter,hidden]`` e4m3 (concat
    gate,up) + ``w13_weight_scale=[E,2*inter,1]`` fp32, ``w2_weight``/``w2_weight_scale`` likewise.
    Weights are stored ``[out,in]`` — same orientation as the store — so no transpose; gate/up fuse on
    dim 0. No post-load transform for per-channel dynamic on CUDA, so a direct row copy matches the
    native load. Scales are bf16 in the checkpoint; the store buffer is fp32 (cast on copy).
    """
    from safetensors import safe_open

    assert store.gpu["w13_weight"].dtype == torch.float8_e4m3fn, (
        "ct fp8 fill expects e4m3fn weights",
        store.gpu["w13_weight"].dtype,
    )
    sdt = store.gpu["w13_weight_scale"].dtype  # fp32 buffer; checkpoint scale is bf16
    snap = _snapshot_dir(model_path)
    wmap = _weight_map(snap)
    pre = _experts_prefix(wmap, layer_idx)
    gate, up, down = _proj_names(wmap, pre)  # (gate,up,down); Mixtral -> (w1,w3,w2)
    by_shard: Dict[str, list] = {}
    for e in range(store.E):
        for proj in (gate, up, down):
            for suffix in ("weight", "weight_scale"):
                by_shard.setdefault(wmap[f"{pre}{e}.{proj}.{suffix}"], []).append(
                    (e, proj, suffix)
                )
    for shard, items in by_shard.items():
        _shard_path = os.path.join(snap, shard)
        with safe_open(_shard_path, framework="pt") as f:
            for e, proj, suffix in items:
                t = f.get_tensor(f"{pre}{e}.{proj}.{suffix}")
                base = "w2_weight" if proj == down else "w13_weight"
                name = base if suffix == "weight" else base + "_scale"
                if suffix == "weight_scale":
                    t = t.to(sdt)
                row = store.row(name, e)
                if proj == down:
                    row.copy_(t)
                    continue
                # w13 packs gate (first half of dim 0) then up (second half); the per-channel scales
                # follow the same row split, so the same halving works for both suffixes.
                half = row.shape[0] // 2
                if proj == gate:
                    row[:half].copy_(t)
                else:  # up
                    row[half:].copy_(t)
        _drop_file_cache(_shard_path)  # release this shard's page cache before the next


class Fp8BlockFill(ExpertFill):
    name = "fp8-block"

    def matches(self, store, quant_method: str) -> bool:
        return "w13_weight_scale_inv" in store.gpu

    def fill(self, store, model_path, layer_idx, device):
        _fill_fp8_block_from_checkpoint(store, model_path, layer_idx)
        return None


class CtFp8ChannelFill(ExpertFill):
    name = "ct-fp8-channel"

    def matches(self, store, quant_method: str) -> bool:
        g = store.gpu
        return (
            "w13_weight_scale" in g
            and "w13_weight" in g
            and g["w13_weight"].dtype == torch.float8_e4m3fn
        )

    def fill(self, store, model_path, layer_idx, device):
        _fill_ct_fp8_channel_from_checkpoint(store, model_path, layer_idx)
        return None
