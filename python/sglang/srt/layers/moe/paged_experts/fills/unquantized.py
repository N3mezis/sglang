"""Unquantized (bf16/fp16) expert fill."""

import os  # noqa: E402  (used by the verbatim body below)
from typing import Dict


from ..store import ExpertStore
from .base import ExpertFill
from .checkpoint import (
    _drop_file_cache,
    _experts_prefix,
    _proj_names,
    _snapshot_dir,
    _weight_map,
)


def _fill_bf16_from_checkpoint(
    store: ExpertStore, model_path: str, layer_idx: int
) -> None:
    """bf16: host w13_weight=[E,2*inter,hidden]=concat(gate,up), w2_weight=[E,hidden,inter]."""
    from safetensors import safe_open

    snap = _snapshot_dir(model_path)
    wmap = _weight_map(snap)
    pre = _experts_prefix(wmap, layer_idx)
    dt = store.gpu["w13_weight"].dtype
    # Gemma-4-style STACKED FUSED experts: one tensor per projection holding all E experts, gate+up
    # already fused. gate_up_proj [E, 2*inter, hidden] == w13, down_proj [E, hidden, inter] == w2 (same
    # orientation as the store), so copy each expert row straight across.
    if f"{pre}gate_up_proj" in wmap:
        for stem, dst in (("gate_up_proj", "w13_weight"), ("down_proj", "w2_weight")):
            with safe_open(
                os.path.join(snap, wmap[f"{pre}{stem}"]), framework="pt"
            ) as f:
                stacked = f.get_tensor(f"{pre}{stem}")  # [E, ...]
            for e in range(store.E):
                store.row(dst, e).copy_(stacked[e].to(dt))
        return
    gate, up, down = _proj_names(wmap, pre)  # (gate,up,down); Mixtral -> (w1,w3,w2)
    by_shard: Dict[str, list] = {}
    for e in range(store.E):
        for proj in (gate, up, down):
            by_shard.setdefault(wmap[f"{pre}{e}.{proj}.weight"], []).append((e, proj))
    for shard, items in by_shard.items():
        _shard_path = os.path.join(snap, shard)
        with safe_open(_shard_path, framework="pt") as f:
            for e, proj in items:
                t = f.get_tensor(f"{pre}{e}.{proj}.weight").to(dt)
                if proj == down:
                    store.row("w2_weight", e).copy_(t)
                    continue
                # w13 packs UP (first half of dim 0) then GATE (second half). The paged MoE GEMM applies
                # silu to the FIRST half, so gate must land in the SECOND half to yield the correct
                # silu(gate)*up (gate-first gives silu(up)*gate garbage — see the _fill_bf16 fix / the
                # in-place NVFP4 store's (up, gate, down) swap).
                row = store.row("w13_weight", e)
                half = row.shape[0] // 2
                if proj == gate:
                    row[half:].copy_(t)
                else:  # up
                    row[:half].copy_(t)
        _drop_file_cache(_shard_path)  # release this shard's page cache before the next


class Bf16Fill(ExpertFill):
    name = "bf16"
    _SCALE_KEYS = (
        "w13_weight_scale",
        "w13_weight_scale_inv",
        "w13_weight_bias",
        "w13_weight_packed",
    )

    def matches(self, store, quant_method: str) -> bool:
        g = store.gpu
        return (
            "w13_weight" in g
            and "w13_qweight" not in g
            and not any(k in g for k in self._SCALE_KEYS)
        )

    def fill(self, store, model_path, layer_idx, device):
        _fill_bf16_from_checkpoint(store, model_path, layer_idx)
        return None
