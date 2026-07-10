"""Transposed-plain-int4 streaming store for Paged Experts compute-through.

The compute-through overflow path streams an expert's int4 weights straight from a pinned host mirror
into the fused streaming FFN kernel (``jit_kernel/stream_moe.py``) instead of paging them into K GPU
slots. That kernel is a warp-per-output-row GEMV, so it wants the weights **output-major** — but the GPTQ
checkpoint packs ``qweight`` as ``[in/pack, out]`` (along the reduction dim), which strides a per-output
read. This module repacks the checkpoint into the coalesced streaming layout:

    qweight [E, out, in/pack] int32  (packed low-nibble-first along ``in``, row-per-output)
    scales  [E, out, in/group] fp16

for ``w13 = concat(gate, up)`` (out = 2*inter, in = hidden) and ``w2 = down`` (out = hidden, in = inter).
sym GPTQ (``desc_act=False``, qzeros all == 2^(bits-1)-1) => effective zero 8 => dequant ``(q-8)*scale``,
no qzeros carried. The repack is a faithful rearrangement: dequant(transposed) == dequant(raw) exactly
(validated bit-for-bit, ``probe/kernels/plain_store_fill.py``). This is the streaming sibling of
``pager._fill_gptq_marlin_from_checkpoint`` (which produces the marlin-permuted resident layout); the two
stores coexist (dual-store: marlin for resident/keep-warm decode, plain for streamed overflow).
"""

from __future__ import annotations

import json
import os
from typing import Dict

import torch

_PLAIN_TENSOR_NAMES = ("w13_qweight", "w13_scales", "w2_qweight", "w2_scales")


def repack_transpose(qw_raw: torch.Tensor, s_raw: torch.Tensor, group: int, pack: int):
    """GPTQ ``[in/pack, out]`` (packed along ``in``) + scales ``[in/group, out]`` -> output-major streaming
    layout ``qw_T [out, in/pack]`` + ``s_T [out, in/group]``. int4 only (``pack == 8``); the nibble width is
    ``32 // pack`` bits. Dequant-identical to the raw layout (a pure rearrangement)."""
    assert pack == 8, f"transposed-plain streaming store is int4-only (pack=8), got pack={pack}"
    nib = 32 // pack  # 4-bit nibbles
    INp, OUT = qw_raw.shape
    IN = INp * pack
    q = torch.zeros(IN, OUT, dtype=torch.int64, device=qw_raw.device)
    for b in range(pack):
        q[b::pack] = (qw_raw.to(torch.int64) >> (nib * b)) & 0xF  # unpack raw -> q[in, out]
    qT = q.t().contiguous()  # [out, in]
    qw_T = torch.zeros(OUT, IN // pack, dtype=torch.int64, device=qw_raw.device)
    for b in range(pack):
        qw_T |= qT[:, b::pack] << (nib * b)  # repack output-major
    return qw_T.to(torch.int32).contiguous(), s_raw.t().contiguous()  # s_T [out, in/group]


def fill_transposed_plain_from_checkpoint(
    model_path: str, layer_idx: int, num_experts: int, device
) -> Dict[str, torch.Tensor]:
    """Build the transposed-plain streaming tensors for ``layer_idx``, all ``num_experts`` experts, from the
    GPTQ checkpoint. Returns a dict of **CPU** tensors (caller pins + resolves UVA devptrs):

        w13_qweight [E, 2*inter, hidden/pack] int32   w13_scales [E, 2*inter, hidden/group] fp16
        w2_qweight  [E, hidden, inter/pack]  int32     w2_scales  [E, hidden, inter/group] fp16

    Mirrors ``pager._fill_gptq_marlin_from_checkpoint``'s checkpoint reading (concat gate+up along out).
    """
    from contextlib import ExitStack

    from safetensors import safe_open

    from sglang.srt.layers.moe.paged_experts.pager import (
        _experts_prefix,
        _snapshot_dir,
        _weight_map,
    )

    snap = _snapshot_dir(model_path)
    cfg = json.load(open(os.path.join(snap, "config.json")))
    tcfg = cfg.get("text_config", cfg)
    qc = cfg["quantization_config"]
    bits, group = qc["bits"], qc["group_size"]
    pack = 32 // bits
    assert bits == 4, f"transposed-plain streaming store is int4-only, got bits={bits}"
    assert not qc.get("desc_act", False), "desc_act=True needs g_idx paging (unsupported)"
    wmap = _weight_map(snap)
    pre = _experts_prefix(wmap, layer_idx)

    stack = ExitStack()
    open_shards: Dict[str, object] = {}

    def get(name: str) -> torch.Tensor:
        sh = wmap[name]
        if sh not in open_shards:
            open_shards[sh] = stack.enter_context(
                safe_open(os.path.join(snap, sh), framework="pt")
            )
        return open_shards[sh].get_tensor(name)

    cols = {n: [] for n in _PLAIN_TENSOR_NAMES}
    for e in range(num_experts):
        p = f"{pre}{e}."
        w13_qw = torch.cat([get(p + "gate_proj.qweight"), get(p + "up_proj.qweight")], dim=1)
        w13_s = torch.cat([get(p + "gate_proj.scales"), get(p + "up_proj.scales")], dim=1)
        w2_qw = get(p + "down_proj.qweight")
        w2_s = get(p + "down_proj.scales")
        w13_qw_T, w13_s_T = repack_transpose(w13_qw.to(device), w13_s.to(device), group, pack)
        w2_qw_T, w2_s_T = repack_transpose(w2_qw.to(device), w2_s.to(device), group, pack)
        cols["w13_qweight"].append(w13_qw_T)
        cols["w13_scales"].append(w13_s_T.half())
        cols["w2_qweight"].append(w2_qw_T)
        cols["w2_scales"].append(w2_s_T.half())
    stack.close()
    return {n: torch.stack(cols[n]).contiguous().cpu() for n in _PLAIN_TENSOR_NAMES}
