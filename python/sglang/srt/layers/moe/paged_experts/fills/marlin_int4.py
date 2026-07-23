"""INT4/INT8 marlin + wna16 expert fills — the largest quant family: gptq-marlin (incl. desc_act
g_idx paging), compressed-tensors pack-quantized (Hub "AWQ-4bit"/"w4a16"), classic AWQ (asymmetric
zero-points), and moe_wna16 uint8 triton (e.g. Intel AutoRound auto_gptq on SM120). gptq and awq
register identical store keys, so the registry splits them by the checkpoint quant_method; AutoRound
lands on the uint8 wna16 layout, split by the qweight dtype."""

import json
import logging
import os  # noqa: E402  (used by verbatim bodies)
from typing import Dict, Optional

import torch

from ..store import ExpertStore
from .base import ExpertFill
from .checkpoint import (
    _experts_prefix,
    _proj_names,
    _snapshot_dir,
    _weight_map,
)

logger = logging.getLogger(__name__)


def _fill_gptq_marlin_from_checkpoint(
    store: ExpertStore, model_path: str, layer_idx: int
) -> None:
    """gptq-int4: repack the GPTQ checkpoint into the on-GPU marlin layout for ALL E experts, using
    sglang's own ops, straight into the host store. sglang's loader repacks only the K resident slots
    (num_local_experts=K); we repack all E so the paged experts match. This is the per-layer repack the
    offline builder did, moved to load time -> no offline store artifact needed. (At runtime the
    quantization package is already imported, so the gptq_kernels/wNa16 circular import doesn't apply.)
    """
    from safetensors import safe_open

    # Load the quantization package fully before importing gptq_kernels directly — gptq_kernels and
    # compressed_tensors_wNa16_moe form an import cycle that only fails when gptq_kernels is the entry
    # point. At server runtime it is already imported; this makes the order-independent too.
    import sglang.srt.layers.quantization  # noqa: F401
    from sglang.srt.hardware_backend.gpu.quantization.gptq_kernels import (
        gptq_marlin_moe_repack,
    )
    from sglang.srt.layers.quantization.marlin_utils import marlin_moe_permute_scales

    snap = _snapshot_dir(model_path)
    cfg = json.load(open(os.path.join(snap, "config.json")))
    tcfg = cfg.get("text_config", cfg)
    inter = tcfg.get("moe_intermediate_size") or tcfg["intermediate_size"]
    qc = cfg["quantization_config"]
    bits, group = qc["bits"], qc["group_size"]
    pack = 32 // bits
    # desc_act (act-order): the checkpoint stores a per-expert g_idx (input-channel -> group map). The
    # marlin repack permutes qweight rows by argsort(g_idx), and the kernel reads the sorted g_idx +
    # sort_indices at gather time. We compute both for ALL E and page them into slots (they are live
    # per-expert tensors — see store._NONPAGED_SUFFIXES). desc_act with group_size==-1 degenerates to
    # no act-order (the config loader forces desc_act=False there), so it can't reach here.
    desc_act = bool(qc.get("desc_act", False)) and group != -1
    wmap = _weight_map(snap)
    pre = _experts_prefix(wmap, layer_idx)
    gate, up, down = _proj_names(wmap, pre)  # (gate,up,down); Mixtral -> (w1,w3,w2)
    dev = store.device

    from contextlib import ExitStack

    _shard_stack = ExitStack()
    open_shards: Dict[str, object] = {}

    def get(name: str) -> torch.Tensor:
        sh = wmap[name]
        if sh not in open_shards:
            open_shards[sh] = _shard_stack.enter_context(
                safe_open(os.path.join(snap, sh), framework="pt")
            )
        return open_shards[sh].get_tensor(name)

    w13_qw, w2_qw, w13_s, w2_s, w13_qz, w2_qz = [], [], [], [], [], []
    w13_gi, w2_gi = [], []
    for e in range(store.E):
        p = f"{pre}{e}."
        w13_qw.append(
            torch.cat([get(f"{p}{gate}.qweight"), get(f"{p}{up}.qweight")], dim=1)
        )
        w2_qw.append(get(f"{p}{down}.qweight"))
        w13_s.append(
            torch.cat([get(f"{p}{gate}.scales"), get(f"{p}{up}.scales")], dim=1)
        )
        w2_s.append(get(f"{p}{down}.scales"))
        w13_qz.append(
            torch.cat([get(f"{p}{gate}.qzeros"), get(f"{p}{up}.qzeros")], dim=1)
        )
        w2_qz.append(get(f"{p}{down}.qzeros"))
        if desc_act:
            # gate & up share the hidden input, so their g_idx is identical; w13 uses that single
            # g_idx, w2 uses down's. (Verified equal on the checkpoint; assert to be safe.)
            g_gate = get(f"{p}{gate}.g_idx")
            assert torch.equal(
                g_gate, get(f"{p}{up}.g_idx")
            ), "[paged-experts] gate/up g_idx differ; fused w13 act-order unsupported"
            w13_gi.append(g_gate)
            w2_gi.append(get(f"{p}{down}.g_idx"))
    _shard_stack.close()  # release shard handles before the (GPU) repack
    w13_qw, w2_qw = torch.stack(w13_qw).to(dev), torch.stack(w2_qw).to(dev)
    w13_s, w2_s = torch.stack(w13_s).to(dev), torch.stack(w2_s).to(dev)
    if desc_act:
        # per-expert argsort(g_idx) -> sort_indices (perm for the repack + kernel) and sorted g_idx.
        w13_gi = torch.stack(w13_gi).to(dev)
        w2_gi = torch.stack(w2_gi).to(dev)
        w13_sort = torch.argsort(w13_gi, dim=1).to(torch.int32)
        w2_sort = torch.argsort(w2_gi, dim=1).to(torch.int32)
        w13_gsorted = torch.gather(w13_gi, 1, w13_sort.to(torch.int64))
        w2_gsorted = torch.gather(w2_gi, 1, w2_sort.to(torch.int64))
    else:
        w13_sort = w2_sort = torch.empty((store.E, 0), dtype=torch.int32, device=dev)
    marlin = {
        "w13_qweight": gptq_marlin_moe_repack(
            w13_qw, w13_sort, w13_qw.shape[1] * pack, w13_qw.shape[2], bits
        ),
        "w2_qweight": gptq_marlin_moe_repack(
            w2_qw, w2_sort, w2_qw.shape[1] * pack, w2_qw.shape[2], bits
        ),
        "w13_scales": marlin_moe_permute_scales(
            s=w13_s, size_k=inter, size_n=w13_s.shape[2], group_size=group
        ),
        "w2_scales": marlin_moe_permute_scales(
            s=w2_s, size_k=w2_s.shape[1] * group, size_n=w2_s.shape[2], group_size=group
        ),
        "w13_qzeros": torch.stack(w13_qz),  # carried unrepacked (sym); kernel ignores
        "w2_qzeros": torch.stack(w2_qz),
    }
    if desc_act:
        marlin["w13_g_idx"] = w13_gsorted
        marlin["w2_g_idx"] = w2_gsorted
        marlin["w13_g_idx_sort_indices"] = w13_sort
        marlin["w2_g_idx_sort_indices"] = w2_sort
    for name in store.gpu:
        t = marlin[name].contiguous().cpu()
        expected = (store.E, *store.gpu[name].shape[1:])
        assert tuple(t.shape) == expected, (name, t.shape, expected)
        store.fill_tensor(name, t)


def _fill_ct_wna16_from_checkpoint(
    store: ExpertStore, model_path: str, layer_idx: int
) -> None:
    """compressed-tensors int pack-quantized (the checkpoints the Hub labels "AWQ-4bit"/"w4a16" —
    which are compressed-tensors, not classic autoawq). The packed layout is bit-compatible with
    GPTQ, so this is the gptq-marlin fill with the compressed-tensors tensor names: per-projection
    ``.weight_packed`` (int32, gptq-layout) + ``.weight_scale`` (fp16 group scales), symmetric
    (no zero-points), group-wise, no act-order. Repack ALL E via sglang's own marlin ops into the
    store keys ``w13_weight_packed``/``w2_weight_packed`` + ``w13_weight_scale``/``w2_weight_scale``
    (exactly the paged params discover_paged_params keeps for this method)."""
    from safetensors import safe_open

    import sglang.srt.layers.quantization  # noqa: F401  (break the gptq_kernels import cycle)
    from sglang.srt.hardware_backend.gpu.quantization.gptq_kernels import (
        gptq_marlin_moe_repack,
    )
    from sglang.srt.layers.quantization.marlin_utils import marlin_moe_permute_scales

    snap = _snapshot_dir(model_path)
    cfg = json.load(open(os.path.join(snap, "config.json")))
    tcfg = cfg.get("text_config", cfg)
    inter = tcfg.get("moe_intermediate_size") or tcfg["intermediate_size"]
    qc = cfg["quantization_config"]
    w = next(iter((qc.get("config_groups") or {}).values()), {}).get("weights", {})
    bits = w["num_bits"]
    group = w.get("group_size") or -1
    pack = 32 // bits
    if w.get("actorder") not in (None, "", "static", "weight"):
        raise RuntimeError(
            "[paged-experts] compressed-tensors runtime act-order (group/dynamic) needs g_idx "
            "paging, unsupported; baked-in 'weight'/'static' act-order is fine."
        )
    if not w.get("symmetric", True):
        raise RuntimeError(
            "[paged-experts] compressed-tensors asymmetric (zero-point) pack-quantized unsupported."
        )
    wmap = _weight_map(snap)
    pre = _experts_prefix(wmap, layer_idx)
    gate, up, down = _proj_names(wmap, pre)  # (gate,up,down); Mixtral -> (w1,w3,w2)
    dev = store.device

    from contextlib import ExitStack

    _shard_stack = ExitStack()
    open_shards: Dict[str, object] = {}

    def get(name: str) -> torch.Tensor:
        sh = wmap[name]
        if sh not in open_shards:
            open_shards[sh] = _shard_stack.enter_context(
                safe_open(os.path.join(snap, sh), framework="pt")
            )
        return open_shards[sh].get_tensor(name)

    # compressed-tensors stores weight_packed as [out, in//pack] and weight_scale as [out, in//group]
    # (is_transposed=True in create_weights; the native weight_loader transposes on load). GPTQ stores
    # [in//pack, out]/[in//group, out] directly, and gptq_marlin_moe_repack expects [in//pack, out].
    # So transpose each ct tensor to the gptq orientation, then fuse gate/up on the output dim (dim=1).
    def _t(t):
        return t.t().contiguous()

    w13_pk, w2_pk, w13_s, w2_s = [], [], [], []
    for e in range(store.E):
        p = f"{pre}{e}."
        w13_pk.append(
            torch.cat(
                [
                    _t(get(f"{p}{gate}.weight_packed")),
                    _t(get(f"{p}{up}.weight_packed")),
                ],
                dim=1,
            )
        )
        w2_pk.append(_t(get(f"{p}{down}.weight_packed")))
        w13_s.append(
            torch.cat(
                [_t(get(f"{p}{gate}.weight_scale")), _t(get(f"{p}{up}.weight_scale"))],
                dim=1,
            )
        )
        w2_s.append(_t(get(f"{p}{down}.weight_scale")))
    _shard_stack.close()  # release shard handles before the (GPU) repack
    w13_pk, w2_pk = torch.stack(w13_pk).to(dev), torch.stack(w2_pk).to(dev)
    w13_s, w2_s = torch.stack(w13_s).to(dev), torch.stack(w2_s).to(dev)
    sort = torch.empty((store.E, 0), dtype=torch.int32, device=dev)
    marlin = {
        "w13_weight_packed": gptq_marlin_moe_repack(
            w13_pk, sort, w13_pk.shape[1] * pack, w13_pk.shape[2], bits
        ),
        "w2_weight_packed": gptq_marlin_moe_repack(
            w2_pk, sort, w2_pk.shape[1] * pack, w2_pk.shape[2], bits
        ),
        "w13_weight_scale": marlin_moe_permute_scales(
            s=w13_s, size_k=inter, size_n=w13_s.shape[2], group_size=group
        ),
        "w2_weight_scale": marlin_moe_permute_scales(
            s=w2_s, size_k=w2_s.shape[1] * group, size_n=w2_s.shape[2], group_size=group
        ),
    }
    for name in store.gpu:
        t = marlin[name].contiguous().cpu()
        expected = (store.E, *store.gpu[name].shape[1:])
        assert tuple(t.shape) == expected, (name, t.shape, expected)
        store.fill_tensor(name, t)


def _fill_awq_marlin_from_checkpoint(
    store: ExpertStore, model_path: str, layer_idx: int
) -> None:
    """classic AWQ (quant_method='awq', asymmetric: real per-group zero-points). The store keys
    (w13_qweight/scales/qzeros) collide with gptq-marlin, so setup_pager routes here by the config's
    quant_method. AWQ packs along the OUTPUT dim: checkpoint qweight [in, out//pack], scales
    [in//group, out], qzeros [in//group, out//pack]; gate/up fuse on the output dim (dim=1, no
    transpose). Repack ALL E via sglang's own awq marlin ops (awq_marlin_moe_repack +
    marlin_moe_permute_scales + moe_awq_to_marlin_zero_points), mirroring the awq MoE kernel's
    process_weights_after_loading exactly."""
    from safetensors import safe_open

    import sglang.srt.layers.quantization  # noqa: F401
    from sglang.srt.hardware_backend.gpu.quantization.awq_kernels import (
        awq_marlin_moe_repack,
    )
    from sglang.srt.layers.quantization.marlin_utils import (
        marlin_moe_permute_scales,
        moe_awq_to_marlin_zero_points,
    )

    snap = _snapshot_dir(model_path)
    cfg = json.load(open(os.path.join(snap, "config.json")))
    tcfg = cfg.get("text_config", cfg)
    inter = tcfg.get("moe_intermediate_size") or tcfg["intermediate_size"]
    qc = cfg["quantization_config"]
    bits, group = qc["bits"], qc["group_size"]
    pack = 32 // bits
    wmap = _weight_map(snap)
    pre = _experts_prefix(wmap, layer_idx)
    gate, up, down = _proj_names(wmap, pre)  # (gate,up,down); Mixtral -> (w1,w3,w2)
    dev = store.device

    from contextlib import ExitStack

    _shard_stack = ExitStack()
    open_shards: Dict[str, object] = {}

    def get(name: str) -> torch.Tensor:
        sh = wmap[name]
        if sh not in open_shards:
            open_shards[sh] = _shard_stack.enter_context(
                safe_open(os.path.join(snap, sh), framework="pt")
            )
        return open_shards[sh].get_tensor(name)

    w13_qw, w2_qw, w13_s, w2_s, w13_qz, w2_qz = [], [], [], [], [], []
    for e in range(store.E):
        p = f"{pre}{e}."
        # AWQ packs along output -> gate/up fuse on dim=1 for all three tensors (no transpose).
        w13_qw.append(
            torch.cat([get(f"{p}{gate}.qweight"), get(f"{p}{up}.qweight")], dim=1)
        )
        w2_qw.append(get(f"{p}{down}.qweight"))
        w13_s.append(
            torch.cat([get(f"{p}{gate}.scales"), get(f"{p}{up}.scales")], dim=1)
        )
        w2_s.append(get(f"{p}{down}.scales"))
        w13_qz.append(
            torch.cat([get(f"{p}{gate}.qzeros"), get(f"{p}{up}.qzeros")], dim=1)
        )
        w2_qz.append(get(f"{p}{down}.qzeros"))
    _shard_stack.close()  # release shard handles before the (GPU) repack
    w13_qw, w2_qw = torch.stack(w13_qw).to(dev), torch.stack(w2_qw).to(dev)
    w13_s, w2_s = torch.stack(w13_s).to(dev), torch.stack(w2_s).to(dev)
    w13_qz, w2_qz = torch.stack(w13_qz).to(dev), torch.stack(w2_qz).to(dev)
    sort = torch.empty((store.E, 0), dtype=torch.int32, device=dev)
    marlin = {
        "w13_qweight": awq_marlin_moe_repack(
            w13_qw,
            sort,
            size_k=w13_qw.shape[1],
            size_n=w13_qw.shape[2] * pack,
            num_bits=bits,
        ),
        "w2_qweight": awq_marlin_moe_repack(
            w2_qw,
            sort,
            size_k=w2_qw.shape[1],
            size_n=w2_qw.shape[2] * pack,
            num_bits=bits,
        ),
        "w13_scales": marlin_moe_permute_scales(
            s=w13_s, size_k=inter, size_n=w13_s.shape[2], group_size=group
        ),
        "w2_scales": marlin_moe_permute_scales(
            s=w2_s, size_k=inter, size_n=w2_s.shape[2], group_size=group
        ),
        "w13_qzeros": moe_awq_to_marlin_zero_points(
            w13_qz, size_k=w13_qz.shape[1], size_n=w13_qz.shape[2] * pack, num_bits=bits
        ),
        "w2_qzeros": moe_awq_to_marlin_zero_points(
            w2_qz, size_k=w2_qz.shape[1], size_n=w2_qz.shape[2] * pack, num_bits=bits
        ),
    }
    for name in store.gpu:
        t = marlin[name].contiguous().to(store.gpu[name].dtype).cpu()
        expected = (store.E, *store.gpu[name].shape[1:])
        assert tuple(t.shape) == expected, (name, t.shape, expected)
        store.fill_tensor(name, t)


def _fill_moe_wna16_from_checkpoint(
    store: ExpertStore, model_path: str, layer_idx: int
) -> None:
    """moe_wna16 uint8 triton path (Intel AutoRound auto_gptq lands here on SM120: sglang's AutoRound
    MoE gates marlin off via check_moe_marlin_supports_layer). The store holds uint8-packed weights
    [E, 2*inter, hidden//2] and per-group fp scales — NOT the marlin int32 layout. This replicates
    MoeWNA16Method's gptq weight_loader conversion (no process_weights_after_loading, so the loaded
    state IS the final state): qweight = gptq_int32[in//8, out].T.view(uint8) -> [out, in//2];
    scales = gptq_scales[in//group, out].T -> [out, in//group]. Symmetric only (has_zp=False -> no
    qzeros paged; group divides hidden/inter here so no group_size_div_factor rescale).
    """
    from safetensors import safe_open

    snap = _snapshot_dir(model_path)
    cfg = json.load(open(os.path.join(snap, "config.json")))
    tcfg = cfg.get("text_config", cfg)
    inter = tcfg.get("moe_intermediate_size") or tcfg["intermediate_size"]
    hidden = tcfg["hidden_size"]
    group = cfg["quantization_config"]["group_size"]
    assert (
        inter % group == 0 and hidden % group == 0
    ), "[paged-experts] moe_wna16 fill assumes group_size divides hidden/inter (no rescale)"
    wmap = _weight_map(snap)
    pre = _experts_prefix(wmap, layer_idx)
    gate, up, down = _proj_names(wmap, pre)  # (gate,up,down); Mixtral -> (w1,w3,w2)
    sdt = store.gpu["w13_scales"].dtype

    from contextlib import ExitStack

    _shard_stack = ExitStack()
    open_shards: Dict[str, object] = {}

    def get(name: str) -> torch.Tensor:
        sh = wmap[name]
        if sh not in open_shards:
            open_shards[sh] = _shard_stack.enter_context(
                safe_open(os.path.join(snap, sh), framework="pt")
            )
        return open_shards[sh].get_tensor(name)

    def qw(name):  # gptq int32 [in//8, out] -> uint8 [out, in//2]
        return get(name).t().contiguous().view(torch.uint8)

    for e in range(store.E):
        p = f"{pre}{e}."
        # w13_qweight [2*inter, hidden//2]: gate -> first half of dim 0, up -> second half.
        rw = store.row("w13_qweight", e)
        h = rw.shape[0] // 2
        rw[:h].copy_(qw(f"{p}{gate}.qweight"))
        rw[h:].copy_(qw(f"{p}{up}.qweight"))
        store.row("w2_qweight", e).copy_(qw(f"{p}{down}.qweight"))
        # scales [2*inter, hidden//group]: gptq [in//group, out].T -> [out, in//group].
        rs = store.row("w13_scales", e)
        rs[:h].copy_(get(f"{p}{gate}.scales").t().contiguous().to(sdt))
        rs[h:].copy_(get(f"{p}{up}.scales").t().contiguous().to(sdt))
        store.row("w2_scales", e).copy_(
            get(f"{p}{down}.scales").t().contiguous().to(sdt)
        )
    _shard_stack.close()


class GptqMarlinFill(ExpertFill):
    name = "gptq-marlin"

    def matches(self, store, quant_method: str) -> bool:
        g = store.gpu
        return (
            "w13_qweight" in g
            and g["w13_qweight"].dtype == torch.int32
            and quant_method != "awq"
        )

    def fill(self, store, model_path, layer_idx, device):
        from . import (
            cache,  # gptq is the only cached fill (deterministic marlin repack)
        )

        cache_dir = cache._store_cache_dir(model_path)
        if cache._fill_store_from_cache(store, cache_dir, layer_idx):
            if not cache._STORE_CACHE_LOGGED:
                cache._STORE_CACHE_LOGGED = True
                logger.info(
                    "[paged-experts] host store loading from the repack cache (%s)",
                    cache_dir,
                )
        else:
            _fill_gptq_marlin_from_checkpoint(store, model_path, layer_idx)
            cache._save_store_to_cache(store, cache_dir, layer_idx)
        return None


class AwqMarlinFill(ExpertFill):
    name = "awq-marlin"

    def matches(self, store, quant_method: str) -> bool:
        g = store.gpu
        return (
            "w13_qweight" in g
            and g["w13_qweight"].dtype == torch.int32
            and quant_method == "awq"
        )

    def fill(self, store, model_path, layer_idx, device):
        _fill_awq_marlin_from_checkpoint(store, model_path, layer_idx)
        return None


class MoeWna16Fill(ExpertFill):
    name = "moe-wna16"

    def matches(self, store, quant_method: str) -> bool:
        g = store.gpu
        return "w13_qweight" in g and g["w13_qweight"].dtype == torch.uint8

    def fill(self, store, model_path, layer_idx, device):
        _fill_moe_wna16_from_checkpoint(store, model_path, layer_idx)
        return None


class CtWna16Fill(ExpertFill):
    name = "ct-int-pack-quantized"

    def matches(self, store, quant_method: str) -> bool:
        return "w13_weight_packed" in store.gpu

    def fill(self, store, model_path, layer_idx, device):
        _fill_ct_wna16_from_checkpoint(store, model_path, layer_idx)
        return None
