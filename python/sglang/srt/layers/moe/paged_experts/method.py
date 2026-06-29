"""PagedExpertsMoEMethod: the K-slot resident expert table.

Wraps the real fused-MoE quant method (unquantized bf16 / gptq-marlin int4) with a K-of-E resident
table; routing stays E-wide (the model's gate is untouched), only the expert TABLE is K, and the
forward remaps logical expert ids -> resident slots per step, paging misses from the pinned host store.

Weight loading for the K residents reuses sglang's NATIVE expert-parallel remap: setting
``layer.num_local_experts = K`` makes the default loader fill slots ``0..K-1`` and skip the rest (no
custom loader). Forward (``apply``) and the host store live in ``forward.py`` / ``pager.py`` (imported
lazily so this module loads without them). K sizing is ``sizing.compute_num_resident_experts``.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from sglang.srt.layers.moe.paged_experts.guard import check_paged_experts_compat
from sglang.srt.layers.moe.paged_experts.sizing import (
    compute_num_resident_experts,
    kv_reserve_bytes_mha,
)

logger = logging.getLogger(__name__)

# Captured at import (BEFORE model weights load) so the resolver sizes K against sglang's PRE-load free
# memory P — the basis of sglang's own KV accounting (KV_pool = post_load_free - P*(1-mem_fraction)).
# Using total board memory over-counts by the CUDA-context overhead and over-sizes K into an OOM.
try:
    _PRE_LOAD_FREE_BYTES = torch.cuda.mem_get_info()[0]
except Exception:
    _PRE_LOAD_FREE_BYTES = 0


def resolve_num_resident_experts(
    num_experts_E: int,
    *,
    nonexpert_reserve_gb: float = 2.5,
) -> int:
    """Resolve K at create_weights from sglang's OWN already-derived config (no CLI re-parse): read
    ``mem_fraction_static`` / ``max_running_requests`` / ``context_length`` / ``kv_cache_dtype`` off
    ``get_global_server_args()`` and the arch off ``ModelConfig``, then call the pure sizing formula.
    Reading the SAME mem_fraction the server runs at keeps K and the KV pool coherent by construction.
    """
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import get_global_server_args

    sa = get_global_server_args()
    mc = ModelConfig.from_server_args(sa)
    htc = mc.hf_text_config
    layers = mc.num_hidden_layers
    fkdr = getattr(htc, "first_k_dense_replace", 0) or 0
    moe_layers = layers - fkdr

    # per-expert-per-layer bytes, estimated from config (gate+up+down at the quant bit-width, +~3% scales/zeros)
    bits = 16
    qc = getattr(htc, "quantization_config", None)
    if isinstance(qc, dict):
        bits = qc.get("bits") or qc.get("weights", {}).get("num_bits") or 16
    per_el = 3 * htc.moe_intermediate_size * htc.hidden_size * (bits / 8.0) * 1.03

    # KV headroom to reserve when sizing K. The K-slot pool is FIXED (it does not grow with concurrency);
    # sglang sizes the real KV pool from the post-weights leftover and derives max_running_requests from
    # THAT. So reserving the worst case (max_running_requests x full context) here double-counts and
    # starves K — a footgun on the constrained cards Paged Experts targets (a high --max-running-requests
    # silently floored K to top_k). Reserve a SINGLE-STREAM context by default; sglang's actual KV pool
    # (the leftover) then supports real concurrency. --paged-experts-kv-reserve-gb overrides to reserve a
    # larger guaranteed KV pool (smaller K). sizing.compute_num_resident_experts clamps it to physical.
    kv_elt = 1 if "fp8" in (sa.kv_cache_dtype or "").lower() else 2
    ctx = sa.context_length or getattr(mc, "context_len", None) or 2048
    kv_gb = getattr(sa, "paged_experts_kv_reserve_gb", -1.0)
    if kv_gb is not None and kv_gb >= 0:
        kv_reserve = kv_gb * 1e9
    elif getattr(mc, "kv_lora_rank", None):  # MLA
        cell = (mc.kv_lora_rank + mc.qk_rope_head_dim) * layers * kv_elt
        kv_reserve = ctx * cell  # single-stream
    else:  # MHA / GQA — reuse the pure helper (get_num_kv_heads handles GQA + TP)
        tp = getattr(sa, "tp_size", 1) or 1
        kv_reserve = kv_reserve_bytes_mha(
            num_layers=layers,
            num_kv_heads=mc.get_num_kv_heads(tp),
            head_dim=(mc.head_dim + mc.v_head_dim) // 2,  # combined K+V per-head width
            kv_dtype_bytes=kv_elt,
            max_running_requests=1,  # single-stream headroom, NOT worst-case concurrency
            context_length=ctx,
        )

    free = _PRE_LOAD_FREE_BYTES or torch.cuda.mem_get_info()[0]
    top_k = getattr(htc, "num_experts_per_tok", 8) or 8
    mem_frac = sa.mem_fraction_static or 0.85
    k = compute_num_resident_experts(
        free_vram_bytes=free,
        mem_fraction=mem_frac,
        nonexpert_bytes=nonexpert_reserve_gb * 1e9,
        kv_reserve_bytes=kv_reserve,
        moe_layers=moe_layers,
        per_expert_layer_bytes=per_el,
        top_k=top_k,
        num_experts=num_experts_E,
    )
    logger.info(
        "[paged-experts] resident K=%d/%d (%d%%): free=%.2fGB mem_fraction=%.3f "
        "KV_reserve=%.2fGB per_expert=%.2fMB moe_layers=%d",
        k,
        num_experts_E,
        k * 100 // num_experts_E,
        free / 1e9,
        mem_frac,
        kv_reserve / 1e9,
        per_el / 1e6,
        moe_layers,
    )
    return k


def _make_method_class():
    """Import the base lazily so this module can be imported before the rest of sglang's MoE stack."""
    from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase

    class PagedExpertsMoEMethod(FusedMoEMethodBase):
        def __init__(
            self,
            base_method,
            num_experts_E: int,
            num_resident_K: int,
            pin_host: bool = True,
            eviction: str = "lru",
            window: int = 0,
        ):
            self.base_method = base_method
            self.E = num_experts_E
            self.num_resident = num_resident_K
            self.pin_host = pin_host
            self.eviction = eviction
            # Pinned-window fallback: 0 = full pin (every expert page-locked); 0 < window < E pins only the
            # W hot experts and keeps the E-W cold tail pageable, for stores past the page-lock ceiling.
            self.window = window
            self._pager = None
            # Initial residents = experts 0..K-1 in slots 0..K-1; the pager re-seeds + pages the rest.
            self.logical_to_gpu_index = torch.full((self.E,), -1, dtype=torch.int32)
            self.logical_to_gpu_index[: self.num_resident] = torch.arange(
                self.num_resident, dtype=torch.int32
            )
            self.logical_to_gpu_index_cuda = None

        def create_weights(
            self,
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra,
        ):
            # K-slot table. Weight loading uses FusedMoE's NATIVE expert-parallel remap: num_local_experts
            # = K -> the default loader fills slots 0..K-1 and skips the rest (no custom loader). Our
            # forward does its OWN routing remap, so K-local only affects load.
            layer.num_local_experts = self.num_resident
            self.base_method.create_weights(
                layer=layer,
                num_experts=self.num_resident,
                hidden_size=hidden_size,
                intermediate_size_per_partition=intermediate_size_per_partition,
                params_dtype=params_dtype,
                **extra,
            )
            dev = next(layer.parameters()).device
            self.logical_to_gpu_index_cuda = self.logical_to_gpu_index.to(dev)

        def create_moe_runner(self, layer, moe_runner_config):
            from dataclasses import replace

            # The runner must size its expert loop to K, not the model's E local experts, else the
            # fused-MoE kernel indexes past the K slots. routed_scaling_factor is applied externally by
            # the model (deepseek_v2) -> strip it here to avoid double-scaling.
            cfg = replace(moe_runner_config, num_local_experts=self.num_resident)
            cfg = replace(cfg, routed_scaling_factor=None)
            self.base_method.create_moe_runner(layer, cfg)
            self.moe_runner_config = getattr(self.base_method, "moe_runner_config", cfg)

        def process_weights_after_loading(self, layer):
            if hasattr(self.base_method, "process_weights_after_loading"):
                self.base_method.process_weights_after_loading(layer)
            from sglang.srt.layers.moe.paged_experts.pager import setup_pager

            self._pager = setup_pager(self, layer)

        def apply(self, layer, dispatch_output):
            from sglang.srt.layers.moe.paged_experts.forward import paged_apply

            return paged_apply(self, layer, dispatch_output)

    return PagedExpertsMoEMethod


def make_for_layer(
    layer,
    base_method,
    server_args: Any,
    *,
    num_resident: Any = "auto",
    nonexpert_reserve_gb: float = 2.5,
    pin_host: Any = None,
) -> Any:
    """Factory invoked from the FusedMoE init hook when paged experts is enabled: enforce the
    compatibility guard, resolve K, and wrap ``base_method``. ``num_resident`` is ``"auto"`` or an int.
    ``pin_host`` defaults from ``--paged-experts-store`` (pinned -> True, paged -> False); the env entry
    overrides it. The eviction policy comes from ``--paged-experts-eviction`` (lru | lfu).
    """
    check_paged_experts_compat(server_args)
    E = int(getattr(layer, "num_local_experts", None) or layer.num_experts)
    if num_resident == "auto":
        K = resolve_num_resident_experts(E, nonexpert_reserve_gb=nonexpert_reserve_gb)
    else:
        K = int(num_resident)
    if pin_host is None:
        pin_host = getattr(server_args, "paged_experts_store", "pinned") != "paged"
    eviction = getattr(server_args, "paged_experts_eviction", "lru")
    window = _resolve_window_size(
        getattr(server_args, "paged_experts_window_size", "0"), E, pin_host=bool(pin_host)
    )
    return _make_method_class()(
        base_method, E, K, pin_host=bool(pin_host), eviction=eviction, window=window
    )


def _resolve_window_size(raw: Any, num_experts_E: int, *, pin_host: bool) -> int:
    """Resolve ``--paged-experts-window-size`` to a pinned-window expert count W (0 = full pin / off).

    ``0`` | ``off`` | ``none`` -> 0 (every expert page-locked, today's behaviour). An integer pins that
    many experts and keeps the cold tail pageable. ``auto`` (greedy-pin-until-fail) is not yet wired — it
    falls back to full pin with a warning. The window only applies to the pinned store (it *is* the
    page-lock fallback), so it is ignored when ``--paged-experts-store paged``.
    """
    if raw is None:
        return 0
    s = str(raw).strip().lower()
    if s in ("", "0", "off", "none"):
        return 0
    if not pin_host:
        logger.warning(
            "[paged-experts] --paged-experts-window-size is ignored with --paged-experts-store paged "
            "(the whole store is already pageable)."
        )
        return 0
    if s == "auto":
        logger.warning(
            "[paged-experts] --paged-experts-window-size=auto (greedy-pin-until-fail) is not yet wired; "
            "specify an integer expert count to pin. Falling back to full pin (window off)."
        )
        return 0
    w = int(s)
    if w <= 0 or w >= num_experts_E:
        return 0  # <=0 off; >=E is a full pin -> the plain pinned store, no window needed
    return w


def make_for_layer_from_env(layer, base_method):
    """Env-driven entry used by the boot/test hook (reads ``KT_*`` envs + the global ServerArgs). The
    upstream contribution uses ``make_for_layer`` with first-class ``ServerArgs`` flags instead.
    ``KT_PIN_HOST=0`` selects the pageable store (== ``--paged-experts-store paged``).
    """
    import os

    from sglang.srt.server_args import get_global_server_args

    return make_for_layer(
        layer,
        base_method,
        get_global_server_args(),
        num_resident=os.environ.get("KT_NUM_GPU_EXPERTS", "auto"),
        pin_host=os.environ.get("KT_PIN_HOST", "1") == "1",
    )
