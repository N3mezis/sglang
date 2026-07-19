"""Unit tests for srt/layers/moe/paged_experts/guard.py"""

import unittest
from types import SimpleNamespace

from sglang.srt.layers.moe.paged_experts.guard import (
    check_paged_experts_compat,
    check_paged_experts_quant,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _sa(**overrides):
    base = dict(
        tp_size=1,
        ep_size=1,
        pp_size=1,
        dp_size=1,
        moe_a2a_backend="none",
        enable_eplb=False,
        load_format="auto",
        paged_experts_store="pinned",
        paged_experts_cold_backing="ram",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestPagedExpertsGuard(CustomTestCase):
    def test_clean_config_passes(self):
        check_paged_experts_compat(_sa())  # must not raise

    def test_rejects_incompatible_placement(self):
        # single-GPU first cut: any multi-device parallelism / placement is rejected
        for overrides, fragment in [
            (dict(tp_size=2), "tensor parallelism"),
            (dict(ep_size=2), "expert parallelism"),
            (dict(pp_size=2), "pipeline parallelism"),
            (dict(dp_size=2), "data parallelism"),
            (dict(enable_eplb=True), "EPLB"),
            (dict(moe_a2a_backend="deepep"), "all-to-all"),
            (dict(load_format="dummy"), "dummy"),
        ]:
            with self.assertRaises(RuntimeError) as cm:
                check_paged_experts_compat(_sa(**overrides))
            self.assertIn(fragment, str(cm.exception))

    def test_disk_cold_backing_passes(self):
        # the window is sized (and freq-ranked) automatically, so a disk cold tier is a plain, coherent
        # choice — no window flags to be incoherent with
        check_paged_experts_compat(
            _sa(paged_experts_cold_backing="disk")
        )  # must not raise

    def test_aggregates_multiple_problems(self):
        with self.assertRaises(RuntimeError) as cm:
            check_paged_experts_compat(
                _sa(ep_size=2, enable_eplb=True, load_format="dummy")
            )
        self.assertEqual(str(cm.exception).count("\n  - "), 3)

    def _quant(self, **qc):
        check_paged_experts_quant(SimpleNamespace(quantization_config=qc or None))

    def _ct_group(self, **weights):
        # a compressed-tensors config_groups blob with one group's weight-quant args
        return {"group_0": {"weights": weights}}

    def test_quant_guard_accepts(self):
        # unquantized
        self._quant()
        # gptq marlin int4, incl. runtime act-order (desc_act -> g_idx paging)
        self._quant(quant_method="gptq", bits=4)
        self._quant(quant_method="gptq", bits=4, group_size=128, desc_act=True)
        # fp8 BLOCK quant
        self._quant(quant_method="fp8", weight_block_size=[128, 128])
        # mxfp4 (gpt-oss)
        self._quant(quant_method="mxfp4")
        # compressed-tensors: nvfp4
        self._quant(quant_method="compressed-tensors", format="nvfp4-pack-quantized")
        # compressed-tensors int pack-quantized (sym), no / baked-in act-order, 4 and 8 bit
        for ao in (None, "static", "weight"):
            self._quant(
                quant_method="compressed-tensors",
                format="pack-quantized",
                config_groups=self._ct_group(
                    type="int", num_bits=4, symmetric=True, actorder=ao
                ),
            )
        self._quant(
            quant_method="compressed-tensors",
            format="pack-quantized",
            config_groups=self._ct_group(type="int", num_bits=8, symmetric=True),
        )
        # compressed-tensors fp8 float-quantized, per-channel + DYNAMIC activations
        self._quant(
            quant_method="compressed-tensors",
            format="float-quantized",
            config_groups={
                "group_0": {
                    "weights": {"type": "float", "num_bits": 8, "strategy": "channel"},
                    "input_activations": {"dynamic": True},
                }
            },
        )
        # classic AWQ (asymmetric) + AutoRound (symmetric auto_gptq)
        self._quant(quant_method="awq", bits=4)
        self._quant(
            quant_method="auto-round",
            packing_format="auto_round:auto_gptq",
            sym=True,
            bits=4,
        )

    def test_quant_guard_rejects(self):
        cases = [
            # per-tensor fp8: unpageable scalar scales
            (dict(quant_method="fp8"), "block"),
            # ct int8 W8A8: no CUDA fused-MoE (NPU-only)
            (dict(quant_method="compressed-tensors", format="int-quantized"), "int8"),
            # ct pack-quantized asymmetric (zero-points unwired)
            (
                dict(
                    quant_method="compressed-tensors",
                    format="pack-quantized",
                    config_groups=self._ct_group(
                        type="int", num_bits=4, symmetric=False
                    ),
                ),
                "symmetric",
            ),
            # ct pack-quantized runtime act-order (group g_idx unwired for ct)
            (
                dict(
                    quant_method="compressed-tensors",
                    format="pack-quantized",
                    config_groups=self._ct_group(
                        type="int", num_bits=4, symmetric=True, actorder="group"
                    ),
                ),
                "act-order",
            ),
            # ct fp8 float-quantized PER-TENSOR (scalar scales unwired; only per-channel supported)
            (
                dict(
                    quant_method="compressed-tensors",
                    format="float-quantized",
                    config_groups={
                        "group_0": {
                            "weights": {
                                "type": "float",
                                "num_bits": 8,
                                "strategy": "tensor",
                            },
                            "input_activations": {"dynamic": True},
                        }
                    },
                ),
                "per-channel",
            ),
            # ct fp8 float-quantized with STATIC input scales (input-scale paging unwired)
            (
                dict(
                    quant_method="compressed-tensors",
                    format="float-quantized",
                    config_groups={
                        "group_0": {
                            "weights": {"type": "float", "num_bits": 8},
                            "input_activations": {"dynamic": False},
                        }
                    },
                ),
                "dynamic",
            ),
            # AutoRound asymmetric / auto_awq (wna16 zero-point path unwired)
            (
                dict(
                    quant_method="auto-round",
                    packing_format="auto_round:auto_awq",
                    sym=False,
                    bits=4,
                ),
                "AutoRound",
            ),
        ]
        for qc, fragment in cases:
            with self.assertRaises(RuntimeError) as cm:
                check_paged_experts_quant(SimpleNamespace(quantization_config=qc))
            self.assertIn(fragment, str(cm.exception))


if __name__ == "__main__":
    unittest.main()
