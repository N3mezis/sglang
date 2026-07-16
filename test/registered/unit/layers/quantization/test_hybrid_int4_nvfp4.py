"""Unit tests for the RTN-int4 backbone quant (hybrid_int4_nvfp4)."""

import unittest

import torch

from sglang.srt.layers.quantization.hybrid_int4_nvfp4 import (
    Int4LinearMethod,
    dequantize_int4,
    rtn_quantize_int4,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestRtnInt4(CustomTestCase):
    def test_pack_shapes_and_roundtrip_error(self):
        torch.manual_seed(0)
        out, cin, g = 128, 512, 128
        w = torch.randn(out, cin)
        packed, scale = rtn_quantize_int4(w, group_size=g)
        self.assertEqual(packed.shape, (out, cin // 2))
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(scale.shape, (out, cin // g))
        deq = dequantize_int4(packed, scale, out_dtype=torch.float32)
        self.assertEqual(deq.shape, (out, cin))
        # symmetric int4 (16 levels) per-128 groups on unit-normal data lands at ~0.10-0.13 mean relative
        # error — inherent to RTN int4 (this lossiness is the MLA-quality risk measured via ppl at boot).
        # The bound only needs to catch a BROKEN pack/unpack (which blows up well past this).
        rel = (deq - w).abs().mean() / w.abs().mean()
        self.assertLess(rel.item(), 0.16, f"int4 RTN relative error too high: {rel.item()}")
        self.assertGreater(rel.item(), 0.02, "suspiciously low error — check the quant is real")

    def test_nibble_encoding_recovers_extremes(self):
        # a row spanning the full int4 range must round-trip its quantized levels exactly (scale-multiplied)
        w = torch.tensor([[-8.0, -1.0, 0.0, 7.0]]) * 0.5  # cin=4, one group
        packed, scale = rtn_quantize_int4(w, group_size=4)
        deq = dequantize_int4(packed, scale, out_dtype=torch.float32)
        # amax=4.0 -> scale=4/7; q=round(w/scale) then *scale. Check monotonic + sign preserved.
        self.assertTrue(torch.all(torch.sign(deq) == torch.sign(w)))
        self.assertAlmostEqual(deq[0, 2].item(), 0.0, places=5)
        self.assertLess(deq[0, 0].item(), deq[0, 1].item())  # -8 level < -1 level

    def test_odd_input_dim_falls_back_to_per_row(self):
        # in not divisible by group_size (but even) -> single per-row group, still round-trips
        out, cin = 8, 96  # 96 % 128 != 0
        w = torch.randn(out, cin)
        packed, scale = rtn_quantize_int4(w, group_size=128)
        self.assertEqual(scale.shape, (out, 1))
        deq = dequantize_int4(packed, scale, out_dtype=torch.float32)
        self.assertEqual(deq.shape, (out, cin))

    def test_linear_method_apply_matches_dequant_reference(self):
        torch.manual_seed(1)
        out, cin, g = 64, 256, 128
        w = torch.randn(out, cin)
        packed, scale = rtn_quantize_int4(w, group_size=g)

        class _Cfg:
            group_size = g

        class _Layer(torch.nn.Module):
            pass

        layer = _Layer()
        layer.qweight = torch.nn.Parameter(packed, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)

        method = Int4LinearMethod(_Cfg())
        x = torch.randn(4, cin)
        out_method = method.apply(layer, x)
        # reference: dequant weight then F.linear
        wq = dequantize_int4(packed, scale, out_dtype=x.dtype)
        out_ref = torch.nn.functional.linear(x, wq)
        self.assertTrue(torch.equal(out_method, out_ref))
        # and it should approximate the true bf16 matmul within the quant error
        out_true = torch.nn.functional.linear(x, w)
        rel = (out_method - out_true).abs().mean() / out_true.abs().mean()
        self.assertLess(rel.item(), 0.15, f"int4 linear apply too far from bf16: {rel.item()}")


if __name__ == "__main__":
    unittest.main()
