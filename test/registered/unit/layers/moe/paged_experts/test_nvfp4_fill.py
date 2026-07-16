"""Numerical-equivalence test for the paged-experts NVFP4 fill adapter.

The fill (`pager._fill_nvfp4_from_checkpoint`) supports TWO on-disk packings that carry identical fp4
weight bytes but differ in per-expert tensor NAMES and the global/input-scale reciprocal convention:

  * compressed-tensors: weight_packed / weight_scale / weight_global_scale (= 1/actual) / input_global_scale (= 1/actual)
  * NVIDIA ModelOpt:     weight        / weight_scale / weight_scale_2      (= actual)  / input_scale        (= actual)

The adapter normalizes ModelOpt to the compressed-tensors convention by inverting the two global scalars
AT READ. This test builds two minimal checkpoints encoding the SAME underlying weights + actual scales,
one in each naming, fills a fake store from each, and asserts the store fills and the returned resident
scalars (g1_alphas/g2_alphas/w*_input_scale_quant) are bit-identical — i.e. a ModelOpt checkpoint loads
exactly like its compressed-tensors twin. Catches the highest-risk failure mode: a wrong reciprocal
convention that silently loads mis-scaled experts.
"""

import os
import tempfile
import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# tiny geometry: E experts, hidden H (mult of 16 for the group), intermediate I (mult of 16).
_E, _H, _I, _GROUP = 2, 32, 16, 16


class _FakeStore:
    """Minimal ExpertStore stand-in exercising exactly the fill's accessors: .E, .gpu (membership +
    w13_weight.dtype), .row(name, e) (writable per-expert host slice), .fill_tensor(name, full)."""

    def __init__(self, swizzled_names):
        self.E = _E
        self._host = {
            "w13_weight": torch.zeros(_E, 2 * _I, _H // 2, dtype=torch.uint8),
            "w2_weight": torch.zeros(_E, _H, _I // 2, dtype=torch.uint8),
        }
        # gpu carries the weights (dtype-checked) plus the swizzled-scale param names the store would
        # have discovered for this packing (compressed -> *_weight_scale; modelopt -> *_blockscale_swizzled).
        self.gpu = {k: v for k, v in self._host.items()}
        for n in swizzled_names:
            self.gpu[n] = None
        self.filled = {}

    def row(self, name, e):
        return self._host[name][e]

    def fill_tensor(self, name, full):
        self.filled[name] = full.clone()


def _write_ckpt(path, *, modelopt, weight_bytes, block_scales, w_actual, in_actual):
    """Write a single-shard safetensors of E experts x {gate,up,down}_proj in the requested naming.

    weight_bytes[proj]  : [rows, cols] uint8 packed fp4 (per expert, same for all experts here)
    block_scales[proj]  : [rows, cols/16] fp8 block scales
    w_actual, in_actual : the ACTUAL per-(proj) weight/input global scalars. compressed stores 1/actual.
    """
    from safetensors.torch import save_file

    tensors = {}
    for e in range(_E):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            pre = f"model.layers.3.mlp.experts.{e}.{proj}"
            tensors[f"{pre}.{'weight' if modelopt else 'weight_packed'}"] = weight_bytes[proj].clone()
            tensors[f"{pre}.weight_scale"] = block_scales[proj].clone()
            wname = "weight_scale_2" if modelopt else "weight_global_scale"
            iname = "input_scale" if modelopt else "input_global_scale"
            w = w_actual[proj] if modelopt else 1.0 / w_actual[proj]
            i = in_actual[proj] if modelopt else 1.0 / in_actual[proj]
            tensors[f"{pre}.{wname}"] = torch.tensor([w], dtype=torch.float32)
            tensors[f"{pre}.{iname}"] = torch.tensor([i], dtype=torch.float32)
    save_file(tensors, os.path.join(path, "model.safetensors"))


class TestNvfp4FillModeloptEquivalence(CustomTestCase):
    def _run_fill(self, modelopt, swizzled_names, tmp, **ck):
        from sglang.srt.layers.moe.paged_experts.pager import (
            _fill_nvfp4_from_checkpoint,
        )

        d = os.path.join(tmp, "modelopt" if modelopt else "compressed")
        os.makedirs(d, exist_ok=True)
        _write_ckpt(d, modelopt=modelopt, **ck)
        store = _FakeStore(swizzled_names)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        scalars = _fill_nvfp4_from_checkpoint(store, d, layer_idx=3, device=dev)
        return store, scalars

    def test_modelopt_matches_compressed_tensors(self):
        torch.manual_seed(0)
        # identical underlying fp4 weight bytes + block scales for both packings
        weight_bytes = {
            "gate_proj": torch.randint(0, 255, (_I, _H // 2), dtype=torch.uint8),
            "up_proj": torch.randint(0, 255, (_I, _H // 2), dtype=torch.uint8),
            "down_proj": torch.randint(0, 255, (_H, _I // 2), dtype=torch.uint8),
        }
        block_scales = {
            "gate_proj": torch.rand(_I, _H // _GROUP).to(torch.float8_e4m3fn),
            "up_proj": torch.rand(_I, _H // _GROUP).to(torch.float8_e4m3fn),
            "down_proj": torch.rand(_H, _I // _GROUP).to(torch.float8_e4m3fn),
        }
        # distinct, non-reciprocal-symmetric actual scales so a wrong inversion can't accidentally pass
        w_actual = {"gate_proj": 3.0, "up_proj": 5.0, "down_proj": 7.0}
        in_actual = {"gate_proj": 2.0, "up_proj": 11.0, "down_proj": 13.0}
        ck = dict(
            weight_bytes=weight_bytes,
            block_scales=block_scales,
            w_actual=w_actual,
            in_actual=in_actual,
        )

        with tempfile.TemporaryDirectory() as tmp:
            ct_store, ct_scalars = self._run_fill(
                False, ("w13_weight_scale", "w2_weight_scale"), tmp, **ck
            )
            mo_store, mo_scalars = self._run_fill(
                True, ("w13_blockscale_swizzled", "w2_blockscale_swizzled"), tmp, **ck
            )

        # 1. packed weights fill identically
        for n in ("w13_weight", "w2_weight"):
            self.assertTrue(
                torch.equal(ct_store._host[n], mo_store._host[n]),
                f"{n} packed weights differ between packings",
            )

        # 2. swizzled block scales fill identically (different param NAMES, same values)
        self.assertTrue(
            torch.equal(
                ct_store.filled["w13_weight_scale"].float(),
                mo_store.filled["w13_blockscale_swizzled"].float(),
            ),
            "w13 swizzled block scales differ",
        )
        self.assertTrue(
            torch.equal(
                ct_store.filled["w2_weight_scale"].float(),
                mo_store.filled["w2_blockscale_swizzled"].float(),
            ),
            "w2 swizzled block scales differ",
        )

        # 3. resident scalars (g*_alphas, w*_input_scale_quant) bit-identical
        self.assertEqual(set(ct_scalars), set(mo_scalars))
        for k in ct_scalars:
            self.assertTrue(
                torch.equal(ct_scalars[k], mo_scalars[k]),
                f"resident scalar {k} differs: {ct_scalars[k]} vs {mo_scalars[k]}",
            )

        # 4. sanity: the alphas match the closed form (max input over gate/up) * weight_scale, so the
        #    test would FAIL if the fill dropped the inversion. g1 = max(2,11)*3 = 33; g2 = 13*7 = 91.
        self.assertAlmostEqual(ct_scalars["g1_alphas"][0].item(), 33.0, places=4)
        self.assertAlmostEqual(ct_scalars["g2_alphas"][0].item(), 91.0, places=4)
        self.assertAlmostEqual(
            ct_scalars["w13_input_scale_quant"][0].item(), 1.0 / 11.0, places=5
        )


if __name__ == "__main__":
    unittest.main()
