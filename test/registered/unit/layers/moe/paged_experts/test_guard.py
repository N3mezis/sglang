"""Unit tests for srt/layers/moe/paged_experts/guard.py"""

import unittest
from types import SimpleNamespace

from sglang.srt.layers.moe.paged_experts.guard import check_paged_experts_compat
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

    def test_aggregates_multiple_problems(self):
        with self.assertRaises(RuntimeError) as cm:
            check_paged_experts_compat(
                _sa(ep_size=2, enable_eplb=True, load_format="dummy")
            )
        self.assertEqual(str(cm.exception).count("\n  - "), 3)


if __name__ == "__main__":
    unittest.main()
