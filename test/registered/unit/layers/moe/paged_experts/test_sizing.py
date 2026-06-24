"""Unit tests for srt/layers/moe/paged_experts/sizing.py"""

import unittest

from sglang.srt.layers.moe.paged_experts.sizing import (
    compute_num_resident_experts,
    kv_reserve_bytes_mha,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# Qwen3-30B-A3B (GQA): 48 layers, 4 KV heads, head_dim 128, fp16 KV -> 98304 B/token.
_QWEN3_KV = dict(
    num_layers=48,
    num_kv_heads=4,
    head_dim=128,
    kv_dtype_bytes=2,
    max_running_requests=1,
    context_length=2048,
)


class TestPagedExpertsSizing(CustomTestCase):
    def test_kv_cell_matches_qwen3(self):
        cell = kv_reserve_bytes_mha(**_QWEN3_KV)
        self.assertEqual(int(cell / 2048), 98304)  # per-token KV bytes

    def test_reproduces_measured_resident_K(self):
        # Reproduces a measured Qwen3-30B-int4 boot: free 6.66 GB, mem_fraction 0.85, K=25/128.
        K = compute_num_resident_experts(
            free_vram_bytes=6.66e9,
            mem_fraction=0.85,
            nonexpert_bytes=2.5e9,
            kv_reserve_bytes=kv_reserve_bytes_mha(**_QWEN3_KV),
            moe_layers=48,
            per_expert_layer_bytes=2.45e6,
            top_k=8,
            num_experts=128,
        )
        self.assertEqual(K, 25)

    def test_clamps_to_topk_and_E(self):
        common = dict(
            mem_fraction=0.85,
            nonexpert_bytes=2.5e9,
            kv_reserve_bytes=0,
            moe_layers=48,
            per_expert_layer_bytes=2.45e6,
            top_k=8,
            num_experts=128,
        )
        self.assertEqual(compute_num_resident_experts(free_vram_bytes=1e9, **common), 8)
        self.assertEqual(
            compute_num_resident_experts(free_vram_bytes=200e9, **common), 128
        )


if __name__ == "__main__":
    unittest.main()
