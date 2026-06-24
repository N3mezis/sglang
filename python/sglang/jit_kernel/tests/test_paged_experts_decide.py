"""Correctness + CUDA-graph-capture tests for the paged_experts_decide JIT kernel.

The kernel computes the per-decode-step Paged Experts residency plan on the GPU (no host sync). Validated
against a pure-Python reference of the same keep-warm/LRU and static-wave logic, and — the whole point of
moving the decision on-device — that the kernel is CUDA-graph-capturable (capture once, replay with the
inputs mutated in place, and the residency state evolves exactly as the eager run, since the step counter
is itself on-device).
"""

import sys

import pytest
import torch

from sglang.jit_kernel.paged_experts_decide import (
    paged_experts_decide,
    paged_experts_decide_wave,
)


# ---------------------------------------------------------------------------
# Pure-Python references (mirror the kernel logic)
# ---------------------------------------------------------------------------
def _ref_decide(topk, slot_expert, expert_slot, slot_lastuse, freq, step, lfu):
    """In-place keep-warm + LRU/LFU decision; returns (src, dst)."""
    distinct = [int(e) for e in topk if e >= 0]
    for e in distinct:
        freq[e] += 1
        s = expert_slot[e]
        if s >= 0:
            slot_lastuse[s] = step
    K = len(slot_expert)
    src, dst = [], []
    for e in distinct:
        if expert_slot[e] >= 0:
            continue
        victim, best_f, best_lu = -1, None, None
        for s in range(K):
            se = slot_expert[s]
            if se in distinct:
                continue
            f = freq[se] if (lfu and se >= 0) else 0
            lu = slot_lastuse[s]
            if best_f is None or f < best_f or (f == best_f and lu < best_lu):
                best_f, best_lu, victim = f, lu, s
        if victim < 0:
            continue
        old = slot_expert[victim]
        if old >= 0:
            expert_slot[old] = -1
        slot_expert[victim] = e
        expert_slot[e] = victim
        slot_lastuse[victim] = step
        src.append(e)
        dst.append(victim)
    return src, dst


def _ref_wave(topk, E, K, w):
    lo, hi = w * K, w * K + K
    idx = [(e - lo) if lo <= e < hi else -1 for e in range(E)]
    src, dst = [], []
    for e in topk:
        if not (lo <= e < hi):
            continue
        if e not in src:
            src.append(e)
            dst.append(e - lo)
    return src, dst, idx


def _i32(x):
    return torch.tensor(x, dtype=torch.int32, device="cuda")


def _new_state(E, K):
    """Fresh device residency state + output buffers, matching the in-tree pager's initial seeding
    (slots 0..K-1 hold experts 0..K-1)."""
    step_ctr = _i32([0])
    slot_expert = _i32(list(range(K)))
    expert_slot = _i32([-1] * E)
    expert_slot[:K] = torch.arange(K, dtype=torch.int32, device="cuda")
    slot_lastuse = _i32([0] * K)
    freq = _i32([0] * E)
    src = _i32([0] * K)
    dst = _i32([0] * K)
    n_out = _i32([0])
    idx = _i32([-1] * E)
    return step_ctr, slot_expert, expert_slot, slot_lastuse, freq, src, dst, n_out, idx


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for the decide kernel"
)


@requires_cuda
@pytest.mark.parametrize("lfu", [False, True])
def test_decide_matches_reference(lfu):
    E, K = 16, 6
    sc, se, es, lu, fq, src, dst, n_out, idx = _new_state(E, K)
    r_se = list(range(K))
    r_es = [-1] * E
    r_es[:K] = list(range(K))
    r_lu = [0] * K
    r_fq = [0] * E
    torch.manual_seed(0)
    for step in range(
        1, 25
    ):  # the kernel's on-device counter starts at 0 and ++s to 1 on the first call
        ndist = int(
            torch.randint(1, K + 1, (1,)).item()
        )  # 1..K distinct (keep-warm regime)
        experts = torch.randperm(E)[:ndist].tolist()
        paged_experts_decide(
            _i32(experts), sc, se, es, lu, fq, lfu, src, dst, n_out, idx
        )
        r_src, r_dst = _ref_decide(experts, r_se, r_es, r_lu, r_fq, step, lfu)
        n = int(n_out.item())
        assert int(sc.item()) == step, f"step {step}: counter"
        assert se.tolist() == r_se, f"step {step}: slot_expert"
        assert es.tolist() == r_es, f"step {step}: expert_slot"
        assert idx.tolist() == r_es, f"step {step}: idx == expert_slot snapshot"
        assert src[:n].tolist() == r_src, f"step {step}: src"
        assert dst[:n].tolist() == r_dst, f"step {step}: dst"


@requires_cuda
def test_decide_wave_matches_reference():
    E, K = 16, 6
    sc, se, es, lu, fq, src, dst, n_out, idx = _new_state(E, K)
    experts = [0, 3, 7, 8, 13, 1, 9]  # distinct > K -> served in waves
    topk = _i32(experts)
    nwaves = (E + K - 1) // K
    served = []
    for w in range(nwaves):
        paged_experts_decide_wave(topk, E, K, w, src, dst, n_out, idx)
        r_src, r_dst, r_idx = _ref_wave(experts, E, K, w)
        n = int(n_out.item())
        assert idx.tolist() == r_idx, f"wave {w}: idx"
        assert src[:n].tolist() == r_src, f"wave {w}: src"
        assert dst[:n].tolist() == r_dst, f"wave {w}: dst"
        served += src[:n].tolist()
    assert sorted(served) == sorted(
        experts
    )  # every active expert served in exactly one wave


@requires_cuda
def test_decide_is_cuda_graph_capturable():
    """Capture the decide kernel once, then replay with topk mutated in place. Because the step counter is
    on-device, the captured replays evolve the residency state *identically* to an eager run.
    """
    E, K = 16, 6
    steps = [[1, 5, 9], [2, 5, 11], [9, 3, 14], [5, 1, 2], [0, 7, 11]]

    # eager reference run (kernel, no capture)
    sc, se, es, lu, fq, src, dst, n_out, idx = _new_state(E, K)
    eager_idx = []
    for experts in steps:
        paged_experts_decide(
            _i32(experts), sc, se, es, lu, fq, False, src, dst, n_out, idx
        )
        eager_idx.append(idx.tolist())

    # captured run: fixed input/state buffers, topk mutated in place between replays
    sc, se, es, lu, fq, src, dst, n_out, idx = _new_state(E, K)
    topk_buf = _i32(steps[0])
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):  # warmup (required before capture)
        paged_experts_decide(topk_buf, sc, se, es, lu, fq, False, src, dst, n_out, idx)
    torch.cuda.current_stream().wait_stream(s)
    # reset state after warmup so the captured replays reproduce the eager sequence from step 1
    sc.zero_()
    se.copy_(_i32(list(range(K))))
    es.copy_(_new_state(E, K)[2])
    lu.zero_()
    fq.zero_()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        paged_experts_decide(topk_buf, sc, se, es, lu, fq, False, src, dst, n_out, idx)

    captured_idx = []
    for experts in steps:
        topk_buf.copy_(_i32(experts))
        g.replay()
        torch.cuda.synchronize()
        captured_idx.append(idx.tolist())

    assert (
        captured_idx == eager_idx
    )  # exact match: on-device counter makes replay == eager


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
