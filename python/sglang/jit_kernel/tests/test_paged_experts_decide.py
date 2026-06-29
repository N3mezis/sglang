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
    paged_experts_decide_bounded,
    paged_experts_decide_wave,
    paged_experts_gather,
    paged_experts_host_devptr,
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


@requires_cuda
def test_gather_dynamic_count():
    """decide -> gather: the gather moves exactly the experts decide chose (count read on-device), placing
    the right host rows in the right slots."""
    E, K, W = 16, 6, 32  # 32 float32 = 128 B/expert (16-byte aligned)
    store = torch.empty((E, W), dtype=torch.float32, device="cpu", pin_memory=True)
    for e in range(E):
        store[e].fill_(float(e + 1))  # expert e's data == e+1
    devptr = paged_experts_host_devptr(store)
    slot = torch.zeros((K, W), dtype=torch.float32, device="cuda")
    for s in range(K):
        slot[s].fill_(
            float(s + 1)
        )  # slots 0..K-1 start holding experts 0..K-1 (value s+1)

    sc, se, es, lu, fq, src, dst, n_out, idx = _new_state(E, K)
    # route to expert 1 (resident hit) + 9, 12 (misses) -> decide pages 2 experts
    paged_experts_decide(
        _i32([1, 9, 12]), sc, se, es, lu, fq, False, src, dst, n_out, idx
    )
    n = int(n_out.item())
    assert n == 2, n
    paged_experts_gather(devptr, slot, src, dst, n_out, W * 4)
    torch.cuda.synchronize()
    # the two paged experts now sit in their assigned slots, with the right host values
    for i in range(n):
        e, s = int(src[i].item()), int(dst[i].item())
        assert (slot[s] == e + 1).all().item(), f"expert {e} -> slot {s}"
    # untouched slots keep their original contents (gather moved exactly n, not K)
    touched = set(int(dst[i].item()) for i in range(n))
    for s in range(K):
        if s not in touched:
            assert (slot[s] == s + 1).all().item(), f"slot {s} should be untouched"


@requires_cuda
def test_decide_gather_capturable():
    """The full per-step primitive (decide -> gather) captured once and replayed with topk mutated in
    place: each replay pages exactly the misses for that step (dynamic count survives capture).
    """
    E, K, W = 16, 6, 32
    store = torch.empty((E, W), dtype=torch.float32, device="cpu", pin_memory=True)
    for e in range(E):
        store[e].fill_(float(e + 1))
    devptr = paged_experts_host_devptr(store)
    slot = torch.zeros((K, W), dtype=torch.float32, device="cuda")
    for s in range(K):
        slot[s].fill_(float(s + 1))
    sc, se, es, lu, fq, src, dst, n_out, idx = _new_state(E, K)
    topk_buf = _i32([1, 9, 12])

    def step():
        paged_experts_decide(topk_buf, sc, se, es, lu, fq, False, src, dst, n_out, idx)
        paged_experts_gather(devptr, slot, src, dst, n_out, W * 4)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        step()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()

    # replay with a fresh routing step; the experts it pages must end up resident in their slots
    topk_buf.copy_(_i32([2, 10, 13]))
    g.replay()
    torch.cuda.synchronize()
    es_l = es.tolist()
    for e in (2, 10, 13):
        sidx = es_l[e]
        assert (
            sidx >= 0 and (slot[sidx] == e + 1).all().item()
        ), f"expert {e} not gathered to its slot"


# ---------------------------------------------------------------------------
# Bounded (pinned-window) decide
# ---------------------------------------------------------------------------
def _ref_decide_bounded(
    topk, slot_expert, expert_slot, slot_lastuse, freq, step, lfu, defer_cold, log2hot, log2cold
):
    """In-place bounded keep-warm decision; returns (src, dst, cold_log, cold_dst, needed)."""
    E, K = len(expert_slot), len(slot_expert)
    distinct = [int(e) for e in topk if 0 <= e < E]
    for e in distinct:
        freq[e] += 1
        s = expert_slot[e]
        if s >= 0:
            slot_lastuse[s] = step
    src, dst, cold_log, cold_dst = [], [], [], []
    for e in distinct:
        if expert_slot[e] >= 0:
            continue
        hi = log2hot[e]
        if defer_cold and hi < 0:  # replay-twice defer: record logical id, no eviction, stays masked
            cold_log.append(e)
            continue
        victim, best_f, best_lu = -1, None, None
        for s in range(K):
            se = slot_expert[s]
            if se in distinct:  # never evict a slot needed this step
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
        if hi >= 0:  # windowed -> on-device gather from host_hot
            src.append(hi)
            dst.append(victim)
        else:  # Rung 1 cold -> host_cold gather (cold-block index)
            cold_log.append(log2cold[e])
            cold_dst.append(victim)
    needed = [1 if (slot_expert[s] >= 0 and slot_expert[s] in distinct) else 0 for s in range(K)]
    return src, dst, cold_log, cold_dst, needed


def _window_maps(E, W):
    """Experts [0, W) are hot (pinned window), [W, E) are cold. log2hot/log2cold as the kernel expects."""
    log2hot = [e if e < W else -1 for e in range(E)]
    log2cold = [(e - W) if e >= W else -1 for e in range(E)]
    return _i32(log2hot), _i32(log2cold)


def _new_bounded_buffers(K):
    cold_log = _i32([0] * K)
    cold_dst = _i32([0] * K)
    cold_n = _i32([0])
    needed = _i32([0] * K)
    return cold_log, cold_dst, cold_n, needed


@requires_cuda
@pytest.mark.parametrize("lfu", [False, True])
@pytest.mark.parametrize("defer_cold", [False, True])
def test_decide_bounded_matches_reference(lfu, defer_cold):
    """Bounded decide matches the pure-Python reference over a random multi-step run: window hits land in
    (src,dst), cold misses either defer (logical id in cold_log, unresident) or take a slot + emit the
    cold-block index, and needed[] marks slots in use this step."""
    E, K, W = 16, 6, 8  # experts 0..7 hot (window), 8..15 cold; K=6 resident slots
    log2hot, log2cold = _window_maps(E, W)
    sc, se, es, lu, fq, src, dst, n_out, idx = _new_state(E, K)
    cold_log, cold_dst, cold_n, needed = _new_bounded_buffers(K)
    r_se, r_es = list(range(K)), [-1] * E
    r_es[:K] = list(range(K))
    r_lu, r_fq = [0] * K, [0] * E
    rlog2hot = [e if e < W else -1 for e in range(E)]
    rlog2cold = [(e - W) if e >= W else -1 for e in range(E)]

    torch.manual_seed(1)
    for step in range(1, 25):
        ndist = int(torch.randint(1, K + 1, (1,)).item())  # 1..K distinct (keep-warm regime)
        experts = torch.randperm(E)[:ndist].tolist()
        paged_experts_decide_bounded(
            _i32(experts), sc, se, es, lu, fq, lfu, defer_cold, log2hot, log2cold,
            src, dst, n_out, cold_log, cold_dst, cold_n, idx, needed,
        )
        r_src, r_dst, r_cl, r_cd, r_needed = _ref_decide_bounded(
            experts, r_se, r_es, r_lu, r_fq, step, lfu, defer_cold, rlog2hot, rlog2cold
        )
        nw, nc = int(n_out.item()), int(cold_n.item())
        assert int(sc.item()) == step, f"step {step}: counter"
        assert se.tolist() == r_se, f"step {step}: slot_expert"
        assert es.tolist() == r_es, f"step {step}: expert_slot"
        assert idx.tolist() == r_es, f"step {step}: idx snapshot"
        assert src[:nw].tolist() == r_src, f"step {step}: windowed src"
        assert dst[:nw].tolist() == r_dst, f"step {step}: windowed dst"
        assert cold_log[:nc].tolist() == r_cl, f"step {step}: cold_log"
        if not defer_cold:  # cold_dst only meaningful for Rung-1 (deferred entries don't take a slot)
            assert cold_dst[:nc].tolist() == r_cd, f"step {step}: cold_dst"
        assert needed.tolist() == r_needed, f"step {step}: needed"
        if defer_cold:  # deferred (window-miss) experts must stay unresident this step
            for e in cold_log[:nc].tolist():
                assert es[e].item() == -1, f"step {step}: deferred expert {e} should be unresident"


@requires_cuda
def test_decide_bounded_is_cuda_graph_capturable():
    """Capture decide_bounded once and replay with topk mutated in place; on-device counter makes the
    captured residency evolution identical to the eager run (the substrate for replay-twice)."""
    E, K, W = 16, 6, 8
    log2hot, log2cold = _window_maps(E, W)
    steps = [[1, 6, 9], [2, 7, 11], [9, 3, 14], [6, 1, 2], [0, 7, 11]]

    def fresh():
        st = _new_state(E, K)
        return (*st, *_new_bounded_buffers(K))

    # eager reference (kernel, no capture)
    sc, se, es, lu, fq, src, dst, n_out, idx, cl, cd, cn, nd = fresh()
    eager_idx = []
    for experts in steps:
        paged_experts_decide_bounded(
            _i32(experts), sc, se, es, lu, fq, False, True, log2hot, log2cold,
            src, dst, n_out, cl, cd, cn, idx, nd,
        )
        eager_idx.append(idx.tolist())

    # captured run
    sc, se, es, lu, fq, src, dst, n_out, idx, cl, cd, cn, nd = fresh()
    topk_buf = _i32(steps[0])
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        paged_experts_decide_bounded(
            topk_buf, sc, se, es, lu, fq, False, True, log2hot, log2cold,
            src, dst, n_out, cl, cd, cn, idx, nd,
        )
    torch.cuda.current_stream().wait_stream(s)
    sc.zero_()
    se.copy_(_i32(list(range(K))))
    es.copy_(_new_state(E, K)[2])
    lu.zero_()
    fq.zero_()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        paged_experts_decide_bounded(
            topk_buf, sc, se, es, lu, fq, False, True, log2hot, log2cold,
            src, dst, n_out, cl, cd, cn, idx, nd,
        )

    captured_idx = []
    for experts in steps:
        topk_buf.copy_(_i32(experts))
        g.replay()
        torch.cuda.synchronize()
        captured_idx.append(idx.tolist())

    assert captured_idx == eager_idx


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
