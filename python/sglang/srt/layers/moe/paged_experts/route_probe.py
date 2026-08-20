"""Routing-geometry probe (SGLANG_PE_ROUTE_PROBE=1) — Qwen3-MoE, bs=1 decode, diagnostic only.

Measures, from the full per-layer router logits the model already computes:
  1. PERSISTENCE      — |top-k(t) ∩ top-k(t-1)| / k per layer (the July probe's 26% number, re-measured).
  2. RUNNER-UP (B)    — of the experts that are NEW at t (not in t-1's top-k), what fraction sat in
                        t-1's runner-up band (ranks k+1..k+δ)? Also conditioned on COLD (outside a
                        top-W frequency resident proxy): do runner-ups predict the misses that COST?
  3. STALE-GATE (A/C) — apply layer L's gate to the SAME token's hidden from n layers earlier
                        (the residual-stream stale-input oracle): top-k overlap; margin-bucketed
                        accuracy (is the old 51% an average over a bimodal population?); and the
                        cold-recall vs wasted-reads grid at net widths k+m, per resident proxy W.

Zero effect on serving: read-only, decode-only (num_tokens==1), env-gated, logs every 100 tokens.
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

ON = os.environ.get("SGLANG_PE_ROUTE_PROBE", "0") != "0"
_MAX = int(os.environ.get("SGLANG_PE_ROUTE_PROBE_MAX", "2000"))          # tokens to sample
_NS = [int(x) for x in os.environ.get("SGLANG_PE_ROUTE_PROBE_NS", "1,2,3").split(",")]
_MS = [int(x) for x in os.environ.get("SGLANG_PE_ROUTE_PROBE_MS", "0,4,8,16").split(",")]
_WS = [int(x) for x in os.environ.get("SGLANG_PE_ROUTE_PROBE_WS", "32,64,91").split(",")]
_DS = [4, 8, 16, 24]                                                     # runner-up band widths
_WARM = 40                                                               # tokens before freq proxy counts

_hid = {}            # layer -> this token's hidden (residual stream ring)
_prev_rank = {}      # layer -> previous token's ranked expert ids (desc by logit)
_freq = {}           # layer -> {expert: count}
_prev_layer = 1 << 30
_tokens = 0

_persist = [0.0, 0]                    # sum, n
_new_cov = {d: [0, 0] for d in _DS}    # new-expert coverage by prev runner-up band
_cold_cov = {}                         # (W, d) -> [covered, total]  cold-miss coverage by prev runner-ups
_cold_rate = {}                        # W -> [cold, total]
_stale_overlap = {}                    # n -> [sum, cnt]
_grid = {}                             # (W, n, m) -> [recall_hit, cold_total]
_waste = {}                            # (W, n, m) -> [wasted, layer_cnt]
_margin_pairs = []                     # (margin, hit) for n=1 stale predictions, capped
_MARGIN_CAP = 400_000


def _report():
    log = logger.info
    log("[route-probe] ===== %d tokens =====", _tokens)
    log("[route-probe] persistence |topk(t) ∩ topk(t-1)|/k = %.1f%%  (n=%d layer-steps)",
        100 * _persist[0] / max(1, _persist[1]), _persist[1])
    log("[route-probe] runner-up coverage of NEW experts: %s",
        "  ".join("δ=%d: %.0f%% (%d)" % (d, 100 * c / max(1, t), t) for d, (c, t) in sorted(_new_cov.items())))
    for W in _WS:
        c, t = _cold_rate.get(W, [0, 1])
        cov = "  ".join("δ=%d: %.0f%%" % (d, 100 * _cold_cov.get((W, d), [0, 1])[0]
                        / max(1, _cold_cov.get((W, d), [0, 1])[1])) for d in _DS)
        log("[route-probe] W=%d cold-rate=%.1f%% | COLD-miss runner-up coverage: %s", W, 100 * c / max(1, t), cov)
    log("[route-probe] stale-gate overlap@k: %s",
        "  ".join("n=%d: %.0f%%" % (n, 100 * s / max(1, c)) for n, (s, c) in sorted(_stale_overlap.items())))
    if _margin_pairs:
        ms = sorted(m for m, _ in _margin_pairs)
        qs = [ms[int(q * (len(ms) - 1))] for q in (0.25, 0.5, 0.75)]
        buck = [[0, 0], [0, 0], [0, 0], [0, 0]]
        for m, hit in _margin_pairs:
            b = sum(m > q for q in qs)
            buck[b][0] += hit
            buck[b][1] += 1
        log("[route-probe] n=1 stale accuracy by margin quartile (Q1 lowest): %s  (edges %.3f/%.3f/%.3f)",
            "  ".join("Q%d: %.0f%% (%d)" % (i + 1, 100 * h / max(1, t), t) for i, (h, t) in enumerate(buck)),
            *qs)
    for W in _WS:
        for n in _NS:
            cells = []
            for m in _MS:
                h, t = _grid.get((W, n, m), [0, 0])
                w, wc = _waste.get((W, n, m), [0, 1])
                cells.append("k+%-2d rec=%2.0f%% waste=%.1f" % (m, 100 * h / max(1, t), w / max(1, wc)))
            log("[route-probe] W=%d n=%d | %s", W, n, "  ".join(cells))


def probe(moe, hidden, router_logits, topk_output):
    global _prev_layer, _tokens
    if _tokens >= _MAX or hidden.shape[0] != 1:
        return
    L = moe.layer_id
    try:
        actual = set(int(x) for x in topk_output.topk_ids.flatten().tolist())
        logits = router_logits.flatten().float()
    except Exception:
        return
    k = len(actual) or 8
    maxrank = k + max(max(_DS), max(_MS))

    if L <= _prev_layer and _hid:          # layer id dropped -> new token
        _tokens += 1
        _hid.clear()
        if _tokens % 100 == 0 or _tokens == _MAX:
            _report()
    _prev_layer = L

    ranked_now = torch.topk(logits, min(maxrank, logits.numel())).indices.tolist()
    fL = _freq.setdefault(L, {})
    warm = _tokens >= _WARM
    residents = {W: set(sorted(fL, key=fL.get, reverse=True)[:W]) for W in _WS} if warm else {}

    # -- cross-token: persistence + runner-up provenance ------------------------------------------
    pr = _prev_rank.get(L)
    if pr is not None:
        prev_topk = set(pr[:k])
        _persist[0] += len(actual & prev_topk) / k
        _persist[1] += 1
        new = actual - prev_topk
        for d in _DS:
            band = set(pr[k:k + d])
            _new_cov[d][0] += len(new & band)
            _new_cov[d][1] += len(new)
        if warm:
            for W, res in residents.items():
                cold = actual - res
                cr = _cold_rate.setdefault(W, [0, 0])
                cr[0] += len(cold)
                cr[1] += len(actual)
                for d in _DS:
                    band = set(pr[k:k + d])
                    cc = _cold_cov.setdefault((W, d), [0, 0])
                    cc[0] += len(cold & band)
                    cc[1] += len(cold)

    # -- cross-layer (same token): stale-gate oracle -----------------------------------------------
    for n in _NS:
        hp = _hid.get(L - n)
        if hp is None:
            continue
        try:
            srl, _ = moe.gate(hp)
            srl = srl.flatten().float()
            svals, sidx = torch.topk(srl, min(maxrank, srl.numel()))
            sranked = sidx.tolist()
        except Exception:
            continue
        so = _stale_overlap.setdefault(n, [0.0, 0])
        so[0] += len(actual & set(sranked[:k])) / k
        so[1] += 1
        if n == 1 and len(svals) > k and len(_margin_pairs) < _MARGIN_CAP:
            thr = float(svals[k])                       # (k+1)-th stale logit
            for j in range(k):
                _margin_pairs.append((float(svals[j]) - thr, sranked[j] in actual))
        if warm:
            for W, res in residents.items():
                cold = actual - res
                if not cold:
                    continue
                for m in _MS:
                    net = set(sranked[:k + m])
                    g = _grid.setdefault((W, n, m), [0, 0])
                    g[0] += len(cold & net)
                    g[1] += len(cold)
                    w = _waste.setdefault((W, n, m), [0, 0])
                    w[0] += len(net - res - actual)     # fetched, not resident, not used
                    w[1] += 1

    for e in actual:
        fL[e] = fL.get(e, 0) + 1
    _prev_rank[L] = ranked_now
    _hid[L] = hidden.detach().clone()
