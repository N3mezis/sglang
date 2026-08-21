"""Speculative-routing prefetch (SGLANG_PE_PREFETCH=1): at layer L, run the NEXT layers' routers on
L's hidden state and hint the cold tier (``store.prefetch_cold`` → MADV_WILLNEED) for the predicted
experts, so the kernel's async read-ahead has them in the page cache by the time those layers gather.
This is the model predicting itself — the real routers evaluated one-to-N layers early on the residual
stream — no draft model, no trained head.

Validated on Qwen3-30B (docs/findings/ROUTE_GEOMETRY_PROBE.md): cross-layer stale-gate overlap@k is
87% at n=1 (82%/78% at n=2/3), and prediction accuracy is sharply bimodal in the router margin
(59/89/99/100% by quartile) — so the net is margin-shaped: the predicted top-k plus boundary experts
within DELTA of the k-th score, optionally floored (drop predictions whose margin over the (k+1)-th
score is below FLOOR — the near-coin-flip quartile) where wasted reads contend for disk queue depth.

Decode-only (bs=1), best-effort, read-only w.r.t. serving state. ``prefetch_cold`` is a no-op unless
the store's cold tier is a disk mmap, so this is safe to leave enabled on RAM-backed configs.

Generalized from the Laguna-only prototype on feature/paged-experts-stream-nvfp4 (models/laguna.py):
model files register their MoE blocks; scoring adapts to the router family (sigmoid+bias when an
``e_score_correction_bias`` exists — Laguna/DeepSeek-style — else raw logits, whose ranking equals
softmax ranking for Qwen-style top-k).
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

ON = os.environ.get("SGLANG_PE_PREFETCH", "0") != "0"
_N = int(
    os.environ.get("SGLANG_PE_PREFETCH_N", "1")
)  # lead: prefetch this many layers ahead
_MMAX = int(
    os.environ.get("SGLANG_PE_PREFETCH_MMAX", "16")
)  # cap on extras beyond top-k
_DELTA = float(
    os.environ.get("SGLANG_PE_PREFETCH_DELTA", "0.10")
)  # boundary width beyond the k-th score
_FLOOR = float(
    os.environ.get("SGLANG_PE_PREFETCH_FLOOR", "0.0")
)  # drop predictions with margin < FLOOR
_LOG_EVERY = int(
    os.environ.get("SGLANG_PE_PREFETCH_LOG", "500")
)  # prove-execution heartbeat (calls)

_layers = {}  # layer_id -> (moe module, top_k)
_calls = 0
_issued = 0
_mode_logged = False


def register(layer_id: int, moe, top_k: int) -> None:
    _layers[layer_id] = (moe, top_k)


_fail_logged = False


def _store(moe):
    global _fail_logged
    try:
        st = moe.experts.quant_method._pager.store
        return st if hasattr(st, "prefetch_cold") else None
    except Exception as e:
        if not _fail_logged:
            _fail_logged = True
            qm = getattr(getattr(moe, "experts", None), "quant_method", None)
            logger.warning(
                "[pe-prefetch] store resolution failed (%s: %s); experts=%s quant_method=%s "
                "_pager=%s — prefetch disabled for such layers",
                type(e).__name__,
                e,
                type(getattr(moe, "experts", None)).__name__,
                type(qm).__name__,
                getattr(qm, "_pager", "<missing>"),
            )
        return None


def _scores(moe, h: torch.Tensor) -> torch.Tensor:
    """The target layer's router scores on a (possibly stale) hidden state, in the same ordering the
    real router uses. Returns a flat [E] tensor."""
    out = moe.gate(h)
    rl = out[0] if isinstance(out, tuple) else out
    cap = getattr(moe, "router_logit_softcapping", 0.0)
    if cap and cap > 0.0:
        rl = torch.tanh(rl / cap) * cap
    bias = getattr(moe.gate, "e_score_correction_bias", None)
    if bias is not None:
        return torch.sigmoid(rl).flatten() + bias.flatten()
    return rl.flatten()  # monotone with softmax — ranking-equivalent


def prefetch_ahead(cur_moe, h: torch.Tensor) -> None:
    """Run layers L+1..L+N routers on h_L; hint the margin-shaped predicted-expert net to each layer's
    cold tier. Call at the top of the MoE forward, bs=1 decode only."""
    global _calls, _issued, _mode_logged
    cur_id = getattr(cur_moe, "layer_id", None)
    if cur_id is None:
        return
    if not _mode_logged:
        st = _store(cur_moe)
        if st is not None:
            _mode_logged = True
            direct = getattr(st, "cold_direct_all", lambda: None)()
            logger.info(
                "[pe-prefetch] active: store=%s cold_direct_all=%s (True means WILLNEED hints are "
                "suppressed — page-ins bypass the cache; hints only help mmap-fault reads)",
                type(st).__name__,
                direct,
            )
    for n in range(1, _N + 1):
        ent = _layers.get(cur_id + n)
        if ent is None:
            continue
        tgt, k = ent
        store = _store(tgt)
        if store is None:
            continue
        try:
            sc = _scores(tgt, h)
            vals, idx = torch.topk(sc, min(k + _MMAX, sc.numel()))
            vl = vals.tolist()
            il = idx.tolist()
            kth, floor_ref = vl[k - 1], vl[k] if len(vl) > k else vl[-1]
            keep = [
                il[i]
                for i in range(len(il))
                if (i < k or vl[i] >= kth - _DELTA)  # top-k + boundary band
                and (
                    not _FLOOR or vl[i] - floor_ref >= _FLOOR
                )  # optional low-margin cutoff
            ]
            if keep:
                store.prefetch_cold(keep)
                _issued += len(keep)
        except Exception:
            pass
    _calls += 1
    if _LOG_EVERY and _calls % _LOG_EVERY == 0:
        logger.info(
            "[pe-prefetch] alive: %d calls, %d expert hints issued (N=%d DELTA=%.2f FLOOR=%.2f)",
            _calls,
            _issued,
            _N,
            _DELTA,
            _FLOOR,
        )
