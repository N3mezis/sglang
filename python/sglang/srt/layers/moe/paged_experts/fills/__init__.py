"""Expert-store fill registry — one ExpertFill per quant format, selected by predicate.

The paged-experts analog of sglang's compressed-tensors scheme registry: ``setup_pager`` builds the
host store, then ``select_fill`` picks the one fill whose ``matches(store, quant_method)`` holds and
runs it. Predicates key on the store's paged params (``store.gpu`` keys + dtypes) — which identify the
layout the wrapped base method produced — with the checkpoint quant_method disambiguating fills that
register identical keys (gptq vs awq marlin). Predicates are mutually exclusive by construction, so
selection asserts exactly one match (no if/elif ordering to keep straight).

Adding a format = one ExpertFill subclass + one entry here.
"""

from .base import ExpertFill
from .fp4 import Dsv4Fp4Fill, Dsv4Mxfp4MarlinFill, Mxfp4Fill, Nvfp4Fill
from .fp8 import CtFp8ChannelFill, Fp8BlockFill
from .marlin_int4 import (
    AwqMarlinFill,
    CtWna16Fill,
    GptqMarlinFill,
    MoeWna16Fill,
)
from .unquantized import Bf16Fill

# Order is cosmetic (predicates are mutually exclusive; select_fill asserts a unique match). Grouped
# by family for readability.
FILLS = [
    GptqMarlinFill(),
    AwqMarlinFill(),
    MoeWna16Fill(),
    CtWna16Fill(),
    Fp8BlockFill(),
    CtFp8ChannelFill(),
    Mxfp4Fill(),
    Nvfp4Fill(),
    Dsv4Fp4Fill(),
    Dsv4Mxfp4MarlinFill(),
    Bf16Fill(),
]


def select_fill(store, quant_method: str) -> ExpertFill:
    """Return the one ExpertFill for this store's layout. Raises if zero or >1 match (a new/ambiguous
    layout — fail loud rather than silently mis-fill)."""
    hits = [f for f in FILLS if f.matches(store, quant_method)]
    if len(hits) != 1:
        raise RuntimeError(
            f"[paged-experts] expected exactly one expert fill for params {list(store.gpu)} "
            f"(quant_method={quant_method!r}); matched {[f.name for f in hits] or 'none'}. "
            "Supported: gptq/awq/moe-wna16 int4, ct int pack-quantized, fp8 block, ct fp8-channel, "
            "mxfp4, nvfp4, dsv4-fp4 (int8-packed + e8m0), unquantized bf16."
        )
    return hits[0]


__all__ = ["ExpertFill", "FILLS", "select_fill"]
