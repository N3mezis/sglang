"""ExpertFill: the per-quant-format strategy for filling the host expert store from a checkpoint.

Mirrors sglang's compressed-tensors scheme pattern (one class per format, selected by predicate,
sharing a common interface) — the paged-experts analog of CompressedTensorsMoEScheme. Each fill
replicates, for ALL E experts, what the wrapped base method's create_weights + process_weights_after_
loading did for the K resident slots. Selection is by ``matches(store, quant_method)``: the store's
paged params (``store.gpu`` keys + dtypes) identify the format the base method produced, with the
checkpoint's quant_method disambiguating fills that register identical keys (gptq vs awq marlin).
Predicates are mutually exclusive, so the registry asserts exactly one match (no ordering fragility).
"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class ExpertFill(ABC):
    #: short human-readable format tag (for logs / the "no fill matched" error)
    name: str = "?"

    @abstractmethod
    def matches(self, store, quant_method: str) -> bool:
        """True iff this fill produces the layout in ``store.gpu``. ``quant_method`` is the lowercased
        checkpoint quant_method (``""`` if unquantized)."""

    @abstractmethod
    def fill(self, store, model_path: str, layer_idx: int, device) -> Optional[Any]:
        """Fill the host store for one layer from the checkpoint (all E experts). Returns a resident
        full-E table to stash on the method (nvfp4's per-expert scalars) or ``None``."""
