"""Expert store: the host backing for all E experts + the page-in transport into the K-slot GPU pool.

The K-slot GPU pool *is* the layer's own expert params (sglang's native loader filled slots 0..K-1). An
``ExpertStore`` holds all E experts per paged tensor on the host and copies the chosen ones into their
slots on a miss. It owns only the *backing and the byte movement* — not the residency *decision* (which
expert goes in which slot, when), which is the pager's job (``pager.py``). Splitting the two lets the
transport vary behind one interface:

* ``PinnedExpertStore`` — page-locked host RAM, paged with sglang's existing ``transfer_kv_per_layer_mla``
  block copy (indices read on-device, dynamic count, capture-safe). The fast default.
* ``PageableExpertStore`` — non-pinned host RAM, paged with a plain indexed copy. Correct but slower; for
  hosts that can't page-lock the full store.

Future tiers (disk-mmap, compressed) are additional ``ExpertStore`` subclasses — they implement the same
``page_in`` contract and need no change to the pager or the forward.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import torch

# packed-quant scaffolding the fused-MoE kernel never reads on the paged path
_NONPAGED_SUFFIXES = ("_g_idx", "_g_idx_sort_indices", "_weight_shape")


def discover_paged_params(layer, num_slots: int) -> Dict[str, torch.Tensor]:
    """Per-expert params on ``layer``: leading dim == num_slots (the K-slot pool) and non-empty per-slot."""
    out = {}
    for name, p in list(layer.named_parameters(recurse=False)) + list(
        layer.named_buffers(recurse=False)
    ):
        if any(name.endswith(s) for s in _NONPAGED_SUFFIXES):
            continue
        if p.dim() >= 1 and p.shape[0] == num_slots and p[0].numel() > 0:
            out[name] = p
    return out


class ExpertStore(ABC):
    """Host backing for all E experts + the page-in transport into the K-slot GPU pool.

    Subclasses choose the host backing and the byte movement; the pager decides which expert goes in
    which slot and hands the plan to ``page_in`` as index tensors. ``host[name]`` is an ``[E, *slot_shape]``
    CPU buffer per paged tensor (filled once at load time); ``gpu[name]`` is the layer's K-slot param;
    ``item_bytes[name]`` is the per-expert block size in bytes. The class attr ``pinned`` records whether
    the backing is page-locked (and gates the 8-byte alignment that the ``transfer_kv`` gather requires).
    """

    pinned: bool = False

    def __init__(self, layer, num_experts_E: int, num_resident_K: int, device):
        self.E = num_experts_E
        self.K = num_resident_K
        self.device = device
        self.gpu = discover_paged_params(
            layer, num_resident_K
        )  # the K-slot GPU pool (layer params)
        assert self.gpu, "no per-expert params found on layer"
        self.host: Dict[str, torch.Tensor] = {}
        self.item_bytes: Dict[str, int] = {}
        for name, p in self.gpu.items():
            self.host[name] = torch.empty(
                (self.E, *p.shape[1:]),
                dtype=p.dtype,
                device="cpu",
                pin_memory=self.pinned,
            )
            self.item_bytes[name] = p[0].numel() * p.element_size()
            # transfer_kv_per_layer_mla requires the per-expert block to be 8-byte aligned. Real weight
            # rows (bf16 / marlin qweight+scales+qzeros) satisfy this; a 1-D per-expert scalar scale
            # (e.g. fp8, 4 B) does not -> that needs the deferred scalar-gather path. The pageable copy
            # path has no such requirement.
            if self.pinned and self.item_bytes[name] % 8 != 0:
                raise RuntimeError(
                    f"[paged-experts] paged tensor {name!r} per-expert size {self.item_bytes[name]} B "
                    "is not 8-byte aligned (transfer_kv requirement); unsupported on the reuse gather "
                    "path. Use --paged-experts-store paged (the pageable copy has no such requirement)."
                )

    @abstractmethod
    def page_in(self, src_experts: torch.Tensor, dst_slots: torch.Tensor) -> None:
        """Copy ``host[src_experts[i]] -> gpu[dst_slots[i]]`` for every paged tensor.

        ``src_experts`` / ``dst_slots`` are device ``int64`` index tensors from the pager's decision; a
        no-op for an empty plan.
        """


class PinnedExpertStore(ExpertStore):
    """Pinned (page-locked) host store, paged with sglang's existing ``transfer_kv_per_layer_mla`` block
    copy — pinned-host -> device, indices read on-device, dynamic count, capture-safe. The fast default."""

    pinned = True

    def page_in(self, src_experts: torch.Tensor, dst_slots: torch.Tensor) -> None:
        if src_experts.numel() == 0:
            return
        from sgl_kernel import transfer_kv_per_layer_mla

        for name, gpu_param in self.gpu.items():
            transfer_kv_per_layer_mla(
                src=self.host[name],
                dst=gpu_param.data,
                src_indices=src_experts,
                dst_indices=dst_slots,
                item_size=self.item_bytes[name],
            )


class PageableExpertStore(ExpertStore):
    """Non-pinned host store, paged with a plain indexed copy (gather rows on the host, one H2D, scatter
    into the slots). Correct but slower; for hosts that can't page-lock the full store. ``transfer_kv``
    would read stale data from non-page-locked memory, so it is not used here."""

    pinned = False

    def page_in(self, src_experts: torch.Tensor, dst_slots: torch.Tensor) -> None:
        if src_experts.numel() == 0:
            return
        src_cpu = src_experts.to("cpu")
        for name, gpu_param in self.gpu.items():
            rows = self.host[name].index_select(0, src_cpu).to(gpu_param.device)
            gpu_param.data.index_copy_(0, dst_slots, rows)


def make_expert_store(
    layer, num_experts_E: int, num_resident_K: int, device, *, pin_host: bool
) -> ExpertStore:
    """Build the host expert store: pinned (fast ``transfer_kv`` path) or pageable (plain indexed copy)."""
    cls = PinnedExpertStore if pin_host else PageableExpertStore
    return cls(layer, num_experts_E, num_resident_K, device)
