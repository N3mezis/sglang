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

import logging
import math
import mmap
import os
import tempfile
from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)

# packed-quant scaffolding the fused-MoE kernel never reads on the paged path
_NONPAGED_SUFFIXES = ("_g_idx", "_g_idx_sort_indices", "_weight_shape")


def _alloc_disk_mmap(
    cold_dir: Optional[str], dims: tuple, dtype: torch.dtype
) -> torch.Tensor:
    """A host tensor backed by a MAP_SHARED file on disk (P4 cold tier) — RAM use is bounded by the OS page
    cache (clean pages evict back to the file under pressure), so a store far larger than RAM still loads.
    The file is unlinked immediately: the inode lives only as long as the mapping (auto-cleaned on free), so
    no stale multi-GB files are left behind. ``cold_dir`` must be on a real disk with room for the cold tier
    (NOT a tmpfs like /tmp, which would defeat the point); falls back to the system temp dir.
    """
    n_bytes = math.prod(dims) * torch.empty([], dtype=dtype).element_size()
    d = cold_dir or tempfile.gettempdir()
    os.makedirs(d, exist_ok=True)
    fd, path = tempfile.mkstemp(dir=d, suffix=".paged_experts_cold")
    try:
        os.ftruncate(fd, n_bytes)
        os.unlink(
            path
        )  # anonymous-on-disk: the inode persists while mmap'd, freed on munmap
        mm = mmap.mmap(
            fd, n_bytes, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE
        )
    finally:
        os.close(fd)  # the mapping keeps the inode alive after the fd is closed
    # torch.frombuffer keeps mm alive inside the tensor storage (munmap fires when the tensor is freed)
    return torch.frombuffer(mm, dtype=dtype, count=math.prod(dims)).reshape(dims)


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

    # --- checkpoint-fill accessors (store-layout-agnostic; used by ``pager.setup_pager``) ---
    # A single ``[E, *]`` host buffer here; ``WindowedExpertStore`` overrides both to route an expert into
    # its hot/cold tier, so the fill code never special-cases the store layout.
    def row(self, name: str, e: int) -> torch.Tensor:
        """Writable host slice backing expert ``e`` for paged tensor ``name`` (per-expert fill)."""
        return self.host[name][e]

    def fill_tensor(self, name: str, full: torch.Tensor) -> None:
        """Fill the whole host backing for ``name`` from a contiguous ``[E, *slot_shape]`` CPU tensor."""
        self.host[name].copy_(full)


class PinnedExpertStore(ExpertStore):
    """Pinned (page-locked) host store, paged with sglang's existing ``transfer_kv_per_layer_mla`` block
    copy — pinned-host -> device, indices read on-device, dynamic count, capture-safe. The fast default.
    """

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


class WindowedExpertStore(ExpertStore):
    """Pinned hot window + pageable cold tail — the fallback for stores that can't be fully page-locked.

    The ``W`` hot experts live in a page-locked ``host_hot[name]`` block (paged with ``transfer_kv``, and —
    in the captured path, pr3 — gatherable on-device through its UVA device pointer); the remaining ``E-W``
    cold experts live in a pageable ``host_cold[name]`` block (paged with a plain indexed copy, or — under
    capture — staged out-of-graph on a deferred miss). ``host[name]`` is *not* allocated: there is no single
    ``[E, *]`` buffer, so the fill goes through ``row`` / ``fill_tensor``.

    Membership defaults to the static ``[0, W)`` split (``hot_pos`` / ``cold_pos``); a frequency profile may
    later pin the hottest ``W`` (the maps make that a fill-order change, not a layout change). This is the
    >pin-ceiling path: ``W`` = what actually fits page-locked, the rest stays pageable but still served.
    """

    pinned = True  # the hot window is page-locked; the cold tail is pageable by design

    def __init__(
        self,
        layer,
        num_experts_E: int,
        num_resident_K: int,
        device,
        *,
        window_W: int,
        cold_backing: str = "ram",
        cold_dir: Optional[str] = None,
    ):
        self.E = num_experts_E
        self.K = num_resident_K
        self.device = device
        self.W = max(0, min(int(window_W), num_experts_E))
        self.cold_backing = (
            cold_backing  # "ram" (pageable) | "disk" (mmap'd file, page-cache-bounded)
        )
        self.gpu = discover_paged_params(layer, num_resident_K)
        assert self.gpu, "no per-expert params found on layer"
        self.host_hot: Dict[str, torch.Tensor] = (
            {}
        )  # [W, *shape] PINNED (transfer_kv / UVA gather)
        self.host_cold: Dict[str, torch.Tensor] = (
            {}
        )  # [E-W, *shape] cold tier (RAM pageable | disk mmap)
        self.item_bytes: Dict[str, int] = {}
        on_disk = cold_backing == "disk"
        for name, p in self.gpu.items():
            self.host_hot[name] = torch.empty(
                (self.W, *p.shape[1:]), dtype=p.dtype, device="cpu", pin_memory=True
            )
            cold_dims = (self.E - self.W, *p.shape[1:])
            # disk: a >RAM cold tier mmap'd to a file (page-cache-bounded) so the store can exceed RAM;
            # ram: a plain pageable tensor (the cold tier must fit RAM).
            self.host_cold[name] = (
                _alloc_disk_mmap(cold_dir, cold_dims, p.dtype)
                if on_disk
                else torch.empty(
                    cold_dims, dtype=p.dtype, device="cpu", pin_memory=False
                )
            )
            self.item_bytes[name] = p[0].numel() * p.element_size()
            # the hot tier feeds transfer_kv -> same 8-byte alignment requirement as the pinned store
            # (see ExpertStore.__init__). The pageable cold tier has none.
            if self.item_bytes[name] % 8 != 0:
                raise RuntimeError(
                    f"[paged-experts] paged tensor {name!r} per-expert size {self.item_bytes[name]} B "
                    "is not 8-byte aligned (transfer_kv requirement on the pinned window); unsupported. "
                    "Use --paged-experts-store paged (the pageable copy has no such requirement)."
                )
        # expert -> (tier, row). v1: static split -> hot experts [0, W), cold experts [W, E). hot_pos[e] is
        # the row of e in host_hot (-1 if cold); cold_pos[e] the row in host_cold (-1 if hot).
        self.hot_pos = torch.full((self.E,), -1, dtype=torch.int64)
        self.cold_pos = torch.full((self.E,), -1, dtype=torch.int64)
        self.hot_pos[: self.W] = torch.arange(self.W, dtype=torch.int64)
        self.cold_pos[self.W :] = torch.arange(self.E - self.W, dtype=torch.int64)

    def is_hot(self, e: int) -> bool:
        return bool(self.hot_pos[e] >= 0)

    # --- fill accessors: route per expert into the hot/cold tier (no single [E,*] buffer) ---
    def row(self, name: str, e: int) -> torch.Tensor:
        hp = int(self.hot_pos[e])
        if hp >= 0:
            return self.host_hot[name][hp]
        return self.host_cold[name][int(self.cold_pos[e])]

    def fill_tensor(self, name: str, full: torch.Tensor) -> None:
        # v1 membership is the contiguous [0, W) split, so the tiers are full[:W] / full[W:]. (A frequency
        # profile would gather by hot_pos/cold_pos instead — a fill-order change, deferred to P3.)
        self.host_hot[name].copy_(full[: self.W])
        self.host_cold[name].copy_(full[self.W :])

    def set_window_membership(self, hot_experts) -> None:
        """Re-pin the window to hold ``hot_experts`` (the top-W by routing frequency) instead of the static
        ``[0, W)`` — the P3 freq-ranked window. Permutes the host_hot/host_cold contents and rebuilds
        ``hot_pos``/``cold_pos`` so the cold tail becomes the *least*-routed experts (rare window-misses ->
        few replay-twice rounds). Runs once, out-of-graph, after a short profiling period; the GPU slots
        keep their (expert-indexed) data unchanged, so only the page-in *source* tier moves.
        """
        hot = list(hot_experts)[: self.W]
        assert len(set(hot)) == len(hot), "hot set has duplicates"
        hot_set = set(int(e) for e in hot)
        cold = [e for e in range(self.E) if e not in hot_set]
        new_hot_pos = torch.full((self.E,), -1, dtype=torch.int64)
        new_cold_pos = torch.full((self.E,), -1, dtype=torch.int64)
        for i, e in enumerate(hot):
            new_hot_pos[int(e)] = i
        for i, e in enumerate(cold):
            new_cold_pos[e] = i
        for name in self.gpu:
            # Gather every expert's current data (via the OLD maps), then re-split into the new tiers. The
            # transient [E,*] buffer is one layer's experts — freed after; only paid once at the refresh.
            full = torch.empty(
                (self.E, *self.host_hot[name].shape[1:]),
                dtype=self.host_hot[name].dtype,
                device="cpu",
            )
            for e in range(self.E):
                full[e].copy_(self.row(name, e))
            for i, e in enumerate(hot):
                self.host_hot[name][i].copy_(full[int(e)])
            for i, e in enumerate(cold):
                self.host_cold[name][i].copy_(full[e])
        self.hot_pos = new_hot_pos
        self.cold_pos = new_cold_pos

    def page_in(self, src_experts: torch.Tensor, dst_slots: torch.Tensor) -> None:
        if src_experts.numel() == 0:
            return
        src_cpu = src_experts.to("cpu")
        hot_mask = (
            self.hot_pos[src_cpu] >= 0
        )  # which planned experts live in the pinned window
        # hot experts -> transfer_kv from the pinned window (fast path), remapped to host_hot rows
        if bool(hot_mask.any()):
            sel = hot_mask.to(dst_slots.device)
            hot_src_rows = self.hot_pos[src_cpu[hot_mask]].to(src_experts.device)
            hot_dst = dst_slots[sel]
            from sgl_kernel import transfer_kv_per_layer_mla

            for name, gpu_param in self.gpu.items():
                transfer_kv_per_layer_mla(
                    src=self.host_hot[name],
                    dst=gpu_param.data,
                    src_indices=hot_src_rows,
                    dst_indices=hot_dst,
                    item_size=self.item_bytes[name],
                )
        # cold experts -> plain indexed copy from the pageable tail, remapped to host_cold rows
        cold_mask = ~hot_mask
        if bool(cold_mask.any()):
            sel = cold_mask.to(dst_slots.device)
            cold_rows = self.cold_pos[src_cpu[cold_mask]]
            cold_dst = dst_slots[sel]
            for name, gpu_param in self.gpu.items():
                rows = (
                    self.host_cold[name].index_select(0, cold_rows).to(gpu_param.device)
                )
                gpu_param.data.index_copy_(0, cold_dst, rows)


def make_expert_store(
    layer,
    num_experts_E: int,
    num_resident_K: int,
    device,
    *,
    pin_host: bool,
    window_W: int = 0,
    cold_backing: str = "ram",
    cold_dir: Optional[str] = None,
) -> ExpertStore:
    """Build the host expert store. ``window_W > 0`` and ``< E`` (with ``pin_host``) selects the windowed
    fallback (pinned hot window + cold tail) for stores that exceed the page-lock ceiling; else pinned (fast
    ``transfer_kv``) or pageable (plain indexed copy). ``cold_backing='disk'`` mmaps the windowed cold tier
    to a file (page-cache-bounded) so the store may exceed RAM (P4)."""
    if pin_host and 0 < window_W < num_experts_E:
        return WindowedExpertStore(
            layer,
            num_experts_E,
            num_resident_K,
            device,
            window_W=window_W,
            cold_backing=cold_backing,
            cold_dir=cold_dir,
        )
    cls = PinnedExpertStore if pin_host else PageableExpertStore
    return cls(layer, num_experts_E, num_resident_K, device)
