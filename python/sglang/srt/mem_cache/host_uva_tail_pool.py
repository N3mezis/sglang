"""Host-resident (mapped-UVA) KV tail pool for KV-streaming: the cold tail of ONE request's live-decode KV
cache lives in pinned host memory, aliased as CUDA tensors so the stock triton decode reads it directly over
PCIe (proven bit-exact + capture-safe + link-rate, docs/design/kv-streaming.md increments 1c/2).

Contrast with HiCache's host pool (``memory_pool_host.py``): that is ``cudaHostRegister``-pinned (copy-only —
a kernel cannot dereference it) and tiers ACROSS requests (prefix reuse). This tiers WITHIN one live sequence
and is UVA-mapped so attention reads it in place — no per-step re-staging to HBM.

Slot-indexed like ``MHATokenToKVPool`` with the same ``get_key_buffer``/``get_value_buffer``/``set_kv_buffer``
surface so it is drop-in for the triton attention backend (the tail becomes a second decode pass over this
pool's buffers; increment 4 wires the dispatch + window→tail eviction into the serving loop).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List

import torch

# Reuse the fork's proven mapped-pinned UVA machinery (same as paged-experts + the KV probes):
from sglang.jit_kernel.paged_experts_decide import paged_experts_host_devptr
from sglang.srt.layers.moe.paged_experts.store import _pinned_empty


def _uva_buffer(shape, dtype: torch.dtype):
    """A CUDA tensor ALIASING mapped-UVA pinned host memory (kernel-readable over PCIe, host-resident).

    Aliases via ``__cuda_array_interface__`` as ``uint16`` then ``.view(dtype)`` so 2-byte dtypes (fp16/bf16 —
    bf16 has no CAI typestr) both work. Returns ``(device_view, host_tensor)``; keep the host tensor alive for
    the lifetime of the view (its pages back the UVA mapping)."""
    assert dtype.itemsize == 2, f"host-UVA tail pool is 2-byte KV only (fp16/bf16), got {dtype}"
    host = _pinned_empty(tuple(shape), dtype)
    host.zero_()
    cai = SimpleNamespace()
    cai.__cuda_array_interface__ = {
        "shape": tuple(shape),
        "typestr": "<u2",  # element-size-only alias; reinterpreted below
        "data": (int(paged_experts_host_devptr(host)), False),
        "version": 3,
        "strides": None,
    }
    device_view = torch.as_tensor(cai, device="cuda").view(dtype)
    return device_view, host


class HostUVATailKVPool:
    """Per-layer mapped-UVA K/V buffers for a request's cold KV tail, plus a bump slot allocator and the
    full-context-position → tail-slot map (SWA ``full_to_swa_index_mapping`` analog)."""

    def __init__(
        self,
        size: int,  # tail slot capacity (bounded by host RAM, not VRAM)
        layer_num: int,
        head_num: int,
        head_dim: int,
        max_context: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self.size = size
        self.layer_num = layer_num
        self.head_num = head_num
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self._k: List[torch.Tensor] = []
        self._v: List[torch.Tensor] = []
        self._host: List[torch.Tensor] = []  # keep host pages alive
        for _ in range(layer_num):
            kd, kh = _uva_buffer((size, head_num, head_dim), dtype)
            vd, vh = _uva_buffer((size, head_num, head_dim), dtype)
            self._k.append(kd)
            self._v.append(vd)
            self._host += [kh, vh]
        self._next = 0  # bump allocator; a live sequence's tail only grows (freed at request end)
        # full-context position -> tail slot (-1 == still in the HBM window). int32, on device (the attention
        # backend gathers tail_kv_indices from it, like req_to_token).
        self.full_to_tail = torch.full((max_context,), -1, dtype=torch.int32, device=device)

    # --- MHATokenToKVPool-compatible surface (drop-in for the triton backend) ---
    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self._k[layer_id]

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        return self._v[layer_id]

    # --- tail management ---
    def alloc(self, n: int) -> torch.Tensor:
        """Reserve ``n`` contiguous tail slots (int64 device indices)."""
        assert self._next + n <= self.size, f"tail pool overflow ({self._next}+{n} > {self.size})"
        s = torch.arange(self._next, self._next + n, dtype=torch.int64, device=self.device)
        self._next += n
        return s

    def evict_layer(
        self,
        layer_id: int,
        src_k: torch.Tensor,  # window pool key buffer [win_size, H, D] (device)
        src_v: torch.Tensor,
        src_slots: torch.Tensor,  # window slots aging out
        tail_slots: torch.Tensor,  # destination tail slots (from alloc)
    ) -> None:
        """Copy KV rows window(HBM) -> tail(UVA host) for one layer — a device->UVA scatter over PCIe.
        1 row/layer/token; trivial bandwidth (96 KB/token all layers)."""
        self._k[layer_id][tail_slots] = src_k[src_slots].to(self.dtype)
        self._v[layer_id][tail_slots] = src_v[src_slots].to(self.dtype)

    def evict_positions(
        self, positions: torch.Tensor, tail_slots: torch.Tensor
    ) -> None:
        """Record the full-context-position -> tail-slot mapping for evicted tokens."""
        self.full_to_tail[positions] = tail_slots.to(torch.int32)

    def tail_indices_for(self, positions: torch.Tensor) -> torch.Tensor:
        """Gather the tail slots for a set of (already-evicted) context positions — the attention backend's
        ``tail_kv_indices`` for this request's cold context."""
        return self.full_to_tail[positions]
