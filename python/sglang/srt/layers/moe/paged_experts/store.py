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

import ctypes
import json
import logging
import math
import mmap
import os
import tempfile
from collections import OrderedDict

from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)


def _moe_layer_count_hint() -> int:
    """Best-effort count of MoE layers in the model being served, for per-layer host budgets that must sum
    to a whole-host number. Reads the live ServerArgs/ModelConfig; falls back to a pessimistic 64 so a
    miss under-sizes the cache (slower) rather than over-committing RAM (fatal)."""
    try:
        from sglang.srt.server_args import get_global_server_args

        mp = get_global_server_args().model_path
        from sglang.srt.configs.model_config import ModelConfig

        mc = ModelConfig.from_server_args(get_global_server_args())
        htc = getattr(mc, "hf_text_config", None) or mc.hf_config
        n = int(getattr(mc, "num_hidden_layers", 0) or 0)
        dense = int(getattr(htc, "first_k_dense_replace", 0) or 0)
        return max(1, n - dense) if n else 64
    except Exception:
        return 64


def _pinned_empty(shape, dtype: torch.dtype) -> torch.Tensor:
    """Pinned host tensor allocated via raw ``cudaHostAlloc`` instead of torch's pinned allocator.

    On WSL2 the two hit DIFFERENT page-lock ceilings: torch's pinned path fails ~12 GB below what raw
    ``cudaHostAlloc`` can pin (measured 19.8 vs 32 GB on a 50 GB-RAM box), which both wastes window
    budget and desynchronizes the store from the sizing probe (which must measure the same ceiling the
    store will hit). The buffer is freed via ``cudaFreeHost`` when the tensor is garbage-collected.
    Falls back to ``torch.empty(pin_memory=True)`` if the CUDA runtime is unreachable.
    """
    import ctypes
    import weakref

    from sglang.srt.layers.moe.paged_experts.method import _cudart_handle

    rt = _cudart_handle()
    nbytes = int(math.prod(shape)) * dtype.itemsize
    if rt is None or nbytes == 0:
        return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
    ptr = ctypes.c_void_p()
    if rt.cudaHostAlloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes), 0) != 0:
        from sglang.srt.layers.moe.paged_experts.method import _pin_ceiling_cache_reset

        _pin_ceiling_cache_reset()  # the cached ceiling no longer holds; re-measure next boot
        raise RuntimeError(
            f"[paged-experts] cudaHostAlloc({nbytes / 1e9:.2f} GB) for the pinned expert store "
            "failed (OS page-lock ceiling). The auto window should have sized under it; if you set "
            "--paged-experts-num-resident or run other pinned-memory workloads, lower them, or use "
            "--paged-experts-store paged."
        )
    # torch.frombuffer keeps ``buf`` alive as long as the tensor's storage, so the finalizer on ``buf``
    # frees the pinned block only after every view of the storage is gone.
    buf = (ctypes.c_uint8 * nbytes).from_address(ptr.value)
    weakref.finalize(buf, rt.cudaFreeHost, ctypes.c_void_p(ptr.value))
    return torch.frombuffer(buf, dtype=torch.uint8).view(dtype).reshape(shape)


# packed-quant scaffolding the fused-MoE kernel never reads on the paged path, plus the nvfp4
# per-expert scalar scales (global/alpha/input-quant): 4-8 B each -> too small for the pinned gather
# (sub-8/16-byte rows). The four runtime nvfp4 scalars (g*_alphas, w*_input_scale_quant) are refreshed
# per-step into the K slots from a resident full-E table (see forward._gemm_hidden); the rest are dead
# after the nvfp4 method's process_weights_after_loading.
#
# NB: ``_g_idx`` / ``_g_idx_sort_indices`` are NOT excluded — for act-order (gptq desc_act=True /
# ct actorder='group') they are live per-expert [in] int32 tensors the marlin kernel reads at gather
# time, so they must page into slots alongside the qweight. For non-act-order the method's
# process_weights_after_loading replaces them with empty tensors, which discover_paged_params drops
# via its ``numel() > 0`` check — so leaving them pageable is safe for both.
# gpt-oss (mxfp4) adds per-expert SwiGLU scalars — swiglu_alpha/beta/limit, each a CONSTANT broadcast
# (full((E,), 1.702/1.0/7.0), not per-expert-varying and absent from the checkpoint). At 4 B/expert they
# are sub-8-byte and would fail the pinned-gather alignment gate; excluding them from paging is correct
# (the layer's [K] copy keeps the constant for every slot regardless of which expert is resident), and it
# keeps gpt-oss on the fast pinned + captured path instead of forcing --paged-experts-store paged (eager).
_NONPAGED_SUFFIXES = (
    "_weight_shape",
    "_global_scale",
    "_scale_2",
    "_alphas",
    "_scale_quant",
    "swiglu_alpha",
    "swiglu_beta",
    "swiglu_limit",
)


def _host_available_bytes() -> int:
    """Available host memory in bytes (Linux ``/proc/meminfo`` ``MemAvailable``), or 0 if unknown."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024  # kB -> bytes
    except Exception:
        pass
    return 0


def _alloc_disk_mmap(cold_dir: Optional[str], dims: tuple, dtype: torch.dtype):
    """A host tensor backed by a MAP_SHARED file on disk (P4 cold tier) — RAM use is bounded by the OS page
    cache (clean pages evict back to the file under pressure), so a store far larger than RAM still loads.
    The file is unlinked immediately: the inode lives only as long as the mapping (auto-cleaned on free), so
    no stale multi-GB files are left behind. ``cold_dir`` must be on a real disk with room for the cold tier
    (NOT a tmpfs like /tmp, which would defeat the point); falls back to the system temp dir.

    Returns ``(tensor, mm, fd_direct)`` — the ``mmap`` object for ``madvise`` read-ahead hints, and an
    O_DIRECT descriptor (or None) for page-cache-bypassing cold reads.
    """
    n_bytes = math.prod(dims) * torch.empty([], dtype=dtype).element_size()
    d = cold_dir or tempfile.gettempdir()
    os.makedirs(d, exist_ok=True)
    fd, path = tempfile.mkstemp(dir=d, suffix=".paged_experts_cold")
    fd_direct = None
    try:
        os.ftruncate(fd, n_bytes)
        # a second, O_DIRECT descriptor (opened while the path still exists) lets cold reads bypass
        # the page cache entirely — no fault storm, no disk->cache->staging double copy. None when the
        # filesystem refuses O_DIRECT; readers fall back to the mmap.
        try:
            fd_direct = os.open(path, os.O_RDONLY | os.O_DIRECT)
        except (OSError, AttributeError):
            fd_direct = None
        os.unlink(
            path
        )  # anonymous-on-disk: the inode persists while mmap'd/open, freed on close
        mm = mmap.mmap(
            fd, n_bytes, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE
        )
        try:
            # Best-effort: with THP/large-folio support, faults map 2 MB at a time instead of 4 KB —
            # a multi-MB expert row costs a handful of faults instead of hundreds. Ignored by kernels
            # without file-backed-THP; never fatal.
            mm.madvise(mmap.MADV_HUGEPAGE)
        except (OSError, ValueError, AttributeError):
            pass
    finally:
        os.close(fd)  # the mapping keeps the inode alive after the fd is closed
    # torch.frombuffer keeps mm alive inside the tensor storage (munmap fires when the tensor is freed)
    t = torch.frombuffer(mm, dtype=dtype, count=math.prod(dims)).reshape(dims)
    return t, mm, fd_direct


# --- shared staging machinery for cold-tier page-ins -----------------------------------------------
# One pinned buffer per (param-name, row-shape) reused across ALL layers (staging is sequential): the
# cold tier is pageable RAM or a disk mmap, and an H2D from pageable memory is silently synchronous and
# slow — gathering the rows into a pinned buffer first makes the H2D a fast async DMA. The gather stays
# SERIAL: with the MADV_WILLNEED read-ahead the page faults are already serviced concurrently by the
# kernel, so the copies run against warm pages — threading them was measured net-negative (pool overhead
# on warm memcpys; the fault parallelism was the winnable part and prefetch already claims it).
_STAGE_PIN: Dict = {}

# Small worker pool for O_DIRECT cold reads: each pread blocks in the kernel (GIL released), so a few
# threads build the queue depth the NVMe needs (~QD8 measured ~2x the single-stream rate on this class
# of disk). Distinct from the refuted "thread the warm memcpys" idea - these are true blocking reads.
_ODIRECT_POOL = None


def _odirect_pool():
    global _ODIRECT_POOL
    if _ODIRECT_POOL is None:
        from concurrent.futures import ThreadPoolExecutor

        _ODIRECT_POOL = ThreadPoolExecutor(
            max_workers=6, thread_name_prefix="pe-odirect"
        )
    return _ODIRECT_POOL


def _stage_pin_buf(name: str, k: int, row_shape, dtype) -> torch.Tensor:
    key = (name, tuple(row_shape), dtype)
    buf = _STAGE_PIN.get(key)
    if buf is None or buf.shape[0] < k:
        # _pinned_empty: page-aligned base (cudaHostAlloc), which O_DIRECT reads into rows require
        buf = _pinned_empty((k, *row_shape), dtype)
        _STAGE_PIN[key] = buf
    return buf


def _aligned_empty(nrows: int, row_shape, dtype, align: int = 4096) -> torch.Tensor:
    """Non-pinned, ``align``-aligned CPU buffer (default page-aligned) — for O_DIRECT reads or in-place
    registration, which require memory ALIGNMENT but not page-locking, so allocating many of these does
    not draw on the OS page-lock ceiling. Over-allocate by ``align`` and slice to the aligned offset;
    the slice keeps the base storage alive. (No consumer on the hot path today; utility for O_DIRECT /
    cudaHostRegister experiments — see RAM_REGIME_IMPROVEMENT_MAP I1.)"""
    nbytes = nrows * int(math.prod(tuple(row_shape))) * dtype.itemsize
    raw = torch.empty(nbytes + align, dtype=torch.uint8)  # anonymous, non-pinned
    off = (-raw.data_ptr()) % align
    return raw[off : off + nbytes].view(dtype).view(nrows, *row_shape)


# packed-quant scaffolding the fused-MoE kernel never reads on the paged path, plus the nvfp4
# per-expert scalar scales (global/alpha/input-quant): 4-8 B each -> too small for the pinned gather
# (sub-8/16-byte rows). The four runtime nvfp4 scalars (g*_alphas, w*_input_scale_quant) are refreshed
# per-step into the K slots from a resident full-E table (see forward._gemm_hidden); the rest are dead
# after the nvfp4 method's process_weights_after_loading.
#
# NB: ``_g_idx`` / ``_g_idx_sort_indices`` are NOT excluded — for act-order (gptq desc_act=True /
# ct actorder='group') they are live per-expert [in] int32 tensors the marlin kernel reads at gather
# time, so they must page into slots alongside the qweight. For non-act-order the method's
# process_weights_after_loading replaces them with empty tensors, which discover_paged_params drops
# via its ``numel() > 0`` check — so leaving them pageable is safe for both.
# gpt-oss (mxfp4) adds per-expert SwiGLU scalars — swiglu_alpha/beta/limit, each a CONSTANT broadcast
# (full((E,), 1.702/1.0/7.0), not per-expert-varying and absent from the checkpoint). At 4 B/expert they
# are sub-8-byte and would fail the pinned-gather alignment gate; excluding them from paging is correct
# (the layer's [K] copy keeps the constant for every slot regardless of which expert is resident), and it
# keeps gpt-oss on the fast pinned + captured path instead of forcing --paged-experts-store paged (eager).
_NONPAGED_SUFFIXES = (
    "_weight_shape",
    "_global_scale",
    "_scale_2",
    "_alphas",
    "_scale_quant",
    "_input_scale",  # ModelOpt nvfp4 per-expert input scalar (compressed-tensors names it
    # ..._input_global_scale, already covered by _global_scale); rides the deferred resident-scalar path.
    "swiglu_alpha",
    "swiglu_beta",
    "swiglu_limit",
)


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
            shape = (self.E, *p.shape[1:])
            self.host[name] = (
                _pinned_empty(shape, p.dtype)
                if self.pinned
                else torch.empty(shape, dtype=p.dtype, device="cpu", pin_memory=False)
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
    def page_in(
        self,
        src_experts: torch.Tensor,
        dst_slots: torch.Tensor,
        *,
        stage_bank: int = 0,
        async_h2d: bool = False,
        src_host: Optional[list] = None,
        dst_host: Optional[list] = None,
    ) -> None:
        """Copy ``host[src_experts[i]] -> gpu[dst_slots[i]]`` for every paged tensor.

        ``src_experts`` / ``dst_slots`` are device ``int64`` index tensors from the pager's decision; a
        no-op for an empty plan. ``stage_bank`` selects an independent staging-buffer set, ``async_h2d``
        skips any trailing stream sync, and ``src_host`` passes the plan as a host list so the store
        needs no D2H read-back — the double-buffered wave path's knobs (the caller sequences
        buffer/slot reuse with events); stores without staging may ignore what they don't use.
        """

    def read_full(
        self, targets: Dict[str, torch.Tensor], *, stage_key: int = 0
    ) -> None:
        """Copy the ENTIRE store (all E experts, expert order) into ``targets[name]`` (``[E, *slot]``
        device tensors) — the streaming-prefill scratch fill. One contiguous async H2D per tensor for
        the single-buffer stores; the windowed store overrides with its hot/cold split. ``stage_key``
        selects an independent staging-buffer set where staging is used (the caller sequences reuse
        with events)."""
        for name, host in self.host.items():
            targets[name].copy_(host, non_blocking=self.pinned)

    # --- checkpoint-fill accessors (store-layout-agnostic; used by ``pager.setup_pager``) ---
    # A single ``[E, *]`` host buffer here; ``WindowedExpertStore`` overrides both to route an expert into
    # its hot/cold tier, so the fill code never special-cases the store layout.
    def row(self, name: str, e: int) -> torch.Tensor:
        """Writable host slice backing expert ``e`` for paged tensor ``name`` (per-expert fill)."""
        return self.host[name][e]

    def fill_tensor(self, name: str, full: torch.Tensor) -> None:
        """Fill the whole host backing for ``name`` from a contiguous ``[E, *slot_shape]`` CPU tensor."""
        self.host[name].copy_(full)


    def gather_rows_into(self, name: str, ids, buf: torch.Tensor) -> None:
        """Copy rows ``ids`` of paged tensor ``name`` into ``buf[:len(ids)]``. Default: per-row copies via
        :meth:`row`. Stores with a faster bulk path (disk pread, mmap) override it."""
        for i, e in enumerate(ids):
            buf[i].copy_(self.row(name, int(e)))

class PinnedExpertStore(ExpertStore):
    """Pinned (page-locked) host store, paged with sglang's existing ``transfer_kv_per_layer_mla`` block
    copy — pinned-host -> device, indices read on-device, dynamic count, capture-safe. The fast default.
    """

    pinned = True

    def page_in(
        self,
        src_experts: torch.Tensor,
        dst_slots: torch.Tensor,
        *,
        stage_bank: int = 0,
        async_h2d: bool = False,
        src_host: Optional[list] = None,
        dst_host: Optional[list] = None,
    ) -> None:
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

    def page_in(
        self,
        src_experts: torch.Tensor,
        dst_slots: torch.Tensor,
        *,
        stage_bank: int = 0,
        async_h2d: bool = False,
        src_host: Optional[list] = None,
        dst_host: Optional[list] = None,
    ) -> None:
        if src_experts.numel() == 0:
            return
        src_cpu = (
            torch.tensor(src_host, dtype=torch.int64)
            if src_host is not None
            else src_experts.to("cpu")
        )
        for name, gpu_param in self.gpu.items():
            rows = self.host[name].index_select(0, src_cpu).to(gpu_param.device)
            dst = gpu_param.data
            # torch has no index_copy_cuda for the 1-byte fp8 scale dtypes (mxfp4 block scales are
            # float8_e8m0fnu; fp8 weights e4m3fn) — the copy is bytewise, so route those through an
            # equal-width uint8 view. Standard int/bf16 params take the direct path.
            if dst.element_size() == 1 and dst.is_floating_point():
                dst.view(torch.uint8).index_copy_(0, dst_slots, rows.view(torch.uint8))
            else:
                dst.index_copy_(0, dst_slots, rows)


class RegisteredExpertStore(PinnedExpertStore):
    """Zero-copy pinned store: the host backing is the persisted repack cache (v2 raw per-tensor files)
    ``mmap``'d and ``cudaHostRegister(Mapped|ReadOnly)``-pinned IN PLACE — no ``cudaHostAlloc``, no fill
    copy. Boot becomes mmap + register (~0.3 s/GB measured), host RAM holds ONE file-backed copy, and the
    ReadOnly registration pins past the alloc ceiling (measured 31.3 vs ~20 GB — see
    RAM_REGIME_IMPROVEMENT_MAP I1 / register_store_probe.py).

    Mechanics: the mapping is MAP_SHARED writable (``ACCESS_WRITE``, fd ``O_RDWR``) and registration passes
    ``cudaHostRegisterReadOnly``. MAP_SHARED is deliberate — ``register_mode_repro.py`` measured that a
    MAP_PRIVATE (``ACCESS_COPY``) mmap HANGS in ``cudaHostRegister`` on WSL2/Blackwell (pre-faulting the
    pages does not help — the private-COW mapping's get_user_pages path is the culprit), while MAP_SHARED
    (read or write) registers in ~0.3 s and UVA-reads bit-exact. Writable (not ``ACCESS_READ``) so
    ``torch.frombuffer`` accepts the buffer without its read-only rejection. CONTRACT: the store is
    READ-ONLY after construction — because the mapping is MAP_SHARED, a stray host write would write THROUGH
    to the cache file (not just desync), so the fill accessors (``row`` / ``fill_tensor``) hard-raise and
    full-pin serving never writes the store (the windowed re-pin does, so this class is full-pin only,
    ``make``-time guarded). ``transfer_kv`` / UVA gather work unchanged: registered memory is page-locked +
    device-mapped, and the per-file mmap base is page-aligned (16-byte gather alignment holds when
    ``item_bytes`` does)."""

    pinned = True

    def __init__(self, layer, num_experts_E: int, num_resident_K: int, device, *, cache_layer_dir: str):
        import ctypes
        import json

        from sglang.srt.layers.moe.paged_experts.method import _cudart_handle

        self.E = num_experts_E
        self.K = num_resident_K
        self.device = device
        self.gpu = discover_paged_params(layer, num_resident_K)
        assert self.gpu, "no per-expert params found on layer"
        self.host: Dict[str, torch.Tensor] = {}
        self.item_bytes: Dict[str, int] = {}
        self._regions: list = []  # (mmap, addr, reg_size) — keep-alive + unregister handles
        rt = _cudart_handle()
        if rt is None:
            raise RuntimeError("[paged-experts] mmap store: CUDA runtime unavailable for cudaHostRegister")
        manifest = json.load(open(os.path.join(cache_layer_dir, "manifest.json")))
        HOST_REGISTER_MAPPED, HOST_REGISTER_READONLY = 0x2, 0x8
        page = mmap.PAGESIZE
        try:
            for name, p in self.gpu.items():
                ent = manifest.get(name)
                shape = (self.E, *p.shape[1:])
                if (
                    ent is None
                    or tuple(ent["shape"]) != shape
                    or ent["dtype"] != str(p.dtype)
                ):
                    raise RuntimeError(
                        f"[paged-experts] mmap store: cache manifest mismatch for {name!r} "
                        f"(want {shape} {p.dtype}, have {ent})"
                    )
                path = os.path.join(cache_layer_dir, f"{name}.bin")
                nbytes = int(ent["nbytes"])
                if os.path.getsize(path) != nbytes:
                    raise RuntimeError(f"[paged-experts] mmap store: torn cache file {path}")
                # MAP_SHARED writable (O_RDWR + ACCESS_WRITE): MAP_PRIVATE hangs cudaHostRegister on WSL2
                # (register_mode_repro.py). Read-only by contract (below) — we never write these pages.
                fd = os.open(path, os.O_RDWR)
                try:
                    mm = mmap.mmap(fd, nbytes, access=mmap.ACCESS_WRITE)
                finally:
                    os.close(fd)
                addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
                reg_size = ((nbytes + page - 1) // page) * page  # mmap maps whole pages; register them all
                rc = rt.cudaHostRegister(
                    ctypes.c_void_p(addr),
                    ctypes.c_size_t(reg_size),
                    HOST_REGISTER_MAPPED | HOST_REGISTER_READONLY,
                )
                if rc != 0:
                    mm.close()
                    raise RuntimeError(
                        f"[paged-experts] cudaHostRegister({reg_size / 1e9:.2f}GB) failed rc={rc} for "
                        f"{name!r} — register ceiling reached; fall back to --paged-experts-store pinned"
                    )
                self._regions.append((mm, addr, reg_size))
                t = torch.frombuffer(mm, dtype=p.dtype, count=self.E * p[0].numel()).view(shape)
                self.host[name] = t
                self.item_bytes[name] = p[0].numel() * p.element_size()
                if self.item_bytes[name] % 8 != 0:
                    raise RuntimeError(
                        f"[paged-experts] paged tensor {name!r} per-expert size {self.item_bytes[name]} B "
                        "is not 8-byte aligned (transfer_kv requirement)."
                    )
        except Exception:
            self.release()
            raise

    def release(self) -> None:
        """Unregister + unmap all regions (weight reload replaces the store; the pin budget must return)."""
        import ctypes

        from sglang.srt.layers.moe.paged_experts.method import _cudart_handle

        rt = _cudart_handle()
        for mm, addr, _sz in self._regions:
            try:
                if rt is not None:
                    rt.cudaHostUnregister(ctypes.c_void_p(addr))
            except Exception:
                pass
        # NOTE: the mmaps stay open while any host tensor still views them (torch.frombuffer holds the
        # buffer); dropping our refs lets GC close them once the tensors go.
        self._regions.clear()
        self.host = {}

    def __del__(self):  # best-effort; release() is the real path (weight reload)
        try:
            self.release()
        except Exception:
            pass

    # The store is READ-ONLY by contract (see class docstring): the fill accessors must never run.
    def row(self, name: str, e: int) -> torch.Tensor:
        raise RuntimeError("[paged-experts] mmap store is read-only (loaded from the repack cache)")

    def fill_tensor(self, name: str, full: torch.Tensor) -> None:
        raise RuntimeError("[paged-experts] mmap store is read-only (loaded from the repack cache)")

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
        self._cold_mm: Dict[str, mmap.mmap] = (
            {}
        )  # disk tier mmap objects, for madvise read-ahead hints
        self._cold_fd: Dict[str, Optional[int]] = (
            {}
        )  # disk tier O_DIRECT fds (None = mmap only)
        on_disk = cold_backing == "disk"
        for name, p in self.gpu.items():
            self.host_hot[name] = _pinned_empty((self.W, *p.shape[1:]), p.dtype)
            cold_dims = (self.E - self.W, *p.shape[1:])
            # disk: a >RAM cold tier mmap'd to a file (page-cache-bounded) so the store can exceed RAM;
            # ram: a plain pageable tensor (the cold tier must fit RAM).
            if on_disk:
                (
                    self.host_cold[name],
                    self._cold_mm[name],
                    self._cold_fd[name],
                ) = _alloc_disk_mmap(cold_dir, cold_dims, p.dtype)
            else:
                self.host_cold[name] = torch.empty(
                    cold_dims, dtype=p.dtype, device="cpu", pin_memory=False
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

    def cold_direct_all(self) -> bool:
        """True when every paged tensor's cold rows can be read O_DIRECT (fd available, block-aligned
        rows) — the page-in paths then bypass the page cache, and WILLNEED read-ahead would only pull
        pages nobody will fault."""
        if not self._cold_fd:
            return False
        return all(
            self._cold_fd.get(name) is not None and self.item_bytes[name] % 4096 == 0
            for name in self.gpu
        )

    def _read_cold_rows_direct(self, name: str, cold_rows, buf: torch.Tensor) -> bool:
        """O_DIRECT preads of the cold rows straight into the pinned staging rows ``buf[0..n)`` —
        page-cache bypass (no fault storm, no double copy), queue depth from a small thread pool
        (preads block in the kernel and release the GIL). Returns False on any precondition/IO failure;
        the caller falls back to the mmap copy loop."""
        import ctypes

        fd = self._cold_fd.get(name)
        nbytes = self.item_bytes[name]
        if fd is None or nbytes % 4096 or buf.data_ptr() % 4096:
            return False
        stride = buf[0].numel() * buf.element_size()
        if stride != nbytes or not buf.is_contiguous():
            return False
        base = buf.data_ptr()

        def _rd(i_r):
            i, r = i_r
            mv = (ctypes.c_char * nbytes).from_address(base + i * stride)
            return os.preadv(fd, [mv], r * nbytes) == nbytes

        try:
            if not all(_odirect_pool().map(_rd, list(enumerate(cold_rows)))):
                raise OSError("short read")
            return True
        except OSError as e:
            logger.warning(
                "[paged-experts] O_DIRECT cold read failed for %s (%s) — mmap fallback",
                name,
                e,
            )
            self._cold_fd[name] = None
            return False

    def prefetch_cold(self, experts, force: bool = False) -> None:
        """Issue MADV_WILLNEED for the disk-mmap rows of ``experts`` so the kernel does parallel async
        read-ahead (high queue depth) instead of the serial one-page-fault-at-a-time the gather would do.
        No-op unless the cold tier is disk-backed. madvise needs a page-aligned start, so we round the row
        offset down to a page and extend the length to cover the row."""
        # O_DIRECT page-ins never fault, so read-ahead would pull pages nobody reads — but callers
        # that still read via the mmap (decode refills, window re-pins) pass force=True
        if not self._cold_mm or (self.cold_direct_all() and not force):
            return
        page = mmap.PAGESIZE
        # Hoist row resolution and coalesce adjacent/overlapping page ranges into one madvise each:
        # small rows (fp8 block scales) share pages, and large sorted batches otherwise cost thousands
        # of syscalls per step in the wave regime.
        rows = sorted(r for r in (int(self.cold_pos[e]) for e in experts) if r >= 0)
        if not rows:
            return
        for name, mm in self._cold_mm.items():
            stride = self.item_bytes[name]
            size = len(mm)
            m_start = m_end = None
            for r in rows:
                off = r * stride
                start = (off // page) * page
                end = min(off + stride, size)
                if m_end is not None and start <= m_end:
                    m_end = max(m_end, end)
                    continue
                if m_end is not None:
                    try:
                        mm.madvise(mmap.MADV_WILLNEED, m_start, m_end - m_start)
                    except (OSError, ValueError):
                        pass  # best-effort hint; never fatal
                m_start, m_end = start, end
            if m_end is not None:
                try:
                    mm.madvise(mmap.MADV_WILLNEED, m_start, m_end - m_start)
                except (OSError, ValueError):
                    pass

    def prefetch_cold_all(self) -> None:
        """Issue MADV_WILLNEED over the ENTIRE cold tier (one call per mmap). Used by the wave path to
        read the NEXT layer's cold file ahead while the current layer transfers/computes: at wave-regime
        batch sizes the distinct set saturates toward E, so the next layer's whole cold tier is a
        predictable read — one syscall queues it at full depth. No-op unless disk-backed; best-effort.
        Gated on memory pressure: read-ahead larger than the available page cache would evict the pages
        the CURRENT layer is reading (IO amplification on true >RAM stores).
        """
        if not self._cold_mm:
            return
        total = sum(len(m) for m in self._cold_mm.values())
        avail = _host_available_bytes()
        if avail and total > avail // 2:
            return
        for mm in self._cold_mm.values():
            try:
                mm.madvise(mmap.MADV_WILLNEED, 0, len(mm))
            except (OSError, ValueError):
                pass

    def read_full(
        self, targets: Dict[str, torch.Tensor], *, stage_key: int = 0
    ) -> None:
        """Windowed read-all for the streaming-prefill scratch fill: hot rows via ``transfer_kv`` from
        the pinned window (expert-ordered destinations), cold rows staged through a pinned buffer.
        Issued on the caller's current stream; no trailing sync (the caller sequences with events).
        """
        hot_mask = self.hot_pos >= 0
        hot_experts = torch.nonzero(hot_mask, as_tuple=False).flatten()
        dev = next(iter(targets.values())).device
        if hot_experts.numel():
            from sgl_kernel import transfer_kv_per_layer_mla

            src_rows = self.hot_pos[hot_experts].to(dev)
            dst_rows = hot_experts.to(dev)
            for name in self.gpu:
                transfer_kv_per_layer_mla(
                    src=self.host_hot[name],
                    dst=targets[name],
                    src_indices=src_rows,
                    dst_indices=dst_rows,
                    item_size=self.item_bytes[name],
                )
        cold_experts = [int(e) for e in torch.nonzero(~hot_mask).flatten().tolist()]
        if cold_experts:
            if not getattr(self, "_step_prefetched", False):
                self.prefetch_cold(cold_experts)
            cold_rows = [int(self.cold_pos[e]) for e in cold_experts]
            n = len(cold_rows)
            for name in self.gpu:
                buf = _stage_pin_buf(
                    f"{name}#rf{stage_key}",
                    n,
                    self.host_hot[name].shape[1:],
                    self.host_hot[name].dtype,
                )
                if not self._read_cold_rows_direct(name, cold_rows, buf):
                    for i, r in enumerate(cold_rows):
                        buf[i].copy_(self.host_cold[name][r])
                for i, e in enumerate(cold_experts):
                    targets[name][e].copy_(buf[i], non_blocking=True)

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
        ``[0, W)`` — the P3 freq-ranked window. Runs once, out-of-graph, after a short profiling period;
        the GPU slots keep their (expert-indexed) data unchanged, so only the page-in *source* tier moves.

        Δ-SET: only the experts that actually CHANGE tier move — each promoted expert (cold -> hot) swaps
        rows with a demoted one (hot -> cold); everything else stays in place (``hot_pos``/``cold_pos`` map
        expert -> row, so row order within a tier is free). The previous full-store rewrite read+wrote
        every expert row per tensor — on a disk cold tier that meant re-reading AND dirtying the entire
        cold file (a multi-second-to-minutes stall on one token, plus page-cache eviction); the Δ set is
        typically a small fraction of E.
        """
        hot = [int(e) for e in list(hot_experts)[: self.W]]
        if len(set(hot)) != len(hot):
            raise RuntimeError(
                "[paged-experts] window membership has duplicate experts — would corrupt the hot tier."
            )
        hot_set = set(hot)
        old_hot = set(e for e in range(self.E) if int(self.hot_pos[e]) >= 0)
        promoted = [e for e in hot if e not in old_hot]  # cold -> hot
        demoted = [e for e in old_hot if e not in hot_set]  # hot -> cold
        if len(promoted) != len(demoted):
            raise RuntimeError(
                "[paged-experts] window size W is fixed; tier moves must pair up "
                f"(promoted={len(promoted)} != demoted={len(demoted)})."
            )
        if not promoted:
            return  # membership unchanged
        # Disk cold tier: queue read-ahead for the promoted rows so the swap below faults them in
        # parallel (force: the swap reads via the mmap even when page-ins go O_DIRECT).
        self.prefetch_cold(promoted, force=True)
        pairs = [
            (p, d, int(self.hot_pos[d]), int(self.cold_pos[p]))
            for p, d in zip(promoted, demoted)
        ]
        for name in self.gpu:
            hh, hc = self.host_hot[name], self.host_cold[name]
            tmp = torch.empty_like(hh[0])
            for _p, _d, hot_row, cold_row in pairs:
                tmp.copy_(hh[hot_row])  # save the demoted expert's data
                hh[hot_row].copy_(hc[cold_row])  # promoted: cold row -> freed hot row
                hc[cold_row].copy_(
                    tmp
                )  # demoted: -> the promoted expert's old cold row
        for p, d, hot_row, cold_row in pairs:
            self.hot_pos[p] = hot_row
            self.hot_pos[d] = -1
            self.cold_pos[d] = cold_row
            self.cold_pos[p] = -1

    def page_in(
        self,
        src_experts: torch.Tensor,
        dst_slots: torch.Tensor,
        *,
        stage_bank: int = 0,
        async_h2d: bool = False,
        src_host: Optional[list] = None,
        dst_host: Optional[list] = None,
    ) -> None:
        if src_experts.numel() == 0:
            return
        # the wave path already holds the plan as a host list — reading it back off the device would
        # stall the CPU on the stream (a D2H sync per wave, fatal to the double-buffered overlap)
        src_cpu = (
            torch.tensor(src_host, dtype=torch.int64)
            if src_host is not None
            else src_experts.to("cpu")
        )
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
        # cold experts -> staged copy from the pageable/disk tail, remapped to host_cold rows
        cold_mask = ~hot_mask
        if bool(cold_mask.any()):
            cold_ids = [int(e) for e in src_cpu[cold_mask].tolist()]
            # Disk cold tier: read ahead the cold group in parallel (MADV_WILLNEED) before the copies
            # below fault — unless the wave path already prefetched the WHOLE step's set upfront
            # (avoids re-issuing the same ranges once per wave).
            if not getattr(self, "_step_prefetched", False):
                self.prefetch_cold(cold_ids)
            cold_rows = self.cold_pos[src_cpu[cold_mask]].tolist()
            # #11: take the cold destinations from the caller's host plan when it passed one (the eager
            # keep-warm and wave paths both have dst as a host list already), avoiding a per-layer
            # ``dst_slots[...].tolist()`` D2H drain. Fall back to the device read-back otherwise.
            if dst_host is not None:
                cm = cold_mask.tolist()
                cold_dst = [d for d, m in zip(dst_host, cm) if m]
            else:
                cold_dst = dst_slots[cold_mask.to(dst_slots.device)].tolist()
            n = len(cold_rows)
            # Gather into PINNED buffers, then direct async H2D per slot: the old
            # index_select -> pageable .to() -> index_copy_ chain crossed the bytes through a pageable
            # temp AND copied device->device again. ``stage_bank`` keys an independent buffer set so the
            # double-buffered wave path can gather wave w+1 while wave w's H2D is still in flight.
            for name, gpu_param in self.gpu.items():
                buf = _stage_pin_buf(
                    f"{name}#b{stage_bank}" if stage_bank else name,
                    max(self.K, n),
                    gpu_param.shape[1:],
                    gpu_param.dtype,
                )
                if not self._read_cold_rows_direct(name, cold_rows, buf):
                    for i, r in enumerate(cold_rows):
                        buf[i].copy_(self.host_cold[name][r])
                for i, s in enumerate(cold_dst):
                    gpu_param.data[s].copy_(buf[i], non_blocking=True)
            if not async_h2d:
                # the shared pinned bufs must not be reused (next layer / next wave) while H2D is in
                # flight; the async caller sequences reuse with events instead
                torch.cuda.current_stream().synchronize()


_SWAP_DISK_LOGGED = False  # one-line boot log for the swap+disk cold tier (once across all layers)

class SwapExpertStore(ExpertStore):
    """Single-copy SWAP store (``--paged-experts-store swap``; eager keep-warm path, ``--disable-cuda-graph``).

    The K experts resident in the VRAM pool carry NO redundant host copy — the host holds only the E-K
    NON-resident experts (+1 spare row for the double-buffered swap), so the host footprint is (E-K+1) rows
    instead of E. Paging is a bidirectional SWAP: promote B (host -> pool slot) while DEMOTING that slot's
    occupant A (pool slot -> host), reclaiming the dynamic pool's RAM redundancy.

    Invariant: exactly (E-K) experts live in host at all times (1 spare row free). ``slot_expert[s]`` = expert
    in slot s; ``expert_slot[e]`` = slot of a pool-resident expert; ``expert_row[e]`` = host row of a
    host-resident expert; ``free_rows`` = the spare host row. The store self-tracks residency so it drops into
    the eager ``decide_keep_warm -> page_in`` flow unchanged; ``reconcile_pager`` re-syncs the pager after a
    wave sequence (which routes experts through slots without touching the pager maps).

    Read-only weights -> the demote is a plain D2H of bytes the normal store would discard; on full-duplex
    PCIe it overlaps the H2D promote (v1 here is correct-but-serial). Uses ``.copy_`` (no 8-byte alignment
    requirement). Static de-dup is orthogonal and unsupported. Captured/on-device decode NOT supported (the
    UVA gather has no writeback) — the method forces the eager placement when this store is selected.
    """

    pinned = True
    single_copy = True

    def __init__(
        self,
        layer,
        num_experts_E: int,
        num_resident_K: int,
        device,
        *,
        cold_backing: str = "ram",
        cold_dir: Optional[str] = None,
        ram_rows: int = 0,
    ):
        self.E = num_experts_E
        self.K = num_resident_K
        self.device = device
        self.gpu = discover_paged_params(layer, num_resident_K)
        assert self.gpu, "no per-expert params found on layer"
        # v2 overlap: B buffer rows (not just 1) let the demote (D2H, copy-out stream) run concurrently with
        # the promote (H2D, current stream), so the writeback drops off the swap's critical path. B>=2 gives
        # a freed row to reuse the next swap without stalling on the previous promote's read (FIFO reuse +
        # per-row release events). RAM is E-K+B rows — still ~E-K (the whole point) at B=2.
        self._spare = max(1, int(os.environ.get("SGLANG_PE_SWAP_BUFFER", "2")))
        self.item_bytes: Dict[str, int] = {
            name: p[0].numel() * p.element_size() for name, p in self.gpu.items()
        }
        self.host: Dict[str, torch.Tensor] = {}
        # --- P5 disk cold tier -----------------------------------------------------------------------
        # cold_backing="disk": the RAM host is no longer the whole E-K non-resident set — it is an R-row
        # LRU CACHE (single-copy: an expert lives in exactly one of {VRAM slot, RAM row, disk}), and the
        # full E-expert store is written once to an mmap'd disk file (read-only backing). Because the
        # weights are immutable, a demote is pure caching — never a mandatory writeback — so on a miss we
        # promote from RAM (hit) or read the checkpoint file (miss) and cache the evicted victim in RAM,
        # spilling the LRU tail to disk. RAM footprint = R rows (tunable), independent of E, so the store
        # serves models whose E-K non-resident set exceeds RAM. cold_backing="ram" keeps the pure
        # single-copy swap (host = E-K+B rows, no disk).
        self._disk = cold_backing == "disk"
        self._cold: Dict[str, torch.Tensor] = {}  # [E, *slot] mmap'd disk backing (disk mode only)
        self._cold_mm: Dict[str, mmap.mmap] = {}
        self._cold_fd: Dict[str, Optional[int]] = {}  # O_DIRECT fds (None = mmap copy only)
        if not self._disk:
            self._rows = self.E - self.K + self._spare  # (E-K) host experts + B buffer rows
            for name, p in self.gpu.items():
                self.host[name] = _pinned_empty((self._rows, *p.shape[1:]), p.dtype)
            # initial layout: expert e in slot e for e<K; expert K+i in host row i.
            self.expert_row: "OrderedDict[int, int]" = OrderedDict(
                (self.K + i, i) for i in range(self.E - self.K)
            )
            self.free_rows = list(range(self.E - self.K, self.E - self.K + self._spare))
        else:
            E_K = self.E - self.K
            R = int(ram_rows) if ram_rows and ram_rows > 0 else self._auto_ram_rows()
            R = max(1, min(R, E_K))  # cache at most the E-K non-resident experts
            self._ram_cap = R
            self._rows = R + self._spare  # R cache rows + B buffer rows
            for name, p in self.gpu.items():
                self.host[name] = _pinned_empty((self._rows, *p.shape[1:]), p.dtype)
                t, mm, fd = _alloc_disk_mmap(cold_dir, (self.E, *p.shape[1:]), p.dtype)
                self._cold[name], self._cold_mm[name], self._cold_fd[name] = t, mm, fd
            # initial cache: experts K..K+R-1 in rows 0..R-1 (most-recent last for the LRU); rest disk-only.
            self.expert_row = OrderedDict((self.K + i, i) for i in range(R))
            self.free_rows = list(range(R, R + self._spare))
            global _SWAP_DISK_LOGGED
            if not _SWAP_DISK_LOGGED:
                _SWAP_DISK_LOGGED = True
                row_mb = sum(self.item_bytes.values()) / 1e6
                logger.info(
                    "[paged-experts] SWAP + disk cold tier: RAM cache R=%d rows x %.1f MB (single-copy) "
                    "+ full %d-expert store mmap'd to disk (O_DIRECT=%s) — RAM footprint independent of E",
                    R, row_mb, self.E, all(v is not None for v in self._cold_fd.values()),
                )
        # residency bookkeeping shared by both modes.
        self.slot_expert = list(range(self.K))  # slot -> expert
        self.expert_slot = {e: e for e in range(self.K)}  # pool-resident expert -> slot
        self._row_release: Dict[int, "torch.cuda.Event"] = {}  # freed row -> event of its last promote-read
        self._d2h_stream = None  # lazy copy-out stream (demotes overlap promotes on full-duplex PCIe)

    def _auto_ram_rows(self) -> int:
        """RAM cache capacity (rows) for the disk tier when not given explicitly. PER-LAYER budget:
        ``SGLANG_PE_SWAP_RAM_GB`` (per MoE layer) if set, else a conservative slice of MemAvailable. Sized
        small forces disk spillover; sized >= E-K degenerates to the pure single-copy swap (no disk)."""
        row_bytes = max(1, sum(self.item_bytes.values()))
        gb = os.environ.get("SGLANG_PE_SWAP_RAM_GB")
        if gb:
            budget = float(gb) * 1e9
        else:
            avail = _host_available_bytes()
            # MemAvailable is a WHOLE-HOST number but this budget is charged PER MoE LAYER, so it must be
            # divided by the layer count: 0.15 * avail per layer on a 48-layer model asks for 7.2x
            # MemAvailable and thrashes the host to death (sshd starves before the loader finishes).
            # Take a quarter of MemAvailable for the WHOLE model and split it evenly across layers.
            nlayers = max(1, _moe_layer_count_hint())
            budget = (0.25 * avail / nlayers) if avail else (2e9 / nlayers)
        return max(1, int(budget // row_bytes))

    def fill_tensor(self, name: str, full: torch.Tensor) -> None:
        self.gpu[name].data[: self.K].copy_(full[: self.K])
        if not self._disk:
            self.host[name][: self.E - self.K].copy_(full[self.K :])
            return
        # disk mode: the FULL E-expert store goes to the mmap'd file (authoritative read-only backing),
        # and the initial R-expert RAM cache mirrors experts K..K+R-1 (rows 0..R-1).
        self._cold[name].copy_(full)
        R = self._ram_cap
        if R:
            self.host[name][:R].copy_(full[self.K : self.K + R])

    def row(self, name: str, e: int) -> torch.Tensor:
        # dynamic: a pool-resident expert lives in its GPU slot; else a RAM cache row, else the disk file.
        s = self.expert_slot.get(e)
        if s is not None:
            return self.gpu[name].data[s]
        r = self.expert_row.get(e)
        if r is not None:
            return self.host[name][r]
        if self._disk:
            return self._cold[name][e]  # disk-only (cold tail): read straight from the mmap
        return self.host[name][self.expert_row[e]]  # non-disk: KeyError is a real bug (invariant broken)

    def _read_cold_into(self, name: str, expert_ids, buf: torch.Tensor) -> None:
        """Read disk experts ``expert_ids`` (by logical id == disk row) into ``buf[0..n)`` — O_DIRECT
        (page-cache bypass, thread-pool queue depth) when the row is 4 KB-aligned and the fd is available,
        else a plain mmap copy loop. ``buf`` rows must be exactly ``item_bytes[name]`` (a staging buffer)."""
        fd = self._cold_fd.get(name)
        nbytes = self.item_bytes[name]
        direct = (
            fd is not None
            and os.environ.get("SGLANG_PAGED_EXPERTS_ODIRECT", "1") != "0"
            and nbytes % 4096 == 0
            and buf.data_ptr() % 4096 == 0
            and buf.is_contiguous()
            and buf[0].numel() * buf.element_size() == nbytes
        )
        if direct:
            import ctypes

            base = buf.data_ptr()

            def _rd(ie):
                i, e = ie
                mv = (ctypes.c_char * nbytes).from_address(base + i * nbytes)
                return os.preadv(fd, [mv], int(e) * nbytes) == nbytes

            try:
                if all(_odirect_pool().map(_rd, list(enumerate(expert_ids)))):
                    return
                raise OSError("short read")
            except OSError as ex:
                logger.warning(
                    "[paged-experts] swap O_DIRECT cold read failed for %s (%s) — mmap fallback", name, ex
                )
                self._cold_fd[name] = None
        cold = self._cold[name]
        for i, e in enumerate(expert_ids):
            buf[i].copy_(cold[int(e)])

    def _ensure_free(self, protect) -> None:
        """Guarantee at least one free RAM row: if none, evict the least-recently-used cache expert (oldest
        in the ``expert_row`` OrderedDict) to disk-only, reclaiming its row. Never evicts an expert in
        ``protect`` (the ones being promoted this batch — their RAM/disk source is already classified). The
        evicted expert's data lives on disk (immutable), so eviction is a pure drop — no writeback."""
        if self.free_rows:
            return
        for e in list(self.expert_row):
            if e not in protect:
                self.free_rows.append(self.expert_row.pop(e))
                return
        raise RuntimeError(
            "[paged-experts] swap+disk: no evictable RAM row — batch distinct experts exceed the cache "
            f"capacity R={self._ram_cap} (raise SGLANG_PE_SWAP_RAM_GB)."
        )

    def _restore_buffer(self) -> None:
        """After a disk batch (whose disk-miss demotes grow the cache into the buffer rows), evict the LRU
        tail until >= ``_spare`` rows are free again — restoring the invariant the all-RAM-hit pipeline
        path relies on (it pops a buffer row per swap) and capping the cache at R = _rows - _spare."""
        while len(self.free_rows) < self._spare and self.expert_row:
            _e, _r = self.expert_row.popitem(last=False)  # LRU oldest -> disk-only
            self.free_rows.append(_r)

    def _page_in_disk(self, promote, slots, names) -> None:
        """Disk-tier page-in (serial, current stream). Each swap: demote the victim A into a RAM cache row
        (D2H, pure caching — A is on disk), promote B from its RAM row (hit) or from a staged O_DIRECT disk
        read (miss); a member already resident in another slot is a slot<->slot D2D swap through a spare
        row. The whole step's disk-miss set is read up front (thread-pool queue depth)."""
        se = self.slot_expert
        dev = self.device
        pairs = [(int(B), int(s)) for B, s in zip(promote, slots) if se[int(s)] != int(B)]
        if not pairs:
            return
        promote_set = {B for B, _ in pairs}
        # PHASE 1 (disk, high QD): stage every disk-miss expert (not in RAM, not resident) into pinned bufs.
        disk_ids = [B for B, _ in pairs if B not in self.expert_row and B not in self.expert_slot]
        staged = {}
        if disk_ids:
            pos = {e: i for i, e in enumerate(disk_ids)}
            for name in names:
                buf = _stage_pin_buf(
                    f"swapdisk#{name}", len(disk_ids), self.gpu[name].shape[1:], self.gpu[name].dtype
                )
                self._read_cold_into(name, disk_ids, buf)
                staged[name] = buf
        # PHASE 2 (GPU, serial on the current stream: demote reads slot s before the promote overwrites it).
        cur = torch.cuda.current_stream(dev)
        for B, s in pairs:
            A = se[s]
            if B in self.expert_slot:
                # slot<->slot: B already resident in slot sB -> D2D swap A<->B through a spare RAM row.
                sB = self.expert_slot[B]
                self._ensure_free(promote_set)
                r = self.free_rows[0]  # borrowed scratch (not consumed; invariant keeps >=1 free)
                for name in names:
                    g, h = self.gpu[name].data, self.host[name]
                    h[r].copy_(g[s], non_blocking=True)  # A: slot s -> spare (D2H)
                    g[s].copy_(g[sB], non_blocking=True)  # B: slot sB -> slot s (D2D)
                    g[sB].copy_(h[r], non_blocking=True)  # A: spare -> slot sB (H2D)
                se[s] = B
                se[sB] = A
                self.expert_slot[B] = s
                self.expert_slot[A] = sB
                continue
            # host<->slot swap: reclaim a row for A's demote (evict LRU if the cache is full).
            self._ensure_free(promote_set)
            r = self.free_rows.pop(0)
            for name in names:
                self.host[name][r].copy_(self.gpu[name].data[s], non_blocking=True)  # demote A (D2H)
            if B in self.expert_row:
                rB = self.expert_row.pop(B)
                for name in names:
                    self.gpu[name].data[s].copy_(self.host[name][rB], non_blocking=True)  # promote B (H2D)
                self.free_rows.append(rB)  # B's freed row replenishes the buffer
            else:
                i = pos[B]
                for name in names:
                    self.gpu[name].data[s].copy_(staged[name][i], non_blocking=True)  # promote B (disk->H2D)
            del self.expert_slot[A]
            self.expert_row[A] = r  # A cached in RAM (most-recent in the LRU)
            se[s] = B
            self.expert_slot[B] = s
        self._restore_buffer()  # re-establish >= _spare free rows for the next all-hit pipeline call
        cur.synchronize()

    def gather_rows_into(self, name: str, ids, buf: torch.Tensor) -> None:
        idx = ids.tolist() if torch.is_tensor(ids) else list(ids)
        for i, e in enumerate(idx):
            buf[i].copy_(self.row(name, int(e)))

    def read_full(self, targets, *, stage_key: int = 0) -> None:
        raise NotImplementedError(
            "SwapExpertStore has no full materialized host store (single-copy); "
            "streaming-prefill scratch is unsupported"
        )

    def _get_d2h_stream(self):
        if self._d2h_stream is None:
            self._d2h_stream = torch.cuda.Stream(self.device)
        return self._d2h_stream

    def page_in(
        self,
        src_experts: torch.Tensor,
        dst_slots: torch.Tensor,
        *,
        stage_bank: int = 0,
        async_h2d: bool = False,
        src_host: Optional[list] = None,
        dst_host: Optional[list] = None,
    ) -> None:
        n = (
            src_experts.numel()
            if hasattr(src_experts, "numel")
            else len(src_experts)
        )
        if n == 0:
            return
        promote = (
            list(src_host)
            if src_host is not None
            else src_experts.detach().cpu().tolist()
        )
        slots = (
            list(dst_host) if dst_host is not None else dst_slots.detach().cpu().tolist()
        )
        names = list(self.gpu)
        se = self.slot_expert
        dev = self.device
        # Pipeline ONLY when every swap is host<->slot (decode keep-warm): the demote (D2H) runs on a
        # copy-out stream while the promote (H2D) runs on the current stream -> full-duplex overlap, the
        # writeback off the critical path. Waves may carry slot<->slot swaps (a group member already
        # resident) -> serial fallback (rare, prefill; correctness over overlap).
        pipeline = all(
            (int(B) in self.expert_row) or (int(B) == se[int(s)])
            for B, s in zip(promote, slots)
        )
        # disk tier: a RAM miss (B on disk) or a slot<->slot swap needs the disk-aware path; an all-RAM-hit
        # step still takes the full-duplex overlap below (RAM rows only, disk untouched).
        if self._disk and not pipeline:
            return self._page_in_disk(promote, slots, names)
        if pipeline:
            cur = torch.cuda.current_stream(dev)
            d2h = self._get_d2h_stream()
            d2h.wait_stream(cur)  # demotes read slots produced by prior compute on cur
            for B, s in zip(promote, slots):
                B, s = int(B), int(s)
                A = se[s]
                if A == B:
                    continue
                rB = self.expert_row.pop(B)
                r = self.free_rows.pop(0)  # FIFO: the oldest freed row (its promote-reader is furthest done)
                rel = self._row_release.pop(r, None)
                with torch.cuda.stream(d2h):
                    if rel is not None:
                        d2h.wait_event(rel)  # don't overwrite r until its last promote finished reading it
                    for name in names:
                        self.host[name][r].copy_(self.gpu[name].data[s], non_blocking=True)  # demote A (D2H)
                    dev_ev = torch.cuda.Event()
                    dev_ev.record(d2h)
                cur.wait_event(dev_ev)  # WAR on slot s: promote overwrites only after A's demote read it
                for name in names:
                    self.gpu[name].data[s].copy_(self.host[name][rB], non_blocking=True)  # promote B (H2D)
                rel_ev = torch.cuda.Event()
                rel_ev.record(cur)  # promote read h[rB]; rB is safe to overwrite only after this
                self.expert_row[A] = r
                del self.expert_slot[A]
                se[s] = B
                self.expert_slot[B] = s
                self.free_rows.append(rB)
                self._row_release[rB] = rel_ev
            cur.wait_stream(d2h)  # fold the copy-out stream back before the GEMM reads the slots
            cur.synchronize()
            return
        # serial fallback (wave path): handles slot<->slot swaps; everything on the current stream.
        for B, s in zip(promote, slots):
            B, s = int(B), int(s)
            A = se[s]
            if A == B:
                continue
            if B in self.expert_row:
                rB = self.expert_row.pop(B)
                r = self.free_rows.pop(0)
                for name in names:
                    g = self.gpu[name].data
                    h = self.host[name]
                    h[r].copy_(g[s], non_blocking=True)  # demote A (slot -> spare, D2H)
                    g[s].copy_(h[rB], non_blocking=True)  # promote B (row -> slot, H2D)
                self.expert_row[A] = r
                del self.expert_slot[A]
                se[s] = B
                self.expert_slot[B] = s
                self.free_rows.append(rB)
            else:
                sB = self.expert_slot[B]
                r = self.free_rows[0]  # borrowed, not consumed (invariant keeps >=1 free)
                for name in names:
                    g = self.gpu[name].data
                    h = self.host[name]
                    h[r].copy_(g[s], non_blocking=True)  # A: slot s -> spare (D2H)
                    g[s].copy_(g[sB], non_blocking=True)  # B: slot sB -> slot s (D2D)
                    g[sB].copy_(h[r], non_blocking=True)  # A: spare -> slot sB (H2D)
                se[s] = B
                se[sB] = A
                self.expert_slot[B] = s
                self.expert_slot[A] = sB
        # serial swaps reused the buffer rows transiently; their release events are stale -> clear.
        self._row_release.clear()
        torch.cuda.current_stream(dev).synchronize()

    def reconcile_pager(self, pager) -> None:
        """After a wave sequence (which routes experts through slots without touching the pager's maps),
        overwrite the pager's persistent residency with this store's true ``slot_expert`` — single-copy
        cannot tolerate a stale map (it would evict a slot the pager thinks is free, destroying the only
        copy, or reload an already-resident expert)."""
        pager.slot_expert = list(self.slot_expert)
        l2g = pager.logical_to_gpu_index
        l2g.fill_(-1)
        for s, e in enumerate(self.slot_expert):
            if e >= 0:
                l2g[e] = s
        pager.logical_to_gpu_index_cuda.copy_(l2g)
        sed = getattr(pager, "_slot_expert_d", None)
        if sed is not None:
            sed.copy_(
                torch.tensor(self.slot_expert, dtype=torch.int32, device=self.device)
            )


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
