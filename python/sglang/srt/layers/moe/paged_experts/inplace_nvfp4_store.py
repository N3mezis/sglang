"""Zero-copy in-place NVFP4 expert store (fixed pinned slab pool + pread).

GLM-5.2-NVFP4 has 427 GB of routed experts — 9x the ~48 GB host RAM, so no host store can hold them. The
packed fp4 bytes in the checkpoint are bit-identical to what the GPU K-slot needs, so this store reads each
resident expert's regions straight from the checkpoint on demand. The decode floor is ~13 GB of experts
moved per token (8 top_k x 75 layers x ~22 MB), so I/O + H2D dominate; the design fights that:

  * FIXED PINNED SLAB POOL: the cache is N uniform page-locked slabs allocated ONCE at init and reused via
    LRU — no per-miss ``cudaHostAlloc`` churn (the earlier killer). A slab holds one expert-layer's
    slot-ready data: [w13(gate|up) packed | w2 packed | w13 swizzled block-scale | w2 swizzled block-scale].
    Weights ``pread`` sequentially straight into slab memory; block scales are swizzled ONCE on fill and
    stored in-slab. A cache HIT is then pure async H2D — no disk, no fault, no swizzle.
  * parallel weight preads (each releases the GIL -> NVMe queue depth); swizzle once per fill.
  * routing has session/domain locality (probed), so the LRU keeps the hot expert subset warm; the bigger
    the pinned budget (``SGLANG_INPLACE_PIN_GB``), the higher the hit rate.

Eager-path only (``--disable-cuda-graph``): implements ``page_in`` + the K-slot ``gpu`` params +
``item_bytes``; ``host`` is None (no materialized host store). ModelOpt NVFP4 per-proj
``{gate,up,down}_proj.{weight,weight_scale,weight_scale_2,input_scale}`` (compressed-tensors names via the
reciprocal normalization). Fused w13 slot = gate rows [:I] + up rows [I:]; w2 = down.
"""

from __future__ import annotations

import json
import os
import struct
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

import torch

from sglang.srt.layers.moe.paged_experts.store import ExpertStore, discover_paged_params

_SHARD_FD: Dict[str, int] = {}
_SHARD_FD_DIRECT: Dict[str, int] = {}
_FD_LOCK = threading.Lock()
_READ_POOL = ThreadPoolExecutor(
    max_workers=int(os.environ.get("SGLANG_INPLACE_READ_WORKERS", "32")),
    thread_name_prefix="inplace-nvfp4-read",
)


def _pin_budget() -> int:
    """Pinned slab-cache budget, auto-sized from LIVE host state. The pool is created lazily at the
    first page-in — after weights, KV and the rest of boot have already claimed their memory — so
    MemAvailable at that moment is the honest budget base: take half of what remains beyond an 8 GB
    host reserve (pinned memory cannot swap; over-pinning thrashes the box and can starve sshd),
    floored at 2 GB so tiny hosts still get a working cache. ``SGLANG_INPLACE_PIN_GB`` remains as an
    explicit override escape hatch."""
    env = os.environ.get("SGLANG_INPLACE_PIN_GB")
    if env not in (None, ""):
        return int(float(env) * 1e9)
    from sglang.srt.layers.moe.paged_experts.store import _host_available_bytes

    avail = _host_available_bytes()
    if not avail:
        return int(8e9)  # /proc/meminfo unreadable: small and safe
    budget = 0.5 * max(0.0, avail - 8e9)
    return int(max(2e9, budget))


# O_DIRECT: DMA weight reads straight past the page cache. The buffered path caps ~2.7 GB/s under RAM
# pressure (page-cache reclaim + a per-read memcpy) while O_DIRECT hit 6.2 GB/s in testing AND stops the
# 427 GB expert stream from thrashing the page cache. Off -> plain buffered pread. Needs 4 KB alignment.
_ODIRECT = os.environ.get("SGLANG_INPLACE_ODIRECT", "1") != "0"
_ALIGN = 4096
_BOUNCE = threading.local()

# Wall-clock stage decomposition (SGLANG_PE_TIMING=1). GPU is idle (14-18% util), so the CPU-side time IS
# the critical path: deliver (disk-read+swizzle wait) vs h2d-launch vs gemm-launch. Logs every ~token.
_PE_TIMING = os.environ.get("SGLANG_PE_TIMING", "0") != "0"
_STAGE_NS = {}
_STAGE_N = [0]

# Cold-read split profiler (PE_PROFILE=1): disk-pread wait vs CPU block-scale swizzle inside the cold
# tier, to size the "repack to pre-swizzled contiguous per-expert" de-scatter lever before building it.
_CR_PROF = os.environ.get("PE_PROFILE", "0") not in ("", "0")
_CR_READ_NS = [0.0]  # weight + small-scale pread (disk) ns
_CR_SWZ_NS = [0.0]  # _cpu_swizzle_blockscale (CPU) ns
_CR_N = [0]  # _read_cold_tensor calls since last log
_CR_LOG_EVERY = 4000

# Lean parallel cold gather (SGLANG_PE_LEAN_GATHER, default on): fan the per-expert cold preads out on
# _READ_POOL for real NVMe queue depth, writing STRAIGHT into the destination rows — unlike the ring
# prefetch (_COLD_STAGE), no extra pinned->pinned staging memcpy. The serial gather_rows_into loop issues
# one blocking pread at a time (QD1 -> ~1 GB/s); fanning out lifts the disk-bound gather toward the
# ~3.2 GB/s O_DIRECT ceiling. Measured: serial gather 1.0 GB/s; the heavy ring path hit 1.44 GB/s but its
# staging overhead made the token slower — this lean path keeps the BW win without that overhead.
_LEAN_GATHER = os.environ.get("SGLANG_PE_LEAN_GATHER", "1") != "0"


def _stage(name, ns):
    _STAGE_NS[name] = _STAGE_NS.get(name, 0) + ns


def _stage_tick(waves_per_log=300):
    # Page-in counter only. Per-forward flush (_stage_flush) is the sole logger so each line = one forward,
    # cleanly aligned to token boundaries (the 300-wave window straddled forwards + mixed prefill/decode).
    _STAGE_N[0] += 1


def _stage_flush(tag):
    """Log the accumulated per-stage wall-clock for ONE forward + clear. Called at each model.forward end,
    tagged with mode+ntok so decode tokens are one clean line (trunk = model_backbone - moe_call;
    wave_loop_other = moe_call - deliver - gemm_launch - h2d_launch; sampling of THIS token lands in the
    next flush, negligible + steady-state-consistent)."""
    if not _STAGE_NS:
        return
    import logging

    bb = _STAGE_NS.get("model_backbone", 0)
    tot = bb + _STAGE_NS.get("sampling", 0) or sum(_STAGE_NS.values()) or 1
    msg = "  ".join(
        f"{k}={v/1e6:.0f}ms/{100*v/tot:.0f}%"
        for k, v in sorted(_STAGE_NS.items(), key=lambda x: -x[1])
    )
    logging.getLogger(__name__).info("[PE-TIMING] %s | %s", tag, msg)
    _STAGE_NS.clear()


# Windowed variant (CUDA-graph-capturable decode): keep a fixed pinned [W,*slot] hot window per layer
# (stable UVA base -> in-graph gather) + in-place checkpoint cold reads. W experts/layer resident hot.
_WINDOWED = os.environ.get("SGLANG_INPLACE_WINDOWED", "0") != "0"
_WINDOW_W = int(os.environ.get("SGLANG_INPLACE_WINDOW_W", "12"))
_MOE_LAYERS_HINT = 75

_DTYPE = {
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn,
    "U8": torch.uint8,
    "I8": torch.int8,
}


def _shard_fd(path: str) -> int:
    real = os.path.realpath(path)
    fd = _SHARD_FD.get(real)
    if fd is None:
        with _FD_LOCK:
            fd = _SHARD_FD.get(real)
            if fd is None:
                fd = os.open(real, os.O_RDONLY)
                _SHARD_FD[real] = fd
    return fd


def _cpu_swizzle_blockscale(scale: torch.Tensor) -> torch.Tensor:
    """swizzle_blockscale (utils.py) on CPU — host-only (no .cuda(), so slab fills use zero device VRAM).

    The swizzle is a pure BYTE permutation (fp8 = 1 byte), but fp8 CPU tensor ops fall back to slow paths
    — profiling showed this at ~23% of decode. Doing the pad/permute/contiguous as uint8 (a plain memcpy)
    is far faster; view back to fp8 at the end. Bit-identical."""
    if scale.ndim == 2:
        scale = scale.unsqueeze(0)
    B, M, K = scale.shape
    ru = lambda x, m: (x + m - 1) // m * m
    Mp, Kp = ru(M, 128), ru(K, 4)
    su = scale.contiguous().view(
        torch.uint8
    )  # fp8 -> uint8 (same bytes); uint8 CPU ops are fast
    padded = torch.zeros((B, Mp, Kp), dtype=torch.uint8)
    padded[:B, :M, :K] = su
    padded = padded.reshape(B, Mp // 128, 4, 32, Kp // 4, 4)
    out = padded.permute((0, 1, 4, 3, 2, 5)).contiguous().reshape(B, Mp, Kp)
    return out.view(scale.dtype)


def _shard_fd_direct(path: str) -> int:
    # Key by the path AS GIVEN — callers pass the already-realpath'd shard (see _index_expert_sources),
    # and os.open follows any symlink itself. The per-read os.path.realpath() here was pure waste (~7% of
    # decode) re-resolving the same symlink on every read.
    fd = _SHARD_FD_DIRECT.get(path)
    if fd is None:
        with _FD_LOCK:
            fd = _SHARD_FD_DIRECT.get(path)
            if fd is None:
                fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
                _SHARD_FD_DIRECT[path] = fd
    return fd


def _bounce_buf(nbytes: int):
    """Per-thread page-aligned bounce buffer (mmap -> 4 KB aligned) for O_DIRECT reads, grown on demand."""
    import mmap

    buf = getattr(_BOUNCE, "buf", None)
    if buf is None or len(buf) < nbytes:
        buf = mmap.mmap(-1, max(nbytes, 1 << 20))  # anonymous mmap is page-aligned
        _BOUNCE.buf = buf
    return buf


def _pread_into(fd: int, offset: int, mv: memoryview):
    """preadv the full ``mv`` window from ``offset`` (loops on short reads)."""
    n = len(mv)
    got = os.preadv(fd, [mv], offset)
    while got < n:
        k = os.preadv(fd, [mv[got:]], offset + got)
        if k <= 0:
            break
        got += k


def _pread_into_direct(path: str, offset: int, mv: memoryview):
    """O_DIRECT read of ``len(mv)`` bytes at ``offset`` into ``mv``. safetensors offsets/sizes are not
    4 KB aligned, so read the aligned superset into a page-aligned bounce buffer, then copy out the exact
    window (an aligned memcpy ~10 GB/s — the win is the DMA disk read, not touching the page cache).
    """
    n = len(mv)
    astart = offset & ~(_ALIGN - 1)
    aend = (offset + n + _ALIGN - 1) & ~(_ALIGN - 1)
    alen = aend - astart
    fd = _shard_fd_direct(path)
    buf = _bounce_buf(alen)
    bmv = memoryview(buf)
    got = os.preadv(fd, [bmv[:alen]], astart)
    while got < alen:
        # O_DIRECT requires aligned length; the file end may be shorter — a short read there is fine.
        k = os.preadv(fd, [bmv[got:alen]], astart + got)
        if k <= 0:
            break
        got += k
    head = offset - astart
    mv[:] = bmv[head : head + n]


class _SlabPool:
    """N uniform pinned slabs (allocated once), LRU-reused across all layers. Key = (layer, expert)."""

    def __init__(self, slab_bytes: int, budget: int):
        from sglang.srt.layers.moe.paged_experts.store import _pinned_empty

        self.slab_bytes = slab_bytes
        self.n = max(1, int(budget // slab_bytes))
        # ONE contiguous pinned block (raw cudaHostAlloc via _pinned_empty), sliced into n slab rows —
        # NOT n separate torch.empty(pin_memory=True) tensors. torch's pinned allocator backs each
        # allocation with mmap(/dev/zero, MAP_SHARED): it shows as non-droppable Shmem that can only SWAP
        # (the thrash source when the pin grows), rounds every slab up (~37% waste: a 24GB budget occupied
        # ~33GB across ~4.5k mappings), and page-locks ~12GB below raw cudaHostAlloc (see _pinned_empty).
        # One raw block fixes all three: exact size, a single mapping, private anon, the higher ceiling.
        self._block = _pinned_empty((self.n, slab_bytes), torch.uint8)
        self.slabs = [
            self._block[i] for i in range(self.n)
        ]  # row views; each is contiguous + pinned
        self.mv = [memoryview(s.numpy()) for s in self.slabs]
        self.free = list(range(self.n))
        self.lru: OrderedDict = OrderedDict()  # key -> slab_idx (most-recent last)
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.inflight = (
            {}
        )  # slot_idx -> list[Future]: a slot with a pending fill is not evictable (its
        # async write would corrupt whoever reused it) and page_in must wait on it. Keyed by SLOT, not
        # (layer,expert), so a prefetched slot that gets evicted/reused can never be mistaken for still-valid.

    def lookup(self, key):
        with self.lock:
            idx = self.lru.get(key)
            if idx is not None:
                self.lru.move_to_end(key)
            return idx

    def acquire(self, key):
        """Reserve a slab for ``key`` (evicting the LRU victim if needed). Returns (idx, need_fill)."""
        with self.lock:
            idx = self.lru.get(key)
            if idx is not None:
                self.lru.move_to_end(key)
                self.hits += 1
                return idx, False
            self.misses += 1
            if self.free:
                idx = self.free.pop()
            else:
                # evict the oldest slot whose fill (if any) has COMPLETED; skip slots with an in-flight
                # fill so a pending prefetch write can't land in a slot now owned by another expert.
                victim = None
                for kk in self.lru:
                    fs = self.inflight.get(self.lru[kk])
                    if not fs or all(f.done() for f in fs):
                        victim = kk
                        break
                if victim is None:
                    victim = next(
                        iter(self.lru)
                    )  # every slot mid-fill (shouldn't happen): take the oldest
                idx = self.lru.pop(victim)
            self.inflight.pop(
                idx, None
            )  # slot reused -> drop any (completed) stale fill record
            self.lru[key] = idx
            if _PE_TIMING and (self.hits + self.misses) % 4800 == 0:
                import logging

                tot = self.hits + self.misses
                logging.getLogger(__name__).info(
                    "[PE-SLAB] n=%d hit=%.1f%% (%d/%d)",
                    self.n,
                    100 * self.hits / tot,
                    self.hits,
                    tot,
                )
            return idx, True

    def note_fill(self, idx, futs):
        """Record a slot's in-flight fill futures (so eviction skips it and page_in can wait on it)."""
        with self.lock:
            self.inflight[idx] = futs

    def take_inflight(self, idx):
        """Pop + return slot ``idx``'s pending fill futures — page_in waits on these on a cache hit so a
        still-in-flight (prefetch) fill is drained before the H2D reads the slab."""
        with self.lock:
            return self.inflight.pop(idx, None)


_POOL: Optional[_SlabPool] = None
_POOL_LOCK = threading.Lock()
_LOGGED = False
_N_STORES = 0  # count of InPlaceNvfp4Store instances = actual MoE layer count (for the cache log)


_MEMCENSUS_DONE = False


def _mem_census(tag: str) -> None:
    """One-shot diagnostic (SGLANG_PE_MEMCENSUS=1): census the process's live CPU-side torch tensors to
    identify what is holding host RAM (the ~27GB load-resident anon). Dedups by storage data_ptr so views
    don't double-count. Writes to the HF-cache mount so the host can read it. Runs once, best-effort.
    """
    global _MEMCENSUS_DONE
    if _MEMCENSUS_DONE or os.environ.get("SGLANG_PE_MEMCENSUS", "0") == "0":
        return
    _MEMCENSUS_DONE = True
    try:
        import collections
        import gc

        rss = {}
        try:
            for ln in open("/proc/self/status"):
                if ln.split(":")[0] in (
                    "VmRSS",
                    "RssAnon",
                    "RssShmem",
                    "RssFile",
                    "VmSwap",
                    "VmData",
                ):
                    rss[ln.split(":")[0]] = int(ln.split()[1]) / 1048576.0  # GB
        except Exception:
            pass

        seen = set()
        by_kind = collections.defaultdict(
            lambda: [0, 0]
        )  # (dtype,ndim,pinned) -> [count, bytes]
        tops = []
        total = 0
        n_tensors = 0
        for obj in gc.get_objects():
            try:
                if not isinstance(obj, torch.Tensor) or obj.device.type != "cpu":
                    continue
                n_tensors += 1
                try:
                    st = obj.untyped_storage()
                except Exception:
                    st = obj.storage()
                ptr = st.data_ptr()
                if ptr in seen:
                    continue
                seen.add(ptr)
                nb = (
                    st.nbytes()
                    if hasattr(st, "nbytes")
                    else st.size() * obj.element_size()
                )
                total += nb
                try:
                    pinned = obj.is_pinned()
                except Exception:
                    pinned = False
                by_kind[(str(obj.dtype), obj.dim(), pinned)][0] += 1
                by_kind[(str(obj.dtype), obj.dim(), pinned)][1] += nb
                tops.append((nb, str(obj.dtype), tuple(obj.shape), pinned))
            except Exception:
                continue
        # referrer walk: for the biggest storages, name the module/attr that holds them
        big = []  # (nbytes, tensor)
        for obj in gc.get_objects():
            try:
                if not isinstance(obj, torch.Tensor) or obj.device.type != "cpu":
                    continue
                st = (
                    obj.untyped_storage()
                    if hasattr(obj, "untyped_storage")
                    else obj.storage()
                )
                nb = (
                    st.nbytes()
                    if hasattr(st, "nbytes")
                    else st.size() * obj.element_size()
                )
                if nb > 400_000_000:  # >0.4 GB
                    big.append((nb, obj))
            except Exception:
                continue
        big.sort(key=lambda x: -x[0])
        owner_lines = []
        _self_ids = {id(big), id(owner_lines), id(tops), id(by_kind)}
        for nb, t in big[:10]:
            desc = []
            try:
                for r in gc.get_referrers(t):
                    if id(r) in _self_ids or r is big:
                        continue
                    if isinstance(r, dict):
                        keys = [k for k, v in r.items() if v is t]
                        for owner in gc.get_referrers(r)[:2]:
                            on = type(owner).__name__
                            if on not in ("frame", "list", "tuple"):
                                desc.append("%s.%s" % (on, keys[0] if keys else "?"))
                    elif isinstance(r, (list, tuple)):
                        for owner in gc.get_referrers(r)[:2]:
                            on = type(owner).__name__
                            if on not in ("frame",):
                                desc.append("%s[list]" % on)
                    else:
                        desc.append(type(r).__name__)
                    if len(desc) >= 3:
                        break
            except Exception as ee:
                desc = ["<err %r>" % ee]
            owner_lines.append(
                "  %6.2f GB %-18s held-by: %s"
                % (nb / 1e9, tuple(t.shape), " | ".join(desc[:3]) or "<none>")
            )
        del big

        tops.sort(reverse=True)
        lines = ["=== PE MEM CENSUS (%s) ===" % tag]
        lines.append("RSS: " + "  ".join("%s=%.2fG" % (k, v) for k, v in rss.items()))
        lines.append(
            "live CPU torch tensors: n=%d unique-storages=%d total=%.2f GB"
            % (n_tensors, len(seen), total / 1e9)
        )
        lines.append("--- by (dtype, ndim, pinned): count, GB ---")
        for k, (c, b) in sorted(by_kind.items(), key=lambda x: -x[1][1])[:15]:
            lines.append("  %-40s n=%-5d %.2f GB" % (str(k), c, b / 1e9))
        lines.append("--- top 25 individual storages ---")
        for nb, dt, shp, pin in tops[:25]:
            lines.append("  %6.2f GB  %-10s pinned=%-5s %s" % (nb / 1e9, dt, pin, shp))
        lines.append("--- OWNERS of the >0.4GB storages ---")
        lines.extend(owner_lines)
        out = "/root/.cache/huggingface/_pe_memcensus.txt"
        with open(out, "w") as f:
            f.write("\n".join(lines) + "\n")
        import logging

        logging.getLogger(__name__).info(
            "[paged-experts] mem census written to %s (CPU tensors=%.1f GB)",
            out,
            total / 1e9,
        )
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning("[paged-experts] mem census failed: %r", e)


def _get_pool(slab_bytes: int) -> _SlabPool:
    """Lazily create the global pinned slab pool on FIRST page-in (inference), not at store __init__
    (which runs in process_weights_after_loading, during the load window). Deferring keeps the multi-GB
    pin off the load-time RAM peak — the load completes on the loader footprint, then the pin allocates
    when host RAM is free — so the cache can be sized near the RAM ceiling without load-time thrash.
    """
    global _POOL, _LOGGED
    if _POOL is None:
        with _POOL_LOCK:
            if _POOL is None:
                _mem_census("pre-slab-alloc (first page-in)")
                pool = _SlabPool(slab_bytes, _pin_budget())
                if not _LOGGED:
                    _LOGGED = True
                    import logging

                    layers = max(
                        1, _N_STORES
                    )  # actual MoE-layer count, not the old hardcoded 75
                    logging.getLogger(__name__).info(
                        "[paged-experts] in-place NVFP4 slab cache (lazy, post-load): %d slabs x %.1f MB "
                        "= %.1f GB pinned (~%d experts/layer over %d MoE layers)",
                        pool.n,
                        slab_bytes / 1e6,
                        pool.n * slab_bytes / 1e9,
                        pool.n // layers,
                        layers,
                    )
                _POOL = (
                    pool  # publish only after fully built (readers see a complete pool)
                )
    return _POOL


class _ColdStagePool:
    """Ring Stage 1 (within-step): a GLOBAL bounded async staging tier for the captured windowed store's
    cold-expert reads. One MoE layer is active at a time under the BCG break loop, so a single small pool of
    uniform pinned bufsets — shared across all layer stores, evicted LRU — covers the current layer's
    distinct cold set (≤ top_k ≈ 4 at bs=1). Reads fan out on ``_READ_POOL`` (deep QD), so the layer's cold
    experts read in PARALLEL instead of serially at each wave break. Keyed on ``(layer_idx, expert)`` — a
    hit is always valid (expert weights are immutable), so cross-step reuse is a free bonus. Per-store the
    pool would be W×layers pinned bufsets (~17 GB); global it is ``cap`` bufsets (~cap×19 MB).
    """

    __slots__ = ("cap", "free", "staged")

    def __init__(self, cap: int = 16):
        self.cap = cap
        self.free = (
            []
        )  # list of {name: pinned tensor} bufsets, reusable across layers (uniform MoE dims)
        self.staged = (
            OrderedDict()
        )  # (layer_idx, e) -> (bufs, futs); LRU by insertion/touch order

    def _acquire(self, store) -> dict:
        # Bufsets are interchangeable across uniform MoE layers; drop a mismatched one (defensive) + alloc.
        want = store.host_hot["w13_weight"][0].shape
        while self.free:
            b = self.free.pop()
            if b["w13_weight"].shape == want:
                return b
        return {
            n: torch.empty_like(store.host_hot[n][0], pin_memory=True)
            for n in store._names
        }

    def prefetch(self, store, e: int) -> None:
        """Kick async reads of expert ``e``'s tensors into a staged bufset (idempotent; LRU-evicts to stay
        within ``cap``). Single-threaded caller (the eager BCG break) — no lock needed.
        """
        key = (store.layer_idx, int(e))
        ent = self.staged.get(key)
        if ent is not None:
            self.staged.move_to_end(key)  # LRU touch
            return
        if len(self.staged) >= self.cap:
            _, (obufs, ofuts) = self.staged.popitem(last=False)  # evict oldest
            for f in ofuts.values():
                f.result()  # drain before recycling its buffers (near-instant for a completed old entry)
            self.free.append(obufs)
        bufs = self._acquire(store)
        futs = {
            n: _READ_POOL.submit(store._read_cold_tensor, n, key[1], bufs[n])
            for n in store._names
        }
        self.staged[key] = (bufs, futs)

    def gather_one(self, store, name: str, e: int, dst: torch.Tensor) -> bool:
        """If expert ``e``'s ``name`` tensor is staged, wait for its read + copy into ``dst`` and return True;
        else return False (caller falls back to a synchronous read)."""
        ent = self.staged.get((store.layer_idx, int(e)))
        if ent is None:
            return False
        bufs, futs = ent
        futs[name].result()
        dst.copy_(bufs[name])
        return True


_COLD_STAGE = _ColdStagePool()


class InPlaceNvfp4Store(ExpertStore):
    pinned = False

    def __init__(
        self, layer, num_experts_E, num_resident_K, device, *, model_path, layer_idx
    ):
        global _N_STORES
        _N_STORES += (
            1  # one instance per MoE layer -> real layer count for the cache log
        )
        self.E = num_experts_E
        self.K = num_resident_K
        self.device = device
        self.layer_idx = layer_idx
        self.host = None
        self.gpu = discover_paged_params(layer, num_resident_K)
        assert self.gpu, "no per-expert params found on layer"
        self.item_bytes = {
            n: p[0].numel() * p.element_size() for n, p in self.gpu.items()
        }
        assert (
            self.gpu["w13_weight"].dtype == torch.uint8
        ), "in-place nvfp4 expects uint8 packed weights"
        self._sc_w13 = (
            "w13_blockscale_swizzled"
            if "w13_blockscale_swizzled" in self.gpu
            else "w13_weight_scale"
        )
        self._sc_w2 = (
            "w2_blockscale_swizzled"
            if "w2_blockscale_swizzled" in self.gpu
            else "w2_weight_scale"
        )

        # slot shapes -> slab layout (uniform across MoE layers)
        w13, w2 = self.gpu["w13_weight"], self.gpu["w2_weight"]
        sc13, sc2 = self.gpu[self._sc_w13], self.gpu[self._sc_w2]
        self._w13_shape, self._w2_shape = tuple(w13.shape[1:]), tuple(w2.shape[1:])
        self._sc13_shape, self._sc2_shape = tuple(sc13.shape[1:]), tuple(sc2.shape[1:])
        w13_b = w13[0].numel()  # uint8
        w2_b = w2[0].numel()
        sc13_b = sc13[0].numel() * sc13.element_size()
        sc2_b = sc2[0].numel() * sc2.element_size()
        self._lay = {
            "w13": (0, w13_b),
            "w2": (w13_b, w2_b),
            "sc13": (w13_b + w2_b, sc13_b),
            "sc2": (w13_b + w2_b + sc13_b, sc2_b),
            "total": w13_b + w2_b + sc13_b + sc2_b,
        }
        self._sc13_dt, self._sc2_dt = sc13.dtype, sc2.dtype

        self._index_expert_sources(model_path, layer_idx)
        self.nvfp4_full_e = self._compute_resident_scalars()

        # LAZY slab-pool allocation (deferred to first page-in — see _get_pool). Allocating the
        # multi-GB pinned pool HERE (setup_pager runs in process_weights_after_loading) stacks it on
        # the loader's transient + the checkpoint page cache, blowing the host-RAM peak past physical
        # RAM (measured: pin 24 -> 61GB used + 25GB swap loading the 93GB Laguna checkpoint on a 62GB
        # box). Deferring to first inference lets the load complete on the loader footprint alone; the
        # pin then allocates when RAM is free, so the cache can be sized large without load-time thrash.
        self._pool_slab_bytes = self._lay["total"]

    # ---- source indexing ---------------------------------------------------------------------------
    def _index_expert_sources(self, model_path, layer_idx):
        from sglang.srt.layers.moe.paged_experts.fills.checkpoint import (
            _experts_prefix,
            _proj_names,
            _snapshot_dir,
            _weight_map,
        )

        snap = _snapshot_dir(model_path)
        wmap = _weight_map(snap)
        pre = _experts_prefix(wmap, layer_idx)
        gate, up, down = _proj_names(wmap, pre)
        # SWAP gate<->up in the fused w13: the MoE GEMM silu's w13's FIRST half, so gate must occupy the
        # SECOND half to yield silu(gate)*up. Every fill (weights + swizzled scales) reads self._projs,
        # so swapping the tuple flips them together, consistently. (Matches the _fill_bf16 fix.)
        self._projs = (up, gate, down)
        self._modelopt = f"{pre}0.{gate}.weight_packed" not in wmap
        self._W = "weight" if self._modelopt else "weight_packed"
        self._WGS = "weight_scale_2" if self._modelopt else "weight_global_scale"
        self._IGS = "input_scale" if self._modelopt else "input_global_scale"

        hdr_cache, start_cache = {}, {}

        def hdr(shard):
            if shard not in hdr_cache:
                p = os.path.join(snap, shard)
                with open(os.path.realpath(p), "rb") as f:
                    (hl,) = struct.unpack("<Q", f.read(8))
                    hdr_cache[shard] = json.loads(f.read(hl))
                start_cache[shard] = 8 + hl
            return hdr_cache[shard], start_cache[shard]

        src = {}
        for e in range(self.E):
            pe = {}
            for proj in (gate, up, down):
                d = {}
                for suf in (self._W, "weight_scale", self._WGS, self._IGS):
                    key = f"{pre}{e}.{proj}.{suf}"
                    shard = wmap[key]
                    h, ds = hdr(shard)
                    v = h[key]
                    b0, b1 = v["data_offsets"]
                    p = os.path.join(snap, shard)
                    d[suf] = (
                        _shard_fd(p),
                        ds + b0,
                        tuple(v["shape"]),
                        _DTYPE[v["dtype"]],
                        b1 - b0,
                        os.path.realpath(p),
                    )
                pe[proj] = d
            src[e] = pe
        self._src = src

    @staticmethod
    def _read_small(entry):
        fd, off, shape, dt, nbytes = entry[:5]
        raw = os.pread(fd, nbytes, off)
        t = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(dt)
        return t.reshape(shape) if shape else t

    def _compute_resident_scalars(self):
        gate, up, down = self._projs
        E = self.E
        z = lambda: torch.empty(E, dtype=torch.float32)
        w1_wgs, w2_wgs, w1_igs, w3_igs, w2_igs = z(), z(), z(), z(), z()
        inv = (lambda x: 1.0 / x) if self._modelopt else (lambda x: x)

        def read_one(e):
            # 5 tiny scalar reads for expert e; fanned across the pool so the ~96K-read boot init is
            # QD-deep instead of seek-bound single-threaded.
            s = self._src[e]
            return (
                e,
                inv(self._read_small(s[gate][self._WGS]).flatten()[0].float()),
                inv(self._read_small(s[down][self._WGS]).flatten()[0].float()),
                inv(self._read_small(s[gate][self._IGS]).flatten()[0].float()),
                inv(self._read_small(s[up][self._IGS]).flatten()[0].float()),
                inv(self._read_small(s[down][self._IGS]).flatten()[0].float()),
            )

        for e, a, b, c, d, f in _READ_POOL.map(read_one, range(E)):
            w1_wgs[e], w2_wgs[e], w1_igs[e], w3_igs[e], w2_igs[e] = a, b, c, d, f
        dev = self.device
        w1_wgs, w2_wgs = w1_wgs.to(dev), w2_wgs.to(dev)
        w1_igs, w3_igs, w2_igs = w1_igs.to(dev), w3_igs.to(dev), w2_igs.to(dev)
        w13_ws2, w2_ws2 = 1.0 / w1_wgs, 1.0 / w2_wgs  # per-expert weight_scale_2 [E]
        w13_iq = torch.minimum(
            w1_igs, w3_igs
        )  # per-expert input_scale_quant (1/input_scale) [E]
        # Upstream's NVFP4 fused-MoE (flashinfer_cutlass/trtllm, ModelOptNvFp4FusedMoEMethod) collapses the
        # per-expert input scale to a PER-TENSOR max: input_scale = 1/iq, max_input_scale = 1/min(iq). The
        # kernel quantizes activations with 1/max, so the alphas must use max_input_scale too (else the
        # x-quant and dequant scales disagree). g*_alphas stay per-expert (max_input_scale * per-expert
        # weight_scale_2); *_input_scale_quant become 0-dim per-tensor scalars (1/max).
        w13_is_max = (1.0 / w13_iq).max()
        w2_is_max = (1.0 / w2_igs).max()
        return {
            "g1_alphas": (w13_is_max * w13_ws2).float(),
            "g2_alphas": (w2_is_max * w2_ws2).float(),
            "w13_input_scale_quant": (1.0 / w13_is_max).float().reshape(()),
            "w2_input_scale_quant": (1.0 / w2_is_max).float().reshape(()),
        }

    # ---- slab fill (on cache miss) -----------------------------------------------------------------
    def _submit_weight_reads(self, idx: int, e: int):
        """Submit the 3 packed-weight preads (gate|up -> w13 region, down -> w2 region) straight into the
        slab; return their futures. Fanning ALL misses' reads out at once gives the NVMe deep queue depth
        (QD ~ 3 x misses) — a serial expert-at-a-time loop only reaches QD~3 and leaves the disk ~80% idle.
        """
        mv = _POOL.mv[idx]
        gate, up, down = self._projs
        s = self._src[e]
        w13_off, w13_b = self._lay["w13"]
        w2_off, w2_b = self._lay["w2"]
        half = w13_b // 2
        # each job: (fd, offset, path, dst_mv). O_DIRECT dispatches on path; buffered on fd.
        jobs = [
            (s[gate][self._W], mv[w13_off : w13_off + half]),
            (s[up][self._W], mv[w13_off + half : w13_off + w13_b]),
            (s[down][self._W], mv[w2_off : w2_off + w2_b]),
        ]
        if _ODIRECT:
            return [
                _READ_POOL.submit(_pread_into_direct, j[0][5], j[0][1], j[1])
                for j in jobs
            ]
        return [_READ_POOL.submit(_pread_into, j[0][0], j[0][1], j[1]) for j in jobs]

    def _fill_scales(self, idx: int, e: int):
        """Read the (small) block scales, swizzle ONCE on CPU (zero device VRAM), store into the slab."""
        mv = _POOL.mv[idx]
        gate, up, down = self._projs
        s = self._src[e]
        g = self._read_small(s[gate]["weight_scale"])
        u = self._read_small(s[up]["weight_scale"])
        dn = self._read_small(s[down]["weight_scale"])
        sw13 = _cpu_swizzle_blockscale(torch.cat([g, u], dim=0).unsqueeze(0))[
            0
        ].contiguous()
        sw2 = _cpu_swizzle_blockscale(dn.unsqueeze(0))[0].contiguous()
        sc13_off, sc13_b = self._lay["sc13"]
        sc2_off, sc2_b = self._lay["sc2"]
        torch.frombuffer(mv[sc13_off : sc13_off + sc13_b], dtype=self._sc13_dt).copy_(
            sw13.view(-1)
        )
        torch.frombuffer(mv[sc2_off : sc2_off + sc2_b], dtype=self._sc2_dt).copy_(
            sw2.view(-1)
        )

    def _submit_fill(self, idx: int, e: int):
        """Fan out one expert's slab fill (3 weight preads + 1 scale swizzle) to the pool; return futures."""
        futs = self._submit_weight_reads(idx, e)
        futs.append(_READ_POOL.submit(self._fill_scales, idx, e))
        _POOL.note_fill(
            idx, futs
        )  # pool won't evict this slot until the fill completes; page_in waits on it
        return futs

    # ---- read-ahead: fan out the WHOLE layer's cold reads at once (deep queue depth) ----------------
    def prefetch_cold(self, experts):
        """Issue background reads for all ``experts`` into their slabs NOW (non-blocking). With the K=1
        stream wave loop this is called once per layer with the layer's full distinct set (~top_k experts),
        so all of them read concurrently (QD ~ 4x experts) and saturate the NVMe — instead of the serial
        one-expert-per-wave pattern that only reaches QD~3 and leaves the disk ~80% idle. ``page_in`` then
        waits on the in-flight fill and just does the H2D."""
        if experts is None:
            return
        _get_pool(
            self._pool_slab_bytes
        )  # lazy: allocate the pinned pool on first use (post-load)
        ids = experts.tolist() if torch.is_tensor(experts) else list(experts)
        for e in ids:
            e = int(e)
            idx, need_fill = _POOL.acquire((self.layer_idx, e))
            if need_fill:
                self._submit_fill(
                    idx, e
                )  # records the fill on the pool (slot-keyed, eviction-safe)

    # ---- hot path ----------------------------------------------------------------------------------
    def page_in(
        self,
        src_experts,
        dst_slots,
        *,
        stage_bank=0,
        async_h2d=False,
        src_host=None,
        dst_host=None,
    ):
        if src_experts.numel() == 0:
            return
        _get_pool(
            self._pool_slab_bytes
        )  # lazy: allocate the pinned pool on first use (post-load)
        # Use caller-supplied host lists to AVOID device->host syncs (.to("cpu").tolist() stalls the pipeline
        # ~2x/wave; the eager wave path already has these on the host).
        srcs = src_host if src_host is not None else src_experts.to("cpu").tolist()
        dsts = dst_host if dst_host is not None else dst_slots.to("cpu").tolist()
        w13, w2 = self.gpu["w13_weight"], self.gpu["w2_weight"]
        sc13, sc2 = self.gpu[self._sc_w13], self.gpu[self._sc_w2]
        dev = self.device
        w13_off, w13_b = self._lay["w13"]
        w2_off, w2_b = self._lay["w2"]
        sc13_off, sc13_b = self._lay["sc13"]
        sc2_off, sc2_b = self._lay["sc2"]

        # PHASE 1 (disk, high QD): resolve each expert to a slab. A prefetched expert has an in-flight fill
        # (wait on it); an un-prefetched miss is filled inline NOW (fanned out); a resident one is a hit.
        plan, pending = [], []
        for e, slot in zip(srcs, dsts):
            e, slot = int(e), int(slot)
            idx, need_fill = _POOL.acquire((self.layer_idx, e))
            if need_fill:
                pending.extend(self._submit_fill(idx, e))  # inline miss: fill now
            else:
                fs = _POOL.take_inflight(
                    idx
                )  # hit: a prefetch fill for this slot may still be running
                if fs:
                    pending.extend(fs)
            plan.append((e, slot, idx))
        _t0 = time.perf_counter_ns() if _PE_TIMING else 0
        for f in pending:
            f.result()
        if _PE_TIMING:
            _stage("deliver", time.perf_counter_ns() - _t0)
            _t0 = time.perf_counter_ns()
        # PHASE 3 (GPU): async H2D per expert DIRECTLY from the pinned slab into the slot. copy_(pinned,
        # non_blocking=True) is ONE DMA; the previous `.to(dev)` then `copy_` was TWO transfers (H2D->temp +
        # D2D) — the small-copy overhead was capping H2D at ~5 GB/s (decode is H2D-bound).
        for e, slot, idx in plan:
            slab = _POOL.slabs[idx]
            w13[slot].copy_(
                slab[w13_off : w13_off + w13_b]
                .view(torch.uint8)
                .reshape(self._w13_shape),
                non_blocking=True,
            )
            w2[slot].copy_(
                slab[w2_off : w2_off + w2_b].view(torch.uint8).reshape(self._w2_shape),
                non_blocking=True,
            )
            sc13[slot].copy_(
                slab[sc13_off : sc13_off + sc13_b]
                .view(self._sc13_dt)
                .reshape(self._sc13_shape),
                non_blocking=True,
            )
            sc2[slot].copy_(
                slab[sc2_off : sc2_off + sc2_b]
                .view(self._sc2_dt)
                .reshape(self._sc2_shape),
                non_blocking=True,
            )
        if not async_h2d:
            torch.cuda.current_stream().synchronize()
        if _PE_TIMING:
            _stage("h2d_launch", time.perf_counter_ns() - _t0)
            _stage_tick()

    def row(self, name, e):
        raise RuntimeError("InPlaceNvfp4Store is read-only (in-place); no fill row")

    def fill_tensor(self, name, full):
        raise RuntimeError("InPlaceNvfp4Store is read-only (in-place); no fill_tensor")


class WindowedNvfp4Store(InPlaceNvfp4Store):
    """Windowed pinned nvfp4 store for the CUDA-graph-CAPTURED decode path.

    The captured decode graph gathers experts in-graph by DMA'ing from a page-locked host buffer whose base
    pointer is baked into the graph. An LRU slab pool can't provide that (an expert's address changes on
    eviction), so this store keeps a FIXED pinned ``host_hot[name]`` window of ``W`` experts/layer at stable
    rows — window *hits* gather in-graph (zero per-wave Python/launch overhead, the eager bottleneck), and
    window *misses* stage from the checkpoint at eager BCG breaks. The cold tier stays IN-PLACE (checkpoint
    O_DIRECT reads on demand) so we never materialize the 427 GB expert set. The pager owns the window
    policy (``refresh_window_freq`` -> :meth:`set_window_membership`); this store only executes row swaps and
    the hot/cold reads. Reuses ``InPlaceNvfp4Store``'s source indexing + resident scalars + CPU swizzle.
    """

    pinned = True

    def __init__(
        self,
        layer,
        num_experts_E,
        num_resident_K,
        device,
        *,
        model_path,
        layer_idx,
        W=None,
    ):
        self.E = num_experts_E
        self.K = num_resident_K
        self.device = device
        self.layer_idx = layer_idx
        self.host = None
        self.W = max(
            int(num_resident_K),
            min(int(W if W is not None else _WINDOW_W), num_experts_E),
        )
        self.cold_backing = "disk"
        self._cold_mm = (
            {}
        )  # no slot-ready mmap tier; cold reads route through gather_rows_into (checkpoint)
        self._cold_fd = {}
        gpu_all = discover_paged_params(layer, num_resident_K)
        assert gpu_all, "no per-expert params found on layer"
        assert (
            gpu_all["w13_weight"].dtype == torch.uint8
        ), "windowed nvfp4 expects uint8 packed weights"
        self._sc_w13 = (
            "w13_blockscale_swizzled"
            if "w13_blockscale_swizzled" in gpu_all
            else "w13_weight_scale"
        )
        self._sc_w2 = (
            "w2_blockscale_swizzled"
            if "w2_blockscale_swizzled" in gpu_all
            else "w2_weight_scale"
        )
        # Keep ONLY the 4 real paged tensors. At K=1, discover_paged_params over-collects [1]-shaped resident
        # scalars (e.g. g1_alphas_up) whose bytes aren't 16-aligned and which the captured gather must not
        # move — the resident scalars ride the separate _refresh_nvfp4_scalars path (nvfp4_full_e).
        keep = ("w13_weight", "w2_weight", self._sc_w13, self._sc_w2)
        self.gpu = {n: gpu_all[n] for n in keep}
        self.item_bytes = {
            n: p[0].numel() * p.element_size() for n, p in self.gpu.items()
        }
        _bad = {
            n: (b, tuple(self.gpu[n].shape))
            for n, b in self.item_bytes.items()
            if b % 16
        }
        assert not _bad, f"captured gather needs 16B-aligned slots; offenders: {_bad}"
        self._names = list(
            self.gpu
        )  # canonical tensor order (must match pager's list(self.gpu))

        self._index_expert_sources(model_path, layer_idx)
        self.nvfp4_full_e = self._compute_resident_scalars()

        # host_hot: fixed pinned [W, *slot] per tensor (stable UVA base for the in-graph gather).
        W = self.W
        self.host_hot = {
            n: torch.empty((W, *p.shape[1:]), dtype=p.dtype, pin_memory=True)
            for n, p in self.gpu.items()
        }
        # tier maps (CPU int64 [E]). LAZY window: start EMPTY (all cold) and let refresh_window_freq +
        # cold-staging warm it — pre-filling a static [0,W) wouldn't match the routed experts anyway, and
        # skipping the ~19 GB boot fill makes preload much faster. Invariant: exactly one of
        # hot_pos[e]>=0 / cold_pos[e]>=0. cold_pos is identity (gather uses logical id; direct path stubbed).
        self.hot_pos = torch.full((self.E,), -1, dtype=torch.int64)
        self.cold_pos = torch.arange(self.E, dtype=torch.int64)

        global _LOGGED
        if not _LOGGED:
            _LOGGED = True
            slot_mb = sum(self.item_bytes.values()) / 1e6
            import logging

            logging.getLogger(__name__).info(
                "[paged-experts] WINDOWED NVFP4 (captured path): W=%d experts/layer x %.1f MB pinned "
                "(~%.1f GB over %d layers); cold tier in-place from checkpoint",
                W,
                slot_mb,
                W * slot_mb * _MOE_LAYERS_HINT / 1e3,
                _MOE_LAYERS_HINT,
            )

    # ---- host_hot fill (a window row <- one checkpoint expert, slot-ready) --------------------------
    def _fill_host_row(self, row: int, e: int):
        s = self._src[e]
        gate, up, down = self._projs
        # weights: O_DIRECT read gate|up -> w13 row (gate [:half], up [half:]); down -> w2 row
        w13mv = memoryview(self.host_hot["w13_weight"][row].view(-1).numpy())
        half = len(w13mv) // 2
        _pread_into_direct(s[gate][self._W][5], s[gate][self._W][1], w13mv[:half])
        _pread_into_direct(s[up][self._W][5], s[up][self._W][1], w13mv[half:])
        w2mv = memoryview(self.host_hot["w2_weight"][row].view(-1).numpy())
        _pread_into_direct(s[down][self._W][5], s[down][self._W][1], w2mv)
        # scales: read raw, swizzle ONCE on CPU, copy into the fp8 host_hot scale rows
        g = self._read_small(s[gate]["weight_scale"])
        u = self._read_small(s[up]["weight_scale"])
        dn = self._read_small(s[down]["weight_scale"])
        sw13 = _cpu_swizzle_blockscale(torch.cat([g, u], dim=0).unsqueeze(0))[
            0
        ].contiguous()
        sw2 = _cpu_swizzle_blockscale(dn.unsqueeze(0))[0].contiguous()
        self.host_hot[self._sc_w13][row].view(-1).copy_(sw13.view(-1))
        self.host_hot[self._sc_w2][row].view(-1).copy_(sw2.view(-1))

    def _read_cold_tensor(self, name: str, e: int, dst: torch.Tensor):
        """Read expert ``e``'s slot-ready tensor ``name`` from the checkpoint into ``dst`` (a pinned row)."""
        s = self._src[e]
        gate, up, down = self._projs
        t0 = time.perf_counter_ns() if _CR_PROF else 0
        if name == "w13_weight":
            mv = memoryview(dst.view(-1).numpy())
            half = len(mv) // 2
            _pread_into_direct(s[gate][self._W][5], s[gate][self._W][1], mv[:half])
            _pread_into_direct(s[up][self._W][5], s[up][self._W][1], mv[half:])
            if _CR_PROF:
                _CR_READ_NS[0] += time.perf_counter_ns() - t0
        elif name == "w2_weight":
            _pread_into_direct(
                s[down][self._W][5],
                s[down][self._W][1],
                memoryview(dst.view(-1).numpy()),
            )
            if _CR_PROF:
                _CR_READ_NS[0] += time.perf_counter_ns() - t0
        elif name == self._sc_w13:
            g = self._read_small(s[gate]["weight_scale"])
            u = self._read_small(s[up]["weight_scale"])
            t1 = time.perf_counter_ns() if _CR_PROF else 0
            sw = _cpu_swizzle_blockscale(torch.cat([g, u], dim=0).unsqueeze(0))[
                0
            ].contiguous()
            dst.view(-1).copy_(sw.view(-1))
            if _CR_PROF:
                _CR_READ_NS[0] += t1 - t0
                _CR_SWZ_NS[0] += time.perf_counter_ns() - t1
        elif name == self._sc_w2:
            dn = self._read_small(s[down]["weight_scale"])
            t1 = time.perf_counter_ns() if _CR_PROF else 0
            sw = _cpu_swizzle_blockscale(dn.unsqueeze(0))[0].contiguous()
            dst.view(-1).copy_(sw.view(-1))
            if _CR_PROF:
                _CR_READ_NS[0] += t1 - t0
                _CR_SWZ_NS[0] += time.perf_counter_ns() - t1
        else:
            raise KeyError(f"unknown paged tensor {name}")
        if _CR_PROF:
            _CR_N[0] += 1
            if _CR_N[0] % _CR_LOG_EVERY == 0:
                import logging

                r, w = _CR_READ_NS[0], _CR_SWZ_NS[0]
                tot = (r + w) or 1.0
                logging.getLogger(__name__).info(
                    "[cold-split] %d cold reads: disk-pread=%.0fms (%.0f%%) | swizzle-CPU=%.0fms (%.0f%%)",
                    _CR_LOG_EVERY,
                    r / 1e6,
                    100 * r / tot,
                    w / 1e6,
                    100 * w / tot,
                )
                _CR_READ_NS[0] = 0.0
                _CR_SWZ_NS[0] = 0.0

    # ---- windowed interface (pager contract) -------------------------------------------------------
    def set_window_membership(self, hot_experts):
        """Re-pin the hot window to ``hot_experts`` (top-W by freq, from the pager). Demoted experts drop to
        cold (hot_pos=-1); promoted experts are read from the checkpoint into the freed rows. Returns moved.
        """
        hot = [
            int(e)
            for e in (
                hot_experts.tolist()
                if torch.is_tensor(hot_experts)
                else list(hot_experts)
            )
        ]
        hot = hot[: self.W]
        hot_set = set(hot)
        occupied = {
            int(self.hot_pos[e]): int(e)
            for e in (self.hot_pos >= 0).nonzero().flatten().tolist()
        }
        free_rows = [r for r in range(self.W) if r not in occupied]
        promoted = [e for e in hot if int(self.hot_pos[e]) < 0]
        demoted = [occ_e for r, occ_e in occupied.items() if occ_e not in hot_set]
        moved = 0
        for pe in promoted:
            if free_rows:
                row = free_rows.pop()  # empty window slot (lazy warm-up)
            elif demoted:
                de = demoted.pop()  # evict a now-cold resident to free its row
                row = int(self.hot_pos[de])
                self.hot_pos[de] = -1
                self.cold_pos[de] = de
            else:
                break  # window full of still-hot experts
            self._fill_host_row(
                row, pe
            )  # cold tier is virtual -> read promoted expert from checkpoint
            self.hot_pos[pe] = row
            self.cold_pos[pe] = -1
            moved += 1
        return moved

    def gather_rows_into(self, name, ids, buf):
        """Write ``buf[i]`` = tensor ``name`` for logical expert ``ids[i]`` (hot from host_hot, cold from
        checkpoint). Used by the BCG cold-staging break and page_in fallback."""
        ids = ids.tolist() if torch.is_tensor(ids) else list(ids)
        cold = []
        for i, e in enumerate(ids):
            e = int(e)
            hp = int(self.hot_pos[e])
            if hp >= 0:
                buf[i].copy_(
                    self.host_hot[name][hp]
                )  # window hit: cheap in-RAM copy, keep serial
            elif not _COLD_STAGE.gather_one(self, name, e, buf[i]):
                cold.append(
                    (i, e)
                )  # window miss: disk read — batch for the lean parallel fan-out
        if _LEAN_GATHER and len(cold) > 1:
            # fan the cold preads out for NVMe queue depth; each writes its own buf row (no aliasing),
            # _read_cold_tensor's bounce buffer is thread-local, so this is race-free.
            futs = [
                _READ_POOL.submit(self._read_cold_tensor, name, e, buf[i])
                for i, e in cold
            ]
            for f in futs:
                f.result()
        else:
            for i, e in cold:
                self._read_cold_tensor(name, e, buf[i])

    def _read_cold_rows_direct(self, name, cold_rows, buf):
        # cold tier is in-place (per-logical-id checkpoint reads), not a slot-ready row file -> use the
        # gather_rows_into fallback (which reads by logical id).
        return False

    def prefetch_cold(self, experts, force=False):
        """Ring Stage 1 (within-step): kick async deep-QD reads of ``experts``' COLD tensors into the global
        staging pool, so the layer's distinct cold set reads in PARALLEL. ``gather_rows_into`` then copies
        from the completed staged buffer instead of blocking on a synchronous O_DIRECT read. Called at the
        wave break with the layer's full distinct set (across-wave overlap) and per-wave with each wave's
        cold ids (idempotent — a re-request of a staged expert is a no-op). The CROSS-STEP temporal variant
        was measured net-negative (4.04->4.86 s/tok); this within-step variant is the path.
        """
        if experts is None:
            return
        if os.environ.get("SGLANG_RING_PREFETCH", "0") != "1":
            return  # MEASURED NET-NEGATIVE at bs=1 (6.62 -> 7.23 s/tok W=8): the QD-parallel win is eaten by
            # thread-pool + extra pinned->pinned memcpy orchestration, and the bs=1 GEMM is too small to hide
            # reads behind. DEFAULT OFF (opt-in for a bs>1 retry, where more distinct experts may amortize).
        ids = experts.tolist() if torch.is_tensor(experts) else list(experts)
        for e in ids:
            e = int(e)
            if (
                int(self.hot_pos[e]) < 0
            ):  # cold only; hot experts gather from host_hot in-window
                _COLD_STAGE.prefetch(self, e)

    # ---- eager fallback (off-graph prefill / eager decode) -----------------------------------------
    def page_in(
        self, src_experts, dst_slots, *, stage_bank=0, async_h2d=False, src_host=None
    ):
        if src_experts.numel() == 0:
            return
        srcs = src_host if src_host is not None else src_experts.to("cpu").tolist()
        dsts = dst_slots.to("cpu").tolist()
        dev = self.device
        scratch = None
        for e, slot in zip(srcs, dsts):
            e, slot = int(e), int(slot)
            hp = int(self.hot_pos[e])
            for name in self._names:
                gp = self.gpu[name]
                if hp >= 0:
                    gp[slot].copy_(self.host_hot[name][hp].to(dev, non_blocking=True))
                else:
                    if scratch is None:
                        scratch = {
                            n: torch.empty_like(self.host_hot[n][0], pin_memory=True)
                            for n in self._names
                        }
                    self._read_cold_tensor(name, e, scratch[name])
                    gp[slot].copy_(scratch[name].to(dev, non_blocking=True))
        if not async_h2d:
            torch.cuda.current_stream().synchronize()
