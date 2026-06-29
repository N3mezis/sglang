// On-device residency decision for Paged Experts (srt/layers/moe/paged_experts).
//
// Replaces the host-side keep-warm/LRU decision so the per-decode-step paging plan is computed entirely
// on the GPU — no host sync (`.tolist()`), which is what makes the decode step CUDA-graph-capturable.
// The kernels only *decide* (which experts to page in, which slots to evict, and the logical->slot remap);
// the actual weight movement is done by the existing transfer_kv_per_layer_mla gather (capturable, reads
// the index buffers these kernels fill). One thread, serial: the residency decision has sequential
// dependencies (each eviction depends on the prior assignments) and the working set (K slots, top-k ids)
// is tiny, so there is no parallelism to exploit.

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck
#include <sgl_kernel/utils.cuh>  // For LaunchKernel, SGL_DEVICE

#include <cuda_runtime.h>  // For cudaHostGetDevicePointer (UVA device pointer of the pinned store)
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <climits>
#include <cstdint>

namespace {

// Victim = a non-needed resident slot, chosen by eviction policy: LFU (lfu != 0) -> minimum use count of
// the resident expert, LRU recency as tiebreak; LRU (lfu == 0) -> minimum last-use step. Empty slots
// (slot_expert < 0) get freq key 0 so they are filled first. Mirrors the host path's KT_EVICT_POLICY.
SGL_DEVICE int pick_victim(
    const int32_t* topk,
    int topk_n,
    int K,
    int lfu,
    const int32_t* slot_expert,
    const int32_t* slot_lastuse,
    const int32_t* freq) {
  int victim = -1, best_f = INT_MAX, best_lu = INT_MAX;
  for (int s = 0; s < K; ++s) {
    const int se = slot_expert[s];
    bool needed = false;
    for (int j = 0; j < topk_n; ++j) {
      if (topk[j] == se) {
        needed = true;
        break;
      }
    }
    if (needed) continue;  // never evict a slot this step still needs
    const int f = (lfu && se >= 0) ? freq[se] : 0;  // LRU: f == 0 always -> falls through to lastuse
    const int lu = slot_lastuse[s];
    if (f < best_f || (f == best_f && lu < best_lu)) {
      best_f = f;
      best_lu = lu;
      victim = s;
    }
  }
  return victim;
}

// Keep-warm + LRU/LFU decision (distinct active experts <= K). Mutates the residency state in place and
// emits the page-in plan: for each distinct active expert not resident, evict a non-needed slot and assign
// it. src[0..n)/dst[0..n) are the (expert -> slot) page-ins; n_out is their count; idx[e] is the updated
// logical->slot map (-1 == not resident) that the forward remap reads.
__global__ void decide_kernel(
    const int32_t* topk,
    int topk_n,
    int E,
    int K,
    int lfu,
    int32_t* step_ctr,       // [1] monotonic step counter, incremented on-device (capture-safe)
    int32_t* slot_expert,    // [K] slot -> expert id (-1 == empty), mutated
    int32_t* expert_slot,    // [E] expert -> slot (-1 == not resident), mutated
    int32_t* slot_lastuse,   // [K] last step each slot was used, mutated
    int32_t* freq,           // [E] per-expert use count (LFU key), mutated
    int32_t* src,            // [>=K] out: page-in source experts
    int32_t* dst,            // [>=K] out: page-in destination slots
    int32_t* n_out,          // [1]  out: number of page-ins
    int32_t* idx) {          // [E]  out: logical -> slot map snapshot
  if (threadIdx.x || blockIdx.x) return;
  // The step counter lives on-device and is bumped here so a captured graph advances LRU recency on every
  // replay (a host-scalar step would be frozen at capture time).
  const int step = ++(*step_ctr);
  // pass 1: bump per-expert use count (LFU key) and resident-hit recency (LRU key / tiebreak)
  for (int i = 0; i < topk_n; ++i) {
    const int e = topk[i];
    if (e < 0 || e >= E) continue;
    freq[e]++;
    const int s = expert_slot[e];
    if (s >= 0) slot_lastuse[s] = step;
  }
  // pass 2: each miss evicts a non-needed slot per the policy and pages its expert in
  int n = 0;
  for (int i = 0; i < topk_n; ++i) {
    const int e = topk[i];
    if (e < 0 || e >= E) continue;
    if (expert_slot[e] >= 0) continue;  // resident (or just assigned this step)
    const int victim = pick_victim(topk, topk_n, K, lfu, slot_expert, slot_lastuse, freq);
    if (victim < 0) continue;  // pool too small (should not happen: distinct <= K)
    const int old = slot_expert[victim];
    if (old >= 0) expert_slot[old] = -1;
    slot_expert[victim] = e;
    expert_slot[e] = victim;
    slot_lastuse[victim] = step;
    src[n] = e;
    dst[n] = victim;
    ++n;
  }
  *n_out = n;
  for (int e = 0; e < E; ++e) idx[e] = expert_slot[e];
}

// Bounded keep-warm + LRU/LFU decision for the pinned-WINDOW store (distinct active experts <= K). Same
// residency logic as ``decide_kernel`` but partitions the page-in plan by window membership so the captured
// gather only ever reads the pinned hot block. ``log2hot[e]`` = hot-block index if expert e is in the
// pinned window, else -1; ``log2cold[e]`` = cold-block (host_cold) index if e is cold, else -1.
//   * window hit (hot)  -> (src=hot index, dst=slot) in the windowed plan -> on-device gather from host_hot.
//   * cold, ``defer_cold`` (replay-twice / Rung 2): the expert isn't gatherable in-graph (pageable/disk).
//     Record its LOGICAL id in ``cold_log`` and leave it UNRESIDENT (idx stays -1 -> masked this replay) with
//     NO eviction (don't displace a window hit for an expert we can't gather). The host stages it into the
//     window out-of-graph and replays the SAME graph again until no miss.
//   * cold, not deferred (Rung 1, registered cold tier): assign a slot and emit (cold_log=COLD-BLOCK index,
//     cold_dst=slot) so the caller's gather indexes host_cold directly.
// ``needed[s]`` = 1 iff slot s holds an expert needed THIS step (logical id in topk): the replay-twice refill
// must not evict these, else a still-needed expert re-misses and the loop never converges.
__global__ void decide_bounded_kernel(
    const int32_t* topk,
    int topk_n,
    int E,
    int K,
    int lfu,
    int defer_cold,
    const int32_t* log2hot,  // [E] hot-block index if e in window, else -1
    const int32_t* log2cold,  // [E] cold-block index if e is cold, else -1
    int32_t* step_ctr,       // [1] monotonic step counter, incremented on-device (capture-safe)
    int32_t* slot_expert,    // [K] slot -> expert id (-1 == empty), mutated
    int32_t* expert_slot,    // [E] expert -> slot (-1 == not resident), mutated
    int32_t* slot_lastuse,   // [K] last step each slot was used, mutated
    int32_t* freq,           // [E] per-expert use count (LFU key), mutated
    int32_t* src,            // [>=K] out: windowed page-in source (hot-block index)
    int32_t* dst,            // [>=K] out: windowed page-in destination slots
    int32_t* n_out,          // [1]  out: number of windowed page-ins
    int32_t* cold_log,       // [>=K] out: cold page-in source (cold-block index, or logical id when deferred)
    int32_t* cold_dst,       // [>=K] out: cold page-in destination slots (unused entries when deferred)
    int32_t* cold_n,         // [1]  out: number of cold entries
    int32_t* idx,            // [E]  out: logical -> slot map snapshot
    int32_t* needed) {       // [K]  out: 1 iff slot holds an expert needed this step
  if (threadIdx.x || blockIdx.x) return;
  const int step = ++(*step_ctr);
  // pass 1: bump per-expert use count (LFU key) and resident-hit recency (LRU key / tiebreak)
  for (int i = 0; i < topk_n; ++i) {
    const int e = topk[i];
    if (e < 0 || e >= E) continue;
    freq[e]++;
    const int s = expert_slot[e];
    if (s >= 0) slot_lastuse[s] = step;
  }
  // pass 2: each miss is split by window membership (hot -> in-graph gather, cold -> defer or host_cold)
  int nw = 0, nc = 0;
  for (int i = 0; i < topk_n; ++i) {
    const int e = topk[i];
    if (e < 0 || e >= E) continue;
    if (expert_slot[e] >= 0) continue;  // resident (or just assigned this step)
    const int hi = log2hot[e];
    if (defer_cold && hi < 0) {
      // Replay-twice: window-miss is not gatherable in-graph -> record logical id, no eviction, stays masked.
      cold_log[nc] = e;
      ++nc;
      continue;
    }
    const int victim = pick_victim(topk, topk_n, K, lfu, slot_expert, slot_lastuse, freq);
    if (victim < 0) continue;  // pool too small (should not happen: distinct <= K)
    const int old = slot_expert[victim];
    if (old >= 0) expert_slot[old] = -1;
    slot_expert[victim] = e;
    expert_slot[e] = victim;
    slot_lastuse[victim] = step;
    if (hi >= 0) {  // windowed hit -> on-device gather from the pinned hot block
      src[nw] = hi;
      dst[nw] = victim;
      ++nw;
    } else {  // Rung 1 cold (registered cold tier) -> host_cold gather
      cold_log[nc] = log2cold[e];
      cold_dst[nc] = victim;
      ++nc;
    }
  }
  *n_out = nw;
  *cold_n = nc;
  for (int e = 0; e < E; ++e) idx[e] = expert_slot[e];
  for (int s = 0; s < K; ++s) {
    const int se = slot_expert[s];
    int nd = 0;
    if (se >= 0) {
      for (int i = 0; i < topk_n; ++i) {
        if (topk[i] == se) {
          nd = 1;
          break;
        }
      }
    }
    needed[s] = nd;
  }
}

// Static fixed-wave decision (distinct active experts > K, e.g. prefill / batched decode). Expert e has a
// STATIC home: wave floor(e/K), slot e%K. For wave w this emits the page-in plan for the distinct in-wave
// experts present in topk (src=e, dst=e-w*K) and writes idx[e] = (e in [w*K, (w+1)*K)) ? e-w*K : -1. The
// caller runs ceil(E/K) waves; each active expert is served in exactly its wave, so summing the per-wave
// GEMM partials reconstructs the full MoE output (lossless). No eviction, no state mutation, no host sync.
__global__ void decide_wave_kernel(
    const int32_t* topk,
    int topk_n,
    int E,
    int K,
    int w,
    int32_t* src,
    int32_t* dst,
    int32_t* n_out,
    int32_t* idx) {
  if (threadIdx.x || blockIdx.x) return;
  const int lo = w * K, hi = lo + K;
  for (int e = 0; e < E; ++e) idx[e] = (e >= lo && e < hi) ? (e - lo) : -1;
  int n = 0;
  for (int i = 0; i < topk_n; ++i) {
    const int e = topk[i];
    if (e < lo || e >= hi) continue;  // not this wave's group
    bool seen = false;
    for (int k = 0; k < n; ++k) {
      if (src[k] == e) {
        seen = true;
        break;
      }
    }
    if (!seen) {  // distinct in-wave hit -> its home slot
      src[n] = e;
      dst[n] = e - lo;
      ++n;
    }
  }
  *n_out = n;
}

// Gather: copy n experts (src[i] -> dst[i]) from the pinned host store into the GPU slot pool, float4.
// The page-in count *n is read ON-DEVICE, so under CUDA-graph capture each replay moves exactly the
// experts the decide kernel chose this step (transfer_kv would move a fixed src_indices.numel() instead).
// ``store`` is the pinned host buffer addressed through its UVA device pointer; ``e16`` = per-expert
// bytes / 16. Copy-only — marlin int4 / bf16 rows travel packed; no dequant.
__global__ void gather_kernel(
    const float4* store, float4* slot, const int32_t* src, const int32_t* dst, const int32_t* n, long e16) {
  const long M = static_cast<long>(*n) * e16;
  const long stride = static_cast<long>(gridDim.x) * blockDim.x;
  for (long j = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x; j < M; j += stride) {
    const long s = j / e16, off = j % e16;
    slot[static_cast<long>(dst[s]) * e16 + off] = store[static_cast<long>(src[s]) * e16 + off];
  }
}

// ---- launchers -------------------------------------------------------------------------------------

void decide(
    tvm::ffi::TensorView topk,
    tvm::ffi::TensorView step_ctr,
    tvm::ffi::TensorView slot_expert,
    tvm::ffi::TensorView expert_slot,
    tvm::ffi::TensorView slot_lastuse,
    tvm::ffi::TensorView freq,
    int64_t lfu,
    tvm::ffi::TensorView src,
    tvm::ffi::TensorView dst,
    tvm::ffi::TensorView n_out,
    tvm::ffi::TensorView idx) {
  using namespace host;

  // All operands are int32 CUDA tensors on the same device. Bind E to expert_slot and K to slot_expert,
  // then verify the rest against those symbolic sizes so a shape mismatch is caught here.
  SymbolicSize E = {"num_experts"}, K = {"num_slots"}, T = {"topk_n"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();
  TensorMatcher({E}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(expert_slot).verify(freq).verify(idx);
  TensorMatcher({K}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(slot_expert).verify(slot_lastuse).verify(src).verify(dst);
  TensorMatcher({T}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(topk);

  const int e = static_cast<int>(E.unwrap());
  const int k = static_cast<int>(K.unwrap());
  const int t = static_cast<int>(T.unwrap());
  const DLDevice device = device_.unwrap();

  LaunchKernel(1, 1, device)(
      decide_kernel,
      static_cast<const int32_t*>(topk.data_ptr()),
      t,
      e,
      k,
      static_cast<int>(lfu),
      static_cast<int32_t*>(step_ctr.data_ptr()),
      static_cast<int32_t*>(slot_expert.data_ptr()),
      static_cast<int32_t*>(expert_slot.data_ptr()),
      static_cast<int32_t*>(slot_lastuse.data_ptr()),
      static_cast<int32_t*>(freq.data_ptr()),
      static_cast<int32_t*>(src.data_ptr()),
      static_cast<int32_t*>(dst.data_ptr()),
      static_cast<int32_t*>(n_out.data_ptr()),
      static_cast<int32_t*>(idx.data_ptr()));
}

void decide_bounded(
    tvm::ffi::TensorView topk,
    int64_t lfu,
    int64_t defer_cold,
    tvm::ffi::TensorView log2hot,
    tvm::ffi::TensorView log2cold,
    tvm::ffi::TensorView step_ctr,
    tvm::ffi::TensorView slot_expert,
    tvm::ffi::TensorView expert_slot,
    tvm::ffi::TensorView slot_lastuse,
    tvm::ffi::TensorView freq,
    tvm::ffi::TensorView src,
    tvm::ffi::TensorView dst,
    tvm::ffi::TensorView n_out,
    tvm::ffi::TensorView cold_log,
    tvm::ffi::TensorView cold_dst,
    tvm::ffi::TensorView cold_n,
    tvm::ffi::TensorView idx,
    tvm::ffi::TensorView needed) {
  using namespace host;

  // E bound to expert_slot, K to slot_expert; the per-expert maps (freq/idx/log2hot/log2cold) are [E], the
  // per-slot ones (slot_lastuse/needed) and page-in plans (src/dst/cold_*) are [K], topk is [T].
  SymbolicSize E = {"num_experts"}, K = {"num_slots"}, T = {"topk_n"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();
  TensorMatcher({E}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(expert_slot).verify(freq).verify(idx).verify(log2hot).verify(log2cold);
  TensorMatcher({K}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(slot_expert).verify(slot_lastuse).verify(src).verify(dst).verify(cold_log).verify(cold_dst).verify(needed);
  TensorMatcher({T}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(topk);

  const int e = static_cast<int>(E.unwrap());
  const int k = static_cast<int>(K.unwrap());
  const int t = static_cast<int>(T.unwrap());
  const DLDevice device = device_.unwrap();

  LaunchKernel(1, 1, device)(
      decide_bounded_kernel,
      static_cast<const int32_t*>(topk.data_ptr()),
      t,
      e,
      k,
      static_cast<int>(lfu),
      static_cast<int>(defer_cold),
      static_cast<const int32_t*>(log2hot.data_ptr()),
      static_cast<const int32_t*>(log2cold.data_ptr()),
      static_cast<int32_t*>(step_ctr.data_ptr()),
      static_cast<int32_t*>(slot_expert.data_ptr()),
      static_cast<int32_t*>(expert_slot.data_ptr()),
      static_cast<int32_t*>(slot_lastuse.data_ptr()),
      static_cast<int32_t*>(freq.data_ptr()),
      static_cast<int32_t*>(src.data_ptr()),
      static_cast<int32_t*>(dst.data_ptr()),
      static_cast<int32_t*>(n_out.data_ptr()),
      static_cast<int32_t*>(cold_log.data_ptr()),
      static_cast<int32_t*>(cold_dst.data_ptr()),
      static_cast<int32_t*>(cold_n.data_ptr()),
      static_cast<int32_t*>(idx.data_ptr()),
      static_cast<int32_t*>(needed.data_ptr()));
}

void decide_wave(
    tvm::ffi::TensorView topk,
    int64_t num_experts,
    int64_t num_slots,
    int64_t wave,
    tvm::ffi::TensorView src,
    tvm::ffi::TensorView dst,
    tvm::ffi::TensorView n_out,
    tvm::ffi::TensorView idx) {
  using namespace host;

  SymbolicSize K = {"num_slots"}, T = {"topk_n"}, Eidx = {"num_experts"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();
  TensorMatcher({K}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(src).verify(dst);
  TensorMatcher({T}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(topk);
  TensorMatcher({Eidx}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(idx);

  const int t = static_cast<int>(T.unwrap());
  const DLDevice device = device_.unwrap();

  LaunchKernel(1, 1, device)(
      decide_wave_kernel,
      static_cast<const int32_t*>(topk.data_ptr()),
      t,
      static_cast<int>(num_experts),
      static_cast<int>(num_slots),
      static_cast<int>(wave),
      static_cast<int32_t*>(src.data_ptr()),
      static_cast<int32_t*>(dst.data_ptr()),
      static_cast<int32_t*>(n_out.data_ptr()),
      static_cast<int32_t*>(idx.data_ptr()));
}

// Resolve the UVA device pointer of a pinned host tensor, once at setup (NOT inside the captured
// region). Returned as int64 and passed back to ``gather`` so no host CUDA call happens during replay.
int64_t host_devptr(tvm::ffi::TensorView pinned) {
  void* d = nullptr;
  cudaError_t e = cudaHostGetDevicePointer(&d, pinned.data_ptr(), 0);
  host::RuntimeCheck(e == cudaSuccess, "cudaHostGetDevicePointer failed: ", cudaGetErrorString(e));
  return reinterpret_cast<int64_t>(d);
}

void gather(
    int64_t store_devptr,
    tvm::ffi::TensorView slot,
    tvm::ffi::TensorView src,
    tvm::ffi::TensorView dst,
    tvm::ffi::TensorView n_out,
    int64_t item_bytes) {
  using namespace host;

  SymbolicSize Nsrc = {"n_src"}, One = {"one"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();
  TensorMatcher({Nsrc}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(src).verify(dst);
  TensorMatcher({One}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(n_out);
  const DLDevice device = device_.unwrap();
  RuntimeCheck(
      item_bytes % 16 == 0,
      "paged_experts gather needs 16-byte-aligned per-expert blocks (float4); got ",
      item_bytes);

  LaunchKernel(2048, 256, device)(
      gather_kernel,
      reinterpret_cast<const float4*>(store_devptr),
      reinterpret_cast<float4*>(slot.data_ptr()),
      static_cast<const int32_t*>(src.data_ptr()),
      static_cast<const int32_t*>(dst.data_ptr()),
      static_cast<const int32_t*>(n_out.data_ptr()),
      static_cast<long>(item_bytes / 16));
}

}  // namespace
