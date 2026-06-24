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

}  // namespace
