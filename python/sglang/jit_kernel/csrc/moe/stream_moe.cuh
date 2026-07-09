// Compute-through streaming expert-FFN for Paged Experts (srt/layers/moe/paged_experts).
//
// The overflow path (bs*top_k > K) can't keep every active expert resident, so today it falls onto the
// wave loop (cycle K slots, ceil(distinct/K) GEMMs) or the eager host wave — both sync-heavy. This kernel
// instead STREAMS an expert's int4 weights straight from the pinned host store (UVA device pointer, like
// the gather kernels) and computes the full gated FFN in one pass, no slot residency:
//   gu = x @ dequant(w13)^T   [T, 2I];   h = SiLU(gu[:, :I]) * gu[:, I:]   [T, I];   y = h @ dequant(w2)^T
// Amortization: each weight row is streamed ONCE into shared (dequanted), then dotted against ALL T tokens
// routed to this expert. Microbenched (probe/kernels/stream_ffn_int4.cu) bit-exact vs a CPU dequant-FFN and
// PCIe-bound (~30-54 GB/s) — the mechanism is de-risked; this is the fork binding of it.
//
// Weights arrive as raw int64 base pointers (UVA devptr of the pinned store, or a device data_ptr for the
// reference test) in the TRANSPOSED-PLAIN layout: qweight [OUT, IN/8] int32 packed low-nibble-first along
// IN, scales [OUT, IN/group] fp16, sym (effective zero = 8 -> dequant (q-8)*scale). Activations / scratch
// are device TensorViews. Caller preallocates gu/h scratch (no cudaMalloc — capture-safe).

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck
#include <sgl_kernel/utils.cuh>  // For LaunchKernel, SGL_DEVICE

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

// out[t,o] = sum_k dequant(W[o,k]) * x[t,k].  W [OUT, IN/8] int32 (UVA/device), S [OUT, IN/G] fp16,
// X [T, IN] fp16, O [T, OUT] fp32. One block per output row: the row's weights are dequanted ONCE into
// shared, then each warp streams one token's dot against them (warps stride over T).
__global__ void stream_gemm_kernel(
    const uint32_t* __restrict__ W,
    const __half* __restrict__ S,
    const __half* __restrict__ X,
    float* __restrict__ O,
    int T,
    int OUT,
    int IN,
    int G) {
  extern __shared__ float dw[];  // dequant W[o]: IN floats
  const int o = blockIdx.x;
  if (o >= OUT) return;
  const int INp = IN >> 3, NG = IN / G;
  const uint32_t* Wr = W + static_cast<size_t>(o) * INp;
  const __half* Sr = S + static_cast<size_t>(o) * NG;
  for (int j = threadIdx.x; j < INp; j += blockDim.x) {
    const uint32_t w = Wr[j];
    const int k0 = j << 3;
    const float s = __half2float(Sr[k0 / G]);
#pragma unroll
    for (int b = 0; b < 8; ++b) dw[k0 + b] = (static_cast<int>((w >> (4 * b)) & 0xF) - 8) * s;
  }
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nwarp = blockDim.x >> 5;
  for (int t = warp; t < T; t += nwarp) {
    const __half* xt = X + static_cast<size_t>(t) * IN;
    float acc = 0.f;
    for (int k = lane; k < IN; k += 32) acc += dw[k] * __half2float(xt[k]);
#pragma unroll
    for (int r = 16; r > 0; r >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, r);
    if (lane == 0) O[static_cast<size_t>(t) * OUT + o] = acc;
  }
}

// h[t,i] = SiLU(gu[t,i]) * gu[t, I+i].  gu [T, 2I] fp32, h [T, I] fp16.
__global__ void silu_mul_kernel(const float* __restrict__ gu, __half* __restrict__ h, int T, int I) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * I) return;
  const int t = idx / I, i = idx % I;
  const float g = gu[static_cast<size_t>(t) * 2 * I + i];
  const float u = gu[static_cast<size_t>(t) * 2 * I + I + i];
  h[idx] = __float2half((g / (1.f + expf(-g))) * u);
}

// ---- launcher --------------------------------------------------------------------------------------

// One expert's full gated FFN over T tokens, weights streamed from the transposed-plain store. Output O
// is the PRE-scatter expert result [T, H] fp32; the caller applies the routing weight + accumulates into
// the MoE output (matching stream_fused_moe's explicit index_add, NOT marlin's mul_topk_weights path).
void stream_expert_ffn(
    int64_t w13_qw,  // UVA/device base [2I, H/8]  int32
    int64_t w13_s,   // UVA/device base [2I, H/group] fp16
    int64_t w2_qw,   // UVA/device base [H, I/8]  int32
    int64_t w2_s,    // UVA/device base [H, I/group] fp16
    tvm::ffi::TensorView X,   // [T, H] fp16
    tvm::ffi::TensorView O,   // [T, H] fp32  (this expert's output, pre-scatter)
    tvm::ffi::TensorView gu,  // [T, 2I] fp32  scratch
    tvm::ffi::TensorView h,   // [T, I] fp16   scratch
    int64_t inter,
    int64_t hidden,
    int64_t group) {
  using namespace host;

  SymbolicSize T = {"tokens"}, HH = {"hidden"}, I2 = {"inter2"}, II = {"inter"};
  SymbolicDevice device_;
  device_.set_options<kDLCUDA>();
  TensorMatcher({T, HH}).with_dtype<fp16_t>().with_device<kDLCUDA>(device_).verify(X);
  TensorMatcher({T, HH}).with_dtype<float>().with_device<kDLCUDA>(device_).verify(O);
  TensorMatcher({T, I2}).with_dtype<float>().with_device<kDLCUDA>(device_).verify(gu);
  TensorMatcher({T, II}).with_dtype<fp16_t>().with_device<kDLCUDA>(device_).verify(h);

  const int t = static_cast<int>(T.unwrap());
  const int H = static_cast<int>(hidden);
  const int I = static_cast<int>(inter);
  const int G = static_cast<int>(group);
  const int O13 = 2 * I;
  const DLDevice device = device_.unwrap();

  const uint32_t* w13q = reinterpret_cast<const uint32_t*>(w13_qw);
  const __half* w13sc = reinterpret_cast<const __half*>(w13_s);
  const uint32_t* w2q = reinterpret_cast<const uint32_t*>(w2_qw);
  const __half* w2sc = reinterpret_cast<const __half*>(w2_s);
  const __half* x = static_cast<const __half*>(X.data_ptr());
  float* o = static_cast<float*>(O.data_ptr());
  float* guf = static_cast<float*>(gu.data_ptr());
  __half* hf = static_cast<__half*>(h.data_ptr());

  // gemm1: [T,H] @ w13[2I,H] -> gu[T,2I];  silu -> h[T,I];  gemm2: [T,I] @ w2[H,I] -> O[T,H]
  LaunchKernel(O13, 256, device, static_cast<size_t>(H) * sizeof(float))(
      stream_gemm_kernel, w13q, w13sc, x, guf, t, O13, H, G);
  LaunchKernel((t * I + 255) / 256, 256, device)(silu_mul_kernel, static_cast<const float*>(guf), hf, t, I);
  LaunchKernel(H, 256, device, static_cast<size_t>(I) * sizeof(float))(
      stream_gemm_kernel, w2q, w2sc, hf, o, t, H, I, G);
}

}  // namespace
