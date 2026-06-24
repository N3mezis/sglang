# Paged Experts

This document describes SGLang **Paged Experts**, a feature that lets you serve a Mixture-of-Experts
(MoE) model whose expert weights do not fit in GPU memory. It covers the motivation, the system design
and per-step workflow, how the resident pool is sized, the supported configurations and their limits,
the performance characteristics, and the related parameters — serving as a complete reference for users
and developers.

## Why and What is Paged Experts?

An MoE model is dominated by its expert weights — often the large majority of the parameter budget — yet
at decode time each token is routed to only a few experts (`top_k` of `E`). The rest of the experts sit
in GPU memory unused for that step. Paged Experts exploits this: it keeps a working set of **K of the E
experts resident on the GPU** and **pages the remaining experts in from pinned host memory over PCIe, on
demand**, as routing requires them.

This mirrors the tiering idea behind [HiCache](hicache_design.md), but applied to *expert weights*
instead of the KV cache: GPU memory is the fast tier that holds the hot working set, and host memory is
the larger backing tier. Where HiCache lets the KV cache spill to host, Paged Experts lets the expert
table spill to host, so the model's expert capacity is bounded by host RAM rather than VRAM.

It is, in spirit, *PagedAttention for MoE experts*: the model's router still operates over the full set
of `E` experts (the gate is untouched), but the on-GPU expert *table* has only `K` slots, and each step
the experts that are needed but not resident are copied into slots before the expert GEMM runs.

Importantly, **computation stays on the GPU**. Only expert *storage* is host-backed; experts are paged to
the GPU and computed there. This distinguishes Paged Experts from `--cpu-offload-gb` and from
expert-parallel CPU offload, both of which move the expert computation itself onto the CPU.

Decode under Paged Experts is **PCIe-bandwidth-bound** — the dominant cost is moving expert weights from
host to GPU each step — which shapes its performance profile (see [Performance](#performance)).

## Related Parameters

- **`--enable-paged-experts`**: Enable Paged Experts. The MoE layers are wrapped with the resident
  expert table and the host-backed paging path. Off by default.

- **`--paged-experts-num-resident PAGED_EXPERTS_NUM_RESIDENT`**: The number of experts kept resident per
  layer (`K`), or `auto` (default) to size it from free VRAM using SGLang's memory model (see
  [Resident-pool sizing](#resident-pool-sizing-auto-k)). Larger `K` raises the cache hit rate and reduces
  paging, at the cost of VRAM that would otherwise go to the KV cache.

- **`--paged-experts-store {pinned,paged}`**: The host expert store kind. `pinned` (default) page-locks
  the store and pages it with the fast transfer kernel. `paged` uses a non-pinned store instead — use it
  only when the pinned store would exceed the host's page-locked memory limit (e.g. an unquantized model
  on a small-RAM box, or under WSL where the page-locked pool is capped at roughly half of system RAM).
  The `paged` store pages via a plain indexed copy, which is correct but noticeably slower than `pinned`.

The following standard arguments interact with Paged Experts and are worth setting deliberately:

- **`--disable-cuda-graph`**: Currently required (the per-step paging decision is not CUDA-graph
  capturable).

- **`--max-running-requests MAX_RUNNING_REQUESTS`**: The admission concurrency. Higher values raise
  aggregate throughput (batching amortizes the per-step weight movement) and also drive the KV reserve
  that `auto`-K sizes against.

- **`--mem-fraction-static MEM_FRACTION_STATIC`**: The fraction of VRAM given to weights + KV. `auto`-K
  reads this exact value, so the resident pool and KV pool remain coherent. Leaving it unset lets SGLang
  derive a safe value; raising it packs more experts at the cost of activation/graph headroom.

- **`--kv-cache-dtype fp8`**: Halves the per-token KV cost, freeing VRAM that `auto`-K turns into a larger
  resident pool.

## Supported Configurations and Limitations

- **Quantization:** unquantized **bf16** and **gptq-marlin int4** MoE models.
- **Single GPU.** Paged Experts currently runs on one device. Tensor / expert / pipeline / data
  parallelism, EPLB, a non-default `--moe-a2a-backend`, and `--load-format dummy` are **rejected at
  startup with an actionable error**, rather than silently producing wrong output — Paged Experts owns
  all `E` experts on a single rank, and a rank-aware store is future work.
- **Eager execution.** `--disable-cuda-graph` is currently required: the per-step residency decision runs
  on the host, which a CUDA graph cannot capture. A captured fast path is planned.
- **Host memory.** The host store holds the full expert weights. By default it is pinned, so the host must
  be able to page-lock that much memory (plus headroom for the rest of the serving process); on a host that
  cannot, use `--paged-experts-store paged` to trade the pinned fast path for a slower pageable one.
- **Startup cost.** For gptq-int4 the host store is repacked to the marlin layout per layer at load time;
  this is a one-time cost paid at startup.

## Performance

Because decode is bound by the PCIe bandwidth of moving expert weights from host to GPU, single-stream
latency is modest, but **aggregate throughput scales with concurrency**: batching lets a single weight
read serve more tokens, amortizing the dominant cost until PCIe saturates. Choose the operating point for
your workload — low concurrency for per-user latency, high concurrency for aggregate throughput.

Tuning guidance:
- For **throughput / many users**, raise `--max-running-requests`; aggregate throughput scales with
  concurrency until PCIe saturates.
- For a **larger resident pool** (higher hit rate), free VRAM for experts: `--kv-cache-dtype fp8` halves
  the KV reserve, which `auto`-K converts into a larger `K`. Keeping `--context-length` no larger than you
  need has the same effect.
- A larger resident fraction `K/E` reduces the number of page-ins per step, so prefer the largest `K` that
  fits within your latency and KV budget.

## System Design

### The resident expert table

When Paged Experts wraps an MoE layer, the layer's on-GPU expert parameters are allocated with only `K`
slots instead of `E` (`num_local_experts = K`). SGLang's native expert weight loader then fills slots
`0..K-1` with the first `K` experts — no custom loader is required, because setting `num_local_experts`
makes the standard expert-parallel remap load exactly those experts and skip the rest.

The model's router is left untouched and continues to emit `top_k` ids over the full `E` experts. Paged
Experts maintains a `logical_to_gpu_index` map (`E` entries; the slot of each expert, or `-1` if it is not
currently resident) that translates those logical ids to physical slots for the GEMM.

### The host expert store

For every paged per-expert tensor on the layer (e.g. `w13_weight`/`w2_weight` for bf16, or the
`qweight`/`scales`/`qzeros` for gptq-marlin int4), Paged Experts allocates a **pinned host buffer holding
all `E` experts**. The store is filled once at load time, directly from the model checkpoint:

- **bf16 (unquantized):** expert rows are copied straight into the host store — the host layout matches
  the on-GPU layout.
- **gptq-marlin int4:** the GPTQ checkpoint is **repacked into the on-GPU marlin layout** for all `E`
  experts, using SGLang's own `gptq_marlin_moe_repack` and `marlin_moe_permute_scales`. SGLang's loader
  only repacks the `K` resident slots, so Paged Experts repacks the full set so the paged experts match
  the resident ones bit-for-bit. This happens at load time; there is no offline artifact to build or
  manage.

### Per-step workflow: decide, page in, remap

Each decode step, for each MoE layer, Paged Experts performs three operations before the expert GEMM:

1. **Decide.** Given the step's `top_k` expert ids, it determines which distinct active experts are
   already resident and which are misses. For each miss it evicts a slot whose expert is *not needed this
   step*, choosing the least-recently-used such slot (a keep-warm + LRU policy), and assigns the missing
   expert to it. The residency maps are updated in place. This decision is data-dependent and runs on the
   host.
2. **Page in.** The chosen experts are copied from the pinned host store into their slots using SGLang's
   existing host↔device transfer kernel, `transfer_kv_*_mla` (see below). Only the misses are moved.
3. **Remap and compute.** The `top_k` logical ids are remapped to slots via `logical_to_gpu_index`. Any
   expert that is still non-resident maps to `-1`; its routing weight is zeroed so its contribution is
   exactly zero, and its id is clamped to a valid slot for the kernel's binning. The real fused-MoE GEMM
   then runs over the `K`-slot pool.

The result is lossless: the correct weights are in the correct slots and the remap is exact, so the
output matches the equivalent fully-resident MoE.

### Reusing the KV transfer kernels

The page-in does not introduce a custom CUDA kernel. It reuses the same indexed host→device block-copy
kernel that HiCache uses to move KV pages between tiers (`transfer_kv_*_mla`): it copies fixed-size
per-expert blocks from a pinned host buffer to a device buffer according to source/destination index
tensors that are **read on-device**, so the count of pages moved is dynamic and the copy is itself
capture-safe. Paged Experts issues one transfer per paged tensor, with `item_size` set to the per-expert
block size in bytes. (A consequence of reusing this kernel is that each per-expert block must be 8-byte
aligned; standard weight rows satisfy this.)

### Resident-pool sizing (auto-K)

The size of the resident pool, `K`, trades directly against the KV cache: SGLang sizes its KV pool from
the VRAM left after weights, and the `K`-slot expert pool counts as weights. Paged Experts therefore sizes
`K` from SGLang's own memory accounting rather than a fixed guess. With `--paged-experts-num-resident
auto` (the default), at weight-creation time it reads the already-derived `mem_fraction_static`,
`max_running_requests`, `context_length`, and `kv_cache_dtype` off the running server config, and computes:

```
K = clamp(top_k, E, floor((free_vram * mem_fraction_static - non_expert_weights - kv_reserve)
                          / (moe_layers * per_expert_bytes)))
```

Because `K` is sized against the *same* `mem_fraction_static` the server runs at, the resident pool and
the KV pool stay coherent by construction. The resolved value is logged at startup:

```
[paged-experts] resident K=25/128 (19%): free=6.66GB mem_fraction=0.850 KV_reserve=0.20GB ...
```

A larger `K` means a higher hit rate and fewer page-ins, so the largest `K` that fits is preferred. You
can pin an explicit integer if you want a fixed pool.
