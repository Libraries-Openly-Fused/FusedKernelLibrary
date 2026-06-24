/* Copyright 2026 Oscar Amoros Huguet, Johnny Nunez

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_COLLECTIVE_REDUCE_H
#define FK_COLLECTIVE_REDUCE_H

/* Collective reductions as first-class FKL building blocks.
 *
 * WHY THIS EXISTS. Cooperative DPPs (SoftmaxDPP today; the tiled mainloop
 * to come) all need cross-thread reductions: row-max and row-sum for
 * softmax, accumulation for GEMM, norms for layernorm. Until now each
 * kernel open-coded the __shfl_xor warp tree and the __shared__ block
 * tree by hand (see SoftmaxDPP, and quantizeQTile in flash_attention_mma).
 * That hand-rolled state is exactly the kind of thing FKL's IOp model is
 * meant to abstract — so this header makes the reduction a COMPOSED
 * operation parameterised by an associative combine functor, keeping the
 * FKL essence: small static structs, host+device parity, no inheritance
 * cost, composability.
 *
 * THE COMBINE FUNCTOR is just an FKL-shaped binary callable:
 *     T operator()(const T& a, const T& b) const;   // associative
 * The library ships ReduceMax / ReduceSum here; SoftmaxDPP's
 * mergeSoftmaxStates is itself a valid combine over OnlineSoftmaxState,
 * so the same primitives reduce scalars AND online-softmax states.
 *
 * TWO TIERS, matching the hardware:
 *   WarpReduce<Combine>  — intra-warp, __shfl_xor butterfly, no smem, no
 *                          barrier. Result broadcast to all lanes.
 *   BlockReduce<Combine> — whole block: each warp reduces, lane0 of each
 *                          warp stages to smem, warp 0 reduces the partials.
 *
 * These are device-side collectives (they call __shfl_xor / __syncthreads),
 * so like the cooperative DPP exec bodies they are NOT constexpr. On the
 * host pass (g++ CPU TU, or nvcc host pass) they degrade to a plain serial
 * fold so the same code type-checks and unit-tests run on CPU.
 */

#include <fused_kernel/core/execution_model/operation_model/operation_model.h>
#include <fused_kernel/core/utils/utils.h>

namespace fk {

/* ---- associative combine functors (host+device, constexpr-friendly) ---- */
template <typename T>
struct ReduceSum {
    FK_HOST_DEVICE_CNST T operator()(const T& a, const T& b) const { return a + b; }
    FK_HOST_DEVICE_CNST static T identity() { return T{}; }
};

template <typename T>
struct ReduceMax {
    FK_HOST_DEVICE_CNST T operator()(const T& a, const T& b) const {
        return a > b ? a : (b >= a ? b : a);   // NaN -> first arg (matches attnMaxF)
    }
};

template <typename T>
struct ReduceMin {
    FK_HOST_DEVICE_CNST T operator()(const T& a, const T& b) const {
        return a < b ? a : (b <= a ? b : a);
    }
};

#if defined(__NVCC__) && defined(__CUDA_ARCH__)

/* ---- WarpReduce: intra-warp all-reduce via __shfl_xor butterfly ----
 * WARP_WIDTH must be a power of two <= 32. Every participating lane in the
 * mask receives the fully reduced value. mask defaults to the full warp;
 * pass a sub-warp mask + WARP_WIDTH for segmented (per-row) reductions. */
template <typename Combine, int WARP_WIDTH = 32, typename T>
__device__ __forceinline__ T warpReduce(T val, Combine combine = {},
                                         const unsigned mask = 0xFFFFFFFFu) {
    #pragma unroll
    for (int offset = WARP_WIDTH / 2; offset > 0; offset >>= 1) {
        const T other = __shfl_xor_sync(mask, val, offset, WARP_WIDTH);
        val = combine(val, other);
    }
    return val;
}

/* ---- BlockReduce: whole-block all-reduce ----
 * Uses a caller-provided smem scratch of >= (BLOCK_SIZE/32) elements so the
 * primitive allocates nothing itself (the kernel owns its smem budget —
 * matches how the cooperative DPPs manage shared memory). Result is
 * returned on lane0 of warp0; pass broadcast=true to publish to all
 * threads through the same scratch. */
template <typename Combine, int BLOCK_SIZE, typename T>
__device__ __forceinline__ T blockReduce(T val, T* smemScratch,
                                          Combine combine = {},
                                          const bool broadcast = true) {
    static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a multiple of 32");
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    val = warpReduce<Combine, 32>(val, combine);   // reduce within each warp
    if (lane == 0) smemScratch[warp] = val;        // stage warp partials
    __syncthreads();

    if (warp == 0) {
        T acc = (lane < NUM_WARPS) ? smemScratch[lane] : smemScratch[0];
        acc = warpReduce<Combine, (NUM_WARPS > 1 ? NUM_WARPS : 1)>(acc, combine,
                                  (NUM_WARPS >= 32) ? 0xFFFFFFFFu : ((1u << NUM_WARPS) - 1u));
        if (lane == 0) smemScratch[0] = acc;
    }
    if (broadcast) {
        __syncthreads();
        return smemScratch[0];
    }
    return val;
}

#else  // host / CPU pass: serial fold so the same template type-checks

template <typename Combine, int WARP_WIDTH = 32, typename T>
inline T warpReduce(T val, Combine = {}, const unsigned = 0xFFFFFFFFu) { return val; }

template <typename Combine, int BLOCK_SIZE, typename T>
inline T blockReduce(T val, T*, Combine = {}, const bool = true) { return val; }

#endif

/* Host-side reference fold over a contiguous range — used by unit tests as
 * the oracle, and usable anywhere a serial associative reduction is wanted. */
template <typename Combine, typename T>
FK_HOST_DEVICE_CNST T serialReduce(const T* data, const int n, T init, Combine combine = {}) {
    T acc = init;
    for (int i = 0; i < n; ++i) acc = combine(acc, data[i]);
    return acc;
}

} // namespace fk

#endif // FK_COLLECTIVE_REDUCE_H
