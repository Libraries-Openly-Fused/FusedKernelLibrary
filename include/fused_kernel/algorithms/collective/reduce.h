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

/* Cross-thread reductions as composable FKL DPPs.
 *
 * WHY THIS SHAPE. Cooperative algorithms (SoftmaxDPP today; the tiled
 * mainloop to come) all need cross-thread reductions: row-max and row-sum
 * for softmax, accumulation for GEMM, norms for layernorm. Per the FKL
 * philosophy these are NOT bespoke functions with bespoke combine functors
 * — the reduction operator is just a standard FKL BINARY Op (Add, Max, Min,
 * Mul, ...) passed as a template argument, and the reduction itself is a
 * Data Parallel Pattern. Three tiers, one per cooperation scope, matching
 * the hardware:
 *
 *   ReduceWarpDPP<ReduceOp, WARP_WIDTH>   intra-warp, __shfl_xor butterfly,
 *                                         no smem, no barrier. All lanes get
 *                                         the result. Sub-warp masks allow
 *                                         segmented (per-row) reductions.
 *   ReduceBlockDPP<ReduceOp, BLOCK_SIZE>  whole block: each warp reduces,
 *                                         lane0 stages its partial to
 *                                         caller-owned smem, warp 0 finishes.
 *                                         The kernel owns its smem budget
 *                                         (matches the cooperative-DPP
 *                                         convention).
 *   ReduceGridDPP<ReduceOp, BLOCK_SIZE>   whole grid: a block-reduce per CTA
 *                                         followed by an atomic combine of
 *                                         the per-block partials into a
 *                                         single global accumulator.
 *
 * THE REDUCE OP is any FKL binary Op, i.e. a static struct exposing
 *     FK_HOST_DEVICE_FUSE O exec(const I a, const P b);     // associative
 * The library already ships Add / Max / Min / Mul (basic_ops). The softmax
 * merge state is itself expressible as a binary Op over OnlineSoftmaxState,
 * so the SAME DPP reduces scalars AND online-softmax states — no new combine
 * type, no duplicated reduction code.
 *
 * These exec bodies call __shfl_xor / __syncthreads, so (like the other
 * cooperative DPP exec bodies) they are NOT constexpr. On the host pass
 * (g++ CPU TU, or nvcc host pass) they degrade to a plain serial fold so
 * the same template type-checks and unit tests run on CPU.
 */

#include <fused_kernel/core/execution_model/operation_model/operation_model.h>
#include <fused_kernel/core/utils/utils.h>
#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/execution_model/stream.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/logical.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <type_traits>

namespace fk {

// Cooperative-DPP exec bodies use __shfl/__shared__/barriers: they cannot be
// constexpr (FK_DEVICE_FUSE). Plain static device inline qualifier:
#define FK_COOP_DEVICE_FUSE static __device__ __forceinline__

/* ReduceWarpDPP: intra-warp all-reduce parameterised by a standard FKL
 * binary Op. WARP_WIDTH must be a power of two <= 32. Every participating
 * lane in the mask receives the fully reduced value. mask defaults to the
 * full warp; pass a sub-warp mask + WARP_WIDTH for segmented reductions. */
template <typename ReduceOp, int WARP_WIDTH = 32>
struct ReduceWarpDPP {
private:
    using SelfType = ReduceWarpDPP<ReduceOp, WARP_WIDTH>;
public:
    FK_STATIC_STRUCT(ReduceWarpDPP, SelfType)
    static_assert(WARP_WIDTH > 0 && WARP_WIDTH <= 32 && (WARP_WIDTH & (WARP_WIDTH - 1)) == 0,
                  "WARP_WIDTH must be a power of two in (0, 32]");

#if defined(__NVCC__) && defined(__CUDA_ARCH__)
    template <typename T>
    FK_COOP_DEVICE_FUSE T exec(T val, const unsigned mask = 0xFFFFFFFFu) {
        #pragma unroll
        for (int offset = WARP_WIDTH / 2; offset > 0; offset >>= 1) {
            const T other = __shfl_xor_sync(mask, val, offset, WARP_WIDTH);
            val = ReduceOp::exec(val, other);   // standard FKL binary Op
        }
        return val;
    }
#else   // host / CPU pass: single thread already holds the whole partial
    template <typename T>
    static inline T exec(T val, const unsigned = 0xFFFFFFFFu) { return val; }
#endif
};

/* ReduceBlockDPP: whole-block all-reduce parameterised by a standard FKL
 * binary Op. Uses a caller-provided smem scratch of >= (BLOCK_SIZE/32)
 * elements so the DPP allocates nothing itself (the kernel owns its smem
 * budget — matches how the cooperative DPPs manage shared memory). With
 * broadcast=true every thread receives the result; otherwise it is returned
 * on lane0 of warp0. */
template <typename ReduceOp, int BLOCK_SIZE>
struct ReduceBlockDPP {
private:
    using SelfType = ReduceBlockDPP<ReduceOp, BLOCK_SIZE>;
public:
    FK_STATIC_STRUCT(ReduceBlockDPP, SelfType)
    static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a multiple of 32");
    static constexpr int NUM_WARPS = BLOCK_SIZE / 32;

#if defined(__NVCC__) && defined(__CUDA_ARCH__)
    template <typename T>
    FK_COOP_DEVICE_FUSE T exec(T val, T* smemScratch, const bool broadcast = true) {
        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;

        val = ReduceWarpDPP<ReduceOp, 32>::exec(val);   // reduce within each warp
        if (lane == 0) smemScratch[warp] = val;         // stage warp partials
        __syncthreads();

        T blockResult = val;
        if (warp == 0) {
            T acc = (lane < NUM_WARPS) ? smemScratch[lane] : smemScratch[0];
            acc = ReduceWarpDPP<ReduceOp, (NUM_WARPS > 1 ? NUM_WARPS : 1)>::exec(
                      acc, (NUM_WARPS >= 32) ? 0xFFFFFFFFu : ((1u << NUM_WARPS) - 1u));
            if (lane == 0) smemScratch[0] = acc;
            blockResult = acc;   // full block reduction lives on lane0/warp0
        }
        if (broadcast) {
            __syncthreads();
            return smemScratch[0];
        }
        // non-broadcast: full result is valid on lane0 of warp0 (smemScratch[0])
        return blockResult;
    }
#else   // host / CPU pass
    template <typename T>
    static inline T exec(T val, T*, const bool = true) { return val; }
#endif
};

/* ReduceGridDPP: whole-grid all-reduce parameterised by a standard FKL
 * binary Op. Each CTA computes a block-level partial (ReduceBlockDPP), then
 * lane0/warp0 combines its partial into a single global accumulator via an
 * atomic apply of ReduceOp. The caller initialises *gAccum to the Op's
 * identity for the data range and reads it back after the grid completes.
 * atomicApply specialises to the hardware atomic for the common Ops
 * (Add->atomicAdd, Max->atomicMax, Min->atomicMin); other Ops fall back to a
 * CAS loop so any associative Op composes. */
template <typename ReduceOp, int BLOCK_SIZE>
struct ReduceGridDPP {
private:
    using SelfType = ReduceGridDPP<ReduceOp, BLOCK_SIZE>;
public:
    FK_STATIC_STRUCT(ReduceGridDPP, SelfType)
    static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a multiple of 32");

#if defined(__NVCC__) && defined(__CUDA_ARCH__)
    template <typename T>
    FK_COOP_DEVICE_FUSE void exec(T val, T* smemScratch, T* gAccum) {
        const T blockPartial =
            ReduceBlockDPP<ReduceOp, BLOCK_SIZE>::exec(val, smemScratch, /*broadcast=*/false);
        if ((threadIdx.x & 31) == 0 && (threadIdx.x >> 5) == 0) {
            atomicApply(gAccum, blockPartial);
        }
    }

private:
    // Hardware atomics for the Ops the library ships; CAS fallback otherwise.
    template <typename T>
    static __device__ __forceinline__ void atomicApply(T* addr, T v) {
        if constexpr (std::is_same_v<ReduceOp, Add<T, T, T, BinaryType>>) {
            atomicAdd(addr, v);
        } else if constexpr (std::is_same_v<ReduceOp, Max<T, T, T, BinaryType>>) {
            atomicMaxOp(addr, v);
        } else if constexpr (std::is_same_v<ReduceOp, Min<T, T, T, BinaryType>>) {
            atomicMinOp(addr, v);
        } else {
            casApply(addr, v);
        }
    }

    // Generic CAS apply of an arbitrary associative ReduceOp (float/int 32-bit).
    template <typename T>
    static __device__ __forceinline__ void casApply(T* addr, T v) {
        static_assert(sizeof(T) == 4, "grid CAS fallback supports 32-bit types");
        unsigned* uaddr = reinterpret_cast<unsigned*>(addr);
        unsigned old = *uaddr, assumed;
        do {
            assumed = old;
            T cur; __builtin_memcpy(&cur, &assumed, 4);
            const T nw = ReduceOp::exec(cur, v);
            unsigned nu; __builtin_memcpy(&nu, &nw, 4);
            old = atomicCAS(uaddr, assumed, nu);
        } while (assumed != old);
    }

    // float atomic max/min via int reinterpretation (monotonic ordering trick).
    static __device__ __forceinline__ void atomicMaxOp(float* addr, float v) {
        if (v >= 0.f) atomicMax(reinterpret_cast<int*>(addr), __float_as_int(v));
        else          atomicMin(reinterpret_cast<unsigned*>(addr), __float_as_uint(v));
    }
    static __device__ __forceinline__ void atomicMinOp(float* addr, float v) {
        if (v >= 0.f) atomicMin(reinterpret_cast<int*>(addr), __float_as_int(v));
        else          atomicMax(reinterpret_cast<unsigned*>(addr), __float_as_uint(v));
    }
    template <typename T>
    static __device__ __forceinline__ void atomicMaxOp(T* addr, T v) { atomicMax(addr, v); }
    template <typename T>
    static __device__ __forceinline__ void atomicMinOp(T* addr, T v) { atomicMin(addr, v); }
#else   // host / CPU pass
    template <typename T>
    static inline void exec(T val, T*, T* gAccum) { if (gAccum) *gAccum = ReduceOp::exec(*gAccum, val); }
#endif
};

/* ReduceDPP: user-facing row-wise reduction as a composable FKL DPP.
 *
 * Honours the full FKL contract: the input is a Read IOp (the prologue) and
 * the output is a Write IOp (the epilogue), so any fused preprocessing chain
 * (read.then(Mul...).then(Cast...)) runs in-register at load time, and the
 * single reduced value per row is emitted through whatever Write IOp the
 * caller supplies (PerThreadWrite, TensorWrite, a fused post-op chain, ...).
 * The reduction operator is a STANDARD FKL binary Op (Add / Max / Min, ...)
 * passed as a template argument — no bespoke combine types. Element
 * addressing matches SoftmaxDPP: thread.x = column, blockIdx.x = row.
 *
 *   InIOp   instantiable Read / ReadBack IOp (the prologue)
 *   OutIOp  instantiable Write IOp (the epilogue); receives one value per row
 *   ReduceOp standard FKL binary Op, invoked as ReduceOp::exec(a, b)
 */

/* DPP details: external (non-data) parameters only — no IOps, no raw
 * pointers. Per the FKL DPP contract, data is reached exclusively through the
 * Read/Write IOps passed to exec(); Details carries just the scalars. */
template <typename T>
struct ReduceDPPDetails {
    int width;       // row length (number of columns to reduce)
    int rows;        // number of rows (grid.x)
    T   identity;    // ReduceOp identity (0 for Add, -inf for Max, ...)
};

template <enum ParArch PA, typename ReduceOp, int BLOCK_SIZE = 256>
struct ReduceDPP;

#if defined(__NVCC__)
template <typename ReduceOp, int BLOCK_SIZE>
struct ReduceDPP<ParArch::GPU_NVIDIA, ReduceOp, BLOCK_SIZE> {
private:
    using SelfType = ReduceDPP<ParArch::GPU_NVIDIA, ReduceOp, BLOCK_SIZE>;
public:
    FK_STATIC_STRUCT(ReduceDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::GPU_NVIDIA;
    static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a multiple of 32");

    /* exec(details, readIOp, writeIOp): data is reached only through the IOps;
     * details holds the external scalars. One reduced value per row. */
    template <typename InIOp, typename OutIOp, typename T>
    static __device__ __forceinline__
    void exec(const ReduceDPPDetails<T>& details, const InIOp& input, const OutIOp& output) {
        static_assert(isAnyReadType<InIOp>,  "ReduceDPP read IOp must be a Read/ReadBack IOp");
        static_assert(isAnyWriteType<OutIOp>, "ReduceDPP write IOp must be a Write IOp");
        __shared__ T scratch[BLOCK_SIZE / 32];

        const int row = blockIdx.x;
        const int tid = threadIdx.x;

        T acc = details.identity;
        for (int x = tid; x < details.width; x += BLOCK_SIZE) {
            acc = ReduceOp::exec(acc, static_cast<T>(InIOp::Operation::exec(Point{ x, row, 0 }, input)));
        }
        const T result = ReduceBlockDPP<ReduceOp, BLOCK_SIZE>::exec(acc, scratch, /*broadcast=*/false);
        if (tid == 0) {
            OutIOp::Operation::exec(Point{ row, 0, 0 }, result, output.params);
        }
    }
};
#endif // defined(__NVCC__)

/* Single-thread CPU implementation (mandatory per the DPP contract). */
template <typename ReduceOp, int BLOCK_SIZE>
struct ReduceDPP<ParArch::CPU, ReduceOp, BLOCK_SIZE> {
private:
    using SelfType = ReduceDPP<ParArch::CPU, ReduceOp, BLOCK_SIZE>;
public:
    FK_STATIC_STRUCT(ReduceDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::CPU;

    template <typename InIOp, typename OutIOp, typename T>
    static inline void exec(const ReduceDPPDetails<T>& details, const InIOp& input, const OutIOp& output) {
        for (int row = 0; row < details.rows; ++row) {
            T acc = details.identity;
            for (int x = 0; x < details.width; ++x) {
                acc = ReduceOp::exec(acc, static_cast<T>(InIOp::Operation::exec(Point{ x, row, 0 }, input)));
            }
            OutIOp::Operation::exec(Point{ row, 0, 0 }, acc, output.params);
        }
    }
};

#if defined(__NVCC__)

template <typename ReduceOp, int BLOCK_SIZE, typename InIOp, typename OutIOp, typename T>
__global__ void launchReduceDPP_Kernel(const __grid_constant__ ReduceDPPDetails<T> details,
                                       const __grid_constant__ InIOp input,
                                       const __grid_constant__ OutIOp output) {
    ReduceDPP<ParArch::GPU_NVIDIA, ReduceOp, BLOCK_SIZE>::exec(details, input, output);
}

/* IOp-first API: input is the prologue Read IOp, output the epilogue Write
 * IOp; one reduced value per row. ReduceOp and identity define the reduction. */
template <typename ReduceOp, int BLOCK_SIZE = 256, typename InIOp, typename OutIOp, typename T>
inline void executeReduce(const InIOp& input, const OutIOp& output,
                          const int rows, const int width, const T identity,
                          Stream_<ParArch::GPU_NVIDIA>& stream) {
    const ReduceDPPDetails<T> details{ width, rows, identity };
    const dim3 grid(rows, 1, 1);
    const dim3 block(BLOCK_SIZE, 1, 1);
    launchReduceDPP_Kernel<ReduceOp, BLOCK_SIZE, InIOp, OutIOp, T>
        <<<grid, block, 0, stream.getCUDAStream()>>>(details, input, output);
    gpuErrchk(cudaGetLastError());
}

#endif // defined(__NVCC__)

/* Host-side reference fold over a contiguous range, parameterised by the
 * same standard FKL binary Op — the unit-test oracle, and usable anywhere a
 * serial associative reduction is wanted. */
template <typename ReduceOp, typename T>
FK_HOST_DEVICE_CNST T serialReduce(const T* data, const int n, T init) {
    T acc = init;
    for (int i = 0; i < n; ++i) acc = ReduceOp::exec(acc, data[i]);
    return acc;
}

#undef FK_COOP_DEVICE_FUSE

} // namespace fk

#endif // FK_COLLECTIVE_REDUCE_H
