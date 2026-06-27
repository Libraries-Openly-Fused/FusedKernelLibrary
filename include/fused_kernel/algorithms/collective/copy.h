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

#ifndef FK_COLLECTIVE_COPY_H
#define FK_COLLECTIVE_COPY_H

/* Asynchronous global->shared staging as a cooperative DPP.
 *
 * Every tiled mainloop overlaps the next tile's load with the current tile's
 * math via 16-byte cp.async.cg. Staging a tile is multi-thread work (the whole
 * block fills the shared buffer), so per the FKL rule it is a DPP, not an Op.
 *
 *   1. detail:: low-level primitives (cpAsync16 / commit / wait) — the raw
 *      PTX, degrading to a synchronous 16-byte copy pre-Ampere and to memcpy
 *      on the host pass (FKL host/device parity).
 *   2. commit / wait FENCES — pipelining policy (how many groups in flight),
 *      exposed as thin free helpers. They order cp.async groups, they don't
 *      move data, so they are control-flow fences the caller / MainloopDPP
 *      drives, not Ops.
 *   3. CpAsyncStageDPP — the cooperative DPP: the block stages N 16-byte
 *      chunks gmem->smem across its threads. */

#include <fused_kernel/core/utils/utils.h>
#include <cstdint>
#include <cstring>

namespace fk {

namespace detail {

#if defined(__NVCC__) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

// 16-byte (128-bit) async copy global->shared, cache-global.
__device__ __forceinline__ void cpAsync16(void* smemDst, const void* gmemSrc) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smemDst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmemSrc));
}
__device__ __forceinline__ void cpAsyncCommit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
__device__ __forceinline__ void cpAsyncWaitGroups() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

#elif defined(__NVCC__) && defined(__CUDA_ARCH__)

// Pre-Ampere device: synchronous 16-byte copy, fences no-ops.
__device__ __forceinline__ void cpAsync16(void* smemDst, const void* gmemSrc) {
    *reinterpret_cast<int4*>(smemDst) = *reinterpret_cast<const int4*>(gmemSrc);
}
__device__ __forceinline__ void cpAsyncCommit() {}
template <int N> __device__ __forceinline__ void cpAsyncWaitGroups() {}

#else

// Host pass: plain memcpy, fences no-ops (keeps the TU compiling on g++).
inline void cpAsync16(void* dst, const void* src) { std::memcpy(dst, src, 16); }
inline void cpAsyncCommit() {}
template <int N> inline void cpAsyncWaitGroups() {}

#endif

} // namespace detail

/* Pipelining fences — control-flow policy for the caller / MainloopDPP.
 * Not constexpr (they emit asm), so plain inline qualifiers. */
#if defined(__NVCC__)
__host__ __device__ __forceinline__ void cpAsyncCommit() { detail::cpAsyncCommit(); }
template <int N>
__host__ __device__ __forceinline__ void cpAsyncWaitGroups() { detail::cpAsyncWaitGroups<N>(); }
#else
inline void cpAsyncCommit() { detail::cpAsyncCommit(); }
template <int N> inline void cpAsyncWaitGroups() { detail::cpAsyncWaitGroups<N>(); }
#endif

#if defined(__NVCC__)

/* CpAsyncStageDPP: cooperatively stage CHUNKS 16-byte chunks gmem->smem across
 * the block's threads through the async pipe. The caller issues a commit + wait
 * fence afterwards (pipelining policy). T is the 16-byte vector element. The
 * kernel owns the shared buffer; this DPP just fills it. Multi-thread => DPP. */
template <typename T = int4, int BLOCK_SIZE = 256>
struct CpAsyncStageDPP {
private:
    using SelfType = CpAsyncStageDPP<T, BLOCK_SIZE>;
public:
    FK_STATIC_STRUCT(CpAsyncStageDPP, SelfType)
    static_assert(sizeof(T) == 16, "CpAsyncStageDPP stages 16-byte chunks");

    // Stage `chunks` 16-byte elements gmem->smem, strided across the block.
    static __device__ __forceinline__
    void stage(T* smemDst, const T* gmemSrc, const int chunks) {
        for (int i = threadIdx.x; i < chunks; i += BLOCK_SIZE) {
            detail::cpAsync16(&smemDst[i], &gmemSrc[i]);
        }
    }
};

#endif // defined(__NVCC__)

} // namespace fk

#endif // FK_COLLECTIVE_COPY_H
