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

/* Asynchronous global->shared staging copy as a composable FKL Op.
 *
 * Every tiled mainloop overlaps the next tile's load with the current tile's
 * math via a 16-byte cp.async.cg. This header exposes that as:
 *   1. detail:: low-level primitives (cpAsync16 / commit / wait) — the raw
 *      PTX, degrading to a synchronous 16-byte copy pre-Ampere and to memcpy
 *      on the host pass (FKL host/device parity).
 *   2. CpAsyncStage16Op — a standard FKL Op (Parent alias, DECLARE_*_PARENT,
 *      static exec, build()) that stages one 16-byte chunk gmem->smem through
 *      the operation model. The commit/wait FENCES stay the caller's /
 *      MainloopDPP's pipelining policy (how many groups in flight), exposed
 *      as thin free helpers — they are control-flow fences, not data Ops. */

#include <fused_kernel/core/utils/utils.h>
#include <fused_kernel/core/data/point.h>
#include <fused_kernel/core/execution_model/operation_model/operation_model.h>
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

/* Pipelining fences — control-flow policy exposed to the caller / MainloopDPP.
 * (Not data Ops: they order cp.async groups, they don't move elements.)
 * Not constexpr (they emit asm), so plain inline qualifiers, not FK_*_FUSE. */
#if defined(__NVCC__)
__host__ __device__ __forceinline__ void cpAsyncCommit() { detail::cpAsyncCommit(); }
template <int N>
__host__ __device__ __forceinline__ void cpAsyncWaitGroups() { detail::cpAsyncWaitGroups<N>(); }
#else
inline void cpAsyncCommit() { detail::cpAsyncCommit(); }
template <int N> inline void cpAsyncWaitGroups() { detail::cpAsyncWaitGroups<N>(); }
#endif

/* ---- CpAsyncStage16Op: stage one 16B chunk gmem->smem as an FKL Op ----
 * Params bundle the destination smem base and source gmem base; exec stages
 * the chunk at Point.x (in 16-byte units). A mainloop issues these across
 * threads, then a commit + wait fence, like PerThreadRead but landing in
 * shared memory through the async pipe. T is the 16-byte vector element.
 * Device-only (emits cp.async asm), so it does not derive the constexpr
 * ReadOperation parent; it follows the same static-struct + Params + exec +
 * build() Op shape the rest of the collective layer uses. */
template <typename T = int4>
struct CpAsyncStage16Op {
private:
    using SelfType = CpAsyncStage16Op<T>;
public:
    FK_STATIC_STRUCT(CpAsyncStage16Op, SelfType)
    static_assert(sizeof(T) == 16, "CpAsyncStage16Op stages 16-byte chunks");

    struct Params {
        T*       smemDst;   // shared-memory destination base
        const T* gmemSrc;   // global-memory source base
    };

    // Stage chunk index Point.x (16-byte granularity). The data lands in smem
    // asynchronously. Not constexpr (emits cp.async asm).
#if defined(__NVCC__)
    __host__ __device__ __forceinline__
#else
    static inline
#endif
    static void exec(const Point thread, const Params& params) {
        detail::cpAsync16(&params.smemDst[thread.x], &params.gmemSrc[thread.x]);
    }
    FK_HOST_FUSE Params build(T* smemDst, const T* gmemSrc) {
        return Params{ smemDst, gmemSrc };
    }
};

} // namespace fk

#endif // FK_COLLECTIVE_COPY_H
