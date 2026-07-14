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

#ifndef FK_COLLECTIVE_MMA_H
#define FK_COLLECTIVE_MMA_H

/* Tensor-core MMA as a cooperative DPP.
 *
 * mma.sync is a WARP-COLLECTIVE instruction: all 32 lanes of the warp execute
 * it together, each contributing its slice of the A/B fragments and receiving
 * its slice of the D accumulator. That is multi-thread work, so per the FKL
 * rule it is a DPP, NOT an Op (an Op is strictly single-thread). MmaWarpDPP
 * wraps one MMA atom; the atom policy carries the shape (M,N,K) and per-lane
 * fragment register counts as part of the TYPE, plus the raw instruction.
 *
 * This replaces the inline mma.sync PTX open-coded at each call site in
 * flash_attention_mma.h (and the validated spikes spike_fp8_mma.cu /
 * spike_fp8_qk_mapping.cu). The per-lane fragment->thread mapping (which lane
 * owns which element) stays the caller's responsibility for now — that is the
 * job of a future TiledMmaDPP — but the warp-collective instruction and its
 * operand register contract become a reusable, documented DPP. A/B/D_REGS are
 * the per-lane register counts the PTX contract mandates (verified against the
 * fp64 oracle in the spikes).
 *
 * DEVICE-ONLY bodies (they emit SASS); host stubs keep the types usable on the
 * CPU pass so shape constants are queryable and TUs naming them compile under
 * g++ — the collective layer's host/device-parity discipline. */

#include <fused_kernel/core/utils/utils.h>
#include <cstdint>

namespace fk {

/* ---- Atom policies: shape-as-type + the raw warp-collective instruction ---- */

/* bf16 x bf16 -> f32, m16n8k16 (Ampere+ canonical HMMA).
 * Per-lane (PTX ISA): A = 4 x .b32 (8 bf16), B = 2 x .b32 (4 bf16),
 * C/D = 4 x .f32. Validated in spikes/spike_fp8_mma.cu (mmaBf16). */
struct MmaBf16_16x8x16 {
    static constexpr int M = 16, N = 8, K = 16;
    static constexpr int A_REGS = 4, B_REGS = 2, D_REGS = 4;
    using AReg = uint32_t;   // packed 2x bf16 per .b32
    using BReg = uint32_t;
    using DReg = float;

#if defined(__NVCC__) && defined(__CUDA_ARCH__)
    __device__ __forceinline__ static void mma(const AReg a[A_REGS],
                                               const BReg b[B_REGS],
                                               DReg d[D_REGS]) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
#else
    static void mma(const AReg[A_REGS], const BReg[B_REGS], DReg[D_REGS]) {}
#endif
};

/* fp8 e4m3 x e4m3 -> f32, kind::f8f6f4 m16n8k32 (sm_120a/sm_121a feature
 * set — plain sm_120 ptxas rejects it). Per-lane: A = 4 x .b32 (16 e4m3),
 * B = 2 x .b32 (8 e4m3), C/D = 4 x .f32. Validated in spike_fp8_mma.cu and
 * spike_fp8_qk_mapping.cu. Opt-in: needs
 * -gencode arch=compute_120a,code=sm_120a + FK_ENABLE_FP8_MMA. */
struct MmaFp8E4M3_16x8x32 {
    static constexpr int M = 16, N = 8, K = 32;
    static constexpr int A_REGS = 4, B_REGS = 2, D_REGS = 4;
    using AReg = uint32_t;   // packed 4x e4m3 per .b32
    using BReg = uint32_t;
    using DReg = float;

#if defined(__NVCC__) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200) && defined(FK_ENABLE_FP8_MMA)
    __device__ __forceinline__ static void mma(const AReg a[A_REGS],
                                               const BReg b[B_REGS],
                                               DReg d[D_REGS]) {
        asm volatile(
            "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
#else
    static void mma(const AReg[A_REGS], const BReg[B_REGS], DReg[D_REGS]) {}
#endif
};

#if defined(__NVCC__)

/* MmaWarpDPP: one warp-collective MMA atom (D += A * B). Each lane passes its
 * own A/B fragment registers and its own D accumulator slice; the 32 lanes
 * cooperate to issue the tensor-core instruction. Multi-thread => DPP. */
template <typename Atom>
struct MmaWarpDPP {
private:
    using SelfType = MmaWarpDPP<Atom>;
public:
    FK_STATIC_STRUCT(MmaWarpDPP, SelfType)
    using AtomType = Atom;
    static constexpr int M = Atom::M, N = Atom::N, K = Atom::K;
    static constexpr int A_REGS = Atom::A_REGS, B_REGS = Atom::B_REGS, D_REGS = Atom::D_REGS;

    // Warp-collective accumulate: D[lane] += A[lane] * B[lane].
    static __device__ __forceinline__
    void exec(const typename Atom::AReg a[Atom::A_REGS],
              const typename Atom::BReg b[Atom::B_REGS],
              typename Atom::DReg d[Atom::D_REGS]) {
        Atom::mma(a, b, d);
    }
};

/* Convenience aliases. */
using MmaBf16WarpDPP = MmaWarpDPP<MmaBf16_16x8x16>;
using MmaFp8E4M3WarpDPP = MmaWarpDPP<MmaFp8E4M3_16x8x32>;

#endif // defined(__NVCC__)

} // namespace fk

#endif // FK_COLLECTIVE_MMA_H
