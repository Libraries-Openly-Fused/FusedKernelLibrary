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

/* MmaOp: tensor-core matrix-multiply-accumulate as a composable FKL Op.
 *
 * WHY. The attention kernel calls mma.sync inline PTX with the fragment
 * register layout open-coded at each call site (flash_attention_mma.h, and
 * the validated spikes spike_fp8_mma.cu / spike_fp8_qk_mapping.cu). This
 * header turns one MMA atom into:
 *   1. An ATOM POLICY (MmaBf16_16x8x16, MmaFp8E4M3_16x8x32) carrying the
 *      shape (M,N,K) and per-thread fragment register counts (A/B/D) as part
 *      of the TYPE, plus the raw instruction.
 *   2. MmaOp<Atom> — a standard FKL Op (Parent alias, DECLARE_*_PARENT,
 *      static exec, build()). The accumulate semantics (D += A*B) make it a
 *      ComputeType: its MmaFragment params bundle the per-lane A/B/D
 *      register arrays, so the tensor-core step is addressable BY THE
 *      OPERATION MODEL and composes in a mainloop like any other Op, instead
 *      of being inline asm scattered through a 2000-line kernel.
 *
 * The fragment->thread mapping (which lane owns which element) stays the
 * caller's responsibility — that is the job of a future TiledMma layer — but
 * the instruction and its operand register contract are now a reusable,
 * documented unit. A/B/D_REGS are the per-lane register counts the PTX
 * contract mandates (verified against the fp64 oracle in the spikes).
 *
 * DEVICE-ONLY bodies (they emit SASS); host stubs keep the type usable on
 * the CPU pass so shape constants are queryable and TUs naming the type
 * compile under g++ — the collective layer's host/device-parity discipline. */

#include <fused_kernel/core/utils/utils.h>
#include <fused_kernel/core/data/point.h>
#include <fused_kernel/core/execution_model/operation_model/operation_model.h>
#include <cstdint>

namespace fk {

/* ---- Atom policies: shape-as-type + the raw tensor-core instruction ---- */

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
 * B = 2 x .b32 (8 e4m3), C/D = 4 x .f32. Validated in spike_fp8_mma.cu
 * (mmaFp8) and spike_fp8_qk_mapping.cu. Opt-in: needs
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

/* ---- MmaFragment: the per-lane operand register contract as params ----
 * Pointers into the caller's register/array fragments for one MMA atom.
 * D is in/out (accumulator): exec performs D += A * B. */
template <typename Atom>
struct MmaFragment {
    const typename Atom::AReg* a;   // [Atom::A_REGS]
    const typename Atom::BReg* b;   // [Atom::B_REGS]
    typename Atom::DReg*       d;   // [Atom::D_REGS], in/out accumulator
};

/* ---- MmaOp: standard FKL Op wrapping one tensor-core MMA atom ----
 * ComputeType (accumulate): exec(thread, params) issues D += A*B for the
 * fragments bundled in params. Composes in a mainloop like any other Op. */
template <typename Atom>
struct MmaOp {
private:
    using SelfType = MmaOp<Atom>;
public:
    FK_STATIC_STRUCT(MmaOp, SelfType)
    using AtomType = Atom;
    using Parent = BinaryOperation<MmaFragment<Atom>, MmaFragment<Atom>, void, MmaOp<Atom>>;
    DECLARE_BINARY_PARENT
    static constexpr int M = Atom::M, N = Atom::N, K = Atom::K;

    // ComputeType exec: accumulate one atom (D += A * B) for these fragments.
    // (params IS the MmaFragment; the macro-generated build(ParamsType) applies.)
    FK_HOST_DEVICE_FUSE void exec(const MmaFragment<Atom>& frag) {
        Atom::mma(frag.a, frag.b, frag.d);
    }
};

/* Convenience aliases. */
using MmaBf16Op = MmaOp<MmaBf16_16x8x16>;
using MmaFp8E4M3Op = MmaOp<MmaFp8E4M3_16x8x32>;

} // namespace fk

#endif // FK_COLLECTIVE_MMA_H
