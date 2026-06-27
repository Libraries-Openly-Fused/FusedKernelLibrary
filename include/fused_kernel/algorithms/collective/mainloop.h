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

#ifndef FK_COLLECTIVE_MAINLOOP_H
#define FK_COLLECTIVE_MAINLOOP_H

/* TileMmaMainloopDPP: the canonical tiled-GEMM inner loop as a DPP that
 * COMPOSES the collective primitives instead of hand-writing it.
 *
 *   D[MxN] = sum over K-tiles of  A_tile[MxK] * B_tile[NxK]
 *
 * Per the FKL DPP contract:
 *   - Inputs A and B are Read IOps, passed as a fk::Tuple<AIOp, BIOp> (an MMA
 *     mainloop needs two read operands — exactly the case the DPP skill cites
 *     for a tuple read IOp). Every operand element enters through
 *     IOp::Operation::exec — no raw device pointers in the exec signature.
 *   - Output D is a Write IOp (the epilogue): the accumulated tile is emitted
 *     through OutIOp::Operation::exec.
 *   - The tensor-core step is MmaWarpDPP (warp-collective). Per the DPP skill,
 *     tensor-core code is the one allowed exception to "a DPP must not contain
 *     data-modifying code"; everything else flows through IOps.
 *   - details (TileMmaMainloopDetails) carries only K (external scalar).
 *
 * The accumulator lives in registers across the whole K loop (online, no
 * global intermediates) — the GEMM analogue of the online-softmax running
 * state. Same structure attention's mainloop has; only the epilogue differs.
 *
 * SCOPE. One warp computes one MmaAtom-shaped output tile (M x N) accumulating
 * over the full K — the minimal composition that proves the primitives stack
 * into a real cooperative kernel, verified against an fp64 oracle. Multi-warp
 * tiling (#279), cp.async pipelining (#276) and a fused epilogue (#282) are
 * additive policies on top of this loop.
 *
 * FRAGMENT MAPPING (which lane reads which element) stays a caller FragLoader
 * policy: it is the delicate, arch-specific part and is independently testable,
 * exactly as MmaWarpDPP keeps the instruction contract separate from mapping.
 * The loader pulls fragments from the value the Read IOp delivers for a Point.
 */

#include <fused_kernel/algorithms/collective/mma.h>
#include <fused_kernel/core/execution_model/parallel_architectures.h>
#include <fused_kernel/core/data/tuple.h>
#include <fused_kernel/core/data/point.h>
#include <fused_kernel/core/utils/utils.h>

namespace fk {

/* External (non-data) parameters: just the K extent of the operand tiles. */
struct TileMmaMainloopDetails {
    int K;
};

template <ParArch PA, typename MmaAtom, typename FragLoader>
struct TileMmaMainloopDPP;

#if defined(__NVCC__)
/* GPU: one warp, one MmaAtom output tile, accumulate over K.
 * ReadIOps is fk::Tuple<AIOp, BIOp>; the FragLoader maps the per-lane A/B
 * fragments out of the operand tiles reached through those IOps. WriteIOp
 * receives the per-lane D accumulator. */
template <typename MmaAtom, typename FragLoader>
struct TileMmaMainloopDPP<ParArch::GPU_NVIDIA, MmaAtom, FragLoader> {
private:
    using SelfType = TileMmaMainloopDPP<ParArch::GPU_NVIDIA, MmaAtom, FragLoader>;
public:
    FK_STATIC_STRUCT(TileMmaMainloopDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::GPU_NVIDIA;
    using AReg = typename MmaAtom::AReg;
    using BReg = typename MmaAtom::BReg;
    using DReg = typename MmaAtom::DReg;

    template <typename ReadIOps, typename WriteIOp>
    static __device__ __forceinline__
    void exec(const TileMmaMainloopDetails& details, const ReadIOps& reads,
              const WriteIOp& output, const FragLoader& loader) {
        const auto& aIOp = get<0>(reads);   // A operand Read IOp
        const auto& bIOp = get<1>(reads);   // B operand Read IOp
        const int lane = threadIdx.x & 31;

        DReg d[MmaAtom::D_REGS];
        #pragma unroll
        for (int i = 0; i < MmaAtom::D_REGS; ++i) d[i] = DReg(0);

        for (int kBase = 0; kBase < details.K; kBase += MmaAtom::K) {
            AReg a[MmaAtom::A_REGS];
            BReg b[MmaAtom::B_REGS];
            loader.loadA(aIOp, lane, kBase, a);   // fragments via the Read IOp
            loader.loadB(bIOp, lane, kBase, b);
            MmaWarpDPP<MmaAtom>::exec(a, b, d);    // warp-collective D += A*B
        }
        loader.storeD(output, lane, d);           // emit via the Write IOp
    }
};
#endif // __NVCC__

/* CPU single-thread implementation (mandatory per the DPP contract): a plain
 * triple loop over the same operand IOps and Write IOp, for host parity. */
template <typename MmaAtom, typename FragLoader>
struct TileMmaMainloopDPP<ParArch::CPU, MmaAtom, FragLoader> {
private:
    using SelfType = TileMmaMainloopDPP<ParArch::CPU, MmaAtom, FragLoader>;
public:
    FK_STATIC_STRUCT(TileMmaMainloopDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::CPU;

    template <typename ReadIOps, typename WriteIOp>
    static inline void exec(const TileMmaMainloopDetails& details, const ReadIOps& reads,
                            const WriteIOp& output, const FragLoader& loader) {
        loader.template cpuGemm<MmaAtom>(get<0>(reads), get<1>(reads), output, details.K);
    }
};

} // namespace fk

#endif // FK_COLLECTIVE_MAINLOOP_H
