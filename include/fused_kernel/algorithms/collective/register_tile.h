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

#ifndef FK_COLLECTIVE_REGISTER_TILE_H
#define FK_COLLECTIVE_REGISTER_TILE_H

/* WarpTileMmaMainloopDPP: register-tiled tiled-GEMM mainloop — one WARP
 * computes a WM x WN grid of MmaAtom output tiles, accumulating over K.
 *
 * WHY. TileMmaMainloopDPP does ONE MmaAtom (16x8) per warp, so the
 * load:compute ratio is fixed and low. A fast GEMM amortizes loads: each warp
 * holds a WM x WN block of accumulators and, per K-step, loads WM A-fragments
 * + WN B-fragments ONCE and issues WM*WN MMAs — every A-fragment reused WN
 * times, every B-fragment WM times, in registers. That reuse turns the kernel
 * compute-bound (cf. CUTLASS warp-tile).
 *
 * FKL conformance (addressing the review — no faked IOps; data flows through
 * real Read/Write IOps): operands A and B arrive as a fk::Tuple<AIOp,BIOp>
 * Read pair, the WM x WN result block leaves through a Write IOp, and the
 * tensor-core step is MmaWarpDPP (the one allowed data-modifying exception).
 * No raw device/smem pointer appears in the exec signature. The per-lane,
 * per-sub-tile fragment mapping stays a pluggable RegTileFragLoader policy
 * that reads operand elements via IOp::Operation::exec.
 *
 * Accumulator d[WM][WN][D_REGS] is caller-owned, register-resident across the
 * whole K loop (online, no global intermediates). */

#include <fused_kernel/algorithms/collective/mma.h>
#include <fused_kernel/core/execution_model/parallel_architectures.h>
#include <fused_kernel/core/data/tuple.h>
#include <fused_kernel/core/data/point.h>
#include <fused_kernel/core/utils/utils.h>

namespace fk {

/* External (non-data) parameters: K extent + the warp-tile shape is in the
 * template, so details carries only K here. */
struct WarpTileMainloopDetails { int K; };

template <ParArch PA, typename MmaAtom, int WM, int WN, typename RegTileFragLoader>
struct WarpTileMmaMainloopDPP;

#if defined(__NVCC__)
template <typename MmaAtom, int WM, int WN, typename RegTileFragLoader>
struct WarpTileMmaMainloopDPP<ParArch::GPU_NVIDIA, MmaAtom, WM, WN, RegTileFragLoader> {
private:
    using SelfType = WarpTileMmaMainloopDPP<ParArch::GPU_NVIDIA, MmaAtom, WM, WN, RegTileFragLoader>;
public:
    FK_STATIC_STRUCT(WarpTileMmaMainloopDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::GPU_NVIDIA;
    using AReg = typename MmaAtom::AReg;
    using BReg = typename MmaAtom::BReg;
    using DReg = typename MmaAtom::DReg;

    template <typename ReadIOps, typename WriteIOp>
    static __device__ __forceinline__
    void exec(const WarpTileMainloopDetails& details, const ReadIOps& reads,
              const WriteIOp& output, const RegTileFragLoader& loader) {
        const auto& aIOp = get<0>(reads);
        const auto& bIOp = get<1>(reads);
        const int lane = threadIdx.x & 31;

        DReg d[WM][WN][MmaAtom::D_REGS];
        #pragma unroll
        for (int i = 0; i < WM; ++i)
            #pragma unroll
            for (int j = 0; j < WN; ++j)
                #pragma unroll
                for (int r = 0; r < MmaAtom::D_REGS; ++r) d[i][j][r] = DReg(0);

        for (int kBase = 0; kBase < details.K; kBase += MmaAtom::K) {
            // Load WM A-fragments and WN B-fragments ONCE through the Read IOps.
            AReg a[WM][MmaAtom::A_REGS];
            BReg b[WN][MmaAtom::B_REGS];
            #pragma unroll
            for (int i = 0; i < WM; ++i) loader.loadA(aIOp, i, kBase, lane, a[i]);
            #pragma unroll
            for (int j = 0; j < WN; ++j) loader.loadB(bIOp, j, kBase, lane, b[j]);
            // Issue WM*WN MMAs reusing the fragments from registers.
            #pragma unroll
            for (int i = 0; i < WM; ++i)
                #pragma unroll
                for (int j = 0; j < WN; ++j)
                    MmaWarpDPP<MmaAtom>::exec(a[i], b[j], d[i][j]);
        }
        // Emit the WM x WN accumulator block through the Write IOp.
        #pragma unroll
        for (int i = 0; i < WM; ++i)
            #pragma unroll
            for (int j = 0; j < WN; ++j)
                loader.storeD(output, i, j, lane, d[i][j]);
    }
};
#endif // __NVCC__

/* CPU single-thread implementation (mandatory per the DPP contract). */
template <typename MmaAtom, int WM, int WN, typename RegTileFragLoader>
struct WarpTileMmaMainloopDPP<ParArch::CPU, MmaAtom, WM, WN, RegTileFragLoader> {
private:
    using SelfType = WarpTileMmaMainloopDPP<ParArch::CPU, MmaAtom, WM, WN, RegTileFragLoader>;
public:
    FK_STATIC_STRUCT(WarpTileMmaMainloopDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::CPU;

    template <typename ReadIOps, typename WriteIOp>
    static inline void exec(const WarpTileMainloopDetails& details, const ReadIOps& reads,
                            const WriteIOp& output, const RegTileFragLoader& loader) {
        loader.template cpuGemm<MmaAtom, WM, WN>(get<0>(reads), get<1>(reads), output, details.K);
    }
};

} // namespace fk

#endif // FK_COLLECTIVE_REGISTER_TILE_H
