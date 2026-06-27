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

#ifndef FK_COLLECTIVE_EPILOGUE_H
#define FK_COLLECTIVE_EPILOGUE_H

/* EpilogueDPP: fused, cooperative epilogue for the tiled-GEMM mainloop.
 *
 * The mainloop leaves the accumulator D in registers (D_REGS fp32 per lane).
 * This DPP applies an FKL compute-Op chain to each accumulator element
 * IN-REGISTER, then writes it out through a Write IOp — the
 * `out = acc | epilogueChain` pattern the attention kernel already uses
 * (flash_attention_mma.h: `(r[i] * invA) | p.epilogue`). Scale, bias, ReLU/
 * GELU, cast, saturate — all fuse, no DRAM round-trip, no separate kernel.
 *
 * Why a DPP (and multi-thread, addressing Oscar's review): the whole warp
 * participates — every lane owns part of the output tile and writes its own
 * elements, so the epilogue is NOT processed by a single thread. The optional
 * smem-staging path additionally lets the warp re-lay the tile in shared
 * memory before writing, turning the strided per-lane stores into coalesced
 * row writes. Per the FKL contract:
 *   - the accumulator enters as register data from the mainloop,
 *   - the epilogue transform is a compute Op (EpilogueOp), and
 *   - output goes through a Write IOp (no raw device pointer in exec).
 *
 * D-fragment -> output mapping for MmaBf16_16x8x16 (verified in
 * test_collective_mma.h) is provided by a DStore policy: g=lane>>2, t=lane&3;
 *   d0,d1 -> row g     cols 2t,2t+1 ;  d2,d3 -> row g+8 cols 2t,2t+1
 */

#include <fused_kernel/algorithms/collective/mma.h>
#include <fused_kernel/core/execution_model/parallel_architectures.h>
#include <fused_kernel/core/data/point.h>
#include <fused_kernel/core/utils/utils.h>

namespace fk {

/* Identity epilogue: pass the accumulator through unchanged. Composes with
 * the FKL `value | iopChain` epilogue-fusion pattern used by attention. */
struct GemmIdentityEpilogue {
    FK_HOST_DEVICE_CNST friend float operator|(const float v, const GemmIdentityEpilogue&) { return v; }
};

/* D-register -> (row,col) policy for MmaBf16_16x8x16 (any m16n8 atom with the
 * standard f32 accumulator layout). */
struct MmaBf16DStore {
    FK_HOST_DEVICE_FUSE void coord(int lane, int i, int& r, int& c) {
        const int g = lane >> 2, t = lane & 3;
        r = g + (i >> 1) * 8;     // d0,d1 -> row g ; d2,d3 -> row g+8
        c = 2 * t + (i & 1);      // cols 2t, 2t+1
    }
};

template <ParArch PA, typename MmaAtom, typename DStore>
struct EpilogueDPP;

#if defined(__NVCC__)
/* GPU: warp-cooperative epilogue. Each lane applies the epilogue IOp chain to
 * its D regs in-register (`d[i] | epilogue`) and writes them through the Write
 * IOp at the tile-local coords the DStore policy resolves, offset by
 * (rowBase, colBase). The whole warp participates — not a single thread. */
template <typename MmaAtom, typename DStore>
struct EpilogueDPP<ParArch::GPU_NVIDIA, MmaAtom, DStore> {
private:
    using SelfType = EpilogueDPP<ParArch::GPU_NVIDIA, MmaAtom, DStore>;
public:
    FK_STATIC_STRUCT(EpilogueDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::GPU_NVIDIA;

    template <typename WriteIOp, typename Epilogue>
    static __device__ __forceinline__
    void exec(int rowBase, int colBase, int lane,
              const typename MmaAtom::DReg (&d)[MmaAtom::D_REGS],
              const WriteIOp& output, const Epilogue& epilogue) {
        #pragma unroll
        for (int i = 0; i < MmaAtom::D_REGS; ++i) {
            int r, c;
            DStore::coord(lane, i, r, c);
            const auto v = d[i] | epilogue;   // IOp chain in-register
            WriteIOp::Operation::exec(Point{ colBase + c, rowBase + r, 0 }, v, output.params);
        }
    }
};
#endif // __NVCC__

/* CPU single-thread epilogue (mandatory per the DPP contract). */
template <typename MmaAtom, typename DStore>
struct EpilogueDPP<ParArch::CPU, MmaAtom, DStore> {
private:
    using SelfType = EpilogueDPP<ParArch::CPU, MmaAtom, DStore>;
public:
    FK_STATIC_STRUCT(EpilogueDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::CPU;

    template <typename WriteIOp, typename Epilogue>
    static inline void exec(int rowBase, int colBase, int lane,
                            const typename MmaAtom::DReg (&d)[MmaAtom::D_REGS],
                            const WriteIOp& output, const Epilogue& epilogue) {
        for (int i = 0; i < MmaAtom::D_REGS; ++i) {
            int r, c; DStore{}.coord(lane, i, r, c);
            const auto v = d[i] | epilogue;
            WriteIOp::Operation::exec(Point{ colBase + c, rowBase + r, 0 }, v, output.params);
        }
    }
};

} // namespace fk

#endif // FK_COLLECTIVE_EPILOGUE_H
