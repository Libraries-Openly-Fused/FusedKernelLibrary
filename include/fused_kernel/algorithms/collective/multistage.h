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

#ifndef FK_COLLECTIVE_MULTISTAGE_H
#define FK_COLLECTIVE_MULTISTAGE_H

/* MultiStageMainloopDPP: multi-stage (N-buffer) software pipeline for the
 * register-tiled GEMM mainloop. Generalises the 2-stage double buffer to
 * STAGES shared-memory buffers, keeping STAGES-1 cp.async groups in flight so
 * more global-load latency is hidden before the math has to wait. STAGES==2
 * reduces to the classic double buffer.
 *
 * FKL conformance: this DPP owns only the PIPELINE CONTROL FLOW (ring
 * rotation, commit/wait fences built from the collective copy primitive).
 * The actual data movement and compute go through IOps via a StageLoader
 * policy: stage(reads, buf, kTile) issues the async load for a K-tile into a
 * ring slot through the operand Read IOps, and compute(buf, output) runs the
 * register-tiled warp body (WarpTileMmaMainloopDPP's body) on a ring slot,
 * writing through the Write IOp. No raw device pointer in exec; operands and
 * result flow as a fk::Tuple<AIOp,BIOp> Read pair + a Write IOp.
 *
 * Caller owns the smem ring (STAGES buffers for A and B). numTiles >= 1;
 * degrades correctly when numTiles < STAGES (drains early). */

#include <fused_kernel/algorithms/collective/copy.h>
#include <fused_kernel/algorithms/collective/register_tile.h>
#include <fused_kernel/core/execution_model/parallel_architectures.h>
#include <fused_kernel/core/data/tuple.h>
#include <fused_kernel/core/utils/utils.h>

namespace fk {

struct MultiStageDetails { int numTiles; };

template <ParArch PA, int STAGES, typename StagePolicy>
struct MultiStageMainloopDPP;

#if defined(__NVCC__)
template <int STAGES, typename StagePolicy>
struct MultiStageMainloopDPP<ParArch::GPU_NVIDIA, STAGES, StagePolicy> {
private:
    using SelfType = MultiStageMainloopDPP<ParArch::GPU_NVIDIA, STAGES, StagePolicy>;
public:
    FK_STATIC_STRUCT(MultiStageMainloopDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::GPU_NVIDIA;
    static_assert(STAGES >= 2, "need at least 2 stages");

    template <typename ReadIOps, typename WriteIOp, typename SmemRing>
    static __device__ __forceinline__
    void exec(const MultiStageDetails& details, const ReadIOps& reads,
              const WriteIOp& output, StagePolicy& stage, SmemRing& ring) {
        const int numTiles = details.numTiles;

        // Prologue: launch the first STAGES-1 tiles, one cp.async group each.
        const int prefetch = (numTiles < STAGES - 1) ? numTiles : (STAGES - 1);
        #pragma unroll
        for (int p = 0; p < STAGES - 1; ++p) {
            if (p < prefetch) { stage.stage(reads, p % STAGES, p, ring); cpAsyncCommit(); }
        }
        // Steady state.
        for (int s = 0; s < numTiles; ++s) {
            const int inFlight = (numTiles - s < STAGES - 1) ? (numTiles - s - 1) : (STAGES - 2);
            switch (inFlight) {
                case 0: cpAsyncWaitGroups<0>(); break;
                case 1: cpAsyncWaitGroups<1>(); break;
                case 2: cpAsyncWaitGroups<2>(); break;
                default: cpAsyncWaitGroups<3>(); break;   // STAGES<=5 covers all
            }
            __syncthreads();
            stage.compute(s % STAGES, ring);              // accumulate this tile
            __syncthreads();
            const int next = s + STAGES - 1;
            if (next < numTiles) { stage.stage(reads, next % STAGES, next, ring); cpAsyncCommit(); }
        }
        stage.epilogue(output);                           // emit via Write IOp
    }
};
#endif // __NVCC__

/* CPU single-thread implementation (mandatory per the DPP contract): no
 * pipelining needed — stage+compute every tile in order, then write out. */
template <int STAGES, typename StagePolicy>
struct MultiStageMainloopDPP<ParArch::CPU, STAGES, StagePolicy> {
private:
    using SelfType = MultiStageMainloopDPP<ParArch::CPU, STAGES, StagePolicy>;
public:
    FK_STATIC_STRUCT(MultiStageMainloopDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::CPU;

    template <typename ReadIOps, typename WriteIOp, typename SmemRing>
    static inline void exec(const MultiStageDetails& details, const ReadIOps& reads,
                            const WriteIOp& output, StagePolicy& stage, SmemRing& ring) {
        for (int s = 0; s < details.numTiles; ++s) { stage.stage(reads, 0, s, ring); stage.compute(0, ring); }
        stage.epilogue(output);
    }
};

} // namespace fk

#endif // FK_COLLECTIVE_MULTISTAGE_H
