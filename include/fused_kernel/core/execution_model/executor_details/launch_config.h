/* Copyright 2025 Grup Mediapro S.L.U (Oscar Amoros Huguet)
   Copyright 2026 Oscar Amoros Huguet, Johnny Nunez

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_LAUNCH_CONFIG_H
#define FK_LAUNCH_CONFIG_H

#include <fused_kernel/core/utils/utils.h>
#include <fused_kernel/core/execution_model/active_threads.h>

#include <climits>
#include <cmath>

namespace fk {

    struct CtxDim3 {
        uint x;
        uint y;
        uint z;
#if defined(__NVCC__)
        constexpr CtxDim3(const dim3& dims) : x(dims.x), y(dims.y), z(dims.z) {}
#endif
        constexpr CtxDim3() : x(1), y(1), z(1) {}
        constexpr CtxDim3(const uint& x) : x(x), y(1), z(1) {}
        constexpr CtxDim3(const uint& x, const uint& y) : x(x), y(y), z(1) {}
        constexpr CtxDim3(const uint& x, const uint& y, const uint& z) : x(x), y(y), z(z) {}
    };

    struct ComputeBestSolutionBase {
        FK_HOST_FUSE uint computeDiscardedThreads(const uint width, const uint height, const uint blockDimx, const uint blockDimy) {
            const uint modX = width % blockDimx;
            const uint modY = height % blockDimy;
            const uint th_disabled_in_X = modX == 0 ? 0 : blockDimx - modX;
            const uint th_disabled_in_Y = modY == 0 ? 0 : blockDimy - modY;
            return (th_disabled_in_X * (modY == 0 ? height : (height + blockDimy)) + th_disabled_in_Y * width);
        }
    };

    template <uint bxS_t, uint byS_t>
    struct computeBestSolution {};

    template <uint bxS_t>
    struct computeBestSolution<bxS_t, 0> final : public ComputeBestSolutionBase {
        FK_HOST_FUSE void exec(const uint width, const uint height, uint& bxS, uint& byS, uint& minDiscardedThreads, const uint(&blockDimX)[4], const uint(&blockDimY)[2][4]) {
            const uint currentDiscardedThreads = computeDiscardedThreads(width, height, blockDimX[bxS_t], blockDimY[0][bxS_t]);
            if (minDiscardedThreads > currentDiscardedThreads) {
                minDiscardedThreads = currentDiscardedThreads;
                bxS = bxS_t;
                byS = 0;
                if (minDiscardedThreads == 0) return;
            }
            computeBestSolution<bxS_t, 1>::exec(width, height, bxS, byS, minDiscardedThreads, blockDimX, blockDimY);
        }
    };

    template <uint bxS_t>
    struct computeBestSolution<bxS_t, 1> final : public ComputeBestSolutionBase {
        FK_HOST_FUSE void exec(const uint width, const uint height, uint& bxS, uint& byS, uint& minDiscardedThreads, const uint(&blockDimX)[4], const uint(&blockDimY)[2][4]) {
            const uint currentDiscardedThreads = computeDiscardedThreads(width, height, blockDimX[bxS_t], blockDimY[1][bxS_t]);
            if (minDiscardedThreads > currentDiscardedThreads) {
                minDiscardedThreads = currentDiscardedThreads;
                bxS = bxS_t;
                byS = 1;
                if constexpr (bxS_t == 3) return;
                if (minDiscardedThreads == 0) return;
            }
            computeBestSolution<bxS_t + 1, 0>::exec(width, height, bxS, byS, minDiscardedThreads, blockDimX, blockDimY);
        }
    };

    template <>
    struct computeBestSolution<3, 1> final : public ComputeBestSolutionBase {
        FK_HOST_FUSE void exec(const uint width, const uint height, uint& bxS, uint& byS, uint& minDiscardedThreads, const uint(&blockDimX)[4], const uint(&blockDimY)[2][4]) {
            const uint currentDiscardedThreads = computeDiscardedThreads(width, height, blockDimX[3], blockDimY[1][3]);
            if (minDiscardedThreads > currentDiscardedThreads) {
                minDiscardedThreads = currentDiscardedThreads;
                bxS = 3;
                byS = 1;
            }
        }
    };

    FK_HOST_CNST CtxDim3 getDefaultBlockSize(const uint& width, const uint& height) {
        constexpr uint blockDimX[4] = { 32, 64, 128, 256 };  // Possible block sizes in the x axis
        constexpr uint blockDimY[2][4] = { { 8,  4,   2,   1},
                                          { 6,  3,   3,   2} };  // Possible block sizes in the y axis according to blockDim.x

        uint minDiscardedThreads = UINT_MAX;
        uint bxS = 0; // from 0 to 3
        uint byS = 0; // from 0 to 1

        computeBestSolution<0, 0>::exec(width, height, bxS, byS, minDiscardedThreads, blockDimX, blockDimY);

        return CtxDim3(blockDimX[bxS], blockDimY[byS][bxS]);
    }

    /**
     * @brief DPPLaunchConfig: backend-agnostic launch configuration for a Data Parallel Pattern.
     * grid is the number of thread blocks and block the number of threads per block, in each dimension.
     * It is returned by the DPP's getLaunchConfig() so that the generic InstantiableDPP execution
     * path can launch any conforming DPP without a hand-written Executor specialization.
     * The CPU backend ignores it: CPU DPP exec() implementations define their own sequential loops.
     */
    struct DPPLaunchConfig {
        ActiveThreads grid;
        ActiveThreads block;
    };

    /**
     * @brief defaultDPPLaunchConfig: one thread per output element, with the default 2D block size
     * heuristic (getDefaultBlockSize) and one grid plane per z element. This is the configuration
     * used by TransformDPP, and a sane default for element-wise DPPs.
     */
    FK_HOST_CNST DPPLaunchConfig defaultDPPLaunchConfig(const ActiveThreads& activeThreads) {
        const CtxDim3 block = getDefaultBlockSize(activeThreads.x, activeThreads.y);
        const ActiveThreads grid(static_cast<uint>(ceil(activeThreads.x / static_cast<float>(block.x))),
                                 static_cast<uint>(ceil(activeThreads.y / static_cast<float>(block.y))),
                                 activeThreads.z);
        return { grid, ActiveThreads(block.x, block.y, 1u) };
    }

} // namespace fk

#endif // FK_LAUNCH_CONFIG_H
