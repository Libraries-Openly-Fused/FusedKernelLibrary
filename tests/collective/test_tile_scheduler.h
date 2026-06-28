/* Copyright 2026 Johnny Nunez

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include <tests/main.h>

#include <fused_kernel/algorithms/collective/tile_scheduler.h>

#include <cstdio>
#include <vector>

using namespace fk;

namespace {

template <typename Scheduler>
bool checkHost(const int mTiles, const int nTiles,
               const bool columnMajor, const char* label) {
    const Scheduler scheduler = Scheduler::Operation::build();
    const int total = mTiles * nTiles;
    std::vector<int> seen(total, 0);
    bool ok = true;
    for (int warpId = 0; warpId < total + 3; ++warpId) {
        const WarpTileAssignment tile =
            make_tuple(warpId, mTiles, nTiles) | scheduler;
        if (warpId >= total) {
            if (tile.valid) ok = false;
            continue;
        }
        if (!tile.valid || tile.mTile < 0 || tile.mTile >= mTiles ||
            tile.nTile < 0 || tile.nTile >= nTiles) {
            ok = false;
            continue;
        }
        const int expectedM = columnMajor
            ? warpId % mTiles : warpId / nTiles;
        const int expectedN = columnMajor
            ? warpId / mTiles : warpId % nTiles;
        if (tile.mTile != expectedM || tile.nTile != expectedN)
            ok = false;
        ++seen[tile.mTile * nTiles + tile.nTile];
    }
    for (const int count : seen)
        if (count != 1) ok = false;
    if (!ok)
        std::printf("host mapping failed: %s %dx%d\n",
                    label, mTiles, nTiles);
    return ok;
}

#if defined(__NVCC__)
template <typename Scheduler>
__global__ void invokeScheduler(WarpTileAssignment* output,
                                const int mTiles, const int nTiles,
                                const Scheduler scheduler) {
    const int warpId = threadIdx.x;
    output[warpId] = make_tuple(warpId, mTiles, nTiles) | scheduler;
}

template <typename Scheduler>
bool checkDevice(const int mTiles, const int nTiles,
                 const bool columnMajor, const char* label) {
    const int total = mTiles * nTiles;
    const int launched = total + 3;
    WarpTileAssignment* device = nullptr;
    cudaMalloc(&device, launched * sizeof(WarpTileAssignment));
    const Scheduler scheduler = Scheduler::Operation::build();
    invokeScheduler<Scheduler><<<1, launched>>>(
        device, mTiles, nTiles, scheduler);
    std::vector<WarpTileAssignment> output(launched);
    cudaMemcpy(output.data(), device,
               launched * sizeof(WarpTileAssignment),
               cudaMemcpyDeviceToHost);
    const cudaError_t error = cudaDeviceSynchronize();
    cudaFree(device);
    if (error != cudaSuccess) return false;

    std::vector<int> seen(total, 0);
    bool ok = true;
    for (int warpId = 0; warpId < launched; ++warpId) {
        const auto tile = output[warpId];
        if (warpId >= total) {
            if (tile.valid) ok = false;
            continue;
        }
        const int expectedM = columnMajor
            ? warpId % mTiles : warpId / nTiles;
        const int expectedN = columnMajor
            ? warpId / mTiles : warpId % nTiles;
        if (!tile.valid || tile.mTile != expectedM ||
            tile.nTile != expectedN) ok = false;
        if (tile.valid && tile.mTile >= 0 && tile.mTile < mTiles &&
            tile.nTile >= 0 && tile.nTile < nTiles)
            ++seen[tile.mTile * nTiles + tile.nTile];
    }
    for (const int count : seen)
        if (count != 1) ok = false;
    if (!ok)
        std::printf("device mapping failed: %s %dx%d\n",
                    label, mTiles, nTiles);
    return ok;
}
#endif

template <typename Scheduler>
bool checkAll(const bool columnMajor, const char* label) {
    bool ok = true;
    const int grids[][2] = {{1, 1}, {1, 7}, {2, 3}, {3, 2}, {3, 5}, {5, 3}};
    for (const auto& grid : grids) {
        ok = checkHost<Scheduler>(
            grid[0], grid[1], columnMajor, label) && ok;
#if defined(__NVCC__)
        ok = checkDevice<Scheduler>(
            grid[0], grid[1], columnMajor, label) && ok;
#endif
    }
    return ok;
}

} // namespace

int launch() {
    using RowScheduler = decltype(
        WarpTileScheduler<RowMajorWarpTileRaster>::build());
    using ColumnScheduler = decltype(
        WarpTileScheduler<ColumnMajorWarpTileRaster>::build());
    static_assert(RowScheduler::template is<UnaryType>);
    static_assert(ColumnScheduler::template is<UnaryType>);

    bool ok = checkAll<RowScheduler>(false, "row-major");
    ok = checkAll<ColumnScheduler>(true, "column-major") && ok;
    if (ok) std::printf("WarpTileScheduler contracts: PASS\n");
    return ok ? 0 : -1;
}
