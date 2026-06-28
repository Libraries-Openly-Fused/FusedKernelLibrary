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

#ifndef FK_COLLECTIVE_TILE_SCHEDULER_H
#define FK_COLLECTIVE_TILE_SCHEDULER_H

#include <fused_kernel/core/data/tuple.h>
#include <fused_kernel/core/execution_model/operation_model/operation_model.h>
#include <fused_kernel/core/utils/utils.h>

namespace fk {

struct WarpTileAssignment {
    int mTile;
    int nTile;
    bool valid;
};

struct RowMajorWarpTileRaster {
private:
    using SelfType = RowMajorWarpTileRaster;

public:
    FK_STATIC_STRUCT(RowMajorWarpTileRaster, SelfType)

    FK_HOST_DEVICE_FUSE WarpTileAssignment map(
        const int warpId, const int, const int nTiles) {
        return {warpId / nTiles, warpId % nTiles, true};
    }
};

struct ColumnMajorWarpTileRaster {
private:
    using SelfType = ColumnMajorWarpTileRaster;

public:
    FK_STATIC_STRUCT(ColumnMajorWarpTileRaster, SelfType)

    FK_HOST_DEVICE_FUSE WarpTileAssignment map(
        const int warpId, const int mTiles, const int) {
        return {warpId % mTiles, warpId / mTiles, true};
    }
};

/**
 * Parameterless Unary Operation mapping
 * Tuple<warpId, mTileExtent, nTileExtent> to a tile assignment.
 * Raster is a static traversal policy; all extents remain runtime inputs.
 */
template <typename RasterPolicy = RowMajorWarpTileRaster>
struct WarpTileScheduler {
private:
    using Input = Tuple<int, int, int>;
    using Parent = UnaryOperation<Input, WarpTileAssignment,
                                  WarpTileScheduler<RasterPolicy>>;
    using SelfType = WarpTileScheduler<RasterPolicy>;

public:
    FK_STATIC_STRUCT(WarpTileScheduler, SelfType)
    DECLARE_UNARY_PARENT

    FK_HOST_DEVICE_FUSE OutputType exec(const InputType input) {
        const int warpId = get<0>(input);
        const int mTiles = get<1>(input);
        const int nTiles = get<2>(input);
        if (warpId < 0 || mTiles <= 0 || nTiles <= 0 ||
            warpId >= mTiles * nTiles)
            return {-1, -1, false};
        return RasterPolicy::map(warpId, mTiles, nTiles);
    }
};

} // namespace fk

#endif // FK_COLLECTIVE_TILE_SCHEDULER_H
