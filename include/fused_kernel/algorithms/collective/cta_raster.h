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

#ifndef FK_COLLECTIVE_CTA_RASTER_H
#define FK_COLLECTIVE_CTA_RASTER_H

#include <fused_kernel/algorithms/collective/tile_scheduler.h>

namespace fk {

using CtaTileAssignment = WarpTileAssignment;
using RowMajorCtaTileRaster = RowMajorWarpTileRaster;
using ColumnMajorCtaTileRaster = ColumnMajorWarpTileRaster;

/**
 * Group output columns into bands and visit all M tiles inside each band.
 * GROUP is compile-time traversal policy; grid extents remain runtime input.
 */
template <int GROUP>
struct GroupedCtaTileRaster {
private:
    using SelfType = GroupedCtaTileRaster<GROUP>;

public:
    static_assert(GROUP > 0, "CTA raster group width must be positive");
    FK_STATIC_STRUCT(GroupedCtaTileRaster, SelfType)

    FK_HOST_DEVICE_FUSE CtaTileAssignment map(
            const int ctaId, const int mTiles, const int nTiles) {
        const int blocksPerFullGroup = GROUP * mTiles;
        const int group = ctaId / blocksPerFullGroup;
        const int firstColumn = group * GROUP;
        const int remainingColumns = nTiles - firstColumn;
        const int groupWidth = remainingColumns < GROUP
            ? remainingColumns : GROUP;
        const int inGroup = ctaId % blocksPerFullGroup;
        return {inGroup / groupWidth,
                firstColumn + inGroup % groupWidth,
                true};
    }
};

/**
 * Pure parameterless Unary Operation mapping
 * Tuple<ctaId, mTileExtent, nTileExtent> to an output-tile assignment.
 */
template <typename RasterPolicy = RowMajorCtaTileRaster>
struct CtaTileScheduler {
private:
    using Input = Tuple<int, int, int>;
    using Parent = UnaryOperation<Input, CtaTileAssignment,
                                  CtaTileScheduler<RasterPolicy>>;
    using SelfType = CtaTileScheduler<RasterPolicy>;

public:
    FK_STATIC_STRUCT(CtaTileScheduler, SelfType)
    DECLARE_UNARY_PARENT

    FK_HOST_DEVICE_FUSE OutputType exec(const InputType input) {
        const int ctaId = get<0>(input);
        const int mTiles = get<1>(input);
        const int nTiles = get<2>(input);
        if (ctaId < 0 || mTiles <= 0 || nTiles <= 0 ||
            ctaId >= mTiles * nTiles)
            return {-1, -1, false};
        return RasterPolicy::map(ctaId, mTiles, nTiles);
    }
};

} // namespace fk

#endif // FK_COLLECTIVE_CTA_RASTER_H
