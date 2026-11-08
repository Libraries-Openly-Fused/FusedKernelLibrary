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

#ifndef FK_NEIGHBORHOOD_DPP_H
#define FK_NEIGHBORHOOD_DPP_H

#include <fused_kernel/core/data/point.h>
#include <fused_kernel/core/utils/utils.h>

namespace fk {

template <typename T, int TILE_W, int TILE_H,
          int MAX_WINDOW_W, int MAX_WINDOW_H>
struct NeighborhoodDPPPolicy {
    using ValueType = T;
    static constexpr int TILE_WIDTH = TILE_W;
    static constexpr int TILE_HEIGHT = TILE_H;
    static constexpr int MAX_WINDOW_WIDTH = MAX_WINDOW_W;
    static constexpr int MAX_WINDOW_HEIGHT = MAX_WINDOW_H;
    static constexpr int MAX_HALO_WIDTH = TILE_W + MAX_WINDOW_W - 1;
    static constexpr int MAX_HALO_HEIGHT = TILE_H + MAX_WINDOW_H - 1;

    static_assert(TILE_W > 0 && TILE_H > 0,
                  "Neighbourhood tile dimensions must be positive");
    static_assert(MAX_WINDOW_W > 0 && MAX_WINDOW_H > 0,
                  "Neighbourhood max window dimensions must be positive");
    static_assert(TILE_W * TILE_H <= 1024,
                  "Neighbourhood tile exceeds CUDA block size");
};

template <typename Policy>
struct NeighborhoodDPPStage {
private:
    using SelfType = NeighborhoodDPPStage<Policy>;

public:
    using T = typename Policy::ValueType;
    FK_STATIC_STRUCT(NeighborhoodDPPStage, SelfType)

    FK_HOST_DEVICE_FUSE bool valid(const int width, const int height,
                                   const int windowWidth,
                                   const int windowHeight,
                                   const int anchorX, const int anchorY,
                                   const bool requireOdd) {
        return width > 0 && height > 0 &&
               windowWidth > 0 && windowHeight > 0 &&
               windowWidth <= Policy::MAX_WINDOW_WIDTH &&
               windowHeight <= Policy::MAX_WINDOW_HEIGHT &&
               anchorX >= 0 && anchorX < windowWidth &&
               anchorY >= 0 && anchorY < windowHeight &&
               (!requireOdd || ((windowWidth & 1) && (windowHeight & 1)));
    }

    template <typename ReadIOp>
    FK_HOST_DEVICE_STATIC T readReplicate(const int width,
                                          const int height,
                                          const ReadIOp& input,
                                          int x, int y) {
        x = x < 0 ? 0 : (x >= width ? width - 1 : x);
        y = y < 0 ? 0 : (y >= height ? height - 1 : y);
        return ReadIOp::Operation::exec(Point{x, y, 0}, input);
    }

#if defined(__NVCC__)
    template <typename ReadIOp>
    FK_DEVICE_STATIC void stageReplicate(const int width,
                                         const int height,
                                         const int windowWidth,
                                         const int windowHeight,
                                         const int anchorX,
                                         const int anchorY,
                                         const ReadIOp& input,
                                         T* halo) {
        const int haloWidth = Policy::TILE_WIDTH + windowWidth - 1;
        const int haloHeight = Policy::TILE_HEIGHT + windowHeight - 1;
        const int tileX = blockIdx.x * Policy::TILE_WIDTH;
        const int tileY = blockIdx.y * Policy::TILE_HEIGHT;
        const int originX = tileX - anchorX;
        const int originY = tileY - anchorY;
        const int tid = threadIdx.y * blockDim.x + threadIdx.x;
        const int threads = blockDim.x * blockDim.y;
        for (int index = tid; index < haloWidth * haloHeight;
             index += threads) {
            const int y = index / haloWidth;
            const int x = index % haloWidth;
            halo[index] = readReplicate(
                width, height, input, originX + x, originY + y);
        }
    }
#endif
};

} // namespace fk

#endif // FK_NEIGHBORHOOD_DPP_H
