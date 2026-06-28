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

#ifndef FK_COLLECTIVE_COPY_H
#define FK_COLLECTIVE_COPY_H

#include <fused_kernel/algorithms/collective/tile.h>
#include <fused_kernel/core/data/point.h>
#include <fused_kernel/core/execution_model/operation_model/operation_model.h>
#include <fused_kernel/core/utils/utils.h>

#include <type_traits>

namespace fk {

template <typename T, typename Layout, int BLOCK_SIZE = 256>
struct CopyTileDPPDetails {
    static_assert(BLOCK_SIZE > 0 && BLOCK_SIZE <= 1024,
                  "BLOCK_SIZE must be in (0, 1024]");

    using ValueType = T;
    using LayoutType = Layout;
    static constexpr int BLOCK_THREADS = BLOCK_SIZE;
    static constexpr uint TILE_ROWS = Layout::rows;
    static constexpr uint TILE_COLS = Layout::cols;
    static constexpr uint TILE_ELEMENTS = Layout::size();

    int originX;
    int originY;
    int extentWidth;
    int extentHeight;
    T boundaryValue;
};

template <ParArch PA, typename DPPDetails>
struct CopyTileDPP;

template <typename DPPDetails>
struct CopyTileDPP<ParArch::CPU, DPPDetails> {
private:
    using SelfType = CopyTileDPP<ParArch::CPU, DPPDetails>;
    using T = typename DPPDetails::ValueType;
    using Layout = typename DPPDetails::LayoutType;

    FK_HOST_FUSE bool isValid(const DPPDetails& details,
                              const int globalX,
                              const int globalY) {
        return globalX >= 0 && globalY >= 0 &&
               globalX < details.extentWidth &&
               globalY < details.extentHeight;
    }

public:
    FK_STATIC_STRUCT(CopyTileDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::CPU;

    template <typename ReadIOp>
    FK_HOST_STATIC void load(const DPPDetails& details,
                             const ReadIOp& input,
                             Tile<T, Layout> tile) {
        static_assert(isAnyCompleteReadType<ReadIOp>,
                      "CopyTileDPP::load requires a complete Read IOp");
        for (uint row = 0; row < Layout::rows; ++row) {
            for (uint col = 0; col < Layout::cols; ++col) {
                const int globalX = details.originX + static_cast<int>(col);
                const int globalY = details.originY + static_cast<int>(row);
                tile.at(row, col) = isValid(details, globalX, globalY)
                    ? ReadIOp::Operation::exec(
                          Point{globalX, globalY, 0}, input)
                    : details.boundaryValue;
            }
        }
    }

    template <typename WriteIOp>
    FK_HOST_STATIC void store(const DPPDetails& details,
                              const Tile<T, Layout> tile,
                              const WriteIOp& output) {
        static_assert(isAnyWriteType<WriteIOp>,
                      "CopyTileDPP::store requires a Write IOp");
        for (uint row = 0; row < Layout::rows; ++row) {
            for (uint col = 0; col < Layout::cols; ++col) {
                const int globalX = details.originX + static_cast<int>(col);
                const int globalY = details.originY + static_cast<int>(row);
                if (isValid(details, globalX, globalY)) {
                    WriteIOp::Operation::exec(
                        Point{globalX, globalY, 0}, tile.at(row, col), output);
                }
            }
        }
    }
};

#if defined(__NVCC__)
template <typename DPPDetails>
struct CopyTileDPP<ParArch::GPU_NVIDIA, DPPDetails> {
private:
    using SelfType = CopyTileDPP<ParArch::GPU_NVIDIA, DPPDetails>;
    using T = typename DPPDetails::ValueType;
    using Layout = typename DPPDetails::LayoutType;

    FK_DEVICE_FUSE bool isValid(const DPPDetails& details,
                                const int globalX,
                                const int globalY) {
        return globalX >= 0 && globalY >= 0 &&
               globalX < details.extentWidth &&
               globalY < details.extentHeight;
    }

public:
    FK_STATIC_STRUCT(CopyTileDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::GPU_NVIDIA;

    template <typename ReadIOp>
    FK_DEVICE_STATIC void load(const DPPDetails& details,
                               const ReadIOp& input,
                               Tile<T, Layout> tile) {
        static_assert(isAnyCompleteReadType<ReadIOp>,
                      "CopyTileDPP::load requires a complete Read IOp");
        for (uint index = threadIdx.x;
             index < DPPDetails::TILE_ELEMENTS;
             index += DPPDetails::BLOCK_THREADS) {
            const uint row = index / Layout::cols;
            const uint col = index % Layout::cols;
            const int globalX = details.originX + static_cast<int>(col);
            const int globalY = details.originY + static_cast<int>(row);
            tile.at(row, col) = isValid(details, globalX, globalY)
                ? ReadIOp::Operation::exec(
                      Point{globalX, globalY, 0}, input)
                : details.boundaryValue;
        }
        __syncthreads();
    }

    template <typename WriteIOp>
    FK_DEVICE_STATIC void store(const DPPDetails& details,
                                const Tile<T, Layout> tile,
                                const WriteIOp& output) {
        static_assert(isAnyWriteType<WriteIOp>,
                      "CopyTileDPP::store requires a Write IOp");
        __syncthreads();
        for (uint index = threadIdx.x;
             index < DPPDetails::TILE_ELEMENTS;
             index += DPPDetails::BLOCK_THREADS) {
            const uint row = index / Layout::cols;
            const uint col = index % Layout::cols;
            const int globalX = details.originX + static_cast<int>(col);
            const int globalY = details.originY + static_cast<int>(row);
            if (isValid(details, globalX, globalY)) {
                WriteIOp::Operation::exec(
                    Point{globalX, globalY, 0}, tile.at(row, col), output);
            }
        }
    }
};
#endif // defined(__NVCC__)

} // namespace fk

#endif // FK_COLLECTIVE_COPY_H

