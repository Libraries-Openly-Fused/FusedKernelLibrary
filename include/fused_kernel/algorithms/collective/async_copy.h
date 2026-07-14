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

#ifndef FK_COLLECTIVE_ASYNC_COPY_H
#define FK_COLLECTIVE_ASYNC_COPY_H

#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/collective/tile.h>
#include <fused_kernel/core/data/point.h>
#include <fused_kernel/core/execution_model/operation_model/operation_model.h>
#include <fused_kernel/core/utils/utils.h>

#include <cstdint>
#include <type_traits>

namespace fk {

template <typename T, typename Layout, int BLOCK_SIZE = 256>
struct AsyncCopyDPPDetails {
    static_assert(BLOCK_SIZE > 0 && BLOCK_SIZE <= 1024,
                  "BLOCK_SIZE must be in (0, 1024]");
    static_assert((Layout::size() * sizeof(T)) % 16 == 0,
                  "Async-copy local storage must be a multiple of 16 bytes");

    using ValueType = T;
    using LayoutType = Layout;
    static constexpr int BLOCK_THREADS = BLOCK_SIZE;
    static constexpr uint TILE_ELEMENTS = Layout::size();

    int origin;
    int elementCount;
    T boundaryValue;
};

// Row-strided source policy used by tiled MMA staging. validCols/validRows
// provide deterministic zero/boundary fill for K tails without over-reading.
template <typename T, typename Layout, int BLOCK_SIZE = 32>
struct AsyncCopy2DDPPDetails {
    static_assert(BLOCK_SIZE > 0 && BLOCK_SIZE <= 1024,
                  "BLOCK_SIZE must be in (0, 1024]");
    static_assert((Layout::cols * sizeof(T)) % 16 == 0,
                  "Each async-copy row must be a multiple of 16 bytes");

    using ValueType = T;
    using LayoutType = Layout;
    static constexpr int BLOCK_THREADS = BLOCK_SIZE;
    static constexpr uint TILE_ELEMENTS = Layout::size();

    int originX;
    int originY;
    int validCols;
    int validRows;
    T boundaryValue;
};

template <ParArch PA, typename DPPDetails>
struct AsyncCopyDPP;

template <typename DPPDetails>
struct AsyncCopyDPP<ParArch::CPU, DPPDetails> {
private:
    using SelfType = AsyncCopyDPP<ParArch::CPU, DPPDetails>;
    using T = typename DPPDetails::ValueType;
    using Layout = typename DPPDetails::LayoutType;

public:
    FK_STATIC_STRUCT(AsyncCopyDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::CPU;

    template <typename ReadIOp>
    FK_HOST_STATIC void load(const DPPDetails& details,
                             const ReadIOp& input,
                             Tile<T, Layout> tile) {
        static_assert(isAnyCompleteReadType<ReadIOp>,
                      "AsyncCopyDPP requires a complete Read IOp");
        for (uint index = 0; index < DPPDetails::TILE_ELEMENTS; ++index) {
            tile.at(index / Layout::cols, index % Layout::cols) =
                static_cast<int>(index) < details.elementCount
                ? ReadIOp::Operation::exec(
                      Point{details.origin + static_cast<int>(index), 0, 0},
                      input)
                : details.boundaryValue;
        }
    }
};

// CPU row-strided specialization, primarily for semantic parity tests.
template <typename T, typename Layout, int BLOCK_SIZE>
struct AsyncCopyDPP<ParArch::CPU,
                    AsyncCopy2DDPPDetails<T, Layout, BLOCK_SIZE>> {
private:
    using DPPDetails = AsyncCopy2DDPPDetails<T, Layout, BLOCK_SIZE>;
    using SelfType = AsyncCopyDPP<ParArch::CPU, DPPDetails>;

public:
    FK_STATIC_STRUCT(AsyncCopyDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::CPU;

    template <typename ReadIOp>
    FK_HOST_STATIC void load(const DPPDetails& details,
                             const ReadIOp& input,
                             Tile<T, Layout> tile) {
        static_assert(isAnyCompleteReadType<ReadIOp>,
                      "AsyncCopyDPP requires a complete Read IOp");
        for (uint row = 0; row < Layout::rows; ++row) {
            for (uint col = 0; col < Layout::cols; ++col) {
                tile.at(row, col) =
                    static_cast<int>(row) < details.validRows &&
                    static_cast<int>(col) < details.validCols
                    ? ReadIOp::Operation::exec(
                          Point{details.originX + static_cast<int>(col),
                                details.originY + static_cast<int>(row), 0},
                          input)
                    : details.boundaryValue;
            }
        }
    }
};

#if defined(__NVCC__)
template <typename DPPDetails>
struct AsyncCopyDPP<ParArch::GPU_NVIDIA, DPPDetails> {
private:
    using SelfType = AsyncCopyDPP<ParArch::GPU_NVIDIA, DPPDetails>;
    using T = typename DPPDetails::ValueType;
    using Layout = typename DPPDetails::LayoutType;
    static constexpr int ELEMENTS_PER_CHUNK = 16 / sizeof(T);
    static constexpr int CHUNKS = DPPDetails::TILE_ELEMENTS /
                                  ELEMENTS_PER_CHUNK;

    template <typename ReadIOp>
    static constexpr bool ADDRESSABLE_READ =
        std::is_same_v<typename ReadIOp::Operation,
                       PerThreadRead<ND::_1D, T>>;

    FK_DEVICE_STATIC void copy16(void* destination,
                                 const void* source,
                                 const int sourceBytes) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
        const uint32_t sharedAddress = static_cast<uint32_t>(
            __cvta_generic_to_shared(destination));
        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], 16, %2;"
            :: "r"(sharedAddress), "l"(source), "r"(sourceBytes));
#else
        if (sourceBytes == 16) {
            *reinterpret_cast<int4*>(destination) =
                *reinterpret_cast<const int4*>(source);
        } else {
            char* dst = reinterpret_cast<char*>(destination);
            const char* src = reinterpret_cast<const char*>(source);
            for (int byte = 0; byte < 16; ++byte)
                dst[byte] = byte < sourceBytes ? src[byte] : 0;
        }
#endif
    }

    FK_DEVICE_STATIC void commitAndWait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
#endif
    }

    template <typename ReadIOp>
    FK_DEVICE_STATIC void fallback(const DPPDetails& details,
                                   const ReadIOp& input,
                                   Tile<T, Layout> tile) {
        for (uint index = threadIdx.x;
             index < DPPDetails::TILE_ELEMENTS;
             index += DPPDetails::BLOCK_THREADS) {
            tile.at(index / Layout::cols, index % Layout::cols) =
                static_cast<int>(index) < details.elementCount
                ? ReadIOp::Operation::exec(
                      Point{details.origin + static_cast<int>(index), 0, 0},
                      input)
                : details.boundaryValue;
        }
    }

public:
    FK_STATIC_STRUCT(AsyncCopyDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::GPU_NVIDIA;

    template <typename ReadIOp>
    FK_DEVICE_STATIC void load(const DPPDetails& details,
                               const ReadIOp& input,
                               Tile<T, Layout> tile) {
        static_assert(isAnyCompleteReadType<ReadIOp>,
                      "AsyncCopyDPP requires a complete Read IOp");
        static_assert(sizeof(T) <= 16 && 16 % sizeof(T) == 0,
                      "AsyncCopyDPP requires an element size dividing 16");

        if constexpr (ADDRESSABLE_READ<ReadIOp>) {
            const bool aligned =
                (reinterpret_cast<uintptr_t>(
                     input.params.data + details.origin) & 0xFu) == 0;
            if (aligned) {
                for (int chunk = threadIdx.x;
                     chunk < CHUNKS;
                     chunk += DPPDetails::BLOCK_THREADS) {
                    const int first = chunk * ELEMENTS_PER_CHUNK;
                    int valid = details.elementCount - first;
                    valid = valid < 0 ? 0 : valid;
                    valid = valid > ELEMENTS_PER_CHUNK
                        ? ELEMENTS_PER_CHUNK : valid;
                    const T* source = valid > 0
                        ? input.params.data + details.origin + first
                        : input.params.data + details.origin;
                    copy16(tile.data() + first, source,
                           valid * static_cast<int>(sizeof(T)));
                }
                commitAndWait();
                for (int index = details.elementCount + threadIdx.x;
                     index < static_cast<int>(DPPDetails::TILE_ELEMENTS);
                     index += DPPDetails::BLOCK_THREADS) {
                    if (index >= 0) {
                        tile.at(index / Layout::cols,
                                index % Layout::cols) = details.boundaryValue;
                    }
                }
            } else {
                fallback(details, input, tile);
            }
        } else {
            fallback(details, input, tile);
        }
        __syncthreads();
    }
};

// GPU row-strided specialization with split issue/commit/wait/complete phases
// so a composing mainloop can keep one cp.async group in flight.
template <typename T, typename Layout, int BLOCK_SIZE>
struct AsyncCopyDPP<ParArch::GPU_NVIDIA,
                    AsyncCopy2DDPPDetails<T, Layout, BLOCK_SIZE>> {
private:
    using DPPDetails = AsyncCopy2DDPPDetails<T, Layout, BLOCK_SIZE>;
    using SelfType = AsyncCopyDPP<ParArch::GPU_NVIDIA, DPPDetails>;
    static constexpr int ELEMENTS_PER_CHUNK = 16 / sizeof(T);
    static constexpr int CHUNKS_PER_ROW =
        static_cast<int>(Layout::cols) / ELEMENTS_PER_CHUNK;
    static constexpr int CHUNKS =
        static_cast<int>(Layout::rows) * CHUNKS_PER_ROW;

    template <typename ReadIOp>
    static constexpr bool ADDRESSABLE_READ =
        std::is_same_v<typename ReadIOp::Operation,
                       PerThreadRead<ND::_2D, T>>;

    FK_DEVICE_STATIC void copy16(void* destination,
                                 const void* source,
                                 const int sourceBytes) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
        const uint32_t sharedAddress = static_cast<uint32_t>(
            __cvta_generic_to_shared(destination));
        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], 16, %2;"
            :: "r"(sharedAddress), "l"(source), "r"(sourceBytes));
#else
        if (sourceBytes == 16) {
            *reinterpret_cast<int4*>(destination) =
                *reinterpret_cast<const int4*>(source);
        } else {
            char* dst = reinterpret_cast<char*>(destination);
            const char* src = reinterpret_cast<const char*>(source);
            for (int byte = 0; byte < 16; ++byte)
                dst[byte] = byte < sourceBytes ? src[byte] : 0;
        }
#endif
    }

public:
    FK_STATIC_STRUCT(AsyncCopyDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::GPU_NVIDIA;

    template <typename ReadIOp>
    FK_DEVICE_STATIC void issue(const DPPDetails& details,
                                const ReadIOp& input,
                                Tile<T, Layout> tile) {
        static_assert(isAnyCompleteReadType<ReadIOp>,
                      "AsyncCopyDPP requires a complete Read IOp");
        static_assert(sizeof(T) <= 16 && 16 % sizeof(T) == 0,
                      "AsyncCopyDPP requires an element size dividing 16");
        if constexpr (ADDRESSABLE_READ<ReadIOp>) {
            const uintptr_t firstAddress = reinterpret_cast<uintptr_t>(
                reinterpret_cast<const char*>(input.params.data) +
                details.originY * input.params.dims.pitch) +
                static_cast<uintptr_t>(details.originX * sizeof(T));
            const bool aligned = (firstAddress & 0xFu) == 0 &&
                                 (input.params.dims.pitch & 0xFu) == 0;
            if (aligned) {
                for (int chunk = threadIdx.x; chunk < CHUNKS;
                     chunk += DPPDetails::BLOCK_THREADS) {
                    const int row = chunk / CHUNKS_PER_ROW;
                    const int chunkInRow = chunk % CHUNKS_PER_ROW;
                    const int firstCol = chunkInRow * ELEMENTS_PER_CHUNK;
                    const bool validRow = row < details.validRows;
                    int valid = validRow ? details.validCols - firstCol : 0;
                    valid = valid < 0 ? 0 : valid;
                    valid = valid > ELEMENTS_PER_CHUNK
                        ? ELEMENTS_PER_CHUNK : valid;
                    const char* rowBase =
                        reinterpret_cast<const char*>(input.params.data) +
                        (details.originY + (validRow ? row : 0)) *
                            input.params.dims.pitch;
                    const T* source = valid > 0
                        ? reinterpret_cast<const T*>(rowBase) +
                              details.originX + firstCol
                        : input.params.data;
                    copy16(tile.data() + row * Layout::cols + firstCol,
                           source, valid * static_cast<int>(sizeof(T)));
                }
                return;
            }
        }
        for (uint index = threadIdx.x; index < Layout::size();
             index += DPPDetails::BLOCK_THREADS) {
            const int row = static_cast<int>(index / Layout::cols);
            const int col = static_cast<int>(index % Layout::cols);
            tile.at(row, col) = row < details.validRows &&
                                col < details.validCols
                ? ReadIOp::Operation::exec(
                      Point{details.originX + col, details.originY + row, 0},
                      input)
                : details.boundaryValue;
        }
    }

    FK_DEVICE_STATIC void commit() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
        asm volatile("cp.async.commit_group;");
#endif
    }

    template <int PendingGroups>
    FK_DEVICE_STATIC void wait() {
        static_assert(PendingGroups == 0 || PendingGroups == 1,
                      "Double-buffer copy supports 0 or 1 pending group");
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
        if constexpr (PendingGroups == 0)
            asm volatile("cp.async.wait_group 0;");
        else
            asm volatile("cp.async.wait_group 1;");
#endif
    }

    FK_DEVICE_STATIC void complete(const DPPDetails& details,
                                   Tile<T, Layout> tile) {
        for (uint index = threadIdx.x; index < Layout::size();
             index += DPPDetails::BLOCK_THREADS) {
            const int row = static_cast<int>(index / Layout::cols);
            const int col = static_cast<int>(index % Layout::cols);
            if (row >= details.validRows || col >= details.validCols)
                tile.at(row, col) = details.boundaryValue;
        }
    }
};
#endif // defined(__NVCC__)

} // namespace fk

#endif // FK_COLLECTIVE_ASYNC_COPY_H
