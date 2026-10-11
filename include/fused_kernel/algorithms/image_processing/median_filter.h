/* Copyright 2025-2026 Oscar Amoros Huguet, Johnny Nunez

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_MEDIAN_FILTER_DPP_H
#define FK_MEDIAN_FILTER_DPP_H

#include <fused_kernel/algorithms/image_processing/neighborhood.h>
#include <fused_kernel/core/data/point.h>
#include <fused_kernel/core/execution_model/operation_model/operation_model.h>
#include <fused_kernel/core/execution_model/parallel_architectures.h>
#include <fused_kernel/core/execution_model/stream.h>
#include <fused_kernel/core/utils/utils.h>

#include <type_traits>
#if defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)
#include <fused_kernel/core/execution_model/cpu_thread_pool.h>

#include <cstddef>
#endif

namespace fk {

/** Fixed-capacity register value handed from a neighbourhood DPP to a
 * single-thread selection Operation. Only values[0..count) are live. */
template <typename T, int CAPACITY>
struct NeighborhoodWindow {
    using ValueType = T;
    static constexpr int MAX_CAPACITY = CAPACITY;
    T values[CAPACITY];
    int count;
};

/** Parameterless Unary Operation selecting the upper median (count/2). */
template <typename T, int CAPACITY>
struct MedianWindowSelect {
private:
    using Window = NeighborhoodWindow<T, CAPACITY>;
    using Parent = UnaryOperation<Window, T,
                                  MedianWindowSelect<T, CAPACITY>>;
    using SelfType = MedianWindowSelect<T, CAPACITY>;

public:
    FK_STATIC_STRUCT(MedianWindowSelect, SelfType)
    DECLARE_UNARY_PARENT

    FK_HOST_DEVICE_FUSE OutputType exec(InputType window) {
        for (int i = 1; i < window.count; ++i) {
            const T key = window.values[i];
            int j = i - 1;
            while (j >= 0 && window.values[j] > key) {
                window.values[j + 1] = window.values[j];
                --j;
            }
            window.values[j + 1] = key;
        }
        return window.values[window.count / 2];
    }
};

/** Alternate selection Operation used to prove orchestration is generic. */
template <typename T, int CAPACITY>
struct MinimumWindowSelect {
private:
    using Window = NeighborhoodWindow<T, CAPACITY>;
    using Parent = UnaryOperation<Window, T,
                                  MinimumWindowSelect<T, CAPACITY>>;
    using SelfType = MinimumWindowSelect<T, CAPACITY>;

public:
    FK_STATIC_STRUCT(MinimumWindowSelect, SelfType)
    DECLARE_UNARY_PARENT

    FK_HOST_DEVICE_FUSE OutputType exec(const InputType window) {
        T result = window.values[0];
        for (int i = 1; i < window.count; ++i)
            result = window.values[i] < result ? window.values[i] : result;
        return result;
    }
};

template <typename T, int TILE_W = 16, int TILE_H = 8,
          int MAX_WINDOW_W = 7, int MAX_WINDOW_H = 7>
struct MedianFilterDPPDetails {
    using ValueType = T;
    using NeighborhoodPolicy = NeighborhoodDPPPolicy<
        T, TILE_W, TILE_H, MAX_WINDOW_W, MAX_WINDOW_H>;
    static constexpr int TILE_WIDTH = NeighborhoodPolicy::TILE_WIDTH;
    static constexpr int TILE_HEIGHT = NeighborhoodPolicy::TILE_HEIGHT;
    static constexpr int MAX_WINDOW_WIDTH =
        NeighborhoodPolicy::MAX_WINDOW_WIDTH;
    static constexpr int MAX_WINDOW_HEIGHT =
        NeighborhoodPolicy::MAX_WINDOW_HEIGHT;
    static constexpr int MAX_HALO_WIDTH =
        NeighborhoodPolicy::MAX_HALO_WIDTH;
    static constexpr int MAX_HALO_HEIGHT =
        NeighborhoodPolicy::MAX_HALO_HEIGHT;
    static constexpr int WINDOW_CAPACITY =
        MAX_WINDOW_WIDTH * MAX_WINDOW_HEIGHT;

    int width;
    int height;
    int windowWidth;
    int windowHeight;
    int anchorX;
    int anchorY;

    // Even windows are intentionally unsupported: there is no unique median.
    FK_HOST_DEVICE_FUSE bool valid(const MedianFilterDPPDetails& details) {
        return NeighborhoodDPPStage<NeighborhoodPolicy>::valid(
            details.width, details.height,
            details.windowWidth, details.windowHeight,
            details.anchorX, details.anchorY, true);
    }
};

template <ParArch PA, typename DPPDetails>
struct MedianFilterDPP;

template <typename DPPDetails>
struct MedianFilterDPP<ParArch::CPU, DPPDetails> {
private:
    using SelfType = MedianFilterDPP<ParArch::CPU, DPPDetails>;
    using T = typename DPPDetails::ValueType;
    using Stage = NeighborhoodDPPStage<
        typename DPPDetails::NeighborhoodPolicy>;
    using Window = NeighborhoodWindow<T, DPPDetails::WINDOW_CAPACITY>;

public:
    FK_STATIC_STRUCT(MedianFilterDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::CPU;

    // Computes output row oy. Used by exec() and by the CPU_MULTITHREAD
    // version; the caller checks DPPDetails::valid(details).
    template <typename ReadIOp, typename WriteIOp, typename SelectionIOp>
    FK_HOST_STATIC void exec_row(const int oy,
                                 const DPPDetails& details,
                                 const ReadIOp& input,
                                 const WriteIOp& output,
                                 const SelectionIOp& selection) {
        static_assert(isAnyCompleteReadType<ReadIOp>,
                      "MedianFilterDPP needs a complete Read IOp");
        static_assert(isAnyWriteType<WriteIOp>,
                      "MedianFilterDPP needs a Write IOp");
        static_assert(SelectionIOp::template is<UnaryType>,
                      "Selection must be a Unary IOp");
        static_assert(std::is_same_v<typename SelectionIOp::Operation::InputType,
                                     Window>,
                      "Selection input must be this DPP's window type");
        static_assert(std::is_same_v<typename SelectionIOp::Operation::OutputType,
                                     T>,
                      "Selection output must match the image value type");
        for (int ox = 0; ox < details.width; ++ox) {
            Window window{};
            for (int wy = 0; wy < details.windowHeight; ++wy) {
                for (int wx = 0; wx < details.windowWidth; ++wx) {
                    window.values[window.count++] = Stage::readReplicate(
                        details.width, details.height, input,
                        ox + wx - details.anchorX,
                        oy + wy - details.anchorY);
                }
            }
            const T selected = window | selection;
            WriteIOp::Operation::exec(
                Point{ox, oy, 0}, selected, output);
        }
    }

    template <typename ReadIOp, typename WriteIOp, typename SelectionIOp>
    FK_HOST_STATIC void exec(const DPPDetails& details,
                             const ReadIOp& input,
                             const WriteIOp& output,
                             const SelectionIOp& selection) {
        if (!DPPDetails::valid(details)) return;

        for (int oy = 0; oy < details.height; ++oy) {
            exec_row(oy, details, input, output, selection);
        }
    }
};

#if defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)
// Runs the per row code of MedianFilterDPP<ParArch::CPU> on cpuThreadPool(),
// one chunk per output row, so both give the same results. As in the GPU
// version, the output must not alias the input: rows run concurrently, and
// a row reads the input rows around it.
template <typename DPPDetails>
struct MedianFilterDPP<ParArch::CPU_MULTITHREAD, DPPDetails> {
private:
    using SelfType = MedianFilterDPP<ParArch::CPU_MULTITHREAD, DPPDetails>;
    using CPUDPP = MedianFilterDPP<ParArch::CPU, DPPDetails>;

public:
    FK_STATIC_STRUCT(MedianFilterDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::CPU_MULTITHREAD;

    // Enqueues the rows in the pool and returns; done is called when all of
    // them have finished. Invalid details enqueue no rows (nothing is
    // written), as the CPU version.
    template <typename ReadIOp, typename WriteIOp, typename SelectionIOp>
    FK_HOST_STATIC void exec(const cpu_multithread::Completion& done,
                             const DPPDetails& details,
                             const ReadIOp& input,
                             const WriteIOp& output,
                             const SelectionIOp& selection) {
        const std::size_t rows = DPPDetails::valid(details)
            ? static_cast<std::size_t>(details.height) : 0;
        cpu_multithread::enqueueChunks(rows,
            [details, input, output, selection](const std::size_t row) {
                CPUDPP::exec_row(static_cast<int>(row), details, input,
                                 output, selection);
            }, done);
    }
};
#endif // defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)

#if defined(__NVCC__) || defined(__HIPCC__)
template <ParArch PA, typename DPPDetails>
    requires(PA == ParArch::GPU_NVIDIA || PA == ParArch::GPU_AMD)
struct MedianFilterDPP<PA, DPPDetails> {
private:
    using SelfType = MedianFilterDPP<PA, DPPDetails>;
    using T = typename DPPDetails::ValueType;
    using Stage = NeighborhoodDPPStage<
        typename DPPDetails::NeighborhoodPolicy>;
    using Window = NeighborhoodWindow<T, DPPDetails::WINDOW_CAPACITY>;

public:
    FK_STATIC_STRUCT(MedianFilterDPP, SelfType)
    static constexpr ParArch PAR_ARCH = PA;

    template <typename ReadIOp, typename WriteIOp, typename SelectionIOp>
    FK_DEVICE_STATIC void exec(const DPPDetails& details,
                               const ReadIOp& input,
                               const WriteIOp& output,
                               const SelectionIOp& selection) {
        static_assert(isAnyCompleteReadType<ReadIOp>,
                      "MedianFilterDPP needs a complete Read IOp");
        static_assert(isAnyWriteType<WriteIOp>,
                      "MedianFilterDPP needs a Write IOp");
        static_assert(SelectionIOp::template is<UnaryType>,
                      "Selection must be a Unary IOp");
        static_assert(std::is_same_v<typename SelectionIOp::Operation::InputType,
                                     Window>,
                      "Selection input must be this DPP's window type");
        static_assert(std::is_same_v<typename SelectionIOp::Operation::OutputType,
                                     T>,
                      "Selection output must match the image value type");

        __shared__ T halo[DPPDetails::MAX_HALO_WIDTH *
                          DPPDetails::MAX_HALO_HEIGHT];
        Stage::stageReplicate(
            details.width, details.height,
            details.windowWidth, details.windowHeight,
            details.anchorX, details.anchorY, input, halo);
        __syncthreads();

        const int tileX = blockIdx.x * DPPDetails::TILE_WIDTH;
        const int tileY = blockIdx.y * DPPDetails::TILE_HEIGHT;
        const int ox = tileX + threadIdx.x;
        const int oy = tileY + threadIdx.y;
        const int haloWidth = DPPDetails::TILE_WIDTH +
                              details.windowWidth - 1;
        if (threadIdx.x < DPPDetails::TILE_WIDTH &&
            threadIdx.y < DPPDetails::TILE_HEIGHT &&
            ox < details.width && oy < details.height) {
            Window window{};
            for (int wy = 0; wy < details.windowHeight; ++wy)
                for (int wx = 0; wx < details.windowWidth; ++wx)
                    window.values[window.count++] = halo[
                        (threadIdx.y + wy) * haloWidth +
                        threadIdx.x + wx];
            const T selected = window | selection;
            WriteIOp::Operation::exec(
                Point{ox, oy, 0}, selected, output);
        }
    }
};

template <ParArch PA, typename DPPDetails, typename ReadIOp,
          typename WriteIOp, typename SelectionIOp>
__global__ void medianFilterDPPKernel(const DPPDetails details,
                                      const ReadIOp input,
                                      const WriteIOp output,
                                      const SelectionIOp selection) {
    MedianFilterDPP<PA, DPPDetails>::exec(
        details, input, output, selection);
}

template <typename DPPDetails, typename ReadIOp,
          typename WriteIOp, typename SelectionIOp, ParArch PA>
    requires(PA == ParArch::GPU_NVIDIA || PA == ParArch::GPU_AMD)
inline bool executeMedianFilter(const DPPDetails& details,
                                const ReadIOp& input,
                                const WriteIOp& output,
                                const SelectionIOp& selection,
                                Stream_<PA>& stream) {
    if (!DPPDetails::valid(details)) return false;
    const dim3 block(DPPDetails::TILE_WIDTH,
                     DPPDetails::TILE_HEIGHT, 1);
    const dim3 grid((details.width + DPPDetails::TILE_WIDTH - 1) /
                        DPPDetails::TILE_WIDTH,
                    (details.height + DPPDetails::TILE_HEIGHT - 1) /
                        DPPDetails::TILE_HEIGHT,
                    1);
#if defined(__HIPCC__)
    medianFilterDPPKernel<PA><<<grid, block, 0, stream.getHIPStream()>>>(
        details, input, output, selection);
    gpuErrchk(hipGetLastError());
#else
    medianFilterDPPKernel<PA><<<grid, block, 0, stream.getCUDAStream()>>>(
        details, input, output, selection);
    gpuErrchk(cudaGetLastError());
#endif
    return true;
}
#endif // defined(__NVCC__) || defined(__HIPCC__)

#if defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)
// Enqueues MedianFilterDPP<ParArch::CPU_MULTITHREAD> on the stream and
// returns; returns false, enqueueing nothing, if the details are not valid.
// The execution keeps copies of the IOps. As with a GPU stream, the buffers
// they read and write must stay valid until it has finished (sync()), and
// the output must not alias the input.
template <typename DPPDetails, typename ReadIOp,
          typename WriteIOp, typename SelectionIOp>
inline bool executeMedianFilter(const DPPDetails& details,
                                const ReadIOp& input,
                                const WriteIOp& output,
                                const SelectionIOp& selection,
                                Stream_<ParArch::CPU_MULTITHREAD>& stream) {
    if (!DPPDetails::valid(details)) return false;
    stream.enqueue([=](const cpu_multithread::Completion& done) {
        MedianFilterDPP<ParArch::CPU_MULTITHREAD, DPPDetails>::exec(
            done, details, input, output, selection);
    });
    return true;
}
#endif // defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)

} // namespace fk

#endif // FK_MEDIAN_FILTER_DPP_H
