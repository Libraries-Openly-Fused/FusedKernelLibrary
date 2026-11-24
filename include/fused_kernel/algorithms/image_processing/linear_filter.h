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

#ifndef FK_LINEAR_FILTER_DPP_H
#define FK_LINEAR_FILTER_DPP_H

#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/image_processing/neighborhood.h>
#include <fused_kernel/core/data/point.h>
#include <fused_kernel/core/data/tuple.h>
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

// LinearFilterDPP computes, for every output pixel (ox, oy) of every image z,
//   accumulator = accumulate(accumulator, multiply(image(x, y, z), coefficient(kx, ky)))
// over the kernelWidth x kernelHeight window anchored at (anchorX, anchorY),
// and writes the accumulator at Point{ox, oy, z}.
// - The images come from the image read: their number is z of its active
//   threads (the batch of a batched read, the planes of a 3D read, 1 for a 2D
//   read), and each image z is filtered at its own size (the read's num_elems
//   at z), with its own replicate border. The grid covers the largest image.
// - One set of coefficients, read at Point{kx, ky, 0}, filters every image.
// - The write must have as many planes as the read has images
//   (neighborhood_detail::canRun); otherwise nothing is computed, and the
//   execute functions return false.
// - The image values are ValueType; the products, the accumulator and the
//   value written are AccumulatorType, float with the channels of ValueType.
//   The DPP converts nothing: the write receives AccumulatorType, and a cast
//   to another type is an operation of the write chain.
template <typename T, int TILE_W = 16, int TILE_H = 8,
          int MAX_KERNEL_W = 7, int MAX_KERNEL_H = 7>
struct LinearFilterDPPDetails {
    using ValueType = T;
    using AccumulatorType = VectorType_t<float, cn<T>>;
    using NeighborhoodPolicy = NeighborhoodDPPPolicy<
        T, TILE_W, TILE_H, MAX_KERNEL_W, MAX_KERNEL_H>;
    static constexpr int TILE_WIDTH = NeighborhoodPolicy::TILE_WIDTH;
    static constexpr int TILE_HEIGHT = NeighborhoodPolicy::TILE_HEIGHT;
    static constexpr int MAX_KERNEL_WIDTH =
        NeighborhoodPolicy::MAX_WINDOW_WIDTH;
    static constexpr int MAX_KERNEL_HEIGHT =
        NeighborhoodPolicy::MAX_WINDOW_HEIGHT;
    static constexpr int MAX_HALO_WIDTH =
        NeighborhoodPolicy::MAX_HALO_WIDTH;
    static constexpr int MAX_HALO_HEIGHT =
        NeighborhoodPolicy::MAX_HALO_HEIGHT;

    int kernelWidth;
    int kernelHeight;
    int anchorX;
    int anchorY;

    FK_HOST_DEVICE_FUSE bool valid(
        const LinearFilterDPPDetails& details) {
        return NeighborhoodDPPStage<NeighborhoodPolicy>::valid(
            details.kernelWidth, details.kernelHeight,
            details.anchorX, details.anchorY, false);
    }
};

namespace linear_filter_detail {
// The operations' types: the multiply takes the image value and the
// coefficient and gives the accumulator type, and the accumulation takes and
// gives the accumulator type
template <typename DPPDetails, typename ReadIOps, typename MultiplyIOp,
          typename AccumulateIOp, typename WriteIOp>
FK_HOST_DEVICE_FUSE void checkIOps() {
    using Acc = typename DPPDetails::AccumulatorType;
    static_assert(isTuple_v<ReadIOps> &&
                  std::decay_t<ReadIOps>::size == 2,
                  "LinearFilterDPP needs image/coefficient Read IOps");
    static_assert(MultiplyIOp::template is<UnaryType> &&
                  AccumulateIOp::template is<UnaryType>,
                  "Multiply and accumulation must be Unary IOps");
    static_assert(isAnyWriteType<WriteIOp>,
                  "LinearFilterDPP needs a Write IOp");
    static_assert(std::is_same_v<typename MultiplyIOp::Operation::OutputType, Acc>,
                  "The multiply must give DPPDetails::AccumulatorType "
                  "(float with the channels of the image values)");
    static_assert(std::is_same_v<typename AccumulateIOp::Operation::OutputType, Acc>,
                  "The accumulation must give DPPDetails::AccumulatorType "
                  "(float with the channels of the image values)");
}
} // namespace linear_filter_detail

// Read IOp used by the box-filter wrapper. It participates in the same
// coefficient data path as arbitrary externally supplied coefficient Reads.
template <typename T>
struct ConstantFilterRead {
private:
    using Parent = ReadOperation<T, T, T, TF::ENABLED,
                                 ConstantFilterRead<T>>;
    using SelfType = ConstantFilterRead<T>;

public:
    FK_STATIC_STRUCT(ConstantFilterRead, SelfType)
    DECLARE_READ_PARENT

    template <uint ELEMS_PER_THREAD = 1>
    FK_HOST_DEVICE_FUSE auto exec(const Point,
                                  const ParamsType& value)
        -> ThreadFusionType<T, ELEMS_PER_THREAD, T> {
        static_assert(ELEMS_PER_THREAD == 1,
                      "ConstantFilterRead returns one coefficient");
        return value;
    }
};

template <ParArch PA, typename DPPDetails>
struct LinearFilterDPP;

template <typename DPPDetails>
struct LinearFilterDPP<ParArch::CPU, DPPDetails> {
private:
    using SelfType = LinearFilterDPP<ParArch::CPU, DPPDetails>;
    using T = typename DPPDetails::ValueType;
    using Acc = typename DPPDetails::AccumulatorType;
    using Stage = NeighborhoodDPPStage<
        typename DPPDetails::NeighborhoodPolicy>;

public:
    FK_STATIC_STRUCT(LinearFilterDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::CPU;

    // Computes row oy of image z; a row past image z's height computes
    // nothing. Used by exec() and by the CPU_MULTITHREAD version; the caller
    // checks DPPDetails::valid(details) and neighborhood_detail::canRun.
    template <typename ReadIOps, typename MultiplyIOp,
              typename AccumulateIOp, typename WriteIOp>
    FK_HOST_STATIC void exec_row(const int z, const int oy,
                                 const DPPDetails& details,
                                 const ReadIOps& reads,
                                 const MultiplyIOp& multiply,
                                 const AccumulateIOp& accumulate,
                                 const WriteIOp& output) {
        linear_filter_detail::checkIOps<DPPDetails, ReadIOps, MultiplyIOp,
                                        AccumulateIOp, WriteIOp>();
        const auto& image = get<0>(reads);
        const auto& coefficients = get<1>(reads);
        using C = typename std::decay_t<decltype(coefficients)>::Operation::OutputType;
        const int width = neighborhood_detail::widthOf(image, z);
        const int height = neighborhood_detail::heightOf(image, z);
        if (oy >= height) return;
        for (int ox = 0; ox < width; ++ox) {
            Acc accumulator{};
            for (int ky = 0; ky < details.kernelHeight; ++ky) {
                for (int kx = 0; kx < details.kernelWidth; ++kx) {
                    const T value = Stage::readReplicate(
                        width, height, z, image,
                        ox + kx - details.anchorX,
                        oy + ky - details.anchorY);
                    const C coefficient =
                        std::decay_t<decltype(coefficients)>::Operation::exec(
                            Point{kx, ky, 0}, coefficients);
                    const Acc product =
                        make_tuple(value, coefficient) | multiply;
                    accumulator =
                        make_tuple(accumulator, product) | accumulate;
                }
            }
            WriteIOp::Operation::exec(Point{ox, oy, z},
                                      accumulator, output);
        }
    }

    template <typename ReadIOps, typename MultiplyIOp,
              typename AccumulateIOp, typename WriteIOp>
    FK_HOST_STATIC void exec(const DPPDetails& details,
                             const ReadIOps& reads,
                             const MultiplyIOp& multiply,
                             const AccumulateIOp& accumulate,
                             const WriteIOp& output) {
        const auto& image = get<0>(reads);
        if (!DPPDetails::valid(details) ||
            !neighborhood_detail::canRun(image, output)) return;
        const int planes = static_cast<int>(
            neighborhood_detail::activeThreadsOf(image).z);
        for (int z = 0; z < planes; ++z) {
            const int height = neighborhood_detail::heightOf(image, z);
            for (int oy = 0; oy < height; ++oy) {
                exec_row(z, oy, details, reads, multiply, accumulate, output);
            }
        }
    }
};

#if defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)
// Runs the per row code of LinearFilterDPP<ParArch::CPU> on cpuThreadPool(),
// one chunk per row of every image (images x the largest height; a chunk
// past its image's height computes nothing), so both give the same results.
// As in the GPU
// version, the output must not alias the image or the coefficients: rows
// run concurrently, and a row reads the input rows around it.
template <typename DPPDetails>
struct LinearFilterDPP<ParArch::CPU_MULTITHREAD, DPPDetails> {
private:
    using SelfType = LinearFilterDPP<ParArch::CPU_MULTITHREAD, DPPDetails>;
    using CPUDPP = LinearFilterDPP<ParArch::CPU, DPPDetails>;

public:
    FK_STATIC_STRUCT(LinearFilterDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::CPU_MULTITHREAD;

    // Enqueues the rows in the pool and returns; done is called when all of
    // them have finished. Invalid details, or a write whose planes differ
    // from the read's images, enqueue no rows (nothing is written), as the
    // CPU version.
    template <typename ReadIOps, typename MultiplyIOp,
              typename AccumulateIOp, typename WriteIOp>
    FK_HOST_STATIC void exec(const cpu_multithread::Completion& done,
                             const DPPDetails& details,
                             const ReadIOps& reads,
                             const MultiplyIOp& multiply,
                             const AccumulateIOp& accumulate,
                             const WriteIOp& output) {
        const auto& image = get<0>(reads);
        const bool run = DPPDetails::valid(details) &&
                         neighborhood_detail::canRun(image, output);
        const ActiveThreads active =
            neighborhood_detail::activeThreadsOf(image);
        const std::size_t rows = static_cast<std::size_t>(active.y);
        const std::size_t chunks =
            run ? rows * static_cast<std::size_t>(active.z) : 0;
        cpu_multithread::enqueueChunks(chunks,
            [details, reads, multiply, accumulate, output, rows](
                const std::size_t chunk) {
                CPUDPP::exec_row(static_cast<int>(chunk / rows),
                                 static_cast<int>(chunk % rows), details,
                                 reads, multiply, accumulate, output);
            }, done);
    }
};
#endif // defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)

#if defined(__NVCC__)
template <typename DPPDetails>
struct LinearFilterDPP<ParArch::GPU_NVIDIA, DPPDetails> {
private:
    using SelfType = LinearFilterDPP<ParArch::GPU_NVIDIA, DPPDetails>;
    using T = typename DPPDetails::ValueType;
    using Acc = typename DPPDetails::AccumulatorType;
    using Stage = NeighborhoodDPPStage<
        typename DPPDetails::NeighborhoodPolicy>;

public:
    FK_STATIC_STRUCT(LinearFilterDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::GPU_NVIDIA;

    template <typename ReadIOps, typename MultiplyIOp,
              typename AccumulateIOp, typename WriteIOp>
    FK_DEVICE_STATIC void exec(const DPPDetails& details,
                               const ReadIOps& reads,
                               const MultiplyIOp& multiply,
                               const AccumulateIOp& accumulate,
                               const WriteIOp& output) {
        linear_filter_detail::checkIOps<DPPDetails, ReadIOps, MultiplyIOp,
                                        AccumulateIOp, WriteIOp>();
        const auto& image = get<0>(reads);
        const auto& coefficientRead = get<1>(reads);
        using C = typename std::decay_t<decltype(coefficientRead)>::Operation::OutputType;

        __shared__ T halo[DPPDetails::MAX_HALO_WIDTH *
                          DPPDetails::MAX_HALO_HEIGHT];
        __shared__ C coefficients[DPPDetails::MAX_KERNEL_WIDTH *
                                  DPPDetails::MAX_KERNEL_HEIGHT];
        // Image z = blockIdx.z, at its own size
        const int z = static_cast<int>(blockIdx.z);
        const int width = neighborhood_detail::widthOf(image, z);
        const int height = neighborhood_detail::heightOf(image, z);
        const int tileX = blockIdx.x * DPPDetails::TILE_WIDTH;
        const int tileY = blockIdx.y * DPPDetails::TILE_HEIGHT;
        // The grid covers the largest image, so a tile can start past image
        // z. The condition is the same for every thread of the block, so the
        // block returns before any __syncthreads.
        if (tileX >= width || tileY >= height) return;
        const int tid = threadIdx.y * blockDim.x + threadIdx.x;
        const int threads = blockDim.x * blockDim.y;
        const int haloWidth = DPPDetails::TILE_WIDTH +
                              details.kernelWidth - 1;

        Stage::stageReplicate(
            width, height, z,
            details.kernelWidth, details.kernelHeight,
            details.anchorX, details.anchorY, image, halo);
        for (int index = tid;
             index < details.kernelWidth * details.kernelHeight;
             index += threads) {
            coefficients[index] =
                std::decay_t<decltype(coefficientRead)>::Operation::exec(
                    Point{index % details.kernelWidth,
                          index / details.kernelWidth, 0},
                    coefficientRead);
        }
        __syncthreads();

        const int ox = tileX + threadIdx.x;
        const int oy = tileY + threadIdx.y;
        if (threadIdx.x < DPPDetails::TILE_WIDTH &&
            threadIdx.y < DPPDetails::TILE_HEIGHT &&
            ox < width && oy < height) {
            Acc accumulator{};
            for (int ky = 0; ky < details.kernelHeight; ++ky) {
                for (int kx = 0; kx < details.kernelWidth; ++kx) {
                    const T value = halo[
                        (threadIdx.y + ky) * haloWidth +
                        threadIdx.x + kx];
                    const C coefficient = coefficients[
                        ky * details.kernelWidth + kx];
                    const Acc product =
                        make_tuple(value, coefficient) | multiply;
                    accumulator =
                        make_tuple(accumulator, product) | accumulate;
                }
            }
            WriteIOp::Operation::exec(Point{ox, oy, z},
                                      accumulator, output);
        }
    }
};

template <ParArch PA, typename DPPDetails, typename ReadIOps,
          typename MultiplyIOp, typename AccumulateIOp, typename WriteIOp>
__global__ void linearFilterDPPKernel(const DPPDetails details,
                                      const ReadIOps reads,
                                      const MultiplyIOp multiply,
                                      const AccumulateIOp accumulate,
                                      const WriteIOp output) {
    LinearFilterDPP<PA, DPPDetails>::exec(
        details, reads, multiply, accumulate, output);
}

template <typename DPPDetails, typename ReadIOps,
          typename MultiplyIOp, typename AccumulateIOp, typename WriteIOp>
inline bool executeLinearFilter(const DPPDetails& details,
                                const ReadIOps& reads,
                                const MultiplyIOp& multiply,
                                const AccumulateIOp& accumulate,
                                const WriteIOp& output,
                                Stream_<ParArch::GPU_NVIDIA>& stream) {
    const auto& image = get<0>(reads);
    if (!DPPDetails::valid(details) ||
        !neighborhood_detail::canRun(image, output)) return false;
    // x and y cover the largest image, z the images
    const ActiveThreads active = neighborhood_detail::activeThreadsOf(image);
    const dim3 block(DPPDetails::TILE_WIDTH,
                     DPPDetails::TILE_HEIGHT, 1);
    const dim3 grid(ceilDiv(active.x, DPPDetails::TILE_WIDTH),
                    ceilDiv(active.y, DPPDetails::TILE_HEIGHT),
                    active.z);
    linearFilterDPPKernel<ParArch::GPU_NVIDIA><<<grid, block, 0, stream.getCUDAStream()>>>(
        details, reads, multiply, accumulate, output);
    gpuErrchk(cudaGetLastError());
    return true;
}

template <typename DPPDetails, typename ImageRead, typename WriteIOp>
inline bool executeBoxFilter(const DPPDetails& details,
                             const ImageRead& image,
                             const WriteIOp& output,
                             const float scale,
                             Stream_<ParArch::GPU_NVIDIA>& stream) {
    using T = typename DPPDetails::ValueType;
    using Acc = typename DPPDetails::AccumulatorType;
    if (!DPPDetails::valid(details)) return false;
    // One scale for every channel: the coefficient is scale / (kernel
    // size) in every channel of the accumulator type
    const Acc coefficient = make_set<Acc>(scale /
        static_cast<float>(details.kernelWidth * details.kernelHeight));
    const auto coefficientRead = ConstantFilterRead<Acc>::build(coefficient);
    const auto multiply = Mul<T, Acc, Acc, UnaryType>::build();
    const auto accumulate = Add<Acc, Acc, Acc, UnaryType>::build();
    return executeLinearFilter(
        details, make_tuple(image, coefficientRead),
        multiply, accumulate, output, stream);
}
#endif // defined(__NVCC__)

#if defined(FK_HIP_BACKEND)
// The GPU_NVIDIA version above, for HIP (ParArch::GPU_AMD), with the HIP runtime API
template <typename DPPDetails>
struct LinearFilterDPP<ParArch::GPU_AMD, DPPDetails> {
private:
    using SelfType = LinearFilterDPP<ParArch::GPU_AMD, DPPDetails>;
    using T = typename DPPDetails::ValueType;
    using Acc = typename DPPDetails::AccumulatorType;
    using Stage = NeighborhoodDPPStage<
        typename DPPDetails::NeighborhoodPolicy>;

public:
    FK_STATIC_STRUCT(LinearFilterDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::GPU_AMD;

    template <typename ReadIOps, typename MultiplyIOp,
              typename AccumulateIOp, typename WriteIOp>
    FK_DEVICE_STATIC void exec(const DPPDetails& details,
                               const ReadIOps& reads,
                               const MultiplyIOp& multiply,
                               const AccumulateIOp& accumulate,
                               const WriteIOp& output) {
        linear_filter_detail::checkIOps<DPPDetails, ReadIOps, MultiplyIOp,
                                        AccumulateIOp, WriteIOp>();
        const auto& image = get<0>(reads);
        const auto& coefficientRead = get<1>(reads);
        using C = typename std::decay_t<decltype(coefficientRead)>::Operation::OutputType;

        __shared__ T halo[DPPDetails::MAX_HALO_WIDTH *
                          DPPDetails::MAX_HALO_HEIGHT];
        __shared__ C coefficients[DPPDetails::MAX_KERNEL_WIDTH *
                                  DPPDetails::MAX_KERNEL_HEIGHT];
        // Image z = blockIdx.z, at its own size
        const int z = static_cast<int>(blockIdx.z);
        const int width = neighborhood_detail::widthOf(image, z);
        const int height = neighborhood_detail::heightOf(image, z);
        const int tileX = blockIdx.x * DPPDetails::TILE_WIDTH;
        const int tileY = blockIdx.y * DPPDetails::TILE_HEIGHT;
        // The grid covers the largest image, so a tile can start past image
        // z. The condition is the same for every thread of the block, so the
        // block returns before any __syncthreads.
        if (tileX >= width || tileY >= height) return;
        const int tid = threadIdx.y * blockDim.x + threadIdx.x;
        const int threads = blockDim.x * blockDim.y;
        const int haloWidth = DPPDetails::TILE_WIDTH +
                              details.kernelWidth - 1;

        Stage::stageReplicate(
            width, height, z,
            details.kernelWidth, details.kernelHeight,
            details.anchorX, details.anchorY, image, halo);
        for (int index = tid;
             index < details.kernelWidth * details.kernelHeight;
             index += threads) {
            coefficients[index] =
                std::decay_t<decltype(coefficientRead)>::Operation::exec(
                    Point{index % details.kernelWidth,
                          index / details.kernelWidth, 0},
                    coefficientRead);
        }
        __syncthreads();

        const int ox = tileX + threadIdx.x;
        const int oy = tileY + threadIdx.y;
        if (threadIdx.x < DPPDetails::TILE_WIDTH &&
            threadIdx.y < DPPDetails::TILE_HEIGHT &&
            ox < width && oy < height) {
            Acc accumulator{};
            for (int ky = 0; ky < details.kernelHeight; ++ky) {
                for (int kx = 0; kx < details.kernelWidth; ++kx) {
                    const T value = halo[
                        (threadIdx.y + ky) * haloWidth +
                        threadIdx.x + kx];
                    const C coefficient = coefficients[
                        ky * details.kernelWidth + kx];
                    const Acc product =
                        make_tuple(value, coefficient) | multiply;
                    accumulator =
                        make_tuple(accumulator, product) | accumulate;
                }
            }
            WriteIOp::Operation::exec(Point{ox, oy, z},
                                      accumulator, output);
        }
    }
};

template <ParArch PA, typename DPPDetails, typename ReadIOps,
          typename MultiplyIOp, typename AccumulateIOp, typename WriteIOp>
__global__ void linearFilterDPPKernel(const DPPDetails details,
                                      const ReadIOps reads,
                                      const MultiplyIOp multiply,
                                      const AccumulateIOp accumulate,
                                      const WriteIOp output) {
    LinearFilterDPP<PA, DPPDetails>::exec(
        details, reads, multiply, accumulate, output);
}

template <typename DPPDetails, typename ReadIOps,
          typename MultiplyIOp, typename AccumulateIOp, typename WriteIOp>
inline bool executeLinearFilter(const DPPDetails& details,
                                const ReadIOps& reads,
                                const MultiplyIOp& multiply,
                                const AccumulateIOp& accumulate,
                                const WriteIOp& output,
                                Stream_<ParArch::GPU_AMD>& stream) {
    const auto& image = get<0>(reads);
    if (!DPPDetails::valid(details) ||
        !neighborhood_detail::canRun(image, output)) return false;
    // x and y cover the largest image, z the images
    const ActiveThreads active = neighborhood_detail::activeThreadsOf(image);
    const dim3 block(DPPDetails::TILE_WIDTH,
                     DPPDetails::TILE_HEIGHT, 1);
    const dim3 grid(ceilDiv(active.x, DPPDetails::TILE_WIDTH),
                    ceilDiv(active.y, DPPDetails::TILE_HEIGHT),
                    active.z);
    linearFilterDPPKernel<ParArch::GPU_AMD><<<grid, block, 0, stream.getHIPStream()>>>(
        details, reads, multiply, accumulate, output);
    gpuErrchk(hipGetLastError());
    return true;
}

template <typename DPPDetails, typename ImageRead, typename WriteIOp>
inline bool executeBoxFilter(const DPPDetails& details,
                             const ImageRead& image,
                             const WriteIOp& output,
                             const float scale,
                             Stream_<ParArch::GPU_AMD>& stream) {
    using T = typename DPPDetails::ValueType;
    using Acc = typename DPPDetails::AccumulatorType;
    if (!DPPDetails::valid(details)) return false;
    // One scale for every channel: the coefficient is scale / (kernel
    // size) in every channel of the accumulator type
    const Acc coefficient = make_set<Acc>(scale /
        static_cast<float>(details.kernelWidth * details.kernelHeight));
    const auto coefficientRead = ConstantFilterRead<Acc>::build(coefficient);
    const auto multiply = Mul<T, Acc, Acc, UnaryType>::build();
    const auto accumulate = Add<Acc, Acc, Acc, UnaryType>::build();
    return executeLinearFilter(
        details, make_tuple(image, coefficientRead),
        multiply, accumulate, output, stream);
}
#endif // defined(FK_HIP_BACKEND)

#if defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)
// Enqueues LinearFilterDPP<ParArch::CPU_MULTITHREAD> on the stream and
// returns; returns false, enqueueing nothing, if the details are not valid
// or the write's planes differ from the read's images.
// The execution keeps copies of the IOps. As with a GPU stream, the buffers
// they read and write must stay valid until it has finished (sync()), and
// the output must not alias the image or the coefficients.
template <typename DPPDetails, typename ReadIOps,
          typename MultiplyIOp, typename AccumulateIOp, typename WriteIOp>
inline bool executeLinearFilter(const DPPDetails& details,
                                const ReadIOps& reads,
                                const MultiplyIOp& multiply,
                                const AccumulateIOp& accumulate,
                                const WriteIOp& output,
                                Stream_<ParArch::CPU_MULTITHREAD>& stream) {
    if (!DPPDetails::valid(details) ||
        !neighborhood_detail::canRun(get<0>(reads), output)) return false;
    stream.enqueue([=](const cpu_multithread::Completion& done) {
        LinearFilterDPP<ParArch::CPU_MULTITHREAD, DPPDetails>::exec(
            done, details, reads, multiply, accumulate, output);
    });
    return true;
}

template <typename DPPDetails, typename ImageRead, typename WriteIOp>
inline bool executeBoxFilter(const DPPDetails& details,
                             const ImageRead& image,
                             const WriteIOp& output,
                             const float scale,
                             Stream_<ParArch::CPU_MULTITHREAD>& stream) {
    using T = typename DPPDetails::ValueType;
    using Acc = typename DPPDetails::AccumulatorType;
    if (!DPPDetails::valid(details)) return false;
    // One scale for every channel: the coefficient is scale / (kernel
    // size) in every channel of the accumulator type
    const Acc coefficient = make_set<Acc>(scale /
        static_cast<float>(details.kernelWidth * details.kernelHeight));
    const auto coefficientRead = ConstantFilterRead<Acc>::build(coefficient);
    const auto multiply = Mul<T, Acc, Acc, UnaryType>::build();
    const auto accumulate = Add<Acc, Acc, Acc, UnaryType>::build();
    // executeLinearFilter copies these locals into the execution
    return executeLinearFilter(
        details, make_tuple(image, coefficientRead),
        multiply, accumulate, output, stream);
}
#endif // defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)

} // namespace fk

#endif // FK_LINEAR_FILTER_DPP_H
