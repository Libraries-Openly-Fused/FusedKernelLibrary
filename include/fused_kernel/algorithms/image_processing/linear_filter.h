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
#include <fused_kernel/fused_kernel.h>
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

template <typename T, int TILE_W = 16, int TILE_H = 8,
          int MAX_KERNEL_W = 7, int MAX_KERNEL_H = 7>
struct LinearFilterDPPDetails {
    using ValueType = T;
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

    int width;
    int height;
    int kernelWidth;
    int kernelHeight;
    int anchorX;
    int anchorY;

    FK_HOST_DEVICE_FUSE bool valid(
        const LinearFilterDPPDetails& details) {
        return NeighborhoodDPPStage<NeighborhoodPolicy>::valid(
            details.width, details.height,
            details.kernelWidth, details.kernelHeight,
            details.anchorX, details.anchorY, false);
    }
};

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
    using Stage = NeighborhoodDPPStage<
        typename DPPDetails::NeighborhoodPolicy>;

public:
    FK_STATIC_STRUCT(LinearFilterDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::CPU;

    // Computes output row oy. Used by exec() and by the CPU_MULTITHREAD
    // version; the caller checks DPPDetails::valid(details).
    template <typename ReadIOps, typename MultiplyIOp,
              typename AccumulateIOp, typename WriteIOp>
    FK_HOST_STATIC void exec_row(const int oy,
                                 const DPPDetails& details,
                                 const ReadIOps& reads,
                                 const MultiplyIOp& multiply,
                                 const AccumulateIOp& accumulate,
                                 const WriteIOp& output) {
        static_assert(isTuple_v<ReadIOps> &&
                      std::decay_t<ReadIOps>::size == 2,
                      "LinearFilterDPP needs image/coefficient Read IOps");
        static_assert(MultiplyIOp::template is<UnaryType> &&
                      AccumulateIOp::template is<UnaryType>,
                      "Multiply and accumulation must be Unary IOps");
        static_assert(isAnyWriteType<WriteIOp>,
                      "LinearFilterDPP needs a Write IOp");
        const auto& image = get<0>(reads);
        const auto& coefficients = get<1>(reads);
        for (int ox = 0; ox < details.width; ++ox) {
            T accumulator{};
            for (int ky = 0; ky < details.kernelHeight; ++ky) {
                for (int kx = 0; kx < details.kernelWidth; ++kx) {
                    const T value = Stage::readReplicate(
                        details.width, details.height, image,
                        ox + kx - details.anchorX,
                        oy + ky - details.anchorY);
                    const T coefficient =
                        std::decay_t<decltype(coefficients)>::Operation::exec(
                            Point{kx, ky, 0}, coefficients);
                    const T product =
                        make_tuple(value, coefficient) | multiply;
                    accumulator =
                        make_tuple(accumulator, product) | accumulate;
                }
            }
            WriteIOp::Operation::exec(Point{ox, oy, 0},
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
        if (!DPPDetails::valid(details)) return;
        for (int oy = 0; oy < details.height; ++oy) {
            exec_row(oy, details, reads, multiply, accumulate, output);
        }
    }
};

#if defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)
// Runs the per row code of LinearFilterDPP<ParArch::CPU> on cpuThreadPool(),
// one chunk per output row, so both give the same results. As in the GPU
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
    // them have finished. Invalid details enqueue no rows (nothing is
    // written), as the CPU version.
    template <typename ReadIOps, typename MultiplyIOp,
              typename AccumulateIOp, typename WriteIOp>
    FK_HOST_STATIC void exec(const cpu_multithread::Completion& done,
                             const DPPDetails& details,
                             const ReadIOps& reads,
                             const MultiplyIOp& multiply,
                             const AccumulateIOp& accumulate,
                             const WriteIOp& output) {
        const std::size_t rows = DPPDetails::valid(details)
            ? static_cast<std::size_t>(details.height) : 0;
        cpu_multithread::enqueueChunks(rows,
            [details, reads, multiply, accumulate, output](
                const std::size_t row) {
                CPUDPP::exec_row(static_cast<int>(row), details, reads,
                                 multiply, accumulate, output);
            }, done);
    }
};
#endif // defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)

#if defined(__NVCC__) || defined(__HIPCC__)
template <ParArch PA, typename DPPDetails>
    requires(PA == ParArch::GPU_NVIDIA || PA == ParArch::GPU_AMD)
struct LinearFilterDPP<PA, DPPDetails> {
private:
    using SelfType = LinearFilterDPP<PA, DPPDetails>;
    using T = typename DPPDetails::ValueType;
    using Stage = NeighborhoodDPPStage<
        typename DPPDetails::NeighborhoodPolicy>;

public:
    FK_STATIC_STRUCT(LinearFilterDPP, SelfType)
    static constexpr ParArch PAR_ARCH = PA;

    template <typename ReadIOps, typename MultiplyIOp,
              typename AccumulateIOp, typename WriteIOp>
    FK_DEVICE_STATIC void exec(const DPPDetails& details,
                               const ReadIOps& reads,
                               const MultiplyIOp& multiply,
                               const AccumulateIOp& accumulate,
                               const WriteIOp& output) {
        static_assert(isTuple_v<ReadIOps> &&
                      std::decay_t<ReadIOps>::size == 2,
                      "LinearFilterDPP needs image/coefficient Read IOps");
        static_assert(MultiplyIOp::template is<UnaryType> &&
                      AccumulateIOp::template is<UnaryType>,
                      "Multiply and accumulation must be Unary IOps");
        static_assert(isAnyWriteType<WriteIOp>,
                      "LinearFilterDPP needs a Write IOp");

        __shared__ T halo[DPPDetails::MAX_HALO_WIDTH *
                          DPPDetails::MAX_HALO_HEIGHT];
        __shared__ T coefficients[DPPDetails::MAX_KERNEL_WIDTH *
                                  DPPDetails::MAX_KERNEL_HEIGHT];
        const auto& image = get<0>(reads);
        const auto& coefficientRead = get<1>(reads);
        const int tid = threadIdx.y * blockDim.x + threadIdx.x;
        const int threads = blockDim.x * blockDim.y;
        const int haloWidth = DPPDetails::TILE_WIDTH +
                              details.kernelWidth - 1;
        const int tileX = blockIdx.x * DPPDetails::TILE_WIDTH;
        const int tileY = blockIdx.y * DPPDetails::TILE_HEIGHT;

        Stage::stageReplicate(
            details.width, details.height,
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
            ox < details.width && oy < details.height) {
            T accumulator{};
            for (int ky = 0; ky < details.kernelHeight; ++ky) {
                for (int kx = 0; kx < details.kernelWidth; ++kx) {
                    const T value = halo[
                        (threadIdx.y + ky) * haloWidth +
                        threadIdx.x + kx];
                    const T coefficient = coefficients[
                        ky * details.kernelWidth + kx];
                    const T product =
                        make_tuple(value, coefficient) | multiply;
                    accumulator =
                        make_tuple(accumulator, product) | accumulate;
                }
            }
            WriteIOp::Operation::exec(Point{ox, oy, 0},
                                      accumulator, output);
        }
    }
};

#endif // defined(__NVCC__) || defined(__HIPCC__)

#if defined(FK_SYCL_ENABLED)
// The SYCL version: the code of the CUDA one. The halo and the coefficients
// are in work-group local memory (the shared memory of the block), which is
// allocated when the kernel is enqueued (by its executor), and the thread
// gets its position in the grid in the nd_item.
template <typename DPPDetails>
struct LinearFilterDPP<ParArch::GPU_INTEL, DPPDetails> {
private:
    using SelfType = LinearFilterDPP<ParArch::GPU_INTEL, DPPDetails>;
    using T = typename DPPDetails::ValueType;
    using Stage = NeighborhoodDPPStage<
        typename DPPDetails::NeighborhoodPolicy>;

public:
    FK_STATIC_STRUCT(LinearFilterDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::GPU_INTEL;
    // Elements of the arrays of T exec needs in local memory: halo and
    // coefficients
    static constexpr std::array<std::size_t, 2> LOCAL_ELEMENTS{
        static_cast<std::size_t>(DPPDetails::MAX_HALO_WIDTH *
                                 DPPDetails::MAX_HALO_HEIGHT),
        static_cast<std::size_t>(DPPDetails::MAX_KERNEL_WIDTH *
                                 DPPDetails::MAX_KERNEL_HEIGHT)};

    template <typename ReadIOps, typename MultiplyIOp,
              typename AccumulateIOp, typename WriteIOp>
    FK_DEVICE_STATIC void exec(const sycl::nd_item<3>& item,
                               T* const halo,
                               T* const coefficients,
                               const DPPDetails& details,
                               const ReadIOps& reads,
                               const MultiplyIOp& multiply,
                               const AccumulateIOp& accumulate,
                               const WriteIOp& output) {
        static_assert(isTuple_v<ReadIOps> &&
                      std::decay_t<ReadIOps>::size == 2,
                      "LinearFilterDPP needs image/coefficient Read IOps");
        static_assert(MultiplyIOp::template is<UnaryType> &&
                      AccumulateIOp::template is<UnaryType>,
                      "Multiply and accumulation must be Unary IOps");
        static_assert(isAnyWriteType<WriteIOp>,
                      "LinearFilterDPP needs a Write IOp");

        const ThreadPosition thread(item);
        const auto& image = get<0>(reads);
        const auto& coefficientRead = get<1>(reads);
        const int tid = thread.threadIdx.y * thread.blockDim.x +
                        thread.threadIdx.x;
        const int threads = thread.blockDim.x * thread.blockDim.y;
        const int haloWidth = DPPDetails::TILE_WIDTH +
                              details.kernelWidth - 1;
        const int tileX = thread.blockIdx.x * DPPDetails::TILE_WIDTH;
        const int tileY = thread.blockIdx.y * DPPDetails::TILE_HEIGHT;

        Stage::stageReplicate(
            item, details.width, details.height,
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
        sycl::group_barrier(item.get_group());

        const int ox = tileX + thread.threadIdx.x;
        const int oy = tileY + thread.threadIdx.y;
        if (thread.threadIdx.x < DPPDetails::TILE_WIDTH &&
            thread.threadIdx.y < DPPDetails::TILE_HEIGHT &&
            ox < details.width && oy < details.height) {
            T accumulator{};
            for (int ky = 0; ky < details.kernelHeight; ++ky) {
                for (int kx = 0; kx < details.kernelWidth; ++kx) {
                    const T value = halo[
                        (thread.threadIdx.y + ky) * haloWidth +
                        thread.threadIdx.x + kx];
                    const T coefficient = coefficients[
                        ky * details.kernelWidth + kx];
                    const T product =
                        make_tuple(value, coefficient) | multiply;
                    accumulator =
                        make_tuple(accumulator, product) | accumulate;
                }
            }
            WriteIOp::Operation::exec(Point{ox, oy, 0},
                                      accumulator, output);
        }
    }
};
#endif // defined(FK_SYCL_ENABLED)

namespace linear_filter_detail {
// The check of the executors: valid details (a non empty image, a kernel of 1 to MAX_KERNEL_WIDTH x MAX_KERNEL_HEIGHT,
// the anchor inside it)
template <typename DPPDetails>
FK_HOST_FUSE void checkArguments(const DPPDetails& details) {
    executor_detail::requireArguments(DPPDetails::valid(details), "LinearFilterDPP",
        "invalid details: the image must not be empty, and the kernel must be 1 to MAX_KERNEL_WIDTH x "
        "MAX_KERNEL_HEIGHT, with the anchor inside it");
}
} // namespace linear_filter_detail

// The executors of LinearFilterDPP, one per back-end:
//   executeOperations<LinearFilterDPP<PA, DPPDetails>>(stream, details, reads, multiply, accumulate, output)
// with reads = make_tuple(image read, coefficient read). Invalid details throw std::invalid_argument before anything is
// launched or enqueued (Oscar's decision, 2026-11-26). A box filter is the coefficient read
// ConstantFilterRead<T>::build(scale / (kernelWidth * kernelHeight)) with Mul<T, T, T, UnaryType> and
// Add<T, T, T, UnaryType>, where T is DPPDetails::ValueType.
template <typename DPPDetails>
struct Executor<LinearFilterDPP<ParArch::CPU, DPPDetails>> {
  private:
    using DPP = LinearFilterDPP<ParArch::CPU, DPPDetails>;
    using SelfType = Executor<DPP>;

  public:
    FK_STATIC_STRUCT(Executor, SelfType)
    FK_HOST_FUSE ParArch parArch() { return ParArch::CPU; }
    // Computes the filter before it returns
    template <typename ReadIOps, typename MultiplyIOp, typename AccumulateIOp, typename WriteIOp>
    FK_HOST_STATIC void executeOperations(Stream_<ParArch::CPU>&, const DPPDetails& details, const ReadIOps& reads,
                                          const MultiplyIOp& multiply, const AccumulateIOp& accumulate,
                                          const WriteIOp& output) {
        linear_filter_detail::checkArguments(details);
        DPP::exec(details, reads, multiply, accumulate, output);
    }
};

#if defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)
template <typename DPPDetails>
struct Executor<LinearFilterDPP<ParArch::CPU_MULTITHREAD, DPPDetails>> {
  private:
    using DPP = LinearFilterDPP<ParArch::CPU_MULTITHREAD, DPPDetails>;
    using SelfType = Executor<DPP>;

  public:
    FK_STATIC_STRUCT(Executor, SelfType)
    FK_HOST_FUSE ParArch parArch() { return ParArch::CPU_MULTITHREAD; }
    // Enqueues LinearFilterDPP<ParArch::CPU_MULTITHREAD> on the stream and returns. The execution keeps copies of the
    // IOps. As with a GPU stream, the buffers they read and write must stay valid until it has finished (sync()), and the
    // output must not alias the image or the coefficients.
    template <typename ReadIOps, typename MultiplyIOp, typename AccumulateIOp, typename WriteIOp>
    FK_HOST_STATIC void executeOperations(Stream_<ParArch::CPU_MULTITHREAD>& stream, const DPPDetails& details,
                                          const ReadIOps& reads, const MultiplyIOp& multiply,
                                          const AccumulateIOp& accumulate, const WriteIOp& output) {
        linear_filter_detail::checkArguments(details);
        stream.enqueue([=](const cpu_multithread::Completion& done) {
            DPP::exec(done, details, reads, multiply, accumulate, output);
        });
    }
};
#endif // defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)

#if defined(__NVCC__) || defined(__HIPCC__)
template <ParArch PA, typename DPPDetails>
    requires(PA == ParArch::GPU_NVIDIA || PA == ParArch::GPU_AMD)
struct Executor<LinearFilterDPP<PA, DPPDetails>> {
  private:
    using DPP = LinearFilterDPP<PA, DPPDetails>;
    using SelfType = Executor<DPP>;

  public:
    FK_STATIC_STRUCT(Executor, SelfType)
    FK_HOST_FUSE ParArch parArch() { return PA; }
    template <typename ReadIOps, typename MultiplyIOp, typename AccumulateIOp, typename WriteIOp>
    FK_HOST_STATIC void executeOperations(Stream_<PA>& stream, const DPPDetails& details, const ReadIOps& reads,
                                          const MultiplyIOp& multiply, const AccumulateIOp& accumulate,
                                          const WriteIOp& output) {
        linear_filter_detail::checkArguments(details);
        const dim3 block(DPPDetails::TILE_WIDTH, DPPDetails::TILE_HEIGHT, 1);
        const dim3 grid((details.width + DPPDetails::TILE_WIDTH - 1) / DPPDetails::TILE_WIDTH,
                        (details.height + DPPDetails::TILE_HEIGHT - 1) / DPPDetails::TILE_HEIGHT, 1);
#if defined(__HIPCC__)
        launchDPP_Kernel<DPP><<<grid, block, 0, stream.getHIPStream()>>>(details, reads, multiply, accumulate, output);
        gpuErrchk(hipGetLastError());
#else
        launchDPP_Kernel<DPP><<<grid, block, 0, stream.getCUDAStream()>>>(details, reads, multiply, accumulate, output);
        gpuErrchk(cudaGetLastError());
#endif
    }
};
#endif // defined(__NVCC__) || defined(__HIPCC__)

#if defined(FK_SYCL_ENABLED)
template <typename DPPDetails>
struct Executor<LinearFilterDPP<ParArch::GPU_INTEL, DPPDetails>> {
  private:
    using DPP = LinearFilterDPP<ParArch::GPU_INTEL, DPPDetails>;
    using SelfType = Executor<DPP>;

  public:
    FK_STATIC_STRUCT(Executor, SelfType)
    FK_HOST_FUSE ParArch parArch() { return ParArch::GPU_INTEL; }
    // Enqueues the kernel in the queue of the stream, one block per tile, and returns: launchDPP_KernelWithLocalMemory
    // (executor_kernels.h) is the SYCL launchDPP_Kernel of a DPP with shared memory, and allocates the halo and the
    // coefficients of the DPP (DPP::LOCAL_ELEMENTS) in work-group local memory, the shared memory of the block. As with
    // the CUDA version, the output must not alias the image or the coefficients.
    template <typename ReadIOps, typename MultiplyIOp, typename AccumulateIOp, typename WriteIOp>
    FK_HOST_STATIC void executeOperations(Stream_<ParArch::GPU_INTEL>& stream, const DPPDetails& details,
                                          const ReadIOps& reads, const MultiplyIOp& multiply,
                                          const AccumulateIOp& accumulate, const WriteIOp& output) {
        linear_filter_detail::checkArguments(details);
        const CtxDim3 block{static_cast<uint>(DPPDetails::TILE_WIDTH), static_cast<uint>(DPPDetails::TILE_HEIGHT), 1};
        const CtxDim3 grid{static_cast<uint>((details.width + DPPDetails::TILE_WIDTH - 1) / DPPDetails::TILE_WIDTH),
                           static_cast<uint>((details.height + DPPDetails::TILE_HEIGHT - 1) / DPPDetails::TILE_HEIGHT),
                           1};
        launchDPP_KernelWithLocalMemory<DPP, typename DPPDetails::ValueType>(
            stream, syclGrid(grid, block), DPP::LOCAL_ELEMENTS, details, reads, multiply, accumulate, output);
    }
};
#endif // defined(FK_SYCL_ENABLED)

} // namespace fk

#endif // FK_LINEAR_FILTER_DPP_H
