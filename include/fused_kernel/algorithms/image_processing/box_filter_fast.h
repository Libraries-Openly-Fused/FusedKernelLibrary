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

#ifndef FK_BOX_FILTER_FAST_H
#define FK_BOX_FILTER_FAST_H

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/image_processing/neighborhood.h>
#include <fused_kernel/core/data/point.h>
#include <fused_kernel/core/execution_model/parallel_architectures.h>
#include <fused_kernel/core/execution_model/stream.h>
#include <fused_kernel/core/utils/vector_utils.h>

#include <type_traits>
#if defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)
#include <fused_kernel/core/execution_model/cpu_thread_pool.h>

#include <cstddef>
#endif

namespace fk {

// The window and its anchor. The images come from the input read: their
// number is z of its active threads (the batch of a batched read, the
// planes of a 3D read, 1 for a 2D read), and image z is filtered at its own
// size (the read's num_elems at z), read and written at Point{x, y, z}.
// The write must have as many planes as the read has images; otherwise the
// executor throws std::invalid_argument and nothing is written.
struct BoxFilterQuadDetails {
    int kernelWidth;
    int kernelHeight;
    int anchorX;
    int anchorY;

    template <int STATIC_KW, int STATIC_KH>
    FK_HOST_DEVICE_CNST bool validFor(
            const int maxRuntimeKernelWidth) const {
        const int effectiveWidth =
            STATIC_KW > 0 ? STATIC_KW : kernelWidth;
        const int effectiveHeight =
            STATIC_KH > 0 ? STATIC_KH : kernelHeight;
        return effectiveWidth > 0 && effectiveHeight > 0 &&
               (STATIC_KW > 0 || effectiveWidth <= maxRuntimeKernelWidth) &&
               anchorX >= 0 && anchorX < effectiveWidth &&
               anchorY >= 0 && anchorY < effectiveHeight;
    }
};

template <ParArch PA, typename T, int EX = 4, int EY = 4,
          int KW = 0, int KH = 0>
struct BoxFilterQuadDPP;

template <typename T, int EX, int EY, int KW, int KH>
struct BoxFilterQuadDPP<ParArch::CPU, T, EX, EY, KW, KH> {
private:
    using SelfType = BoxFilterQuadDPP<ParArch::CPU, T, EX, EY, KW, KH>;
    // The values read, the sums and the means written: float with the
    // channels of T
    using Acc = VectorType_t<float, cn<T>>;

    FK_HOST_FUSE int clamp(const int value, const int upper) {
        return value < 0 ? 0 : (value >= upper ? upper - 1 : value);
    }

public:
    FK_STATIC_STRUCT(BoxFilterQuadDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::CPU;
    static constexpr int MAX_RUNTIME_KERNEL_WIDTH = 31;

    FK_HOST_DEVICE_FUSE bool accepts(const BoxFilterQuadDetails& details) {
        return details.template validFor<KW, KH>(
            MAX_RUNTIME_KERNEL_WIDTH);
    }

    // Computes the output pixels of row y of image z. Used by exec() and by
    // the CPU_MULTITHREAD specialization (one row of one image per chunk).
    // Expects accepts(details) and y >= 0; a row past the height of image z
    // computes nothing.
    template <typename InIOp, typename ComputeIOps, typename OutIOp>
    FK_HOST_FUSE void exec_row(const int z, const int y,
                               const BoxFilterQuadDetails& details,
                               const InIOp& input,
                               const ComputeIOps& compute,
                               const OutIOp& output) {
        static_assert(isAnyCompleteReadType<InIOp>,
                      "BoxFilterQuadDPP requires a complete Read IOp");
        static_assert(isAnyWriteType<OutIOp>,
                      "BoxFilterQuadDPP requires a Write IOp");
        static_assert(std::is_same_v<
                          typename InIOp::Operation::OutputType, Acc>,
                      "BoxFilterQuadDPP Read IOp must produce "
                      "VectorType_t<float, cn<T>>");
        static_assert(std::is_same_v<
                          typename OutIOp::Operation::InputType, Acc>,
                      "BoxFilterQuadDPP Write IOp must consume "
                      "VectorType_t<float, cn<T>>");
        static_assert(EX > 0 && EY > 0,
                      "BoxFilterQuadDPP output tile must be positive");
        static_assert(KW >= 0 && KH >= 0,
                      "BoxFilterQuadDPP static kernel sizes cannot be negative");
        static_assert((KW == 0) == (KH == 0),
                      "BoxFilterQuadDPP kernel dimensions are both static or both runtime");

        const int width = neighborhood_detail::widthOf(input, z);
        const int height = neighborhood_detail::heightOf(input, z);
        if (y >= height) return;
        const int kernelWidth = KW > 0 ? KW : details.kernelWidth;
        const int kernelHeight = KH > 0 ? KH : details.kernelHeight;
        const auto& combine = get<0>(compute);
        const auto& normalize = get<2>(compute);

        for (int x = 0; x < width; ++x) {
            Acc sum = make_set<Acc>(0.f);
            for (int ky = 0; ky < kernelHeight; ++ky) {
                for (int kx = 0; kx < kernelWidth; ++kx) {
                    const Point source{
                        clamp(x + kx - details.anchorX, width),
                        clamp(y + ky - details.anchorY, height), z};
                    const Acc value =
                        InIOp::Operation::exec(source, input);
                    sum = make_tuple(sum, value) | combine;
                }
            }
            const Acc mean = sum | normalize;
            OutIOp::Operation::exec(Point{x, y, z}, mean, output);
        }
    }

    template <typename InIOp, typename ComputeIOps, typename OutIOp>
    FK_HOST_FUSE void exec(const BoxFilterQuadDetails& details,
                           const InIOp& input,
                           const ComputeIOps& compute,
                           const OutIOp& output) {
        if (!accepts(details) ||
            !neighborhood_detail::canRun(input, output)) return;
        const int planes = static_cast<int>(
            neighborhood_detail::activeThreadsOf(input).z);
        for (int z = 0; z < planes; ++z) {
            const int height = neighborhood_detail::heightOf(input, z);
            for (int y = 0; y < height; ++y) {
                exec_row(z, y, details, input, compute, output);
            }
        }
    }
};

#if defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)
// Runs the rows of BoxFilterQuadDPP<ParArch::CPU> on cpuThreadPool(), one
// chunk per row of every image (images x the largest height). As on the GPU,
// the output must not alias the input: a row reads the input rows around it,
// which other chunks may be writing.
template <typename T, int EX, int EY, int KW, int KH>
struct BoxFilterQuadDPP<ParArch::CPU_MULTITHREAD, T, EX, EY, KW, KH> {
private:
    using SelfType = BoxFilterQuadDPP<
        ParArch::CPU_MULTITHREAD, T, EX, EY, KW, KH>;
    using CPUDPP = BoxFilterQuadDPP<ParArch::CPU, T, EX, EY, KW, KH>;

public:
    FK_STATIC_STRUCT(BoxFilterQuadDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::CPU_MULTITHREAD;
    static constexpr int MAX_RUNTIME_KERNEL_WIDTH =
        CPUDPP::MAX_RUNTIME_KERNEL_WIDTH;

    FK_HOST_DEVICE_FUSE bool accepts(const BoxFilterQuadDetails& details) {
        return CPUDPP::accepts(details);
    }

    // Enqueues the rows in the pool and returns; done is called when all of
    // them have finished. Rejected details enqueue no rows, like exec() of
    // the CPU specialization, which returns without writing.
    template <typename InIOp, typename ComputeIOps, typename OutIOp>
    FK_HOST_STATIC void exec(const cpu_multithread::Completion& done,
                             const BoxFilterQuadDetails& details,
                             const InIOp& input,
                             const ComputeIOps& compute,
                             const OutIOp& output) {
        const bool run = accepts(details) &&
                         neighborhood_detail::canRun(input, output);
        const ActiveThreads active =
            neighborhood_detail::activeThreadsOf(input);
        const std::size_t rows = static_cast<std::size_t>(active.y);
        const std::size_t chunks =
            run ? rows * static_cast<std::size_t>(active.z) : 0;
        cpu_multithread::enqueueChunks(chunks,
            [details, input, compute, output, rows](const std::size_t chunk) {
                CPUDPP::exec_row(static_cast<int>(chunk / rows),
                                 static_cast<int>(chunk % rows), details,
                                 input, compute, output);
            }, done);
    }
};
#endif // defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)

#if defined(__NVCC__)
template <typename T, int EX, int EY, int KW, int KH>
struct BoxFilterQuadDPP<ParArch::GPU_NVIDIA, T, EX, EY, KW, KH> {
private:
    using SelfType = BoxFilterQuadDPP<
        ParArch::GPU_NVIDIA, T, EX, EY, KW, KH>;
    // The values read, the sums and the means written: float with the
    // channels of T
    using Acc = VectorType_t<float, cn<T>>;
    static constexpr int MAX_RUNTIME_KERNEL_WIDTH = 31;
    static constexpr int MAX_SPAN =
        EX + (KW > 0 ? KW : MAX_RUNTIME_KERNEL_WIDTH) - 1;

public:
    FK_STATIC_STRUCT(BoxFilterQuadDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::GPU_NVIDIA;
    static constexpr int BLOCK_THREADS = 256;

    FK_HOST_DEVICE_FUSE bool accepts(const BoxFilterQuadDetails& details) {
        return details.template validFor<KW, KH>(
            MAX_RUNTIME_KERNEL_WIDTH);
    }

    struct LaunchConfig {
        unsigned int blocks;
        unsigned int threads;
        unsigned int planes;
    };

    // The work items cover the largest image (active.x x active.y), and
    // there is one grid plane per image (active.z)
    FK_HOST_FUSE LaunchConfig launchConfig(const ActiveThreads& active) {
        const int columns = (static_cast<int>(active.x) + EX - 1) / EX;
        const int rows = (static_cast<int>(active.y) + EY - 1) / EY;
        const int workItems = columns * rows;
        return {static_cast<unsigned int>(
                    (workItems + BLOCK_THREADS - 1) / BLOCK_THREADS),
                static_cast<unsigned int>(BLOCK_THREADS), active.z};
    }

    template <typename InIOp, typename ComputeIOps, typename OutIOp>
    FK_DEVICE_STATIC void exec(const BoxFilterQuadDetails& details,
                               const InIOp& input,
                               const ComputeIOps& compute,
                               const OutIOp& output) {
        static_assert(isAnyCompleteReadType<InIOp>,
                      "BoxFilterQuadDPP requires a complete Read IOp");
        static_assert(isAnyWriteType<OutIOp>,
                      "BoxFilterQuadDPP requires a Write IOp");
        static_assert(std::is_same_v<
                          typename InIOp::Operation::OutputType, Acc>,
                      "BoxFilterQuadDPP Read IOp must produce "
                      "VectorType_t<float, cn<T>>");
        static_assert(std::is_same_v<
                          typename OutIOp::Operation::InputType, Acc>,
                      "BoxFilterQuadDPP Write IOp must consume "
                      "VectorType_t<float, cn<T>>");
        static_assert(EX > 0 && EY > 0,
                      "BoxFilterQuadDPP output tile must be positive");
        static_assert(KW >= 0 && KH >= 0,
                      "BoxFilterQuadDPP static kernel sizes cannot be negative");
        static_assert((KW == 0) == (KH == 0),
                      "BoxFilterQuadDPP kernel dimensions are both static or both runtime");
#if defined(__CUDA_ARCH__)
        if (!accepts(details)) return;
        // Image z = blockIdx.z, at its own size
        const int z = static_cast<int>(blockIdx.z);
        const int width = neighborhood_detail::widthOf(input, z);
        const int height = neighborhood_detail::heightOf(input, z);
        if (width <= 0 || height <= 0) return;

        const int kernelWidth = KW > 0 ? KW : details.kernelWidth;
        const int kernelHeight = KH > 0 ? KH : details.kernelHeight;
        const auto& combine = get<0>(compute);
        const auto& subtract = get<1>(compute);
        const auto& normalize = get<2>(compute);

        const int columns = (width + EX - 1) / EX;
        const int workItem = static_cast<int>(
            blockIdx.x * blockDim.x + threadIdx.x);
        const int x0 = (workItem % columns) * EX;
        const int y0 = (workItem / columns) * EY;
        if (x0 >= width || y0 >= height) return;

        const int span = EX + kernelWidth - 1;
        if (span > MAX_SPAN) return;
        const int firstColumn = x0 - details.anchorX;

        auto clamp = [](const int value, const int upper) {
            return value < 0 ? 0 : (value >= upper ? upper - 1 : value);
        };
        auto readSource = [&](const int x, const int y) {
            return InIOp::Operation::exec(
                Point{clamp(x, width),
                      clamp(y, height), z}, input);
        };

        Acc columnsSum[MAX_SPAN];
        #pragma unroll
        for (int index = 0; index < span; ++index) {
            Acc sum = make_set<Acc>(0.f);
            for (int ky = 0; ky < kernelHeight; ++ky) {
                sum = make_tuple(
                    sum, readSource(firstColumn + index,
                                    y0 + ky - details.anchorY)) | combine;
            }
            columnsSum[index] = sum;
        }

        #pragma unroll
        for (int localY = 0; localY < EY; ++localY) {
            const int y = y0 + localY;
            if (y >= height) break;
            #pragma unroll
            for (int localX = 0; localX < EX; ++localX) {
                const int x = x0 + localX;
                if (x >= width) break;
                Acc sum = make_set<Acc>(0.f);
                for (int kx = 0; kx < kernelWidth; ++kx) {
                    sum = make_tuple(
                        sum, columnsSum[localX + kx]) | combine;
                }
                const Acc mean = sum | normalize;
                OutIOp::Operation::exec(Point{x, y, z}, mean, output);
            }

            if (localY + 1 < EY && y + 1 < height) {
                const int top = y - details.anchorY;
                const int bottom = top + kernelHeight;
                #pragma unroll
                for (int index = 0; index < span; ++index) {
                    const Acc withBottom = make_tuple(
                        columnsSum[index],
                        readSource(firstColumn + index, bottom)) | combine;
                    columnsSum[index] = make_tuple(
                        withBottom,
                        readSource(firstColumn + index, top)) | subtract;
                }
            }
        }
#endif // defined(__CUDA_ARCH__)
    }
};

#endif // defined(__NVCC__)

#if defined(FK_HIP_BACKEND)
// The GPU_NVIDIA version above, for HIP (ParArch::GPU_AMD), with the HIP runtime API
template <typename T, int EX, int EY, int KW, int KH>
struct BoxFilterQuadDPP<ParArch::GPU_AMD, T, EX, EY, KW, KH> {
private:
    using SelfType = BoxFilterQuadDPP<
        ParArch::GPU_AMD, T, EX, EY, KW, KH>;
    // The values read, the sums and the means written: float with the
    // channels of T
    using Acc = VectorType_t<float, cn<T>>;
    static constexpr int MAX_RUNTIME_KERNEL_WIDTH = 31;
    static constexpr int MAX_SPAN =
        EX + (KW > 0 ? KW : MAX_RUNTIME_KERNEL_WIDTH) - 1;

public:
    FK_STATIC_STRUCT(BoxFilterQuadDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::GPU_AMD;
    static constexpr int BLOCK_THREADS = 256;

    FK_HOST_DEVICE_FUSE bool accepts(const BoxFilterQuadDetails& details) {
        return details.template validFor<KW, KH>(
            MAX_RUNTIME_KERNEL_WIDTH);
    }

    struct LaunchConfig {
        unsigned int blocks;
        unsigned int threads;
        unsigned int planes;
    };

    // The work items cover the largest image (active.x x active.y), and
    // there is one grid plane per image (active.z)
    FK_HOST_FUSE LaunchConfig launchConfig(const ActiveThreads& active) {
        const int columns = (static_cast<int>(active.x) + EX - 1) / EX;
        const int rows = (static_cast<int>(active.y) + EY - 1) / EY;
        const int workItems = columns * rows;
        return {static_cast<unsigned int>(
                    (workItems + BLOCK_THREADS - 1) / BLOCK_THREADS),
                static_cast<unsigned int>(BLOCK_THREADS), active.z};
    }

    template <typename InIOp, typename ComputeIOps, typename OutIOp>
    FK_DEVICE_STATIC void exec(const BoxFilterQuadDetails& details,
                               const InIOp& input,
                               const ComputeIOps& compute,
                               const OutIOp& output) {
        static_assert(isAnyCompleteReadType<InIOp>,
                      "BoxFilterQuadDPP requires a complete Read IOp");
        static_assert(isAnyWriteType<OutIOp>,
                      "BoxFilterQuadDPP requires a Write IOp");
        static_assert(std::is_same_v<
                          typename InIOp::Operation::OutputType, Acc>,
                      "BoxFilterQuadDPP Read IOp must produce "
                      "VectorType_t<float, cn<T>>");
        static_assert(std::is_same_v<
                          typename OutIOp::Operation::InputType, Acc>,
                      "BoxFilterQuadDPP Write IOp must consume "
                      "VectorType_t<float, cn<T>>");
        static_assert(EX > 0 && EY > 0,
                      "BoxFilterQuadDPP output tile must be positive");
        static_assert(KW >= 0 && KH >= 0,
                      "BoxFilterQuadDPP static kernel sizes cannot be negative");
        static_assert((KW == 0) == (KH == 0),
                      "BoxFilterQuadDPP kernel dimensions are both static or both runtime");
#if defined(__HIP_DEVICE_COMPILE__)
        if (!accepts(details)) return;
        // Image z = blockIdx.z, at its own size
        const int z = static_cast<int>(blockIdx.z);
        const int width = neighborhood_detail::widthOf(input, z);
        const int height = neighborhood_detail::heightOf(input, z);
        if (width <= 0 || height <= 0) return;

        const int kernelWidth = KW > 0 ? KW : details.kernelWidth;
        const int kernelHeight = KH > 0 ? KH : details.kernelHeight;
        const auto& combine = get<0>(compute);
        const auto& subtract = get<1>(compute);
        const auto& normalize = get<2>(compute);

        const int columns = (width + EX - 1) / EX;
        const int workItem = static_cast<int>(
            blockIdx.x * blockDim.x + threadIdx.x);
        const int x0 = (workItem % columns) * EX;
        const int y0 = (workItem / columns) * EY;
        if (x0 >= width || y0 >= height) return;

        const int span = EX + kernelWidth - 1;
        if (span > MAX_SPAN) return;
        const int firstColumn = x0 - details.anchorX;

        auto clamp = [](const int value, const int upper) {
            return value < 0 ? 0 : (value >= upper ? upper - 1 : value);
        };
        auto readSource = [&](const int x, const int y) {
            return InIOp::Operation::exec(
                Point{clamp(x, width),
                      clamp(y, height), z}, input);
        };

        Acc columnsSum[MAX_SPAN];
        #pragma unroll
        for (int index = 0; index < span; ++index) {
            Acc sum = make_set<Acc>(0.f);
            for (int ky = 0; ky < kernelHeight; ++ky) {
                sum = make_tuple(
                    sum, readSource(firstColumn + index,
                                    y0 + ky - details.anchorY)) | combine;
            }
            columnsSum[index] = sum;
        }

        #pragma unroll
        for (int localY = 0; localY < EY; ++localY) {
            const int y = y0 + localY;
            if (y >= height) break;
            #pragma unroll
            for (int localX = 0; localX < EX; ++localX) {
                const int x = x0 + localX;
                if (x >= width) break;
                Acc sum = make_set<Acc>(0.f);
                for (int kx = 0; kx < kernelWidth; ++kx) {
                    sum = make_tuple(
                        sum, columnsSum[localX + kx]) | combine;
                }
                const Acc mean = sum | normalize;
                OutIOp::Operation::exec(Point{x, y, z}, mean, output);
            }

            if (localY + 1 < EY && y + 1 < height) {
                const int top = y - details.anchorY;
                const int bottom = top + kernelHeight;
                #pragma unroll
                for (int index = 0; index < span; ++index) {
                    const Acc withBottom = make_tuple(
                        columnsSum[index],
                        readSource(firstColumn + index, bottom)) | combine;
                    columnsSum[index] = make_tuple(
                        withBottom,
                        readSource(firstColumn + index, top)) | subtract;
                }
            }
        }
#endif // defined(__HIP_DEVICE_COMPILE__)
    }
};

#endif // defined(FK_HIP_BACKEND)

namespace box_filter_quad_detail {
// The checks of the executors: details the DPP accepts, a non empty read and a write with as many planes as the read
// has images (neighborhood_detail::canRunIOps)
template <typename DPP, typename... IOps>
FK_HOST_FUSE void checkArguments(const BoxFilterQuadDetails& details, const IOps&... iOps) {
    executor_detail::requireArguments(DPP::accepts(details), "BoxFilterQuadDPP",
        "details it does not accept (the kernel size, or its maximum at run time)");
    executor_detail::requireArguments(neighborhood_detail::canRunIOps(iOps...), "BoxFilterQuadDPP",
        "the input read is empty, or the write's planes differ from the read's images");
}
} // namespace box_filter_quad_detail

// The executors of BoxFilterQuadDPP, one per back-end:
//   executeOperations<BoxFilterQuadDPP<PA, T, EX, EY, KW, KH>>(stream, details, input, ..., output)
// Details the DPP does not accept, an empty input read, or a write whose planes differ from the read's images throw
// std::invalid_argument before anything is launched or enqueued (Oscar's decision, 2026-11-26).
template <typename T, int EX, int EY, int KW, int KH>
struct Executor<BoxFilterQuadDPP<ParArch::CPU, T, EX, EY, KW, KH>> {
private:
    using DPP = BoxFilterQuadDPP<ParArch::CPU, T, EX, EY, KW, KH>;
    using SelfType = Executor<DPP>;
public:
    FK_STATIC_STRUCT(Executor, SelfType)
    FK_HOST_FUSE ParArch parArch() { return ParArch::CPU; }
    // Computes the filter before it returns
    template <typename... IOps>
    FK_HOST_STATIC void executeOperations(Stream_<ParArch::CPU>&, const BoxFilterQuadDetails& details,
                                          const IOps&... iOps) {
        box_filter_quad_detail::checkArguments<DPP>(details, iOps...);
        DPP::exec(details, iOps...);
    }
};

#if defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)
template <typename T, int EX, int EY, int KW, int KH>
struct Executor<BoxFilterQuadDPP<ParArch::CPU_MULTITHREAD, T, EX, EY, KW, KH>> {
private:
    using DPP = BoxFilterQuadDPP<ParArch::CPU_MULTITHREAD, T, EX, EY, KW, KH>;
    using SelfType = Executor<DPP>;
public:
    FK_STATIC_STRUCT(Executor, SelfType)
    FK_HOST_FUSE ParArch parArch() { return ParArch::CPU_MULTITHREAD; }
    // Enqueues the execution on the stream and returns. The buffers the IOps use must stay valid until
    // it has finished (for instance, until stream.sync()), and the output must not alias the input.
    template <typename... IOps>
    FK_HOST_STATIC void executeOperations(Stream_<ParArch::CPU_MULTITHREAD>& stream, const BoxFilterQuadDetails& details,
                                          const IOps&... iOps) {
        box_filter_quad_detail::checkArguments<DPP>(details, iOps...);
        stream.enqueue([details, iOps...](const cpu_multithread::Completion& done) {
            DPP::exec(done, details, iOps...);
        });
    }
};
#endif // defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)

#if defined(__NVCC__)
template <typename T, int EX, int EY, int KW, int KH>
struct Executor<BoxFilterQuadDPP<ParArch::GPU_NVIDIA, T, EX, EY, KW, KH>> {
private:
    using DPP = BoxFilterQuadDPP<ParArch::GPU_NVIDIA, T, EX, EY, KW, KH>;
    using SelfType = Executor<DPP>;
public:
    FK_STATIC_STRUCT(Executor, SelfType)
    FK_HOST_FUSE ParArch parArch() { return ParArch::GPU_NVIDIA; }
    template <typename... IOps>
    FK_HOST_STATIC void executeOperations(Stream_<ParArch::GPU_NVIDIA>& stream, const BoxFilterQuadDetails& details,
                                          const IOps&... iOps) {
        box_filter_quad_detail::checkArguments<DPP>(details, iOps...);
        const auto launch = DPP::launchConfig(
            neighborhood_detail::activeThreadsOf(neighborhood_detail::firstOf(iOps...)));
        launchDPP_Kernel<DPP><<<dim3(launch.blocks, 1, launch.planes), launch.threads, 0, stream.getCUDAStream()>>>(
            details, iOps...);
        gpuErrchk(cudaGetLastError());
    }
};
#endif // defined(__NVCC__)

#if defined(FK_HIP_BACKEND)
template <typename T, int EX, int EY, int KW, int KH>
struct Executor<BoxFilterQuadDPP<ParArch::GPU_AMD, T, EX, EY, KW, KH>> {
private:
    using DPP = BoxFilterQuadDPP<ParArch::GPU_AMD, T, EX, EY, KW, KH>;
    using SelfType = Executor<DPP>;
public:
    FK_STATIC_STRUCT(Executor, SelfType)
    FK_HOST_FUSE ParArch parArch() { return ParArch::GPU_AMD; }
    template <typename... IOps>
    FK_HOST_STATIC void executeOperations(Stream_<ParArch::GPU_AMD>& stream, const BoxFilterQuadDetails& details,
                                          const IOps&... iOps) {
        box_filter_quad_detail::checkArguments<DPP>(details, iOps...);
        const auto launch = DPP::launchConfig(
            neighborhood_detail::activeThreadsOf(neighborhood_detail::firstOf(iOps...)));
        launchDPP_Kernel<DPP><<<dim3(launch.blocks, 1, launch.planes), launch.threads, 0, stream.getHIPStream()>>>(
            details, iOps...);
        gpuErrchk(hipGetLastError());
    }
};
#endif // defined(FK_HIP_BACKEND)

} // namespace fk

#endif // FK_BOX_FILTER_FAST_H
