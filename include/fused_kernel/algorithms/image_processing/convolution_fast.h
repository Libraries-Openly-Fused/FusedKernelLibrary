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

#ifndef FK_CONVOLUTION_FAST_H
#define FK_CONVOLUTION_FAST_H

#include <fused_kernel/algorithms/image_processing/neighborhood.h>
#include <fused_kernel/core/execution_model/operation_model/operation_model.h>
#include <fused_kernel/core/execution_model/parallel_architectures.h>
#include <fused_kernel/core/execution_model/stream.h>

#include <type_traits>
#if defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)
#include <fused_kernel/core/execution_model/cpu_thread_pool.h>

#include <cstddef>
#endif

namespace fk {

constexpr int FK_CONV_MAX_KERNEL_SIDE = 15;
constexpr int FK_CONV_MAX_COEFFICIENTS =
    FK_CONV_MAX_KERNEL_SIDE * FK_CONV_MAX_KERNEL_SIDE;

// The window and its anchor. The images come from the input read: their
// number is z of its active threads (the batch of a batched read, the
// planes of a 3D read, 1 for a 2D read), and image z is filtered at its own
// size (the read's num_elems at z), read and written at Point{x, y, z}.
// The write must have as many planes as the read has images; otherwise the
// execute functions return false and nothing is written.
struct ConvQuadDetails {
    int kernelWidth;
    int kernelHeight;
    int anchorX;
    int anchorY;
    float coefficients[FK_CONV_MAX_COEFFICIENTS];

    template <int STATIC_KW, int STATIC_KH>
    FK_HOST_DEVICE_CNST bool validFor() const {
        const int effectiveWidth =
            STATIC_KW > 0 ? STATIC_KW : kernelWidth;
        const int effectiveHeight =
            STATIC_KH > 0 ? STATIC_KH : kernelHeight;
        return effectiveWidth > 0 && effectiveHeight > 0 &&
               effectiveWidth <= FK_CONV_MAX_KERNEL_SIDE &&
               effectiveHeight <= FK_CONV_MAX_KERNEL_SIDE &&
               anchorX >= 0 && anchorX < effectiveWidth &&
               anchorY >= 0 && anchorY < effectiveHeight;
    }
};

template <ParArch PA, typename T,
          int EX = 4, int EY = 4, int KW = 0, int KH = 0>
struct ConvQuadDPP;

template <typename T, int EX, int EY, int KW, int KH>
struct ConvQuadDPP<ParArch::CPU, T, EX, EY, KW, KH> {
private:
    using SelfType = ConvQuadDPP<ParArch::CPU, T, EX, EY, KW, KH>;
    // The products, the accumulator and the value written: float with the
    // channels of T. The coefficients are float, one set for every channel.
    using Acc = VectorType_t<float, cn<T>>;

    FK_HOST_FUSE int clamp(const int value, const int upper) {
        return value < 0 ? 0 : (value >= upper ? upper - 1 : value);
    }

public:
    FK_STATIC_STRUCT(ConvQuadDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::CPU;

    FK_HOST_DEVICE_FUSE bool accepts(const ConvQuadDetails& details) {
        return details.template validFor<KW, KH>();
    }

    // Computes the output pixels of row y of image z. Used by exec() and by
    // the CPU_MULTITHREAD specialization (one row of one image per chunk).
    // Expects accepts(details) and y >= 0; a row past the height of image z
    // computes nothing.
    template <typename InIOp, typename ComputeIOps, typename OutIOp>
    FK_HOST_FUSE void exec_row(const int z, const int y,
                               const ConvQuadDetails& details,
                               const InIOp& input,
                               const ComputeIOps& compute,
                               const OutIOp& output) {
        static_assert(isAnyCompleteReadType<InIOp>,
                      "ConvQuadDPP requires a complete Read IOp");
        static_assert(isAnyWriteType<OutIOp>,
                      "ConvQuadDPP requires a Write IOp");
        static_assert(std::is_same_v<
                          typename InIOp::Operation::OutputType, T>,
                      "ConvQuadDPP Read IOp must produce T");
        static_assert(std::is_same_v<
                          typename OutIOp::Operation::InputType, Acc>,
                      "ConvQuadDPP Write IOp must consume "
                      "VectorType_t<float, cn<T>>");
        static_assert(EX > 0 && EY > 0,
                      "ConvQuadDPP output tile must be positive");
        static_assert(KW >= 0 && KH >= 0,
                      "ConvQuadDPP static kernel sizes cannot be negative");
        static_assert((KW == 0) == (KH == 0),
                      "ConvQuadDPP sizes are both static or both runtime");

        const int width = neighborhood_detail::widthOf(input, z);
        const int height = neighborhood_detail::heightOf(input, z);
        if (y >= height) return;
        const int kernelWidth = KW > 0 ? KW : details.kernelWidth;
        const int kernelHeight = KH > 0 ? KH : details.kernelHeight;
        const auto& multiply = get<0>(compute);
        const auto& combine = get<1>(compute);
        auto source = [&](const int x, const int y) {
            return InIOp::Operation::exec(
                Point{clamp(x, width),
                      clamp(y, height), z}, input);
        };

        for (int x = 0; x < width; ++x) {
            Acc value = make_set<Acc>(0.f);
            for (int ky = 0; ky < kernelHeight; ++ky) {
                for (int kx = 0; kx < kernelWidth; ++kx) {
                    const Acc product = make_tuple(
                        details.coefficients[ky * kernelWidth + kx],
                        source(x + kx - details.anchorX,
                               y + ky - details.anchorY)) | multiply;
                    value = make_tuple(value, product) | combine;
                }
            }
            OutIOp::Operation::exec(Point{x, y, z}, value, output);
        }
    }

    template <typename InIOp, typename ComputeIOps, typename OutIOp>
    FK_HOST_FUSE void exec(const ConvQuadDetails& details,
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
// Runs the rows of ConvQuadDPP<ParArch::CPU> on cpuThreadPool(), one chunk per
// row of every image (images x the largest height). As on the GPU, the output
// must not alias the input: a row reads the input rows around it, which other
// chunks may be writing.
template <typename T, int EX, int EY, int KW, int KH>
struct ConvQuadDPP<ParArch::CPU_MULTITHREAD, T, EX, EY, KW, KH> {
private:
    using SelfType = ConvQuadDPP<
        ParArch::CPU_MULTITHREAD, T, EX, EY, KW, KH>;
    using CPUDPP = ConvQuadDPP<ParArch::CPU, T, EX, EY, KW, KH>;

public:
    FK_STATIC_STRUCT(ConvQuadDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::CPU_MULTITHREAD;

    FK_HOST_DEVICE_FUSE bool accepts(const ConvQuadDetails& details) {
        return CPUDPP::accepts(details);
    }

    // Enqueues the rows in the pool and returns; done is called when all of
    // them have finished. Rejected details enqueue no rows, like exec() of
    // the CPU specialization, which returns without writing. The chunks share
    // one copy of details (with its coefficients).
    template <typename InIOp, typename ComputeIOps, typename OutIOp>
    FK_HOST_STATIC void exec(const cpu_multithread::Completion& done,
                             const ConvQuadDetails& details,
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
struct ConvQuadDPP<ParArch::GPU_NVIDIA, T, EX, EY, KW, KH> {
private:
    using SelfType = ConvQuadDPP<
        ParArch::GPU_NVIDIA, T, EX, EY, KW, KH>;
    // The products, the accumulator and the value written: float with the
    // channels of T. The coefficients are float, one set for every channel.
    using Acc = VectorType_t<float, cn<T>>;
    static constexpr int MAX_KERNEL_WIDTH =
        KW > 0 ? KW : FK_CONV_MAX_KERNEL_SIDE;
    static constexpr int MAX_SPAN = EX + MAX_KERNEL_WIDTH - 1;

public:
    FK_STATIC_STRUCT(ConvQuadDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::GPU_NVIDIA;
    static constexpr int BLOCK_THREADS = 256;

    struct LaunchConfig {
        unsigned int blocks;
        unsigned int threads;
        unsigned int planes;
    };

    FK_HOST_DEVICE_FUSE bool accepts(const ConvQuadDetails& details) {
        return details.template validFor<KW, KH>();
    }

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
    FK_DEVICE_STATIC void exec(const ConvQuadDetails& details,
                               const InIOp& input,
                               const ComputeIOps& compute,
                               const OutIOp& output) {
        static_assert(isAnyCompleteReadType<InIOp>,
                      "ConvQuadDPP requires a complete Read IOp");
        static_assert(isAnyWriteType<OutIOp>,
                      "ConvQuadDPP requires a Write IOp");
        static_assert(std::is_same_v<
                          typename InIOp::Operation::OutputType, T>,
                      "ConvQuadDPP Read IOp must produce T");
        static_assert(std::is_same_v<
                          typename OutIOp::Operation::InputType, Acc>,
                      "ConvQuadDPP Write IOp must consume "
                      "VectorType_t<float, cn<T>>");
        static_assert(EX > 0 && EY > 0,
                      "ConvQuadDPP output tile must be positive");
        static_assert(KW >= 0 && KH >= 0,
                      "ConvQuadDPP static kernel sizes cannot be negative");
        static_assert((KW == 0) == (KH == 0),
                      "ConvQuadDPP sizes are both static or both runtime");
#if defined(__CUDA_ARCH__)
        if (!accepts(details)) return;
        // Image z = blockIdx.z, at its own size
        const int z = static_cast<int>(blockIdx.z);
        const int width = neighborhood_detail::widthOf(input, z);
        const int height = neighborhood_detail::heightOf(input, z);
        if (width <= 0 || height <= 0) return;
        const int kernelWidth = KW > 0 ? KW : details.kernelWidth;
        const int kernelHeight = KH > 0 ? KH : details.kernelHeight;
        const auto& multiply = get<0>(compute);
        const auto& combine = get<1>(compute);
        const int columns = (width + EX - 1) / EX;
        const int workItem = static_cast<int>(
            blockIdx.x * blockDim.x + threadIdx.x);
        const int x0 = (workItem % columns) * EX;
        const int y0 = (workItem / columns) * EY;
        if (x0 >= width || y0 >= height) return;

        const int span = EX + kernelWidth - 1;
        if (span > MAX_SPAN) return;
        const int rowsNeeded = EY + kernelHeight - 1;
        const int firstColumn = x0 - details.anchorX;
        auto clamp = [](const int value, const int upper) {
            return value < 0 ? 0 : (value >= upper ? upper - 1 : value);
        };
        auto source = [&](const int x, const int y) {
            return InIOp::Operation::exec(
                Point{clamp(x, width),
                      clamp(y, height), z}, input);
        };

        Acc accumulated[EY][EX];
        #pragma unroll
        for (int localY = 0; localY < EY; ++localY) {
            #pragma unroll
            for (int localX = 0; localX < EX; ++localX) {
                accumulated[localY][localX] = make_set<Acc>(0.f);
            }
        }

        #pragma unroll
        for (int row = 0; row < rowsNeeded; ++row) {
            T sourceRow[MAX_SPAN];
            #pragma unroll
            for (int index = 0; index < span; ++index) {
                sourceRow[index] = source(
                    firstColumn + index,
                    y0 - details.anchorY + row);
            }
            #pragma unroll
            for (int localY = 0; localY < EY; ++localY) {
                const int ky = row - localY;
                if (ky < 0 || ky >= kernelHeight ||
                    y0 + localY >= height) continue;
                #pragma unroll
                for (int kx = 0; kx < kernelWidth; ++kx) {
                    const float coefficient =
                        details.coefficients[ky * kernelWidth + kx];
                    #pragma unroll
                    for (int localX = 0; localX < EX; ++localX) {
                        const Acc product = make_tuple(
                            coefficient,
                            sourceRow[localX + kx]) | multiply;
                        accumulated[localY][localX] = make_tuple(
                            accumulated[localY][localX], product) | combine;
                    }
                }
            }
        }

        #pragma unroll
        for (int localY = 0; localY < EY; ++localY) {
            const int y = y0 + localY;
            if (y >= height) break;
            #pragma unroll
            for (int localX = 0; localX < EX; ++localX) {
                const int x = x0 + localX;
                if (x >= width) break;
                OutIOp::Operation::exec(
                    Point{x, y, z}, accumulated[localY][localX], output);
            }
        }
#endif // defined(__CUDA_ARCH__)
    }
};

template <typename DPP, typename... IOps>
__global__ void launchConvQuadDPP_Kernel(
        const __grid_constant__ ConvQuadDetails details,
        const __grid_constant__ IOps... iOps) {
    DPP::exec(details, iOps...);
}

template <typename DPP, typename... IOps>
FK_HOST_STATIC bool executeConvQuad(
        Stream_<ParArch::GPU_NVIDIA>& stream,
        const ConvQuadDetails& details,
        const IOps&... iOps) {
    static_assert(DPP::PAR_ARCH == ParArch::GPU_NVIDIA,
                  "GPU stream requires the matching ConvQuadDPP specialization");
    if (!DPP::accepts(details) ||
        !neighborhood_detail::canRunIOps(iOps...)) return false;
    const auto launch = DPP::launchConfig(
        neighborhood_detail::activeThreadsOf(neighborhood_detail::firstOf(iOps...)));
    launchConvQuadDPP_Kernel<DPP, IOps...>
        <<<dim3(launch.blocks, 1, launch.planes), launch.threads, 0, stream.getCUDAStream()>>>(
            details, iOps...);
    gpuErrchk(cudaGetLastError());
    return true;
}
#endif // defined(__NVCC__)

#if defined(FK_HIP_BACKEND)
// The GPU_NVIDIA version above, for HIP (ParArch::GPU_AMD), with the HIP runtime API
template <typename T, int EX, int EY, int KW, int KH>
struct ConvQuadDPP<ParArch::GPU_AMD, T, EX, EY, KW, KH> {
private:
    using SelfType = ConvQuadDPP<
        ParArch::GPU_AMD, T, EX, EY, KW, KH>;
    // The products, the accumulator and the value written: float with the
    // channels of T. The coefficients are float, one set for every channel.
    using Acc = VectorType_t<float, cn<T>>;
    static constexpr int MAX_KERNEL_WIDTH =
        KW > 0 ? KW : FK_CONV_MAX_KERNEL_SIDE;
    static constexpr int MAX_SPAN = EX + MAX_KERNEL_WIDTH - 1;

public:
    FK_STATIC_STRUCT(ConvQuadDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::GPU_AMD;
    static constexpr int BLOCK_THREADS = 256;

    struct LaunchConfig {
        unsigned int blocks;
        unsigned int threads;
        unsigned int planes;
    };

    FK_HOST_DEVICE_FUSE bool accepts(const ConvQuadDetails& details) {
        return details.template validFor<KW, KH>();
    }

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
    FK_DEVICE_STATIC void exec(const ConvQuadDetails& details,
                               const InIOp& input,
                               const ComputeIOps& compute,
                               const OutIOp& output) {
        static_assert(isAnyCompleteReadType<InIOp>,
                      "ConvQuadDPP requires a complete Read IOp");
        static_assert(isAnyWriteType<OutIOp>,
                      "ConvQuadDPP requires a Write IOp");
        static_assert(std::is_same_v<
                          typename InIOp::Operation::OutputType, T>,
                      "ConvQuadDPP Read IOp must produce T");
        static_assert(std::is_same_v<
                          typename OutIOp::Operation::InputType, Acc>,
                      "ConvQuadDPP Write IOp must consume "
                      "VectorType_t<float, cn<T>>");
        static_assert(EX > 0 && EY > 0,
                      "ConvQuadDPP output tile must be positive");
        static_assert(KW >= 0 && KH >= 0,
                      "ConvQuadDPP static kernel sizes cannot be negative");
        static_assert((KW == 0) == (KH == 0),
                      "ConvQuadDPP sizes are both static or both runtime");
#if defined(__HIP_DEVICE_COMPILE__)
        if (!accepts(details)) return;
        // Image z = blockIdx.z, at its own size
        const int z = static_cast<int>(blockIdx.z);
        const int width = neighborhood_detail::widthOf(input, z);
        const int height = neighborhood_detail::heightOf(input, z);
        if (width <= 0 || height <= 0) return;
        const int kernelWidth = KW > 0 ? KW : details.kernelWidth;
        const int kernelHeight = KH > 0 ? KH : details.kernelHeight;
        const auto& multiply = get<0>(compute);
        const auto& combine = get<1>(compute);
        const int columns = (width + EX - 1) / EX;
        const int workItem = static_cast<int>(
            blockIdx.x * blockDim.x + threadIdx.x);
        const int x0 = (workItem % columns) * EX;
        const int y0 = (workItem / columns) * EY;
        if (x0 >= width || y0 >= height) return;

        const int span = EX + kernelWidth - 1;
        if (span > MAX_SPAN) return;
        const int rowsNeeded = EY + kernelHeight - 1;
        const int firstColumn = x0 - details.anchorX;
        auto clamp = [](const int value, const int upper) {
            return value < 0 ? 0 : (value >= upper ? upper - 1 : value);
        };
        auto source = [&](const int x, const int y) {
            return InIOp::Operation::exec(
                Point{clamp(x, width),
                      clamp(y, height), z}, input);
        };

        Acc accumulated[EY][EX];
        #pragma unroll
        for (int localY = 0; localY < EY; ++localY) {
            #pragma unroll
            for (int localX = 0; localX < EX; ++localX) {
                accumulated[localY][localX] = make_set<Acc>(0.f);
            }
        }

        #pragma unroll
        for (int row = 0; row < rowsNeeded; ++row) {
            T sourceRow[MAX_SPAN];
            #pragma unroll
            for (int index = 0; index < span; ++index) {
                sourceRow[index] = source(
                    firstColumn + index,
                    y0 - details.anchorY + row);
            }
            #pragma unroll
            for (int localY = 0; localY < EY; ++localY) {
                const int ky = row - localY;
                if (ky < 0 || ky >= kernelHeight ||
                    y0 + localY >= height) continue;
                #pragma unroll
                for (int kx = 0; kx < kernelWidth; ++kx) {
                    const float coefficient =
                        details.coefficients[ky * kernelWidth + kx];
                    #pragma unroll
                    for (int localX = 0; localX < EX; ++localX) {
                        const Acc product = make_tuple(
                            coefficient,
                            sourceRow[localX + kx]) | multiply;
                        accumulated[localY][localX] = make_tuple(
                            accumulated[localY][localX], product) | combine;
                    }
                }
            }
        }

        #pragma unroll
        for (int localY = 0; localY < EY; ++localY) {
            const int y = y0 + localY;
            if (y >= height) break;
            #pragma unroll
            for (int localX = 0; localX < EX; ++localX) {
                const int x = x0 + localX;
                if (x >= width) break;
                OutIOp::Operation::exec(
                    Point{x, y, z}, accumulated[localY][localX], output);
            }
        }
#endif // defined(__HIP_DEVICE_COMPILE__)
    }
};

template <typename DPP, typename... IOps>
__global__ void launchConvQuadDPP_Kernel(
        const __grid_constant__ ConvQuadDetails details,
        const __grid_constant__ IOps... iOps) {
    DPP::exec(details, iOps...);
}

template <typename DPP, typename... IOps>
FK_HOST_STATIC bool executeConvQuad(
        Stream_<ParArch::GPU_AMD>& stream,
        const ConvQuadDetails& details,
        const IOps&... iOps) {
    static_assert(DPP::PAR_ARCH == ParArch::GPU_AMD,
                  "GPU stream requires the matching ConvQuadDPP specialization");
    if (!DPP::accepts(details) ||
        !neighborhood_detail::canRunIOps(iOps...)) return false;
    const auto launch = DPP::launchConfig(
        neighborhood_detail::activeThreadsOf(neighborhood_detail::firstOf(iOps...)));
    launchConvQuadDPP_Kernel<DPP, IOps...>
        <<<dim3(launch.blocks, 1, launch.planes), launch.threads, 0, stream.getHIPStream()>>>(
            details, iOps...);
    gpuErrchk(hipGetLastError());
    return true;
}
#endif // defined(FK_HIP_BACKEND)

template <typename DPP, typename... IOps>
FK_HOST_FUSE bool executeConvQuad(
        Stream_<ParArch::CPU>&,
        const ConvQuadDetails& details,
        const IOps&... iOps) {
    static_assert(DPP::PAR_ARCH == ParArch::CPU,
                  "CPU stream requires the CPU ConvQuadDPP specialization");
    if (!DPP::accepts(details) ||
        !neighborhood_detail::canRunIOps(iOps...)) return false;
    DPP::exec(details, iOps...);
    return true;
}

#if defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)
// Enqueues the execution on the stream and returns. details (with its
// coefficients) is copied; the buffers the IOps use must stay valid until the
// execution has finished (for instance, until stream.sync()), and the output
// must not alias the input.
template <typename DPP, typename... IOps>
FK_HOST_STATIC bool executeConvQuad(
        Stream_<ParArch::CPU_MULTITHREAD>& stream,
        const ConvQuadDetails& details,
        const IOps&... iOps) {
    static_assert(DPP::PAR_ARCH == ParArch::CPU_MULTITHREAD,
                  "CPU_MULTITHREAD stream requires the CPU_MULTITHREAD ConvQuadDPP specialization");
    if (!DPP::accepts(details) ||
        !neighborhood_detail::canRunIOps(iOps...)) return false;
    stream.enqueue([details, iOps...](const cpu_multithread::Completion& done) {
        DPP::exec(done, details, iOps...);
    });
    return true;
}
#endif // defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)

} // namespace fk

#endif // FK_CONVOLUTION_FAST_H
