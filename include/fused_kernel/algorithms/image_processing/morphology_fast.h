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

#ifndef FK_MORPHOLOGY_FAST_H
#define FK_MORPHOLOGY_FAST_H

#include <fused_kernel/core/execution_model/operation_model/operation_model.h>
#include <fused_kernel/core/execution_model/parallel_architectures.h>
#include <fused_kernel/core/execution_model/stream.h>
#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/core/utils/vector_utils.h>
#if defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)
#include <fused_kernel/core/execution_model/cpu_thread_pool.h>

#include <cstddef>
#endif

#include <type_traits>

namespace fk {

struct MorphQuadDetails {
    int width;
    int height;
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
        const bool runtimeWidthSupported =
            STATIC_KW > 0 || effectiveWidth <= maxRuntimeKernelWidth;
        return width > 0 && height > 0 &&
               effectiveWidth > 0 && effectiveHeight > 0 &&
               runtimeWidthSupported &&
               anchorX >= 0 && anchorX < effectiveWidth &&
               anchorY >= 0 && anchorY < effectiveHeight;
    }
};

template <ParArch PA, typename T,
          int EX = 4, int EY = 4, int KW = 0, int KH = 0>
struct MorphQuadDPP;

template <typename T, int EX, int EY, int KW, int KH>
struct MorphQuadDPP<ParArch::CPU, T, EX, EY, KW, KH> {
private:
    using SelfType = MorphQuadDPP<ParArch::CPU, T, EX, EY, KW, KH>;

    FK_HOST_FUSE int clamp(const int value, const int upper) {
        return value < 0 ? 0 : (value >= upper ? upper - 1 : value);
    }

public:
    FK_STATIC_STRUCT(MorphQuadDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::CPU;
    static constexpr int MAX_RUNTIME_KERNEL_WIDTH = 31;

    FK_HOST_DEVICE_FUSE bool accepts(const MorphQuadDetails& details) {
        return details.template validFor<KW, KH>(
            MAX_RUNTIME_KERNEL_WIDTH);
    }

    // Computes the output row y, with 0 <= y < details.height. Does nothing
    // unless accepts(details). Also used by
    // MorphQuadDPP<ParArch::CPU_MULTITHREAD>.
    template <typename InIOp, typename ReduceIOp, typename OutIOp>
    FK_HOST_FUSE void exec_row(const int y,
                               const MorphQuadDetails& details,
                               const InIOp& input,
                               const ReduceIOp& reduce,
                               const OutIOp& output) {
        static_assert(isAnyCompleteReadType<InIOp>,
                      "MorphQuadDPP requires a complete Read IOp");
        static_assert(isAnyWriteType<OutIOp>,
                      "MorphQuadDPP requires a Write IOp");
        static_assert(std::is_same_v<
                          typename InIOp::Operation::OutputType, T>,
                      "MorphQuadDPP Read IOp must produce T");
        static_assert(std::is_same_v<
                          typename OutIOp::Operation::InputType, T>,
                      "MorphQuadDPP Write IOp must consume T");
        static_assert(EX > 0 && EY > 0,
                      "MorphQuadDPP output tile must be positive");
        static_assert(KW >= 0 && KH >= 0,
                      "MorphQuadDPP static kernel sizes cannot be negative");
        static_assert((KW == 0) == (KH == 0),
                      "MorphQuadDPP sizes are both static or both runtime");
        static_assert(cn<T> == 1,
                      "MorphQuadDPP currently supports scalar pixel types");
        if (!accepts(details)) return;

        const int kernelWidth = KW > 0 ? KW : details.kernelWidth;
        const int kernelHeight = KH > 0 ? KH : details.kernelHeight;
        auto source = [&](const int x, const int y) {
            return InIOp::Operation::exec(
                Point{clamp(x, details.width),
                      clamp(y, details.height), 0}, input);
        };

        for (int x = 0; x < details.width; ++x) {
            T value = source(x - details.anchorX,
                             y - details.anchorY);
            for (int ky = 0; ky < kernelHeight; ++ky) {
                for (int kx = 0; kx < kernelWidth; ++kx) {
                    if (kx == 0 && ky == 0) continue;
                    value = make_tuple(
                        value,
                        source(x + kx - details.anchorX,
                               y + ky - details.anchorY)) | reduce;
                }
            }
            OutIOp::Operation::exec(Point{x, y, 0}, value, output);
        }
    }

    template <typename InIOp, typename ReduceIOp, typename OutIOp>
    FK_HOST_FUSE void exec(const MorphQuadDetails& details,
                           const InIOp& input,
                           const ReduceIOp& reduce,
                           const OutIOp& output) {
        if (!accepts(details)) return;

        for (int y = 0; y < details.height; ++y) {
            exec_row(y, details, input, reduce, output);
        }
    }
};

#if defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)
// Runs the per row code of MorphQuadDPP<ParArch::CPU>, one chunk per output
// row, on cpuThreadPool(). As with the GPU version, the output must not alias
// the input: rows are written while other rows still read their neighbours.
template <typename T, int EX, int EY, int KW, int KH>
struct MorphQuadDPP<ParArch::CPU_MULTITHREAD, T, EX, EY, KW, KH> {
private:
    using SelfType = MorphQuadDPP<
        ParArch::CPU_MULTITHREAD, T, EX, EY, KW, KH>;
    using CPUDPP = MorphQuadDPP<ParArch::CPU, T, EX, EY, KW, KH>;

public:
    FK_STATIC_STRUCT(MorphQuadDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::CPU_MULTITHREAD;
    static constexpr int MAX_RUNTIME_KERNEL_WIDTH =
        CPUDPP::MAX_RUNTIME_KERNEL_WIDTH;

    FK_HOST_FUSE bool accepts(const MorphQuadDetails& details) {
        return CPUDPP::accepts(details);
    }

    // Enqueues the rows in the pool and returns; done is called when all of
    // them have finished (right away, with no work, unless accepts(details))
    template <typename InIOp, typename ReduceIOp, typename OutIOp>
    FK_HOST_STATIC void exec(const cpu_multithread::Completion& done,
                             const MorphQuadDetails& details,
                             const InIOp& input,
                             const ReduceIOp& reduce,
                             const OutIOp& output) {
        if (!accepts(details)) {
            done();
            return;
        }
        cpu_multithread::enqueueChunks(
            static_cast<std::size_t>(details.height),
            [details, input, reduce, output](const std::size_t row) {
                CPUDPP::exec_row(static_cast<int>(row), details,
                                 input, reduce, output);
            }, done);
    }
};
#endif // defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)

#if defined(__NVCC__) || defined(__HIPCC__)
template <ParArch PA, typename T, int EX, int EY, int KW, int KH>
    requires(PA == ParArch::GPU_NVIDIA || PA == ParArch::GPU_AMD)
struct MorphQuadDPP<PA, T, EX, EY, KW, KH> {
private:
    using SelfType = MorphQuadDPP<
        PA, T, EX, EY, KW, KH>;
    static constexpr int MAX_RUNTIME_KERNEL_WIDTH = 31;
    static constexpr int MAX_KERNEL_WIDTH =
        KW > 0 ? KW : MAX_RUNTIME_KERNEL_WIDTH;
    static constexpr int MAX_SPAN = EX + MAX_KERNEL_WIDTH - 1;

public:
    FK_STATIC_STRUCT(MorphQuadDPP, SelfType)
    static constexpr ParArch PAR_ARCH = PA;
    static constexpr int BLOCK_THREADS = 256;

    struct LaunchConfig {
        unsigned int blocks;
        unsigned int threads;
    };

    FK_HOST_DEVICE_FUSE bool accepts(const MorphQuadDetails& details) {
        return details.template validFor<KW, KH>(
            MAX_RUNTIME_KERNEL_WIDTH);
    }

    FK_HOST_FUSE LaunchConfig launchConfig(
            const MorphQuadDetails& details) {
        const int columns = (details.width + EX - 1) / EX;
        const int rows = (details.height + EY - 1) / EY;
        const int workItems = columns * rows;
        return {static_cast<unsigned int>(
                    (workItems + BLOCK_THREADS - 1) / BLOCK_THREADS),
                static_cast<unsigned int>(BLOCK_THREADS)};
    }

    template <typename InIOp, typename ReduceIOp, typename OutIOp>
    FK_DEVICE_FUSE void exec(const MorphQuadDetails& details,
                             const InIOp& input,
                             const ReduceIOp& reduce,
                             const OutIOp& output) {
        static_assert(isAnyCompleteReadType<InIOp>,
                      "MorphQuadDPP requires a complete Read IOp");
        static_assert(isAnyWriteType<OutIOp>,
                      "MorphQuadDPP requires a Write IOp");
        static_assert(std::is_same_v<
                          typename InIOp::Operation::OutputType, T>,
                      "MorphQuadDPP Read IOp must produce T");
        static_assert(std::is_same_v<
                          typename OutIOp::Operation::InputType, T>,
                      "MorphQuadDPP Write IOp must consume T");
        static_assert(EX > 0 && EY > 0,
                      "MorphQuadDPP output tile must be positive");
        static_assert(KW >= 0 && KH >= 0,
                      "MorphQuadDPP static kernel sizes cannot be negative");
        static_assert((KW == 0) == (KH == 0),
                      "MorphQuadDPP sizes are both static or both runtime");
        static_assert(cn<T> == 1,
                      "MorphQuadDPP currently supports scalar pixel types");
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        if (!accepts(details)) return;
        const int kernelWidth = KW > 0 ? KW : details.kernelWidth;
        const int kernelHeight = KH > 0 ? KH : details.kernelHeight;
        const int columns = (details.width + EX - 1) / EX;
        const int workItem = static_cast<int>(
            blockIdx.x * blockDim.x + threadIdx.x);
        const int x0 = (workItem % columns) * EX;
        const int y0 = (workItem / columns) * EY;
        if (x0 >= details.width || y0 >= details.height) return;

        const int span = EX + kernelWidth - 1;
        if (span > MAX_SPAN) return;
        const int firstColumn = x0 - details.anchorX;
        auto clamp = [](const int value, const int upper) {
            return value < 0 ? 0 : (value >= upper ? upper - 1 : value);
        };
        auto source = [&](const int x, const int y) {
            return InIOp::Operation::exec(
                Point{clamp(x, details.width),
                      clamp(y, details.height), 0}, input);
        };

        #pragma unroll
        for (int localY = 0; localY < EY; ++localY) {
            const int y = y0 + localY;
            if (y >= details.height) break;
            T columnsReduced[MAX_SPAN];
            #pragma unroll
            for (int index = 0; index < span; ++index) {
                T value = source(firstColumn + index,
                                 y - details.anchorY);
                for (int ky = 1; ky < kernelHeight; ++ky) {
                    value = make_tuple(
                        value,
                        source(firstColumn + index,
                               y + ky - details.anchorY)) | reduce;
                }
                columnsReduced[index] = value;
            }

            #pragma unroll
            for (int localX = 0; localX < EX; ++localX) {
                const int x = x0 + localX;
                if (x >= details.width) break;
                T value = columnsReduced[localX];
                for (int kx = 1; kx < kernelWidth; ++kx) {
                    value = make_tuple(
                        value, columnsReduced[localX + kx]) | reduce;
                }
                OutIOp::Operation::exec(Point{x, y, 0}, value, output);
            }
        }
#endif // defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    }
};

#endif // defined(__NVCC__) || defined(__HIPCC__)

namespace morph_quad_detail {
// The check of the executors: details the DPP accepts (DPP::accepts)
template <typename DPP>
FK_HOST_FUSE void checkArguments(const MorphQuadDetails& details) {
    executor_detail::requireArguments(DPP::accepts(details), "MorphQuadDPP", "details it does not accept (DPP::accepts)");
}
} // namespace morph_quad_detail

// The executors of MorphQuadDPP, one per back-end:
//   executeOperations<MorphQuadDPP<PA, T, EX, EY, KW, KH>>(stream, details, input, ..., output)
// Details the DPP does not accept throw std::invalid_argument before anything is launched or enqueued (Oscar's
// decision, 2026-11-26).
template <typename T, int EX, int EY, int KW, int KH>
struct Executor<MorphQuadDPP<ParArch::CPU, T, EX, EY, KW, KH>> {
  private:
    using DPP = MorphQuadDPP<ParArch::CPU, T, EX, EY, KW, KH>;
    using SelfType = Executor<DPP>;

  public:
    FK_STATIC_STRUCT(Executor, SelfType)
    FK_HOST_FUSE ParArch parArch() { return ParArch::CPU; }
    // Computes the filter before it returns
    template <typename... IOps>
    FK_HOST_STATIC void executeOperations(Stream_<ParArch::CPU>&, const MorphQuadDetails& details, const IOps&... iOps) {
        morph_quad_detail::checkArguments<DPP>(details);
        DPP::exec(details, iOps...);
    }
};

#if defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)
template <typename T, int EX, int EY, int KW, int KH>
struct Executor<MorphQuadDPP<ParArch::CPU_MULTITHREAD, T, EX, EY, KW, KH>> {
  private:
    using DPP = MorphQuadDPP<ParArch::CPU_MULTITHREAD, T, EX, EY, KW, KH>;
    using SelfType = Executor<DPP>;

  public:
    FK_STATIC_STRUCT(Executor, SelfType)
    FK_HOST_FUSE ParArch parArch() { return ParArch::CPU_MULTITHREAD; }
    // Enqueues the execution on the stream and returns. The buffers the IOps use must stay valid until it has finished
    // (for instance, until stream.sync()), and the output must not alias the input.
    template <typename... IOps>
    FK_HOST_STATIC void executeOperations(Stream_<ParArch::CPU_MULTITHREAD>& stream, const MorphQuadDetails& details,
                                          const IOps&... iOps) {
        morph_quad_detail::checkArguments<DPP>(details);
        stream.enqueue([details, iOps...](const cpu_multithread::Completion& done) {
            DPP::exec(done, details, iOps...);
        });
    }
};
#endif // defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)

#if defined(__NVCC__) || defined(__HIPCC__)
template <ParArch PA, typename T, int EX, int EY, int KW, int KH>
    requires(PA == ParArch::GPU_NVIDIA || PA == ParArch::GPU_AMD)
struct Executor<MorphQuadDPP<PA, T, EX, EY, KW, KH>> {
  private:
    using DPP = MorphQuadDPP<PA, T, EX, EY, KW, KH>;
    using SelfType = Executor<DPP>;

  public:
    FK_STATIC_STRUCT(Executor, SelfType)
    FK_HOST_FUSE ParArch parArch() { return PA; }
    template <typename... IOps>
    FK_HOST_STATIC void executeOperations(Stream_<PA>& stream, const MorphQuadDetails& details, const IOps&... iOps) {
        morph_quad_detail::checkArguments<DPP>(details);
        const auto launch = DPP::launchConfig(details);
#if defined(__HIPCC__)
        launchDPP_Kernel<DPP><<<launch.blocks, launch.threads, 0, stream.getHIPStream()>>>(details, iOps...);
        gpuErrchk(hipGetLastError());
#else
        launchDPP_Kernel<DPP><<<launch.blocks, launch.threads, 0, stream.getCUDAStream()>>>(details, iOps...);
        gpuErrchk(cudaGetLastError());
#endif
    }
};
#endif // defined(__NVCC__) || defined(__HIPCC__)

} // namespace fk

#endif // FK_MORPHOLOGY_FAST_H
