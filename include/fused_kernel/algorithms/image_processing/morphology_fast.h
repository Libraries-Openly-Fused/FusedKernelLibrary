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
#include <fused_kernel/core/utils/vector_utils.h>

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

    template <typename InIOp, typename ReduceIOp, typename OutIOp>
    FK_HOST_FUSE void exec(const MorphQuadDetails& details,
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

        for (int y = 0; y < details.height; ++y) {
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
    }
};

#if defined(__NVCC__)
template <typename T, int EX, int EY, int KW, int KH>
struct MorphQuadDPP<ParArch::GPU_NVIDIA, T, EX, EY, KW, KH> {
private:
    using SelfType = MorphQuadDPP<
        ParArch::GPU_NVIDIA, T, EX, EY, KW, KH>;
    static constexpr int MAX_RUNTIME_KERNEL_WIDTH = 31;
    static constexpr int MAX_KERNEL_WIDTH =
        KW > 0 ? KW : MAX_RUNTIME_KERNEL_WIDTH;
    static constexpr int MAX_SPAN = EX + MAX_KERNEL_WIDTH - 1;

public:
    FK_STATIC_STRUCT(MorphQuadDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::GPU_NVIDIA;
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
#if defined(__CUDA_ARCH__)
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
#endif // defined(__CUDA_ARCH__)
    }
};

template <typename DPP, typename... IOps>
__global__ void launchMorphQuadDPP_Kernel(
        const __grid_constant__ MorphQuadDetails details,
        const __grid_constant__ IOps... iOps) {
    DPP::exec(details, iOps...);
}

template <typename DPP, typename... IOps>
FK_HOST_FUSE void executeMorphQuad(
        Stream_<ParArch::GPU_NVIDIA>& stream,
        const MorphQuadDetails& details,
        const IOps&... iOps) {
    static_assert(DPP::PAR_ARCH == ParArch::GPU_NVIDIA,
                  "GPU stream requires the NVIDIA MorphQuadDPP specialization");
    if (!DPP::accepts(details)) return;
    const auto launch = DPP::launchConfig(details);
    launchMorphQuadDPP_Kernel<DPP, IOps...>
        <<<launch.blocks, launch.threads, 0, stream.getCUDAStream()>>>(
            details, iOps...);
    gpuErrchk(cudaGetLastError());
}
#endif // defined(__NVCC__)

template <typename DPP, typename... IOps>
FK_HOST_FUSE void executeMorphQuad(
        Stream_<ParArch::CPU>&,
        const MorphQuadDetails& details,
        const IOps&... iOps) {
    static_assert(DPP::PAR_ARCH == ParArch::CPU,
                  "CPU stream requires the CPU MorphQuadDPP specialization");
    DPP::exec(details, iOps...);
}

} // namespace fk

#endif // FK_MORPHOLOGY_FAST_H
