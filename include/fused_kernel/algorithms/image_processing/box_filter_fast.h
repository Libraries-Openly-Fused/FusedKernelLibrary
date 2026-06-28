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

#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/core/data/point.h>
#include <fused_kernel/core/execution_model/parallel_architectures.h>
#include <fused_kernel/core/execution_model/stream.h>
#include <fused_kernel/core/utils/vector_utils.h>

#include <type_traits>

namespace fk {

struct BoxFilterQuadDetails {
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
        return width > 0 && height > 0 &&
               effectiveWidth > 0 && effectiveHeight > 0 &&
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

    template <typename InIOp, typename ComputeIOps, typename OutIOp>
    FK_HOST_FUSE void exec(const BoxFilterQuadDetails& details,
                           const InIOp& input,
                           const ComputeIOps& compute,
                           const OutIOp& output) {
        static_assert(isAnyCompleteReadType<InIOp>,
                      "BoxFilterQuadDPP requires a complete Read IOp");
        static_assert(isAnyWriteType<OutIOp>,
                      "BoxFilterQuadDPP requires a Write IOp");
        static_assert(std::is_same_v<
                          typename InIOp::Operation::OutputType, float>,
                      "BoxFilterQuadDPP Read IOp must produce float");
        static_assert(std::is_same_v<
                          typename OutIOp::Operation::InputType, float>,
                      "BoxFilterQuadDPP Write IOp must consume float");
        static_assert(EX > 0 && EY > 0,
                      "BoxFilterQuadDPP output tile must be positive");
        static_assert(KW >= 0 && KH >= 0,
                      "BoxFilterQuadDPP static kernel sizes cannot be negative");
        static_assert((KW == 0) == (KH == 0),
                      "BoxFilterQuadDPP kernel dimensions are both static or both runtime");
        static_assert(cn<T> == 1,
                      "BoxFilterQuadDPP currently supports scalar pixel types");
        if (!accepts(details)) return;

        const int kernelWidth = KW > 0 ? KW : details.kernelWidth;
        const int kernelHeight = KH > 0 ? KH : details.kernelHeight;
        const auto& combine = get<0>(compute);
        const auto& normalize = get<2>(compute);

        for (int y = 0; y < details.height; ++y) {
            for (int x = 0; x < details.width; ++x) {
                float sum = 0.f;
                for (int ky = 0; ky < kernelHeight; ++ky) {
                    for (int kx = 0; kx < kernelWidth; ++kx) {
                        const Point source{
                            clamp(x + kx - details.anchorX, details.width),
                            clamp(y + ky - details.anchorY, details.height), 0};
                        const float value =
                            InIOp::Operation::exec(source, input);
                        sum = make_tuple(sum, value) | combine;
                    }
                }
                const float mean = sum | normalize;
                OutIOp::Operation::exec(Point{x, y, 0}, mean, output);
            }
        }
    }
};

#if defined(__NVCC__)
template <typename T, int EX, int EY, int KW, int KH>
struct BoxFilterQuadDPP<ParArch::GPU_NVIDIA, T, EX, EY, KW, KH> {
private:
    using SelfType = BoxFilterQuadDPP<
        ParArch::GPU_NVIDIA, T, EX, EY, KW, KH>;
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
    };

    FK_HOST_FUSE LaunchConfig launchConfig(
            const BoxFilterQuadDetails& details) {
        const int columns = (details.width + EX - 1) / EX;
        const int rows = (details.height + EY - 1) / EY;
        const int workItems = columns * rows;
        return {static_cast<unsigned int>(
                    (workItems + BLOCK_THREADS - 1) / BLOCK_THREADS),
                static_cast<unsigned int>(BLOCK_THREADS)};
    }

    template <typename InIOp, typename ComputeIOps, typename OutIOp>
    FK_DEVICE_FUSE void exec(const BoxFilterQuadDetails& details,
                             const InIOp& input,
                             const ComputeIOps& compute,
                             const OutIOp& output) {
        static_assert(isAnyCompleteReadType<InIOp>,
                      "BoxFilterQuadDPP requires a complete Read IOp");
        static_assert(isAnyWriteType<OutIOp>,
                      "BoxFilterQuadDPP requires a Write IOp");
        static_assert(std::is_same_v<
                          typename InIOp::Operation::OutputType, float>,
                      "BoxFilterQuadDPP Read IOp must produce float");
        static_assert(std::is_same_v<
                          typename OutIOp::Operation::InputType, float>,
                      "BoxFilterQuadDPP Write IOp must consume float");
        static_assert(EX > 0 && EY > 0,
                      "BoxFilterQuadDPP output tile must be positive");
        static_assert(KW >= 0 && KH >= 0,
                      "BoxFilterQuadDPP static kernel sizes cannot be negative");
        static_assert((KW == 0) == (KH == 0),
                      "BoxFilterQuadDPP kernel dimensions are both static or both runtime");
        static_assert(cn<T> == 1,
                      "BoxFilterQuadDPP currently supports scalar pixel types");
#if defined(__CUDA_ARCH__)
        if (!accepts(details)) return;

        const int kernelWidth = KW > 0 ? KW : details.kernelWidth;
        const int kernelHeight = KH > 0 ? KH : details.kernelHeight;
        const auto& combine = get<0>(compute);
        const auto& subtract = get<1>(compute);
        const auto& normalize = get<2>(compute);

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
        auto readSource = [&](const int x, const int y) {
            return InIOp::Operation::exec(
                Point{clamp(x, details.width),
                      clamp(y, details.height), 0}, input);
        };

        float columnsSum[MAX_SPAN];
        #pragma unroll
        for (int index = 0; index < span; ++index) {
            float sum = 0.f;
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
            if (y >= details.height) break;
            #pragma unroll
            for (int localX = 0; localX < EX; ++localX) {
                const int x = x0 + localX;
                if (x >= details.width) break;
                float sum = 0.f;
                for (int kx = 0; kx < kernelWidth; ++kx) {
                    sum = make_tuple(
                        sum, columnsSum[localX + kx]) | combine;
                }
                const float mean = sum | normalize;
                OutIOp::Operation::exec(Point{x, y, 0}, mean, output);
            }

            if (localY + 1 < EY && y + 1 < details.height) {
                const int top = y - details.anchorY;
                const int bottom = top + kernelHeight;
                #pragma unroll
                for (int index = 0; index < span; ++index) {
                    const float withBottom = make_tuple(
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

template <typename DPP, typename... IOps>
__global__ void launchBoxFilterQuadDPP_Kernel(
        const __grid_constant__ BoxFilterQuadDetails details,
        const __grid_constant__ IOps... iOps) {
    DPP::exec(details, iOps...);
}

template <typename DPP, typename... IOps>
FK_HOST_FUSE void executeBoxFilterQuad(
        Stream_<ParArch::GPU_NVIDIA>& stream,
        const BoxFilterQuadDetails& details,
        const IOps&... iOps) {
    static_assert(DPP::PAR_ARCH == ParArch::GPU_NVIDIA,
                  "GPU stream requires the NVIDIA BoxFilterQuadDPP specialization");
    if (!DPP::accepts(details)) return;
    const auto launch = DPP::launchConfig(details);
    launchBoxFilterQuadDPP_Kernel<DPP, IOps...>
        <<<launch.blocks, launch.threads, 0, stream.getCUDAStream()>>>(
            details, iOps...);
    gpuErrchk(cudaGetLastError());
}
#endif // defined(__NVCC__)

template <typename DPP, typename... IOps>
FK_HOST_FUSE void executeBoxFilterQuad(
        Stream_<ParArch::CPU>&,
        const BoxFilterQuadDetails& details,
        const IOps&... iOps) {
    static_assert(DPP::PAR_ARCH == ParArch::CPU,
                  "CPU stream requires the CPU BoxFilterQuadDPP specialization");
    DPP::exec(details, iOps...);
}

} // namespace fk

#endif // FK_BOX_FILTER_FAST_H
