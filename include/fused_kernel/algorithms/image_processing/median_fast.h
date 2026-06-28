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

#ifndef FK_MEDIAN_FAST_H
#define FK_MEDIAN_FAST_H

#include <fused_kernel/core/execution_model/operation_model/operation_model.h>
#include <fused_kernel/core/execution_model/parallel_architectures.h>
#include <fused_kernel/core/execution_model/stream.h>
#include <fused_kernel/core/utils/vector_utils.h>
#include <fused_kernel/core/utils/vlimits.h>

#include <type_traits>

namespace fk {

constexpr int FK_MEDIAN_MAX_KERNEL_SIDE = 7;
constexpr int FK_MEDIAN_SORT_SIZE = 64;

struct MedianQuadDetails {
    int width;
    int height;
    int kernelWidth;
    int kernelHeight;
    int anchorX;
    int anchorY;

    template <int STATIC_KW, int STATIC_KH>
    FK_HOST_DEVICE_CNST bool validFor() const {
        const int effectiveWidth =
            STATIC_KW > 0 ? STATIC_KW : kernelWidth;
        const int effectiveHeight =
            STATIC_KH > 0 ? STATIC_KH : kernelHeight;
        return width > 0 && height > 0 &&
               effectiveWidth > 0 && effectiveHeight > 0 &&
               effectiveWidth <= FK_MEDIAN_MAX_KERNEL_SIDE &&
               effectiveHeight <= FK_MEDIAN_MAX_KERNEL_SIDE &&
               effectiveWidth * effectiveHeight <= FK_MEDIAN_SORT_SIZE &&
               anchorX >= 0 && anchorX < effectiveWidth &&
               anchorY >= 0 && anchorY < effectiveHeight;
    }
};

namespace median_detail {
template <typename T, typename MinIOp, typename MaxIOp>
FK_HOST_DEVICE_FUSE void compareSwap(
        T* values, const int first, const int second,
        const MinIOp& minimum, const MaxIOp& maximum,
        const bool ascending = true) {
    const T left = values[first];
    const T right = values[second];
    const T low = make_tuple(left, right) | minimum;
    const T high = make_tuple(left, right) | maximum;
    values[first] = ascending ? low : high;
    values[second] = ascending ? high : low;
}

template <typename T, typename MinIOp, typename MaxIOp>
FK_HOST_DEVICE_FUSE T medianNet9(
        T* values, const MinIOp& minimum, const MaxIOp& maximum) {
#define FK_MEDIAN_SWAP(a, b) \
    compareSwap(values, a, b, minimum, maximum)
    FK_MEDIAN_SWAP(0,1); FK_MEDIAN_SWAP(2,3); FK_MEDIAN_SWAP(4,5);
    FK_MEDIAN_SWAP(6,7); FK_MEDIAN_SWAP(0,2); FK_MEDIAN_SWAP(1,3);
    FK_MEDIAN_SWAP(4,6); FK_MEDIAN_SWAP(5,7); FK_MEDIAN_SWAP(1,2);
    FK_MEDIAN_SWAP(5,6); FK_MEDIAN_SWAP(0,4); FK_MEDIAN_SWAP(1,5);
    FK_MEDIAN_SWAP(2,6); FK_MEDIAN_SWAP(3,7); FK_MEDIAN_SWAP(2,4);
    FK_MEDIAN_SWAP(3,5); FK_MEDIAN_SWAP(1,2); FK_MEDIAN_SWAP(3,4);
    FK_MEDIAN_SWAP(5,6); FK_MEDIAN_SWAP(0,8); FK_MEDIAN_SWAP(4,8);
    FK_MEDIAN_SWAP(2,4); FK_MEDIAN_SWAP(3,5); FK_MEDIAN_SWAP(6,8);
    FK_MEDIAN_SWAP(1,2); FK_MEDIAN_SWAP(3,4); FK_MEDIAN_SWAP(5,6);
    FK_MEDIAN_SWAP(7,8);
#undef FK_MEDIAN_SWAP
    return values[4];
}

template <typename T, typename MinIOp, typename MaxIOp>
FK_HOST_DEVICE_FUSE T medianNet25(
        T* values, const MinIOp& minimum, const MaxIOp& maximum) {
#define FK_MEDIAN_SWAP(a, b) \
    compareSwap(values, a, b, minimum, maximum)
    FK_MEDIAN_SWAP(0,1); FK_MEDIAN_SWAP(2,3); FK_MEDIAN_SWAP(4,5);
    FK_MEDIAN_SWAP(6,7); FK_MEDIAN_SWAP(8,9); FK_MEDIAN_SWAP(10,11);
    FK_MEDIAN_SWAP(12,13); FK_MEDIAN_SWAP(14,15); FK_MEDIAN_SWAP(16,17);
    FK_MEDIAN_SWAP(18,19); FK_MEDIAN_SWAP(20,21); FK_MEDIAN_SWAP(22,23);
    FK_MEDIAN_SWAP(0,2); FK_MEDIAN_SWAP(1,3); FK_MEDIAN_SWAP(4,6);
    FK_MEDIAN_SWAP(5,7); FK_MEDIAN_SWAP(8,10); FK_MEDIAN_SWAP(9,11);
    FK_MEDIAN_SWAP(12,14); FK_MEDIAN_SWAP(13,15); FK_MEDIAN_SWAP(16,18);
    FK_MEDIAN_SWAP(17,19); FK_MEDIAN_SWAP(20,22); FK_MEDIAN_SWAP(21,23);
    FK_MEDIAN_SWAP(1,2); FK_MEDIAN_SWAP(5,6); FK_MEDIAN_SWAP(9,10);
    FK_MEDIAN_SWAP(13,14); FK_MEDIAN_SWAP(17,18); FK_MEDIAN_SWAP(21,22);
    FK_MEDIAN_SWAP(0,4); FK_MEDIAN_SWAP(1,5); FK_MEDIAN_SWAP(2,6);
    FK_MEDIAN_SWAP(3,7); FK_MEDIAN_SWAP(8,12); FK_MEDIAN_SWAP(9,13);
    FK_MEDIAN_SWAP(10,14); FK_MEDIAN_SWAP(11,15); FK_MEDIAN_SWAP(16,20);
    FK_MEDIAN_SWAP(17,21); FK_MEDIAN_SWAP(18,22); FK_MEDIAN_SWAP(19,23);
    FK_MEDIAN_SWAP(2,4); FK_MEDIAN_SWAP(3,5); FK_MEDIAN_SWAP(10,12);
    FK_MEDIAN_SWAP(11,13); FK_MEDIAN_SWAP(18,20); FK_MEDIAN_SWAP(19,21);
    FK_MEDIAN_SWAP(1,2); FK_MEDIAN_SWAP(3,4); FK_MEDIAN_SWAP(5,6);
    FK_MEDIAN_SWAP(9,10); FK_MEDIAN_SWAP(11,12); FK_MEDIAN_SWAP(13,14);
    FK_MEDIAN_SWAP(17,18); FK_MEDIAN_SWAP(19,20); FK_MEDIAN_SWAP(21,22);
    FK_MEDIAN_SWAP(0,8); FK_MEDIAN_SWAP(1,9); FK_MEDIAN_SWAP(2,10);
    FK_MEDIAN_SWAP(3,11); FK_MEDIAN_SWAP(4,12); FK_MEDIAN_SWAP(5,13);
    FK_MEDIAN_SWAP(6,14); FK_MEDIAN_SWAP(7,15); FK_MEDIAN_SWAP(16,24);
    FK_MEDIAN_SWAP(4,8); FK_MEDIAN_SWAP(5,9); FK_MEDIAN_SWAP(6,10);
    FK_MEDIAN_SWAP(7,11); FK_MEDIAN_SWAP(20,24); FK_MEDIAN_SWAP(2,4);
    FK_MEDIAN_SWAP(3,5); FK_MEDIAN_SWAP(6,8); FK_MEDIAN_SWAP(7,9);
    FK_MEDIAN_SWAP(10,12); FK_MEDIAN_SWAP(11,13); FK_MEDIAN_SWAP(18,20);
    FK_MEDIAN_SWAP(19,21); FK_MEDIAN_SWAP(22,24); FK_MEDIAN_SWAP(1,2);
    FK_MEDIAN_SWAP(3,4); FK_MEDIAN_SWAP(5,6); FK_MEDIAN_SWAP(7,8);
    FK_MEDIAN_SWAP(9,10); FK_MEDIAN_SWAP(11,12); FK_MEDIAN_SWAP(13,14);
    FK_MEDIAN_SWAP(17,18); FK_MEDIAN_SWAP(19,20); FK_MEDIAN_SWAP(21,22);
    FK_MEDIAN_SWAP(23,24); FK_MEDIAN_SWAP(0,16); FK_MEDIAN_SWAP(1,17);
    FK_MEDIAN_SWAP(2,18); FK_MEDIAN_SWAP(3,19); FK_MEDIAN_SWAP(4,20);
    FK_MEDIAN_SWAP(5,21); FK_MEDIAN_SWAP(6,22); FK_MEDIAN_SWAP(7,23);
    FK_MEDIAN_SWAP(8,24); FK_MEDIAN_SWAP(8,16); FK_MEDIAN_SWAP(9,17);
    FK_MEDIAN_SWAP(10,18); FK_MEDIAN_SWAP(11,19); FK_MEDIAN_SWAP(12,20);
    FK_MEDIAN_SWAP(13,21); FK_MEDIAN_SWAP(14,22); FK_MEDIAN_SWAP(15,23);
    FK_MEDIAN_SWAP(4,8); FK_MEDIAN_SWAP(5,9); FK_MEDIAN_SWAP(6,10);
    FK_MEDIAN_SWAP(7,11); FK_MEDIAN_SWAP(12,16); FK_MEDIAN_SWAP(13,17);
    FK_MEDIAN_SWAP(14,18); FK_MEDIAN_SWAP(15,19); FK_MEDIAN_SWAP(20,24);
    FK_MEDIAN_SWAP(2,4); FK_MEDIAN_SWAP(3,5); FK_MEDIAN_SWAP(6,8);
    FK_MEDIAN_SWAP(7,9); FK_MEDIAN_SWAP(10,12); FK_MEDIAN_SWAP(11,13);
    FK_MEDIAN_SWAP(14,16); FK_MEDIAN_SWAP(15,17); FK_MEDIAN_SWAP(18,20);
    FK_MEDIAN_SWAP(19,21); FK_MEDIAN_SWAP(22,24); FK_MEDIAN_SWAP(1,2);
    FK_MEDIAN_SWAP(3,4); FK_MEDIAN_SWAP(5,6); FK_MEDIAN_SWAP(7,8);
    FK_MEDIAN_SWAP(9,10); FK_MEDIAN_SWAP(11,12); FK_MEDIAN_SWAP(13,14);
    FK_MEDIAN_SWAP(15,16); FK_MEDIAN_SWAP(17,18); FK_MEDIAN_SWAP(19,20);
    FK_MEDIAN_SWAP(21,22); FK_MEDIAN_SWAP(23,24);
#undef FK_MEDIAN_SWAP
    return values[12];
}

template <typename T, typename MinIOp, typename MaxIOp>
FK_HOST_DEVICE_FUSE T medianBitonic64(
        T* values, const int count,
        const MinIOp& minimum, const MaxIOp& maximum) {
    for (int index = count; index < FK_MEDIAN_SORT_SIZE; ++index) {
        values[index] = maxValue<T>;
    }
    for (int size = 2; size <= FK_MEDIAN_SORT_SIZE; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int index = 0; index < FK_MEDIAN_SORT_SIZE; ++index) {
                const int partner = index ^ stride;
                if (partner > index) {
                    compareSwap(values, index, partner, minimum, maximum,
                                (index & size) == 0);
                }
            }
        }
    }
    return values[count / 2];
}

template <typename T, typename ComputeIOps>
FK_HOST_DEVICE_FUSE T median(
        T* values, const int count, const int kernelWidth,
        const int kernelHeight, const ComputeIOps& compute) {
    const auto& minimum = get<0>(compute);
    const auto& maximum = get<1>(compute);
    if (kernelWidth == 3 && kernelHeight == 3) {
        return medianNet9(values, minimum, maximum);
    }
    if (kernelWidth == 5 && kernelHeight == 5) {
        return medianNet25(values, minimum, maximum);
    }
    return medianBitonic64(values, count, minimum, maximum);
}
} // namespace median_detail

template <ParArch PA, typename T,
          int EX = 4, int EY = 4, int KW = 0, int KH = 0>
struct MedianQuadDPP;

template <typename T, int EX, int EY, int KW, int KH>
struct MedianQuadDPP<ParArch::CPU, T, EX, EY, KW, KH> {
private:
    using SelfType = MedianQuadDPP<ParArch::CPU, T, EX, EY, KW, KH>;

    FK_HOST_FUSE int clamp(const int value, const int upper) {
        return value < 0 ? 0 : (value >= upper ? upper - 1 : value);
    }

public:
    FK_STATIC_STRUCT(MedianQuadDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::CPU;

    FK_HOST_DEVICE_FUSE bool accepts(const MedianQuadDetails& details) {
        return details.template validFor<KW, KH>();
    }

    template <typename InIOp, typename ComputeIOps, typename OutIOp>
    FK_HOST_FUSE void exec(const MedianQuadDetails& details,
                           const InIOp& input,
                           const ComputeIOps& compute,
                           const OutIOp& output) {
        static_assert(isAnyCompleteReadType<InIOp>,
                      "MedianQuadDPP requires a complete Read IOp");
        static_assert(isAnyWriteType<OutIOp>,
                      "MedianQuadDPP requires a Write IOp");
        static_assert(std::is_same_v<
                          typename InIOp::Operation::OutputType, T>,
                      "MedianQuadDPP Read IOp must produce T");
        static_assert(std::is_same_v<
                          typename OutIOp::Operation::InputType, T>,
                      "MedianQuadDPP Write IOp must consume T");
        static_assert(EX > 0 && EY > 0,
                      "MedianQuadDPP output tile must be positive");
        static_assert(KW >= 0 && KH >= 0,
                      "MedianQuadDPP static kernel sizes cannot be negative");
        static_assert((KW == 0) == (KH == 0),
                      "MedianQuadDPP sizes are both static or both runtime");
        static_assert(cn<T> == 1,
                      "MedianQuadDPP currently supports scalar pixel types");
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
                T values[FK_MEDIAN_SORT_SIZE];
                int count = 0;
                for (int ky = 0; ky < kernelHeight; ++ky) {
                    for (int kx = 0; kx < kernelWidth; ++kx) {
                        values[count++] = source(
                            x + kx - details.anchorX,
                            y + ky - details.anchorY);
                    }
                }
                const T value = median_detail::median(
                    values, count, kernelWidth, kernelHeight, compute);
                OutIOp::Operation::exec(Point{x, y, 0}, value, output);
            }
        }
    }
};

#if defined(__NVCC__)
template <typename T, int EX, int EY, int KW, int KH>
struct MedianQuadDPP<ParArch::GPU_NVIDIA, T, EX, EY, KW, KH> {
private:
    using SelfType = MedianQuadDPP<
        ParArch::GPU_NVIDIA, T, EX, EY, KW, KH>;

public:
    FK_STATIC_STRUCT(MedianQuadDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::GPU_NVIDIA;
    static constexpr int BLOCK_THREADS = 256;

    struct LaunchConfig {
        unsigned int blocks;
        unsigned int threads;
    };

    FK_HOST_DEVICE_FUSE bool accepts(const MedianQuadDetails& details) {
        return details.template validFor<KW, KH>();
    }

    FK_HOST_FUSE LaunchConfig launchConfig(const MedianQuadDetails& details) {
        const int columns = (details.width + EX - 1) / EX;
        const int rows = (details.height + EY - 1) / EY;
        const int workItems = columns * rows;
        return {static_cast<unsigned int>(
                    (workItems + BLOCK_THREADS - 1) / BLOCK_THREADS),
                static_cast<unsigned int>(BLOCK_THREADS)};
    }

    template <typename InIOp, typename ComputeIOps, typename OutIOp>
    FK_DEVICE_FUSE void exec(const MedianQuadDetails& details,
                             const InIOp& input,
                             const ComputeIOps& compute,
                             const OutIOp& output) {
        static_assert(isAnyCompleteReadType<InIOp>,
                      "MedianQuadDPP requires a complete Read IOp");
        static_assert(isAnyWriteType<OutIOp>,
                      "MedianQuadDPP requires a Write IOp");
        static_assert(std::is_same_v<
                          typename InIOp::Operation::OutputType, T>,
                      "MedianQuadDPP Read IOp must produce T");
        static_assert(std::is_same_v<
                          typename OutIOp::Operation::InputType, T>,
                      "MedianQuadDPP Write IOp must consume T");
        static_assert(EX > 0 && EY > 0,
                      "MedianQuadDPP output tile must be positive");
        static_assert(KW >= 0 && KH >= 0,
                      "MedianQuadDPP static kernel sizes cannot be negative");
        static_assert((KW == 0) == (KH == 0),
                      "MedianQuadDPP sizes are both static or both runtime");
        static_assert(cn<T> == 1,
                      "MedianQuadDPP currently supports scalar pixel types");
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
            #pragma unroll
            for (int localX = 0; localX < EX; ++localX) {
                const int x = x0 + localX;
                if (x >= details.width) break;
                T values[FK_MEDIAN_SORT_SIZE];
                int count = 0;
                for (int ky = 0; ky < kernelHeight; ++ky) {
                    for (int kx = 0; kx < kernelWidth; ++kx) {
                        values[count++] = source(
                            x + kx - details.anchorX,
                            y + ky - details.anchorY);
                    }
                }
                const T value = median_detail::median(
                    values, count, kernelWidth, kernelHeight, compute);
                OutIOp::Operation::exec(Point{x, y, 0}, value, output);
            }
        }
#endif // defined(__CUDA_ARCH__)
    }
};

template <typename DPP, typename... IOps>
__global__ void launchMedianQuadDPP_Kernel(
        const __grid_constant__ MedianQuadDetails details,
        const __grid_constant__ IOps... iOps) {
    DPP::exec(details, iOps...);
}

template <typename DPP, typename... IOps>
FK_HOST_FUSE void executeMedianQuad(
        Stream_<ParArch::GPU_NVIDIA>& stream,
        const MedianQuadDetails& details,
        const IOps&... iOps) {
    static_assert(DPP::PAR_ARCH == ParArch::GPU_NVIDIA,
                  "GPU stream requires the NVIDIA MedianQuadDPP specialization");
    if (!DPP::accepts(details)) return;
    const auto launch = DPP::launchConfig(details);
    launchMedianQuadDPP_Kernel<DPP, IOps...>
        <<<launch.blocks, launch.threads, 0, stream.getCUDAStream()>>>(
            details, iOps...);
    gpuErrchk(cudaGetLastError());
}
#endif // defined(__NVCC__)

template <typename DPP, typename... IOps>
FK_HOST_FUSE void executeMedianQuad(
        Stream_<ParArch::CPU>&,
        const MedianQuadDetails& details,
        const IOps&... iOps) {
    static_assert(DPP::PAR_ARCH == ParArch::CPU,
                  "CPU stream requires the CPU MedianQuadDPP specialization");
    DPP::exec(details, iOps...);
}

} // namespace fk

#endif // FK_MEDIAN_FAST_H
