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

    template <typename ReadIOps, typename MultiplyIOp,
              typename AccumulateIOp, typename WriteIOp>
    FK_HOST_STATIC void exec(const DPPDetails& details,
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
        if (!DPPDetails::valid(details)) return;
        const auto& image = get<0>(reads);
        const auto& coefficients = get<1>(reads);
        for (int oy = 0; oy < details.height; ++oy) {
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
    }
};

#if defined(__NVCC__)
template <typename DPPDetails>
struct LinearFilterDPP<ParArch::GPU_NVIDIA, DPPDetails> {
private:
    using SelfType = LinearFilterDPP<ParArch::GPU_NVIDIA, DPPDetails>;
    using T = typename DPPDetails::ValueType;
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

template <typename DPPDetails, typename ReadIOps,
          typename MultiplyIOp, typename AccumulateIOp, typename WriteIOp>
__global__ void linearFilterDPPKernel(const DPPDetails details,
                                      const ReadIOps reads,
                                      const MultiplyIOp multiply,
                                      const AccumulateIOp accumulate,
                                      const WriteIOp output) {
    LinearFilterDPP<ParArch::GPU_NVIDIA, DPPDetails>::exec(
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
    if (!DPPDetails::valid(details)) return false;
    const dim3 block(DPPDetails::TILE_WIDTH,
                     DPPDetails::TILE_HEIGHT, 1);
    const dim3 grid((details.width + DPPDetails::TILE_WIDTH - 1) /
                        DPPDetails::TILE_WIDTH,
                    (details.height + DPPDetails::TILE_HEIGHT - 1) /
                        DPPDetails::TILE_HEIGHT,
                    1);
    linearFilterDPPKernel<<<grid, block, 0, stream.getCUDAStream()>>>(
        details, reads, multiply, accumulate, output);
    gpuErrchk(cudaGetLastError());
    return true;
}

template <typename DPPDetails, typename ImageRead, typename WriteIOp>
inline bool executeBoxFilter(const DPPDetails& details,
                             const ImageRead& image,
                             const WriteIOp& output,
                             const typename DPPDetails::ValueType scale,
                             Stream_<ParArch::GPU_NVIDIA>& stream) {
    using T = typename DPPDetails::ValueType;
    if (!DPPDetails::valid(details)) return false;
    const T coefficient = scale /
        static_cast<T>(details.kernelWidth * details.kernelHeight);
    const auto coefficientRead = ConstantFilterRead<T>::build(coefficient);
    const auto multiply = Mul<T, T, T, UnaryType>::build();
    const auto accumulate = Add<T, T, T, UnaryType>::build();
    return executeLinearFilter(
        details, make_tuple(image, coefficientRead),
        multiply, accumulate, output, stream);
}
#endif // defined(__NVCC__)

} // namespace fk

#endif // FK_LINEAR_FILTER_DPP_H
