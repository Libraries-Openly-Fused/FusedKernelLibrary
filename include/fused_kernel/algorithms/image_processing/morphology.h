/* Copyright 2023-2026 Oscar Amoros Huguet, Johnny Nunez

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_MORPHOLOGY
#define FK_MORPHOLOGY

#include <fused_kernel/algorithms/basic_ops/logical.h>
#include <fused_kernel/core/data/tuple.h>
#include <fused_kernel/core/execution_model/operation_model/operation_model.h>
#include <fused_kernel/core/execution_model/stream.h>
#include <fused_kernel/core/utils/utils.h>

#include <type_traits>
#include <utility>
#include <vector>

namespace fk {

template <typename T, int TILE_W = 16, int TILE_H = 8,
          int MAX_MASK_W = 7, int MAX_MASK_H = 7>
struct MorphologyDPPDetails {
    using ValueType = T;
    static constexpr int TILE_WIDTH = TILE_W;
    static constexpr int TILE_HEIGHT = TILE_H;
    static constexpr int MAX_MASK_WIDTH = MAX_MASK_W;
    static constexpr int MAX_MASK_HEIGHT = MAX_MASK_H;
    static constexpr int MAX_PASSES = 4;
    static constexpr int MAX_STAGE_WIDTH =
        TILE_W + MAX_PASSES * (MAX_MASK_W - 1);
    static constexpr int MAX_STAGE_HEIGHT =
        TILE_H + MAX_PASSES * (MAX_MASK_H - 1);
    static constexpr int MAX_STAGE_ELEMENTS =
        MAX_STAGE_WIDTH * MAX_STAGE_HEIGHT;

    static_assert(TILE_W > 0 && TILE_H > 0,
                  "Morphology tile dimensions must be positive");
    static_assert(MAX_MASK_W > 0 && MAX_MASK_H > 0,
                  "Morphology maximum mask dimensions must be positive");

    int width;
    int height;
    int maskW;
    int maskH;
    int anchorX;
    int anchorY;

    FK_HOST_DEVICE_FUSE bool valid(const MorphologyDPPDetails& details) {
        return details.width > 0 && details.height > 0 &&
               details.maskW > 0 && details.maskH > 0 &&
               details.maskW <= MAX_MASK_W &&
               details.maskH <= MAX_MASK_H &&
               details.anchorX >= 0 && details.anchorX < details.maskW &&
               details.anchorY >= 0 && details.anchorY < details.maskH;
    }
};

namespace morphology_detail {

template <int Index = 0, typename T, typename Reducers>
FK_HOST_DEVICE_STATIC T applyReducer(const int pass, const T a, const T b,
                                     const Reducers& reducers) {
    if constexpr (Index < std::decay_t<Reducers>::size) {
        if (pass == Index) {
            return static_cast<T>(make_tuple(a, b) | get<Index>(reducers));
        }
        return applyReducer<Index + 1>(pass, a, b, reducers);
    } else {
        return a;
    }
}

template <int Index = 0, typename Reducers>
FK_HOST_CNST bool reducersAreUnary() {
    if constexpr (Index == std::decay_t<Reducers>::size) {
        return true;
    } else {
        using Reducer = std::decay_t<decltype(
            get<Index>(std::declval<const Reducers&>()))>;
        return Reducer::template is<UnaryType> &&
               reducersAreUnary<Index + 1, Reducers>();
    }
}

FK_HOST_DEVICE_FUSE int clamp(const int value, const int low,
                              const int high) {
    return value < low ? low : (value > high ? high : value);
}

} // namespace morphology_detail

template <ParArch PA, typename DPPDetails>
struct MorphologyDPP;

template <typename DPPDetails>
struct MorphologyDPP<ParArch::CPU, DPPDetails> {
private:
    using SelfType = MorphologyDPP<ParArch::CPU, DPPDetails>;
    using T = typename DPPDetails::ValueType;

public:
    FK_STATIC_STRUCT(MorphologyDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::CPU;

    template <typename Reducers>
    static constexpr bool validReducerCount =
        isTuple_v<Reducers> && std::decay_t<Reducers>::size >= 1 &&
        std::decay_t<Reducers>::size <= DPPDetails::MAX_PASSES;

    template <typename InputReadIOp, typename OutputWriteIOp,
              typename Reducers>
    FK_HOST_STATIC void exec(const DPPDetails& details,
                             const InputReadIOp& input,
                             const OutputWriteIOp& output,
                             const Reducers& reducers) {
        static_assert(validReducerCount<Reducers>,
                      "MorphologyDPP requires 1 through 4 reducers");
        static_assert(morphology_detail::reducersAreUnary<0, Reducers>(),
                      "Morphology reducers must be Unary IOps");
        static_assert(isAnyCompleteReadType<InputReadIOp>,
                      "Morphology input must be a complete Read IOp");
        static_assert(isAnyWriteType<OutputWriteIOp>,
                      "Morphology output must be a Write IOp");
        if (!DPPDetails::valid(details)) return;

        const int elements = details.width * details.height;
        std::vector<T> current(static_cast<size_t>(elements));
        std::vector<T> next(static_cast<size_t>(elements));
        for (int y = 0; y < details.height; ++y)
            for (int x = 0; x < details.width; ++x)
                current[static_cast<size_t>(y * details.width + x)] =
                    static_cast<T>(InputReadIOp::Operation::exec(
                        Point{x, y, 0}, input));

        for (int pass = 0; pass < std::decay_t<Reducers>::size; ++pass) {
            for (int y = 0; y < details.height; ++y) {
                for (int x = 0; x < details.width; ++x) {
                    bool first = true;
                    T accumulator{};
                    for (int my = 0; my < details.maskH; ++my) {
                        const int sy = morphology_detail::clamp(
                            y + my - details.anchorY, 0,
                            details.height - 1);
                        for (int mx = 0; mx < details.maskW; ++mx) {
                            const int sx = morphology_detail::clamp(
                                x + mx - details.anchorX, 0,
                                details.width - 1);
                            const T value = current[static_cast<size_t>(
                                sy * details.width + sx)];
                            if (first) {
                                accumulator = value;
                                first = false;
                            } else {
                                accumulator =
                                    morphology_detail::applyReducer(
                                        pass, accumulator, value, reducers);
                            }
                        }
                    }
                    next[static_cast<size_t>(y * details.width + x)] =
                        accumulator;
                }
            }
            current.swap(next);
        }

        for (int y = 0; y < details.height; ++y)
            for (int x = 0; x < details.width; ++x)
                OutputWriteIOp::Operation::exec(
                    Point{x, y, 0},
                    current[static_cast<size_t>(y * details.width + x)],
                    output);
    }
};

#if defined(__NVCC__)
template <typename DPPDetails>
struct MorphologyDPP<ParArch::GPU_NVIDIA, DPPDetails> {
private:
    using SelfType = MorphologyDPP<ParArch::GPU_NVIDIA, DPPDetails>;
    using T = typename DPPDetails::ValueType;

    template <typename InputReadIOp>
    FK_DEVICE_FUSE T readClamped(const DPPDetails& details,
                                 const InputReadIOp& input,
                                 const int x, const int y) {
        const int sx = morphology_detail::clamp(x, 0, details.width - 1);
        const int sy = morphology_detail::clamp(y, 0, details.height - 1);
        return static_cast<T>(InputReadIOp::Operation::exec(
            Point{sx, sy, 0}, input));
    }

public:
    FK_STATIC_STRUCT(MorphologyDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::GPU_NVIDIA;

    template <typename Reducers>
    static constexpr bool validReducerCount =
        isTuple_v<Reducers> && std::decay_t<Reducers>::size >= 1 &&
        std::decay_t<Reducers>::size <= DPPDetails::MAX_PASSES;

    template <typename InputReadIOp, typename OutputWriteIOp,
              typename Reducers>
    FK_HOST_DEVICE_STATIC void exec(const DPPDetails& details,
                                    const InputReadIOp& input,
                                    const OutputWriteIOp& output,
                                    const Reducers& reducers) {
        static_assert(validReducerCount<Reducers>,
                      "MorphologyDPP requires 1 through 4 reducers");
        static_assert(morphology_detail::reducersAreUnary<0, Reducers>(),
                      "Morphology reducers must be Unary IOps");
        static_assert(isAnyCompleteReadType<InputReadIOp>,
                      "Morphology input must be a complete Read IOp");
        static_assert(isAnyWriteType<OutputWriteIOp>,
                      "Morphology output must be a Write IOp");
#if defined(__CUDA_ARCH__)
        __shared__ T stage0[DPPDetails::MAX_STAGE_ELEMENTS];
        __shared__ T stage1[DPPDetails::MAX_STAGE_ELEMENTS];

        constexpr int PASSES = std::decay_t<Reducers>::size;
        const int tileX = blockIdx.x * DPPDetails::TILE_WIDTH;
        const int tileY = blockIdx.y * DPPDetails::TILE_HEIGHT;
        const int tid = threadIdx.y * blockDim.x + threadIdx.x;
        const int threads = blockDim.x * blockDim.y;
        const int initialW = DPPDetails::TILE_WIDTH +
                             PASSES * (details.maskW - 1);
        const int initialH = DPPDetails::TILE_HEIGHT +
                             PASSES * (details.maskH - 1);
        const int initialX = tileX - PASSES * details.anchorX;
        const int initialY = tileY - PASSES * details.anchorY;

        for (int index = tid; index < initialW * initialH;
             index += threads) {
            const int lx = index % initialW;
            const int ly = index / initialW;
            stage0[index] = readClamped(
                details, input, initialX + lx, initialY + ly);
        }
        __syncthreads();

        T* source = stage0;
        T* target = stage1;
        T finalValue{};
        #pragma unroll
        for (int pass = 0; pass < PASSES; ++pass) {
            const int remainingSource = PASSES - pass;
            const int remainingTarget = remainingSource - 1;
            const int sourceW = DPPDetails::TILE_WIDTH +
                                remainingSource * (details.maskW - 1);
            const int targetW = DPPDetails::TILE_WIDTH +
                                remainingTarget * (details.maskW - 1);
            const int targetH = DPPDetails::TILE_HEIGHT +
                                remainingTarget * (details.maskH - 1);
            const int sourceX = tileX -
                                remainingSource * details.anchorX;
            const int sourceY = tileY -
                                remainingSource * details.anchorY;
            const int targetX = tileX -
                                remainingTarget * details.anchorX;
            const int targetY = tileY -
                                remainingTarget * details.anchorY;
            const bool finalPass = pass == PASSES - 1;
            const int targetElements = targetW * targetH;

            for (int index = tid; index < targetElements;
                 index += threads) {
                const int lx = index % targetW;
                const int ly = index / targetW;
                const int centerX = morphology_detail::clamp(
                    targetX + lx, 0, details.width - 1);
                const int centerY = morphology_detail::clamp(
                    targetY + ly, 0, details.height - 1);
                bool first = true;
                T accumulator{};
                for (int my = 0; my < details.maskH; ++my) {
                    const int sampleY = morphology_detail::clamp(
                        centerY + my - details.anchorY,
                        0, details.height - 1);
                    const int sourceLY = sampleY - sourceY;
                    for (int mx = 0; mx < details.maskW; ++mx) {
                        const int sampleX = morphology_detail::clamp(
                            centerX + mx - details.anchorX,
                            0, details.width - 1);
                        const int sourceLX = sampleX - sourceX;
                        const T value = source[sourceLY * sourceW + sourceLX];
                        if (first) {
                            accumulator = value;
                            first = false;
                        } else {
                            accumulator = morphology_detail::applyReducer(
                                pass, accumulator, value, reducers);
                        }
                    }
                }
                if (finalPass)
                    finalValue = accumulator;
                else
                    target[index] = accumulator;
            }
            if (!finalPass) {
                __syncthreads();
                T* temporary = source;
                source = target;
                target = temporary;
            }
        }

        const int outputX = tileX + threadIdx.x;
        const int outputY = tileY + threadIdx.y;
        if (threadIdx.x < DPPDetails::TILE_WIDTH &&
            threadIdx.y < DPPDetails::TILE_HEIGHT &&
            outputX < details.width && outputY < details.height) {
            OutputWriteIOp::Operation::exec(
                Point{outputX, outputY, 0}, finalValue, output);
        }
#else
        (void)details;
        (void)input;
        (void)output;
        (void)reducers;
#endif
    }
};

template <typename DPPDetails, typename InputReadIOp,
          typename OutputWriteIOp, typename Reducers>
__global__ void morphologyDPPKernel(const DPPDetails details,
                                    const InputReadIOp input,
                                    const OutputWriteIOp output,
                                    const Reducers reducers) {
    MorphologyDPP<ParArch::GPU_NVIDIA, DPPDetails>::exec(
        details, input, output, reducers);
}

template <typename DPPDetails, typename InputReadIOp,
          typename OutputWriteIOp, typename Reducers>
inline bool executeMorphology(const DPPDetails& details,
                              const InputReadIOp& input,
                              const OutputWriteIOp& output,
                              const Reducers& reducers,
                              Stream_<ParArch::GPU_NVIDIA>& stream) {
    if (!DPPDetails::valid(details)) return false;
    static_assert(MorphologyDPP<ParArch::GPU_NVIDIA, DPPDetails>::
                      template validReducerCount<Reducers>,
                  "MorphologyDPP requires 1 through 4 reducers");
    const dim3 block(DPPDetails::TILE_WIDTH, DPPDetails::TILE_HEIGHT, 1);
    const dim3 grid((details.width + DPPDetails::TILE_WIDTH - 1) /
                        DPPDetails::TILE_WIDTH,
                    (details.height + DPPDetails::TILE_HEIGHT - 1) /
                        DPPDetails::TILE_HEIGHT,
                    1);
    morphologyDPPKernel<<<grid, block, 0, stream.getCUDAStream()>>>(
        details, input, output, reducers);
    gpuErrchk(cudaGetLastError());
    return true;
}

template <typename DPPDetails, typename InputReadIOp,
          typename OutputWriteIOp>
inline bool executeErode(const DPPDetails& details,
                         const InputReadIOp& input,
                         const OutputWriteIOp& output,
                         Stream_<ParArch::GPU_NVIDIA>& stream) {
    using T = typename DPPDetails::ValueType;
    return executeMorphology(
        details, input, output,
        make_tuple(Min<T, T, T, UnaryType>::build()), stream);
}

template <typename DPPDetails, typename InputReadIOp,
          typename OutputWriteIOp>
inline bool executeDilate(const DPPDetails& details,
                          const InputReadIOp& input,
                          const OutputWriteIOp& output,
                          Stream_<ParArch::GPU_NVIDIA>& stream) {
    using T = typename DPPDetails::ValueType;
    return executeMorphology(
        details, input, output,
        make_tuple(Max<T, T, T, UnaryType>::build()), stream);
}

template <typename DPPDetails, typename InputReadIOp,
          typename OutputWriteIOp>
inline bool executeOpen(const DPPDetails& details,
                        const InputReadIOp& input,
                        const OutputWriteIOp& output,
                        Stream_<ParArch::GPU_NVIDIA>& stream) {
    using T = typename DPPDetails::ValueType;
    return executeMorphology(
        details, input, output,
        make_tuple(Min<T, T, T, UnaryType>::build(),
                   Max<T, T, T, UnaryType>::build()), stream);
}

template <typename DPPDetails, typename InputReadIOp,
          typename OutputWriteIOp>
inline bool executeClose(const DPPDetails& details,
                         const InputReadIOp& input,
                         const OutputWriteIOp& output,
                         Stream_<ParArch::GPU_NVIDIA>& stream) {
    using T = typename DPPDetails::ValueType;
    return executeMorphology(
        details, input, output,
        make_tuple(Max<T, T, T, UnaryType>::build(),
                   Min<T, T, T, UnaryType>::build()), stream);
}
#endif // defined(__NVCC__)

} // namespace fk

#endif // FK_MORPHOLOGY
