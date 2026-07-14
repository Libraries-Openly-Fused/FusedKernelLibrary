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

#ifndef FK_ATTENTION_SOFTMAX_H
#define FK_ATTENTION_SOFTMAX_H

#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/collective/reduce.h>
#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/data/tuple.h>
#include <fused_kernel/core/execution_model/operation_model/operation_model.h>
#include <fused_kernel/core/execution_model/parallel_architectures.h>
#include <fused_kernel/core/execution_model/stream.h>
#include <fused_kernel/core/utils/utils.h>

#include <cfloat>
#include <cmath>
#include <type_traits>
#if defined(__NVCC__)
#include <cuda_fp16.h>
#endif

namespace fk {

FK_HOST_DEVICE_CNST float attnMaxF(const float a, const float b) {
    return a > b ? a : (b >= a ? b : (a == a ? a : b));
}

FK_HOST_DEVICE_STATIC float attnExpF(const float x) {
    return ::expf(x);
}

template <typename T>
FK_HOST_DEVICE_CNST float attnToF32(const T& value) {
#if defined(__NVCC__)
    if constexpr (std::is_same_v<T, __half>) {
        return __half2float(value);
    } else
#endif
    {
        return static_cast<float>(value);
    }
}

template <typename T>
FK_HOST_DEVICE_CNST T attnFromF32(const float value) {
#if defined(__NVCC__)
    if constexpr (std::is_same_v<T, __half>) {
        return __float2half(value);
    } else
#endif
    {
        return static_cast<T>(value);
    }
}

struct OnlineSoftmaxState {
    float maximum;
    float denominator;
};

FK_HOST_DEVICE_STATIC OnlineSoftmaxState mergeSoftmaxStates(
        const OnlineSoftmaxState& lhs, const OnlineSoftmaxState& rhs) {
    const float maximum = attnMaxF(lhs.maximum, rhs.maximum);
    const float lhsDenominator = lhs.denominator == 0.f
        ? 0.f
        : lhs.denominator * attnExpF(lhs.maximum - maximum);
    const float rhsDenominator = rhs.denominator == 0.f
        ? 0.f
        : rhs.denominator * attnExpF(rhs.maximum - maximum);
    return {maximum, lhsDenominator + rhsDenominator};
}

template <typename I1 = OnlineSoftmaxState,
          typename I2 = I1,
          typename O = I1,
          typename IType = UnaryType>
struct MergeSoftmaxState;

template <typename I1, typename I2, typename O>
struct MergeSoftmaxState<I1, I2, O, UnaryType> {
private:
    using SelfType = MergeSoftmaxState<I1, I2, O, UnaryType>;

public:
    FK_STATIC_STRUCT(MergeSoftmaxState, SelfType)
    using Parent = UnaryOperation<Tuple<I1, I2>, O, SelfType>;
    DECLARE_UNARY_PARENT

    FK_HOST_DEVICE_STATIC OutputType exec(const InputType input) {
        return mergeSoftmaxStates(get<0>(input), get<1>(input));
    }
};

template <typename T, int BLOCK_SIZE = 256>
struct SoftmaxDPPDetails {
    static_assert(BLOCK_SIZE >= 32 && BLOCK_SIZE <= 1024 &&
                  BLOCK_SIZE % 32 == 0,
                  "SoftmaxDPP block size must be a warp multiple in [32, 1024]");

    using ScalarType = T;
    using ValueType = OnlineSoftmaxState;
    static constexpr int BLOCK_THREADS = BLOCK_SIZE;

    int rows;
    int width;
    OnlineSoftmaxState identity{-FLT_MAX, 0.f};
    int rowOffset{0};
};

template <ParArch PA, typename DPPDetails>
struct SoftmaxDPP;

template <typename DPPDetails>
struct SoftmaxDPP<ParArch::CPU, DPPDetails> {
private:
    using SelfType = SoftmaxDPP<ParArch::CPU, DPPDetails>;
    using Scalar = typename DPPDetails::ScalarType;
    using State = typename DPPDetails::ValueType;

public:
    FK_STATIC_STRUCT(SoftmaxDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::CPU;

    template <typename ReadIOp, typename WriteIOp, typename MergeIOp>
    FK_HOST_STATIC void exec(const DPPDetails& details,
                             const ReadIOp& input,
                             const WriteIOp& output,
                             const MergeIOp& merge) {
        static_assert(isAnyReadType<ReadIOp>,
                      "SoftmaxDPP input must be a Read or ReadBack IOp");
        static_assert(isAnyWriteType<WriteIOp>,
                      "SoftmaxDPP output must be a Write IOp");
        static_assert(opIs<UnaryType, MergeIOp>,
                      "SoftmaxDPP merge must be an instantiable Unary IOp");

        for (int localRow = 0; localRow < details.rows; ++localRow) {
            const int row = details.rowOffset + localRow;
            State state = details.identity;
            for (int x = 0; x < details.width; ++x) {
                const float value = attnToF32(
                    ReadIOp::Operation::exec(Point{x, row, 0}, input));
                state = make_tuple(state, State{value, 1.f}) | merge;
            }
            const float inverseDenominator = 1.f / state.denominator;
            for (int x = 0; x < details.width; ++x) {
                const float value = attnToF32(
                    ReadIOp::Operation::exec(Point{x, row, 0}, input));
                const Scalar result = attnFromF32<Scalar>(
                    attnExpF(value - state.maximum) * inverseDenominator);
                WriteIOp::Operation::exec(Point{x, row, 0}, result, output);
            }
        }
    }
};

#if defined(__NVCC__)
template <typename DPPDetails>
struct SoftmaxDPP<ParArch::GPU_NVIDIA, DPPDetails> {
private:
    using SelfType = SoftmaxDPP<ParArch::GPU_NVIDIA, DPPDetails>;
    using Scalar = typename DPPDetails::ScalarType;
    using State = typename DPPDetails::ValueType;

    // CUDA rejects function-local __shared__ storage in FK_DEVICE_FUSE's
    // constexpr function. Keep only this shared-memory runtime body private.
    template <typename ReadIOp, typename WriteIOp, typename MergeIOp>
    FK_DEVICE_STATIC void execRuntime(
            const DPPDetails& details,
            const ReadIOp& input,
            const WriteIOp& output,
            const MergeIOp& merge) {
        __shared__ State blockScratch[DPPDetails::BLOCK_THREADS];

        const int localRow = static_cast<int>(blockIdx.x);
        if (localRow >= details.rows) return;
        const int row = details.rowOffset + localRow;
        const int tid = static_cast<int>(threadIdx.x);

        State state = details.identity;
        for (int x = tid; x < details.width; x += DPPDetails::BLOCK_THREADS) {
            const float value = attnToF32(
                ReadIOp::Operation::exec(Point{x, row, 0}, input));
            state = make_tuple(state, State{value, 1.f}) | merge;
        }

        const State reduced =
            ReduceBlockDPP<ParArch::GPU_NVIDIA, DPPDetails>::exec(
                details, state, merge, blockScratch);
        const float inverseDenominator = 1.f / reduced.denominator;
        for (int x = tid; x < details.width; x += DPPDetails::BLOCK_THREADS) {
            const float value = attnToF32(
                ReadIOp::Operation::exec(Point{x, row, 0}, input));
            const Scalar result = attnFromF32<Scalar>(
                attnExpF(value - reduced.maximum) * inverseDenominator);
            WriteIOp::Operation::exec(Point{x, row, 0}, result, output);
        }
    }

public:
    FK_STATIC_STRUCT(SoftmaxDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::GPU_NVIDIA;

    template <typename ReadIOp, typename WriteIOp, typename MergeIOp>
    FK_DEVICE_FUSE void exec(const DPPDetails& details,
                             const ReadIOp& input,
                             const WriteIOp& output,
                             const MergeIOp& merge) {
        static_assert(isAnyReadType<ReadIOp>,
                      "SoftmaxDPP input must be a Read or ReadBack IOp");
        static_assert(isAnyWriteType<WriteIOp>,
                      "SoftmaxDPP output must be a Write IOp");
        static_assert(opIs<UnaryType, MergeIOp>,
                      "SoftmaxDPP merge must be an instantiable Unary IOp");
        execRuntime(details, input, output, merge);
    }
};

template <typename DPPDetails, typename ReadIOp,
          typename WriteIOp, typename MergeIOp>
__global__ void launchSoftmaxDPPKernel(
        const __grid_constant__ DPPDetails details,
        const __grid_constant__ ReadIOp input,
        const __grid_constant__ WriteIOp output,
        const __grid_constant__ MergeIOp merge) {
    SoftmaxDPP<ParArch::GPU_NVIDIA, DPPDetails>::exec(
        details, input, output, merge);
}

template <typename DPPDetails, typename ReadIOp,
          typename WriteIOp, typename MergeIOp>
inline void executeSoftmax(const DPPDetails& details,
                           const ReadIOp& input,
                           const WriteIOp& output,
                           const MergeIOp& merge,
                           Stream_<ParArch::GPU_NVIDIA>& stream) {
    static_assert(std::is_trivially_copyable_v<DPPDetails>,
                  "SoftmaxDPP Details must be trivially copyable");
    if (details.rows <= 0 || details.width <= 0) return;
    launchSoftmaxDPPKernel<<<details.rows, DPPDetails::BLOCK_THREADS, 0,
                            stream.getCUDAStream()>>>(
        details, input, output, merge);
    gpuErrchk(cudaGetLastError());
}

template <int BLOCK_SIZE = 256, typename ReadIOp, typename OT>
inline void executeSoftmax(const ReadIOp& input,
                           const Ptr2D<OT>& output,
                           const int rows, const int width,
                           Stream_<ParArch::GPU_NVIDIA>& stream) {
    const SoftmaxDPPDetails<OT, BLOCK_SIZE> details{rows, width};
    const auto write = PerThreadWrite<ND::_2D, OT>::build(output);
    const auto merge = MergeSoftmaxState<>::build();
    executeSoftmax(details, input, write, merge, stream);
}

template <typename T, int BLOCK_SIZE = 256>
inline void executeSoftmax(const Ptr2D<T>& input,
                           const Ptr2D<T>& output,
                           Stream_<ParArch::GPU_NVIDIA>& stream) {
    const SoftmaxDPPDetails<T, BLOCK_SIZE> details{
        static_cast<int>(input.dims().height),
        static_cast<int>(input.dims().width)};
    const auto read = PerThreadRead<ND::_2D, T>::build(input);
    const auto write = PerThreadWrite<ND::_2D, T>::build(output);
    const auto merge = MergeSoftmaxState<>::build();
    executeSoftmax(details, read, write, merge, stream);
}
#endif // defined(__NVCC__)

} // namespace fk

#endif // FK_ATTENTION_SOFTMAX_H
