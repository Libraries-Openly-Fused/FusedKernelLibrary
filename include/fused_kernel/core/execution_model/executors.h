/* Copyright 2025 Grup Mediapro S.L.U (Oscar Amoros Huguet)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

// In the future we will change all .h files to .h files, since we are going to be multi platform
#ifndef FK_EXECUTORS_CUH
#define FK_EXECUTORS_CUH

#include <fused_kernel/core/execution_model/parallel_architectures.h>
#include <fused_kernel/core/execution_model/data_parallel_patterns.h>
#include <fused_kernel/core/execution_model/memory_operations.h>
#include <fused_kernel/algorithms/basic_ops/set.h>
#include <fused_kernel/core/execution_model/stream.h>

#if defined(__NVCC__) || defined(__CUDA__) || defined(__HIP__)
#include <fused_kernel/core/execution_model/executor_details/executor_kernels.h>
#endif

namespace fk {

#if defined(__NVCC__) || defined(__CUDA__) || defined(__HIP__) || defined(NVRTC_ENABLED)
    struct CtxDim3 {
        uint x;
        uint y;
        uint z;
        constexpr CtxDim3(const dim3& dims) : x(dims.x), y(dims.y), z(dims.z) {}
        constexpr CtxDim3() : x(1), y(1), z(1) {}
        constexpr CtxDim3(const uint& x) : x(x), y(1), z(1) {}
        constexpr CtxDim3(const uint& x, const uint& y) : x(x), y(y), z(1) {}
        constexpr CtxDim3(const uint& x, const uint& y, const uint& z) : x(x), y(y), z(z) {}
    };
#endif

    template <typename Child>
    struct BaseExecutor {
    private:
        template <typename First, typename Second, typename... IOps>
        FK_HOST_FUSE auto fuseBack(const First& first, const Second& second, const IOps&... iOps) {
            if constexpr (sizeof...(iOps) == 0 || noneIncompleteReadBackType<IOps...>) {
                if constexpr (isIncompleteReadBackType<Second>) {
                    return first.then(second);
                } else {
                    return first;
                }
            } else {
                return fuseBack(first.then(second), iOps...);
            }
        }

        template <typename First, typename... IOps>
        FK_HOST_FUSE size_t idxFirstNonBack() {
            if constexpr (sizeof...(IOps) == 0) {
                return 1;
            } else {
                if constexpr (!noneIncompleteReadBackType<IOps...>) {
                    return 1 + idxFirstNonBack<IOps...>();
                } else {
                    if constexpr (isReadBackType<First> || isIncompleteReadBackType<First>) {
                        return 1;
                    } else {
                        return 0;
                    }
                }
            }
        }

    public:
        template <size_t initialIdx, ParArch PA, size_t... Idx, typename... IOps>
        FK_HOST_FUSE void executeOperationsBase_helper(const std::index_sequence<Idx...>&, Stream_<PA>& stream, const IOps&... iOps) {
            if constexpr (initialIdx == 0) {
                Child::executeOperations_helper(stream, iOps...);
            } else {
                const auto firstOp = fuseBack(iOps...);
                Child::executeOperations_helper(stream, firstOp, get<Idx + initialIdx>(iOps...)...);
            }
        }

        FK_STATIC_STRUCT(BaseExecutor, BaseExecutor)

        template <enum ParArch PA, typename... IOps>
        FK_HOST_FUSE void executeOperations(Stream_<PA>& stream, const IOps&... iOps) {
            constexpr size_t firstNonBackIdx = idxFirstNonBack<IOps...>();
            constexpr size_t remainingOps = sizeof...(iOps) - firstNonBackIdx;
            executeOperationsBase_helper<firstNonBackIdx>(std::make_index_sequence<remainingOps>{}, stream, iOps...);
        }

        template <enum ParArch PA, enum ND D, typename I, typename... IOps>
        FK_HOST_FUSE void executeOperations(const Ptr<D, I>& input, Stream_<PA>& stream,
                                            const IOps&... iOps) {
            executeOperations(stream, PerThreadRead<D, I>::build({ input }), iOps...);
        }

        template <enum ParArch PA, enum ND D, typename I, typename O, typename... IOps>
        FK_HOST_FUSE void executeOperations(const Ptr<D, I>& input, const Ptr<D, O>& output,
                                            Stream_<PA>& stream, const IOps&... iOps) {
            executeOperations(stream,
            PerThreadRead<D, I>::build({ input }), iOps..., PerThreadWrite<D, O>::build({ output }));
        }

        template <enum ParArch PA, typename I, size_t BATCH, typename... IOps>
        FK_HOST_FUSE void executeOperations(const std::array<Ptr2D<I>, BATCH>& input, const int& activeBatch, const I& defaultValue,
                                            Stream_<PA>& stream, const IOps&... iOps) {
            const auto batchReadIOp = PerThreadRead<ND::_2D, I>::build(activeBatch, defaultValue, input);
            executeOperations(stream, batchReadIOp, iOps...);
        }

        template <enum ParArch PA, typename I, size_t BATCH, typename... IOps>
        FK_HOST_FUSE void executeOperations(const std::array<Ptr2D<I>, BATCH>& input,
                                            Stream_<PA>& stream, const IOps&... iOps) {
            const auto batchReadIOp = PerThreadRead<ND::_2D, I>::build(input);
            executeOperations(stream, batchReadIOp, iOps...);
        }

        template <enum ParArch PA, typename I, typename O, size_t Batch, typename... IOps>
        FK_HOST_FUSE void executeOperations(const std::array<Ptr2D<I>, Batch>& input, const int& activeBatch, const I& defaultValue,
                                            const Tensor<O>& output, Stream_<PA>& stream, const IOps&... iOps) {
            const auto batchReadIOp = PerThreadRead<ND::_2D, I>::build(activeBatch, defaultValue, input);
            const auto writeOp = PerThreadWrite<ND::_3D, O>::build(output);
            executeOperations(stream, batchReadIOp, iOps..., writeOp);
        }

        template <enum ParArch PA, typename I, typename O, size_t Batch, typename... IOps>
        FK_HOST_FUSE void executeOperations(const std::array<Ptr2D<I>, Batch>& input, const Tensor<O>& output,
                                            Stream_<PA>& stream, const IOps&... iOps) {
            const auto batchReadIOp = PerThreadRead<ND::_2D, I>::build(input);
            const auto writeOp = PerThreadWrite<ND::_3D, O>::build(output);
            executeOperations(stream, batchReadIOp, iOps..., writeOp);
        }
    };

#define DECLARE_EXECUTOR_PARENT_IMPL \
friend class BaseExecutor<Child>; \
template <enum ParArch PA, typename... IOps> \
FK_HOST_FUSE void executeOperations(Stream_<PA>& stream, const IOps&... iOps) { \
    Parent::executeOperations(stream, iOps...); \
} \
template <enum ParArch PA, enum ND D, typename I, typename... IOps> \
FK_HOST_FUSE void executeOperations(const Ptr<D, I>& input, Stream_<PA>& stream, const IOps&... iOps) { \
    Parent::executeOperations(input, stream, iOps...); \
} \
template <enum ParArch PA, enum ND D, typename I, typename O, typename... IOps> \
FK_HOST_FUSE void executeOperations(const Ptr<D, I>& input, const Ptr<D, O>& output, Stream_<PA>& stream, const IOps&... iOps) { \
    Parent::executeOperations(input, output, stream, iOps...); \
} \
template <enum ParArch PA, typename I, size_t BATCH, typename... IOps> \
FK_HOST_FUSE void executeOperations(const std::array<Ptr2D<I>, BATCH>& input, const int& activeBatch, const I& defaultValue, \
                                    Stream_<PA>& stream, const IOps&... iOps) { \
    Parent::executeOperations(input, activeBatch, defaultValue, stream, iOps...); \
} \
template <enum ParArch PA, typename I, size_t BATCH, typename... IOps> \
FK_HOST_FUSE void executeOperations(const std::array<Ptr2D<I>, BATCH>& input, Stream_<PA>& stream, const IOps&... iOps) { \
    Parent::executeOperations(input, stream, iOps...); \
} \
template <enum ParArch PA, typename I, typename O, size_t Batch, typename... IOps> \
FK_HOST_FUSE void executeOperations(const std::array<Ptr2D<I>, Batch>& input, const int& activeBatch, const I& defaultValue, \
                                    const Tensor<O>& output, Stream_<PA>& stream, const IOps&... iOps) { \
    Parent::executeOperations(input, activeBatch, defaultValue, output, stream, iOps...); \
} \
template <enum ParArch PA, typename I, typename O, size_t Batch, typename... IOps> \
FK_HOST_FUSE void executeOperations(const std::array<Ptr2D<I>, Batch>& input, const Tensor<O>& output, \
                                    Stream_<PA>& stream, const IOps&... iOps) { \
    Parent::executeOperations(input, output, stream, iOps...); \
}

#ifdef NVRTC_ENABLED
    template <typename DataParallelPattern>
    struct Executor {
        FK_STATIC_STRUCT(Executor, Executor)
        static_assert(DataParallelPattern::PAR_ARCH == ParArch::GPU_NVIDIA ||
                      DataParallelPattern::PAR_ARCH == ParArch::CPU ||
                      DataParallelPattern::PAR_ARCH == ParArch::GPU_NVIDIA_JIT, "Only GPU_NVIDIA, CPU and GPU_NVIDIA_JIT are supported");
    };
#else
    template <typename DataParallelPattern>
    struct Executor {
        FK_STATIC_STRUCT(Executor, Executor)
        static_assert(DataParallelPattern::PAR_ARCH == ParArch::GPU_NVIDIA ||
                      DataParallelPattern::PAR_ARCH == ParArch::CPU,
                      "Only GPU_NVIDIA and CPU supported");
    };
#endif

    template <enum TF TFEN>
    struct Executor<TransformDPP<ParArch::CPU, TFEN, void>> {
        FK_STATIC_STRUCT(Executor, Executor)
    private:
        using Child = Executor<TransformDPP<ParArch::CPU, TFEN>>;
        using Parent = BaseExecutor<Child>;
        template <typename... IOps>
        FK_HOST_FUSE void executeOperations_helper(Stream_<ParArch::CPU>& stream, const IOps&... iOps) {
            constexpr ParArch PA = ParArch::CPU;
            const auto tDetails = TransformDPP<PA, TFEN>::build_details(iOps...);
            using TDPPDetails = std::decay_t<decltype(tDetails)>;
            if constexpr (TDPPDetails::TFI::ENABLED) {
                if (!tDetails.threadDivisible) {
                    TransformDPP<PA, TFEN, TDPPDetails, false>::exec(tDetails, iOps...);
                } else {
                    TransformDPP<PA, TFEN, TDPPDetails, true>::exec(tDetails, iOps...);
                }
            } else {
                TransformDPP<PA, TFEN, TDPPDetails, true>::exec(tDetails, iOps...);
            }
        }
    public:
        FK_HOST_FUSE ParArch parArch() {
            return ParArch::CPU;
        }
        DECLARE_EXECUTOR_PARENT_IMPL
    };

#if defined(__NVCC__) || defined(__CUDA__) || defined(__HIP__) || defined(NVRTC_ENABLED)
    struct ComputeBestSolutionBase {
        FK_HOST_FUSE uint computeDiscardedThreads(const uint width, const uint height, const uint blockDimx, const uint blockDimy) {
            const uint modX = width % blockDimx;
            const uint modY = height % blockDimy;
            const uint th_disabled_in_X = modX == 0 ? 0 : blockDimx - modX;
            const uint th_disabled_in_Y = modY == 0 ? 0 : blockDimy - modY;
            return (th_disabled_in_X * (modY == 0 ? height : (height + blockDimy)) + th_disabled_in_Y * width);
        }
    };

    template <uint bxS_t, uint byS_t>
    struct computeBestSolution {};

    template <uint bxS_t>
    struct computeBestSolution<bxS_t, 0> final : public ComputeBestSolutionBase {
        FK_HOST_FUSE void exec(const uint width, const uint height, uint& bxS, uint& byS, uint& minDiscardedThreads, const uint(&blockDimX)[4], const uint(&blockDimY)[2][4]) {
            const uint currentDiscardedThreads = computeDiscardedThreads(width, height, blockDimX[bxS_t], blockDimY[0][bxS_t]);
            if (minDiscardedThreads > currentDiscardedThreads) {
                minDiscardedThreads = currentDiscardedThreads;
                bxS = bxS_t;
                byS = 0;
                if (minDiscardedThreads == 0) return;
            }
            computeBestSolution<bxS_t, 1>::exec(width, height, bxS, byS, minDiscardedThreads, blockDimX, blockDimY);
        }
    };

    template <uint bxS_t>
    struct computeBestSolution<bxS_t, 1> final : public ComputeBestSolutionBase {
        FK_HOST_FUSE void exec(const uint width, const uint height, uint& bxS, uint& byS, uint& minDiscardedThreads, const uint(&blockDimX)[4], const uint(&blockDimY)[2][4]) {
            const uint currentDiscardedThreads = computeDiscardedThreads(width, height, blockDimX[bxS_t], blockDimY[1][bxS_t]);
            if (minDiscardedThreads > currentDiscardedThreads) {
                minDiscardedThreads = currentDiscardedThreads;
                bxS = bxS_t;
                byS = 1;
                if constexpr (bxS_t == 3) return;
                if (minDiscardedThreads == 0) return;
            }
            computeBestSolution<bxS_t + 1, 0>::exec(width, height, bxS, byS, minDiscardedThreads, blockDimX, blockDimY);
        }
    };

    template <>
    struct computeBestSolution<3, 1> final : public ComputeBestSolutionBase {
        FK_HOST_FUSE void exec(const uint width, const uint height, uint& bxS, uint& byS, uint& minDiscardedThreads, const uint(&blockDimX)[4], const uint(&blockDimY)[2][4]) {
            const uint currentDiscardedThreads = computeDiscardedThreads(width, height, blockDimX[3], blockDimY[1][3]);
            if (minDiscardedThreads > currentDiscardedThreads) {
                minDiscardedThreads = currentDiscardedThreads;
                bxS = 3;
                byS = 1;
            }
        }
    };

    FK_HOST_CNST CtxDim3 getDefaultBlockSize(const uint& width, const uint& height) {
        constexpr uint blockDimX[4] = { 32, 64, 128, 256 };  // Possible block sizes in the x axis
        constexpr uint blockDimY[2][4] = { { 8,  4,   2,   1},
                                          { 6,  3,   3,   2} };  // Possible block sizes in the y axis according to blockDim.x

        uint minDiscardedThreads = UINT_MAX;
        uint bxS = 0; // from 0 to 3
        uint byS = 0; // from 0 to 1

        computeBestSolution<0, 0>::exec(width, height, bxS, byS, minDiscardedThreads, blockDimX, blockDimY);

        return CtxDim3(blockDimX[bxS], blockDimY[byS][bxS]);
    }
#endif
#if defined(__NVCC__) || defined(__CUDA__) || defined(__HIP__)
    template <enum TF TFEN>
    struct Executor<TransformDPP<ParArch::GPU_NVIDIA, TFEN>> {
        FK_STATIC_STRUCT(Executor, Executor)
    private:
        using Child = Executor<TransformDPP<ParArch::GPU_NVIDIA, TFEN>>;
        using Parent = BaseExecutor<Child>;

        template <typename... IOps>
        FK_HOST_FUSE void executeOperations_helper(Stream_<ParArch::GPU_NVIDIA>& stream_, const IOps&... iOps) {
            const cudaStream_t stream = stream_.getCUDAStream();
            constexpr ParArch PA = ParArch::GPU_NVIDIA;
            const auto tDetails = TransformDPP<PA, TFEN>::build_details(iOps...);
            if constexpr (decltype(tDetails)::TFI::ENABLED) {
                const ActiveThreads activeThreads = tDetails.activeThreads;

                const CtxDim3 ctx_block = getDefaultBlockSize(activeThreads.x, activeThreads.y);

                const dim3 block{ ctx_block.x, ctx_block.y, 1 };
                const dim3 grid{ static_cast<uint>(ceil(activeThreads.x / static_cast<float>(block.x))),
                                 static_cast<uint>(ceil(activeThreads.y / static_cast<float>(block.y))),
                                 activeThreads.z };
                if (!tDetails.threadDivisible) {
                    launchTransformDPP_Kernel<PA, TFEN, false><<<grid, block, 0, stream>>>(tDetails, iOps...);
                    gpuErrchk(cudaGetLastError());
                } else {
                    launchTransformDPP_Kernel<PA, TFEN, true><<<grid, block, 0, stream>>>(tDetails, iOps...);
                    gpuErrchk(cudaGetLastError());
                }
            } else {
                const auto readOp = get<0>(iOps...);

                const ActiveThreads activeThreads = readOp.getActiveThreads();

                const CtxDim3 ctx_block = getDefaultBlockSize(activeThreads.x, activeThreads.y);

                const dim3 block{ ctx_block.x, ctx_block.y, 1 };
                const dim3 grid{ static_cast<uint>(ceil(activeThreads.x / static_cast<float>(block.x))),
                                 static_cast<uint>(ceil(activeThreads.y / static_cast<float>(block.y))),
                                 activeThreads.z };
                launchTransformDPP_Kernel<PA, TFEN, true><<<grid, block, 0, stream>>>(tDetails, iOps...);
                gpuErrchk(cudaGetLastError());
            }
        }
    public:
        FK_HOST_FUSE ParArch parArch() {
            return ParArch::GPU_NVIDIA;
        }
        DECLARE_EXECUTOR_PARENT_IMPL
    };
#endif
} // namespace fk

#endif // FK_EXECUTORS_CUH