/* Copyright 2025 Grup Mediapro S.L.U (Oscar Amoros Huguet)
   Copyright 2026 Oscar Amoros Huguet

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
#include <fused_kernel/core/execution_model/executor_details/launch_config.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/basic_ops/set.h>
#include <fused_kernel/core/execution_model/stream.h>

#if defined(__NVCC__)
#include <fused_kernel/core/execution_model/executor_details/executor_kernels.h>
#endif

namespace fk {

    template <typename Child>
    struct BaseExecutor {
      private:
        template <typename StreamT, typename... Args>
        FK_HOST_FUSE void applyExecuteOperations_helper(Tuple<StreamT, Args...>& argTuple) {
            apply(Child::template executeOperations_helper<Args...>, argTuple);
        }
      public:
        template <ParArch PA, typename... IOps>
        FK_HOST_FUSE void executeOperationsBase_helper(Stream_<PA>& stream, const IOps&... iOps) {
            const auto fb_iOps = BackFuser::fuse_back(iOps...);
            auto fullArgs = tuple_cat(forward_as_tuple(stream), fb_iOps);
            applyExecuteOperations_helper(fullArgs);
        }

        FK_STATIC_STRUCT(BaseExecutor, BaseExecutor)

        template <enum ParArch PA, typename... IOps>
        FK_HOST_FUSE void executeOperations(Stream_<PA>& stream, const IOps&... iOps) {
            executeOperationsBase_helper(stream, iOps...);
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
template <ParArch PA, ND D, typename I, typename... IOps> \
FK_HOST_FUSE void executeOperations(const Ptr<D, I>& input, Stream_<PA>& stream, const IOps&... iOps) { \
    Parent::executeOperations(input, stream, iOps...); \
} \
template <ParArch PA, ND D, typename I, typename O, typename... IOps> \
FK_HOST_FUSE void executeOperations(const Ptr<D, I>& input, const Ptr<D, O>& output, Stream_<PA>& stream, const IOps&... iOps) { \
    Parent::executeOperations(input, output, stream, iOps...); \
} \
template <ParArch PA, typename I, size_t BATCH, typename... IOps> \
FK_HOST_FUSE void executeOperations(const std::array<Ptr2D<I>, BATCH>& input, const int& activeBatch, const I& defaultValue, \
                                    Stream_<PA>& stream, const IOps&... iOps) { \
    Parent::executeOperations(input, activeBatch, defaultValue, stream, iOps...); \
} \
template <ParArch PA, typename I, size_t BATCH, typename... IOps> \
FK_HOST_FUSE void executeOperations(const std::array<Ptr2D<I>, BATCH>& input, Stream_<PA>& stream, const IOps&... iOps) { \
    Parent::executeOperations(input, stream, iOps...); \
} \
template <ParArch PA, typename I, typename O, size_t Batch, typename... IOps> \
FK_HOST_FUSE void executeOperations(const std::array<Ptr2D<I>, Batch>& input, const int& activeBatch, const I& defaultValue, \
                                    const Tensor<O>& output, Stream_<PA>& stream, const IOps&... iOps) { \
    Parent::executeOperations(input, activeBatch, defaultValue, output, stream, iOps...); \
} \
template <ParArch PA, typename I, typename O, size_t Batch, typename... IOps> \
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
        FK_STATIC_STRUCT(Executor, Child)
        FK_HOST_FUSE ParArch parArch() {
            return ParArch::CPU;
        }
        DECLARE_EXECUTOR_PARENT_IMPL
    };

#if defined(__NVCC__)
    template <enum TF TFEN>
    struct Executor<TransformDPP<ParArch::GPU_NVIDIA, TFEN>> {
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
                    launchTransformDPP_Kernel<ParArch::GPU_NVIDIA, TFEN, false><<<grid, block, 0, stream>>>(tDetails, iOps...);
                    gpuErrchk(cudaGetLastError());
                } else {
                    launchTransformDPP_Kernel<ParArch::GPU_NVIDIA, TFEN, true><<<grid, block, 0, stream>>>(tDetails, iOps...);
                    gpuErrchk(cudaGetLastError());
                }
            } else {
                const auto readOp = get_arg<0>(iOps...);

                const ActiveThreads activeThreads = readOp.getActiveThreads();

                const CtxDim3 ctx_block = getDefaultBlockSize(activeThreads.x, activeThreads.y);

                const dim3 block{ ctx_block.x, ctx_block.y, 1 };
                const dim3 grid{ static_cast<uint>(ceil(activeThreads.x / static_cast<float>(block.x))),
                                 static_cast<uint>(ceil(activeThreads.y / static_cast<float>(block.y))),
                                 activeThreads.z };
                launchTransformDPP_Kernel<ParArch::GPU_NVIDIA, TFEN, true><<<grid, block, 0, stream>>>(tDetails, iOps...);
                gpuErrchk(cudaGetLastError());
            }
        }
    public:
        FK_STATIC_STRUCT(Executor, Child)
        FK_HOST_FUSE ParArch parArch() {
            return ParArch::GPU_NVIDIA;
        }
        DECLARE_EXECUTOR_PARENT_IMPL
    };

    template <typename SequenceSelector>
    struct Executor<DivergentBatchTransformDPP<ParArch::GPU_NVIDIA, SequenceSelector>> {
    private:
        using DPPType = DivergentBatchTransformDPP<ParArch::GPU_NVIDIA, SequenceSelector>;
        using DPPDetails = typename DPPType::DPPDetails;
        using SelfType = Executor<DPPType>;

        template <typename... IOpSequenceTypes>
        FK_HOST_FUSE ActiveThreads getActiveThreads(const IOpSequenceTypes&... iOpSequences) {
            const uint x = cxp::max::f(get<0>(iOpSequences.iOps).getActiveThreads().x...);
            const uint y = cxp::max::f(get<0>(iOpSequences.iOps).getActiveThreads().y...);
            const uint z = cxp::sum::f(get<0>(iOpSequences.iOps).getActiveThreads().z...);
            return ActiveThreads{ x, y, z }; 
        }

        template <typename... IOps>
        FK_HOST_FUSE auto fuseBackSequence(const IOpSequence<IOps...>& iOpSeq) {
            return buildOperationSequence_tup(
                apply([](auto&&... args) {
                    // Now fuse_back deduces the types naturally and preserves value categories via perfect forwarding
                    return BackFuser::fuse_back(std::forward<decltype(args)>(args)...);
                }, iOpSeq.iOps)
            );
        }

        template <typename... IOpSequenceTypes>
        FK_HOST_FUSE void executeOperationsFused(Stream_<ParArch::GPU_NVIDIA>& stream, const IOpSequenceTypes&... iOpSequences) {
            const ActiveThreads activeThreads = getActiveThreads(iOpSequences...);
            const DPPDetails details{};

            const dim3 block(cxp::min::f(activeThreads.x, 32u), cxp::min::f(activeThreads.y, 8u));
            const dim3 grid(ceil(activeThreads.x / static_cast<float>(block.x)),
                            ceil(activeThreads.y / static_cast<float>(block.y)), activeThreads.z);
            launchDivergentBatchTransformDPP_Kernel<ParArch::GPU_NVIDIA, SequenceSelector><<<grid, block, 0, stream.getCUDAStream()>>>(details, iOpSequences...);
            gpuErrchk(cudaGetLastError());
        }

        template <typename... IOpSequenceTypes>
        FK_HOST_FUSE void executeOperations_helper(Stream_<ParArch::GPU_NVIDIA>& stream, const IOpSequenceTypes&... iOpSequences) {
            executeOperationsFused(stream, fuseBackSequence(iOpSequences)...);
        }

    public:
        FK_STATIC_STRUCT(Executor, SelfType)
        FK_HOST_FUSE ParArch parArch() {
            return ParArch::GPU_NVIDIA;
        }
        template <typename... IOpSequenceTypes>
        FK_HOST_FUSE void executeOperations(Stream_<ParArch::GPU_NVIDIA>& stream, const IOpSequenceTypes&... iOpSequences) {
            executeOperations_helper(stream, iOpSequences...);
        }
    };
#endif

    // ===============================================================================
    // Generic InstantiableDPP execution path
    // ===============================================================================
    // ONE executor path for every DPP that conforms to the InstantiableDPP protocol
    // (see instantiable_dpp.h): no per-DPP Executor specialization and no per-DPP
    // __global__ kernel are needed.

    // hasGetLaunchConfig trait: detects the getLaunchConfig(details, iOps...) hook
    template <typename DPP, typename Details, typename Enabler, typename... IOps>
    struct HasGetLaunchConfig_ : std::false_type {};
    template <typename DPP, typename Details, typename... IOps>
    struct HasGetLaunchConfig_<DPP, Details,
        std::void_t<decltype(DPP::getLaunchConfig(std::declval<const Details&>(), std::declval<const IOps&>()...))>,
        IOps...> : std::true_type {};
    template <typename DPP, typename Details, typename... IOps>
    constexpr bool hasGetLaunchConfig_v = HasGetLaunchConfig_<DPP, Details, void, IOps...>::value;

    /**
     * @brief execute: executes an InstantiableDPP (built with DPP::build(iOps...)) on the
     * CPU backend. The DPP's exec() implements the whole sequential loop.
     */
    template <typename DPP, typename DPPDetails, typename... IOps>
    inline void execute(Stream_<ParArch::CPU>& stream, const InstantiableDPP<DPP, DPPDetails, IOps...>& iDPP) {
        static_assert(DPP::PAR_ARCH == ParArch::CPU,
            "fk::execute: the InstantiableDPP was built for a different backend than the Stream_<ParArch::CPU> provided.");
        static_assert(dppIOContractSatisfied<DPP, IOps...>(),
            "fk::execute: the InstantiableDPP does not conform to the IO contract (IO_SPEC) of its DPP.");
        apply([&](const auto&... iOps) {
            DPP::exec(iDPP.details, iOps...);
        }, iDPP.iOps);
        (void)stream; // CPU streams are synchronous
    }

#if defined(__NVCC__)
    /**
     * @brief execute: executes an InstantiableDPP (built with DPP::build(iOps...)) on the
     * GPU_NVIDIA backend, launching the generic launchInstantiableDPP_Kernel with the
     * grid/block configuration provided by the DPP's getLaunchConfig().
     */
    template <typename DPP, typename DPPDetails, typename... IOps>
    inline void execute(Stream_<ParArch::GPU_NVIDIA>& stream, const InstantiableDPP<DPP, DPPDetails, IOps...>& iDPP) {
        static_assert(DPP::PAR_ARCH == ParArch::GPU_NVIDIA,
            "fk::execute: the InstantiableDPP was built for a different backend than the Stream_<ParArch::GPU_NVIDIA> provided.");
        static_assert(dppIOContractSatisfied<DPP, IOps...>(),
            "fk::execute: the InstantiableDPP does not conform to the IO contract (IO_SPEC) of its DPP.");
        static_assert(hasGetLaunchConfig_v<DPP, DPPDetails, IOps...>,
            "fk::execute: a GPU DPP must implement 'FK_HOST_FUSE DPPLaunchConfig getLaunchConfig(const Details&, const IOps&...)'.");
        apply([&](const auto&... iOps) {
            const DPPLaunchConfig config = DPP::getLaunchConfig(iDPP.details, iOps...);
            const dim3 grid{ config.grid.x, config.grid.y, config.grid.z };
            const dim3 block{ config.block.x, config.block.y, config.block.z };
            launchInstantiableDPP_Kernel<DPP><<<grid, block, 0, stream.getCUDAStream()>>>(iDPP.details, iOps...);
            gpuErrchk(cudaGetLastError());
        }, iDPP.iOps);
    }
#endif
} // namespace fk

#endif // FK_EXECUTORS_CUH