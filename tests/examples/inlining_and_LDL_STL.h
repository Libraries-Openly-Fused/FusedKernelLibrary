/* Copyright 2025 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#define __ONLY_CU__
#include <tests/main.h>

#include <fused_kernel/core/execution_model/memory_operations.h>
#include <fused_kernel/core/utils/utils.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/image_processing/crop.h>
#include <fused_kernel/algorithms/image_processing/color_conversion.h>
#include <fused_kernel/algorithms/image_processing/resize.h>
#include <fused_kernel/fused_kernel.h>

using namespace fk;

template <ParArch PA = defaultParArch> struct SimpleTransformDPPValue;
template <ParArch PA = defaultParArch> struct SimpleTransformDPPValueLessCallDepth;
template <ParArch PA = defaultParArch> struct SimpleTransformDPPReference;
template <ParArch PA = defaultParArch> struct SimpleTransformDPPReferenceFoldExpr;

struct SimpleTransformDPPBaseValue {
    friend struct SimpleTransformDPPValue<ParArch::GPU_NVIDIA>; // Allow TransformDPP to access private members
  private:
    template <typename ReadIOp, typename... IOps>
    FK_HOST_DEVICE_FUSE void execute_thread(const Point thread, const ReadIOp readDF, const IOps... iOps) {
        using ReadOperation = typename ReadIOp::Operation;
        using WriteOperation = typename LastType_t<IOps...>::Operation;

        const auto& writeDF = ppLast(iOps...);

        if constexpr (sizeof...(iOps) > 1) {
            const auto tempO = ((thread | readDF) | ... | iOps).input;
            WriteOperation::exec(thread, tempO, writeDF);
        } else {
            WriteOperation::exec(thread, ReadIOp::Operation::exec(thread, readDF), writeDF);
        }
    }

    template <typename FirstIOp> FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const FirstIOp iOp) {
        return FirstIOp::Operation::getActiveThreads(iOp);
    }
};

struct SimpleTransformDPPBaseReference {
    friend struct SimpleTransformDPPReference<ParArch::GPU_NVIDIA>; // Allow TransformDPP to access private members
  private:
    template <typename ReadIOp, typename... IOps>
    FK_HOST_DEVICE_FUSE void execute_thread(const Point& thread, const ReadIOp& readDF, const IOps&... iOps) {
        using ReadOperation = typename ReadIOp::Operation;
        using WriteOperation = typename LastType_t<IOps...>::Operation;

        const auto& writeDF = ppLast(iOps...);

        if constexpr (sizeof...(iOps) > 1) {
            const auto tempO = ((thread | readDF) | ... | iOps).input;
            WriteOperation::exec(thread, tempO, writeDF);
        } else {
            WriteOperation::exec(thread, ReadIOp::Operation::exec(thread, readDF), writeDF);
        }
    }

    template <typename FirstIOp>
    FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const FirstIOp& iOp) {
        return FirstIOp::Operation::getActiveThreads(iOp);
    }
};

struct SimpleTransformDPPBaseReferenceFoldExpr {
    friend struct SimpleTransformDPPReferenceFoldExpr<ParArch::GPU_NVIDIA>; // Allow SimpleTransformDPPReferenceFoldExpr
                                                                            // to access private members
  private:
    template <typename ReadIOp, typename... IOps>
    FK_HOST_DEVICE_FUSE void execute_thread(const Point &thread, const ReadIOp &readDF, const IOps &...iOps) {
        using ReadOperation = typename ReadIOp::Operation;
        using WriteOperation = typename LastType_t<IOps...>::Operation;

        const auto& writeDF = ppLast(iOps...);

        if constexpr (sizeof...(iOps) > 1) {
            const auto tempO = ((thread | readDF) | ... | iOps);
            WriteOperation::exec(thread, tempO.input, writeDF);
        } else {
            WriteOperation::exec(thread, ReadIOp::Operation::exec(thread, readDF), writeDF);
        }
    }

    template <typename FirstIOp> FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const FirstIOp &iOp) {
        return FirstIOp::Operation::getActiveThreads(iOp);
    }
};

template <>
struct SimpleTransformDPPValue<ParArch::GPU_NVIDIA> {
  private:
    using Parent = SimpleTransformDPPBaseValue;

  public:
    static constexpr ParArch PAR_ARCH = ParArch::GPU_NVIDIA;
    template <typename FirstIOp> FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const FirstIOp iOp) {
        return Parent::getActiveThreads(iOp);
    }

    template <typename... IOps> FK_DEVICE_FUSE void exec(const IOps... iOps) {
        const int x = (blockDim.x * blockIdx.x) + threadIdx.x;
        const int y = (blockDim.y * blockIdx.y) + threadIdx.y;
        const int z = blockIdx.z;

        const Point thread{x, y, z};

        const ActiveThreads activeThreads = getActiveThreads(get_arg<0>(iOps...));

        if (x < activeThreads.x && y < activeThreads.y) {
            Parent::execute_thread(thread, iOps...);
        }
    }
};

template <>
struct SimpleTransformDPPValueLessCallDepth<ParArch::GPU_NVIDIA> {
    static constexpr ParArch PAR_ARCH = ParArch::GPU_NVIDIA;
};

template <>
struct SimpleTransformDPPReference<ParArch::GPU_NVIDIA> {
  private:
    using Parent = SimpleTransformDPPBaseReference;

  public:
    static constexpr ParArch PAR_ARCH = ParArch::GPU_NVIDIA;
    template <typename FirstIOp>
    FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const FirstIOp& iOp) {
        return Parent::getActiveThreads(iOp);
    }

    template <typename... IOps> FK_DEVICE_FUSE void exec(const IOps&... iOps) {
        const int x = (blockDim.x * blockIdx.x) + threadIdx.x;
        const int y = (blockDim.y * blockIdx.y) + threadIdx.y;
        const int z = blockIdx.z;

        const Point thread{x, y, z};

        const ActiveThreads activeThreads = getActiveThreads(get_arg<0>(iOps...));

        if (x < activeThreads.x && y < activeThreads.y) {
            Parent::execute_thread(thread, iOps...);
        }
    }
};

template <>
struct SimpleTransformDPPReferenceFoldExpr<ParArch::GPU_NVIDIA> {
  private:
    using Parent = SimpleTransformDPPBaseReferenceFoldExpr;

  public:
    static constexpr ParArch PAR_ARCH = ParArch::GPU_NVIDIA;
    template <typename FirstIOp> FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const FirstIOp &iOp) {
        return Parent::getActiveThreads(iOp);
    }

    template <typename... IOps>
    FK_DEVICE_FUSE void exec(const IOps &...iOps) {
        const int x = (blockDim.x * blockIdx.x) + threadIdx.x;
        const int y = (blockDim.y * blockIdx.y) + threadIdx.y;
        const int z = blockIdx.z;

        const Point thread{x, y, z};

        const ActiveThreads activeThreads = getActiveThreads(get_arg<0>(iOps...));

        if (x < activeThreads.x && y < activeThreads.y) {
            Parent::execute_thread(thread, iOps...);
        }
    }
};

struct InstantiableSimpleTransformDPPReference {
    static constexpr ParArch PAR_ARCH = ParArch::GPU_NVIDIA;
};

template <typename DPP_, typename... Operations>
struct InstantiableDPP {
    using DPP = DPP_;
    using OperationsTuple = Tuple<Operations...>;
    OperationsTuple ops;
};

struct SimpleTransformDPPReferenceBuilder {
    template <typename... IOps>
    FK_HOST_FUSE auto build(const IOps&... iops) {
        return InstantiableDPP<SimpleTransformDPPReferenceFoldExpr<ParArch::GPU_NVIDIA>, IOps...>{make_tuple(iops...)};
    }
};

template <typename IDPP>
__global__ void launchInstantiableDPP_Kernel(const __grid_constant__ IDPP idpp) {
    fk::apply([](auto &&...args) { return std::decay_t<IDPP>::DPP::exec(std::forward<decltype(args)>(args)...); },
              idpp.ops);
}

template <typename... IOps>
__global__ void launchSimpleTransformDPPValue_Kernel(const __grid_constant__ IOps... iOps) {
    SimpleTransformDPPValue<ParArch::GPU_NVIDIA>::exec(iOps...);
}

template <size_t N, typename T> 
FK_HOST_DEVICE_CNST T dummyCalls(const T something) { 
    if constexpr (N == 0) {
        return something;
    } else {
        return dummyCalls<N - 1, T>(something);
    }
}

template <typename I, typename P, typename O, typename ChildImplementation, bool IS_FUSED = false>
struct BinaryOperationValue {
  private:
    using SelfType = BinaryOperationValue<I, P, O, ChildImplementation, IS_FUSED>;

  public:
    FK_STATIC_STRUCT(BinaryOperationValue, SelfType)
    using Child = ChildImplementation;
    using InputType = I;
    using OutputType = O;
    using ParamsType = P;
    using InstanceType = BinaryType;
    using OperationDataType = OperationData<Child>;
    using InstantiableType = Binary<Child>;
    static constexpr bool IS_FUSED_OP = IS_FUSED;
    static constexpr int N = 1;
    FK_HOST_DEVICE_FUSE OutputType exec(const InputType input, const OperationDataType opData) {
        if constexpr (N == 0) {
            return Child::exec(input, opData.params);
        } else {
            return Child::exec(input, dummyCalls<N - 1>(opData).params);
        }
    }
    FK_HOST_FUSE InstantiableType build(const OperationDataType &opData) { return InstantiableType{opData}; }
    FK_HOST_FUSE InstantiableType build(const ParamsType &params) { return InstantiableType{{params}}; }
};

template <typename I, typename P = I, typename O = I>
struct DummyOp {
  private:
    using Self = DummyOp<I, P, O>;
    using Parent = BinaryOperationValue<I, P, O, Self>;
  public:
    using InputType = typename Parent::InputType;
    using OutputType = typename Parent::OutputType;
    using ParamsType = typename Parent::ParamsType;
    using InstanceType = typename Parent::InstanceType;
    using OperationDataType = typename Parent::OperationDataType;
    using InstantiableType = typename Parent::InstantiableType;
    static constexpr bool IS_FUSED_OP = Parent::IS_FUSED_OP;

    FK_HOST_DEVICE_FUSE OutputType exec(const InputType input, const OperationDataType opData) {
        return Parent::exec(input, opData);
    }

    static constexpr inline InstantiableType build(const OperationDataType &opData) { return Parent::build(opData); }
    static constexpr inline InstantiableType build(const ParamsType &params) { return Parent::build(params); }
    
    FK_HOST_DEVICE_FUSE O exec(const I input, const P params) { 
        return static_cast<O>(input + params);
    }
};

template <typename ReadIOp, typename... IOps>
__global__ void launchSimpleTransformDPPValueLessCallDepth_Kernel(const __grid_constant__ ReadIOp readIOp, const __grid_constant__ IOps... iOps) {
    const int x = (blockDim.x * blockIdx.x) + threadIdx.x;
    const int y = (blockDim.y * blockIdx.y) + threadIdx.y;
    const int z = blockIdx.z;

    const Point thread{x, y, z};

    const ActiveThreads activeThreads = ReadIOp::Operation::getActiveThreads(readIOp);

    if (x < activeThreads.x && y < activeThreads.y) {
        using ReadOperation = typename ReadIOp::Operation;
        using WriteOperation = typename LastType_t<IOps...>::Operation;

        const auto writeIOp = ppLast(iOps...);
        const auto readIOpTemp = dummyCalls<0>(readIOp);

        if constexpr (sizeof...(iOps) > 1) {
            const auto tempO = ((thread | readIOpTemp) | ... | iOps).input;
            WriteOperation::exec(thread, tempO, writeIOp);
        } else {
            WriteOperation::exec(thread, (thread | readIOpTemp).input, writeIOp);
        }
    }
}

template <typename... IOps>
__global__ void launchSimpleTransformDPPReference_Kernel(const __grid_constant__ IOps... iOps) {
    SimpleTransformDPPReference<ParArch::GPU_NVIDIA>::exec(iOps...);
}

template <typename... IOps>
__global__ void launchSimpleTransformDPPReferenceFoldExpr_Kernel(const __grid_constant__ IOps... iOps) {
    SimpleTransformDPPReferenceFoldExpr<ParArch::GPU_NVIDIA>::exec(iOps...);
}

template <>
struct Executor<InstantiableSimpleTransformDPPReference> {
  private:
    using Child = Executor<InstantiableSimpleTransformDPPReference>;
    using Parent = BaseExecutor<Child>;
    template <typename... IOps>
    FK_HOST_FUSE void executeOperations_helper(Stream_<ParArch::GPU_NVIDIA> &stream_, const IOps &...iOps) {
        const cudaStream_t stream = stream_.getCUDAStream();

        const auto idpp = SimpleTransformDPPReferenceBuilder::build(iOps...);

        const auto readOp = get<0>(idpp.ops);

        const ActiveThreads activeThreads = readOp.getActiveThreads();

        const CtxDim3 ctx_block = getDefaultBlockSize(activeThreads.x, activeThreads.y);

        const dim3 block{ctx_block.x, ctx_block.y, 1};
        const dim3 grid{static_cast<uint>(ceil(activeThreads.x / static_cast<float>(block.x))),
                        static_cast<uint>(ceil(activeThreads.y / static_cast<float>(block.y))), activeThreads.z};
        launchInstantiableDPP_Kernel<<<grid, block, 0, stream>>>(idpp);
        gpuErrchk(cudaGetLastError());
    }

  public:
    FK_STATIC_STRUCT(Executor, Child)
    FK_HOST_FUSE ParArch parArch() { return ParArch::GPU_NVIDIA; }
    DECLARE_EXECUTOR_PARENT_IMPL
};

template <>
struct Executor<SimpleTransformDPPValue<ParArch::GPU_NVIDIA>> {
  private:
    using Child = Executor<SimpleTransformDPPValue<ParArch::GPU_NVIDIA>>;
    using Parent = BaseExecutor<Child>;
    template <typename... IOps>
    FK_HOST_FUSE void executeOperations_helper(Stream_<ParArch::GPU_NVIDIA> &stream_, const IOps &...iOps) {
        const cudaStream_t stream = stream_.getCUDAStream();
        
        const auto readOp = get_arg<0>(iOps...);

        const ActiveThreads activeThreads = readOp.getActiveThreads();

        const CtxDim3 ctx_block = getDefaultBlockSize(activeThreads.x, activeThreads.y);

        const dim3 block{ctx_block.x, ctx_block.y, 1};
        const dim3 grid{static_cast<uint>(ceil(activeThreads.x / static_cast<float>(block.x))),
                        static_cast<uint>(ceil(activeThreads.y / static_cast<float>(block.y))), activeThreads.z};
        launchSimpleTransformDPPValue_Kernel<<<grid, block, 0, stream>>>(iOps...);
        gpuErrchk(cudaGetLastError());
        
    }

  public:
    FK_STATIC_STRUCT(Executor, Child)
    FK_HOST_FUSE ParArch parArch() { return ParArch::GPU_NVIDIA; }
    DECLARE_EXECUTOR_PARENT_IMPL
};

template <>
struct Executor<SimpleTransformDPPValueLessCallDepth<ParArch::GPU_NVIDIA>> {
  private:
    using Child = Executor<SimpleTransformDPPValueLessCallDepth<ParArch::GPU_NVIDIA>>;
    using Parent = BaseExecutor<Child>;
    template <typename... IOps>
    FK_HOST_FUSE void executeOperations_helper(Stream_<ParArch::GPU_NVIDIA> &stream_, const IOps &...iOps) {
        const cudaStream_t stream = stream_.getCUDAStream();

        const auto& readOp = get_arg<0>(iOps...);

        const ActiveThreads activeThreads = readOp.getActiveThreads();

        const CtxDim3 ctx_block = getDefaultBlockSize(activeThreads.x, activeThreads.y);

        const dim3 block{ctx_block.x, ctx_block.y, 1};
        const dim3 grid{static_cast<uint>(ceil(activeThreads.x / static_cast<float>(block.x))),
                        static_cast<uint>(ceil(activeThreads.y / static_cast<float>(block.y))), activeThreads.z};
        launchSimpleTransformDPPValueLessCallDepth_Kernel<<<grid, block, 0, stream>>>(iOps...);
        gpuErrchk(cudaGetLastError());
    }

  public:
    FK_STATIC_STRUCT(Executor, Child)
    FK_HOST_FUSE ParArch parArch() { return ParArch::GPU_NVIDIA; }
    DECLARE_EXECUTOR_PARENT_IMPL
};

template <>
struct Executor<SimpleTransformDPPReference<ParArch::GPU_NVIDIA>> {
  private:
    using Child = Executor<SimpleTransformDPPReference<ParArch::GPU_NVIDIA>>;
    using Parent = BaseExecutor<Child>;
    template <typename... IOps>
    FK_HOST_FUSE void executeOperations_helper(Stream_<ParArch::GPU_NVIDIA> &stream_, const IOps &...iOps) {
        const cudaStream_t stream = stream_.getCUDAStream();

        const auto readOp = get_arg<0>(iOps...);

        const ActiveThreads activeThreads = readOp.getActiveThreads();

        const CtxDim3 ctx_block = getDefaultBlockSize(activeThreads.x, activeThreads.y);

        const dim3 block{ctx_block.x, ctx_block.y, 1};
        const dim3 grid{static_cast<uint>(ceil(activeThreads.x / static_cast<float>(block.x))),
                        static_cast<uint>(ceil(activeThreads.y / static_cast<float>(block.y))), activeThreads.z};
        launchSimpleTransformDPPReference_Kernel<<<grid, block, 0, stream>>>(iOps...);
        gpuErrchk(cudaGetLastError());
    }

  public:
    FK_STATIC_STRUCT(Executor, Child)
    FK_HOST_FUSE ParArch parArch() { return ParArch::GPU_NVIDIA; }
    DECLARE_EXECUTOR_PARENT_IMPL
};

template <>
struct Executor<SimpleTransformDPPReferenceFoldExpr<ParArch::GPU_NVIDIA>> {
  private:
    using Child = Executor<SimpleTransformDPPReferenceFoldExpr<ParArch::GPU_NVIDIA>>;
    using Parent = BaseExecutor<Child>;
    template <typename... IOps>
    FK_HOST_FUSE void executeOperations_helper(Stream_<ParArch::GPU_NVIDIA> &stream_, const IOps &...iOps) {
        const cudaStream_t stream = stream_.getCUDAStream();

        const auto readOp = get_arg<0>(iOps...);

        const ActiveThreads activeThreads = readOp.getActiveThreads();

        const CtxDim3 ctx_block = getDefaultBlockSize(activeThreads.x, activeThreads.y);

        const dim3 block{ctx_block.x, ctx_block.y, 1};
        const dim3 grid{static_cast<uint>(ceil(activeThreads.x / static_cast<float>(block.x))),
                        static_cast<uint>(ceil(activeThreads.y / static_cast<float>(block.y))), activeThreads.z};
        launchSimpleTransformDPPReferenceFoldExpr_Kernel<<<grid, block, 0, stream>>>(iOps...);
        gpuErrchk(cudaGetLastError());
    }

  public:
    FK_STATIC_STRUCT(Executor, Child)
    FK_HOST_FUSE ParArch parArch() { return ParArch::GPU_NVIDIA; }
    DECLARE_EXECUTOR_PARENT_IMPL
};

void testCompareReferenceVSValueVSInstantiableDPP() {
    Stream stream;

    // We set all outputs to the same size
    const Size outputSize(128, 128);

    // We perform 5 crops on the image
    constexpr int BATCH = 200;

    // We have a 4K source image
    Ptr2D<uchar3> inputImage(3840, 2160);
    Ptr2D<float3> outputImage(3840, 2160);

    // We want a Tensor of contiguous memory for all images
    Tensor<float3> output(outputSize.width, outputSize.height, BATCH);

    // Crops can be of different sizes
    std::array<Rect, 10> crops10{Rect(0, 0, 34, 25),      Rect(40, 40, 70, 15),     Rect(100, 200, 60, 59),
                                  Rect(300, 1000, 20, 23), Rect(3000, 2000, 12, 11), Rect(0, 0, 34, 25),
                                  Rect(40, 40, 70, 15),    Rect(100, 200, 60, 59),   Rect(300, 1000, 20, 23),
                                  Rect(3000, 2000, 12, 11)};
    std::array<Rect, BATCH> crops{};
    for (int i = 0; i < BATCH; ++i) {
        crops[i] = crops10[i % 10];
    }

    // initImageValues(inputImage);
    const float3 backgroundColor{0.f, 0.f, 0.f};
    const float3 mulValue = make_set<float3>(1.4f);
    const float3 subValue = make_set<float3>(0.5f);
    const float3 divValue = make_set<float3>(255.f);

    // Create the operation instances once, and use them multiple times
    const auto readIOp = PerThreadRead<ND::_2D, uchar3>::build(inputImage);
    const auto cropIOp = Crop<>::build(crops);
    const auto resizeIOp =
        Resize<InterpolationType::INTER_LINEAR, AspectRatio::PRESERVE_AR>::build(outputSize, backgroundColor);
    const auto mulIOp = Mul<float3>::build(mulValue);
    const auto subIOp = Sub<float3>::build(subValue);
    const auto divIOp = Div<float3>::build(divValue);
    const auto colorIOp = ColorConversion<ColorConversionCodes::COLOR_RGB2BGR, float3, float3>::build();
    const auto tensorWriteIOp = TensorWrite<float3>::build(output);

    // Execute the operations in a single kernel
    // At compile time, the types are used to define the kernel code
    // At runtime, the kernel is executed with the provided parameters

    // First, execute the operations with a transform that passes all parameters as const &
    executeOperations<SimpleTransformDPPReference<>>(
        stream, readIOp, cropIOp, resizeIOp, mulIOp, subIOp, divIOp, colorIOp, tensorWriteIOp);

    // Second, execute the same operaitions with a transform that passes all parameters by value
    executeOperations<SimpleTransformDPPValue<>>(
        stream, readIOp, cropIOp, resizeIOp, mulIOp, subIOp, divIOp, colorIOp, tensorWriteIOp);

    // Third, execute the same operations with a transform that uses an instantiable transform that passes
    // all parameters as const &, but stores the operations in a tuple inside the instantiable transform,
    // and passes that transform instance to the CUDA kernel.
    executeOperations<InstantiableSimpleTransformDPPReference>(
        stream, readIOp, cropIOp, resizeIOp, mulIOp, subIOp, divIOp, colorIOp, tensorWriteIOp);

    // Fourth, test the fold expression solution to expand the IOp parameter pack
    executeOperations<SimpleTransformDPPReferenceFoldExpr<>>(stream, readIOp, cropIOp, resizeIOp, mulIOp, subIOp,
                                                               divIOp, colorIOp, tensorWriteIOp);

    // Fifth, test (partial) pass by value, with less device function call depth
    const auto dummyIOp = DummyOp<uchar3, float3, float3>::build(make_set<float3>(2.f));
    using DummyOpType = typename std::decay_t<decltype(dummyIOp)>::Operation;
    static_assert(std::is_same_v<typename DummyOpType::OutputType, float3>, "Not float3");
    auto result = (Point(0, 0, 0) | readIOp.then(cropIOp) | dummyIOp);
    using ResultType = decltype(result.input);
    static_assert(std::is_same_v<ResultType, float3>, "Not float3");
    executeOperations<SimpleTransformDPPValueLessCallDepth<>>(stream, readIOp.then(cropIOp), Cast<uchar3, float3>::build(), mulIOp,
                                                              subIOp, divIOp, colorIOp, tensorWriteIOp);                                                      
        //PerThreadWrite<ND::_2D, float3>::build(outputImage));

    stream.sync();
}

int launch() {
    testCompareReferenceVSValueVSInstantiableDPP();

    return 0; 
}