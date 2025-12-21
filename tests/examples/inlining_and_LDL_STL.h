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
template <ParArch PA = defaultParArch> struct SimpleTransformDPPReference;

struct SimpleTransformDPPBaseValue {
    friend struct SimpleTransformDPPValue<ParArch::GPU_NVIDIA>; // Allow TransformDPP to access private members
  private:
    template <typename T, typename IOp, typename... IOpTypes>
    FK_HOST_DEVICE_FUSE auto operate(const Point& thread, const T& i_data, const IOp& iOp,
                                     const IOpTypes&... iOpInstances) {
        static_assert(!isIncompleteReadBackType<IOp>, "Trying to execute an incomplete IOp");
        if constexpr (IOp::template is<WriteType>) {
            return i_data;
            // MidWriteOperation with continuations, based on FusedOperation
        } else if constexpr (IOp::template is<MidWriteType> && isMidWriteType<typename IOp::Operation>) {
            return IOp::Operation::exec(thread, i_data, iOp);
        } else if constexpr (IOp::template is<MidWriteType> && !isMidWriteType<typename IOp::Operation>) {
            IOp::Operation::exec(thread, i_data, iOp);
            return i_data;
        } else {
            return operate(thread, compute(i_data, iOp), iOpInstances...);
        }
    }

    template <typename ReadIOp, typename... IOps>
    FK_HOST_DEVICE_FUSE void execute_thread(const Point thread, const ReadIOp readDF, const IOps... iOps) {
        using ReadOperation = typename ReadIOp::Operation;
        using WriteOperation = typename LastType_t<IOps...>::Operation;

        const auto writeDF = ppLast(iOps...);

        const auto tempI = ReadIOp::Operation::exec(thread, readDF);
        if constexpr (sizeof...(iOps) > 1) {
            const auto tempO = operate(thread, tempI, iOps...);
            WriteOperation::exec(thread, tempO, writeDF);
        } else {
            WriteOperation::exec(thread, tempI, writeDF);
        }
    }

    template <typename FirstIOp> FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const FirstIOp& iOp) {
        return FirstIOp::Operation::getActiveThreads(iOp);
    }
};

struct SimpleTransformDPPBaseReference {
    friend struct SimpleTransformDPPReference<ParArch::GPU_NVIDIA>; // Allow TransformDPP to access private members
  private:
    template <typename T, typename IOp, typename... IOpTypes>
    FK_HOST_DEVICE_FUSE auto operate(const Point& thread, const T& i_data, const IOp& iOp,
                                     const IOpTypes&... iOpInstances) {
        static_assert(!isIncompleteReadBackType<IOp>, "Trying to execute an incomplete IOp");
        if constexpr (IOp::template is<WriteType>) {
            return i_data;
            // MidWriteOperation with continuations, based on FusedOperation
        } else if constexpr (IOp::template is<MidWriteType> && isMidWriteType<typename IOp::Operation>) {
            return IOp::Operation::exec(thread, i_data, iOp);
        } else if constexpr (IOp::template is<MidWriteType> && !isMidWriteType<typename IOp::Operation>) {
            IOp::Operation::exec(thread, i_data, iOp);
            return i_data;
        } else {
            return operate(thread, compute(i_data, iOp), iOpInstances...);
        }
    }

    template <typename ReadIOp, typename... IOps>
    FK_HOST_DEVICE_FUSE void execute_thread(const Point& thread, const ReadIOp& readDF, const IOps&... iOps) {
        using ReadOperation = typename ReadIOp::Operation;
        using WriteOperation = typename LastType_t<IOps...>::Operation;

        const auto writeDF = ppLast(iOps...);

        const auto tempI = ReadIOp::Operation::exec(thread, readDF);
        if constexpr (sizeof...(iOps) > 1) {
            const auto tempO = operate(thread, tempI, iOps...);
            WriteOperation::exec(thread, tempO, writeDF);
        } else {
            WriteOperation::exec(thread, tempI, writeDF);
        }
    }

    template <typename FirstIOp>
    FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const FirstIOp& iOp) {
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

        const ActiveThreads activeThreads = getActiveThreads(get<0>(iOps...));

        if (x < activeThreads.x && y < activeThreads.y) {
            Parent::execute_thread(thread, iOps...);
        }
    }
};

template <> struct SimpleTransformDPPReference<ParArch::GPU_NVIDIA> {
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

        const ActiveThreads activeThreads = getActiveThreads(get<0>(iOps...));

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
        return InstantiableDPP<SimpleTransformDPPReference<ParArch::GPU_NVIDIA>, IOps...>{make_tuple(iops...)};
    }
};

template <typename IDPP, size_t... Idx>
FK_DEVICE_CNST void exec_helper(const IDPP& idpp, const std::index_sequence<Idx...>&) {
    IDPP::DPP::exec(fk::get<Idx>(idpp.ops)...);
}

template <typename IDPP>
__global__ void launchInstantiableDPP_Kernel(const __grid_constant__ IDPP idpp) {
    constexpr std::make_index_sequence<IDPP::OperationsTuple::size> seq{};
    exec_helper(idpp, seq);
}

template <typename... IOps>
__global__ void launchSimpleTransformDPPValue_Kernel(const __grid_constant__ IOps... iOps) {
    SimpleTransformDPPValue<ParArch::GPU_NVIDIA>::exec(iOps...);
}

template <typename... IOps>
__global__ void launchSimpleTransformDPPReference_Kernel(const __grid_constant__ IOps... iOps) {
    SimpleTransformDPPReference<ParArch::GPU_NVIDIA>::exec(iOps...);
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
        
        const auto readOp = get<0>(iOps...);

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
struct Executor<SimpleTransformDPPReference<ParArch::GPU_NVIDIA>> {
  private:
    using Child = Executor<SimpleTransformDPPReference<ParArch::GPU_NVIDIA>>;
    using Parent = BaseExecutor<Child>;
    template <typename... IOps>
    FK_HOST_FUSE void executeOperations_helper(Stream_<ParArch::GPU_NVIDIA> &stream_, const IOps &...iOps) {
        const cudaStream_t stream = stream_.getCUDAStream();

        const auto readOp = get<0>(iOps...);

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

void testCompareReferenceVSValueVSInstantiableDPP() {
    Stream stream;

    // We set all outputs to the same size
    const Size outputSize(60, 60);

    // We perform 5 crops on the image
    constexpr int BATCH = 10;

    // We have a 4K source image
    Ptr2D<uchar3> inputImage(3840, 2160);

    // We want a Tensor of contiguous memory for all images
    Tensor<float3> output(outputSize.width, outputSize.height, BATCH);

    // Crops can be of different sizes
    std::array<Rect, BATCH> crops{Rect(0, 0, 34, 25),      Rect(40, 40, 70, 15),     Rect(100, 200, 60, 59),
                                  Rect(300, 1000, 20, 23), Rect(3000, 2000, 12, 11), Rect(0, 0, 34, 25),
                                  Rect(40, 40, 70, 15),    Rect(100, 200, 60, 59),   Rect(300, 1000, 20, 23),
                                  Rect(3000, 2000, 12, 11)};

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

    stream.sync();
}

int launch() {
    testCompareReferenceVSValueVSInstantiableDPP();

    return 0; 
}