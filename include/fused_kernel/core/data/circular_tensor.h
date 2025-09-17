/* Copyright 2023-2025 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.*/

#ifndef FK_CIRCULAR_TENSOR
#define FK_CIRCULAR_TENSOR

#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/execution_model/executors.h>
#include <fused_kernel/core/execution_model/memory_operations.h>

namespace fk {

    enum class CircularTensorOrder { NewestFirst, OldestFirst };

    template <CircularTensorOrder CTO, int BATCH>
    struct SequenceSelectorType {
        FK_HOST_DEVICE_FUSE uint at(const uint& index) {
            if constexpr (CTO == CircularTensorOrder::NewestFirst) {
                return index > 0 ? 2u : 1u;
            } else {
                return index != BATCH - 1 ? 2u : 1u;
            }
        }
    };

    template <CircularTensorOrder CT_ORDER>
    struct CTReadDirection;

    template <>
    struct CTReadDirection<CircularTensorOrder::NewestFirst> {
        static const CircularDirection dir{ CircularDirection::Descendent };
    };

    template <>
    struct CTReadDirection<CircularTensorOrder::OldestFirst> {
        static const CircularDirection dir{ CircularDirection::Ascendent };
    };

    template <CircularTensorOrder CT_ORDER>
    static constexpr CircularDirection CTReadDirection_v = CTReadDirection<CT_ORDER>::dir;

    enum class ColorPlanes { Standard, Transposed };

    template <typename T, ColorPlanes CP_MODE>
    struct CoreType;

    template <typename T>
    struct CoreType<T, ColorPlanes::Standard> {
        using type = Tensor<T>;
    };

    template <typename T>
    struct CoreType<T, ColorPlanes::Transposed> {
        using type = TensorT<T>;
    };

    template <typename T, ColorPlanes CP_MODE>
    using CoreType_t = typename CoreType<T, CP_MODE>::type;

    template <typename T, int COLOR_PLANES, typename Enabler = void>
    struct CircularTensorStoreType {};

    template <typename T, int COLOR_PLANES>
    struct CircularTensorStoreType<T, COLOR_PLANES, std::enable_if_t<std::is_aggregate_v<T>&& COLOR_PLANES == 1>> {
        using type = T;
    };

    template <typename T, int COLOR_PLANES>
    struct CircularTensorStoreType<T, COLOR_PLANES, std::enable_if_t<!std::is_aggregate_v<T> && (COLOR_PLANES > 1)>> {
        using type = VectorType_t<T, COLOR_PLANES>;
    };

    template <typename T, int COLOR_PLANES, int BATCH, CircularTensorOrder CT_ORDER, ColorPlanes CP_MODE>
    class CircularTensor : public CoreType_t<T, CP_MODE> {

        using ParentType = CoreType_t<T, CP_MODE>;

        using StoreT = typename CircularTensorStoreType<T, COLOR_PLANES>::type;

        using WriteInstantiableOperations = TypeList<Write<TensorWrite<StoreT>>,
            Write<TensorSplit<StoreT>>,
            Write<TensorTSplit<StoreT>>>;

        using ReadInstantiableOperations = TypeList<Read<CircularTensorRead<CTReadDirection_v<CT_ORDER>, TensorRead<StoreT>, BATCH>>,
            Read<CircularTensorRead<CTReadDirection_v<CT_ORDER>, TensorPack<StoreT>, BATCH>>,
            Read<CircularTensorRead<CTReadDirection_v<CT_ORDER>, TensorTPack<StoreT>, BATCH>>>;

    public:
        FK_HOST_CNST CircularTensor() {};

        FK_HOST_CNST CircularTensor(const uint& width_, const uint& height_, const MemType& type_ = defaultMemType, const int& deviceID_ = 0) :
            ParentType(width_, height_, BATCH, COLOR_PLANES, type_, deviceID_),
            m_tempTensor(width_, height_, BATCH, COLOR_PLANES, type_, deviceID_) {};

        FK_HOST_CNST void Alloc(const uint& width_, const uint& height_, const MemType& type_ = defaultMemType, const int& deviceID_ = 0) {
            this->allocTensor(width_, height_, BATCH, COLOR_PLANES, type_, deviceID_);
            m_tempTensor.allocTensor(width_, height_, BATCH, COLOR_PLANES, type_, deviceID_);
        }

        template <typename... IOpTypes>
        FK_HOST_CNST void update(const Stream& stream,
                                 const IOpTypes&... instantiableOperationInstances) {
            const auto writeInstantiableOperation = ppLast(instantiableOperationInstances...);
            using writeDFType = std::decay_t<decltype(writeInstantiableOperation)>;
            using writeOpType = typename writeDFType::Operation;
            if constexpr (CP_MODE == ColorPlanes::Transposed) {
                static_assert(std::is_same_v<writeDFType, Write<TensorTSplit<StoreT>>>,
                    "Need to use TensorTSplitWrite as write function because you are using a transposed CircularTensor (CP_MODE = Transposed)");
            }
            using equivalentReadDFType = EquivalentType_t<writeDFType, WriteInstantiableOperations, ReadInstantiableOperations>;

            MidWrite<CircularTensorWrite<CircularDirection::Ascendent, writeOpType, BATCH>> updateWriteToTemp;
            updateWriteToTemp.params.first = m_nextUpdateIdx;
            updateWriteToTemp.params.opData.params = m_tempTensor.ptr();

            const auto updateOps = buildOperationSequence_tup(insert_before_last(updateWriteToTemp, instantiableOperationInstances...));

            // Build copy pipeline
            equivalentReadDFType nonUpdateRead;
            nonUpdateRead.params.first = m_nextUpdateIdx;
            nonUpdateRead.params.opData.params = m_tempTensor.ptr();

            const auto copyOps = buildOperationSequence(nonUpdateRead, writeInstantiableOperation);

            /*const dim3 block(std::min(static_cast<int>(this->ptr_a.dims.width), 32),
                             std::min(static_cast<int>(this->ptr_a.dims.height), 8));
            const dim3 grid((uint)ceil((float)this->ptr_a.dims.width / static_cast<float>(block.x)),
                            (uint)ceil((float)this->ptr_a.dims.height / static_cast<float>(block.y)),
                            BATCH);

            launchDivergentBatchTransformDPP_Kernel<ParArch::GPU_NVIDIA, SequenceSelectorType<CT_ORDER, BATCH>><<<grid, block, 0, stream>>>(updateOps, copyOps);*/

            if (this->type == MemType::Device || this->type == MemType::DeviceAndPinned) {
#if defined(__NVCC__) || CLANG_HOST_DEVICE 
                Executor<DivergentBatchTransformDPP<ParArch::GPU_NVIDIA, SequenceSelectorType<CT_ORDER, BATCH>>>::executeOperations(stream, static_cast<uint>(BATCH), updateOps, copyOps);
                gpuErrchk(cudaGetLastError());
#else
                throw std::runtime_error("CircularTensor operations on Device memory only supported in nvcc or hipcc compilation.");
#endif
            } else {
                Executor<DivergentBatchTransformDPP<ParArch::CPU, SequenceSelectorType<CT_ORDER, BATCH>>>::executeOperations(stream, static_cast<uint>(BATCH), updateOps, copyOps);
            }

            m_nextUpdateIdx = (m_nextUpdateIdx + 1) % BATCH;
        }

    private:
        CoreType_t<T, CP_MODE> m_tempTensor;
        int m_nextUpdateIdx{ 0 };
    };
} // namespace fk

#endif
