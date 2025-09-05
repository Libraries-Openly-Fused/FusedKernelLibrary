/* Copyright 2023-2025 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_MEMORY_OPERATIONS
#define FK_MEMORY_OPERATIONS

#include <fused_kernel/core/data/size.h>
#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/execution_model/thread_fusion.h>
#include <fused_kernel/core/execution_model/operation_model/operation_model.h>
#include <fused_kernel/core/data/array.h>

#if !defined(NVRTC_COMPILER)
#include <vector>
#else
namespace std {
    template <typename T>
    class vector;
}
#endif

namespace fk {
    template <ND D, typename T>
    struct PerThreadRead {
    private:
        using Parent = ReadOperation<T, RawPtr<D, T>, T, TF::ENABLED, PerThreadRead<D, T>>;
        using SelfType = PerThreadRead<D, T>;
    public:
        FK_STATIC_STRUCT(PerThreadRead, SelfType)
        DECLARE_READ_PARENT
        template <uint ELEMS_PER_THREAD=1>
        FK_HOST_DEVICE_FUSE ThreadFusionType<ReadDataType, ELEMS_PER_THREAD, OutputType>
        exec(const Point& thread, const ParamsType& params) {
            return *PtrAccessor<D>::template cr_point<T, ThreadFusionType<ReadDataType, ELEMS_PER_THREAD, OutputType>>(thread, params);
        }

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return opData.params.dims.width;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const OperationDataType& opData) {
            if constexpr (D == ND::_1D) {
                return 1;
            } else {
                return opData.params.dims.height;
            }
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const OperationDataType& opData) {
            if constexpr (D == ND::_1D || D == ND::_2D) {
                return 1;
            } else {
                return opData.params.dims.planes;
            }
        }

        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const OperationDataType& opData) {
            return opData.params.dims.pitch;
        }

        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
            return { num_elems_x(Point(), opData), num_elems_y(Point(), opData), num_elems_z(Point(), opData) };
        }
    };

    template <enum ND D, typename T>
    struct PerThreadWrite {
    private:
        using Parent = WriteOperation<T, RawPtr<D, T>, T, TF::ENABLED, PerThreadWrite<D, T>>;
        using SelfType = PerThreadWrite<D, T>;
    public:
        FK_STATIC_STRUCT(PerThreadWrite, SelfType)
        DECLARE_WRITE_PARENT
        template <uint ELEMS_PER_THREAD = 1>
        FK_HOST_DEVICE_FUSE void exec(const Point& thread,
                                      const ThreadFusionType<T, ELEMS_PER_THREAD, T>& input,
                                      const ParamsType& params) {
            *PtrAccessor<D>::template point<T, ThreadFusionType<T, ELEMS_PER_THREAD, T>>(thread, params) = input;
        }
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return opData.params.dims.width;
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const OperationDataType& opData) {
            return opData.params.dims.pitch;
        }
        FK_HOST_FUSE InstantiableType build(const Ptr<D, T>& ptr) {
            return InstantiableType{ {ptr.ptr()} };
        }
    };

    template <typename T>
    struct TensorRead {
    private:
        using Parent = ReadOperation<T, RawPtr<ND::_3D, T>, T, TF::ENABLED, TensorRead<T>>;
        using SelfType = TensorRead<T>;
    public:
        FK_STATIC_STRUCT(TensorRead, SelfType)
        DECLARE_READ_PARENT

        template <uint ELEMS_PER_THREAD = 1>
        FK_HOST_DEVICE_FUSE ThreadFusionType<ReadDataType, ELEMS_PER_THREAD, OutputType> exec(const Point& thread, const ParamsType& params) {
            return *PtrAccessor<ND::_3D>::template cr_point<T, ThreadFusionType<ReadDataType, ELEMS_PER_THREAD, OutputType>>(thread, params);
        }
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return opData.params.dims.width;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const OperationDataType& opData) {
            return opData.params.dims.height;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const OperationDataType& opData) {
            return opData.params.dims.planes;
        }

        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const OperationDataType& opData) {
            return opData.params.dims.pitch;
        }

        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
            return { num_elems_x(Point(), opData), num_elems_y(Point(), opData), num_elems_z(Point(), opData) };
        }
    };

    template <typename T>
    struct TensorWrite {
    private:
        using Parent = WriteOperation<T, RawPtr<ND::_3D, T>, T, TF::ENABLED, TensorWrite<T>>;
        using SelfType = TensorWrite<T>;
    public:
        FK_STATIC_STRUCT(TensorWrite, SelfType)
        DECLARE_WRITE_PARENT
        template <uint ELEMS_PER_THREAD = 1>
        FK_HOST_DEVICE_FUSE void exec(const Point& thread, const ThreadFusionType<InputType, ELEMS_PER_THREAD, InputType>& input, const ParamsType& params) {
            *PtrAccessor<ND::_3D>::template point<T, ThreadFusionType<InputType, ELEMS_PER_THREAD, InputType>>(thread, params) = input;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return opData.params.dims.width;
        }

        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const OperationDataType& opData) {
            return opData.params.dims.pitch;
        }
    };

    template <typename T>
    struct TensorSplit {
    private:
        using Parent = WriteOperation<T, RawPtr<ND::_3D, VBase<T>>, VBase<T>, TF::DISABLED, TensorSplit<T>>;
        using SelfType = TensorSplit<T>;
    public:
        FK_STATIC_STRUCT(TensorSplit, SelfType)
        DECLARE_WRITE_PARENT
        FK_HOST_DEVICE_FUSE void exec(const Point& thread, const T& input, const ParamsType& params) {
            static_assert(cn<InputType> >= 2,
                          "Wrong type for split tensor write. It must be one of <type>2, <type>3 or <type>4.");

            const int planePixels = params.dims.width * params.dims.height;

            WriteDataType* const work_plane = PtrAccessor<ND::_3D>::point(thread, params);

            *work_plane = input.x;
            *(work_plane + planePixels) = input.y;
            if constexpr (cn<InputType> >= 3) {
                *(work_plane + (planePixels * 2)) = input.z;
            }
            if constexpr (cn<InputType> == 4) {
                *(work_plane + (planePixels * 3)) = input.w;
            }
        }
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return opData.params.dims.width;
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const OperationDataType& opData) {
            return opData.params.dims.pitch;
        }
    };

    template <typename T>
    struct TensorTSplit {
    private:
        using Parent = WriteOperation<T, RawPtr<ND::T3D, VBase<T>>, VBase<T>, TF::DISABLED, TensorTSplit<T>>;
        using SelfType = TensorTSplit<T>;
    public:
        FK_STATIC_STRUCT(TensorTSplit, SelfType)
        DECLARE_WRITE_PARENT
        FK_HOST_DEVICE_FUSE void exec(const Point& thread, const InputType& input, const ParamsType& params) {
            static_assert(cn<InputType> >= 2,
                          "Wrong type for split tensor write. It must be one of <type>2, <type>3 or <type>4.");

            *PtrAccessor<ND::T3D>::point(thread, params, 0) = input.x;
            *PtrAccessor<ND::T3D>::point(thread, params, 1) = input.y;
            if constexpr (cn<InputType> >= 3) {
                *PtrAccessor<ND::T3D>::point(thread, params, 2) = input.z;
            }
            if constexpr (cn<InputType> == 4) {
                *PtrAccessor<ND::T3D>::point(thread, params, 3) = input.w;
            }
        }
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return opData.params.dims.width;
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const OperationDataType& opData) {
            return opData.params.dims.pitch;
        }
    };

    template <typename T>
    struct TensorPack {
    private:
        using Parent = ReadOperation<VBase<T>, RawPtr<ND::_3D, VBase<T>>, T, TF::DISABLED, TensorPack<T>>;
        using SelfType = TensorPack<T>;
    public:
        FK_STATIC_STRUCT(TensorPack, SelfType)
        DECLARE_READ_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const ParamsType& params) {
            static_assert(cn<OutputType> >= 2,
                          "Wrong type for split tensor read. It must be one of <type>2, <type>3 or <type>4.");

            const int planePixels = params.dims.width * params.dims.height;

            const ReadDataType* const work_plane = PtrAccessor<ND::_3D>::cr_point(thread, params);
            if constexpr (cn<OutputType> == 2) {
                return make_<OutputType>(*work_plane, *(work_plane + planePixels));
            } else if constexpr (cn<OutputType> == 3) {
                return make_<OutputType>(*work_plane, *(work_plane + planePixels),
                    *(work_plane + (planePixels * 2)));
            } else {
                return make_<OutputType>(*work_plane,
                    *(work_plane + planePixels),
                    *(work_plane + (planePixels * 2)),
                    *(work_plane + (planePixels * 3)));
            }
        }

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return opData.params.dims.width;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const OperationDataType& opData) {
            return opData.params.dims.height;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const OperationDataType& opData) {
            return opData.params.dims.planes;
        }

        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const OperationDataType& opData) {
            return opData.params.dims.pitch;
        }

        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
            return { num_elems_x(Point(), opData), num_elems_y(Point(), opData), num_elems_z(Point(), opData) };
        }
    };

    template <typename T>
    struct TensorTPack {
    private:
        using Parent = ReadOperation<T, RawPtr<ND::T3D, VBase<T>>, T, TF::DISABLED, TensorTPack<T>>;
        using SelfType = TensorTPack<T>;
    public:
        FK_STATIC_STRUCT(TensorTPack, SelfType)
        DECLARE_READ_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const ParamsType& params) {
            static_assert(cn<OutputType> >= 2,
                          "Wrong type for split tensor read. It must be one of <type>2, <type>3 or <type>4.");

            const VBase<T> x = *PtrAccessor<ND::T3D>::cr_point(thread, params, 0);
            if constexpr (cn<OutputType> == 2) {
                const VBase<T> y = *PtrAccessor<ND::T3D>::cr_point(thread, params, 1);
                return make_<OutputType>(x, y);
            } else if constexpr (cn<OutputType> == 3) {
                const VBase<T> y = *PtrAccessor<ND::T3D>::cr_point(thread, params, 1);
                const VBase<T> z = *PtrAccessor<ND::T3D>::cr_point(thread, params, 2);
                return make_<OutputType>(x, y, z);
            } else {
                const VBase<T> y = *PtrAccessor<ND::T3D>::cr_point(thread, params, 1);
                const VBase<T> z = *PtrAccessor<ND::T3D>::cr_point(thread, params, 2);
                const VBase<T> w = *PtrAccessor<ND::T3D>::cr_point(thread, params, 3);
                return make_<OutputType>(x, y, z, w);
            }
        }
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return opData.params.dims.width;
        }
        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const OperationDataType& opData) {
            return opData.params.dims.height;
        }
        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const OperationDataType& opData) {
            return opData.params.dims.planes;
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const OperationDataType& opData) {
            return opData.params.dims.pitch;
        }
        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
            return { num_elems_x(Point(), opData), num_elems_y(Point(), opData), num_elems_z(Point(), opData) };
        }
    };

    template <ND D, typename T, typename Enabler = void>
    struct SplitWriteParams {};

    template <ND D, typename T>
    struct SplitWriteParams<D, T, typename std::enable_if_t<cn<T> == 2>> {
        RawPtr<D, decltype(T::x)> x;
        RawPtr<D, decltype(T::y)> y;
    };

    template <ND D, typename T>
    struct SplitWriteParams<D, T, typename std::enable_if_t<cn<T> == 3>> {
        RawPtr<D, decltype(T::x)> x;
        RawPtr<D, decltype(T::y)> y;
        RawPtr<D, decltype(T::z)> z;
    };

    template <ND D, typename T>
    struct SplitWriteParams<D, T, typename std::enable_if_t<cn<T> == 4>> {
        RawPtr<D, decltype(T::x)> x;
        RawPtr<D, decltype(T::y)> y;
        RawPtr<D, decltype(T::z)> z;
        RawPtr<D, decltype(T::w)> w;
    };

    template <ND D, typename T>
    struct SplitWrite {
    private:
        using Parent = WriteOperation<T, SplitWriteParams<D, T>, VBase<T>, TF::DISABLED, SplitWrite<D, T>>;
        using SelfType = SplitWrite<D, T>;
    public:
        FK_STATIC_STRUCT(SplitWrite, SelfType)
        DECLARE_WRITE_PARENT
        FK_HOST_DEVICE_FUSE void exec(const Point& thread, const InputType& input, const ParamsType& params) {
            static_assert(cn<InputType> >= 2,
                          "Wrong type for split write. It must be one of <type>2, <type>3 or <type>4.");
            *PtrAccessor<D>::point(thread, params.x) = input.x;
            *PtrAccessor<D>::point(thread, params.y) = input.y;
            if constexpr (cn<InputType> >= 3) *PtrAccessor<D>::point(thread, params.z) = input.z;
            if constexpr (cn<InputType> == 4) *PtrAccessor<D>::point(thread, params.w) = input.w;
        }
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return opData.params.x.dims.width;
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const OperationDataType& opData) {
            return opData.params.x.dims.pitch;
        }

        FK_HOST_FUSE auto build(const std::vector<Ptr2D<VBase<T>>>& output) {
            static_assert(cn<T> >= 2, "Split operations can only be used with types of 2, 3 or 4 channels.");
            if constexpr (cn<T> == 2) {
                return InstantiableType{ {{output.at(0).ptr(), output.at(1).ptr()}} };
            } else if constexpr (cn<T> == 3) {
                return InstantiableType{ {{output.at(0).ptr(), output.at(1).ptr(), output.at(2).ptr()}} };
            } else {
                return InstantiableType{ {{output.at(0).ptr(), output.at(1).ptr(), output.at(2).ptr(), output.at(3).ptr()}} };
            }
        }
    };

    /* The following code has the following copy right

       Copyright 2024-2025 Oscar Amoros Huguet
       Copyright 2023 Mediaproduccion S.L.U. (Oscar Amoros Huget)
       Copyright 2023 Mediaproduccion S.L.U. (Guillermo Oyarzun Altamirano)

       Licensed under the Apache License, Version 2.0 (the "License");
       you may not use this file except in compliance with the License.
       You may obtain a copy of the License at

           http://www.apache.org/licenses/LICENSE-2.0

       Unless required by applicable law or agreed to in writing, software
       distributed under the License is distributed on an "AS IS" BASIS,
       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
       See the License for the specific language governing permissions and
       limitations under the License. */

    enum class CircularDirection { Ascendent, Descendent };

    template <typename OperationDataTypeArray>
    struct CircularMemoryParams {
        int first;
        OperationDataTypeArray opData;
    };

    namespace circular_batch_internal {
        template <CircularDirection direction, int BATCH>
        FK_HOST_DEVICE_CNST Point computeCircularThreadIdx(const Point& currentIdx, const int& fst) {
            if constexpr (direction == CircularDirection::Ascendent) {
                const int z = currentIdx.z + fst;
                return { currentIdx.x, currentIdx.y, z >= BATCH ? z - BATCH : z };
            } else {
                const int z = fst - currentIdx.z;
                return { currentIdx.x, currentIdx.y, z < 0 ? static_cast<int>(BATCH + z) : static_cast<int>(z) };
            }
        }
    } // namespace circular_batch_internal

    template <CircularDirection direction, typename Operation, int BATCH>
    struct CircularBatchRead {
    private:
        using Parent = ReadOperation<typename Operation::ReadDataType,
                                    CircularMemoryParams<OperationData<Operation>[BATCH]>,
                                    typename Operation::OutputType,
                                    Operation::THREAD_FUSION ? TF::ENABLED : TF::DISABLED,
                                    CircularBatchRead<direction, Operation, BATCH>>;
        using SelfType = CircularBatchRead<direction, Operation, BATCH>;
    public:
        FK_STATIC_STRUCT(CircularBatchRead, SelfType)
        DECLARE_READ_PARENT
        template <uint ELEMS_PER_THREAD = 1>
        FK_HOST_DEVICE_FUSE ThreadFusionType<ReadDataType, ELEMS_PER_THREAD, OutputType> exec(const Point& thread, const ParamsType& params) {
            const Point newThreadIdx = circular_batch_internal::computeCircularThreadIdx<direction, BATCH>(thread, params.first);
            if constexpr (THREAD_FUSION) {
                return Operation::template exec<ELEMS_PER_THREAD>(newThreadIdx, params.opData[newThreadIdx.z]);
            } else {
                return Operation::exec(newThreadIdx, params.opData[newThreadIdx.z]);
            }
        }
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return Operation::num_elems_x(thread, opData.params.opData[thread.z]);
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const OperationDataType& opData) {
            return Operation::num_elems_y(thread, opData.params.opData[thread.z]);
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const OperationDataType& opData) {
            return BATCH;
        }

        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const OperationDataType& opData) {
            return Operation::pitch(thread, opData.params.opData[thread.z]);
        }

        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
            return { num_elems_x(Point(), opData), num_elems_y(Point(), opData), num_elems_z(Point(), opData) };
        }
    };

    template <CircularDirection direction, typename Operation, int BATCH>
    struct CircularBatchWrite {
    private:
        using Parent = WriteOperation<typename Operation::InputType,
                                      CircularMemoryParams<OperationData<Operation>[BATCH]>,
                                      typename Operation::WriteDataType,
                                      Operation::THREAD_FUSION ? TF::ENABLED : TF::DISABLED,
                                      CircularBatchWrite<direction, Operation, BATCH>>;
        using SelfType = CircularBatchWrite<direction, Operation, BATCH>;
    public:
        FK_STATIC_STRUCT(CircularBatchWrite, SelfType)
        DECLARE_WRITE_PARENT
        template <uint ELEMS_PER_THREAD = 1>
        FK_HOST_DEVICE_FUSE void exec(const Point& thread, const ThreadFusionType<InputType, ELEMS_PER_THREAD, InputType>& input, const ParamsType& params) {
            const Point newThreadIdx = circular_batch_internal::computeCircularThreadIdx<direction, BATCH>(thread, params.first);
            if constexpr (THREAD_FUSION) {
                Operation::template exec<ELEMS_PER_THREAD>(newThreadIdx, input, params.opData[newThreadIdx.z]);
            } else {
                Operation::exec(newThreadIdx, input, params.opData[newThreadIdx.z]);
            }
        }
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opBatch) {
            return Operation::num_elems_x(thread, opBatch.params.opData[thread.z]);
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const OperationDataType& opBatch) {
            return Operation::pitch(thread, opBatch.params.opData[thread.z]);
        }
    };

    template <CircularDirection direction, typename Operation, int BATCH>
    struct CircularTensorRead {
    private:
        using Parent = ReadOperation<typename Operation::ReadDataType,
                                     CircularMemoryParams<OperationData<Operation>>,
                                     typename Operation::OutputType,
                                     Operation::THREAD_FUSION ? TF::ENABLED : TF::DISABLED,
                                     CircularTensorRead<direction, Operation, BATCH>>;
        using SelfType = CircularTensorRead<direction, Operation, BATCH>;
    public:
        FK_STATIC_STRUCT(CircularTensorRead, SelfType)
        DECLARE_READ_PARENT
        template <uint ELEMS_PER_THREAD = 1>
        FK_HOST_DEVICE_FUSE const ThreadFusionType<ReadDataType, ELEMS_PER_THREAD, OutputType> exec(const Point& thread, const ParamsType& params) {
            const Point newThreadIdx = circular_batch_internal::computeCircularThreadIdx<direction, BATCH>(thread, params.first);
            if constexpr (THREAD_FUSION) {
                return Operation::template exec<ELEMS_PER_THREAD>(newThreadIdx, params.opData);
            } else {
                return Operation::exec(newThreadIdx, params.opData);
            }
        }
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return Operation::num_elems_x(thread, opData.params.opData);
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const OperationDataType& opData) {
            return Operation::num_elems_y(thread, opData.params.opData);
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const OperationDataType& opData) {
            return BATCH;
        }

        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const OperationDataType& opData) {
            return Operation::pitch(thread, opData.params.opData);
        }

        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
            return { num_elems_x(Point(), opData), num_elems_y(Point(), opData), num_elems_z(Point(), opData) };
        }
    };

    template <CircularDirection direction, typename Operation, int BATCH>
    struct CircularTensorWrite {
    private:
        using Parent = WriteOperation<typename Operation::InputType,
                                      CircularMemoryParams<OperationData<Operation>>,
                                      typename Operation::WriteDataType,
                                      Operation::THREAD_FUSION ? TF::ENABLED : TF::DISABLED,
                                      CircularTensorWrite<direction, Operation, BATCH>>;
        using SelfType = CircularTensorWrite<direction, Operation, BATCH>;
    public:
        FK_STATIC_STRUCT(CircularTensorWrite, SelfType)
        DECLARE_WRITE_PARENT
        template <uint ELEMS_PER_THREAD = 1>
        FK_HOST_DEVICE_FUSE void exec(const Point& thread,
                                      const ThreadFusionType<InputType, ELEMS_PER_THREAD, InputType>& input,
                                      const ParamsType& params) {
            const Point newThreadIdx = circular_batch_internal::computeCircularThreadIdx<direction, BATCH>(thread, params.first);
            if constexpr (THREAD_FUSION) {
                Operation::template exec<ELEMS_PER_THREAD>(newThreadIdx, input, params.opData);
            } else {
                Operation::exec(newThreadIdx, input, params.opData);
            }
        }
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return Operation::num_elems_x(thread, opData.params.opData);
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const OperationDataType& opData) {
            return Operation::pitch(thread, opData.params.opData);
        }
    };

} //namespace fk

#endif // FK_MEMORY_OPERATIONS
