/* Copyright 2023-2026 Oscar Amoros Huguet

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
#include <vector>

namespace fk {
    template <ND D, typename T>
    struct PerThreadRead {
    private:
        using Parent = ReadOperation<T, RawPtr<D, T>, T, PerThreadRead<D, T>>;
        using SelfType = PerThreadRead<D, T>;
    public:
        FK_STATIC_STRUCT(PerThreadRead, SelfType)
        DECLARE_READ_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const Point thread, const ParamsType& params) {
            return *PtrAccessor<D>::cr_point(thread, params);
        }

        // Thread fusion opt-in: exec() is a plain contiguous read from this pointer, so the
        // vectorized path can be synthesized centrally by ThreadFusionAdapter.
        FK_HOST_DEVICE_FUSE const ParamsType& contiguous_data(const ParamsType& params) {
            return params;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point thread, const OperationDataType& opData) {
            return opData.params.dims.width;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point thread, const OperationDataType& opData) {
            if constexpr (D == ND::_1D) {
                return 1;
            } else {
                return opData.params.dims.height;
            }
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point thread, const OperationDataType& opData) {
            if constexpr (D == ND::_1D || D == ND::_2D) {
                return 1;
            } else {
                return opData.params.dims.planes;
            }
        }

        FK_HOST_DEVICE_FUSE uint pitch(const Point thread, const OperationDataType& opData) {
            return opData.params.dims.pitch;
        }

        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
            return { num_elems_x(Point{0,0,0}, opData), num_elems_y(Point{0,0,0}, opData), num_elems_z(Point{0,0,0}, opData) };
        }
    };

    struct ReadOp {
        template <typename PtrType>
        FK_HOST_FUSE decltype(auto) build(PtrType&& ptr) {
            constexpr ND D = std::decay_t<PtrType>::nd;
            using PtrDataType = typename std::decay_t<PtrType>::Type;
            return PerThreadRead<D, PtrDataType>::build(std::forward<PtrType>(ptr));
        }
        template <typename PtrType, size_t N>
        FK_HOST_FUSE decltype(auto) build(const std::array<PtrType, N>& ptrs) {
            constexpr ND D = PtrType::nd;
            using PtrDataType = typename PtrType::Type;
            return PerThreadRead<D, PtrDataType>::build(ptrs);
        }
    };

    template <enum ND D, typename T>
    struct PerThreadWrite {
    private:
        using Parent = WriteOperation<T, RawPtr<D, T>, T, PerThreadWrite<D, T>>;
        using SelfType = PerThreadWrite<D, T>;
    public:
        FK_STATIC_STRUCT(PerThreadWrite, SelfType)
        DECLARE_WRITE_PARENT
        FK_HOST_DEVICE_FUSE void exec(const Point thread, const InputType input, const ParamsType& params) {
            *PtrAccessor<D>::point(thread, params) = input;
        }

        // Thread fusion opt-in: exec() is a plain contiguous write to this pointer, so the
        // vectorized path can be synthesized centrally by ThreadFusionAdapter.
        FK_HOST_DEVICE_FUSE const ParamsType& contiguous_data(const ParamsType& params) {
            return params;
        }
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point thread, const OperationDataType& opData) {
            return opData.params.dims.width;
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point thread, const OperationDataType& opData) {
            return opData.params.dims.pitch;
        }
        FK_HOST_FUSE InstantiableType build(const Ptr<D, T>& ptr) {
            return { {ptr.ptr()} };
        }
    };

    struct WriteOp {
        template <typename PtrType>
        FK_HOST_FUSE decltype(auto) build(PtrType&& ptr) {
            constexpr ND D = std::decay_t<PtrType>::nd;
            using PtrDataType = typename std::decay_t<PtrType>::Type;
            return PerThreadWrite<D, PtrDataType>::build(std::forward<PtrType>(ptr));
        }
        template <typename PtrType, size_t N>
        FK_HOST_FUSE decltype(auto) build(const std::array<PtrType, N>& ptrs) {
            constexpr ND D = PtrType::nd;
            using PtrDataType = typename PtrType::Type;
            return PerThreadWrite<D, PtrDataType>::build(ptrs);
        }
    };

    template <typename T>
    struct TensorRead {
    private:
        using Parent = ReadOperation<T, RawPtr<ND::_3D, T>, T, TensorRead<T>>;
        using SelfType = TensorRead<T>;
    public:
        FK_STATIC_STRUCT(TensorRead, SelfType)
        DECLARE_READ_PARENT

        FK_HOST_DEVICE_FUSE OutputType exec(const Point thread, const ParamsType& params) {
            return *PtrAccessor<ND::_3D>::cr_point(thread, params);
        }

        // Thread fusion opt-in: exec() is a plain contiguous read from this pointer, so the
        // vectorized path can be synthesized centrally by ThreadFusionAdapter.
        FK_HOST_DEVICE_FUSE const ParamsType& contiguous_data(const ParamsType& params) {
            return params;
        }
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point thread, const OperationDataType& opData) {
            return opData.params.dims.width;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point thread, const OperationDataType& opData) {
            return opData.params.dims.height;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point thread, const OperationDataType& opData) {
            return opData.params.dims.planes;
        }

        FK_HOST_DEVICE_FUSE uint pitch(const Point thread, const OperationDataType& opData) {
            return opData.params.dims.pitch;
        }

        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
            return { num_elems_x(Point{0,0,0}, opData), num_elems_y(Point{0,0,0}, opData), num_elems_z(Point{0,0,0}, opData) };
        }
    };

    template <typename T>
    struct TensorWrite {
    private:
        using Parent = WriteOperation<T, RawPtr<ND::_3D, T>, T, TensorWrite<T>>;
        using SelfType = TensorWrite<T>;
    public:
        FK_STATIC_STRUCT(TensorWrite, SelfType)
        DECLARE_WRITE_PARENT
        FK_HOST_DEVICE_FUSE void exec(const Point thread, const InputType input, const ParamsType& params) {
            *PtrAccessor<ND::_3D>::point(thread, params) = input;
        }

        // Thread fusion opt-in: exec() is a plain contiguous write to this pointer, so the
        // vectorized path can be synthesized centrally by ThreadFusionAdapter.
        FK_HOST_DEVICE_FUSE const ParamsType& contiguous_data(const ParamsType& params) {
            return params;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point thread, const OperationDataType& opData) {
            return opData.params.dims.width;
        }

        FK_HOST_DEVICE_FUSE uint pitch(const Point thread, const OperationDataType& opData) {
            return opData.params.dims.pitch;
        }
    };

    template <typename T>
    struct TensorSplit {
    private:
        using Parent = WriteOperation<T, RawPtr<ND::_3D, VBase<T>>, VBase<T>, TensorSplit<T>>;
        using SelfType = TensorSplit<T>;
    public:
        FK_STATIC_STRUCT(TensorSplit, SelfType)
        DECLARE_WRITE_PARENT
        FK_HOST_DEVICE_FUSE void exec(const Point thread, const InputType input, const ParamsType& params) {
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
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point thread, const OperationDataType& opData) {
            return opData.params.dims.width;
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point thread, const OperationDataType& opData) {
            return opData.params.dims.pitch;
        }
    };

    template <typename T>
    struct TensorTSplit {
    private:
        using Parent = WriteOperation<T, RawPtr<ND::T3D, VBase<T>>, VBase<T>, TensorTSplit<T>>;
        using SelfType = TensorTSplit<T>;
    public:
        FK_STATIC_STRUCT(TensorTSplit, SelfType)
        DECLARE_WRITE_PARENT
        FK_HOST_DEVICE_FUSE void exec(const Point thread, const InputType input, const ParamsType& params) {
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
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point thread, const OperationDataType& opData) {
            return opData.params.dims.width;
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point thread, const OperationDataType& opData) {
            return opData.params.dims.pitch;
        }
    };

    template <typename T>
    struct TensorPack {
    private:
        using Parent = ReadOperation<VBase<T>, RawPtr<ND::_3D, VBase<T>>, T, TensorPack<T>>;
        using SelfType = TensorPack<T>;
    public:
        FK_STATIC_STRUCT(TensorPack, SelfType)
        DECLARE_READ_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const Point thread, const ParamsType& params) {
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

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point thread, const OperationDataType& opData) {
            return opData.params.dims.width;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point thread, const OperationDataType& opData) {
            return opData.params.dims.height;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point thread, const OperationDataType& opData) {
            return opData.params.dims.planes;
        }

        FK_HOST_DEVICE_FUSE uint pitch(const Point thread, const OperationDataType& opData) {
            return opData.params.dims.pitch;
        }

        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
            return { num_elems_x(Point{0,0,0}, opData), num_elems_y(Point{0,0,0}, opData), num_elems_z(Point{0,0,0}, opData) };
        }
    };

    template <typename T>
    struct TensorTPack {
    private:
        using Parent = ReadOperation<T, RawPtr<ND::T3D, VBase<T>>, T, TensorTPack<T>>;
        using SelfType = TensorTPack<T>;
    public:
        FK_STATIC_STRUCT(TensorTPack, SelfType)
        DECLARE_READ_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const Point thread, const ParamsType& params) {
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
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point thread, const OperationDataType& opData) {
            return opData.params.dims.width;
        }
        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point thread, const OperationDataType& opData) {
            return opData.params.dims.height;
        }
        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point thread, const OperationDataType& opData) {
            return opData.params.dims.planes;
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point thread, const OperationDataType& opData) {
            return opData.params.dims.pitch;
        }
        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
            return { num_elems_x(Point{0,0,0}, opData), num_elems_y(Point{0,0,0}, opData), num_elems_z(Point{0,0,0}, opData) };
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
        using Parent = WriteOperation<T, SplitWriteParams<D, T>, VBase<T>, SplitWrite<D, T>>;
        using SelfType = SplitWrite<D, T>;
    public:
        FK_STATIC_STRUCT(SplitWrite, SelfType)
        DECLARE_WRITE_PARENT
        FK_HOST_DEVICE_FUSE void exec(const Point thread, const InputType input, const ParamsType& params) {
            static_assert(cn<InputType> >= 2,
                          "Wrong type for split write. It must be one of <type>2, <type>3 or <type>4.");
            *PtrAccessor<D>::point(thread, params.x) = input.x;
            *PtrAccessor<D>::point(thread, params.y) = input.y;
            if constexpr (cn<InputType> >= 3) *PtrAccessor<D>::point(thread, params.z) = input.z;
            if constexpr (cn<InputType> == 4) *PtrAccessor<D>::point(thread, params.w) = input.w;
        }
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point thread, const OperationDataType& opData) {
            return opData.params.x.dims.width;
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point thread, const OperationDataType& opData) {
            return opData.params.x.dims.pitch;
        }

        FK_HOST_FUSE InstantiableType build(const std::vector<Ptr2D<VBase<T>>>& output) {
            static_assert(cn<T> >= 2, "Split operations can only be used with types of 2, 3 or 4 channels.");
            if constexpr (cn<T> == 2) {
                return { {{output.at(0).ptr(), output.at(1).ptr()}} };
            } else if constexpr (cn<T> == 3) {
                return { {{output.at(0).ptr(), output.at(1).ptr(), output.at(2).ptr()}} };
            } else {
                return { {{output.at(0).ptr(), output.at(1).ptr(), output.at(2).ptr(), output.at(3).ptr()}} };
            }
        }
    };

    /* The following code has the following copy right

       Copyright 2024-2026 Oscar Amoros Huguet
       Copyright 2023 Grup Mediapro S.L.U. (Oscar Amoros Huguet)
       Copyright 2023 Grup Mediapro S.L.U. (Guillermo Oyarzun Altamirano)

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
        FK_HOST_DEVICE_CNST Point computeCircularThreadIdx(const Point currentIdx, const int fst) {
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
                                    CircularBatchRead<direction, Operation, BATCH>>;
        using SelfType = CircularBatchRead<direction, Operation, BATCH>;
    public:
        FK_STATIC_STRUCT(CircularBatchRead, SelfType)
        DECLARE_READ_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const Point thread, const ParamsType& params) {
            const auto access = forwarded_access(thread, params);
            return Operation::exec(access.thread, access.opData);
        }

        // Thread fusion hook: this Operation only remaps the thread circularly; the memory
        // access is delegated to the wrapped Operation, so thread fusion is available whenever
        // the wrapped Operation supports it.
        FK_HOST_DEVICE_FUSE ForwardedAccess<Operation> forwarded_access(const Point thread, const ParamsType& params) {
            const Point newThreadIdx = circular_batch_internal::computeCircularThreadIdx<direction, BATCH>(thread, params.first);
            return { newThreadIdx, params.opData[newThreadIdx.z] };
        }
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point thread, const OperationDataType& opData) {
            return Operation::num_elems_x(thread, opData.params.opData[thread.z]);
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point thread, const OperationDataType& opData) {
            return Operation::num_elems_y(thread, opData.params.opData[thread.z]);
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point thread, const OperationDataType& opData) {
            return BATCH;
        }

        FK_HOST_DEVICE_FUSE uint pitch(const Point thread, const OperationDataType& opData) {
            return Operation::pitch(thread, opData.params.opData[thread.z]);
        }

        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
            return { num_elems_x(Point{0,0,0}, opData), num_elems_y(Point{0,0,0}, opData), num_elems_z(Point{0,0,0}, opData) };
        }
    };

    template <CircularDirection direction, typename Operation, int BATCH>
    struct CircularBatchWrite {
    private:
        using Parent = WriteOperation<typename Operation::InputType,
                                      CircularMemoryParams<OperationData<Operation>[BATCH]>,
                                      typename Operation::WriteDataType,
                                      CircularBatchWrite<direction, Operation, BATCH>>;
        using SelfType = CircularBatchWrite<direction, Operation, BATCH>;
    public:
        FK_STATIC_STRUCT(CircularBatchWrite, SelfType)
        DECLARE_WRITE_PARENT
        FK_HOST_DEVICE_FUSE void exec(const Point thread, const InputType input, const ParamsType& params) {
            const auto access = forwarded_access(thread, params);
            Operation::exec(access.thread, input, access.opData);
        }

        // Thread fusion hook: this Operation only remaps the thread circularly; the memory
        // access is delegated to the wrapped Operation, so thread fusion is available whenever
        // the wrapped Operation supports it.
        FK_HOST_DEVICE_FUSE ForwardedAccess<Operation> forwarded_access(const Point thread, const ParamsType& params) {
            const Point newThreadIdx = circular_batch_internal::computeCircularThreadIdx<direction, BATCH>(thread, params.first);
            return { newThreadIdx, params.opData[newThreadIdx.z] };
        }
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point thread, const OperationDataType& opBatch) {
            return Operation::num_elems_x(thread, opBatch.params.opData[thread.z]);
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point thread, const OperationDataType& opBatch) {
            return Operation::pitch(thread, opBatch.params.opData[thread.z]);
        }
    };

    template <CircularDirection direction, typename Operation, int BATCH>
    struct CircularTensorRead {
    private:
        using Parent = ReadOperation<typename Operation::ReadDataType,
                                     CircularMemoryParams<OperationData<Operation>>,
                                     typename Operation::OutputType,
                                     CircularTensorRead<direction, Operation, BATCH>>;
        using SelfType = CircularTensorRead<direction, Operation, BATCH>;
    public:
        FK_STATIC_STRUCT(CircularTensorRead, SelfType)
        DECLARE_READ_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const Point thread, const ParamsType& params) {
            const auto access = forwarded_access(thread, params);
            return Operation::exec(access.thread, access.opData);
        }

        // Thread fusion hook: this Operation only remaps the thread circularly; the memory
        // access is delegated to the wrapped Operation, so thread fusion is available whenever
        // the wrapped Operation supports it.
        FK_HOST_DEVICE_FUSE ForwardedAccess<Operation> forwarded_access(const Point thread, const ParamsType& params) {
            const Point newThreadIdx = circular_batch_internal::computeCircularThreadIdx<direction, BATCH>(thread, params.first);
            return { newThreadIdx, params.opData };
        }
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point thread, const OperationDataType& opData) {
            return Operation::num_elems_x(thread, opData.params.opData);
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point thread, const OperationDataType& opData) {
            return Operation::num_elems_y(thread, opData.params.opData);
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point thread, const OperationDataType& opData) {
            return BATCH;
        }

        FK_HOST_DEVICE_FUSE uint pitch(const Point thread, const OperationDataType& opData) {
            return Operation::pitch(thread, opData.params.opData);
        }

        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
            return { num_elems_x(Point{0,0,0}, opData), num_elems_y(Point{0,0,0}, opData), num_elems_z(Point{0,0,0}, opData) };
        }
    };

    template <CircularDirection direction, typename Operation, int BATCH>
    struct CircularTensorWrite {
    private:
        using Parent = WriteOperation<typename Operation::InputType,
                                      CircularMemoryParams<OperationData<Operation>>,
                                      typename Operation::WriteDataType,
                                      CircularTensorWrite<direction, Operation, BATCH>>;
        using SelfType = CircularTensorWrite<direction, Operation, BATCH>;
    public:
        FK_STATIC_STRUCT(CircularTensorWrite, SelfType)
        DECLARE_WRITE_PARENT
        FK_HOST_DEVICE_FUSE void exec(const Point thread, const InputType input, const ParamsType& params) {
            const auto access = forwarded_access(thread, params);
            Operation::exec(access.thread, input, access.opData);
        }

        // Thread fusion hook: this Operation only remaps the thread circularly; the memory
        // access is delegated to the wrapped Operation, so thread fusion is available whenever
        // the wrapped Operation supports it.
        FK_HOST_DEVICE_FUSE ForwardedAccess<Operation> forwarded_access(const Point thread, const ParamsType& params) {
            const Point newThreadIdx = circular_batch_internal::computeCircularThreadIdx<direction, BATCH>(thread, params.first);
            return { newThreadIdx, params.opData };
        }
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point thread, const OperationDataType& opData) {
            return Operation::num_elems_x(thread, opData.params.opData);
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point thread, const OperationDataType& opData) {
            return Operation::pitch(thread, opData.params.opData);
        }
    };

    // Reads two sources at the same Point and returns a Tuple<T, T>, intended to
    // feed the two-input Unary form of binary operators (Add/Sub/Mul/Div, BwAnd/
    // BwOr/BwXor). This enables image-by-image element-wise operations as a single
    // fused kernel. Thread fusion is disabled to keep the dual-pointer read simple.
    template <ND D, typename T>
    struct DualSourceReadParams {
        RawPtr<D, T> src1;
        RawPtr<D, T> src2;
    };

    template <ND D, typename T>
    struct DualSourceRead {
    private:
        using Parent = ReadOperation<T, DualSourceReadParams<D, T>, Tuple<T, T>,
                                     DualSourceRead<D, T>>;
        using SelfType = DualSourceRead<D, T>;
    public:
        FK_STATIC_STRUCT(DualSourceRead, SelfType)
        DECLARE_READ_PARENT
        FK_HOST_DEVICE_FUSE auto exec(const Point thread, const ParamsType& params) -> Tuple<T, T> {
            const T a = *PtrAccessor<D>::template cr_point<T, T>(thread, params.src1);
            const T b = *PtrAccessor<D>::template cr_point<T, T>(thread, params.src2);
            return { a, b };
        }
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point thread, const OperationDataType& opData) {
            return opData.params.src1.dims.width;
        }
        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point thread, const OperationDataType& opData) {
            if constexpr (D == ND::_1D) {
                return 1;
            } else {
                return opData.params.src1.dims.height;
            }
        }
        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point thread, const OperationDataType& opData) {
            if constexpr (D == ND::_1D || D == ND::_2D) {
                return 1;
            } else {
                return opData.params.src1.dims.planes;
            }
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point thread, const OperationDataType& opData) {
            return opData.params.src1.dims.pitch;
        }
        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
            return { num_elems_x(Point{0,0,0}, opData), num_elems_y(Point{0,0,0}, opData), num_elems_z(Point{0,0,0}, opData) };
        }
        FK_HOST_FUSE InstantiableType build(const Ptr<D, T>& src1, const Ptr<D, T>& src2) {
            return { { DualSourceReadParams<D, T>{ src1.ptr(), src2.ptr() } } };
        }
    };

} //namespace fk

#endif // FK_MEMORY_OPERATIONS
