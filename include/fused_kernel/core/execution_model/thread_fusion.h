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

#ifndef FK_THREAD_FUSION
#define FK_THREAD_FUSION

#include <fused_kernel/core/utils/vector_utils.h>
#include <fused_kernel/core/data/point.h>
#include <fused_kernel/core/data/rawptr.h>
#include <fused_kernel/core/execution_model/operation_model/operation_data.h>

namespace fk {

    enum class TF : bool {
        ENABLED = true,
        DISABLED = false
    };

    /* Possible combinations:
    Size, Channels, Types
    1,    1         char, uchar                     8,  8  char4, uchar4        x4
    2,    1         short, ushort                   4,  2  short2, ushort2      x2
    4,    1         int, uint, float                8,  2  int2, uint2, float2  x2
    8,    1         longlong, ulonglong, double     8,  1
    2,    2         char2, uchar2                   4,  4  char4, uchar4        x2
    4,    2         short2, ushort2                 4,  2
    8,    2         int2, uint2, float2             8,  2
    16,   2         longlong2, ulonglong2, double2  16, 2
    3,    3         char3, uchar3                   3,  3
    6,    3         short3, ushort3                 6,  3
    12,   3         int3, uint3, float3             12, 3
    24,   3         longlong3, ulonglong3, double3  24, 3
    4,    4         char4, uchar4                   4,  4
    8,    4         short4, ushort4                 8,  4
    16,   4         int4, uint4, float4             16, 4
    32,   4         longlong4, ulonglong4, double4  32, 4

    Times bigger can be: 1, 2, 4
    */

    using TFSourceTypes = TypeListCat_t<BaseTypes, VOne, VTwo, VThree, VFour>;
    using TFBiggerTypes = TypeList<bool4, uchar4,  char4,  ushort2,  short2,  uint2, int2, ulong,  long,  ulonglong,  longlong,  float2, double,
                                   bool4, uchar4,  char4,  ushort2,  short2,  uint2, int2, ulong,  long,  ulonglong,  longlong,  float2, double,
                                   bool4, uchar4,  char4,  ushort2,  short2,  uint2, int2, ulong2, long2, ulonglong2, longlong2, float2, double2,
                                   bool3, uchar3,  char3,  ushort3,  short3,  uint3, int3, ulong3, long3, ulonglong3, longlong3, float3, double3,
                                   bool4, uchar4,  char4,  ushort4,  short4,  uint4, int4, ulong4, long4, ulonglong4, longlong4, float4, double4>;
    template <typename T>
    using FilteredType_t = std::conditional_t<std::is_same_v<T, char>, typename VectorTraits<T>::base, T>;
    template <typename SourceType>
    using TFBiggerType_t = EquivalentType_t<FilteredType_t<SourceType>, TFSourceTypes, TFBiggerTypes>;

    constexpr std::integer_sequence<uint, 1, 2, 3, 4> validChannelsSequence;

    template <uint channelNumber>
    constexpr bool isValidChannelNumber = Find<uint, channelNumber>::one_of(validChannelsSequence);

    template <typename SourceType, uint ELEMS_PER_THREAD, typename OutputType = SourceType, typename=void>
    struct ThreadFusionTypeImpl : std::false_type {
    private:
        using VectorType_ = VectorType<VBase<SourceType>, (cn<SourceType>)* ELEMS_PER_THREAD>;
    public:
        using type = std::conditional_t<validCUDAVec<SourceType>, typename VectorType_::type_v, typename VectorType_::type>;
    };

    template <typename SourceType, uint ELEMS_PER_THREAD, typename OutputType>
    struct ThreadFusionTypeImpl<SourceType, ELEMS_PER_THREAD, OutputType, std::enable_if_t<!std::is_same_v<SourceType, OutputType> || std::is_same_v<SourceType, NullType>, void>> : std::true_type {
        using type = OutputType;
    };

    template <typename SourceType, uint ELEMS_PER_THREAD, typename OutputType>
    using ThreadFusionType = typename ThreadFusionTypeImpl<SourceType, ELEMS_PER_THREAD, OutputType>::type;

    template <typename ReadType, typename WriteType, bool ENABLED_>
    struct ThreadFusionInfo {
        public:
            static constexpr bool ENABLED = ENABLED_ && isValidChannelNumber<(cn<TFBiggerType_t<ReadType>> / cn<ReadType>) * cn<WriteType>>;
            using BiggerReadType = std::conditional_t<ENABLED, TFBiggerType_t<ReadType>, ReadType>;
            static constexpr uint elems_per_thread{ static_cast<uint>(cn<BiggerReadType> / cn<ReadType>) };
            using BiggerWriteType = VectorType_t<VBase<WriteType>, elems_per_thread * cn<WriteType>>;

            template <int IDX>
            FK_HOST_DEVICE_FUSE ReadType get(const BiggerReadType& data) {
                static_assert(IDX < elems_per_thread, "Index out of range for this ThreadFusionInfo");
                if constexpr (cn<ReadType> == 1) {
                    if constexpr (IDX == 0) {
                        return data.x;
                    } else if constexpr (IDX == 1) {
                        return data.y;
                    } else if constexpr (IDX == 2) {
                        return data.z;
                    } else if constexpr (IDX == 3) {
                        return data.w;
                    }
                } else if constexpr (cn<ReadType> == 2) {
                    if constexpr (IDX == 0) {
                        return make_<ReadType>(data.x, data.y);
                    } else if constexpr (IDX == 1) {
                        return make_<ReadType>(data.z, data.w);
                    }
                } else if constexpr (cn<ReadType> == 3) {
                    if constexpr (IDX == 0) {
                        return make_<ReadType>(data.x, data.y, data.z);
                    }
                } else if constexpr (cn<ReadType> == 4) {
                    if constexpr (IDX == 0) {
                        return make_<ReadType>(data.x, data.y, data.z, data.w);
                    }
                }
            }
            template <typename... OriginalTypes>
            FK_HOST_DEVICE_FUSE BiggerWriteType make(const OriginalTypes&... data) {
                static_assert(and_v<std::is_same_v<WriteType, OriginalTypes>...>, "Not all types are the same when making the ThreadFusion BiggerType value");
                if constexpr (cn<WriteType> > 1) {
                    return make_impl(data...);
                } else {
                    return make_<BiggerWriteType>(data...);
                }
            }

        private:
            FK_HOST_DEVICE_FUSE BiggerWriteType make_impl(const WriteType& data0,
                                                          const WriteType& data1) {
                if constexpr (cn<WriteType> == 2) {
                    return make_<BiggerWriteType>(data0.x, data0.y, data1.x, data1.y);
                }
            }
    };

    /* THREAD FUSION AUTHORING HOOKS
    *
    *  Operations never implement the multi-element (vectorized) exec() variants themselves.
    *  A memory Operation only writes its scalar exec(). To opt into thread fusion it declares
    *  one of the following static hooks, and ThreadFusionAdapter (below) synthesizes the wide
    *  load/store in a single central place:
    *
    *  1. contiguous_data: for Operations whose exec() is a plain per-thread access into a
    *     pitch-linear RawPtr, where thread.x indexes consecutive elements of ReadDataType
    *     (Read Operations) or WriteDataType (Write Operations):
    *         FK_HOST_DEVICE_FUSE const ParamsType& contiguous_data(const ParamsType& params) {
    *             return params;
    *         }
    *     The returned object must be a RawPtr<D, T> where T is the Operation's memory data type.
    *
    *  2. forwarded_access: for wrapper Operations (batch/circular) that only remap the thread
    *     and/or select per-plane parameters, delegating the actual memory access to a wrapped
    *     Operation:
    *         FK_HOST_DEVICE_FUSE ForwardedAccess<Operation> forwarded_access(const Point thread,
    *                                                                         const ParamsType& params) {
    *             return { remappedThread, params.opData[remappedThread.z] };
    *         }
    *
    *  Operations that declare no hook simply do not support thread fusion (the default).
    *  An Operation must declare at most ONE of the two hooks: declaring both is a contract
    *  violation, rejected at compile time by ThreadFusionAdapter.
    */

    template <typename Operation>
    struct ForwardedAccess {
        using Op = Operation;
        Point thread;
        OperationData<Operation> opData;
    };

    template <typename Op, typename = void>
    struct HasContiguousData : std::false_type {};
    template <typename Op>
    struct HasContiguousData<Op,
        std::void_t<decltype(Op::contiguous_data(std::declval<const typename Op::ParamsType&>()))>>
        : std::true_type {};
    template <typename Op>
    constexpr bool hasContiguousData = HasContiguousData<Op>::value;

    template <typename Op, typename = void>
    struct HasForwardedAccess : std::false_type {};
    template <typename Op>
    struct HasForwardedAccess<Op,
        std::void_t<decltype(Op::forwarded_access(std::declval<Point>(), std::declval<const typename Op::ParamsType&>()))>>
        : std::true_type {};
    template <typename Op>
    constexpr bool hasForwardedAccess = HasForwardedAccess<Op>::value;

    template <typename Op>
    using ForwardTargetOp_t =
        typename decltype(Op::forwarded_access(std::declval<Point>(),
                                               std::declval<const typename Op::ParamsType&>()))::Op;

    template <typename Op, typename = void>
    struct IsThreadFusionCapable : std::false_type {};
    template <typename Op>
    struct IsThreadFusionCapable<Op, std::enable_if_t<hasContiguousData<Op>>> : std::true_type {};
    template <typename Op>
    struct IsThreadFusionCapable<Op, std::enable_if_t<hasForwardedAccess<Op> && !hasContiguousData<Op>>>
        : IsThreadFusionCapable<ForwardTargetOp_t<Op>> {};
    template <typename Op>
    constexpr bool isThreadFusionCapable = IsThreadFusionCapable<Op>::value;

    /* ThreadFusionAdapter: the single place where the vectorized (multi-element) memory accesses
    *  are synthesized out of the Operations' declarative hooks. The DPPs never call an Operation's
    *  exec() with an ELEMS_PER_THREAD template parameter; they go through this adapter instead. */
    template <typename Op>
    struct ThreadFusionAdapter {
        static_assert(!(hasContiguousData<Op> && hasForwardedAccess<Op>),
            "An Operation must declare exactly one thread fusion hook: either contiguous_data or forwarded_access, not both");
    private:
        using SelfType = ThreadFusionAdapter<Op>;
    public:
        FK_STATIC_STRUCT(ThreadFusionAdapter, SelfType)

        template <uint ELEMS_PER_THREAD>
        FK_HOST_DEVICE_FUSE auto read(const Point thread, const OperationData<Op>& opData) {
            if constexpr (ELEMS_PER_THREAD == 1) {
                return Op::exec(thread, opData);
            } else if constexpr (hasForwardedAccess<Op>) {
                const auto access = Op::forwarded_access(thread, opData.params);
                using TargetOp = typename std::decay_t<decltype(access)>::Op;
                return ThreadFusionAdapter<TargetOp>::template read<ELEMS_PER_THREAD>(access.thread, access.opData);
            } else {
                static_assert(hasContiguousData<Op>,
                    "Thread fusion requires the Operation to declare a contiguous_data or forwarded_access hook");
                const auto& ptr = Op::contiguous_data(opData.params);
                using RawPtrType = std::decay_t<decltype(ptr)>;
                using DataType = typename RawPtrType::type;
                static_assert(std::is_same_v<DataType, typename Op::ReadDataType>,
                    "contiguous_data must return a RawPtr of the Operation's ReadDataType");
                constexpr ND D = static_cast<ND>(RawPtrType::NDim);
                using BiggerType = ThreadFusionType<typename Op::ReadDataType, ELEMS_PER_THREAD, typename Op::OutputType>;
                return *PtrAccessor<D>::template cr_point<DataType, BiggerType>(thread, ptr);
            }
        }

        template <uint ELEMS_PER_THREAD, typename BiggerInputType>
        FK_HOST_DEVICE_FUSE void write(const Point thread, const BiggerInputType& input,
                                       const OperationData<Op>& opData) {
            if constexpr (ELEMS_PER_THREAD == 1) {
                Op::exec(thread, input, opData);
            } else if constexpr (hasForwardedAccess<Op>) {
                const auto access = Op::forwarded_access(thread, opData.params);
                using TargetOp = typename std::decay_t<decltype(access)>::Op;
                ThreadFusionAdapter<TargetOp>::template write<ELEMS_PER_THREAD>(access.thread, input, access.opData);
            } else {
                static_assert(hasContiguousData<Op>,
                    "Thread fusion requires the Operation to declare a contiguous_data or forwarded_access hook");
                const auto& ptr = Op::contiguous_data(opData.params);
                using RawPtrType = std::decay_t<decltype(ptr)>;
                using DataType = typename RawPtrType::type;
                static_assert(std::is_same_v<DataType, typename Op::WriteDataType>,
                    "contiguous_data must return a RawPtr of the Operation's WriteDataType");
                // The wide store below is computed from InputType, so the hook is only valid
                // for Write Operations that store their input as-is.
                static_assert(std::is_same_v<typename Op::InputType, typename Op::WriteDataType>,
                    "contiguous_data on a Write Operation requires InputType == WriteDataType");
                constexpr ND D = static_cast<ND>(RawPtrType::NDim);
                using BiggerType = ThreadFusionType<typename Op::InputType, ELEMS_PER_THREAD, typename Op::InputType>;
                *PtrAccessor<D>::template point<DataType, BiggerType>(thread, ptr) = input;
            }
        }
    };

    // Thread Fusion hepler functions

    template <bool THREAD_FUSION_ENABLED, typename... IOpTypes>
    FK_HOST_DEVICE_CNST bool isThreadFusionEnabled() {
        using ReadOperation = typename FirstType_t<IOpTypes...>::Operation;
        using WriteOperation = typename LastType_t<IOpTypes...>::Operation;
        return isThreadFusionCapable<ReadOperation> && isThreadFusionCapable<WriteOperation> && THREAD_FUSION_ENABLED;
    }

    template <bool THREAD_FUSION_ENABLED, typename... IOpTypes>
    FK_HOST_INLINE bool isThreadDivisible(const uint& elems_per_thread, const IOpTypes&... iOps) {
        if constexpr (THREAD_FUSION_ENABLED) {
            const auto& readOp = ppFirst(iOps...);
            const auto& writeOp = ppLast(iOps...);
            using ReadOperation = typename FirstType_t<IOpTypes...>::Operation;
            using WriteOperation = typename LastType_t<IOpTypes...>::Operation;
            // The IOps are OperationData already, so they can be passed directly to num_elems_x
            // (constructing a new OperationData from .params would not compile for Operations
            // with array ParamsType, like BatchWrite).
            // Batch Operations can hold a different row width per z plane, so every plane the
            // kernel will execute must be divisible; otherwise the wide accesses of the
            // divisible kernel variant would go past the end of the non-divisible rows.
            const int numPlanes = static_cast<int>(ReadOperation::num_elems_z(Point{ 0, 0, 0 }, readOp));
            for (int z = 0; z < numPlanes; ++z) {
                const Point plane{ 0, 0, z };
                const uint readRow = ReadOperation::num_elems_x(plane, readOp);
                const uint writeRow = WriteOperation::num_elems_x(plane, writeOp);
                if ((readRow % elems_per_thread != 0) || (writeRow % elems_per_thread != 0)) {
                    return false;
                }
            }
            return true;
        } else {
            return true;
        }
    }
} // namespace fk

#endif
