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

#ifndef FK_BATCH_OPERATIONS_CUH
#define FK_BATCH_OPERATIONS_CUH

#include <fused_kernel/core/execution_model/operation_model/parent_operations.h>
#include <fused_kernel/core/constexpr_libs/constexpr_cmath.h>

/*
BatchRead implementation gidelines, including ReadBackOperations:

##### READ_BACK
template <..., typename BackIOp> ReadBackOperation:
    Has default template <typename... ParamsTypes> build(const std::array<N, ParamsTypes>&... params)
        - Which calls build_batch(params...) and gets std::array<N, IOp>
        - Then returns BatchRead<N, PlanePolicy::PROCESS_ALL>::build(std::array<N, IOp>)
    Has default template <typename... ParamsTypes> build(usedPlanes, defaultValue, std::array<N, ParamsTypes>&... params)
        - Which calls build_batch(params...) and gets std::array<N, IOp>
        - Then returns BatchRead<N, PlanePolicy::CONDITIONAL_WITH_DEFAULT>::build(std::array<N, IOp>, usedPlanes, DefType defaultValue)

ReadBackOperation<TypeList<void>>:
    Instantiable
    ParamsType = ???
    Implements template ReadBackOperation<TypeList<void>>::build(Params&...)
        - Returns either ReadBackOperation<TypeList<void>> with direct {} instantiation or ReadBackOperation<TypeList<void, ...>> with direct {} instantiation
    Implements template <typename BackIOp> ReadBackOperation<TypeList<void>>::build(BackIOp, Params&...)
        - BackIOp is ALWAYS COMPLETE
        - Returns ReadBackOperation<BackIOp> with direct {} instantiation
    Implements template <typename BackIOp> ReadBackOperation<TypeList<void>>::build(const BackIOp& backIOp, const InstantiableType& iOp)
        - Combines the information in InstantiableType::ParamsType with BackIOp to create a complete ReadBackOperation
        - Returns ReadBackOperation<BackIOp> with direct {} instantiation

ReadBackOperation<TypeList<void, ...>>:
    Instantiable
    ParamsType = ??? can be different for each TypeList<void, ...>
    Implements template <typename BackIOp> ReadBackOperation<TypeList<void, ...>>::build(const BackIOp& backIOp, const InstantiableType& iOp)
        - Combines the information in InstantiableType::ParamsType with BackIOp to create a complete ReadBackOperation
        - Returns ReadBackOperation<BackIOp> with direct {} instantiation

ReadBackOperation<BackIOp>:
    Instantiable
    Implements exec(const Point& thread, const OperationData<ReadBackOperation<BackIOp>>& opData)
    Implements exec(const Point& thread, const ParamsType& params, const BackIOp& backIOp)
    Implements build(const OperationData<ReadBackOperation<BackIOp>>& opData)
    Implements build(const ParamsType& params, const BackIOp& backIOp)

##### BATCH_READ

template <PlanePolicy PP, size_t BATCH = 1, typename Operation = TypeList<void>> BatchRead:
    Does not call Operation::build!!
    Only builds with build(std::array<IOp, BATCH>), or build(std::array<IOp, BATCH>, usedPlanes, defaultValue)
    And of course the default builders build(OperationData<SelfType>) and build(ParamsType)
*/

namespace fk {
    // BATCH OPERATIONS

    // Helper template to check for existence of static constexpr int BATCH 
    template <typename T, typename = void>
    struct has_batch : std::false_type {};
    template <typename T>
    struct has_batch<T, std::void_t<decltype(T::BATCH)>> : std::is_integral<decltype(T::BATCH)> {};
    // Helper template to check for existence of type alias Operation
    template <typename T, typename = void>
    struct has_operation : std::false_type {};
    template <typename T>
    struct has_operation<T, std::void_t<typename T::Operation>> : std::true_type {};
    // Combine checks into a single struct
    template <typename T> struct IsBatchOperation :
        std::integral_constant<bool, has_batch<T>::value&& has_operation<T>::value> {};
    // Helper variable template
    template <typename T>
    static constexpr bool isBatchOperation = IsBatchOperation<T>::value;

    // ################### START BATCH READ #####################
    enum class PlanePolicy { PROCESS_ALL = 0, CONDITIONAL_WITH_DEFAULT = 1 };

    template <size_t BATCH, enum PlanePolicy PP, typename OpParamsType, typename DefaultType = NullType>
    struct BatchReadParams;

    template <size_t BATCH, typename Operation, typename DefaultType>
    struct BatchReadParams<BATCH, PlanePolicy::CONDITIONAL_WITH_DEFAULT, Operation, DefaultType> {
        OperationData<Operation> opData[BATCH];
        int usedPlanes;
        DefaultType default_value;
        ActiveThreads activeThreads;
    };

    template <size_t BATCH, typename Operation>
    struct BatchReadParams<BATCH, PlanePolicy::PROCESS_ALL, Operation, NullType> {
        OperationData<Operation> opData[BATCH];
        ActiveThreads activeThreads;
    };

    struct BatchUtils {
        FK_STATIC_STRUCT(BatchUtils, BatchUtils)
#ifndef NVRTC_COMPILER
            template <typename InstantiableType>
        FK_HOST_FUSE auto toArray(const InstantiableType& batchIOp) {
            static_assert(isBatchOperation<typename InstantiableType::Operation>,
                "The IOp passed as parameter is not a batch operation");
            constexpr size_t BATCH = InstantiableType::Operation::BATCH;
            return toArray_helper(std::make_index_sequence<BATCH>{}, batchIOp);
        }
        template <typename Operation, size_t BATCH_N, typename FirstType, typename... ArrayTypes>
        FK_HOST_FUSE auto build_batch(const std::array<FirstType, BATCH_N>& firstInstance, const ArrayTypes &...arrays) {
            static_assert(allArraysSameSize_v<BATCH_N, ArrayTypes...>, "Not all arrays have the same size as BATCH");
            return build_helper_generic<Operation>(std::make_index_sequence<BATCH_N>(), firstInstance, arrays...);
        }
    private:
        template <typename InstantiableType, size_t... Idx>
        FK_HOST_FUSE auto toArray_helper(const std::index_sequence<Idx...>&, const InstantiableType& batchIOp) {
            using Operation = typename InstantiableType::Operation::Operation;
            using OutputArrayType = std::array<Instantiable<Operation>, sizeof...(Idx)>;
            if constexpr (InstantiableType::template is<ReadType>) {
                return OutputArrayType{ Operation::build(batchIOp.params.opData[Idx])... };
            } else {
                static_assert(InstantiableType::template is<WriteType>, "InstantiableType is not a ReadType or WriteType. It means it is not a batch operation");
                return OutputArrayType{ Operation::build(batchIOp.params[Idx])... };
            }
        }
        template <size_t Idx, typename Array>
        FK_HOST_FUSE auto get_element_at_index(const Array& paramArray) -> decltype(paramArray[Idx]) {
            return paramArray[Idx];
        }
        template <typename Operation, size_t Idx, typename... Arrays>
        FK_HOST_FUSE auto call_build_at_index(const Arrays &...arrays) {
            return Operation::build(get_element_at_index<Idx>(arrays)...);
        }
        template <typename Operation, size_t... Idx, typename... Arrays>
        FK_HOST_FUSE auto build_helper_generic(const std::index_sequence<Idx...>&, const Arrays &...arrays) {
            using OutputArrayType = decltype(call_build_at_index<Operation, 0>(std::declval<Arrays>()...));
            return std::array<OutputArrayType, sizeof...(Idx)>{call_build_at_index<Operation, Idx>(arrays...)...};
        }
#endif // NVRTC_COMPILER
    };

    template <PlanePolicy PP = PlanePolicy::PROCESS_ALL, size_t BATCH = 1, typename Operation = TypeList<void>>
    struct BatchRead;

    /*
    BatchRead<PP, BATCH, Operation>:
        Instantiable
        ParamsType = BatchReadParams<BATCH, PP, Operation::ParamsType>
        Implements exec(const Point& thread, const OperationDataType& opData)
        Implements exec(const Point& thread, const ParamsType& params)
        Implements build(const OperationDataType& opData)
        Implements build(const ParamsType& params)
    */
    template <size_t BATCH_, typename Op>
    struct BatchRead<PlanePolicy::PROCESS_ALL, BATCH_, Op> {
        static_assert(isCompleteOperation<Op>,
            "The IOp passed as template parameter is not a complete operation");
    private:
        using SelfType = BatchRead<PlanePolicy::PROCESS_ALL, BATCH_, Op>;
    public:
        FK_STATIC_STRUCT(BatchRead, SelfType)
        using Operation = Op;
        static constexpr size_t BATCH = BATCH_;
        static constexpr PlanePolicy PP = PlanePolicy::PROCESS_ALL;

        using ParamsType = BatchReadParams<BATCH, PP, Operation>;
        using ReadDataType = typename Operation::ReadDataType;
        using InstanceType = ReadType;
        using OutputType = typename Operation::OutputType;
        using OperationDataType = OperationData<SelfType>;
        using InstantiableType = Read<SelfType>;
        static constexpr bool IS_FUSED_OP = Operation::IS_FUSED_OP;
        static constexpr bool THREAD_FUSION = Operation::THREAD_FUSION;

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
            return opData.params.activeThreads;
        }

        template <uint ELEMS_PER_THREAD = 1>
        FK_HOST_DEVICE_FUSE auto exec(const Point& thread, const OperationDataType& opData) {
            return exec<ELEMS_PER_THREAD>(thread, opData.params);
        }
        template <uint ELEMS_PER_THREAD = 1>
        FK_HOST_DEVICE_FUSE auto exec(const Point& thread, const ParamsType& params) {
            if constexpr (THREAD_FUSION) {
                return Operation::template exec<ELEMS_PER_THREAD>(thread, params.opData[thread.z]);
            } else {
                return Operation::exec(thread, params.opData[thread.z]);
            }
        }
        FK_HOST_FUSE InstantiableType build(const OperationDataType& opData) {
            return InstantiableType{ opData };
        }
        FK_HOST_FUSE InstantiableType build(const ParamsType& params) {
            return InstantiableType{ {params} };
        }
    };

    template <size_t BATCH_, typename Op>
    struct BatchRead<PlanePolicy::CONDITIONAL_WITH_DEFAULT, BATCH_, Op> {
        static_assert(isCompleteOperation<Op>,
            "The IOp passed as template parameter is not a complete operation");
    private:
        using SelfType = BatchRead<PlanePolicy::CONDITIONAL_WITH_DEFAULT, BATCH_, Op>;
    public:
        FK_STATIC_STRUCT(BatchRead, SelfType)
        using Operation = Op;
        static constexpr size_t BATCH = BATCH_;
        static constexpr PlanePolicy PP = PlanePolicy::CONDITIONAL_WITH_DEFAULT;

        using ParamsType = BatchReadParams<BATCH, PP, Operation, typename Operation::OutputType>;
        using ReadDataType = typename Operation::ReadDataType;
        using InstanceType = ReadType;
        using OutputType = typename Operation::OutputType;
        using OperationDataType = OperationData<SelfType>;
        using InstantiableType = Read<SelfType>;
        static constexpr bool IS_FUSED_OP = Operation::IS_FUSED_OP;
        static constexpr bool THREAD_FUSION = false;

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
            return opData.params.activeThreads;
        }

        template <uint ELEMS_PER_THREAD = 1>
        FK_HOST_DEVICE_FUSE auto exec(const Point& thread, const OperationDataType& opData) {
            return exec<ELEMS_PER_THREAD>(thread, opData.params);
        }
        template <uint ELEMS_PER_THREAD = 1>
        FK_HOST_DEVICE_FUSE auto exec(const Point& thread, const ParamsType& params) {
            if (params.usedPlanes <= thread.z) {
                return params.default_value;
            } else {
                if constexpr (THREAD_FUSION) {
                    return Operation::template exec<ELEMS_PER_THREAD>(thread, params.opData[thread.z]);
                } else {
                    return Operation::exec(thread, params.opData[thread.z]);
                }
            }
        }

        FK_HOST_FUSE InstantiableType build(const OperationDataType& opData) {
            return InstantiableType{ opData };
        }
        FK_HOST_FUSE InstantiableType build(const ParamsType& params) {
            return InstantiableType{ {params} };
        }
    };

    /*
    BatchRead<PlanePolicy::PROCESS_ALL, BATCH, TypeList<Op>>:
        Instantiable
        ParamsType = BatchReadParams<PlanePolicy::PROCESS_ALL, BATCH, Op>
    */
    template <size_t BATCH_, typename Op>
    struct BatchRead<PlanePolicy::PROCESS_ALL, BATCH_, TypeList<Op>> {
    private:
        using SelfType = BatchRead<PlanePolicy::PROCESS_ALL, BATCH_, TypeList<Op>>;
    public:
        FK_STATIC_STRUCT(BatchRead, SelfType)
        using Operation = Op;
        static constexpr size_t BATCH = BATCH_;
        static constexpr PlanePolicy PP = PlanePolicy::PROCESS_ALL;

        using ParamsType = BatchReadParams<BATCH, PlanePolicy::PROCESS_ALL, Operation>;
        using ReadDataType = typename Operation::ReadDataType;
        using InstanceType = ReadType;
        using OutputType = typename Operation::OutputType;
        using OperationDataType = OperationData<SelfType>;
        using InstantiableType = Read<SelfType>;
        static constexpr bool IS_FUSED_OP = Operation::IS_FUSED_OP;
        static constexpr bool THREAD_FUSION = Operation::THREAD_FUSION;

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
            return opData.params.activeThreads;
        }

        FK_HOST_FUSE InstantiableType build(const OperationDataType& opData) {
            return InstantiableType{ opData };
        }
        FK_HOST_FUSE InstantiableType build(const ParamsType& params) {
            return InstantiableType{ {params} };
        }
    };

    /*
    BatchRead<PlanePolicy::CONDITIONAL_WITH_DEFAULT, BATCH, TypeList<Op, DefType>>:
        Instantiable
        ParamsType = BatchReadParams<BATCH, PlanePolicy::CONDITIONAL_WITH_DEFAULT, Op, DefType>
    */
    template <size_t BATCH_, typename Op, typename DefType>
    struct BatchRead<PlanePolicy::CONDITIONAL_WITH_DEFAULT, BATCH_, TypeList<Op, DefType>> {
    private:
        using SelfType = BatchRead<PlanePolicy::CONDITIONAL_WITH_DEFAULT, BATCH_, TypeList<Op, DefType>>;
    public:
        FK_STATIC_STRUCT(BatchRead, SelfType)
        using Operation = Op;
        static constexpr size_t BATCH = BATCH_;
        static constexpr PlanePolicy PP = PlanePolicy::CONDITIONAL_WITH_DEFAULT;

        using ParamsType = BatchReadParams<BATCH, PlanePolicy::CONDITIONAL_WITH_DEFAULT, Operation, DefType>;
        using ReadDataType = typename Operation::ReadDataType;
        using InstanceType = ReadType;
        using OutputType = typename Operation::OutputType;
        using OperationDataType = OperationData<SelfType>;
        using InstantiableType = Read<SelfType>;
        static constexpr bool IS_FUSED_OP = Operation::IS_FUSED_OP;
        static constexpr bool THREAD_FUSION = false;

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
            return opData.params.activeThreads;
        }

        FK_HOST_FUSE InstantiableType build(const OperationDataType& opData) {
            return InstantiableType{ opData };
        }
        FK_HOST_FUSE InstantiableType build(const ParamsType& params) {
            return InstantiableType{ {params} };
        }
    };

    /*
    BatchRead<N, PP, TypeList<void>>:
        Can NOT be instantiated
        Implements BatchRead<PlanePolicy::PROCESS_ALL>::build(std::array<BATCH, IOp>) with direct {} instantiation
            For complete Op's:
                Returns BatchRead<PlanePolicy::PROCESS_ALL, BATCH, Op> with direct {} instantiation
            For incomplete Op's:
                Returns BatchRead<PlanePolicy::PROCESS_ALL, BATCH, TypeList<Op>> with direct {} instantiation
        Implements BatchRead<PlanePolicy::CONDITIONAL_WITH_DEFAULT>::build(std::array<BATCH, IOp>, usedPlanes, DefType defaultValue) with direct {} instantiation
            For complete Op's:
                Casts defaultValue to the Op::OutputType
                Returns BatchRead<PlanePolicy::CONDITIONAL_WITH_DEFAULT, BATCH, Op> with direct {} instantiation
            For incomplete IOp's
                Returns BatchRead<PlanePolicy::CONDITIONAL_WITH_DEFAULT, BATCH, TypeList<Op, DefType>> with direct {} instantiation
    */

    template <>
    struct BatchRead<PlanePolicy::PROCESS_ALL> {
    private:
        using SelfType = BatchRead<PlanePolicy::PROCESS_ALL>;
    public:
        FK_STATIC_STRUCT(BatchRead, SelfType)
        template <typename IOp, size_t BATCH>
        FK_HOST_FUSE auto build(const std::array<IOp, BATCH>& iOps) {
            return build_helper(iOps, std::make_index_sequence<BATCH>{});
        }
    private:
        template <typename IOp, size_t BATCH, size_t... Idx>
        FK_HOST_FUSE auto build_helper(const std::array<IOp, BATCH>& iOps,
                                       const std::index_sequence<Idx...>&) {
            using NewOperation = typename IOp::Operation;
#ifdef NDEBUG
            // Release mode. Use const variables and variadic template recursion for best performance
            const uint max_width = cxp::max::f(NewOperation::num_elems_x(Point(0u, 0u, 0u), iOps[Idx])...);
            const uint max_height = cxp::max::f(NewOperation::num_elems_y(Point(0u, 0u, 0u), iOps[Idx])...);
#else
            // Debug mode. Loop to avoid stack overflow
            uint max_width = NewOperation::num_elems_x(Point(0u, 0u, 0u), iOps[0]);
            uint max_height = NewOperation::num_elems_y(Point(0u, 0u, 0u), iOps[0]);
            for (int i = 1; i < BATCH; ++i) {
                max_width = cxp::max::f(max_width, NewOperation::num_elems_x(Point(0u, 0u, 0u), iOps[i]));
                max_height = cxp::max::f(max_height, NewOperation::num_elems_y(Point(0u, 0u, 0u), iOps[i]));
            }
#endif
            using BatchReadType = std::conditional_t<isCompleteOperation<NewOperation>,
                                                        BatchRead<PlanePolicy::PROCESS_ALL, BATCH, NewOperation>,
                                                        BatchRead<PlanePolicy::PROCESS_ALL, BATCH, TypeList<NewOperation>>>;
            return BatchReadType::build( { {iOps[Idx]...},
                                           ActiveThreads{ max_width, max_height, static_cast<uint>(BATCH) }});
        }
    };

    template <>
    struct BatchRead<PlanePolicy::CONDITIONAL_WITH_DEFAULT> {
    private:
        using SelfType = BatchRead<PlanePolicy::CONDITIONAL_WITH_DEFAULT>;
    public:
        FK_STATIC_STRUCT(BatchRead, SelfType)
        template <typename IOp, size_t BATCH, typename DefaultType, typename... ArrayTypes>
        FK_HOST_FUSE auto build(const std::array<IOp, BATCH>& iOps, const int& usedPlanes, const DefaultType& defaultValue) {
            return build_helper(iOps, usedPlanes, defaultValue, std::make_index_sequence<BATCH>{});
        }
    private:
        template <typename IOp, size_t BATCH, typename DefaultType, size_t... Idx>
        FK_HOST_FUSE auto build_helper(const std::array<IOp, BATCH>& iOps,
                                       const int& usedPlanes, const DefaultType& defaultValue,
                                       const std::index_sequence<Idx...>&) {
            using NewOperation = typename IOp::Operation;
            const uint max_width = cxp::max::f(NewOperation::num_elems_x(Point(0u, 0u, 0u), iOps[Idx])...);
            const uint max_height = cxp::max::f(NewOperation::num_elems_y(Point(0u, 0u, 0u), iOps[Idx])...);

            using NewOutputType = std::conditional_t<std::is_same_v<typename NewOperation::OutputType, NullType>, DefaultType, typename NewOperation::OutputType>;
            using BatchReadType = std::conditional_t<isCompleteOperation<NewOperation>,
                                                        BatchRead<PlanePolicy::CONDITIONAL_WITH_DEFAULT, BATCH, NewOperation>,
                                                        BatchRead<PlanePolicy::CONDITIONAL_WITH_DEFAULT, BATCH, TypeList<NewOperation, NewOutputType>>>;
            return BatchReadType::build( { {iOps[Idx]...}, usedPlanes, cxp::cast<NewOutputType>::f(defaultValue),
                                           ActiveThreads{ max_width, max_height, static_cast<uint>(BATCH) }});
        }
    };
    // ##################### END BATCH_READ #####################


    // ##################### BATCH_WRITE #####################
    template <size_t BATCH, typename Operation = void>
    struct BatchWrite {
    private:
        using SelfType = BatchWrite<BATCH, Operation>;
    public:
        FK_STATIC_STRUCT(BatchWrite, SelfType)
        using Parent = WriteOperation<typename Operation::InputType, typename Operation::ParamsType[BATCH],
                                      typename Operation::WriteDataType,
                                      Operation::THREAD_FUSION ? TF::ENABLED : TF::DISABLED, BatchWrite<BATCH, Operation>>;
        DECLARE_WRITE_PARENT_BASIC

        template <uint ELEMS_PER_THREAD = 1>
        FK_HOST_DEVICE_FUSE void exec(const Point& thread,
            const ThreadFusionType<InputType, ELEMS_PER_THREAD, InputType>& input,
            const ParamsType& params) {
            if constexpr (THREAD_FUSION) {
                Operation::template exec<ELEMS_PER_THREAD>(thread, input, params[thread.z]);
            } else {
                Operation::exec(thread, input, params[thread.z]);
            }
        }
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return Operation::num_elems_x(thread, opData.params[thread.z]);
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const OperationDataType& opData) {
            return Operation::pitch(thread, opData.params[thread.z]);
        }
        // Build WriteBatch from array of IOps
        FK_HOST_FUSE InstantiableType build(const std::array<Instantiable<Operation>, BATCH>& iOps) {
            return build_helper(iOps, std::make_index_sequence<BATCH>{});
        }
    private:
        template <size_t... Idx>
        FK_HOST_FUSE InstantiableType build_helper(const std::array<Instantiable<Operation>, BATCH>& iOps,
                                                   const std::index_sequence<Idx...>&) {
            return { {{(iOps[Idx].params)...}} };
        }
    };

    template <size_t BATCH>
    struct BatchWrite<BATCH, void> {
    private:
        using SelfType = BatchWrite<BATCH, void>;
    public:
        FK_STATIC_STRUCT(BatchWrite, SelfType)
        using InstaceType = WriteType;
        template <typename IOp> FK_HOST_FUSE auto build(const std::array<IOp, BATCH>& iOps) {
            return BatchWrite<BATCH, typename IOp::Operation>::build(iOps);
        }
    };

// MEMORY OPERATION BATCH BUILDERS

// BATCH BUILDERS FOR READ AND READBACK OPERATIONS
#define DECLARE_PARENT_BATCH_BUILDER                                                                                   \
   template <size_t B, typename... ArrayTypes>                                                                         \
    FK_HOST_FUSE auto build_batch(const std::array<ArrayTypes, B>&... arrays) {                                        \
        return BatchUtils::template build_batch<typename Parent::Child>(arrays...);                                    \
    }

#define BATCHREAD_BUILDERS \
    template <size_t B, typename... ArrayTypes>                                                                        \
    FK_HOST_FUSE auto build(const std::array<ArrayTypes, B>&... arrays) {                                              \
        const auto iOpArray = build_batch(arrays...);                                                                  \
        return BatchRead<PlanePolicy::PROCESS_ALL>::build(iOpArray);                                                   \
    }                                                                                                                  \
    template <size_t B, typename DefaultValueType, typename... ArrayTypes>                                             \
    FK_HOST_FUSE auto build(const int& usedPlanes, const DefaultValueType& defaultValue,                               \
                            const std::array<ArrayTypes, B>&...arrays) {                                               \
        const auto iOpArray = build_batch(arrays...);                                                                  \
        return BatchRead<PlanePolicy::CONDITIONAL_WITH_DEFAULT>::build(iOpArray, usedPlanes, defaultValue);            \
    }

#define DECLARE_PARENT_READBATCH_BUILDERS                                                                              \
    DECLARE_PARENT_BATCH_BUILDER                                                                                       \
    BATCHREAD_BUILDERS

// DECLARE PARENT FOR READ OPERATIONS
#define DECLARE_READ_PARENT                                                                                            \
  DECLARE_READ_PARENT_BASIC                                                                                            \
  DECLARE_PARENT_READBATCH_BUILDERS

// DECLARE PARENT FOR WRITE OPERATIONS
#define DECLARE_WRITE_PARENT_BATCH                                                                                     \
  template <size_t BATCH_N, typename FirstType, typename... ArrayTypes>                                                \
  FK_HOST_FUSE auto build_batch(const std::array<FirstType, BATCH_N> &firstInstance, const ArrayTypes &...arrays) {    \
    return BatchUtils::template build_batch<typename Parent::Child>(firstInstance, arrays...);                         \
  }                                                                                                                    \
  template <size_t BATCH_N, typename FirstType, typename... ArrayTypes>                                                \
  FK_HOST_FUSE auto build(const std::array<FirstType, BATCH_N> &firstInstance, const ArrayTypes &...arrays) {          \
    const auto iOpArray = build_batch(firstInstance, arrays...);                                                       \
    return BatchWrite<BATCH_N>::build(iOpArray);                                                                       \
  }
#define DECLARE_WRITE_PARENT                                                                                           \
  DECLARE_WRITE_PARENT_BASIC                                                                                           \
  DECLARE_WRITE_PARENT_BATCH

// DECLARE PARENT FOR READBACK OPERATIONS
#define DECLARE_READBACK_PARENT                                                                                        \
  DECLARE_READBACK_PARENT_BASIC                                                                                        \
  DECLARE_PARENT_READBATCH_BUILDERS

#define DECLARE_INCOMPLETEREADBACK_PARENT                                                                \
  DECLARE_INCOMPLETEREADBACK_PARENT_BASIC                                                                              \
  DECLARE_PARENT_READBATCH_BUILDERS

  // END MEMORY OPERATIONS BATCH BUILDERS
  // END BATCH OPERATIONS
} // namespace fk

#endif