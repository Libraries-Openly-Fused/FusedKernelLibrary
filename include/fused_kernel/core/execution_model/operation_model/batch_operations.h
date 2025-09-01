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
        std::integral_constant<bool, has_batch<T>::value&& has_operation<T>::value> {
    };
    // Helper variable template
    template <typename T>
    static constexpr bool isBatchOperation = IsBatchOperation<T>::value;

    enum class PlanePolicy { PROCESS_ALL = 0, CONDITIONAL_WITH_DEFAULT = 1 };

    struct BatchOperation {
        FK_STATIC_STRUCT(BatchOperation, BatchOperation)
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

    template <size_t BATCH, enum PlanePolicy PP, typename OpParamsType, typename DefaultType>
    struct BatchReadParams;

    template <size_t BATCH, typename Operation, typename DefaultType>
    struct BatchReadParams<BATCH, PlanePolicy::CONDITIONAL_WITH_DEFAULT, Operation, DefaultType> {
        OperationData<Operation> opData[BATCH];
        int usedPlanes;
        DefaultType default_value;
        ActiveThreads activeThreads;
    };

    template <size_t BATCH, typename Operation, typename DefaultType>
    struct BatchReadParams<BATCH, PlanePolicy::PROCESS_ALL, Operation, DefaultType> {
        OperationData<Operation> opData[BATCH];
        ActiveThreads activeThreads;
    };

    template <size_t BATCH, typename BatchOp>
    struct BatchReadBase {
    private:
        using SelfType = BatchReadBase<BATCH, BatchOp>;
    public:
        FK_STATIC_STRUCT(BatchReadBase, SelfType)
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationData<BatchOp>& opData) {
            return BatchOp::Operation::num_elems_x(thread, opData.params.opData[thread.z]);
        }
        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const OperationData<BatchOp>& opData) {
            return BatchOp::Operation::num_elems_y(thread, opData.params.opData[thread.z]);
        }
        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const OperationData<BatchOp>& opData) {
            return BatchOp::BATCH;
        }
        FK_HOST_DEVICE_FUSE uint pitch(const Point& thread, const OperationData<BatchOp>& opData) {
            return BatchOp::Operation::pitch(thread, opData.params.opData[thread.z]);
        }
        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationData<BatchOp>& opData) {
            return opData.params.activeThreads;
        }
    };

    template <size_t BATCH, enum PlanePolicy PP = PlanePolicy::PROCESS_ALL, typename Operation = void, typename OutputType = NullType>
    struct BatchRead;

    /// @brief struct BatchRead
    /// @tparam BATCH: number of thread planes and number of data planes to process
    /// @tparam Operation: the read Operation to perform on the data
    /// @tparam PP: enum to select if all planes will be processed equally, or only some
    /// with the remainder not reading and returning a default value
    template <size_t BATCH_, typename Operation_, typename OutputType_>
    struct BatchRead<BATCH_, PlanePolicy::PROCESS_ALL, Operation_, OutputType_> final
        : public BatchReadBase<BATCH_, BatchRead<BATCH_, PlanePolicy::PROCESS_ALL, Operation_, OutputType_>> {
    private:
        using SelfType = BatchRead<BATCH_, PlanePolicy::PROCESS_ALL, Operation_, OutputType_>;
    public:
        FK_STATIC_STRUCT(BatchRead, SelfType)
        using Operation = Operation_;
        static constexpr size_t BATCH = BATCH_;
        static constexpr PlanePolicy PP = PlanePolicy::PROCESS_ALL;
        using Parent = ReadOperation<typename Operation::ReadDataType,
                                     BatchReadParams<BATCH, PP, Operation, typename Operation::OutputType>,
                                     typename Operation::OutputType, Operation::THREAD_FUSION ? TF::ENABLED : TF::DISABLED,
                                     BatchRead<BATCH, PlanePolicy::PROCESS_ALL, Operation_, OutputType_>>;
        DECLARE_READ_PARENT_BASIC

        static_assert(isAnyReadType<Operation>, "The Operation is not of any Read type");

        template <uint ELEMS_PER_THREAD = 1>
        FK_HOST_DEVICE_FUSE auto exec(const Point& thread, const ParamsType& params) {
            if constexpr (THREAD_FUSION) {
                return Operation::template exec<ELEMS_PER_THREAD>(thread, params.opData[thread.z]);
            } else {
                return Operation::exec(thread, params.opData[thread.z]);
            }
        }
#ifndef NVRTC_COMPILER
        // Build BatchRead from an array of InstantiableOperations
        template <typename IOp>
        FK_HOST_FUSE std::enable_if_t<isAnyReadType<IOp>, InstantiableType>
            build(const std::array<IOp, BATCH>& instantiableOperations) {
            static_assert(isAnyReadType<IOp>);
            return build_helper(instantiableOperations, std::make_integer_sequence<int, BATCH>{});
        }
        // Build BatchRead from arrays of Operation::build() parameters
        template <size_t BATCH_N, typename FirstType, typename... ArrayTypes>
        FK_HOST_FUSE std::enable_if_t<!isAnyReadType<FirstType>, InstantiableType>
            build(const std::array<FirstType, BATCH_N>& firstInstance, const ArrayTypes &...arrays) {
            const auto arrayOfIOps = Operation::build_batch(firstInstance, arrays...);
            return BatchRead<BATCH_N, PP>::build(arrayOfIOps);
        }

    private:
        template <int... Idx>
        FK_HOST_FUSE InstantiableType build_helper(const std::array<Instantiable<Operation>, BATCH>& instantiableOperations,
            const std::integer_sequence<int, Idx...>&) {
#ifdef NDEBUG
            // Release mode. Use const variables and variadic template recursion for best performance
            const uint max_width = cxp::max(Operation::num_elems_x(Point(0u, 0u, 0u), instantiableOperations[Idx])...);
            const uint max_height = cxp::max(Operation::num_elems_y(Point(0u, 0u, 0u), instantiableOperations[Idx])...);
#else
            // Debug mode. Loop to avoid stack overflow
            uint max_width = Operation::num_elems_x(Point(0u, 0u, 0u), instantiableOperations[0]);
            uint max_height = Operation::num_elems_y(Point(0u, 0u, 0u), instantiableOperations[0]);
            for (int i = 1; i < BATCH; ++i) {
                max_width = cxp::max(max_width, Operation::num_elems_x(Point(0u, 0u, 0u), instantiableOperations[i]));
                max_height = cxp::max(max_height, Operation::num_elems_y(Point(0u, 0u, 0u), instantiableOperations[i]));
            }
#endif

#if defined(_MSC_VER) && (_MSC_VER >= 1910) && (_MSC_VER < 1920)
            // VS2017 compilers need the BatchReadParams type specified
            return {   
                       BatchReadParams<BATCH, PP, Operation, typename Operation::OutputType>{
                           {instantiableOperations[Idx]...},
                           {max_width, max_height, static_cast<uint>(BATCH)}
                       }
                   };
#else
            // gcc, clang or VS2022
            return { {{{static_cast<OperationData<Operation>>(instantiableOperations[Idx])...},
                      ActiveThreads{max_width, max_height, static_cast<uint>(BATCH)}}} };
#endif // _MSC_VER
        }
#endif // NVRTC_COMPILER
    };

    template <typename T>
    using NullTypeToUchar = std::conditional_t<std::is_same_v<T, NullType>, uchar, T>;
    template <typename Nullable, typename Alternative>
    using NullTypeToAlternative = std::conditional_t<std::is_same_v<Nullable, NullType>, Alternative, Nullable>;

    template <size_t BATCH_, typename Operation_, typename OutputType_>
    struct BatchRead<BATCH_, PlanePolicy::CONDITIONAL_WITH_DEFAULT, Operation_, OutputType_> final
        : public BatchReadBase<BATCH_, BatchRead<BATCH_, PlanePolicy::CONDITIONAL_WITH_DEFAULT, Operation_, OutputType_>> {
    private:
        using SelfType = BatchRead<BATCH_, PlanePolicy::CONDITIONAL_WITH_DEFAULT, Operation_, OutputType_>;
    public:
        FK_STATIC_STRUCT(BatchRead, SelfType)
        using Operation = Operation_;
        static constexpr int BATCH = BATCH_;
        static constexpr PlanePolicy PP = PlanePolicy::CONDITIONAL_WITH_DEFAULT;
        using Parent = ReadOperation<NullTypeToUchar<typename Operation::ReadDataType>,
                                     BatchReadParams<BATCH, PP, Operation, NullTypeToAlternative<typename Operation::OutputType, OutputType_>>,
                                     NullTypeToAlternative<typename Operation::OutputType, OutputType_>,
                                     Operation::THREAD_FUSION ? TF::ENABLED : TF::DISABLED, BatchRead<BATCH, PP, Operation, OutputType_>>;
        DECLARE_READ_PARENT_BASIC
        static_assert(isAnyReadType<Operation>, "The Operation is not of any Read type");

        template <uint ELEMS_PER_THREAD = 1>
        FK_HOST_DEVICE_FUSE auto exec(const Point& thread, const ParamsType& params) {
            if (params.usedPlanes <= thread.z) {
                return params.default_value;
            } else {
                return Operation::exec(thread, params.opData[thread.z]);
            }
        }

        template <typename IOp, typename DefaultValueType>
        FK_HOST_FUSE auto build(const std::array<IOp, BATCH>& instantiableOperations,
                                const int& usedPlanes, const DefaultValueType& defaultValue) {
            static_assert(isAnyReadType<IOp>);
            if constexpr (std::is_same_v<OutputType, NullType>) {
                return BatchRead<BATCH, PP, typename IOp::Operation, DefaultValueType>::build_helper(
                    instantiableOperations, usedPlanes, defaultValue, std::make_integer_sequence<int, BATCH>{});
            } else {
                static_assert(std::is_same_v<OutputType, DefaultValueType>, "OutputType and DefaultValueType should be the same.");
                return build_helper(instantiableOperations, usedPlanes, defaultValue, std::make_integer_sequence<int, BATCH>{});
            }
        }
        template <size_t BATCH_N, typename DefaultValueType, typename FirstType, typename... ArrayTypes>
        FK_HOST_FUSE auto build(const int& usedPlanes, const DefaultValueType& defaultValue,
                                const std::array<FirstType, BATCH_N>& firstInstance, const ArrayTypes &...arrays) {
            const auto arrayOfIOps = Operation::build_batch(firstInstance, arrays...);
            return BatchRead<BATCH_N, PP>::build(arrayOfIOps, usedPlanes, defaultValue);
        }

    private:
        template <int... Idx>
        FK_HOST_FUSE InstantiableType build_helper(const std::array<Instantiable<Operation>, BATCH>& instantiableOperations,
                                                   const int& usedPlanes, const OutputType& defaultValue,
                                                   const std::integer_sequence<int, Idx...>&) {
            const uint max_width = cxp::max(Operation::num_elems_x(Point(0u, 0u, 0u), instantiableOperations[Idx])...);
            const uint max_height = cxp::max(Operation::num_elems_y(Point(0u, 0u, 0u), instantiableOperations[Idx])...);
#if defined(_MSC_VER) && (_MSC_VER >= 1910) && (_MSC_VER < 1920)
            // VS2017 compilers need the BatchReadParams type specified
            return {
                       BatchReadParams<BATCH, PP, Operation, NullTypeToAlternative<typename Operation::OutputType, OutputType_>>{
                           {instantiableOperations[Idx]...},
                           usedPlanes,
                           defaultValue,
                           {max_width, max_height, static_cast<uint>(BATCH)}
                       }
            };
#else
            // gcc, clang or VS2022
            return { {{{static_cast<OperationData<Operation>>(instantiableOperations[Idx])...},
                      usedPlanes,
                      defaultValue,
                      ActiveThreads{max_width, max_height, BATCH}}} };
#endif
        }
    };

    template <size_t BATCH>
    struct BatchRead<BATCH, PlanePolicy::PROCESS_ALL, void> {
    private:
        using SelfType = BatchRead<BATCH, PlanePolicy::PROCESS_ALL, void, NullType>;
    public:
        FK_STATIC_STRUCT(BatchRead, SelfType)
        template <typename IOp> FK_HOST_FUSE auto build(const std::array<IOp, BATCH>& instantiableOperations) {
            return BatchRead<BATCH, PlanePolicy::PROCESS_ALL, typename IOp::Operation>::build(instantiableOperations);
        }
    };

    template <size_t BATCH>
    struct BatchRead<BATCH, PlanePolicy::CONDITIONAL_WITH_DEFAULT, void> {
    private:
        using SelfType = BatchRead<BATCH, PlanePolicy::CONDITIONAL_WITH_DEFAULT, void, NullType>;
    public:
        FK_STATIC_STRUCT(BatchRead, SelfType)
        template <typename IOp, typename DefaultValueType>
        FK_HOST_FUSE auto build(const std::array<IOp, BATCH>& instantiableOperations, const int& usedPlanes,
            const DefaultValueType& defaultValue) {
            if constexpr (std::is_same_v<typename IOp::Operation::OutputType, NullType>) {
                return BatchRead<BATCH, PlanePolicy::CONDITIONAL_WITH_DEFAULT, typename IOp::Operation, DefaultValueType>::build(
                    instantiableOperations, usedPlanes, defaultValue);
            }
            else {
                return BatchRead<BATCH, PlanePolicy::CONDITIONAL_WITH_DEFAULT, typename IOp::Operation>::build(instantiableOperations,
                    usedPlanes, defaultValue);
            }
        }
    };

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
        template <typename IOp>
        FK_HOST_FUSE std::enable_if_t<isWriteType<IOp>, InstantiableType> build(const std::array<IOp, BATCH>& iOps) {
            static_assert(isWriteType<IOp>, "The IOps in the array are not WriteType");
            return build_helper(iOps, std::make_integer_sequence<int, BATCH>{});
        }
        template <size_t BATCH_N, typename FirstType, typename... ArrayTypes>
        FK_HOST_FUSE std::enable_if_t<!isWriteType<FirstType>, InstantiableType>
            build(const std::array<FirstType, BATCH_N>& firstInstance, const ArrayTypes &...arrays) {
            const auto arrayOfIOps = Operation::build_batch(firstInstance, arrays...);
            return BatchWrite<BATCH_N>::build(arrayOfIOps);
        }

    private:
        template <int... Idx>
        FK_HOST_FUSE InstantiableType build_helper(const std::array<Instantiable<Operation>, BATCH>& iOps,
            const std::integer_sequence<int, Idx...>&) {
            return { {{(iOps[Idx].params)...}} };
        }
    };

    template <size_t BATCH> struct BatchWrite<BATCH, void> {
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
    template <typename ReadOperation>
    struct ReadOperationBatchBuilders {
    private:
        using SelfType = ReadOperationBatchBuilders<ReadOperation>;
        FK_STATIC_STRUCT(ReadOperationBatchBuilders, SelfType)
    public:
        template <size_t BATCH_N, typename FirstType, typename... ArrayTypes>
        FK_HOST_FUSE auto build(const std::array<FirstType, BATCH_N>& firstInstance, const ArrayTypes &...arrays) {
            using BuilderType = BatchRead<BATCH_N, PlanePolicy::PROCESS_ALL, ReadOperation>;
            return BuilderType::build(firstInstance, arrays...);
        }
        template <size_t BATCH_N, typename DefaultValueType, typename FirstType, typename... ArrayTypes>
        FK_HOST_FUSE auto build(const int& usedPlanes, const DefaultValueType& defaultValue,
                                const std::array<FirstType, BATCH_N>& firstInstance, const ArrayTypes &...arrays) {
            using BuilderType = BatchRead<BATCH_N, PlanePolicy::CONDITIONAL_WITH_DEFAULT, ReadOperation>;
            if constexpr (sizeof...(ArrayTypes) > 0) {
                return BuilderType::build(usedPlanes, defaultValue, firstInstance, arrays...);
            } else {
                if constexpr (isAnyReadType<FirstType>) {
                    return BuilderType::build(firstInstance, usedPlanes, defaultValue);
                } else {
                    static_assert(!isAnyReadType<FirstType>, "FirstType is a Read or ReadBack type and should not be.");
                    return BuilderType::build(usedPlanes, defaultValue, firstInstance);
                }
            }
        }
    };

#define DECLARE_READ_PARENT_BATCH                                                                                      \
  template <size_t BATCH_N, typename FirstType, typename... ArrayTypes>                                                \
  FK_HOST_FUSE auto build_batch(const std::array<FirstType, BATCH_N> &firstInstance, const ArrayTypes &...arrays) {    \
    return BatchOperation::build_batch<typename Parent::Child>(firstInstance, arrays...);                              \
  }                                                                                                                    \
  template <size_t BATCH_N, typename FirstType, typename... ArrayTypes>                                                \
  FK_HOST_FUSE auto build(const std::array<FirstType, BATCH_N> &firstInstance, const ArrayTypes &...arrays) {          \
    return ReadOperationBatchBuilders<typename Parent::Child>::build(firstInstance, arrays...);                                        \
  }                                                                                                                    \
  template <size_t BATCH_N, typename DefaultValueType, typename FirstType, typename... ArrayTypes>                     \
  FK_HOST_FUSE auto build(const int &usedPlanes, const DefaultValueType &defaultValue,                                 \
                          const std::array<FirstType, BATCH_N> &firstInstance, const ArrayTypes &...arrays) {          \
    return ReadOperationBatchBuilders<typename Parent::Child>::build(usedPlanes, defaultValue, firstInstance, arrays...);              \
  }

#define DECLARE_READ_PARENT                                                                                            \
  DECLARE_READ_PARENT_BASIC                                                                                            \
  DECLARE_READ_PARENT_BATCH

#define DECLARE_WRITE_PARENT_BATCH                                                                                     \
  template <size_t BATCH_N, typename FirstType, typename... ArrayTypes>                                                \
  FK_HOST_FUSE auto build_batch(const std::array<FirstType, BATCH_N> &firstInstance, const ArrayTypes &...arrays) {    \
    return BatchOperation::build_batch<typename Parent::Child>(firstInstance, arrays...);                              \
  }                                                                                                                    \
  template <size_t BATCH_N, typename FirstType, typename... ArrayTypes>                                                \
  FK_HOST_FUSE auto build(const std::array<FirstType, BATCH_N> &firstInstance, const ArrayTypes &...arrays) {          \
    return BatchWrite<BATCH_N, typename Parent::Child>::build(firstInstance, arrays...);                               \
  }
#define DECLARE_WRITE_PARENT                                                                                           \
  DECLARE_WRITE_PARENT_BASIC                                                                                           \
  DECLARE_WRITE_PARENT_BATCH

    // DECLARE_READBACK_PARENT
#define DECLARE_READBACK_PARENT                                                                                        \
  DECLARE_READBACK_PARENT_ALIAS                                                                                        \
  FK_DEVICE_FUSE OutputType exec(const Point &thread, const OperationDataType &opData) {                               \
    return Parent::exec(thread, opData);                                                                               \
  }                                                                                                                    \
  FK_HOST_DEVICE_FUSE auto build(const OperationDataType &opData) { return Parent::build(opData); }                    \
  FK_HOST_DEVICE_FUSE auto build(const ParamsType &params, const BackIOp &backIOp) {                        \
    return Parent::build(params, backIOp);                                                                       \
  }                                                                                                                    \
  DECLARE_READ_PARENT_BATCH

    template <typename ReadOperation>
    struct ReadBackIncompleteOperationBatchBuilders {
    private:
        using SelfType = ReadBackIncompleteOperationBatchBuilders<ReadOperation>;
    public:
        FK_STATIC_STRUCT(ReadBackIncompleteOperationBatchBuilders, SelfType)
        template <size_t BATCH_N, typename FirstType, typename... ArrayTypes>
        FK_HOST_FUSE auto build(const std::array<FirstType, BATCH_N>& firstInstance, const ArrayTypes &...arrays) {
            const auto arrayOfIOps = BatchOperation::build_batch<ReadOperation>(firstInstance, arrays...);
            return BatchRead<BATCH_N, PlanePolicy::PROCESS_ALL>::build(arrayOfIOps);
        }
        template <size_t BATCH_N, typename DefaultValueType, typename FirstType, typename... ArrayTypes>
        FK_HOST_FUSE auto build(const int& usedPlanes, const DefaultValueType& defaultValue,
                                const std::array<FirstType, BATCH_N>& firstInstance, const ArrayTypes &...arrays) {
            using BuilderType = BatchRead<BATCH_N, PlanePolicy::CONDITIONAL_WITH_DEFAULT>;
            if constexpr (sizeof...(ArrayTypes) > 0) {
                const auto arrayOfIOps = BatchOperation::build_batch<ReadOperation>(firstInstance, arrays...);
                return BuilderType::build(arrayOfIOps, usedPlanes, defaultValue);
            } else {
                if constexpr (isAnyReadType<FirstType>) {
                    return BuilderType::build(firstInstance, usedPlanes, defaultValue);
                } else {
                    const auto arrayOfIOps = BatchOperation::build_batch<ReadOperation>(firstInstance);
                    return BuilderType::build(arrayOfIOps, usedPlanes, defaultValue);
                }
            }
        }
    };
    template <typename ReadOperation>
    using RBIncompleteOpBB = ReadBackIncompleteOperationBatchBuilders<ReadOperation>;

#define DECLARE_READBACK_PARENT_BATCH_INCOMPLETE                                                                       \
  template <size_t BATCH_N, typename FirstType, typename... ArrayTypes>                                                \
  FK_HOST_FUSE auto build_batch(const std::array<FirstType, BATCH_N> &firstInstance, const ArrayTypes &...arrays) {    \
    return BatchOperation::build_batch<typename Parent::Child>(firstInstance, arrays...);                              \
  }                                                                                                                    \
  template <size_t BATCH_N, typename FirstType, typename... ArrayTypes>                                                \
  FK_HOST_FUSE auto build(const std::array<FirstType, BATCH_N> &firstInstance, const ArrayTypes &...arrays) {          \
    return RBIncompleteOpBB<typename Parent::Child>::build(firstInstance, arrays...);                                                  \
  }                                                                                                                    \
  template <size_t BATCH_N, typename DefaultValueType, typename FirstType, typename... ArrayTypes>                     \
  FK_HOST_FUSE auto build(const int &usedPlanes, const DefaultValueType &defaultValue,                                 \
                          const std::array<FirstType, BATCH_N> &firstInstance, const ArrayTypes &...arrays) {          \
    return RBIncompleteOpBB<typename Parent::Child>::build(usedPlanes, defaultValue, firstInstance, arrays...);                        \
  }

#define DECLARE_READBACK_PARENT_INCOMPLETE                                                                             \
  DECLARE_READBACK_PARENT_ALIAS                                                                                        \
  FK_HOST_DEVICE_FUSE auto build(const OperationDataType &opData) { return Parent::build(opData); }                    \
  FK_HOST_DEVICE_FUSE auto build(const ParamsType &params, const BackIOp &backIOp) {                        \
    return Parent::build(params, backIOp);                                                                       \
  }                                                                                                                    \
  DECLARE_READBACK_PARENT_BATCH_INCOMPLETE
  // END MEMORY OPERATIONS BATCH BUILDERS
  // END BATCH OPERATIONS
} // namespace fk

#endif