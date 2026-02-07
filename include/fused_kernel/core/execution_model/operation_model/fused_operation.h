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

#ifndef FK_FUSED_OPERATION
#define FK_FUSED_OPERATION

#include <fused_kernel/core/execution_model/operation_model/batch_operations.h>
#include <fused_kernel/core/execution_model/operation_model/operation_tuple.h>

namespace fk {
    // FusedOperation
    /*namespace fused_operation_impl {
        // FusedOperation implementation struct
        template <typename Operation>
        FK_HOST_DEVICE_CNST typename Operation::OutputType
            exec_operate(const typename Operation::InputType& i_data) {
            return Operation::exec(i_data);
        }
        template <typename Tuple_>
        FK_HOST_DEVICE_CNST auto
            exec_operate(const typename Tuple_::Operation::InputType& i_data, const Tuple_& tuple) {
            using Operation = typename Tuple_::Operation;
            static_assert(isComputeType<Operation>, "The operation is WriteType and shouldn't be.");
            if constexpr (isUnaryType<Operation>) {
                return Operation::exec(i_data);
            } else {
                return Operation::exec(i_data, tuple.instance);
            }
        }
        template <typename Tuple_>
        FK_HOST_DEVICE_CNST auto exec_operate(const Point& thread, const Tuple_& tuple) {
            return Tuple_::Operation::exec(thread, tuple.instance);
        }
        template <typename Tuple_>
        FK_HOST_DEVICE_CNST auto exec_operate(const Point& thread,
            const typename Tuple_::Operation::InputType& i_data,
            const Tuple_& tuple) {
            using Operation = typename Tuple_::Operation;
            if constexpr (isComputeType<Operation> && !isUnaryType<Operation>) {
                return Operation::exec(i_data, tuple.instance);
            } else if constexpr (isUnaryType<Operation>) {
                return Operation::exec(i_data);
            } else if constexpr (isWriteType<Operation>) {
                // Assuming the behavior of a MidWriteType IOp
                Operation::exec(thread, i_data, tuple.instance);
                return i_data;
            } else {
                static_assert(isMidWriteType<Operation>, "The operation should be MidWriteType, and it's not.");
                // We are executing another FusedOperation that is MidWriteType
                return Operation::exec(thread, i_data, tuple.instance);
            }
        }

        template <typename FirstOp, typename... RemOps>
        FK_HOST_DEVICE_CNST auto tuple_operate(const typename FirstOp::InputType& i_data) {
            if constexpr (sizeof...(RemOps) == 0) {
                return FirstOp::exec(i_data);
            } else {
                return tuple_operate<RemOps...>(FirstOp::exec(i_data));
            }
        }

        template <typename FirstOp, typename... RemOps>
        FK_HOST_DEVICE_CNST auto tuple_operate(const typename FirstOp::InputType& i_data,
            const OperationTuple<FirstOp, RemOps...>& tuple) {
            const auto result = exec_operate(i_data, tuple);
            if constexpr (sizeof...(RemOps) > 0) {
                if constexpr (has_next_v<OperationTuple<FirstOp, RemOps...>>) {
                    return tuple_operate(result, tuple.next);
                } else {
                    return tuple_operate<RemOps...>(result);
                }
            } else {
                return result;
            }
        }
        template <typename FirstOp, typename... RemOps>
        FK_HOST_DEVICE_CNST auto tuple_operate(const Point& thread,
            const typename FirstOp::InputType& input,
            const OperationTuple<FirstOp, RemOps...>& tuple) {
            const auto result = exec_operate(thread, input, tuple);
            if constexpr (sizeof...(RemOps) > 0) {
                if constexpr (has_next_v<OperationTuple<FirstOp, RemOps...>>) {
                    return tuple_operate(thread, result, tuple.next);
                } else {
                    return tuple_operate<RemOps...>(result);
                }
            } else {
                return result;
            }
        }
        template <typename FirstOp, typename... RemOps>
        FK_HOST_DEVICE_CNST auto tuple_operate(const Point& thread,
            const OperationTuple<FirstOp, RemOps...>& tuple) {
            const auto result = exec_operate(thread, tuple);
            if constexpr (sizeof...(RemOps) > 0) {
                if constexpr (has_next_v<OperationTuple<FirstOp, RemOps...>>) {
                    return tuple_operate(thread, result, tuple.next);
                } else {
                    return tuple_operate<RemOps...>(result);
                }
            } else {
                return result;
            }
        }
    } // namespace fused_operation_impl

    template <typename Enabler, typename... Operations>
    struct FusedOperationOutputType;

    template <typename Operation>
    struct FusedOperationOutputType<std::enable_if_t<isWriteType<Operation>>, Operation> {
        using type = typename Operation::InputType;
    };

    template <typename Operation>
    struct FusedOperationOutputType<std::enable_if_t<!isWriteType<Operation>>, Operation> {
        using type = typename Operation::OutputType;
    };

    template <typename... Operations>
    using FOOT = typename FusedOperationOutputType<void, Operations...>::type;

    template <typename Enabler, typename... Operations>
    struct FusedOperation_;

    template <typename FirstOp, typename... RemOps>
    struct FusedOperation_<std::enable_if_t<allUnaryTypes<FirstOp, RemOps...> && (sizeof...(RemOps) + 1 > 1)>, FirstOp, RemOps...> {
    private:
        using SelfType = FusedOperation_<std::enable_if_t<allUnaryTypes<FirstOp, RemOps...> && (sizeof...(RemOps) + 1 > 1)>, FirstOp, RemOps...>;
    public:
        FK_STATIC_STRUCT(FusedOperation_, SelfType)
        using Parent =
            UnaryOperation<typename FirstOp::InputType,
            typename LastType_t<RemOps...>::OutputType,
            FusedOperation_<std::enable_if_t<allUnaryTypes<FirstOp, RemOps...> && (sizeof...(RemOps) + 1 > 1)>, FirstOp, RemOps...>,
            true>;
        DECLARE_UNARY_PARENT

        using Operations = TypeList<FirstOp, RemOps...>;
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            return fused_operation_impl::tuple_operate<FirstOp, RemOps...>(input);
        }
    };

    template <typename Operation>
    struct FusedOperation_<std::enable_if_t<isUnaryType<Operation>>, Operation> {
    private:
        using SelfType = FusedOperation_<std::enable_if_t<isUnaryType<Operation>>, Operation>;
    public:
        FK_STATIC_STRUCT(FusedOperation_, SelfType)
        using Parent =
            UnaryOperation<typename Operation::InputType,
            typename Operation::OutputType,
            FusedOperation_<std::enable_if_t<isUnaryType<Operation>>, Operation>,
            true>;
        DECLARE_UNARY_PARENT

        using Operations = TypeList<Operation>;
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            return Operation::exec(input);
        }
    };

    template <typename... Operations>
    struct FusedOperation_<std::enable_if_t<isComputeType<FirstType_t<Operations...>> &&
                           !allUnaryTypes<Operations...>>, Operations...> {
    private:
        using SelfType = FusedOperation_<std::enable_if_t<isComputeType<FirstType_t<Operations...>> &&
                                         !allUnaryTypes<Operations...>>, Operations...>;
    public:
        FK_STATIC_STRUCT(FusedOperation_, SelfType)
        using Parent =
            BinaryOperation<typename FirstType_t<Operations...>::InputType,
            OperationTuple<Operations...>,
            FOOT<LastType_t<Operations...>>,
            SelfType, true>;
        DECLARE_BINARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input,
                                            const ParamsType& params) {
            return fused_operation_impl::tuple_operate(input, params);
        }

    };

    template <typename... Operations>
    struct FusedOperation_<std::enable_if_t<isAnyReadType<FirstType_t<Operations...>>>, Operations...> {
    private:
        static constexpr bool isTFEnabled = std::is_same_v<typename FirstType_t<Operations...>::ReadDataType, FOOT<LastType_t<Operations...>>> && ((sizeof...(Operations) > 1) ? false :
            FirstType_t<Operations...>::THREAD_FUSION);
        using SelfType = FusedOperation_<std::enable_if_t<isAnyReadType<FirstType_t<Operations...>>>, Operations...>;
    public:
        FK_STATIC_STRUCT(FusedOperation_, SelfType)
        using Parent = ReadOperation<typename FirstType_t<Operations...>::ReadDataType,
                                     OperationTuple<Operations...>,
                                     FOOT<LastType_t<Operations...>>,
                                     isTFEnabled ? TF::ENABLED : TF::DISABLED,
                                     SelfType, true>;
        DECLARE_READ_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread,
                                            const ParamsType& params) {
            return fused_operation_impl::tuple_operate(thread, params);
        }

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread,
                                             const OperationDataType& opData) {
            return ParamsType::Operation::num_elems_x(thread, opData.params.instance);
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread,
                                             const OperationDataType& opData) {
            return ParamsType::Operation::num_elems_y(thread, opData.params.instance);
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread,
                                             const OperationDataType& opData) {
            return ParamsType::Operation::num_elems_z(thread, opData.params.instance);
        }

        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
            return { num_elems_x(Point(), opData), num_elems_y(Point(), opData), num_elems_z(Point(), opData) };
        }
    };

    template <typename... Operations>
    struct FusedOperation_<std::enable_if_t<isWriteType<FirstType_t<Operations...>>>, Operations...> {
    private:
        using SelfType = FusedOperation_<std::enable_if_t<isWriteType<FirstType_t<Operations...>>>, Operations...>;
    public:
        FK_STATIC_STRUCT(FusedOperation_, SelfType)
        using ParamsType = OperationTuple<Operations...>;
        using OutputType = FOOT<LastType_t<Operations...>>;
        using InputType = typename FirstType_t<Operations...>::InputType;
        using InstanceType = MidWriteType;
        // THREAD_FUSION in this case will not be used in the current Transform implementation
        // May be used in future implementations
        static constexpr bool IS_FUSED_OP{ true };
        static constexpr bool THREAD_FUSION{ false };
        using WriteDataType = typename FirstType_t<Operations...>::WriteDataType;
        using OperationDataType = OperationData<FusedOperation_<void, Operations...>>;
        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const InputType& input,
                                            const ParamsType& params) {
            return fused_operation_impl::tuple_operate(thread, input, params);
        }
        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const InputType& input,
                                            const OperationDataType& opData) {
            return exec(thread, input, opData.params);
        }
        using InstantiableType = MidWrite<FusedOperation_<void, Operations...>>;
        FK_HOST_DEVICE_FUSE auto build(const OperationDataType& opData) {
            return InstantiableType{ opData };
        }
        FK_HOST_DEVICE_FUSE auto build(const ParamsType& params) {
            return InstantiableType{ { params } };
        }
        template <size_t BATCH_N, typename FirstType, typename... ArrayTypes>
        FK_HOST_FUSE auto build_batch(const std::array<FirstType, BATCH_N>& firstInstance,
            const ArrayTypes&... arrays) {
            return BatchUtils::build_batch<SelfType>(firstInstance, arrays...);
        }
        template <size_t BATCH_N, typename FirstType, typename... ArrayTypes>
        FK_HOST_FUSE auto build(const std::array<FirstType, BATCH_N>& firstInstance,
            const ArrayTypes&... arrays) {
            return BatchWrite<BATCH_N, SelfType>::build(firstInstance, arrays...);
        }
    };

    template <typename... Operations>
    struct FusedOperation_<std::enable_if_t<isMidWriteType<FirstType_t<Operations...>>>, Operations...> {
    private:
        using SelfType = FusedOperation_<std::enable_if_t<isMidWriteType<FirstType_t<Operations...>>>, Operations...>;
    public:
        FK_STATIC_STRUCT(FusedOperation_, SelfType)
        using ParamsType = OperationTuple<Operations...>;
        using OutputType = FOOT<LastType_t<Operations...>>;
        using InputType = typename FirstType_t<Operations...>::InputType;
        using InstanceType = MidWriteType;
        // THREAD_FUSION in this case will not be used in the current Transform implementation
        // May be used in future implementations
        static constexpr bool IS_FUSED_OP{ true };
        static constexpr bool THREAD_FUSION{ false };
        using WriteDataType = typename FirstType_t<Operations...>::WriteDataType;
        using OperationDataType = OperationData<FusedOperation_<void, Operations...>>;
        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const InputType& input,
                                            const ParamsType& params) {
            return fused_operation_impl::tuple_operate(thread, input, params);
        }
        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const InputType& input,
                                            const OperationDataType& opData) {
            return exec(thread, input, opData.params);
        }
        using InstantiableType = MidWrite<FusedOperation_<void, Operations...>>;
        FK_HOST_DEVICE_FUSE auto build(const OperationDataType& opData) {
            return InstantiableType{ opData };
        }
        FK_HOST_DEVICE_FUSE auto build(const ParamsType& params) {
            return InstantiableType{ { params } };
        }
        template <size_t BATCH_N, typename FirstType, typename... ArrayTypes>
        FK_HOST_FUSE auto build_batch(const std::array<FirstType, BATCH_N>& firstInstance,
                                      const ArrayTypes&... arrays) {
            return BatchUtils::build_batch<SelfType>(firstInstance, arrays...);
        }
        template <size_t BATCH_N, typename FirstType, typename... ArrayTypes>
        FK_HOST_FUSE auto build(const std::array<FirstType, BATCH_N>& firstInstance,
                                const ArrayTypes&... arrays) {
            return BatchWrite<BATCH_N, SelfType>::build(firstInstance, arrays...);
        }
    };

    template <typename... Operations>
    using FusedOperation = FusedOperation_<void, Operations...>;

    template <typename FusedOperationType, typename = void>
    struct IsAllUnaryFusedOperation : std::false_type {};

    template <typename FusedOperationType>
    struct IsAllUnaryFusedOperation<FusedOperationType, std::void_t<typename FusedOperationType::Operations>> : std::true_type {};

    template <typename FusedOperationType, typename = void>
    struct IsNotAllUnaryFusedOperation : std::true_type {};

    template <typename FusedOperationType>
    struct IsNotAllUnaryFusedOperation<FusedOperationType, std::void_t<typename FusedOperationType::Operations>> : std::false_type {};

    template <typename FusedOperationType>
    constexpr bool isAllUnaryFusedOperation = IsAllUnaryFusedOperation<FusedOperationType>::value;

    template <typename FusedOperationType>
    constexpr bool isNotAllUnaryFusedOperation = IsNotAllUnaryFusedOperation<FusedOperationType>::value;

    template <typename IOp, typename Enabler = void>
    struct InstantiableFusedOperationToOperationTuple {};

    template <typename FusedIOp>
    struct InstantiableFusedOperationToOperationTuple<FusedIOp, std::enable_if_t<isAllUnaryFusedOperation<typename FusedIOp::Operation>, void>> {
    private:
        using SelfType = InstantiableFusedOperationToOperationTuple<FusedIOp, std::enable_if_t<isAllUnaryFusedOperation<typename FusedIOp::Operation>, void>>;
    public:
        FK_STATIC_STRUCT(InstantiableFusedOperationToOperationTuple, SelfType)
        FK_HOST_FUSE auto value(const FusedIOp& iOp) {
            return TypeListToOT<typename FusedIOp::Operation::Operations>{};
        }
    };
    template <typename FusedIOp>
    struct InstantiableFusedOperationToOperationTuple<FusedIOp, std::enable_if_t<isNotAllUnaryFusedOperation<typename FusedIOp::Operation>, void>> {
    private:
        using SelfType = InstantiableFusedOperationToOperationTuple<FusedIOp, std::enable_if_t<isNotAllUnaryFusedOperation<typename FusedIOp::Operation>, void>>;
    public:
        FK_STATIC_STRUCT(InstantiableFusedOperationToOperationTuple, SelfType)
        FK_HOST_FUSE auto value(const FusedIOp& iOp) {
            return iOp.params;
        }
    };

    template <typename OperationTupleType, typename Enabler = void>
    struct OperationTupleToInstantiableOperation;

    template <typename... Operations>
    struct OperationTupleToInstantiableOperation<OperationTuple<Operations...>, std::enable_if_t<allUnaryTypes<Operations...>, void>> {
    private:
        using SelfType = OperationTupleToInstantiableOperation<OperationTuple<Operations...>, std::enable_if_t<allUnaryTypes<Operations...>, void>>;
    public:
        FK_STATIC_STRUCT(OperationTupleToInstantiableOperation, SelfType)
        FK_HOST_FUSE auto value(const OperationTuple<Operations...>& opTuple) {
            return Instantiable<FusedOperation<Operations...>>{};
        }
    };

    template <typename... Operations>
    struct OperationTupleToInstantiableOperation<OperationTuple<Operations...>, std::enable_if_t<notAllUnaryTypes<Operations...>, void>> {
    private:
        using SelfType = OperationTupleToInstantiableOperation<OperationTuple<Operations...>, std::enable_if_t<notAllUnaryTypes<Operations...>, void>>;
    public:
        FK_STATIC_STRUCT(OperationTupleToInstantiableOperation, SelfType)
        FK_HOST_FUSE auto value(const OperationTuple<Operations...>& opTuple) {
            return Instantiable<FusedOperation<Operations...>>{opTuple};
        }
    };

    template <typename IOp>
    FK_HOST_CNST auto iOpsToOperationTuple(const IOp& iOp) {
        using Op = typename IOp::Operation;
        if constexpr (is_fused_operation<Op>::value) {
            return InstantiableFusedOperationToOperationTuple<IOp>::value(iOp);
        } else if constexpr (hasParamsAndBackIOp_v<Op>) {
            return OperationTuple<Op>{ {iOp.params, iOp.backIOp} };
        } else if constexpr (hasParams_v<Op>) {
            return OperationTuple<Op>{ {iOp.params} };
        } else { // UnaryType case
            return OperationTuple<Op>{};
        }
    }

    template <typename IOp, typename... InstantiableOperations>
    FK_HOST_CNST auto iOpsToOperationTuple(const IOp& iOp, const InstantiableOperations&... iOps) {
        return cat(iOpsToOperationTuple(iOp), iOpsToOperationTuple(iOps...));
    }

    template <typename OperationTuple>
    FK_HOST_CNST auto operationTupleToIOp(const OperationTuple& opTuple) {
        return OperationTupleToInstantiableOperation<OperationTuple>::value(opTuple);
    }*/

    template <typename Enabler, typename... Operations>
    struct NewFusedOperation_;

    template <typename FirstOp, typename... RemOps>
    struct NewFusedOperation_<std::enable_if_t<!isAnyReadType<FirstOp> && !isWriteType<LastType_t<FirstOp, RemOps...>>>,
                              FirstOp, RemOps...> {
    private:
        using SelfType = NewFusedOperation_<std::enable_if_t<!isAnyReadType<FirstOp>>, FirstOp, RemOps...>;
    public:
        FK_STATIC_STRUCT(NewFusedOperation_, SelfType)
        using Parent =
            OpenOperationParent<typename FirstOp::Operation::InputType, NewOperationTuple<FirstOp, RemOps...>,
            typename LastType_t<RemOps...>::Operation::OutputType, SelfType, true>;
        DECLARE_FUSED_PARENT

        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const InputType& input, const ParamsType& params) {
            return exec_helper(std::make_index_sequence<ParamsType::size>{}, thread, input, params);
        }
    private:
        template <size_t... Idx>
        FK_HOST_DEVICE_FUSE OutputType exec_helper(const std::index_sequence<Idx...>&,
                                                   const Point& thread,
                                                   const InputType& input,
                                                   const ParamsType& params) {
            return (InputFoldType<InputType>{thread, input} | ... | get_opt<Idx>(params));
        }
    };

    template <typename FirstOp, typename... RemOps>
    struct NewFusedOperation_<
        std::enable_if_t<isAnyReadType<FirstOp> &&
                         !isWriteType<LastType_t<FirstOp, RemOps...>>>,
        FirstOp, RemOps...> {
    private:
      using SelfType = NewFusedOperation_<std::enable_if_t<isAnyReadType<FirstOp>>, FirstOp, RemOps...>;
      // 1. Resolve the type of the last operation in the sequence.
      //    We separate this to isolate the 'RemOps...' expansion from the Parent definition.
      using LastOpResolved = typename LastType_t<std::decay_t<FirstOp>, std::decay_t<RemOps>...>;

      // 2. Extract the output type from that resolved last operation.
      using FinalOutputType = typename LastOpResolved::Operation::OutputType;

      // 3. Define the tuple of operations separately.
      using OpTupleType = NewOperationTuple<FirstOp, RemOps...>;

      // 4. Extract the read data type from the first operation.
      using FirstReadDataType = typename std::decay_t<FirstOp>::Operation::ReadDataType;
      // 5. Finally, assemble Parent using these pre-calculated, clean types.
      //    Note: No '...' expansions happen inside this specific statement anymore.
      using Parent = ReadOperation<FirstReadDataType, OpTupleType, FinalOutputType, TF::DISABLED, SelfType, true>;
    public:
        FK_STATIC_STRUCT(NewFusedOperation_, SelfType)
        DECLARE_READ_PARENT

        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const ParamsType& params) {
            return exec_helper(std::make_index_sequence<ParamsType::size>{}, thread, params);
        }

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread,
                                             const OperationDataType& opData) {
            return FirstOp::Operation::num_elems_x(thread, get_opt<0>(opData.params));
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread,
                                             const OperationDataType& opData) {
            return FirstOp::Operation::num_elems_y(thread, get_opt<0>(opData.params));
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread,
                                             const OperationDataType& opData) {
            return FirstOp::Operation::num_elems_z(thread, get_opt<0>(opData.params));
        }

        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
            return { num_elems_x(Point(), opData), num_elems_y(Point(), opData), num_elems_z(Point(), opData) };
        }

    private:
        template <size_t... Idx>
        FK_HOST_DEVICE_FUSE OutputType exec_helper(const std::index_sequence<Idx...>&,
                                                   const Point& thread,
                                                   const ParamsType& params) {
            return (thread | ... | get_opt<Idx>(params));
        }
    };

    template <>
    struct NewFusedOperation_<void> {
      private:
        template <typename T>
        struct FuseProxy {
            T value;

            template <typename IOp1, typename IOp2>
            FK_HOST_FUSE decltype(auto) fuse(IOp1 &&iOp1, IOp2 &&iOp2) {
                constexpr bool iOp1Fused = std::decay_t<IOp1>::Operation::IS_FUSED_OP;
                constexpr bool iOp2Fused = std::decay_t<IOp2>::Operation::IS_FUSED_OP;
                if constexpr (iOp1Fused && iOp2Fused) {
                    return NewFusedOperation_<void>::build(
                        cat(std::forward<IOp1>(iOp1).params, std::forward<IOp2>(iOp2).params));
                } else if constexpr (iOp1Fused) {
                    return NewFusedOperation_<void>::build(
                        cat(std::forward<IOp1>(iOp1).params, make_new_operation_tuple(std::forward<IOp2>(iOp2))));
                } else if constexpr (iOp2Fused) {
                    return NewFusedOperation_<void>::build(
                        cat(make_new_operation_tuple(std::forward<IOp1>(iOp1)), std::forward<IOp2>(iOp2).params));
                } else {
                    return NewFusedOperation_<void>::build(make_new_operation_tuple(std::forward<IOp1>(iOp1), std::forward<IOp2>(iOp2)));
                }
            }

            // The friend operator is found via ADL even if the struct is private
            template <typename LeftIOp>
            FK_HOST_CNST friend auto operator&&(const FuseProxy<LeftIOp>& left, const FuseProxy<T>& right) {
                // Call your global 'fuse' function
                auto fused = FuseProxy<T>::fuse(left.value, right.value);

                return FuseProxy<decltype(fused)>{std::move(fused)};
            }
        };

        template <typename... IOps>
        FK_HOST_FUSE decltype(auto) build_helper(const NewOperationTuple_<void, IOps...>& opTup) {
            return NewFusedOperation_<void, IOps...>::build(opTup);
        }


      public:
        template <typename... IOps>
        FK_HOST_FUSE decltype(auto) build(IOps&&... iOps) {
            if constexpr (and_v<isOperation<std::decay_t<IOps>>...>) {
                return (... && FuseProxy<std::decay_t<IOps>>{std::forward<IOps>(iOps)}).value;
            } else { // Assuming it's a NewOperationTuple
                static_assert(sizeof...(IOps) == 1,
                              "If the argument is not an operation, it should be a single OperationTuple");
                return build_helper(iOps...);
            }
        }
    };

    template <typename... IOps>
    using FusedOperation = NewFusedOperation_<void, IOps...>;
    // END FusedOperation
} // namespace fk

#endif // FK_FUSED_OPERATION