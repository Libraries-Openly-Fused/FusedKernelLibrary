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

#ifndef FK_FUSED_OPERATION
#define FK_FUSED_OPERATION

#include <fused_kernel/core/execution_model/operation_model/batch_operations.h>
#include <fused_kernel/core/execution_model/operation_model/operation_tuple.h>

namespace fk {
 /*
 * FusedOperation can be of the follwing types:
 * - ReadType: if the first Op is ReadType and the last one is not WriteType.
 * - WriteType: if the last Op is WriteType and the first one is not ReadType nor ReadBackType.
 * - UnaryType: if all Ops are UnaryType.
 * - BinaryType: if there is no ReadType, ReadBackType, WriteType nor MidWriteType, and not all Ops are UnaryType.
 * - OpenType: if there is at least one MidWriteType, and no ReadType, ReadBackType nor WriteType.
 * - ClosedType: if the first Op is ReadType or ReadBackType and the last Op is WriteType.
 */

    // FusedOperation
    template <typename Enabler, typename... Operations>
    struct FusedOperation_;

    // 1. OpenType Specialization
    // Changed to use IOps... for consistency
    template <typename... IOps>
    struct FusedOperation_<std::enable_if_t<!isAnyReadType<FirstType_t<IOps...>> && 
                                            !opIs<WriteType, LastType_t<IOps...>> && 
                                            atLeastOneMidWriteType<IOps...>>,
                           IOps...> {
    private:
        using SelfType = FusedOperation_<std::enable_if_t<!isAnyReadType<FirstType_t<IOps...>> && 
                                                          !opIs<WriteType, LastType_t<IOps...>> && 
                                                          atLeastOneMidWriteType<IOps...>>,
                                         IOps...>;
        
        using Parent = OpenOperationParent<typename FirstType_t<IOps...>::Operation::InputType, 
                                           OperationTuple<IOps...>,
                                           typename LastType_t<IOps...>::Operation::OutputType, 
                                           SelfType, true>;
    public:
        FK_STATIC_STRUCT(FusedOperation_, SelfType)
        DECLARE_OPEN_PARENT
        using Operations = TypeList<IOps...>;
        
        FK_HOST_DEVICE_FUSE OutputType exec(const Point thread, const InputType input, const ParamsType& params) {
            return exec_helper(std::make_index_sequence<ParamsType::size>{}, thread, input, params);
        }
    private:
        template <size_t... Idx>
        FK_HOST_DEVICE_FUSE OutputType exec_helper(const std::index_sequence<Idx...>&,
                                                   const Point thread,
                                                   const InputType input,
                                                   const ParamsType& params) {
            return (InputFoldType<InputType>{thread, input} | ... | get_opt<Idx>(params)).input;
        }
    };

    // 2. ReadType Specialization
    // Changed to use IOps... for consistency
    template <typename... IOps>
    struct FusedOperation_<
        std::enable_if_t<isAnyReadType<FirstType_t<IOps...>> && !(opIs<WriteType, LastType_t<IOps...>>)>,
        IOps...> {
    private:
      using SelfType = FusedOperation_<
        std::enable_if_t<isAnyReadType<FirstType_t<IOps...>> && !(opIs<WriteType, LastType_t<IOps...>>)>,
        IOps...>;
      
      using FusedReadDataType = typename std::decay_t<FirstType_t<IOps...>>::Operation::ReadDataType;
      using FusedOutputType = typename LastType_t<IOps...>::Operation::OutputType;

      using Parent = ReadOperation<FusedReadDataType, OperationTuple<IOps...>, FusedOutputType, TF::DISABLED, SelfType, true>;
    public:
        FK_STATIC_STRUCT(FusedOperation_, SelfType)
        DECLARE_READ_PARENT
        using Operations = TypeList<IOps...>;
        
        FK_HOST_DEVICE_FUSE OutputType exec(const Point thread, const ParamsType& params) {
            return exec_helper(std::make_index_sequence<ParamsType::size>{}, thread, params);
        }

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point thread,
                                             const OperationDataType& opData) {
            return FirstType_t<IOps...>::Operation::num_elems_x(thread, get_opt<0>(opData.params));
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point thread,
                                             const OperationDataType& opData) {
            return FirstType_t<IOps...>::Operation::num_elems_y(thread, get_opt<0>(opData.params));
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point thread,
                                             const OperationDataType& opData) {
            return FirstType_t<IOps...>::Operation::num_elems_z(thread, get_opt<0>(opData.params));
        }

        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
            return { num_elems_x(Point{0,0,0}, opData), num_elems_y(Point{0,0,0}, opData), num_elems_z(Point{0,0,0}, opData) };
        }

    private:
        template <size_t... Idx>
        FK_HOST_DEVICE_FUSE OutputType exec_helper(const std::index_sequence<Idx...>&,
                                                   const Point thread,
                                                   const ParamsType& params) {
            return (thread | ... | get_opt<Idx>(params)).input;
        }
    };

    // 3. UnaryType Specialization
    template <typename... IOps>
    struct FusedOperation_<std::enable_if_t<allUnaryTypes<IOps...>>, IOps...> {
      private:
        using SelfType = FusedOperation_<std::enable_if_t<allUnaryTypes<IOps...>>, IOps...>;
        using Parent = UnaryOperation<typename FirstType_t<IOps...>::Operation::InputType,
                                      typename LastType_t<IOps...>::Operation::OutputType, SelfType, true>;
      public:
        FK_STATIC_STRUCT(FusedOperation_, SelfType)
        DECLARE_UNARY_PARENT
        using Operations = TypeList<IOps...>;
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType input) {
            return exec_helper(std::make_index_sequence<OperationTuple<IOps...>::size>{}, input);
        }

      private:
        template <size_t... Idx>
        FK_HOST_DEVICE_FUSE OutputType exec_helper(const std::index_sequence<Idx...>&, const InputType input) {
            constexpr OperationTuple<IOps...> poTup{};
            // Optimization, we use a version of operator| that does not use InputTypeFold,
            // thus it does not propagate Point thread, because it is not needed.
            return (input | ... | get_opt<Idx>(poTup));
        }
    };

    // 4. ClosedType Specialization
    template <typename... IOps>
    struct FusedOperation_<std::enable_if_t<isAnyReadType<FirstType_t<IOps...>> && opIs<WriteType, LastType_t<IOps...>>>,
                           IOps...> {
      private:
        using SelfType =
            FusedOperation_<std::enable_if_t<isAnyReadType<FirstType_t<IOps...>> && opIs<WriteType, LastType_t<IOps...>>>, IOps...>;
        using Parent = ClosedOperation<OperationTuple<IOps...>, SelfType, true>;

      public:
        FK_STATIC_STRUCT(FusedOperation_, SelfType)
        DECLARE_CLOSED_PARENT
        using Operations = TypeList<IOps...>;
        FK_HOST_DEVICE_FUSE void exec(const Point thread, const ParamsType &params) {
            exec_helper(std::make_index_sequence<ParamsType::size>{}, thread, params);
        }

      private:
        template <size_t... Idx>
        FK_HOST_DEVICE_FUSE void exec_helper(const std::index_sequence<Idx...>&, const Point thread,
                                             const ParamsType &params) {
            LastType_t<typename ParamsType::Operations>::Operation::exec(thread,
                (thread | ... | get_opt<Idx>(params)).input,
                get_opt<sizeof...(Idx) - 1>(params));
        }
    };

    // 5. WriteType Specialization
    template <typename... IOps>
    struct FusedOperation_<std::enable_if_t<!isAnyReadType<FirstType_t<IOps...>> && opIs<WriteType, LastType_t<IOps...>>>,
                           IOps...> {
      private:
        using SelfType =
            FusedOperation_<std::enable_if_t<!isAnyReadType<FirstType_t<IOps...>> && opIs<WriteType, LastType_t<IOps...>>>,
                            IOps...>;
        using FusedInputType = typename FirstType_t<IOps...>::Operation::InputType;
        using FusedWriteDataType = typename LastType_t<IOps...>::Operation::WriteDataType;
        using Parent = WriteOperation<FusedInputType,
            OperationTuple<IOps...>, FusedWriteDataType, TF::DISABLED, SelfType, true>;

      public:
        FK_STATIC_STRUCT(FusedOperation_, SelfType)
        DECLARE_WRITE_PARENT
        using Operations = TypeList<IOps...>;
        FK_HOST_DEVICE_FUSE void exec(const Point thread, const InputType input, const ParamsType &params) {
            exec_helper(std::make_index_sequence<ParamsType::size - 1>{}, thread, input, params);
        }

      private:
        template <size_t... Idx>
        FK_HOST_DEVICE_FUSE void exec_helper(const std::index_sequence<Idx...> &, const Point thread,
                                             const InputType input, const ParamsType &params) {
            LastType_t<typename ParamsType::Operations>::Operation::exec(
                thread, (InputFoldType<>::build(thread, input) | ... | get_opt<Idx>(params)).input,
                get_opt<ParamsType::size - 1>(params));
        }
    };

    // 6. BinaryType (ComputeType) Specialization
    template <typename... IOps>
    struct FusedOperation_<std::enable_if_t<allComputeTypes<IOps...> && !allUnaryTypes<IOps...>>, IOps...> {
      private:
        using SelfType =
            FusedOperation_<std::enable_if_t<allComputeTypes<IOps...> && !allUnaryTypes<IOps...>>, IOps...>;
        using FusedInputType = typename FirstType_t<IOps...>::Operation::InputType;
        using FusedOutputType = typename LastType_t<IOps...>::Operation::OutputType;
        using Parent = BinaryOperation<FusedInputType, OperationTuple<IOps...>, FusedOutputType, SelfType, true>;

      public:
        FK_STATIC_STRUCT(FusedOperation_, SelfType)
        DECLARE_BINARY_PARENT
        using Operations = TypeList<IOps...>;
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType input, const ParamsType &params) {
            return exec_helper(std::make_index_sequence<ParamsType::size>{}, input, params);
        }

      private:
        template <size_t... Idx>
        FK_HOST_DEVICE_FUSE OutputType exec_helper(const std::index_sequence<Idx...>&,
                                                   const InputType input, const ParamsType& params) {
            // Optimization, we use a version of operator| that does not use InputTypeFold,
            // thus it does not propagate Point thread, because it is not needed.
            return (input | ... | get_opt<Idx>(params));
        }
    };

    // 7. Builder / Proxy Specialization
    template <>
    struct FusedOperation_<void> {
      private:
        template <typename T>
        struct FuseProxy {
            T iOp;

            template <typename U>
            FK_HOST_CNST FuseProxy(U &&u) : iOp(std::forward<U>(u)) {}

            template <typename IOp1, typename IOp2>
            FK_HOST_FUSE auto fuse(IOp1 &&iOp1, IOp2 &&iOp2) {
                constexpr bool iOp1Fused = std::decay_t<IOp1>::Operation::IS_FUSED_OP;
                constexpr bool iOp2Fused = std::decay_t<IOp2>::Operation::IS_FUSED_OP;
                
                if constexpr (iOp1Fused && iOp2Fused) {
                    return FusedOperation_<void>::build(
                        cat(get_params(std::forward<IOp1>(iOp1)), get_params(std::forward<IOp2>(iOp2))));
                } else if constexpr (iOp1Fused) {
                    return FusedOperation_<void>::build(
                        cat(get_params(std::forward<IOp1>(iOp1)), make_new_operation_tuple(std::forward<IOp2>(iOp2))));
                } else if constexpr (iOp2Fused) {
                    return FusedOperation_<void>::build(
                        cat(make_new_operation_tuple(std::forward<IOp1>(iOp1)), get_params(std::forward<IOp2>(iOp2))));
                } else {
                    return FusedOperation_<void>::build(make_new_operation_tuple(std::forward<IOp1>(iOp1), std::forward<IOp2>(iOp2)));
                }
            }

            template <typename LeftIOp>
            FK_HOST_CNST friend auto operator&&(const FuseProxy<LeftIOp>& left, const FuseProxy<T>& right) {
                return FuseProxy<decltype(FuseProxy<T>::fuse(std::declval<LeftIOp>(), std::declval<T>()))>{
                    FuseProxy<T>::fuse(left.iOp, right.iOp)
                };
            }

            private:
            template <typename... IOps>
            FK_HOST_FUSE auto get_unary_params(const Unary<FusedOperation_<void, IOps...>>&)
                -> OperationTuple<IOps...> {
                return OperationTuple<IOps...>{};
            }
            template <typename IOp>
            FK_HOST_FUSE decltype(auto) get_params(IOp&& iOp) {
                if constexpr (opIs<UnaryType, std::decay_t<IOp>>) {
                    return get_unary_params(std::forward<IOp>(iOp));
                } else {
                    return std::forward<IOp>(iOp).params;
                }
            }
        };

        template <typename... IOps>
        FK_HOST_FUSE auto build_helper(const OperationTuple_<void, IOps...>& opTup) {
            if constexpr (allUnaryTypes<IOps...>) {
                return Unary<FusedOperation_<void, IOps...>>{};
            } else {
                return FusedOperation_<void, IOps...>::build(opTup);
            }
        }

      public:
        template <typename... IOps>
        FK_HOST_FUSE auto build(IOps&&... iOps) {
            if constexpr (and_v<isOperation<std::decay_t<IOps>>...>) {
                return (... && FuseProxy<std::decay_t<IOps>>{std::forward<IOps>(iOps)}).iOp;
            } else { 
                static_assert(sizeof...(IOps) == 1,
                              "If the argument is not an operation, it should be a single OperationTuple");
                return build_helper(iOps...);
            }
        }
    };

    template <typename... IOps>
    using FusedOperation = FusedOperation_<void, IOps...>;
    // END FusedOperation
} // namespace fk

#endif // FK_FUSED_OPERATION