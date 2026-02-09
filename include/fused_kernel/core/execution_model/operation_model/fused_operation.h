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
    template <typename Enabler, typename... Operations>
    struct FusedOperation_;

    template <typename FirstOp, typename... RemOps>
    struct FusedOperation_<std::enable_if_t<!isAnyReadType<FirstOp> && !isWriteType<LastType_t<FirstOp, RemOps...>> && !allUnaryTypes<FirstOp, RemOps...>>,
                              FirstOp, RemOps...> {
    private:
        using SelfType = FusedOperation_<std::enable_if_t<!isAnyReadType<FirstOp>>, FirstOp, RemOps...>;
        using Parent = OpenOperationParent<typename FirstOp::Operation::InputType, OperationTuple<FirstOp, RemOps...>,
                                         typename LastType_t<RemOps...>::Operation::OutputType, SelfType, true>;
    public:
        FK_STATIC_STRUCT(FusedOperation_, SelfType)
        DECLARE_OPEN_PARENT
        using Operations = TypeList<FirstOp, RemOps...>;
        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const InputType& input, const ParamsType& params) {
            return exec_helper(std::make_index_sequence<ParamsType::size>{}, thread, input, params);
        }
    private:
        template <size_t... Idx>
        FK_HOST_DEVICE_FUSE OutputType exec_helper(const std::index_sequence<Idx...>&,
                                                   const Point& thread,
                                                   const InputType& input,
                                                   const ParamsType& params) {
            return (InputFoldType<InputType>{thread, input} | ... | get_opt<Idx>(params)).input;
        }
    };

    template <typename FirstOp, typename... RemOps>
    struct FusedOperation_<
        std::enable_if_t<isAnyReadType<FirstOp> &&
                         !isWriteType<LastType_t<FirstOp, RemOps...>>>,
        FirstOp, RemOps...> {
    private:
      using SelfType = FusedOperation_<std::enable_if_t<isAnyReadType<FirstOp>>, FirstOp, RemOps...>;
      // 1. Resolve the type of the last operation in the sequence.
      //    We separate this to isolate the 'RemOps...' expansion from the Parent definition.
      using LastOpResolved = typename LastType_t<std::decay_t<FirstOp>, std::decay_t<RemOps>...>;

      // 2. Extract the output type from that resolved last operation.
      using FinalOutputType = typename LastOpResolved::Operation::OutputType;

      // 3. Define the tuple of operations separately.
      using OpTupleType = OperationTuple<FirstOp, RemOps...>;

      // 4. Extract the read data type from the first operation.
      using FirstReadDataType = typename std::decay_t<FirstOp>::Operation::ReadDataType;
      // 5. Finally, assemble Parent using these pre-calculated, clean types.
      //    Note: No '...' expansions happen inside this specific statement anymore.
      using Parent = ReadOperation<FirstReadDataType, OpTupleType, FinalOutputType, TF::DISABLED, SelfType, true>;
    public:
        FK_STATIC_STRUCT(FusedOperation_, SelfType)
        DECLARE_READ_PARENT
        using Operations = TypeList<FirstOp, RemOps...>;
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
            return (thread | ... | get_opt<Idx>(params)).input;
        }
    };

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
        FK_HOST_DEVICE_FUSE OutputType exec(InputType &&input) {
            return exec_helper(std::make_index_sequence<OperationTuple<IOps...>::size>{}, std::forward<InputType>(input));
        }

      private:
        template <size_t... Idx>
        FK_HOST_DEVICE_FUSE OutputType exec_helper(const std::index_sequence<Idx...> &, InputType &&input) {
            constexpr OperationTuple<IOps...> poTup{};
            return (input | ... | get_opt<Idx>(poTup));
        }
    };

    template <typename... IOps>
    struct FusedOperation_<std::enable_if_t<isReadType<FirstType_t<IOps...>> && isWriteType<LastType_t<IOps...>>>,
                          IOps...> {
      private:
        using SelfType =
            FusedOperation_<std::enable_if_t<isReadType<FirstType_t<IOps...>> && isWriteType<LastType_t<IOps...>>>, IOps...>;
        using Parent = ClosedOperation<OperationTuple<IOps...>, SelfType, true>;

      public:
        FK_STATIC_STRUCT(FusedOperation_, SelfType)
        DECLARE_CLOSED_PARENT
        using Operations = TypeList<IOps...>;
        FK_HOST_DEVICE_FUSE void exec(const Point &thread, const ParamsType &params) {
            exec_helper(std::make_index_sequence<ParamsType::size>{}, thread, params);
        }

      private:
        template <size_t... Idx>
        FK_HOST_DEVICE_FUSE void exec_helper(const std::index_sequence<Idx...> &, const Point &thread,
                                              const ParamsType &params) {
            (thread | ... | get_opt<Idx>(params));
        }
    };

    template <>
    struct FusedOperation_<void> {
      private:
        template <typename T>
        struct FuseProxy {
            T value;

            template <typename U>
            FK_HOST_CNST FuseProxy(U &&u) : value(std::forward<U>(u)) {}

            template <typename IOp1, typename IOp2>
            FK_HOST_FUSE decltype(auto) fuse(IOp1 &&iOp1, IOp2 &&iOp2) {
                constexpr bool iOp1Fused = std::decay_t<IOp1>::Operation::IS_FUSED_OP;
                constexpr bool iOp2Fused = std::decay_t<IOp2>::Operation::IS_FUSED_OP;
                // Missing taking into account the unary case of FusedOperation
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

            // The friend operator is found via ADL even if the struct is private
            template <typename LeftIOp>
            FK_HOST_CNST friend decltype(auto) operator&&(const FuseProxy<LeftIOp>& left, const FuseProxy<T>& right) {
                // Call your global 'fuse' function
                using ReturnType = decltype(FuseProxy<T>::fuse(std::declval<decltype(left.value)>(), std::declval<decltype(right.value)>()));
                return FuseProxy<ReturnType>{FuseProxy<T>::fuse(left.value, right.value)};
            }

            private:
            template <typename... IOps>
            FK_HOST_FUSE decltype(auto) get_unary_params(const Unary<FusedOperation_<void, IOps...>>&) {
                return OperationTuple<IOps...>{};
            }
            template <typename IOp>
            FK_HOST_FUSE decltype(auto) get_params(IOp&& iOp) {
                if constexpr (isUnaryType<std::decay_t<IOp>>) {
                    return get_unary_params(std::forward<IOp>(iOp));
                } else {
                    return std::forward<IOp>(iOp).params;
                }
            }
        };

        template <typename... IOps>
        FK_HOST_FUSE decltype(auto) build_helper(const OperationTuple_<void, IOps...>& opTup) {
            if constexpr (allUnaryTypes<IOps...>) {
                return Unary<FusedOperation_<void, IOps...>>{};
            } else {
                return FusedOperation_<void, IOps...>::build(opTup);
            }
        }


      public:
        template <typename... IOps>
        FK_HOST_FUSE decltype(auto) build(IOps&&... iOps) {
            if constexpr (and_v<isOperation<std::decay_t<IOps>>...>) {
                return (... && FuseProxy<std::decay_t<IOps>>{std::forward<IOps>(iOps)}).value;
            } else { // Assuming it's a OperationTuple
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