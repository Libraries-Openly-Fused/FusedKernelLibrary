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

#ifndef FK_OPERATION_TUPLE
#define FK_OPERATION_TUPLE

#include <fused_kernel/core/data/tuple.h>
#include <fused_kernel/core/execution_model/operation_model/operation_data.h>

namespace fk {
   /*template <typename Enabler, typename... Operations>
    struct OperationTuple_;

    template <typename Operation_t, typename... Operations>
    struct OperationTuple_<std::enable_if_t<!isUnaryType<Operation_t> &&
                                            (sizeof...(Operations) > 0), void>, Operation_t, Operations...> {
        using Operation = Operation_t;
        using Next = OperationTuple_<void, Operations...>;
        OperationData<Operation> instance;
        OperationTuple_<void, Operations...> next;
        enum { size = sizeof...(Operations) + 1 };
    };

    template <typename Operation_t, typename... Operations>
    struct OperationTuple_<std::enable_if_t<isUnaryType<Operation_t> &&
                                            !allUnaryTypes<Operations...> &&
                                            (sizeof...(Operations) > 0), void>, Operation_t, Operations...> {
        using Operation = Operation_t;
        using Next = OperationTuple_<void, Operations...>;
        OperationTuple_<void, Operations...> next;
        enum { size = sizeof...(Operations) + 1 };
    };

    template <typename Operation_t, typename... Operations>
    struct OperationTuple_<std::enable_if_t<allUnaryTypes<Operation_t, Operations...> &&
                                            (sizeof...(Operations) > 0), void>, Operation_t, Operations...> {
        using Operation = Operation_t;
        using Next = OperationTuple_<void, Operations...>;
        enum { size = sizeof...(Operations) + 1 };
    };

    template <typename Operation_t>
    struct OperationTuple_<std::enable_if_t<!isUnaryType<Operation_t>, void>, Operation_t> {
        using Operation = Operation_t;
        OperationData<Operation> instance;
        enum { size = 1 };
    };

    template <typename Operation_t>
    struct OperationTuple_<std::enable_if_t<isUnaryType<Operation_t>, void>, Operation_t> {
        using Operation = Operation_t;
        enum { size = 1 };
    };

    template <typename... Operations>
    using OperationTuple = OperationTuple_<void, Operations...>;

    template <typename... Operations>
    FK_HOST_DEVICE_CNST bool allOpTupleUnary_f(const OperationTuple<Operations...>& opTup) {
        return allUnaryTypes<Operations...>;
    }

    template <typename OpTuple>
    constexpr bool allOpTupleUnary = allOpTupleUnary_f(OpTuple{});

    template <int INDEX, typename... Instances>
    struct GetType<INDEX, OperationTuple_<void, Instances...>> {
        using type = TypeAt_t<INDEX, TypeList<Instances...>>;
    };

    template <typename... Operations, typename... OperationDatas>
    FK_HOST_DEVICE_CNST OperationTuple<Operations...> make_operation_tuple_(const OperationDatas&... instances) {
        return OperationTuple<Operations...>{instances...};
    }

    template <int INDEX, typename... InstanceTypes>
    FK_HOST_DEVICE_CNST auto& get(OperationTuple<InstanceTypes...>& instances) {
        using Operation = typename OperationTuple<InstanceTypes...>::Operation;
        constexpr int numberOfInstances = OperationTuple<InstanceTypes...>::size;
        static_assert(INDEX < numberOfInstances, "Index out of range. There are not so many instances in the tuple.");
        if constexpr (INDEX > 0) {
            return get<INDEX - 1>(instances.next);
        } else if constexpr (INDEX == -1) {
            if constexpr (numberOfInstances > 0) {
                return get<numberOfInstances - 1>(instances.next);
            } else {
                static_assert(hasParams_v<Operation>, "This is an Unary operation, and it does not have params.");
                return instances.instance;
            }
        } else {
            static_assert(hasParams_v<Operation>, "This is an Unary operation, and it does not have params.");
            return instances.instance;
        }
    }

    template <int INDEX, typename... InstanceTypes>
    FK_HOST_DEVICE_CNST auto get(const OperationTuple<InstanceTypes...>& instances) {
        using Operation = typename OperationTuple<InstanceTypes...>::Operation;
        constexpr int numberOfInstances = OperationTuple<InstanceTypes...>::size;
        static_assert(INDEX < numberOfInstances, "Index out of range. There are not so many instances in the tuple.");
        if constexpr (INDEX > 0) {
            return get<INDEX - 1>(instances.next);
        } else if constexpr (INDEX == -1) {
            if constexpr (numberOfInstances > 0) {
                return get<numberOfInstances - 1>(instances.next);
            } else {
                static_assert(hasParams_v<Operation>, "This is an Unary operation, and it does not have params.");
                return instances.instance;
            }
        } else {
            static_assert(hasParams_v<Operation>, "This is an Unary operation, and it does not have params.");
            return instances.instance;
        }
    }

    template <typename TupleType>
    struct OperationTupleTypeToTypeList;

    template <typename... Types>
    struct OperationTupleTypeToTypeList<OperationTuple<Types...>> {
        using type = TypeList<Types...>;
    };

    template <typename TupleType>
    using OTToTypeList = typename OperationTupleTypeToTypeList<TupleType>::type;

    template <typename TypeList_t>
    struct TypeListToOperationTuple;

    template <typename... Operations>
    struct TypeListToOperationTuple<TypeList<Operations...>> {
        using type = OperationTuple<Operations...>;
    };

    template <typename TypeList_t>
    using TypeListToOT = typename TypeListToOperationTuple<TypeList_t>::type;

    template <int INDEX, typename TupleType>
    using get_ot = TypeAt_t<INDEX, OTToTypeList<TupleType>>;

    template <int INDEX, typename... OperationTypes>
    FK_HOST_DEVICE_CNST auto getIOp(const OperationTuple<OperationTypes...>& instances) {
        using SelectedOperation = get_ot<INDEX, OperationTuple<OperationTypes...>>;
        if constexpr (isUnaryType<SelectedOperation>) {
            return SelectedOperation::build();
        } else {
            return SelectedOperation::build(get<INDEX>(instances));
        }
    }

    struct NotUnaryRestriction {
        template <typename Type>
        FK_HOST_DEVICE_FUSE bool complies() {
            return !std::is_same_v<Type, UnaryType>;
        }
    };

    template <typename... Operations1, typename... Operations2, int... I1, int... I2>
    FK_HOST_DEVICE_CNST auto cat_impl(const OperationTuple<Operations1...>& t1, const std::integer_sequence<int, I1...>& is1,
                                      const OperationTuple<Operations2...>& t2, const std::integer_sequence<int, I2...>& is2) {
        return make_operation_tuple_<Operations1..., Operations2...>(get<I1>(t1)..., get<I2>(t2)...);
    }

    template <typename... Operations1, typename... Operations2>
    FK_HOST_DEVICE_CNST auto cat(OperationTuple<Operations1...>& t1, OperationTuple<Operations2...>& t2) {
        return cat_impl(t1, filtered_integer_sequence_t<int, NotUnaryRestriction, TypeList<typename Operations1::InstanceType...>>{},
                        t2, filtered_integer_sequence_t<int, NotUnaryRestriction, TypeList<typename Operations2::InstanceType...>>{});
    }

    template <typename... Operations1, typename... Operations2>
    FK_HOST_DEVICE_CNST auto cat(const OperationTuple<Operations1...>& t1, const OperationTuple<Operations2...>& t2) {
        return cat_impl(t1, filtered_integer_sequence_t<int, NotUnaryRestriction, TypeList<typename Operations1::InstanceType...>>{},
                        t2, filtered_integer_sequence_t<int, NotUnaryRestriction, TypeList<typename Operations2::InstanceType...>>{});
    }

    template <typename... IOps>
    FK_HOST_DEVICE_CNST auto make_operation_tuple(const IOps&... iOps) {
        const auto fusedOp = fuseIOps(iOps...);
        return fusedOp.params;
    }


    template <typename Type, typename = void>
    struct HasOperation : std::false_type {};

    template <typename Type>
    struct HasOperation<Type, std::void_t<typename Type::Operation>> : std::true_type {};

    struct IsInstantiableOperation {
        template <typename Type>
        FK_HOST_DEVICE_FUSE bool complies() {
            return HasOperation<Type>::value;
        }
    };*/

    struct NotIsUnaryRestriction {
        template <typename Type>
        FK_HOST_DEVICE_FUSE bool complies() {
            return !isUnaryType<Type>;
        }
    };

    template <typename IndexSequence, typename... Operations>
    struct FilteredOps;

    template <size_t... Idx, typename... Operations>
    struct FilteredOps<std::index_sequence<Idx...>, Operations...> {
        using type = Tuple<TypeAt_t<Idx, TypeList<Operations...>>...>;
    };

    template <typename... Operations>
    using FilteredOperations = typename FilteredOps<filtered_index_sequence_t<NotIsUnaryRestriction, TypeList<std::decay_t<Operations>...>>, Operations...>::type;

    template <typename Enabler, typename... Operations>
    struct NewOperationTuple_;

    template <typename... Operations_>
    struct NewOperationTuple_<std::enable_if_t<!allUnaryTypes<Operations_...>, void>, Operations_...> {
        using Operations = TypeList<std::decay_t<Operations_>...>;
        using Indexes = filtered_index_sequence_t<NotIsUnaryRestriction, Operations>;
        using InstancesType = FilteredOperations<Operations_...>;
        static constexpr size_t size{sizeof...(Operations_)};
        InstancesType instances;
    };

    template <typename... Operations_>
    struct NewOperationTuple_<std::enable_if_t<allUnaryTypes<Operations_...>, void>, Operations_...> {
        using Operations = TypeList<std::decay_t<Operations_>...>;
        using Indexes = filtered_index_sequence_t<NotIsUnaryRestriction, Operations>;
        static constexpr size_t size{ sizeof...(Operations_) };
    };

    template <typename... Operations>
    using NewOperationTuple = NewOperationTuple_<void, Operations...>;

    template <typename IndexSeq, size_t IdxValue>
    struct GetIndex;

    template <typename IndexSeq, typename IndexSeqOut, size_t IdxValue>
    struct GetIndexHelper;

    template <size_t... Idx, size_t... IdxOut, size_t IdxValue>
    struct  GetIndexHelper<std::index_sequence<Idx...>, std::index_sequence<IdxOut...>, IdxValue> {
        static constexpr size_t value = []() {
            size_t result = 0;
            ((result = (Idx == IdxValue ? IdxOut : result)), ...);
            return result;
            }();
    };

    template <size_t... Idx, size_t IdxValue>
    struct GetIndex<std::index_sequence<Idx...>, IdxValue> {
        static constexpr size_t value = GetIndexHelper<std::index_sequence<Idx...>, decltype(std::make_index_sequence<sizeof...(Idx)>()), IdxValue>::value;
    };

    // We need to return auto instead of auto& because we may be returning temporaries
    // As observed in get<>(Tuple<...>), returning a const& as auto,
    // may lead to local memory accesses in the GPU
    template <size_t Idx, typename... Operations>
    FK_HOST_DEVICE_CNST decltype(auto) get(const NewOperationTuple<Operations...>& opTuple){
        if constexpr (isUnaryType<TypeAt_t<Idx, TypeList<Operations...>>>) {
            return typename TypeAt_t<Idx, TypeList<Operations...>>::Operation::InstantiableType{};
        } else {
            return get<GetIndex<typename NewOperationTuple<Operations...>::Indexes, Idx>::value>(opTuple.instances);
        }
    }

    template <size_t... Idx, typename... IOps>
    FK_HOST_DEVICE_CNST decltype(auto) make_new_operation_tuple_helper(const std::index_sequence<Idx...>&, const IOps&... iOps) {
        // 1. Pack arguments into a tuple ONCE.
        //    Using forward_as_tuple preserves references (Op1&, Op2&...)
        auto args_tuple = forward_as_tuple(iOps...);

        // 2. Expand using the tuple-based 'get'.
        //    We move(args_tuple) to ensure we forward the r-valueness if needed.
        using ResultType = NewOperationTuple<std::decay_t<IOps>...>;

        // Note: We use fk::get (the one that takes a Tuple) here, not the pack one.
        return ResultType{{get<Idx>(std::move(args_tuple))...}};
    }

    template <typename... IOps>
    FK_HOST_DEVICE_CNST decltype(auto) make_new_operation_tuple(const IOps&... iOps) {
        if constexpr (allUnaryTypes<IOps...>) {
            return NewOperationTuple<std::decay_t<IOps>...>{};
        } else {
            using IdxType = typename NewOperationTuple<std::decay_t<IOps>...>::Indexes;
            return make_new_operation_tuple_helper(IdxType{}, iOps...);
        }
    }

    namespace detail {
        template <size_t... Idx1, size_t... Idx2,
                  typename... IOps1, typename... IOps2>
        FK_HOST_DEVICE_CNST decltype(auto) cat_helper(const std::index_sequence<Idx1...>&,
                                               const std::index_sequence<Idx2...>&,
                                               const NewOperationTuple<IOps1...>& opTup1,
                                               const NewOperationTuple<IOps2...>& opTup2) {
            return make_new_operation_tuple(get<Idx1>(opTup1)..., get<Idx2>(opTup2)...);
        }
    } // namespace detail

    template <typename... IOps1, typename... IOps2>
    FK_HOST_DEVICE_CNST decltype(auto) cat(const NewOperationTuple<IOps1...>& opTup1,
                                           const NewOperationTuple<IOps2...>& opTup2) {
        return detail::cat_helper(std::make_index_sequence<NewOperationTuple<IOps1...>::size>{},
                          std::make_index_sequence<NewOperationTuple<IOps2...>::size>{},
                          opTup1, opTup2);
    }
} // namespace fk

#endif
