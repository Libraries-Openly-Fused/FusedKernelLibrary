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

#ifndef FK_OPERATION_TUPLE
#define FK_OPERATION_TUPLE

#include <fused_kernel/core/data/tuple.h>
#include <fused_kernel/core/execution_model/operation_model/operation_data.h>

namespace fk {
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
    struct OperationTuple_;

    template <typename... Operations_>
    struct OperationTuple_<std::enable_if_t<!allUnaryTypes<Operations_...>, void>, Operations_...> {
        using Operations = TypeList<std::decay_t<Operations_>...>;
        using Indexes = filtered_index_sequence_t<NotIsUnaryRestriction, Operations>;
        using InstancesType = FilteredOperations<Operations_...>;
        static constexpr size_t size{sizeof...(Operations_)};
        InstancesType instances{};
    };

    template <typename... Operations_>
    struct OperationTuple_<std::enable_if_t<allUnaryTypes<Operations_...>, void>, Operations_...> {
        using Operations = TypeList<std::decay_t<Operations_>...>;
        using Indexes = filtered_index_sequence_t<NotIsUnaryRestriction, Operations>;
        static constexpr size_t size{ sizeof...(Operations_) };
    };

    template <typename... Operations>
    using OperationTuple = OperationTuple_<void, Operations...>;

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

    // As observed in get<>(Tuple<...>), returning a const& as auto,
    // may lead to local memory accesses in the GPU
    template <size_t Idx, typename... Operations>
    FK_HOST_DEVICE_CNST decltype(auto) get_opt(const OperationTuple<Operations...>& opTuple){
        if constexpr (isUnaryType<TypeAt_t<Idx, TypeList<Operations...>>>) {
            return typename TypeAt_t<Idx, TypeList<Operations...>>::Operation::InstantiableType{};
        } else {
            return get<GetIndex<typename OperationTuple<Operations...>::Indexes, Idx>::value>(opTuple.instances);
        }
    }

    template <size_t... Idx, typename... IOps>
    FK_HOST_DEVICE_CNST auto make_new_operation_tuple_helper(const std::index_sequence<Idx...>&, const IOps&... iOps) {
        // 1. Pack arguments into a tuple ONCE.
        auto args_tuple = forward_as_tuple(iOps...);

        // 2. Capture elements as value (std::decay_t)
        using ResultType = OperationTuple<std::decay_t<IOps>...>;

        // 3. Expand a single tuple using fk::get
        return ResultType{{get<Idx>(args_tuple)...}};
    }

    template <typename... IOps>
    FK_HOST_DEVICE_CNST auto make_new_operation_tuple(const IOps&... iOps) {
        if constexpr (allUnaryTypes<IOps...>) {
            return OperationTuple<std::decay_t<IOps>...>{};
        } else {
            using IdxType = typename OperationTuple<std::decay_t<IOps>...>::Indexes;
            return make_new_operation_tuple_helper(IdxType{}, iOps...);
        }
    }

    // cat and cat_helper must return auto, since they are creating a new OperationTuple 
    namespace detail {
        template <size_t... Idx1, size_t... Idx2,
                  typename... IOps1, typename... IOps2>
        FK_HOST_DEVICE_CNST auto cat_helper(const std::index_sequence<Idx1...>&,
                                               const std::index_sequence<Idx2...>&,
                                               const OperationTuple<IOps1...>& opTup1,
                                               const OperationTuple<IOps2...>& opTup2) {
            return make_new_operation_tuple(get_opt<Idx1>(opTup1)..., get_opt<Idx2>(opTup2)...);
        }
    } // namespace detail

    template <typename... IOps1, typename... IOps2>
    FK_HOST_DEVICE_CNST auto cat(const OperationTuple<IOps1...>& opTup1,
                                           const OperationTuple<IOps2...>& opTup2) {
        return detail::cat_helper(std::make_index_sequence<OperationTuple<IOps1...>::size>{},
                          std::make_index_sequence<OperationTuple<IOps2...>::size>{},
                          opTup1, opTup2);
    }
} // namespace fk

#endif
