/* Copyright 2023 Mediaproduccion S.L.U. (Oscar Amoros Huguet)
   Copyright 2023-2025 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_PARAMETER_PACK_UTILS
#define FK_PARAMETER_PACK_UTILS

#include <fused_kernel/core/utils/utils.h>
#include <fused_kernel/core/data/tuple.h>
#include <utility>

namespace fk { // namespace fused kernel

    template <size_t N>
    constexpr auto indexSequence = std::make_index_sequence<N>{};

    // Util to get the parameters of a parameter pack
    template <size_t I, typename... Args>
    FK_HOST_DEVICE_CNST decltype(auto) get_arg(Args&&... args) {
        // 1. Args&& captures the exact category (L-value vs R-value)
        // 2. std::forward preserves it
        // 3. forward_as_tuple creates a Tuple of references (Tuple<T&, U&&...>)
        // 4. get extracts it
        // 5. decltype(auto) returns exactly that reference
        return TupleUtil::get<I>(forward_as_tuple(std::forward<Args>(args)...));
    }

    // Util to get the last parameter of a parameter pack
    template <typename... Args>
    FK_HOST_DEVICE_CNST const auto& ppLast(const Args&... args) {
        return get_arg<sizeof...(args) - 1>(args...);
    }

    // Util to get the first parameter of a parameter pack
    template <typename... Args>
    FK_HOST_DEVICE_CNST const auto& ppFirst(const Args&... args) {
        return get_arg<0>(args...);
    }

    template <size_t idx, size_t... iseq>
    constexpr inline size_t get_index_f(const std::index_sequence<iseq...>&) {
        return get_arg<static_cast<int>(idx)>(iseq...);
    }

    template <size_t idx, typename ISeq>
    constexpr size_t get_index = get_index_f<idx>(ISeq{});

    template <typename T, T idx, T... iseq>
    constexpr inline size_t get_integer_f(const std::integer_sequence<T, iseq...>&) {
        return get_arg<static_cast<size_t>(idx)>(iseq...);
    }

    template <typename T, T idx, typename ISeq>
    constexpr T get_integer = get_integer_f<T, idx>(ISeq{});
} // namespace fk

#endif
