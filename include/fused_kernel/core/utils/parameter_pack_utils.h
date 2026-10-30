/* Copyright 2023 Grup Mediapro S.L.U. (Oscar Amoros Huguet)
   Copyright 2023-2026 Oscar Amoros Huguet

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

    namespace pp_detail {
        // A parameter that accepts any argument and ignores it: the I leading parameters of Nth<I>::pick
        template <size_t Skipped>
        struct Skip {
            template <typename U>
            FK_HOST_DEVICE_CNST Skip(U&&) {}
        };
        template <size_t I, typename Seq = std::make_index_sequence<I>>
        struct Nth;
        template <size_t I, size_t... S>
        struct Nth<I, std::index_sequence<S...>> {
            template <typename T, typename... Rest>
            FK_HOST_DEVICE_CNST static T&& pick(Skip<S>..., T&& x, Rest&&...) { return static_cast<T&&>(x); }
        };
    } // namespace pp_detail

    // Util to get the parameters of a parameter pack.
    // It does not build a Tuple of references to all the arguments: such an object has one pointer per argument at every call, the optimizers
    // remove it for small packs, but when the arguments are the operations of a fused kernel (each one a __grid_constant__ kernel parameter)
    // and there are more than 1024 of them, nvcc keeps it as a local array of pointers (8 KB), takes the address of every kernel parameter and
    // the kernel goes from 17 to 255 registers with thousands of spills. pick() skips the first I arguments with I empty parameters instead
    // (O(1) template depth, nothing is built), and returns exactly the reference that the original returned: T&& for Args&&... = T&&...
    template <size_t I, typename... Args>
    FK_HOST_DEVICE_CNST decltype(auto) get_arg(Args&&... args) {
        static_assert(I < sizeof...(Args), "get_arg: the index is out of range");
        return pp_detail::Nth<I>::pick(std::forward<Args>(args)...);
    }

    // Util to get the last parameter of a parameter pack
    template <typename... Args>
    FK_HOST_DEVICE_CNST decltype(auto) ppLast(Args&&... args) {
        return get_arg<sizeof...(args) - 1>(std::forward<Args>(args)...);
    }

    // Util to get the first parameter of a parameter pack
    template <typename... Args>
    FK_HOST_DEVICE_CNST decltype(auto) ppFirst(Args&&... args) {
        return get_arg<0>(std::forward<Args>(args)...);
    }

    template <size_t idx, size_t... iseq>
    constexpr inline size_t get_index_f(const std::index_sequence<iseq...>&) {
        return get_arg<idx>(iseq...);
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
