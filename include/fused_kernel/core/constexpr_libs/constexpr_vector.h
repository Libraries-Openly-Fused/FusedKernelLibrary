/* Copyright 2025 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */


#ifndef CXP_CONSTEXPR_VECTOR_H
#define CXP_CONSTEXPR_VECTOR_H

#include <fused_kernel/core/constexpr_libs/constexpr_cmath.h>
#include <fused_kernel/core/constexpr_libs/constexpr_vector_exec.h>

namespace cxp {
    template <typename T>
    FK_HOST_DEVICE_CNST bool v_and(const T& value) {
        const auto valBool = cast<fk::VectorType_t<bool, fk::cn<T>>>::f(value);
        return valBool;
    }

    namespace internal {
        template <typename T>
        FK_HOST_DEVICE_CNST auto is_even_helper(const T& value)
            -> std::enable_if_t<std::is_integral_v<T>, bool> {
            return (value & 1) == 0;
        }
        template <size_t... Idx, typename T>
        FK_HOST_DEVICE_CNST auto is_even_helper(const std::index_sequence<Idx...>&, const T& value)
            -> std::enable_if_t<fk::validCUDAVec<T>, fk::VectorType_t<bool, fk::cn<T>>> {
            return fk::make_<fk::VectorType_t<bool, fk::cn<T>>>(is_even_heper(fk::get<Idx>(value))...);
        }
    }

    template <typename T>
    FK_HOST_DEVICE_CNST auto is_even(const T& value)
        -> std::enable_if_t<fk::validCUDAVec<T>
                            && std::is_integral_v<fk::VBase<T>>,
                                fk::VectorType_t<bool, fk::cn<T>>> {
        return internal::is_even_helper(std::make_index_sequence<fk::cn<T>>{}, value);
    }

    template <typename T>
    FK_HOST_DEVICE_CNST auto is_even(const T& value)
        -> std::enable_if_t<std::is_integral_v<T>, bool> {
        return internal::is_even_helper(value);
    }
 } // namespace cxp

#endif // CXP_CONSTEXPR_VECTOR_H