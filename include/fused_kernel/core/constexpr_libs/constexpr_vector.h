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
#include <fused_kernel/core/utils/vector_utils.h>

namespace cxp {
    template <size_t Idx, typename T>
    FK_HOST_DEVICE_CNST typename fk::VBase<T> v_get(const T& vecVal) {
        static_assert(fk::validCUDAVec<T>, "v_get can only be used with vector types");
        static_assert(Idx < fk::cn<T>, "Index out of bounds");
        return fk::vectorAt(Idx, vecVal);
    }

    namespace internal {
        template <size_t... Idx, typename T>
        FK_HOST_DEVICE_CNST auto v_sum_helper(const std::index_sequence<Idx...>&, const T& vecVal) {
            return sum(v_get<Idx>(vecVal)...);
        }
        template <typename TargetT, typename SourceT, size_t... Idx>
        FK_HOST_DEVICE_CNST TargetT v_static_cast_helper(const SourceT& source, const std::index_sequence<Idx...>&) {
            return fk::make_<TargetT>(static_cast<fk::VBase<TargetT>>(v_get<Idx>(source))...);
        }
    } // namespace internal

    template <typename T>
    FK_HOST_DEVICE_CNST auto v_sum(const T& vecVal) {
        static_assert(fk::validCUDAVec<T>, "v_sum can only be used with vector types");
        return internal::v_sum_helper(std::make_index_sequence<fk::cn<T>>{}, vecVal);
    }

    template <typename TargetT, typename SourceT>
    FK_HOST_DEVICE_CNST TargetT v_static_cast(const SourceT& source) {
        if constexpr (std::is_same_v<TargetT, SourceT>) {
            return source;
        } else if constexpr (fk::validCUDAVec<SourceT>) {
            static_assert(fk::cn<TargetT> == fk::cn<SourceT>, "Can not cast to a type with different number of channels");
            return internal::v_static_cast_helper<TargetT>(source, std::make_index_sequence<fk::cn<SourceT>>{});
        } else {
            static_assert(!fk::validCUDAVec<TargetT> || (fk::cn<TargetT> == 1),
                "Can not convert a fundamental type to a vetor type with more than one channel");
            if constexpr (fk::validCUDAVec<TargetT>) {
                return fk::make_<TargetT>(static_cast<fk::VBase<TargetT>>(source));
            } else {
                return static_cast<TargetT>(source);
            }
        }
    }

    template <typename T>
    FK_HOST_DEVICE_CNST bool v_and(const T& value) {
        const auto valBool = v_static_cast<fk::VectorType_t<bool, fk::cn<T>>>(value);
        return valBool;
    }
 } // namespace cxp

#endif // CXP_CONSTEXPR_VECTOR_H