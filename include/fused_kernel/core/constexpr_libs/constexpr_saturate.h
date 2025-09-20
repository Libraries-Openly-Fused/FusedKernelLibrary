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

#ifndef CXP_CONSTEXPR_SATURATE_H
#define CXP_CONSTEXPR_SATURATE_H

#include <fused_kernel/core/utils/utils.h>
#include <fused_kernel/core/constexpr_libs/constexpr_cmath.h>

namespace cxp {
    template <typename O, typename I>
    FK_HOST_DEVICE_CNST O saturate_cast(const I& input) {
        constexpr auto maxValOutput = maxValue<O>;
        constexpr auto minValueOutput = minValue<O>;
        if (cxp::cmp_greater::f(input, maxValOutput)) {
            return maxValOutput;
        } else if (cxp::cmp_less::f(input, minValueOutput)) {
            return minValueOutput;
        } else {
            // We know that the value of input is within the
            // numerical range of OutputType.
            if constexpr (std::is_floating_point_v<I> && std::is_integral_v<O>) {
                // For floating point to integral conversion, we need to round
                return static_cast<O>(cxp::round::f(input));
            } else {
                // For any other case, we can cast directly
                return static_cast<O>(input);
            }
        }
    }

    namespace internal {
        template <typename O, size_t... Idx, typename I>
        FK_HOST_DEVICE_CNST O v_saturate_cast_helper(const std::index_sequence<Idx...>&,
                                                     const I& input) {
            return {saturate_cast<fk::VBase<O>>(fk::get<Idx>(input))...};
        }
    }

    template <typename O, typename I>
    FK_HOST_DEVICE_CNST O v_saturate_cast(const I& input) {
        static_assert(fk::cn<I> == fk::cn<O>, "Input and Output number of channels should be the same.");
        if constexpr (fk::validCUDAVec<O>) {
            return internal::v_saturate_cast_helper<O>(std::make_index_sequence<fk::cn<I>>{}, input);
        } else {
            if constexpr (fk::validCUDAVec<I>) {
                return saturate_cast<O>(input.x);
            } else {
                return saturate_cast<O>(input);
            }
        }
    }
}


#endif // CXP_CONSTEXPR_SATURATE_H
