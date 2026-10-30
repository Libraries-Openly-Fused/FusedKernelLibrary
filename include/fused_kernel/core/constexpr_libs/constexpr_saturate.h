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
#include <fused_kernel/core/constexpr_libs/constexpr_vector_exec.h>

namespace cxp {
    template <typename OT>
    struct saturate_cast {
        private:
        struct BaseFunc {
            using InstanceType = fk::UnaryType;
            template <typename ST>
            FK_HOST_DEVICE_FUSE auto exec(const ST& s) {
                constexpr auto maxValOutput = maxValue<fk::VBase<OT>>;
                constexpr auto minValueOutput = minValue<fk::VBase<OT>>;
#if defined(FK_HOST_RUNTIME_MATH)
                if constexpr (std::is_floating_point_v<ST> && std::is_integral_v<fk::VBase<OT>>) {
                    if (!__builtin_is_constant_evaluated()) {
                        // At run time on the host, from a floating point type to an integer type: no branches (the ones below mispredict
                        // when the values are not predictable) and one instruction to round. The results are the ones of the code
                        // below, except where that code is not defined: a NaN gives 0 for every output type (below, it is cast to
                        // the integer type: 0 for the narrow types, the smallest int with an x86 CPU; a GPU gives 0), and a value
                        // equal to the limit of the output type when the limit is not exact in the floating point type (2^31 as a
                        // float for int) gives the limit (below, it is cast unchanged: the smallest int with an x86 CPU).
                        using Out = fk::VBase<OT>;
                        const ST hi = static_cast<ST>(maxValOutput);
                        const ST lo = static_cast<ST>(minValueOutput);
                        const bool isNaN = s != s;
                        const bool over = s >= hi;
                        const bool under = s <= lo;
                        // The value that is rounded and cast is inside the range of the output type
                        const ST inRange = (isNaN | over | under) ? static_cast<ST>(0) : s;
                        const Out rounded = static_cast<Out>(std::nearbyint(inRange));
                        return over ? static_cast<Out>(maxValOutput)
                                    : (under ? static_cast<Out>(minValueOutput) : (isNaN ? static_cast<Out>(0) : rounded));
                    }
                }
#endif
                if (cxp::cmp_greater::BaseFunc::exec(s, maxValOutput)) {
                    return maxValOutput;
                } else if (cxp::cmp_less::BaseFunc::exec(s, minValueOutput)) {
                    return minValueOutput;
                } else {
                    // We know that the value of input is within the
                    // numerical range of OutputType.
                    if constexpr (std::is_floating_point_v<ST> && std::is_integral_v<fk::VBase<OT>>) {
                        // For floating point to integral conversion, we need to round
                        return static_cast<fk::VBase<OT>>(cxp::nearbyint::BaseFunc::exec(s));
                    } else {
                        // For any other case, we can cast directly
                        return static_cast<fk::VBase<OT>>(s);
                    }
                }
            }
        };
    public:
        template <typename T>
        FK_HOST_DEVICE_FUSE auto f(const T& val) {
            static_assert(fk::AreSS<OT, T>::value || fk::AreVVEqCN<OT, T>::value,
                "saturate_cast can not cast vector to non vector or the other way arround, or from vector to vector of different channel number.");
            return Exec<BaseFunc>::exec(val);
        }
    };
    
}


#endif // CXP_CONSTEXPR_SATURATE_H
