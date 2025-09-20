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
                if (cxp::cmp_greater::f(s, maxValOutput)) {
                    return maxValOutput;
                } else if (cxp::cmp_less::f(s, minValueOutput)) {
                    return minValueOutput;
                } else {
                    // We know that the value of input is within the
                    // numerical range of OutputType.
                    if constexpr (std::is_floating_point_v<ST> && std::is_integral_v<OT>) {
                        // For floating point to integral conversion, we need to round
                        return static_cast<fk::VBase<OT>>(cxp::round::f(s));
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
