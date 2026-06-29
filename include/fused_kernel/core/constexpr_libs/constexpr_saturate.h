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

    struct banker_round {
        struct BaseFunc {
            using InstanceType = fk::UnaryType;
            template <std::floating_point ST>
            FK_HOST_DEVICE_FUSE ST exec(const ST u) {
                if (u != u) return u; // Propagate NaN

                bool is_negative = u < 0.0;
                ST abs_v = is_negative ? -u : u;

                // For extreme limits, the float is inherently an exact integer.
                // 2^53 is 9007199254740992.0, which fits safely in a 64-bit integer.
                constexpr ST safe_limit = static_cast<ST>(9007199254740992.0);
                if (abs_v >= safe_limit) {
                    return u;
                }

                // Mathematical exact extraction (UB-free since abs_v < 2^53)
                std::int64_t i = static_cast<std::int64_t>(abs_v);

                // By Sterbenz's Lemma, this subtraction is exactly representable.
                ST diff = abs_v - static_cast<ST>(i);

                // Bypassing `==` gracefully averts pedantic `-Wfloat-equal` warnings
                if (diff > static_cast<ST>(0.5)) {
                    i += 1;
                } else if (diff >= static_cast<ST>(0.5)) {
                    if (i % 2 != 0) { // Push odd ties up to even
                        i += 1;
                    }
                }

                ST res = static_cast<ST>(i);
                return is_negative ? -res : res;
            }
        };
    };

    template <typename OT>
    struct saturate_cast {
        private:
        struct BaseFunc {
            using InstanceType = fk::UnaryType;
            template <typename ST>
            FK_HOST_DEVICE_FUSE auto exec(const ST& s) {
                using Target_T = fk::VBase<OT>;
                using Target = std::remove_cv_t<Target_T>;
                using Source_T = ST;
                using Source = std::remove_cv_t<Source_T>;

                constexpr auto maxValueTarget = maxValue<Target>;
                constexpr auto minValueTarget = minValue<Target>;
                
                // 1. Same exact types -> No-op
                if constexpr (std::is_same_v<Target, Source>) {
                    return static_cast<Target>(s);
                }
                // 2. Target is boolean -> Standard bool conversion (!= 0)
                else if constexpr (std::is_same_v<Target, bool>) {
                    return s != 0;
                }
                // 3. Floating-point to Integer -> Banker's Rounding & Clamping
                else if constexpr (std::integral<Target> && std::floating_point<Source>) {
                    // OpenCV uniquely maps NaN directly to 0
                    if (s != s)
                        return static_cast<Target>(0);

                    Source rounded = banker_round::BaseFunc::exec(s);

                    constexpr Source t_max_float = static_cast<Source>(maxValueTarget);
                    constexpr Source t_min_float = static_cast<Source>(minValueTarget);

                    // Safe bounds checking prevents C++ UB when casting floats.
                    // It seamlessly manages the inherent rounding imprecision of floats
                    // (e.g., INT32_MAX evaluates exactly to 2^31 as a float).
                    if (rounded >= t_max_float) {
                        return maxValueTarget;
                    }
                    if (rounded <= t_min_float) {
                        return minValueTarget;
                    }

                    // Safely inside bounds
                    return static_cast<Target_T>(rounded);
                }
                // 4. Integer to Integer -> Safe Mixed-Sign comparisons
                else if constexpr (std::integral<Target> && std::integral<Source>) {
                    // C++20 safe comparators avert implicit sign-promotion disasters
                    if (cmp_greater_u::BaseFunc::exec(s, maxValueTarget)) {
                        return maxValueTarget;
                    }
                    if (cmp_less_u::BaseFunc::exec(s, minValueTarget)) {
                        return minValueTarget;
                    }
                    return static_cast<Target>(s);
                }
                // 5. Floating point clamping
                else if constexpr (std::floating_point<Target> && std::floating_point<Source>) {
                    if (cmp_greater_u::BaseFunc::exec(s, maxValueTarget)) {
                        return maxValueTarget;
                    }
                    if (cmp_less_u::BaseFunc::exec(s, minValueTarget)) {
                        return minValueTarget;
                    }
                    return static_cast<Target>(s);
                }
                // 6. Fallbacks 
                else {
                    // OpenCV does not saturate continuous bounds when casting upward
                    return static_cast<Target>(s);
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
