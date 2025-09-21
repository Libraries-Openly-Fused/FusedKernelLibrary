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

#ifndef FK_CONSTEXPR_CMATH
#define FK_CONSTEXPR_CMATH

#include <fused_kernel/core/utils/utils.h>
#include <fused_kernel/core/utils/type_lists.h>
#include <fused_kernel/core/constexpr_libs/constexpr_vector_exec.h>

#include <type_traits>
#include <limits>

namespace cxp {
    template <typename T>
    constexpr T minValue = std::numeric_limits<T>::lowest();

    template <typename T>
    constexpr T maxValue = std::numeric_limits<T>::max();

    template <typename T>
    constexpr T smallestPositiveValue = std::is_floating_point_v<T> ? std::numeric_limits<T>::min() : static_cast<T>(1);

#define CXP_F_FUNC \
public: \
    template <typename... Types> \
    FK_HOST_DEVICE_FUSE auto f(const Types&... vals) { \
        return Exec<BaseFunc>::exec(vals...); \
    }

    struct isnan {
        private:
        struct BaseFunc {
            using InstanceType = fk::UnaryType;
            template <typename ST>
            FK_HOST_DEVICE_FUSE bool exec(const ST& s) {
                return s != s;
            }
        };
        CXP_F_FUNC
    };

    struct isinf {
        private:
        struct BaseFunc {
            using InstanceType = fk::UnaryType;
            template <typename ST>
            FK_HOST_DEVICE_FUSE bool exec(const ST& s) {
                return s == s && s != ST(0) && s + s == s;
            }
        };
        CXP_F_FUNC
    };

    // safe_cmp_equal
    struct cmp_equal {
        private:
        struct BaseFunc {
            using InstanceType = fk::BinaryType;
            template<typename ST1, typename ST2>
            FK_HOST_DEVICE_FUSE bool exec(const ST1& s1, const ST2& s2) {
                static_assert(!std::is_same_v<ST1, bool> && std::is_fundamental_v<ST1>,
                    "First parameter must be a fundamental type other than bool");
                static_assert(!std::is_same_v<ST2, bool> && std::is_fundamental_v<ST2>,
                    "Second parameter must be a fundamental type other than bool");
                constexpr bool isAnyFloatingPoint = std::is_floating_point_v<ST1> || std::is_floating_point_v<ST2>;
                constexpr bool areBothSigned = std::is_signed_v<ST1> == std::is_signed_v<ST2>;
                if constexpr (isAnyFloatingPoint || areBothSigned) {
                    // Safe comparison cases
                    return s1 == s2;
                } else if constexpr (std::is_signed_v<ST1>) {
                    // T is signed, U is unsigned, both are integers
                    if (s1 < 0) return false; // Negative cannot equal any unsigned.
                    return static_cast<std::make_unsigned_t<ST1>>(s1) == s2;
                } else {
                    // T is unsigned, U is signed, both are integers
                    if (s2 < 0) return false; // Negative cannot equal any unsigned.
                    return s1 == static_cast<std::make_unsigned_t<ST2>>(s2);
                }
            }
        };
        CXP_F_FUNC
    };

    // safe_cmp_not_equal
    struct cmp_not_equal {
        private:
        struct BaseFunc {
            using InstanceType = fk::BinaryType;
            template<typename ST1, typename ST2>
            FK_HOST_DEVICE_FUSE bool exec(const ST1& s1, const ST2& s2) {
                return !cmp_equal::f(s1, s2);
            }
        };
        CXP_F_FUNC
    };

    // safe_cmp_less
    struct cmp_less {
        private:
        struct BaseFunc {
            using InstanceType = fk::BinaryType;
            template<typename ST1, typename ST2>
            FK_HOST_DEVICE_FUSE bool exec(const ST1& s1, const ST2& s2) {
                static_assert(!std::is_same_v<ST1, bool> && std::is_fundamental_v<ST1>,
                    "First parameter must be a fundamental type other than bool");
                static_assert(!std::is_same_v<ST2, bool> && std::is_fundamental_v<ST2>,
                    "Second parameter must be a fundamental type other than bool");
                constexpr bool isAnyFloatingPoint = std::is_floating_point_v<ST1> || std::is_floating_point_v<ST2>;
                constexpr bool areBothSigned = std::is_signed_v<ST1> == std::is_signed_v<ST2>;
                if constexpr (isAnyFloatingPoint || areBothSigned) {
                    // Safe comparison cases
                    return s1 < s2;
                } else if constexpr (std::is_signed_v<ST1>) {
                    // T is signed, U is unsigned, both are integers
                    if (s1 < 0) return true; // Signed negative is always less than unsigned.
                    return static_cast<std::make_unsigned_t<ST1>>(s1) < s2;
                } else {
                    // T is unsigned, U is signed, both are integers
                    if (s2 < 0) return false; // Unsigned is never less than a signed negative.
                    return s1 < static_cast<std::make_unsigned_t<ST2>>(s2);
                }
            }
        };
        CXP_F_FUNC
    };

    // safe_cmp_greater
    struct cmp_greater {
        private:
        struct BaseFunc {
            using InstanceType = fk::BinaryType;
            template<typename ST1, typename ST2>
            FK_HOST_DEVICE_FUSE bool exec(const ST1& s1, const ST2& s2) {
                return cmp_less::f(s2, s1);
            }
        };
        CXP_F_FUNC
    };

    // safe_cmp_less_equal
    struct cmp_less_equal {
        private:
        struct BaseFunc {
            using InstanceType = fk::BinaryType;
            template<typename ST1, typename ST2>
            FK_HOST_DEVICE_FUSE bool exec(const ST1& s1, const ST2& s2) {
                // Equivalent to "not greater than".
                return !cmp_greater::f(s1, s2);
            }
        };
        CXP_F_FUNC
    };

    // safe_cmp_greater_equal
    struct cmp_greater_equal {
        private:
        struct BaseFunc {
            using InstanceType = fk::BinaryType;
            template<typename ST1, typename ST2>
            FK_HOST_DEVICE_FUSE bool exec(const ST1& s1, const ST2& s2) {
                // Equivalent to "not less than".
                return !cmp_less::f(s1, s2);
            }
        };
        CXP_F_FUNC
    };

    struct round {
        private:
        struct BaseFunc {
            using InstanceType = fk::UnaryType;
            template <typename ST>
            FK_HOST_DEVICE_FUSE ST exec(const ST& s) {
                static_assert(std::is_floating_point_v<ST>, "Input must be a floating-point type");
                if (isnan::f(s) || isinf::f(s)) {
                    return s;
                }
                // Casted to int instead of long long, because long long is very slow on GPU
                return (s > ST(0))
                    ? static_cast<ST>(static_cast<int>(s + ST(0.5)))
                    : static_cast<ST>(static_cast<int>(s - ST(0.5)));
            }
        };
        CXP_F_FUNC
    };

    struct floor {
    private:
        struct BaseFunc {
            using InstanceType = fk::UnaryType;
            template <typename ST>
            FK_HOST_DEVICE_FUSE ST exec(const ST& s) {
                static_assert(std::is_floating_point_v<ST>, "Input must be a floating-point type");
                if (isnan::f(s) || isinf::f(s) || (s == static_cast<ST>(0))) {
                    return s;
                }
                if constexpr (std::is_same_v<ST, double>) {
                    // For double, we can use long long safely
                    const long long intPart = static_cast<long long>(s);
                    if (s < ST(0) && s != static_cast<ST>(intPart)) {
                        return static_cast<ST>(intPart - 1);
                    }
                    return static_cast<ST>(intPart);
                } else {
                    // For float, we use int to avoid performance issues with long long}
                    const ST intPart = static_cast<int>(s);
                    if (s < ST(0) && s != static_cast<ST>(intPart)) {
                        return static_cast<ST>(intPart - 1);
                    }
                    return static_cast<ST>(intPart);
                }
            }
        };
        CXP_F_FUNC
    };

    struct max {
        private:
        struct BaseFunc {
            using InstanceType = fk::BinaryType;
            template <typename ST>
            FK_HOST_DEVICE_FUSE auto exec(const ST& s1, const ST& s2)
                -> std::enable_if_t<std::is_fundamental_v<ST>, ST> {
                return s1 >= s2 ? s1 : s2;
            }
        };
        CXP_F_FUNC
        template <typename ST>
        FK_HOST_DEVICE_FUSE ST f(const ST& value) {
            return value; 
        }
    };

    struct min {
        private:
        struct BaseFunc {
            using InstanceType = fk::BinaryType;
            template <typename ST>
            FK_HOST_DEVICE_FUSE auto exec(const ST& s1, const ST& s2) 
                -> std::enable_if_t<std::is_fundamental_v<ST>, ST> {
                return s1 <= s2 ? s1 : s2;
            }
        };
        CXP_F_FUNC
        template <typename ST>
        FK_HOST_DEVICE_FUSE ST f(const ST& value) {
            return value; 
        }
    };

    struct abs {
        private:
        struct BaseFunc {
            using InstanceType = fk::UnaryType;
            template <typename ST>
            FK_HOST_DEVICE_FUSE auto exec(const ST& s) {
                static_assert(std::is_fundamental_v<ST>, "abs does not support non fundamental types");
                if constexpr (std::is_signed_v<ST>) {
                    // For signed integrals, when x is std::numerical_limits<T>::lowest(),
                    // the result is undefined behavior in C++. So, for the sake of performance,
                    // we will not do any special treatment for those cases.
                    return s < static_cast<ST>(0) ? -s : s;
                } else {
                    return s;
                }
            }
        };
        CXP_F_FUNC
    };

    // NON SDT FUNCTIONS
    struct sum {
    private:
        struct BaseFunc {
            using InstanceType = fk::BinaryType;
            template <typename ST1, typename ST2>
            FK_HOST_DEVICE_FUSE auto exec(const ST1& s1, const ST2& s2) {
                return s1 + s2;
            }
        };
        CXP_F_FUNC
    };

    template <typename OT>
    struct cast {
    private:
        struct BaseFunc {
            using InstanceType = fk::UnaryType;
            template <typename ST>
            FK_HOST_DEVICE_FUSE auto exec(const ST& s) {
                return static_cast<fk::VBase<OT>>(s);
            }
        };
    public: 
        template <typename T>
        FK_HOST_DEVICE_FUSE auto f(const T& val) {
            static_assert(fk::AreSS<OT, T>::value || fk::AreVVEqCN<OT, T>::value,
                "Can only cast from scalar to scalar or from vector to vector of the same number of channels.");
            return Exec<BaseFunc>::exec(val);
        }
    };

    struct is_even {
        private:
        struct BaseFunc {
            using InstanceType = fk::UnaryType;
            template <typename ST>
            FK_HOST_DEVICE_FUSE bool exec(const ST& s) {
                static_assert(std::is_integral_v<ST>, "is_even only works with integral types");
                return (s & 1) == 0;
            }
        };
        CXP_F_FUNC
    };

#undef CXP_F_FUNC

} // namespace cxp

#endif
