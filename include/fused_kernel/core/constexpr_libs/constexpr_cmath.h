/* Copyright 2025 Oscar Amoros Huguet
   Copyright 2025 Grup Mediapro S.L.U.

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

#ifdef __CUDACC__
#include <cuda/std/bit>
namespace cxp {
    using cuda::std::bit_cast;
}
#else
#include <bit>
namespace cxp {
    using std::bit_cast;
}
#endif

namespace cxp {
    template <typename T>
    constexpr T minValue = std::numeric_limits<T>::lowest();

    template <typename T>
    constexpr T maxValue = std::numeric_limits<T>::max();

    template <typename T>
    constexpr T smallestPositiveValue = std::is_floating_point_v<T> ? std::numeric_limits<T>::min() : static_cast<T>(1);

#define CXP_F_FUNC                                     \
    template <typename... Types>                       \
    FK_HOST_DEVICE_FUSE auto f(const Types&... vals) { \
        return Exec<BaseFunc>::exec(vals...);          \
    }

    struct isnan {
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
        struct BaseFunc {
            using InstanceType = fk::UnaryType;
            template <typename ST>
            FK_HOST_DEVICE_FUSE bool exec(const ST& s) {
                return s == s && s != ST(0) && s + s == s;
            }
        };
        CXP_F_FUNC
    };

    struct is_even {
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

    // safe_cmp_equal
    struct cmp_equal {
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
        struct BaseFunc {
            using InstanceType = fk::BinaryType;
            template<typename ST1, typename ST2>
            FK_HOST_DEVICE_FUSE bool exec(const ST1& s1, const ST2& s2) {
                return !cmp_equal::BaseFunc::exec(s1, s2);
            }
        };
        CXP_F_FUNC
    };

    // safe_cmp_less
    struct cmp_less {
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
        struct BaseFunc {
            using InstanceType = fk::BinaryType;
            template<typename ST1, typename ST2>
            FK_HOST_DEVICE_FUSE bool exec(const ST1& s1, const ST2& s2) {
                return cmp_less::BaseFunc::exec(s2, s1);
            }
        };
        CXP_F_FUNC
    };

    // safe_cmp_less_equal
    struct cmp_less_equal {
        struct BaseFunc {
            using InstanceType = fk::BinaryType;
            template<typename ST1, typename ST2>
            FK_HOST_DEVICE_FUSE bool exec(const ST1& s1, const ST2& s2) {
                // Equivalent to "not greater than".
                return !cmp_greater::BaseFunc::exec(s1, s2);
            }
        };
        CXP_F_FUNC
    };

    // safe_cmp_greater_equal
    struct cmp_greater_equal {
        struct BaseFunc {
            using InstanceType = fk::BinaryType;
            template<typename ST1, typename ST2>
            FK_HOST_DEVICE_FUSE bool exec(const ST1& s1, const ST2& s2) {
                // Equivalent to "not less than".
                return !cmp_less::BaseFunc::exec(s1, s2);
            }
        };
        CXP_F_FUNC
    };

    struct round {
        struct BaseFunc {
            using InstanceType = fk::UnaryType;
            template <typename ST>
            FK_HOST_DEVICE_FUSE ST exec(const ST& s) {
                static_assert(std::is_floating_point_v<ST>, "Input must be a floating-point type");
                if (isnan::BaseFunc::exec(s) || isinf::BaseFunc::exec(s)) {
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
        struct BaseFunc {
            using InstanceType = fk::UnaryType;
            template <typename ST>
            FK_HOST_DEVICE_FUSE ST exec(const ST& s) {
                static_assert(std::is_floating_point_v<ST>, "Input must be a floating-point type");
                if (isnan::BaseFunc::exec(s) || isinf::BaseFunc::exec(s) || (s == static_cast<ST>(0))) {
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

    struct nearbyint {
        struct BaseFunc {
            using InstanceType = fk::UnaryType;
            template <typename ST>
            FK_HOST_DEVICE_FUSE ST exec(const ST& s) {
                static_assert(std::is_floating_point_v<ST>, "Input must be a floating-point type");
                if (isnan::BaseFunc::exec(s) || isinf::BaseFunc::exec(s) || (s == static_cast<ST>(0))) {
                    return s;
                }
                ST fl{0};
                // This is doing the same as floor, but we don't want to execute the previous if again
                if constexpr (std::is_same_v<ST, double>) {
                    // For double, we can use long long safely
                    const long long intPart = static_cast<long long>(s);
                    if (s < ST(0) && s != static_cast<ST>(intPart)) {
                        fl = static_cast<ST>(intPart - 1);
                    } else {
                        fl = static_cast<ST>(intPart);
                    }
                } else {
                    // For float, we use int to avoid performance issues with long long
                    const ST intPart = static_cast<int>(s);
                    if (s < ST(0) && s != static_cast<ST>(intPart)) {
                        fl = static_cast<ST>(intPart - 1);
                    } else {
                        fl = static_cast<ST>(intPart);
                    }
                }
                const ST frac = s - fl;
                if (frac < static_cast<ST>(0.5)) {
                    // Closer to the floor.
                    return fl;
                } else if (frac > static_cast<ST>(0.5)) {
                    // Closer to the ceiling.
                    return fl + static_cast<ST>(1.0);
                } else {
                    // Exactly 0.5, the tie-breaker case.
                    // We must round to the nearest *even* integer.
                    if constexpr (std::is_same_v<ST, double>) {
                        const auto i_f = static_cast<long long>(fl);
                        if (is_even::BaseFunc::exec(i_f)) {
                            // Floor is even, so round to it.
                            return fl;
                        } else {
                            // Floor is odd, so round to the ceiling (which must be even).
                            return fl + static_cast<ST>(1.0);
                        }
                    } else {
                        const auto i_f = static_cast<int>(fl);
                        if (is_even::BaseFunc::exec(i_f)) {
                            // Floor is even, so round to it.
                            return fl;
                        } else {
                            // Floor is odd, so round to the ceiling (which must be even).
                            return fl + static_cast<ST>(1.0);
                        }
                    }
                }
            }
        };
        CXP_F_FUNC
    };

    struct max {
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
        FK_HOST_DEVICE_FUSE ST f(const ST& s) {
            return s; 
        }
    };

    struct min {
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
        struct BaseFunc {
            using InstanceType = fk::UnaryType;
            template <typename ST>
            FK_HOST_DEVICE_FUSE auto exec(const ST& s) {
                return static_cast<fk::VBase<OT>>(s);
            }
        };
        template <typename T>
        FK_HOST_DEVICE_FUSE auto f(const T& val) {
            static_assert(fk::AreSS<OT, T>::value || fk::AreVVEqCN<OT, T>::value,
                "Can only cast from scalar to scalar or from vector to vector of the same number of channels.");
            return Exec<BaseFunc>::exec(val);
        }
    };

    struct ldexpf {
        struct BaseFunc {
            using InstanceType = fk::BinaryType;
            FK_HOST_DEVICE_FUSE
            float exec(const float x, const int exp) 
            { 
                // Replace union with standard C++20 bit_cast
                uint32_t ui = cxp::bit_cast<uint32_t>(x);

                // Extract the 8-bit exponent
                int32_t e = (ui >> 23) & 0xFF;

                // 1. Handle edge cases (Zero, Subnormals, Infinity, NaN)
                if (e == 0) {
                    // If the number is exactly 0.0f or -0.0f, return it.
                    if ((ui & 0x7FFFFFFF) == 0)
                        return x;

                    // Subnormal normalization: multiply by 2^24 to push into normal range
                    ui = cxp::bit_cast<uint32_t>(x * 16777216.0f); // Re-cast after math
                    e = ((ui >> 23) & 0xFF) - 24;
                } else if (e == 0xFF) {
                    // Infinity or NaN
                    return x;
                }

                // 2. Apply the exponent shift
                e += exp;

                // 3. Check for Overflow
                if (e > 254) {
                    // Force the exponent to the infinity marker while keeping the sign bit
                    ui = (ui & 0x80000000) | 0x7F800000;
                    return cxp::bit_cast<float>(ui);
                }

                // 4. Check for Underflow
                if (e <= 0) {
                    // Total underflow to zero
                    if (e <= -24) {
                        ui &= 0x80000000; // Preserve sign bit, zero everything else
                        return cxp::bit_cast<float>(ui);
                    }

                    // Partial underflow to subnormal
                    ui = (ui & 0x807FFFFF) | ((e + 24) << 23);
                    return cxp::bit_cast<float>(ui) * 5.960464477539063e-8f;
                }

                // 5. Standard Reconstruction
                ui = (ui & 0x807FFFFF) | (e << 23);
                return cxp::bit_cast<float>(ui);
            }
        };
        CXP_F_FUNC
    };

    struct expf {
        struct BaseFunc {
            using InstanceType = fk::UnaryType;
            FK_HOST_DEVICE_FUSE auto exec(const float x) {
                // 1. Handle edge cases FIRST to protect constexpr evaluation
                if (cxp::isnan::f(x))
                    return x;
                if (x < -103.972f)
                    return 0.0f; // Hard underflow
                if (x > 88.7228f)
                    return cxp::bit_cast<float>(0x7F800000); // 0x7F800000 is +INFINITY

                // 2. Range Reduction: x = k * ln(2) + r
                const float INV_LN2 = 1.44269504f;

                float k_f = cxp::round::f(x * INV_LN2);
                int32_t k = static_cast<int32_t>(k_f);

                // Split ln(2) to prevent catastrophic cancellation
                const float LN2_HI = 0.693145751953125f;
                const float LN2_LO = 1.428606765330187e-06f;

                float r = x - (k_f * LN2_HI) - (k_f * LN2_LO);

                // 3. Degree-6 Polynomial Approximation (ILP Optimized)
                float r2 = r * r;
                float r3 = r2 * r;

                // Group 1: 1 + r + r^2/2
                float p1 = 1.0f + r + r2 * 0.5f;

                // Group 2: 1/6 + r/24 + r^2/120
                float p2 = 0.16666667f + r * 0.04166667f + r2 * 0.00833333f;

                // Recombine with 1/720 term
                float poly = p1 + r3 * (p2 + r3 * 0.00138889f);

                // 4. Reconstruction: e^x = e^r * 2^k
                if (k >= -126) {
                    // Fast bit manipulation for normal IEEE 754 floats
                    uint32_t twok_i = static_cast<uint32_t>(k + 127) << 23;
                    return poly * cxp::bit_cast<float>(twok_i);
                } else {
                    // Fallback for subnormal numbers using your custom ldexpf
                    return cxp::ldexpf::f(poly, k);
                }
            }
        };
        CXP_F_FUNC
    };

#undef CXP_F_FUNC

} // namespace cxp

#endif
