/* Copyright 2025-2026 Oscar Amoros Huguet
   Copyright 2025-2026 Grup Mediapro S.L.U.

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
#include <cuda/std/algorithm> // Requires CUDA 13.3+ (libcu++/CCCL >= 3.3); enforced in core/utils/utils.h
#include <cuda/std/bit>
#include <cuda/std/utility>

namespace cxp {
using cuda::std::bit_cast;
namespace base {
using cuda::std::clamp;
using cuda::std::cmp_equal;
using cuda::std::cmp_greater;
using cuda::std::cmp_greater_equal;
using cuda::std::cmp_less;
using cuda::std::cmp_less_equal;
using cuda::std::cmp_not_equal;
using cuda::std::max;
using cuda::std::min;
} // namespace base
} // namespace cxp
#else
#include <algorithm>
#include <bit>
#include <utility>
namespace cxp {
using std::bit_cast;
namespace base {
using std::clamp;
using std::cmp_equal;
using std::cmp_greater;
using std::cmp_greater_equal;
using std::cmp_less;
using std::cmp_less_equal;
using std::cmp_not_equal;
using std::max;
using std::min;
} // namespace base
} // namespace cxp
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

    // Hand-rolled: std::isnan is constexpr only since C++23 (P0533) and the CPU backend compiles as plain C++20.
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

    // Hand-rolled: std::isinf is constexpr only since C++23 (P0533) and the CPU backend compiles as plain C++20.
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

    // Hand-rolled: no std equivalent.
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

    struct cmp_equal {
        struct BaseFunc {
            using InstanceType = fk::BinaryType;
            template <std::integral ST1, std::integral ST2>
            FK_HOST_DEVICE_FUSE bool exec(const ST1 &s1, const ST2 &s2) {
                return base::cmp_equal(s1, s2);
            }
        };
        CXP_F_FUNC
    };

    // safe cmp_equal all types, including floating point
    // Hand-rolled: std::cmp_equal and friends only accept integral types, the *_u variants also take floating point.
    struct cmp_equal_u {
        struct BaseFunc {
            using InstanceType = fk::BinaryType;
            template <typename ST1, typename ST2>
            FK_HOST_DEVICE_FUSE bool exec(const ST1 &s1, const ST2 &s2) {
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
                    if (s1 < 0)
                        return false; // Negative cannot equal any unsigned.
                    return static_cast<std::make_unsigned_t<ST1>>(s1) == s2;
                } else {
                    // T is unsigned, U is signed, both are integers
                    if (s2 < 0)
                        return false; // Negative cannot equal any unsigned.
                    return s1 == static_cast<std::make_unsigned_t<ST2>>(s2);
                }
            }
        };
        CXP_F_FUNC
    };

    // safe cmp_not_equal universal
    struct cmp_not_equal_u {
        struct BaseFunc {
            using InstanceType = fk::BinaryType;
            template<typename ST1, typename ST2>
            FK_HOST_DEVICE_FUSE bool exec(const ST1& s1, const ST2& s2) {
                return !cmp_equal_u::BaseFunc::exec(s1, s2);
            }
        };
        CXP_F_FUNC
    };

    struct cmp_not_equal {
        struct BaseFunc {
            using InstanceType = fk::BinaryType;
            template <std::integral ST1, std::integral ST2>
            FK_HOST_DEVICE_FUSE bool exec(const ST1 &s1, const ST2 &s2) {
                return base::cmp_not_equal(s1, s2);
            }
        };
        CXP_F_FUNC
    };

    // safe_cmp_less
    struct cmp_less_u {
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
                    if (s2 < 0)
                        return false; // Unsigned is never less than a signed negative.
                    return s1 < static_cast<std::make_unsigned_t<ST2>>(s2);
                }
            }
        };
        CXP_F_FUNC
    };

    struct cmp_less {
        struct BaseFunc {
            using InstanceType = fk::BinaryType;
            template <std::integral ST1, std::integral ST2>
            FK_HOST_DEVICE_FUSE bool exec(const ST1 &s1, const ST2 &s2) {
                return base::cmp_less(s1, s2);
            }
        };
        CXP_F_FUNC
    };

    // safe_cmp_greater
    struct cmp_greater {
        struct BaseFunc {
            using InstanceType = fk::BinaryType;
            template<std::integral ST1, std::integral ST2>
            FK_HOST_DEVICE_FUSE bool exec(const ST1& s1, const ST2& s2) {
                return base::cmp_greater(s1, s2);
            }
        };
        CXP_F_FUNC
    };

    struct cmp_greater_u {
        struct BaseFunc {
            using InstanceType = fk::BinaryType;
            template <typename ST1, typename ST2> FK_HOST_DEVICE_FUSE bool exec(const ST1 &s1, const ST2 &s2) {
                return cmp_less_u::BaseFunc::exec(s2, s1);
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
                return !cmp_greater_u::BaseFunc::exec(s1, s2);
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
                return !cmp_less_u::BaseFunc::exec(s1, s2);
            }
        };
        CXP_F_FUNC
    };

    // Hand-rolled: neither C++20 std::round nor cuda::std::round (libcu++/CCCL 3.3) is constexpr.
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

    // Hand-rolled: neither C++20 std::floor nor cuda::std::floor (libcu++/CCCL 3.3) is constexpr.
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

    // Hand-rolled: neither C++20 std::nearbyint nor cuda::std::nearbyint (libcu++/CCCL 3.3) is constexpr.
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
                return base::max(s1, s2);
            }
        };
        CXP_F_FUNC
        template <typename ST>
        FK_HOST_DEVICE_FUSE ST f(const ST& s) {
            return s; 
        }
    };

    // Hand-rolled: cuda::std::fmaxf drops the IEEE-754 signed-zero ordering when constant-evaluated
    // (fmax(-0.f, +0.f) yields -0.f) and host libm differs from the device instruction on signed zeros,
    // breaking the compile-time/runtime and CPU/GPU bit-identity this library guarantees.
    struct fmaxf {
        struct BaseFunc {
            using InstanceType = fk::BinaryType;
            FK_HOST_DEVICE_FUSE float exec(const float s1, const float s2) {
                // 1. IEEE-754 NaN Rules: If one is NaN, return the other.
                const bool s1_is_nan = isnan::BaseFunc::exec(s1);
                const bool s2_is_nan = isnan::BaseFunc::exec(s2);
                if (s1_is_nan && s2_is_nan) {
                    return s1; // return NaN if both are NaN
                } else if (s1_is_nan) {
                    return s2;
                } else if (s2_is_nan) {
                    return s1;
                }

                // 2. The Signed Zero Trap (-0.0 vs +0.0)
                // If they evaluate as equal, return the one with the positive sign bit.
                if (s1 == s2) {
                    // Extract the 31st bit (sign bit) via our unified bit_cast
                    bool s1_is_negative = (cxp::bit_cast<uint>(s1) & 0x80000000) != 0;
                    return s1_is_negative ? s2 : s1;
                }

                // 3. base::max accepts floats and we already handled NaN and signed zero
                return base::max(s1, s2);
            }
        };
        CXP_F_FUNC
        FK_HOST_DEVICE_FUSE float f(const float &s) { return s; }
    };

    struct min {
        struct BaseFunc {
            using InstanceType = fk::BinaryType;
            template <typename ST>
            FK_HOST_DEVICE_FUSE auto exec(const ST& s1, const ST& s2) 
                -> std::enable_if_t<std::is_fundamental_v<ST>, ST> {
                return base::min(s1, s2);
            }
        };
        CXP_F_FUNC
        template <typename ST>
        FK_HOST_DEVICE_FUSE ST f(const ST& value) {
            return value; 
        }
    };

    // Hand-rolled: same signed-zero bit-identity rationale as fmaxf.
    struct fminf {
        struct BaseFunc {
            using InstanceType = fk::BinaryType;
            FK_HOST_DEVICE_FUSE float exec(const float s1, const float s2) {
                // 1. IEEE-754 NaN Rules: If one is NaN, return the other.
                const bool s1_is_nan = isnan::BaseFunc::exec(s1);
                const bool s2_is_nan = isnan::BaseFunc::exec(s2);
                if (s1_is_nan && s2_is_nan) {
                    return s1; // return NaN if both are NaN
                } else if (s1_is_nan) {
                    return s2;
                } else if (s2_is_nan) {
                    return s1;
                }

                // 2. The Signed Zero Trap (-0.0 vs +0.0)
                // If they evaluate as equal, return the one with the negative sign bit.
                if (s1 == s2) {
                    // Extract the 31st bit (sign bit) via our unified bit_cast
                    bool s1_is_negative = (cxp::bit_cast<uint>(s1) & 0x80000000) != 0;
                    return s1_is_negative ? s1 : s2;
                }

                // 3. base::min accepts floats and we already handled NaN and signed zero
                return base::min(s1, s2);
            }
        };
        CXP_F_FUNC
        FK_HOST_DEVICE_FUSE float f(const float &s) { return s; }
    };

    // Hand-rolled: same signed-zero bit-identity rationale as fmaxf.
    struct fmax {
        struct BaseFunc {
            using InstanceType = fk::BinaryType;
            FK_HOST_DEVICE_FUSE auto exec(const double s1, const double s2) {
                // 1. IEEE-754 NaN Rules: If one is NaN, return the other.
                const bool s1_is_nan = isnan::BaseFunc::exec(s1);
                const bool s2_is_nan = isnan::BaseFunc::exec(s2);
                if (s1_is_nan && s2_is_nan) {
                    return s1; // return NaN if both are NaN
                } else if (s1_is_nan) {
                    return s2;
                } else if (s2_is_nan) {
                    return s1;
                }

                // 2. The Signed Zero Trap (-0.0 vs +0.0)
                if (s1 == s2) {
                    // Use ulonglong and the 64-bit sign mask
                    bool s1_is_negative = (cxp::bit_cast<ulonglong>(s1) & 0x8000000000000000ULL) != 0;
                    return s1_is_negative ? s2 : s1;
                }

                // 3. base::max accepts double and we already handled NaN and signed zero
                return base::max(s1, s2);
            }
        };
        CXP_F_FUNC
    };

    // Hand-rolled: same signed-zero bit-identity rationale as fmaxf.
    struct fmin {
        struct BaseFunc {
            using InstanceType = fk::BinaryType;
            FK_HOST_DEVICE_FUSE auto exec(const double s1, const double s2) {
                // 1. IEEE-754 NaN Rules: If one is NaN, return the other.
                const bool s1_is_nan = isnan::BaseFunc::exec(s1);
                const bool s2_is_nan = isnan::BaseFunc::exec(s2);
                if (s1_is_nan && s2_is_nan) {
                    return s1; // return NaN if both are NaN
                } else if (s1_is_nan) {
                    return s2;
                } else if (s2_is_nan) {
                    return s1;
                }

                // 2. The Signed Zero Trap (-0.0 vs +0.0)
                if (s1 == s2) {
                    // Use ulonglong and the 64-bit sign mask
                    bool s1_is_negative = (cxp::bit_cast<ulonglong>(s1) & 0x8000000000000000ULL) != 0;
                    return s1_is_negative ? s1 : s2;
                }

                // 3. base::min accepts double and we already handled NaN and signed zero
                return base::min(s1, s2);
            }
        };

        CXP_F_FUNC
    };

    // Hand-rolled: std::abs is constexpr only since C++23 and has no overloads for unsigned types.
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

    // NON STD FUNCTIONS
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

    // Hand-rolled: neither C++20 std::ldexp nor cuda::std::ldexpf (libcu++/CCCL 3.3) is constexpr.
    struct ldexpf {
        struct BaseFunc {
            using InstanceType = fk::BinaryType;
            FK_HOST_DEVICE_FUSE
            float exec(const float x, const int exp) 
            { 
                // Replace union with standard C++20 bit_cast
                uint ui = cxp::bit_cast<uint>(x);

                // Extract the 8-bit exponent
                int32_t e = (ui >> 23) & 0xFF;

                // 1. Handle edge cases (Zero, Subnormals, Infinity, NaN)
                if (e == 0) {
                    // If the number is exactly 0.0f or -0.0f, return it.
                    if ((ui & 0x7FFFFFFF) == 0)
                        return x;

                    // Subnormal normalization: multiply by 2^24 to push into normal range
                    ui = cxp::bit_cast<uint>(x * 16777216.0f); // Re-cast after math
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

    // Hand-rolled: std::exp is constexpr only since C++26 (P1383) and cuda::std::expf (libcu++/CCCL 3.3)
    // is not constexpr; also keeps CPU/GPU results bit-identical, which libm vs libdevice does not.
    struct expf {
        struct BaseFunc {
            using InstanceType = fk::UnaryType;
            FK_HOST_DEVICE_FUSE auto exec(const float x) {
                // 1. Handle edge cases FIRST to protect constexpr evaluation
                if (isnan::BaseFunc::exec(x))
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
                    uint twok_i = static_cast<uint>(k + 127) << 23;
                    return poly * cxp::bit_cast<float>(twok_i);
                } else {
                    // Fallback for subnormal numbers using your custom ldexpf
                    return cxp::ldexpf::f(poly, k);
                }
            }
        };
        CXP_F_FUNC
    };

    struct clamp {
        struct BaseFunc {
            using InstanceType = fk::TernaryType;
            template <typename T>
            FK_HOST_DEVICE_FUSE T exec(const T val, const T minV, const T maxV) {
                return base::clamp(val, minV, maxV);
            }
        };
        CXP_F_FUNC
    };

#undef CXP_F_FUNC

} // namespace cxp

#endif
