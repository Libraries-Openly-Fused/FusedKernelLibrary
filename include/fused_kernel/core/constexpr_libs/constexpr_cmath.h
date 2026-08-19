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
#include <cmath>

#ifdef __CUDACC__
#include <cuda/std/bit>
#include <cuda/std/utility>

// Conditionally include the algorithm header if compiling on CUDA 13.3+
#if __has_include(<cuda/std/algorithm>)
#include <cuda/std/algorithm>
#endif

namespace cxp {
using cuda::std::bit_cast;
namespace base {
using cuda::std::cmp_equal;
using cuda::std::cmp_greater;
using cuda::std::cmp_greater_equal;
using cuda::std::cmp_less;
using cuda::std::cmp_less_equal;
using cuda::std::cmp_not_equal;
using cuda::std::is_constant_evaluated;

// If the header exists, alias the cuda::std versions
#if __has_include(<cuda/std/algorithm>)
using cuda::std::clamp;
using cuda::std::max;
using cuda::std::min;
#else
// Polyfill for CUDA < 13.3 where <cuda/std/algorithm> is missing
template <typename T> constexpr __host__ __device__ const T &max(const T &a, const T &b) { return (a < b) ? b : a; }

template <typename T> constexpr __host__ __device__ const T &min(const T &a, const T &b) { return (b < a) ? b : a; }

template <typename T> constexpr __host__ __device__ const T &clamp(const T &v, const T &lo, const T &hi) {
    return (v < lo) ? lo : ((hi < v) ? hi : v);
}
#endif
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
using std::is_constant_evaluated;
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
    FK_HOST_DEVICE_FUSE auto f(const Types... vals) {  \
        return Exec<BaseFunc>::exec(vals...);          \
    }

    struct isnan {
        struct BaseFunc {
            // isnan only works for double or float
            using InstanceType = fk::UnaryType;
            template <std::floating_point ST>
            FK_HOST_DEVICE_FUSE bool exec(const ST s) {
                if (base::is_constant_evaluated()) {
                    if constexpr (std::is_same_v<ST, float>) {
                        uint bits = bit_cast<uint>(s);
                        return (bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0;
                    } else {
                        ulonglong bits = bit_cast<ulonglong>(s);
                        return (bits & 0x7FF0000000000000ull) == 0x7FF0000000000000ull && 
                               (bits & 0x000FFFFFFFFFFFFFull) != 0;
                    }
                } else {
#if defined(__CUDA_ARCH__)
                    if constexpr (std::is_same_v<ST, float>) {
                        return __isnanf(s);
                    } else {
                        return __isnan(s);
                    }
#else
                    return std::isnan(s);
#endif
                }
            }
        };
        CXP_F_FUNC
    };

    struct isinf {
        struct BaseFunc {
            // isinf only works for double or float
            using InstanceType = fk::UnaryType;
            template <std::floating_point ST>
            FK_HOST_DEVICE_FUSE bool exec(const ST s) {
                if (base::is_constant_evaluated()) {
                    if constexpr (std::is_same_v<ST, float>) {
                        const uint bits = bit_cast<uint>(s);
                        // Infinity: exponent all ones, mantissa zero (sign bit ignored)
                        return (bits & 0x7FFFFFFFu) == 0x7F800000u;
                    } else {
                        const ulonglong bits = bit_cast<ulonglong>(s);
                        return (bits & 0x7FFFFFFFFFFFFFFFull) == 0x7FF0000000000000ull;
                    }
                } else {
#if defined(__CUDA_ARCH__)
                    if constexpr (std::is_same_v<ST, float>) {
                        return __isinff(s);
                    } else {
                        return __isinf(s);
                    }
#else
                    return std::isinf(s);
#endif
                }
            }
        };
        CXP_F_FUNC
    };

    struct is_even {
        struct BaseFunc {
            using InstanceType = fk::UnaryType;
            template <std::integral ST>
            FK_HOST_DEVICE_FUSE bool exec(const ST s) {
                return (s & 1) == 0;
            }
        };
        CXP_F_FUNC
    };

    struct cmp_equal {
        struct BaseFunc {
            using InstanceType = fk::BinaryType;
            template <std::integral ST1, std::integral ST2>
            FK_HOST_DEVICE_FUSE bool exec(const ST1 s1, const ST2 s2) {
                return base::cmp_equal(s1, s2);
            }
        };
        CXP_F_FUNC
    };

    // safe cmp_equal all types, including floating point
    struct cmp_equal_u {
        struct BaseFunc {
            using InstanceType = fk::BinaryType;
            template <typename ST1, typename ST2>
            FK_HOST_DEVICE_FUSE bool exec(const ST1 s1, const ST2 s2) {
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
            FK_HOST_DEVICE_FUSE bool exec(const ST1 s1, const ST2 s2) {
                return !cmp_equal_u::BaseFunc::exec(s1, s2);
            }
        };
        CXP_F_FUNC
    };

    struct cmp_not_equal {
        struct BaseFunc {
            using InstanceType = fk::BinaryType;
            template <std::integral ST1, std::integral ST2>
            FK_HOST_DEVICE_FUSE bool exec(const ST1 s1, const ST2 s2) {
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
            FK_HOST_DEVICE_FUSE bool exec(const ST1 s1, const ST2 s2) {
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
            FK_HOST_DEVICE_FUSE bool exec(const ST1 s1, const ST2 s2) {
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
            FK_HOST_DEVICE_FUSE bool exec(const ST1 s1, const ST2 s2) {
                return base::cmp_greater(s1, s2);
            }
        };
        CXP_F_FUNC
    };

    struct cmp_greater_u {
        struct BaseFunc {
            using InstanceType = fk::BinaryType;
            template <typename ST1, typename ST2>
            FK_HOST_DEVICE_FUSE bool exec(const ST1 s1, const ST2 s2) {
                return cmp_less_u::BaseFunc::exec(s2, s1);
            }
        };
        CXP_F_FUNC
    };

    struct cmp_less_equal {
        struct BaseFunc {
            using InstanceType = fk::BinaryType;
            template <std::integral ST1, std::integral ST2>
            FK_HOST_DEVICE_FUSE bool exec(const ST1 s1, const ST2 s2) {
                return base::cmp_less_equal(s1, s2);
            }
        };
        CXP_F_FUNC
    };

    // safe_cmp_less_equal
    struct cmp_less_equal_u {
        struct BaseFunc {
            using InstanceType = fk::BinaryType;
            template<typename ST1, typename ST2>
            FK_HOST_DEVICE_FUSE bool exec(const ST1 s1, const ST2 s2) {
                // Equivalent to "not greater than".
                return !cmp_greater_u::BaseFunc::exec(s1, s2);
            }
        };
        CXP_F_FUNC
    };

    struct cmp_greater_equal {
        struct BaseFunc {
            using InstanceType = fk::BinaryType;
            template <std::integral ST1, std::integral ST2>
            FK_HOST_DEVICE_FUSE bool exec(const ST1 s1, const ST2 s2) {
                return base::cmp_greater_equal(s1, s2);
            }
        };
        CXP_F_FUNC
    };

    // safe_cmp_greater_equal
    struct cmp_greater_equal_u {
        struct BaseFunc {
            using InstanceType = fk::BinaryType;
            template<typename ST1, typename ST2>
            FK_HOST_DEVICE_FUSE bool exec(const ST1 s1, const ST2 s2) {
                // Equivalent to "not less than".
                return !cmp_less_u::BaseFunc::exec(s1, s2);
            }
        };
        CXP_F_FUNC
    };

    struct round {
        struct BaseFunc {
            using InstanceType = fk::UnaryType;
            template <std::floating_point ST>
            FK_HOST_DEVICE_FUSE ST exec(const ST s) {
                if (base::is_constant_evaluated()) {
                    // Rounds half away from zero, like std::round
                    if (isnan::BaseFunc::exec(s) || isinf::BaseFunc::exec(s) || s == static_cast<ST>(0)) {
                        return s;
                    }
                    const ST absS = s < static_cast<ST>(0) ? -s : s;
                    // Above this threshold every representable value is already an integer
                    constexpr ST integralThreshold =
                        std::is_same_v<ST, float> ? static_cast<ST>(8388608.0) : static_cast<ST>(4503599627370496.0);
                    if (absS >= integralThreshold) {
                        return s;
                    }
                    // The integer part always fits in the chosen integral type thanks to the check above.
                    // int is used for float, because long long is very slow on GPU
                    using IntType = std::conditional_t<std::is_same_v<ST, float>, int, long long>;
                    const ST truncated = static_cast<ST>(static_cast<IntType>(absS));
                    const ST rounded = (absS - truncated) >= static_cast<ST>(0.5)
                                           ? truncated + static_cast<ST>(1)
                                           : truncated;
                    return s < static_cast<ST>(0) ? -rounded : rounded;
                } else {
                    return std::round(s);
                }
            }
        };
        CXP_F_FUNC
    };

    struct floor {
        struct BaseFunc {
            using InstanceType = fk::UnaryType;
            template <std::floating_point ST> 
            FK_HOST_DEVICE_FUSE ST exec(const ST s) {
                if (base::is_constant_evaluated()) {
                    // 1. Handle special cases
                    if (isnan::BaseFunc::exec(s) || isinf::BaseFunc::exec(s) || (s == static_cast<ST>(0))) {
                        return s;
                    }

                    // 2. Compute absolute value safely for constexpr
                    const ST abs_s = s < static_cast<ST>(0) ? -s : s;

                    if constexpr (std::is_same_v<ST, float>) {
                        // 2^23: Floats beyond this cannot have a fractional part.
                        // Returning early prevents Undefined Behavior during the int cast.
                        if (abs_s >= static_cast<ST>(8388608.0)) {
                            return s;
                        }

                        const ST intPart = static_cast<ST>(static_cast<int>(s));
                        if (s < static_cast<ST>(0) && s != intPart) {
                            return intPart - static_cast<ST>(1);
                        }
                        return intPart;

                    } else {
                        // 2^52: Doubles beyond this cannot have a fractional part.
                        if (abs_s >= static_cast<ST>(4503599627370496.0)) {
                            return s;
                        }

                        const ST intPart = static_cast<ST>(static_cast<long long>(s));
                        if (s < static_cast<ST>(0) && s != intPart) {
                            return intPart - static_cast<ST>(1);
                        }
                        return intPart;
                    }
                } else {
                    return std::floor(s);
                }
            }
        };
        CXP_F_FUNC
    };

    struct nearbyint {
        struct BaseFunc {
            using InstanceType = fk::UnaryType;
            template <std::floating_point ST>
            FK_HOST_DEVICE_FUSE ST exec(const ST s) {
                if (base::is_constant_evaluated()) {
                    // NaN, infinities and +-0 have no meaningful fractional part, and the
                    // subtraction below would produce a NaN for them.
                    if (isnan::BaseFunc::exec(s) || isinf::BaseFunc::exec(s) || (s == static_cast<ST>(0))) {
                        return s;
                    }
                    // Any value too large to hold a fractional part is returned by floor unchanged,
                    // so from here on fl is always representable in the integral type used for the
                    // tie-breaker.
                    const ST fl = floor::BaseFunc::exec(s);
                    const ST frac = s - fl;
                    if (frac < static_cast<ST>(0.5)) {
                        // Closer to the floor.
                        return fl;
                    } else if (frac > static_cast<ST>(0.5)) {
                        // Closer to the ceiling.
                        return fl + static_cast<ST>(1);
                    } else {
                        // Exactly 0.5, the tie-breaker case: round to the nearest *even* integer.
                        // int is used for float, because long long is very slow on GPU
                        using IntType = std::conditional_t<std::is_same_v<ST, float>, int, long long>;
                        return is_even::BaseFunc::exec(static_cast<IntType>(fl)) ? fl : fl + static_cast<ST>(1);
                    }
                } else {
                    return std::nearbyint(s);
                }
            }
        };
        CXP_F_FUNC
    };

    struct max {
        struct BaseFunc {
            using InstanceType = fk::BinaryType;
            template <typename ST>
                requires(std::is_fundamental_v<ST>)
            FK_HOST_DEVICE_FUSE auto exec(const ST s1, const ST s2) {
                return base::max(s1, s2);
            }
        };
        CXP_F_FUNC
        template <typename ST>
        FK_HOST_DEVICE_FUSE ST f(const ST& s) {
            return s; 
        }
    };

    struct signbit {
        struct BaseFunc {
            using InstanceType = fk::UnaryType;
            template <std::floating_point ST>
            FK_HOST_DEVICE_FUSE bool exec(ST s) {
                if constexpr (std::is_same_v<ST, float>) {
                    return (cxp::bit_cast<uint>(s) & 0x80000000u) != 0;
                } else {
                    return (cxp::bit_cast<ulonglong>(s) & 0x8000000000000000ULL) != 0;
                }
            }
        };
        CXP_F_FUNC
    };

    struct fmax {
        struct BaseFunc {
            using InstanceType = fk::BinaryType;
            template <std::floating_point ST>
            FK_HOST_DEVICE_FUSE ST exec(const ST s1, const ST s2) {
                if (base::is_constant_evaluated()) {
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
                        return signbit::BaseFunc::exec(s1) ? s2 : s1;
                    }

                    // 3. base::max accepts floating point and we already handled NaN and signed zero
                    return base::max(s1, s2);
                } else {
                    // fmax/fmaxf already implement the IEEE-754 NaN and signed zero rules, and map
                    // to a single FMNMX instruction on device
                    return std::fmax(s1, s2);
                }
            }
        };
        CXP_F_FUNC
        template <std::floating_point ST>
        FK_HOST_DEVICE_FUSE ST f(const ST s) {
            return s;
        }
    };

    // Kept as an alias of fmax, which is now generic over the floating point type
    using fmaxf = fmax;

    struct min {
        struct BaseFunc {
            using InstanceType = fk::BinaryType;
            template <typename ST>
            FK_HOST_DEVICE_FUSE auto exec(const ST s1, const ST s2) 
                -> std::enable_if_t<std::is_fundamental_v<ST>, ST> {
                return base::min(s1, s2);
            }
        };
        CXP_F_FUNC
        template <typename ST>
        FK_HOST_DEVICE_FUSE ST f(const ST value) {
            return value; 
        }
    };

    struct fmin {
        struct BaseFunc {
            using InstanceType = fk::BinaryType;
            template <std::floating_point ST>
            FK_HOST_DEVICE_FUSE ST exec(const ST s1, const ST s2) {
                if (base::is_constant_evaluated()) {
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
                        return signbit::BaseFunc::exec(s1) ? s1 : s2;
                    }

                    // 3. base::min accepts floating point and we already handled NaN and signed zero
                    return base::min(s1, s2);
                } else {
                    // fmin/fminf already implement the IEEE-754 NaN and signed zero rules, and map
                    // to a single FMNMX instruction on device
                    return std::fmin(s1, s2);
                }
            }
        };
        CXP_F_FUNC
        template <std::floating_point ST>
        FK_HOST_DEVICE_FUSE ST f(const ST s) {
            return s;
        }
    };

    // Kept as an alias of fmin, which is now generic over the floating point type
    using fminf = fmin;

    struct abs {
        struct BaseFunc {
            using InstanceType = fk::UnaryType;
            template <typename ST>
            FK_HOST_DEVICE_FUSE auto exec(const ST s) {
                static_assert(std::is_fundamental_v<ST>, "abs does not support non fundamental types");
                if constexpr (std::is_floating_point_v<ST>) {
                    if (base::is_constant_evaluated()) {
                        // Clearing the sign bit, instead of negating, so that -0.0 yields +0.0 and
                        // NaNs keep their payload, matching fabs
                        if constexpr (std::is_same_v<ST, float>) {
                            return bit_cast<float>(bit_cast<uint>(s) & 0x7FFFFFFFu);
                        } else {
                            return bit_cast<double>(bit_cast<ulonglong>(s) & 0x7FFFFFFFFFFFFFFFULL);
                        }
                    } else {
                        // Maps to a single instruction on device, and is a sign bit mask on host
                        return std::fabs(s);
                    }
                } else if constexpr (std::is_signed_v<ST>) {
                    // For signed integrals, when x is std::numerical_limits<T>::lowest(),
                    // the result is undefined behavior in C++. So, for the sake of performance,
                    // we will not do any special treatment for those cases.
                    // Note this keeps the integer promotion of -s, so types narrower than int
                    // return int, exactly like std::abs.
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
            FK_HOST_DEVICE_FUSE auto exec(const ST1 s1, const ST2 s2) {
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
            FK_HOST_DEVICE_FUSE auto exec(const ST s) {
                return static_cast<fk::VBase<OT>>(s);
            }
        };
        template <typename T>
        FK_HOST_DEVICE_FUSE auto f(const T val) {
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
                if (base::is_constant_evaluated()) {
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
                } else {
                    // Bit exact with the constexpr path below, and a single instruction sequence
                    // on device instead of the manual exponent surgery
                    return std::ldexp(x, exp);
                }
            }
        };
        CXP_F_FUNC
    };

    struct expf {
        struct BaseFunc {
            using InstanceType = fk::UnaryType;
            FK_HOST_DEVICE_FUSE auto exec(const float x) {
                if (base::is_constant_evaluated()) {
                    // 1. Handle edge cases FIRST to protect constexpr evaluation.
                    // The finite cutoffs sit safely outside the domain where the result is still
                    // representable, so they only fire where the answer is exactly inf or zero.
                    // Narrower bounds would misreport values near the overflow boundary, whose
                    // true result is still finite.
                    if (isnan::BaseFunc::exec(x))
                        return x;
                    if (x > 89.0f)
                        return cxp::bit_cast<float>(0x7F800000); // 0x7F800000 is +INFINITY
                    if (x < -104.0f)
                        return 0.0f; // Hard underflow, below half the smallest subnormal

                    // The whole reduction is carried out in double and rounded to float exactly once
                    // at the end. float has 24 mantissa bits against double's 53, so the ~29 bits of
                    // headroom make the single final rounding correct, which is what makes this path
                    // agree with std::exp on a float input.
                    const double xd = static_cast<double>(x);

                    // 2. Range Reduction: x = k * ln(2) + r, with |r| <= ln(2)/2
                    const double INV_LN2 = 1.4426950408889634;
                    const double kd = xd * INV_LN2;
                    // Round half away from zero without leaving constant evaluation
                    const int32_t k = static_cast<int32_t>(kd >= 0.0 ? kd + 0.5 : kd - 0.5);
                    const double k_d = static_cast<double>(k);

                    // ln(2) split so that k * LN2_HI is exact for every k we can reach here
                    const double LN2_HI = 6.93147180369123816490e-01;
                    const double LN2_LO = 1.90821492927058770002e-10;
                    const double r = (xd - k_d * LN2_HI) - k_d * LN2_LO;

                    // 3. Degree-13 Taylor series in double. The truncation error over |r| <= 0.3466
                    // is below 0.02 double ULP, far under half a float ULP.
                    const double poly =
                        1.0 +
                        r * (1.0 +
                             r * (1.0 / 2.0 +
                                  r * (1.0 / 6.0 +
                                       r * (1.0 / 24.0 +
                                            r * (1.0 / 120.0 +
                                                 r * (1.0 / 720.0 +
                                                      r * (1.0 / 5040.0 +
                                                           r * (1.0 / 40320.0 +
                                                                r * (1.0 / 362880.0 +
                                                                     r * (1.0 / 3628800.0 +
                                                                          r * (1.0 / 39916800.0 +
                                                                               r * (1.0 / 479001600.0 +
                                                                                    r * (1.0 /
                                                                                         6227020800.0)))))))))))));

                    // 4. Reconstruction: e^x = e^r * 2^k. k stays well inside the double exponent
                    // range, so this scaling is exact and cannot underflow before the final rounding.
                    const double twoK = cxp::bit_cast<double>(static_cast<ulonglong>(k + 1023) << 52);
                    const double result = poly * twoK;

                    // Anything at or above the float rounding midpoint towards 2^128 becomes infinity.
                    // Guarding here keeps the narrowing conversion below in range.
                    if (result >= 3.4028235677973366e38) {
                        return cxp::bit_cast<float>(0x7F800000);
                    }
                    // Single, correctly rounded narrowing. Underflow to (sub)normal or zero is
                    // well defined and also correctly rounded.
                    return static_cast<float>(result);
                } else {
                    // The constexpr body was verified to be bit
                    // identical to std::exp for every finite float in [-104, 89].
                    return std::exp(x);
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
        template <typename... Types>
        static constexpr inline auto f(const Types &...vals) {
            return Exec<BaseFunc>::exec(vals...);
        }
    };

#undef CXP_F_FUNC

} // namespace cxp

#endif
