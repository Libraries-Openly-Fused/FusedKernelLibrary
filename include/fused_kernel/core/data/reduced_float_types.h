/* Copyright 2026 Oscar Amoros Huguet, Johnny Nunez

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_REDUCED_FLOAT_TYPES
#define FK_REDUCED_FLOAT_TYPES

#include <fused_kernel/core/utils/utils.h>
#include <fused_kernel/core/utils/type_lists.h>

#if defined(__NVCC__) && !defined(NVRTC_COMPILER)
// Native types are used only as runtime fast paths and for interop; the fk types below are
// the element types of the library on every backend. cuda_fp8.h/cuda_fp4.h are deliberately
// NOT included: fp8/fp4 conversions are pure software on all paths (see PR notes), which keeps
// the API independent of the installed toolkit version.
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

#if !defined(NVRTC_COMPILER)
#include <limits>
#endif

namespace fk {

    namespace detail_rf {
        // Description of a reduced floating point format plus its conversion policy.
        // HAS_INF: the format encodes infinities (fp16, bf16, e5m2).
        // HAS_NAN: the format encodes NaN (all but e2m1).
        // SAT_FINITE: out of range values (including infinities) saturate to the maximum finite
        //             value instead of producing Inf, matching NVIDIA's __NV_SATFINITE fp8/fp4
        //             converters. NaN handling under SAT_FINITE is canonical (sign dropped).
        template <int EXP_BITS_, int MAN_BITS_, bool HAS_INF_, bool HAS_NAN_, bool SAT_FINITE_>
        struct MiniFloatFormat {
            static constexpr int EXP_BITS = EXP_BITS_;
            static constexpr int MAN_BITS = MAN_BITS_;
            static constexpr int BIAS = (1 << (EXP_BITS - 1)) - 1;
            static constexpr bool HAS_INF = HAS_INF_;
            static constexpr bool HAS_NAN = HAS_NAN_;
            static constexpr bool SAT_FINITE = SAT_FINITE_;
            static constexpr unsigned int SIGN_MASK = 1u << (EXP_BITS + MAN_BITS);
            static constexpr unsigned int EXP_MASK = ((1u << EXP_BITS) - 1u) << MAN_BITS;
            static constexpr unsigned int MAN_MASK = (1u << MAN_BITS) - 1u;
            // Maximum finite value (sign bit not included):
            // - Formats with Inf: largest exponent field is reserved, so exp = max-1, mantissa all ones.
            // - e4m3 (no Inf, NaN = exp and mantissa all ones): everything but the NaN code is finite.
            // - e2m1 (no Inf, no NaN): every code is finite.
            static constexpr unsigned int MAX_FINITE =
                HAS_INF ? ((((EXP_MASK >> MAN_BITS) - 1u) << MAN_BITS) | MAN_MASK)
                        : (HAS_NAN ? (EXP_MASK | (MAN_MASK - 1u)) : (EXP_MASK | MAN_MASK));
            // Canonical NaN produced when converting a NaN into this format. NVIDIA's fp8
            // converters produce 0x7F for both e4m3 and e5m2; fp16/bf16 use a quiet NaN.
            // For e2m1 (no NaN) NVIDIA converts NaN to +MAX_FINITE (+6.0).
            static constexpr unsigned int NAN_CODE =
                HAS_NAN ? (SAT_FINITE ? (EXP_MASK | MAN_MASK) : (EXP_MASK | (1u << (MAN_BITS - 1))))
                        : MAX_FINITE;
            static constexpr unsigned int INF_CODE = EXP_MASK;
            static constexpr unsigned int MIN_NORMAL = 1u << MAN_BITS;
        };

        using FmtF16 = MiniFloatFormat<5, 10, true, true, false>;
        using FmtBF16 = MiniFloatFormat<8, 7, true, true, false>;
        using FmtE4M3 = MiniFloatFormat<4, 3, false, true, true>;
        using FmtE5M2 = MiniFloatFormat<5, 2, true, true, true>;
        using FmtE2M1 = MiniFloatFormat<2, 1, false, false, true>;

        // Round a 64 bit significand to the lowest "shift" bits using round to nearest, ties to even.
        FK_HOST_DEVICE_CNST unsigned long long roundShiftRNE(const unsigned long long value, const int shift) {
            if (shift <= 0) {
                return value << (-shift);
            } else if (shift >= 64) {
                return 0ull;
            }
            const unsigned long long truncated = value >> shift;
            const unsigned long long guard = (value >> (shift - 1)) & 1ull;
            const unsigned long long stickyMask = (1ull << (shift - 1)) - 1ull;
            const bool sticky = (value & stickyMask) != 0ull;
            const bool roundUp = guard && (sticky || ((truncated & 1ull) != 0ull));
            return truncated + (roundUp ? 1ull : 0ull);
        }

        // Encode an IEEE 754 binary32/binary64 value into the target reduced format, with a
        // single rounding step (round to nearest even) and the format's overflow/NaN policy.
        template <typename Fmt, typename FloatType>
        FK_HOST_DEVICE_CNST unsigned int encode(const FloatType value) {
            static_assert(std::is_same_v<FloatType, float> || std::is_same_v<FloatType, double>,
                          "encode only accepts float or double sources");
            constexpr bool IS_F32 = std::is_same_v<FloatType, float>;
            using UIntT = std::conditional_t<IS_F32, unsigned int, unsigned long long>;
            constexpr int SRC_MAN_BITS = IS_F32 ? 23 : 52;
            constexpr int SRC_EXP_BITS = IS_F32 ? 8 : 11;
            constexpr int SRC_BIAS = IS_F32 ? 127 : 1023;
            constexpr UIntT SRC_MAN_MASK = (UIntT(1) << SRC_MAN_BITS) - 1;
            constexpr unsigned int SRC_EXP_MAX = (1u << SRC_EXP_BITS) - 1u;

            const UIntT srcBits = __builtin_bit_cast(UIntT, value);
            const unsigned int sign = static_cast<unsigned int>(srcBits >> (SRC_MAN_BITS + SRC_EXP_BITS)) & 1u;
            const unsigned int srcExp = static_cast<unsigned int>(srcBits >> SRC_MAN_BITS) & SRC_EXP_MAX;
            const unsigned long long srcMan = static_cast<unsigned long long>(srcBits & SRC_MAN_MASK);
            const unsigned int signBits = sign << (Fmt::EXP_BITS + Fmt::MAN_BITS);

            if (srcExp == SRC_EXP_MAX) {
                if (srcMan != 0ull) { // NaN
                    if constexpr (!Fmt::HAS_NAN) {
                        return Fmt::NAN_CODE; // e2m1: NaN converts to +MAX_FINITE, sign dropped
                    } else if constexpr (Fmt::SAT_FINITE) {
                        return Fmt::NAN_CODE; // fp8: canonical NaN, sign dropped
                    } else {
                        // fp16/bf16: quiet NaN preserving the truncated payload
                        const unsigned int payload =
                            static_cast<unsigned int>(srcMan >> (SRC_MAN_BITS - Fmt::MAN_BITS)) & Fmt::MAN_MASK;
                        return signBits | Fmt::EXP_MASK | (1u << (Fmt::MAN_BITS - 1)) | payload;
                    }
                } else { // Inf
                    if constexpr (Fmt::HAS_INF && !Fmt::SAT_FINITE) {
                        return signBits | Fmt::INF_CODE;
                    } else {
                        return signBits | Fmt::MAX_FINITE;
                    }
                }
            }

            // Finite input: build (unbiased exponent, 1.xxx significand) normalizing subnormal sources
            unsigned long long significand = srcMan;
            int exponent = static_cast<int>(srcExp) - SRC_BIAS;
            if (srcExp == 0u) {
                if (significand == 0ull) {
                    return signBits; // +/- 0
                }
                exponent = 1 - SRC_BIAS;
                while ((significand & (UIntT(1) << SRC_MAN_BITS)) == 0ull) {
                    significand <<= 1;
                    --exponent;
                }
            } else {
                significand |= (UIntT(1) << SRC_MAN_BITS);
            }

            const int targetBiasedExp = exponent + Fmt::BIAS;
            unsigned long long candidate = 0ull;
            if (targetBiasedExp >= 1) {
                // Normal range: round the significand to MAN_BITS fractional bits
                const unsigned long long rounded = roundShiftRNE(significand, SRC_MAN_BITS - Fmt::MAN_BITS);
                // rounded holds the implicit bit at position MAN_BITS (or MAN_BITS+1 after a carry);
                // adding (exp-1)<<MAN_BITS makes any mantissa carry bump the exponent for free.
                candidate = rounded + (static_cast<unsigned long long>(targetBiasedExp - 1) << Fmt::MAN_BITS);
            } else {
                // Subnormal range for the target: shift out extra bits before rounding.
                // A round up out of the subnormal range naturally produces the minimum normal.
                const int extraShift = 1 - targetBiasedExp;
                candidate = roundShiftRNE(significand, (SRC_MAN_BITS - Fmt::MAN_BITS) + extraShift);
            }
            if (candidate > static_cast<unsigned long long>(Fmt::MAX_FINITE)) {
                if constexpr (Fmt::HAS_INF && !Fmt::SAT_FINITE) {
                    return signBits | Fmt::INF_CODE;
                } else {
                    return signBits | Fmt::MAX_FINITE;
                }
            }
            return signBits | static_cast<unsigned int>(candidate);
        }

        // Decode a reduced format value into IEEE 754 binary32. Every finite value of every
        // supported format is exactly representable in binary32.
        template <typename Fmt>
        FK_HOST_DEVICE_CNST float decode(const unsigned int bits) {
            const unsigned int sign = (bits & Fmt::SIGN_MASK) ? 1u : 0u;
            const unsigned int expField = (bits & Fmt::EXP_MASK) >> Fmt::MAN_BITS;
            const unsigned int mantissa = bits & Fmt::MAN_MASK;
            const unsigned int maxExpField = (1u << Fmt::EXP_BITS) - 1u;
            const unsigned int signBit32 = sign << 31;

            if constexpr (Fmt::HAS_INF) {
                if (expField == maxExpField) {
                    if (mantissa == 0u) {
                        return __builtin_bit_cast(float, signBit32 | 0x7F800000u);
                    }
                    // Quiet NaN preserving the payload in the top mantissa bits
                    return __builtin_bit_cast(float, signBit32 | 0x7FC00000u | (mantissa << (23 - Fmt::MAN_BITS)));
                }
            } else if constexpr (Fmt::HAS_NAN) {
                if ((bits & (Fmt::EXP_MASK | Fmt::MAN_MASK)) == (Fmt::EXP_MASK | Fmt::MAN_MASK)) {
                    return __builtin_bit_cast(float, signBit32 | 0x7FC00000u);
                }
            }

            if (expField == 0u) {
                if (mantissa == 0u) {
                    return __builtin_bit_cast(float, signBit32); // +/- 0
                }
                // Subnormal: normalize. The result is usually a binary32 normal value, except
                // for the smallest bf16 subnormals (down to 2^-133), which land below the
                // binary32 normal range (2^-126) and must be emitted as binary32 subnormals.
                int exponent = 1 - Fmt::BIAS;
                unsigned int man = mantissa;
                while ((man & Fmt::MIN_NORMAL) == 0u) {
                    man <<= 1;
                    --exponent;
                }
                man &= Fmt::MAN_MASK;
                const int exp32 = exponent + 127;
                if (exp32 >= 1) {
                    return __builtin_bit_cast(float, signBit32 | (static_cast<unsigned int>(exp32) << 23)
                                                              | (man << (23 - Fmt::MAN_BITS)));
                }
                // Binary32 subnormal: shift the full significand (implicit bit included) into
                // place. The shift never discards set bits: the smallest representable input
                // (bf16 2^-133) still has 16 trailing zero bits available.
                const unsigned int sig32 = (1u << 23) | (man << (23 - Fmt::MAN_BITS));
                return __builtin_bit_cast(float, signBit32 | (sig32 >> (1 - exp32)));
            }
            const int exponent = static_cast<int>(expField) - Fmt::BIAS;
            return __builtin_bit_cast(float, signBit32 | (static_cast<unsigned int>(exponent + 127) << 23)
                                                      | (mantissa << (23 - Fmt::MAN_BITS)));
        }
    } // namespace detail_rf

    // Reduced precision floating point types, usable as element types in any FKL pipeline on
    // every backend. They are the same fk-owned types under nvcc, g++, clang and MSVC (identical
    // layout, traits and semantics), fully constexpr through a software conversion core, with
    // native CUDA fast paths taken automatically at runtime under nvcc (fp16/bf16).
    // Layouts are bit-compatible with the CUDA types (__half, __nv_bfloat16, __nv_fp8_e4m3,
    // __nv_fp8_e5m2, __nv_fp4_e2m1), so device buffers can be reinterpreted freely.
// VALUE_MASK canonicalizes the stored bits on construction from raw bits: for fp4 the upper
// nibble is unspecified storage (NVIDIA converters write 0 and ignore it on read), and keeping
// it would make numerically equal values compare bit-different through toBits().
#define FK_RF_COMMON(TypeName, StorageT, Fmt, VALUE_MASK)                                              \
    private:                                                                                           \
        StorageT bits_;                                                                                \
        struct FromBitsTag {};                                                                         \
        FK_HOST_DEVICE_CNST TypeName(const StorageT bits, const FromBitsTag&)                          \
            : bits_(static_cast<StorageT>(bits & VALUE_MASK)) {}                                       \
    public:                                                                                            \
        using Format = Fmt;                                                                            \
        using StorageType = StorageT;                                                                  \
        TypeName() = default;                                                                          \
        FK_HOST_DEVICE_STATIC constexpr TypeName fromBits(const StorageT bits) {                       \
            return TypeName(bits, FromBitsTag{});                                                      \
        }                                                                                              \
        FK_HOST_DEVICE_CNST StorageT bits() const { return bits_; }

    struct Fp16 {
        FK_RF_COMMON(Fp16, unsigned short, detail_rf::FmtF16, 0xFFFFu)
        FK_HOST_DEVICE_CNST explicit Fp16(const float value) {
#if defined(__NVCC__) && !defined(NVRTC_COMPILER)
            if (!__builtin_is_constant_evaluated()) {
                bits_ = __builtin_bit_cast(unsigned short, __half(value));
                return;
            }
#endif
            bits_ = static_cast<unsigned short>(detail_rf::encode<Format>(value));
        }
        FK_HOST_DEVICE_CNST explicit Fp16(const double value)
            : bits_(static_cast<unsigned short>(detail_rf::encode<Format>(value))) {}
        template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
        FK_HOST_DEVICE_CNST explicit Fp16(const I value) : Fp16(static_cast<double>(value)) {}
        FK_HOST_DEVICE_CNST operator float() const {
#if defined(__NVCC__) && !defined(NVRTC_COMPILER)
            if (!__builtin_is_constant_evaluated()) {
                return static_cast<float>(__builtin_bit_cast(__half, bits_));
            }
#endif
            return detail_rf::decode<Format>(bits_);
        }
    };

    struct Bf16 {
        FK_RF_COMMON(Bf16, unsigned short, detail_rf::FmtBF16, 0xFFFFu)
        FK_HOST_DEVICE_CNST explicit Bf16(const float value) {
#if defined(__NVCC__) && !defined(NVRTC_COMPILER)
            if (!__builtin_is_constant_evaluated()) {
                bits_ = __builtin_bit_cast(unsigned short, __nv_bfloat16(value));
                return;
            }
#endif
            bits_ = static_cast<unsigned short>(detail_rf::encode<Format>(value));
        }
        FK_HOST_DEVICE_CNST explicit Bf16(const double value)
            : bits_(static_cast<unsigned short>(detail_rf::encode<Format>(value))) {}
        template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
        FK_HOST_DEVICE_CNST explicit Bf16(const I value) : Bf16(static_cast<double>(value)) {}
        FK_HOST_DEVICE_CNST operator float() const {
#if defined(__NVCC__) && !defined(NVRTC_COMPILER)
            if (!__builtin_is_constant_evaluated()) {
                return static_cast<float>(__builtin_bit_cast(__nv_bfloat16, bits_));
            }
#endif
            return detail_rf::decode<Format>(bits_);
        }
    };

    struct Fp8E4m3 {
        FK_RF_COMMON(Fp8E4m3, unsigned char, detail_rf::FmtE4M3, 0xFFu)
        FK_HOST_DEVICE_CNST explicit Fp8E4m3(const float value)
            : bits_(static_cast<unsigned char>(detail_rf::encode<Format>(value))) {}
        FK_HOST_DEVICE_CNST explicit Fp8E4m3(const double value)
            : bits_(static_cast<unsigned char>(detail_rf::encode<Format>(value))) {}
        FK_HOST_DEVICE_CNST explicit Fp8E4m3(const Fp16 value) : Fp8E4m3(static_cast<float>(value)) {}
        FK_HOST_DEVICE_CNST explicit Fp8E4m3(const Bf16 value) : Fp8E4m3(static_cast<float>(value)) {}
        template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
        FK_HOST_DEVICE_CNST explicit Fp8E4m3(const I value) : Fp8E4m3(static_cast<double>(value)) {}
        FK_HOST_DEVICE_CNST explicit operator float() const { return detail_rf::decode<Format>(bits_); }
    };

    struct Fp8E5m2 {
        FK_RF_COMMON(Fp8E5m2, unsigned char, detail_rf::FmtE5M2, 0xFFu)
        FK_HOST_DEVICE_CNST explicit Fp8E5m2(const float value)
            : bits_(static_cast<unsigned char>(detail_rf::encode<Format>(value))) {}
        FK_HOST_DEVICE_CNST explicit Fp8E5m2(const double value)
            : bits_(static_cast<unsigned char>(detail_rf::encode<Format>(value))) {}
        FK_HOST_DEVICE_CNST explicit Fp8E5m2(const Fp16 value) : Fp8E5m2(static_cast<float>(value)) {}
        FK_HOST_DEVICE_CNST explicit Fp8E5m2(const Bf16 value) : Fp8E5m2(static_cast<float>(value)) {}
        template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
        FK_HOST_DEVICE_CNST explicit Fp8E5m2(const I value) : Fp8E5m2(static_cast<double>(value)) {}
        FK_HOST_DEVICE_CNST explicit operator float() const { return detail_rf::decode<Format>(bits_); }
    };

    // One element per byte, value in the low nibble, matching CUDA's scalar __nv_fp4_e2m1.
    // The truly packed two-per-byte format (__nv_fp4x2_e2m1) cannot be an element type in the
    // RawPtr model (one addressable element per T); use a fused dequantizing Read for that.
    struct Fp4E2m1 {
        FK_RF_COMMON(Fp4E2m1, unsigned char, detail_rf::FmtE2M1, 0x0Fu)
        FK_HOST_DEVICE_CNST explicit Fp4E2m1(const float value)
            : bits_(static_cast<unsigned char>(detail_rf::encode<Format>(value))) {}
        FK_HOST_DEVICE_CNST explicit Fp4E2m1(const double value)
            : bits_(static_cast<unsigned char>(detail_rf::encode<Format>(value))) {}
        FK_HOST_DEVICE_CNST explicit Fp4E2m1(const Fp16 value) : Fp4E2m1(static_cast<float>(value)) {}
        FK_HOST_DEVICE_CNST explicit Fp4E2m1(const Bf16 value) : Fp4E2m1(static_cast<float>(value)) {}
        template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
        FK_HOST_DEVICE_CNST explicit Fp4E2m1(const I value) : Fp4E2m1(static_cast<double>(value)) {}
        // bits_ is canonical (upper nibble masked on construction), so no masking is needed here.
        FK_HOST_DEVICE_CNST explicit operator float() const { return detail_rf::decode<Format>(bits_); }
    };

#undef FK_RF_COMMON

    using fp16 = Fp16;
    using bf16 = Bf16;
    using fp8_e4m3 = Fp8E4m3;
    using fp8_e5m2 = Fp8E5m2;
    using fp4_e2m1 = Fp4E2m1;

    static_assert(sizeof(fp16) == 2 && alignof(fp16) == 2, "fp16 layout must match __half");
    static_assert(sizeof(bf16) == 2 && alignof(bf16) == 2, "bf16 layout must match __nv_bfloat16");
    static_assert(sizeof(fp8_e4m3) == 1 && alignof(fp8_e4m3) == 1, "fp8_e4m3 layout must match __nv_fp8_e4m3");
    static_assert(sizeof(fp8_e5m2) == 1 && alignof(fp8_e5m2) == 1, "fp8_e5m2 layout must match __nv_fp8_e5m2");
    static_assert(sizeof(fp4_e2m1) == 1 && alignof(fp4_e2m1) == 1, "fp4_e2m1 layout must match __nv_fp4_e2m1");
    static_assert(std::is_trivially_copyable_v<fp16> && std::is_trivially_copyable_v<bf16> &&
                  std::is_trivially_copyable_v<fp8_e4m3> && std::is_trivially_copyable_v<fp8_e5m2> &&
                  std::is_trivially_copyable_v<fp4_e2m1>, "Reduced float types must be trivially copyable");

    // Type classification. validScalar/validFloatingPoint are the library-wide replacements for
    // std::is_fundamental_v / std::is_floating_point_v wherever reduced floats must be admitted.
    using ReducedFloatTypes = TypeList<fp16, bf16, fp8_e4m3, fp8_e5m2, fp4_e2m1>;
    template <typename T>
    constexpr bool isReducedFloat = one_of_v<T, ReducedFloatTypes>;
    // Types with a full arithmetic and comparison operator surface (fp8/fp4 are conversion-only,
    // exactly like the CUDA types, which define no operators for them).
    template <typename T>
    constexpr bool isArithmeticReducedFloat = one_of_v<T, TypeList<fp16, bf16>>;
    template <typename T>
    constexpr bool validScalar = std::is_fundamental_v<T> || isReducedFloat<T>;
    template <typename T>
    constexpr bool validFloatingPoint = std::is_floating_point_v<T> || isReducedFloat<T>;

    template <typename T>
    struct ReducedFloatTraits {
        using Format = typename T::Format;
        using StorageType = typename T::StorageType;
        static constexpr StorageType maxBits = static_cast<StorageType>(Format::MAX_FINITE);
        static constexpr StorageType lowestBits = static_cast<StorageType>(Format::SIGN_MASK | Format::MAX_FINITE);
        static constexpr StorageType minNormalBits = static_cast<StorageType>(Format::MIN_NORMAL);
        static constexpr StorageType minSubnormalBits = static_cast<StorageType>(1u);
        static constexpr StorageType quietNaNBits = static_cast<StorageType>(Format::NAN_CODE);
        static constexpr StorageType infBits = static_cast<StorageType>(Format::INF_CODE);
        static constexpr bool hasInf = Format::HAS_INF;
        static constexpr bool hasNaN = Format::HAS_NAN;
        FK_HOST_DEVICE_FUSE bool isNaN(const T& value) {
            if constexpr (Format::HAS_INF) {
                return (value.bits() & Format::EXP_MASK) == Format::EXP_MASK &&
                       (value.bits() & Format::MAN_MASK) != 0u;
            } else if constexpr (Format::HAS_NAN) {
                return (value.bits() & (Format::EXP_MASK | Format::MAN_MASK)) ==
                       (Format::EXP_MASK | Format::MAN_MASK);
            } else {
                return false;
            }
        }
        FK_HOST_DEVICE_FUSE bool isInf(const T& value) {
            if constexpr (Format::HAS_INF) {
                return (value.bits() & (Format::EXP_MASK | Format::MAN_MASK)) == Format::EXP_MASK;
            } else {
                return false;
            }
        }
        FK_HOST_DEVICE_FUSE T abs(const T& value) {
            return T::fromBits(static_cast<StorageType>(value.bits() & ~Format::SIGN_MASK));
        }
    };

    template <typename T, typename = std::enable_if_t<isReducedFloat<T>>>
    FK_HOST_DEVICE_CNST auto toBits(const T& value) {
        return value.bits();
    }

#if defined(__NVCC__) && !defined(NVRTC_COMPILER)
    // Zero cost interop with the CUDA native types (bit-identical layouts).
    // NOTE: mixed expressions between fk::fp16 and __half (or fk::bf16 and __nv_bfloat16) are
    // ambiguous by design - both types convert implicitly to several built in arithmetic types.
    // Convert one operand explicitly with toNative()/fromNative() (both are free, bit casts).
    FK_HOST_DEVICE_CNST __half toNative(const fp16& value) {
        return __builtin_bit_cast(__half, value.bits());
    }
    FK_HOST_DEVICE_CNST __nv_bfloat16 toNative(const bf16& value) {
        return __builtin_bit_cast(__nv_bfloat16, value.bits());
    }
    FK_HOST_DEVICE_CNST fp16 fromNative(const __half& value) {
        return fp16::fromBits(__builtin_bit_cast(unsigned short, value));
    }
    FK_HOST_DEVICE_CNST bf16 fromNative(const __nv_bfloat16& value) {
        return bf16::fromBits(__builtin_bit_cast(unsigned short, value));
    }
#endif

} // namespace fk

// Arithmetic and comparisons for fp16/bf16: promote to float, compute, demote (RN).
// binary32 has more than 2p+2 mantissa bits for both formats, so the results are the
// correctly rounded reduced precision results (identical to CUDA's host/device operators).
// Declared at global scope like the vector operators in vector_utils.h: declaring them inside
// namespace fk would hide every global operator from unqualified lookup within fk.
#define FK_RF_ARITHMETIC(TypeName)                                                                      \
    FK_HOST_DEVICE_CNST TypeName operator+(const TypeName& a, const TypeName& b) {                      \
        return TypeName(static_cast<float>(a) + static_cast<float>(b));                                 \
    }                                                                                                   \
    FK_HOST_DEVICE_CNST TypeName operator-(const TypeName& a, const TypeName& b) {                      \
        return TypeName(static_cast<float>(a) - static_cast<float>(b));                                 \
    }                                                                                                   \
    FK_HOST_DEVICE_CNST TypeName operator*(const TypeName& a, const TypeName& b) {                      \
        return TypeName(static_cast<float>(a) * static_cast<float>(b));                                 \
    }                                                                                                   \
    FK_HOST_DEVICE_CNST TypeName operator/(const TypeName& a, const TypeName& b) {                      \
        return TypeName(static_cast<float>(a) / static_cast<float>(b));                                 \
    }                                                                                                   \
    FK_HOST_DEVICE_CNST TypeName operator-(const TypeName& a) {                                         \
        return TypeName::fromBits(static_cast<typename TypeName::StorageType>(                          \
            a.bits() ^ TypeName::Format::SIGN_MASK));                                                   \
    }                                                                                                   \
    FK_HOST_DEVICE_CNST TypeName& operator+=(TypeName& a, const TypeName& b) { a = a + b; return a; }   \
    FK_HOST_DEVICE_CNST TypeName& operator-=(TypeName& a, const TypeName& b) { a = a - b; return a; }   \
    FK_HOST_DEVICE_CNST TypeName& operator*=(TypeName& a, const TypeName& b) { a = a * b; return a; }   \
    FK_HOST_DEVICE_CNST TypeName& operator/=(TypeName& a, const TypeName& b) { a = a / b; return a; }   \
    FK_HOST_DEVICE_CNST bool operator==(const TypeName& a, const TypeName& b) {                         \
        return static_cast<float>(a) == static_cast<float>(b);                                          \
    }                                                                                                   \
    FK_HOST_DEVICE_CNST bool operator!=(const TypeName& a, const TypeName& b) {                         \
        return static_cast<float>(a) != static_cast<float>(b);                                          \
    }                                                                                                   \
    FK_HOST_DEVICE_CNST bool operator<(const TypeName& a, const TypeName& b) {                          \
        return static_cast<float>(a) < static_cast<float>(b);                                           \
    }                                                                                                   \
    FK_HOST_DEVICE_CNST bool operator>(const TypeName& a, const TypeName& b) {                          \
        return static_cast<float>(a) > static_cast<float>(b);                                           \
    }                                                                                                   \
    FK_HOST_DEVICE_CNST bool operator<=(const TypeName& a, const TypeName& b) {                         \
        return static_cast<float>(a) <= static_cast<float>(b);                                          \
    }                                                                                                   \
    FK_HOST_DEVICE_CNST bool operator>=(const TypeName& a, const TypeName& b) {                         \
        return static_cast<float>(a) >= static_cast<float>(b);                                          \
    }

FK_RF_ARITHMETIC(fk::Fp16)
FK_RF_ARITHMETIC(fk::Bf16)
#undef FK_RF_ARITHMETIC

#if !defined(NVRTC_COMPILER)
// std::numeric_limits for the reduced float types: makes fk::maxValue/minValue (vlimits.h) and
// cxp::minValue/maxValue work unmodified, and is the canonical source for the limit values.
#define FK_RF_NUMERIC_LIMITS(TypeName, DIGITS_, DIGITS10_, MAX_DIGITS10_, MAX_EXP_, MAX_EXP10_,         \
                             MIN_EXP_, MIN_EXP10_)                                                      \
    template <>                                                                                         \
    class std::numeric_limits<TypeName> {                                                               \
    public:                                                                                             \
        using Traits = fk::ReducedFloatTraits<TypeName>;                                                \
        static constexpr bool is_specialized = true;                                                    \
        static constexpr bool is_signed = true;                                                         \
        static constexpr bool is_integer = false;                                                       \
        static constexpr bool is_exact = false;                                                         \
        static constexpr bool has_infinity = Traits::hasInf;                                            \
        static constexpr bool has_quiet_NaN = Traits::hasNaN;                                           \
        static constexpr bool has_signaling_NaN = false;                                                \
        static constexpr std::float_denorm_style has_denorm = std::denorm_present;                      \
        static constexpr bool has_denorm_loss = false;                                                  \
        static constexpr std::float_round_style round_style = std::round_to_nearest;                    \
        static constexpr bool is_iec559 = false;                                                        \
        static constexpr bool is_bounded = true;                                                        \
        static constexpr bool is_modulo = false;                                                        \
        static constexpr bool traps = false;                                                            \
        static constexpr bool tinyness_before = false;                                                  \
        static constexpr int digits = DIGITS_;                                                          \
        static constexpr int digits10 = DIGITS10_;                                                      \
        static constexpr int max_digits10 = MAX_DIGITS10_;                                              \
        static constexpr int max_exponent = MAX_EXP_;                                                   \
        static constexpr int max_exponent10 = MAX_EXP10_;                                               \
        static constexpr int min_exponent = MIN_EXP_;                                                   \
        static constexpr int min_exponent10 = MIN_EXP10_;                                               \
        static constexpr int radix = 2;                                                                 \
        static constexpr TypeName max() noexcept { return TypeName::fromBits(Traits::maxBits); }        \
        static constexpr TypeName lowest() noexcept { return TypeName::fromBits(Traits::lowestBits); }  \
        static constexpr TypeName min() noexcept { return TypeName::fromBits(Traits::minNormalBits); }  \
        static constexpr TypeName denorm_min() noexcept {                                               \
            return TypeName::fromBits(Traits::minSubnormalBits);                                        \
        }                                                                                               \
        static constexpr TypeName epsilon() noexcept {                                                  \
            return TypeName(1.0f / static_cast<float>(1 << (DIGITS_ - 1)));                             \
        }                                                                                               \
        static constexpr TypeName round_error() noexcept { return TypeName(0.5f); }                     \
        static constexpr TypeName infinity() noexcept { return TypeName::fromBits(Traits::infBits); }   \
        static constexpr TypeName quiet_NaN() noexcept {                                                \
            return TypeName::fromBits(Traits::quietNaNBits);                                            \
        }                                                                                               \
        static constexpr TypeName signaling_NaN() noexcept {                                            \
            return TypeName::fromBits(Traits::quietNaNBits);                                            \
        }                                                                                               \
    };

FK_RF_NUMERIC_LIMITS(fk::Fp16, 11, 3, 5, 16, 4, -13, -4)
FK_RF_NUMERIC_LIMITS(fk::Bf16, 8, 2, 4, 128, 38, -125, -37)
FK_RF_NUMERIC_LIMITS(fk::Fp8E4m3, 4, 0, 3, 9, 2, -5, -1)
FK_RF_NUMERIC_LIMITS(fk::Fp8E5m2, 3, 0, 2, 16, 4, -13, -4)
FK_RF_NUMERIC_LIMITS(fk::Fp4E2m1, 2, 0, 2, 3, 0, 1, 0)
#undef FK_RF_NUMERIC_LIMITS
#endif // !NVRTC_COMPILER

#endif // FK_REDUCED_FLOAT_TYPES
