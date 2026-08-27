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

#include <tests/main.h>

#include <fused_kernel/core/data/reduced_float_types.h>
#include <fused_kernel/core/utils/vector_utils.h>
#include <fused_kernel/core/utils/vlimits.h>
#include <fused_kernel/core/constexpr_libs/constexpr_cmath.h>
#include <fused_kernel/core/constexpr_libs/constexpr_saturate.h>

#include <cmath>
#include <iostream>

// The CPU backend must never see a CUDA header: cuda_fp16.h compiles silently under plain
// g++ when the toolkit is installed, so this tripwire is the only reliable guard.
#if !defined(__NVCC__) && defined(__CUDA_FP16_TYPES_EXIST__)
#error "cuda_fp16.h leaked into a CPU-only translation unit"
#endif

using namespace fk;

// ---- Layout: bit compatible with the CUDA types ----
static_assert(sizeof(fp16) == 2 && alignof(fp16) == 2);
static_assert(sizeof(bf16) == 2 && alignof(bf16) == 2);
static_assert(sizeof(fp8_e4m3) == 1 && alignof(fp8_e4m3) == 1);
static_assert(sizeof(fp8_e5m2) == 1 && alignof(fp8_e5m2) == 1);
static_assert(sizeof(fp4_e2m1) == 1 && alignof(fp4_e2m1) == 1);
static_assert(sizeof(fp16_2) == 4 && alignof(fp16_2) == 4);   // __half2 is alignas(4)
static_assert(sizeof(bf16_2) == 4 && alignof(bf16_2) == 4);   // __nv_bfloat162 is alignas(4)
static_assert(sizeof(fp16_3) == 6 && sizeof(fp16_4) == 8 && alignof(fp16_4) == 8);
static_assert(sizeof(fp8_e4m3_2) == 2 && alignof(fp8_e4m3_2) == 2); // __nv_fp8x2_e4m3
static_assert(sizeof(fp8_e4m3_4) == 4 && alignof(fp8_e4m3_4) == 4); // __nv_fp8x4_e4m3

// ---- Trait registration ----
static_assert(validScalar<fp16> && validScalar<bf16> && validScalar<fp8_e4m3> &&
              validScalar<fp8_e5m2> && validScalar<fp4_e2m1>);
static_assert(validFloatingPoint<fp16> && !validFloatingPoint<int>);
static_assert(isArithmeticReducedFloat<fp16> && isArithmeticReducedFloat<bf16> &&
              !isArithmeticReducedFloat<fp8_e4m3>);
static_assert(validCUDAVec<fp16_2> && validCUDAVec<bf16_4> && validCUDAVec<fp4_e2m1_3>);
static_assert(cn<fp16> == 1 && cn<fp16_2> == 2 && cn<fp8_e4m3_3> == 3 && cn<bf16_4> == 4);
static_assert(std::is_same_v<VBase<fp16_3>, fp16> && std::is_same_v<VBase<fp8_e5m2_2>, fp8_e5m2>);
static_assert(std::is_same_v<VectorType_t<fp16, 2>, fp16_2> &&
              std::is_same_v<VectorType_t<fp4_e2m1, 4>, fp4_e2m1_4>);

// ---- Constexpr conversions (the software path is always taken at constant evaluation) ----
static_assert(fp16(1.0f).bits() == 0x3C00 && fp16(-2.0f).bits() == 0xC000);
static_assert(fp16(65504.0f).bits() == 0x7BFF && fp16(1e20f).bits() == 0x7C00);
static_assert(static_cast<float>(fp16::fromBits(0x0001)) == 5.960464477539063e-08f); // min subnormal
static_assert(bf16(1.0f).bits() == 0x3F80);
static_assert(fp8_e4m3(448.0f).bits() == 0x7E && fp8_e4m3(1000.0f).bits() == 0x7E);  // SATFINITE
static_assert(fp8_e5m2(1e6f).bits() == 0x7B);                                        // SATFINITE 57344
static_assert(fp4_e2m1(1.5f).bits() == 0x03 && fp4_e2m1(-0.5f).bits() == 0x09);
static_assert(fp4_e2m1(1000.0f).bits() == 0x07);                                     // saturates to 6.0
static_assert(static_cast<float>(fp4_e2m1::fromBits(0xF2)) == 1.0f);                  // upper nibble ignored
// Round to nearest even ties
static_assert(fp4_e2m1(5.0f).bits() == 0x06);   // tie between 4 (even mantissa) and 6 -> 4
static_assert(fp4_e2m1(0.25f).bits() == 0x00);  // tie between 0 and 0.5 -> 0
static_assert(fp16(65505.0f).bits() == 0x7BFF); // rounds down to max, not Inf
// Single rounding from double: 2049.0000001 is above the 2048/2050 tie, so it must round to
// 2050 (0x6801). Converting through float first would round to 2049.0f and then to 2048 (tie
// to even) - a double rounding error this assert would catch.
static_assert(fp16(2049.0000001).bits() == 0x6801);
// NaN handling
static_assert(fp8_e4m3(std::numeric_limits<float>::quiet_NaN()).bits() == 0x7F);
static_assert(fp4_e2m1(std::numeric_limits<float>::quiet_NaN()).bits() == 0x07); // e2m1 has no NaN

// ---- Constexpr arithmetic and comparisons (fp16/bf16) ----
static_assert((fp16(2.0f) * fp16(3.0f)).bits() == fp16(6.0f).bits());
static_assert((bf16(2.0f) + bf16(3.0f)).bits() == bf16(5.0f).bits());
static_assert((-fp16(1.5f)).bits() == fp16(-1.5f).bits());
static_assert(fp16(1.0f) < fp16(2.0f) && bf16(-1.0f) <= bf16(1.0f) && fp16(3.0f) == fp16(3.0f));

// ---- Limits: fk::maxValue/minValue and cxp limits work through std::numeric_limits ----
static_assert(toBits(maxValue<fp16>) == 0x7BFF && toBits(minValue<fp16>) == 0xFBFF);
static_assert(static_cast<float>(maxValue<fp8_e4m3>) == 448.0f);
static_assert(static_cast<float>(maxValue<fp8_e5m2>) == 57344.0f);
static_assert(static_cast<float>(maxValue<fp4_e2m1>) == 6.0f);
static_assert(static_cast<float>(maxValue<bf16>) == 3.3895313892515355e+38f);
static_assert(toBits(maxValue<fp16_2>.x) == 0x7BFF && toBits(maxValue<fp16_2>.y) == 0x7BFF);
static_assert(static_cast<float>(cxp::maxValue<fp16>) == 65504.0f);

// ---- cxp classification and saturate_cast ----
static_assert(cxp::isnan::f(fp16::fromBits(0x7E00)) && !cxp::isnan::f(fp16(1.0f)));
static_assert(cxp::isinf::f(fp16::fromBits(0x7C00)) && !cxp::isinf::f(fp16(1.0f)));
static_assert(cxp::isnan::f(fp8_e4m3::fromBits(0x7F)) && !cxp::isnan::f(fp8_e4m3::fromBits(0x7E)));
static_assert(!cxp::isinf::f(fp4_e2m1::fromBits(0x07)));
static_assert(toBits(cxp::abs::f(fp16(-3.5f))) == fp16(3.5f).bits());
static_assert(toBits(cxp::abs::f(fp8_e5m2(-2.0f))) == fp8_e5m2(2.0f).bits());
static_assert(toBits(cxp::saturate_cast<fp16>::f(1e20f)) == 0x7BFF);        // clamps, no Inf
static_assert(cxp::saturate_cast<int>::f(fp16(3.7f)) == 4);                 // rounds via float
static_assert(toBits(cxp::saturate_cast<fp8_e4m3>::f(fp16(1000.0f))) == 0x7E);
static_assert(cxp::saturate_cast<uchar>::f(fp16(300.0f)) == 255);
static_assert(toBits(cxp::max::f(fp8_e4m3(2.0f), fp8_e4m3(-3.0f))) == fp8_e4m3(2.0f).bits());

int launch() {
    int failures = 0;

    // Exhaustive roundtrip on the software path: decode every code, encode it back.
    for (unsigned int code = 0; code < 0x10000u; ++code) {
        const auto h = fp16::fromBits(static_cast<unsigned short>(code));
        if (!ReducedFloatTraits<fp16>::isNaN(h)) {
            const float f = static_cast<float>(h);
            if (toBits(fp16(f)) != code) {
                std::cout << "fp16 roundtrip failed for code " << code << std::endl;
                ++failures;
            }
        }
        const auto b = bf16::fromBits(static_cast<unsigned short>(code));
        if (!ReducedFloatTraits<bf16>::isNaN(b)) {
            const float f = static_cast<float>(b);
            if (toBits(bf16(f)) != code) {
                std::cout << "bf16 roundtrip failed for code " << code << std::endl;
                ++failures;
            }
        }
    }
    for (unsigned int code = 0; code < 256u; ++code) {
        const auto e43 = fp8_e4m3::fromBits(static_cast<unsigned char>(code));
        if (!ReducedFloatTraits<fp8_e4m3>::isNaN(e43)) {
            if (toBits(fp8_e4m3(static_cast<float>(e43))) != code) {
                std::cout << "fp8_e4m3 roundtrip failed for code " << code << std::endl;
                ++failures;
            }
        }
        const auto e52 = fp8_e5m2::fromBits(static_cast<unsigned char>(code));
        if (!ReducedFloatTraits<fp8_e5m2>::isNaN(e52) && !ReducedFloatTraits<fp8_e5m2>::isInf(e52)) {
            if (toBits(fp8_e5m2(static_cast<float>(e52))) != code) {
                std::cout << "fp8_e5m2 roundtrip failed for code " << code << std::endl;
                ++failures;
            }
        }
    }
    for (unsigned int code = 0; code < 16u; ++code) {
        const auto e21 = fp4_e2m1::fromBits(static_cast<unsigned char>(code));
        // -0.0 encodes back to +0.0? No: sign is preserved for zero, so identity must hold.
        if (toBits(fp4_e2m1(static_cast<float>(e21))) != code) {
            std::cout << "fp4_e2m1 roundtrip failed for code " << code << std::endl;
            ++failures;
        }
    }

    // Arithmetic double oracle: the float promote-compute-demote path must produce the
    // correctly rounded reduced result (binary32 has > 2p+2 mantissa bits for both formats).
    unsigned int lcg = 42u;
    for (int i = 0; i < 200000; ++i) {
        lcg = lcg * 1664525u + 1013904223u;
        const auto a16 = fp16::fromBits(static_cast<unsigned short>(lcg & 0xFFFFu));
        const auto b16 = fp16::fromBits(static_cast<unsigned short>((lcg >> 16) & 0xFFFFu));
        if (!ReducedFloatTraits<fp16>::isNaN(a16) && !ReducedFloatTraits<fp16>::isNaN(b16)) {
            const auto sum = a16 + b16;
            const auto oracle = fp16(static_cast<double>(static_cast<float>(a16)) +
                                     static_cast<double>(static_cast<float>(b16)));
            if (toBits(sum) != toBits(oracle) &&
                !(ReducedFloatTraits<fp16>::isNaN(sum) && ReducedFloatTraits<fp16>::isNaN(oracle))) {
                std::cout << "fp16 add oracle mismatch: " << toBits(a16) << " + " << toBits(b16) << std::endl;
                ++failures;
            }
            const auto prod = a16 * b16;
            const auto prodOracle = fp16(static_cast<double>(static_cast<float>(a16)) *
                                         static_cast<double>(static_cast<float>(b16)));
            if (toBits(prod) != toBits(prodOracle) &&
                !(ReducedFloatTraits<fp16>::isNaN(prod) && ReducedFloatTraits<fp16>::isNaN(prodOracle))) {
                std::cout << "fp16 mul oracle mismatch: " << toBits(a16) << " * " << toBits(b16) << std::endl;
                ++failures;
            }
        }
        const auto a8 = bf16::fromBits(static_cast<unsigned short>(lcg & 0xFFFFu));
        const auto b8 = bf16::fromBits(static_cast<unsigned short>((lcg >> 16) & 0xFFFFu));
        if (!ReducedFloatTraits<bf16>::isNaN(a8) && !ReducedFloatTraits<bf16>::isNaN(b8)) {
            const auto sum = a8 + b8;
            const auto oracle = bf16(static_cast<double>(static_cast<float>(a8)) +
                                     static_cast<double>(static_cast<float>(b8)));
            if (toBits(sum) != toBits(oracle) &&
                !(ReducedFloatTraits<bf16>::isNaN(sum) && ReducedFloatTraits<bf16>::isNaN(oracle))) {
                std::cout << "bf16 add oracle mismatch: " << toBits(a8) << " + " << toBits(b8) << std::endl;
                ++failures;
            }
        }
    }

    // Vector operators over reduced vectors
    {
        const fp16_2 a{ fp16(1.0f), fp16(2.0f) };
        const fp16_2 b{ fp16(0.5f), fp16(0.25f) };
        const auto c = a * b;
        static_assert(std::is_same_v<std::decay_t<decltype(c)>, fp16_2>);
        if (toBits(c.x) != fp16(0.5f).bits() || toBits(c.y) != fp16(0.5f).bits()) {
            std::cout << "fp16_2 vector multiply failed" << std::endl;
            ++failures;
        }
        const auto d = a * fp16(2.0f); // vector x scalar
        if (toBits(d.x) != fp16(2.0f).bits() || toBits(d.y) != fp16(4.0f).bits()) {
            std::cout << "fp16_2 vector x scalar multiply failed" << std::endl;
            ++failures;
        }
        const auto eq = (a == a);
        if (!Bool::vAnd(eq)) {
            std::cout << "fp16_2 vector comparison failed" << std::endl;
            ++failures;
        }
        const auto ms = make_set<bf16_4>(bf16(1.5f));
        if (toBits(ms.x) != bf16(1.5f).bits() || toBits(ms.w) != bf16(1.5f).bits()) {
            std::cout << "make_set<bf16_4> failed" << std::endl;
            ++failures;
        }
    }

    if (failures == 0) {
        std::cout << "utest_reduced_float_types: all checks passed" << std::endl;
    }
    return failures == 0 ? 0 : -1;
}
