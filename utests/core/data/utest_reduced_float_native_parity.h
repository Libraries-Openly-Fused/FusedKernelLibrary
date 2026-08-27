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

// Verifies bit exact parity between the fk software conversions and the NVIDIA native
// converters, for every representable code and a large sample of float inputs.
#define __ONLY_CU__

#include <tests/main.h>

#include <fused_kernel/core/data/reduced_float_types.h>
#include <fused_kernel/core/data/vector_types.h>

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#if __has_include(<cuda_fp8.h>)
#include <cuda_fp8.h>
#define UTEST_HAS_FP8 1
#endif
#if __has_include(<cuda_fp4.h>)
#include <cuda_fp4.h>
#define UTEST_HAS_FP4 1
#endif

#include <cmath>
#include <iostream>

using namespace fk;

// The constexpr software path and the runtime native path must agree inside the same TU.
static_assert(fp16(1.5f).bits() == 0x3E00);
static_assert(bf16(1.5f).bits() == 0x3FC0);
static_assert(sizeof(fp16) == sizeof(__half) && alignof(fp16) == alignof(__half));
static_assert(sizeof(bf16) == sizeof(__nv_bfloat16) && alignof(bf16) == alignof(__nv_bfloat16));
static_assert(sizeof(fp16_2) == sizeof(__half2) && alignof(fp16_2) == alignof(__half2));
static_assert(sizeof(bf16_2) == sizeof(__nv_bfloat162) && alignof(bf16_2) == alignof(__nv_bfloat162));
#if defined(UTEST_HAS_FP8)
static_assert(sizeof(fp8_e4m3) == sizeof(__nv_fp8_e4m3) && alignof(fp8_e4m3) == alignof(__nv_fp8_e4m3));
static_assert(sizeof(fp8_e5m2) == sizeof(__nv_fp8_e5m2) && alignof(fp8_e5m2) == alignof(__nv_fp8_e5m2));
static_assert(sizeof(fp8_e4m3_2) == sizeof(__nv_fp8x2_e4m3) && alignof(fp8_e4m3_2) == alignof(__nv_fp8x2_e4m3));
static_assert(sizeof(fp8_e4m3_4) == sizeof(__nv_fp8x4_e4m3) && alignof(fp8_e4m3_4) == alignof(__nv_fp8x4_e4m3));
#endif
#if defined(UTEST_HAS_FP4)
static_assert(sizeof(fp4_e2m1) == sizeof(__nv_fp4_e2m1) && alignof(fp4_e2m1) == alignof(__nv_fp4_e2m1));
#endif

namespace {
    int failures = 0;

    void reportFailure(const char* what, const unsigned int detail) {
        if (failures < 50) {
            std::cout << what << " (0x" << std::hex << detail << std::dec << ")" << std::endl;
        }
        ++failures;
    }

    void checkEncodeFp16(const float f) {
        const unsigned int sw = detail_rf::encode<detail_rf::FmtF16>(f);
        const unsigned short nat = __builtin_bit_cast(unsigned short, __half(f));
        if (std::isnan(f)) {
            const bool swNan = ((sw & 0x7C00u) == 0x7C00u) && ((sw & 0x3FFu) != 0u);
            const bool natNan = ((nat & 0x7C00u) == 0x7C00u) && ((nat & 0x3FFu) != 0u);
            if (!swNan || !natNan) reportFailure("fp16 NaN encode parity", sw);
        } else if (sw != nat) {
            reportFailure("fp16 encode parity", __builtin_bit_cast(unsigned int, f));
        }
    }

    void checkEncodeBf16(const float f) {
        const unsigned int sw = detail_rf::encode<detail_rf::FmtBF16>(f);
        const unsigned short nat = __builtin_bit_cast(unsigned short, __nv_bfloat16(f));
        if (std::isnan(f)) {
            const bool swNan = ((sw & 0x7F80u) == 0x7F80u) && ((sw & 0x7Fu) != 0u);
            const bool natNan = ((nat & 0x7F80u) == 0x7F80u) && ((nat & 0x7Fu) != 0u);
            if (!swNan || !natNan) reportFailure("bf16 NaN encode parity", sw);
        } else if (sw != nat) {
            reportFailure("bf16 encode parity", __builtin_bit_cast(unsigned int, f));
        }
    }
} // namespace

int launch() {
    // Exhaustive decode parity for the 2 byte formats. detail_rf::decode is called directly:
    // fp16/bf16 operator float takes the native fast path at runtime under nvcc, and the point
    // here is to compare the SOFTWARE decoder (the one CPU builds and constexpr use) against
    // the native converter.
    for (unsigned int code = 0; code < 0x10000u; ++code) {
        const float nativeF16 = __half2float(__builtin_bit_cast(__half, static_cast<unsigned short>(code)));
        const float fkF16 = detail_rf::decode<detail_rf::FmtF16>(code);
        if (std::isnan(nativeF16) ? !std::isnan(fkF16)
                                  : (nativeF16 != fkF16 || std::signbit(nativeF16) != std::signbit(fkF16))) {
            reportFailure("fp16 decode parity", code);
        }
        const float nativeBf16 = __bfloat162float(__builtin_bit_cast(__nv_bfloat16, static_cast<unsigned short>(code)));
        const float fkBf16 = detail_rf::decode<detail_rf::FmtBF16>(code);
        if (std::isnan(nativeBf16) ? !std::isnan(fkBf16)
                                   : (nativeBf16 != fkBf16 || std::signbit(nativeBf16) != std::signbit(fkBf16))) {
            reportFailure("bf16 decode parity", code);
        }
    }

    // Encode parity: directed corpus plus 2M sampled bit patterns
    const float corpus[] = { 0.0f, -0.0f, 1.0f, -1.0f, 0.5f, 1.5f, 2.5f, 65504.0f, 65505.0f, 65520.0f,
                             65536.0f, 1e20f, -1e20f, 5.96e-8f, 2.98e-8f, 1e-40f, -1e-45f, 3.4e38f,
                             INFINITY, -INFINITY, NAN, 448.0f, 449.0f, 464.0f, 465.0f, 57344.0f,
                             6.0f, 5.0f, 7.0f, 0.25f, 0.75f, 3.3895314e38f, 1.7014118e38f };
    for (const float f : corpus) {
        checkEncodeFp16(f);
        checkEncodeBf16(f);
    }
    unsigned int lcg = 12345u;
    for (int i = 0; i < 2000000; ++i) {
        lcg = lcg * 1664525u + 1013904223u;
        const float f = __builtin_bit_cast(float, lcg);
        checkEncodeFp16(f);
        checkEncodeBf16(f);
    }

    // Native interop roundtrip
    {
        const fp16 h(3.25f);
        const __half nh = toNative(h);
        if (toBits(fromNative(nh)) != h.bits() || __half2float(nh) != 3.25f) {
            reportFailure("fp16 toNative/fromNative roundtrip", h.bits());
        }
        const bf16 b(-7.5f);
        if (toBits(fromNative(toNative(b))) != b.bits()) {
            reportFailure("bf16 toNative/fromNative roundtrip", b.bits());
        }
    }

#if defined(UTEST_HAS_FP8)
    for (unsigned int code = 0; code < 256u; ++code) {
        __nv_fp8_e4m3 n43; n43.__x = static_cast<__nv_fp8_storage_t>(code);
        const float nat43 = static_cast<float>(n43);
        const float fk43 = static_cast<float>(fp8_e4m3::fromBits(static_cast<unsigned char>(code)));
        if (std::isnan(nat43) ? !std::isnan(fk43)
                              : (nat43 != fk43 || std::signbit(nat43) != std::signbit(fk43))) {
            reportFailure("fp8_e4m3 decode parity", code);
        }
        __nv_fp8_e5m2 n52; n52.__x = static_cast<__nv_fp8_storage_t>(code);
        const float nat52 = static_cast<float>(n52);
        const float fk52 = static_cast<float>(fp8_e5m2::fromBits(static_cast<unsigned char>(code)));
        if (std::isnan(nat52) ? !std::isnan(fk52)
                              : (nat52 != fk52 || std::signbit(nat52) != std::signbit(fk52))) {
            reportFailure("fp8_e5m2 decode parity", code);
        }
    }
    lcg = 999u;
    for (int i = 0; i < 500000; ++i) {
        lcg = lcg * 1664525u + 1013904223u;
        const float f = __builtin_bit_cast(float, lcg);
        if (detail_rf::encode<detail_rf::FmtE4M3>(f) !=
            static_cast<unsigned char>(__nv_cvt_float_to_fp8(f, __NV_SATFINITE, __NV_E4M3))) {
            reportFailure("fp8_e4m3 encode parity", lcg);
        }
        if (detail_rf::encode<detail_rf::FmtE5M2>(f) !=
            static_cast<unsigned char>(__nv_cvt_float_to_fp8(f, __NV_SATFINITE, __NV_E5M2))) {
            reportFailure("fp8_e5m2 encode parity", lcg);
        }
        // Also sample the fp8-relevant exponent range densely
        const float g = std::ldexp(static_cast<float>(static_cast<int>(lcg % 4096u)) / 256.0f - 8.0f,
                                   static_cast<int>(lcg % 40u) - 20);
        if (detail_rf::encode<detail_rf::FmtE4M3>(g) !=
            static_cast<unsigned char>(__nv_cvt_float_to_fp8(g, __NV_SATFINITE, __NV_E4M3))) {
            reportFailure("fp8_e4m3 encode parity (ranged)", __builtin_bit_cast(unsigned int, g));
        }
    }
#endif

#if defined(UTEST_HAS_FP4)
    for (unsigned int code = 0; code < 16u; ++code) {
        __nv_fp4_e2m1 n4; n4.__x = static_cast<__nv_fp4_storage_t>(code);
        const float nat = static_cast<float>(n4);
        const float fkv = static_cast<float>(fp4_e2m1::fromBits(static_cast<unsigned char>(code)));
        if (nat != fkv || std::signbit(nat) != std::signbit(fkv)) {
            reportFailure("fp4_e2m1 decode parity", code);
        }
    }
    for (int i = -20000; i <= 20000; ++i) {
        const float f = static_cast<float>(i) / 1000.0f;
        const __nv_fp4_e2m1 n4(f);
        if (detail_rf::encode<detail_rf::FmtE2M1>(f) != static_cast<unsigned int>(n4.__x & 0x0Fu)) {
            reportFailure("fp4_e2m1 encode parity", static_cast<unsigned int>(i));
        }
    }
    {
        const __nv_fp4_e2m1 nNan(NAN);
        if (detail_rf::encode<detail_rf::FmtE2M1>(NAN) != static_cast<unsigned int>(nNan.__x & 0x0Fu)) {
            reportFailure("fp4_e2m1 NaN encode parity", 0u);
        }
        const __nv_fp4_e2m1 nInf(INFINITY);
        if (detail_rf::encode<detail_rf::FmtE2M1>(INFINITY) != static_cast<unsigned int>(nInf.__x & 0x0Fu)) {
            reportFailure("fp4_e2m1 Inf encode parity", 0u);
        }
    }
#endif

    if (failures == 0) {
        std::cout << "utest_reduced_float_native_parity: all checks passed" << std::endl;
    }
    return failures == 0 ? 0 : -1;
}
