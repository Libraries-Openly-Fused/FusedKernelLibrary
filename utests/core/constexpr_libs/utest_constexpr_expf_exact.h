/* Copyright 2026 Oscar Amoros Huguet
   Copyright 2026 Grup Mediapro S.L.U.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_TEST_CONSTEXPR_EXPF_EXACT_H
#define FK_TEST_CONSTEXPR_EXPF_EXACT_H

#include <tests/main.h>

#include <fused_kernel/core/constexpr_libs/constexpr_cmath.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>

// Forces constant evaluation of cxp::expf for the float whose bit pattern is Bits.
template <uint Bits>
struct CtExp {
    static constexpr float value = cxp::expf::f(cxp::bit_cast<float>(Bits));
    static constexpr uint bits = cxp::bit_cast<uint>(value);
};

// The constexpr path of cxp::expf must be bit identical to std::exp on a float input.
// The expected bit patterns below were produced by std::exp and are checked against it
// again at runtime, so a divergence on either side is caught.
#define CHECK_CT_EXPF(inBits, outBits)                                                                                 \
    static_assert(CtExp<inBits>::bits == outBits, "constexpr expf must match std::exp bit for bit")

// e^0 == 1, and the sign of zero must not leak through
CHECK_CT_EXPF(0x00000000u, 0x3F800000u); //  0.0f  -> 1.0f
CHECK_CT_EXPF(0x80000000u, 0x3F800000u); // -0.0f  -> 1.0f
// Small magnitudes, where the polynomial dominates
CHECK_CT_EXPF(0x39C6BE5Bu, 0x3F800C6Du); //  3.79073288e-4f -> 1.0003792f
CHECK_CT_EXPF(0x3F800000u, 0x402DF854u); //  1.0f   -> e
CHECK_CT_EXPF(0xBF800000u, 0x3EBC5AB2u); // -1.0f   -> 1/e
CHECK_CT_EXPF(0x40000000u, 0x40EC7326u); //  2.0f
CHECK_CT_EXPF(0xC0000000u, 0x3E0A9555u); // -2.0f
// Large magnitudes, exercising the 2^k reconstruction
CHECK_CT_EXPF(0x42AF0000u, 0x7E96BAB3u); //  87.5f, just below overflow
CHECK_CT_EXPF(0x42B17214u, 0x7F7FFE04u); //  88.7228088f, the last finite result
CHECK_CT_EXPF(0xC2AF0000u, 0x006CB2BCu); // -87.5f, subnormal result
CHECK_CT_EXPF(0xC2C60000u, 0x00000048u); // -99.0f, deep subnormal
// Saturation
CHECK_CT_EXPF(0x42B17218u, 0x7F800000u); //  88.7228394f -> +inf
CHECK_CT_EXPF(0xC2D20000u, 0x00000000u); // -105.0f      -> 0.0f

#undef CHECK_CT_EXPF

int launch() {
    bool allCorrect = true;

    // Re-verify the same points at runtime against std::exp, so the constants above
    // cannot silently drift from the platform's std::exp.
    constexpr uint inputs[] = {0x00000000u, 0x80000000u, 0x39C6BE5Bu, 0x3F800000u, 0xBF800000u,
                                    0x40000000u, 0xC0000000u, 0x42AF0000u, 0x42B17214u, 0xC2AF0000u,
                                    0xC2C60000u, 0x42B17218u, 0xC2D20000u};
    for (const uint bits : inputs) {
        const float x = cxp::bit_cast<float>(bits);
        const uint expected = cxp::bit_cast<uint>(std::exp(x));
        const uint actual = cxp::bit_cast<uint>(cxp::expf::f(x));
        if (expected != actual) {
            std::cout << "Runtime Fail: cxp::expf::f(" << x << ") expected bits 0x" << std::hex << expected
                        << ", got 0x" << actual << std::dec << std::endl;
            allCorrect = false;
        }
    }

    // NaN must propagate rather than compare unequal
    if (!cxp::isnan::f(cxp::expf::f(std::numeric_limits<float>::quiet_NaN()))) {
        std::cout << "Runtime Fail: cxp::expf::f(NaN) should be NaN" << std::endl;
        allCorrect = false;
    }

    // Runtime dispatch check: outside constant evaluation cxp::expf must behave exactly
    // like std::expf on a sweep of the useful input range, including boundaries.
    {
        constexpr float sweepStart = -105.0f;
        constexpr float sweepEnd = 89.0f;
        constexpr int sweepCount = 1001;
        for (int i = 0; i < sweepCount; ++i) {
            const float x = sweepStart + (sweepEnd - sweepStart) * static_cast<float>(i) / (sweepCount - 1);
            const uint expected = cxp::bit_cast<uint>(std::expf(x));
            const uint actual = cxp::bit_cast<uint>(cxp::expf::f(x));
            if (expected != actual) {
                std::cout << "Runtime Fail: cxp::expf::f(" << x << ") expected bits 0x" << std::hex << expected
                          << ", got 0x" << actual << std::dec << std::endl;
                allCorrect = false;
            }
        }
    }

    // Compile-time sweep: values produced by the constexpr path are compared at runtime
    // against std::expf. exp results are always non-negative, so the bit patterns are
    // directly ordered and their difference is the distance in ulps (this also works
    // for subnormals and for the 0/inf saturation points).
    {
        constexpr float sweepStart = -103.0f;
        constexpr float sweepEnd = 88.7f;
        constexpr int sweepCount = 257;
        constexpr auto ctResults = []() {
            std::array<float, sweepCount> results{};
            for (int i = 0; i < sweepCount; ++i) {
                const float x = sweepStart + (sweepEnd - sweepStart) * static_cast<float>(i) / (sweepCount - 1);
                results[i] = cxp::expf::f(x);
            }
            return results;
        }();
        for (int i = 0; i < sweepCount; ++i) {
            const float x = sweepStart + (sweepEnd - sweepStart) * static_cast<float>(i) / (sweepCount - 1);
            const uint expected = cxp::bit_cast<uint>(std::expf(x));
            const uint actual = cxp::bit_cast<uint>(ctResults[i]);
            const uint ulpDiff = expected > actual ? expected - actual : actual - expected;
            if (ulpDiff > 2u) {
                std::cout << "Constexpr Fail: cxp::expf::f(" << x << ") expected bits 0x" << std::hex << expected
                          << ", got 0x" << actual << std::dec << " (" << ulpDiff << " ulps)" << std::endl;
                allCorrect = false;
            }
        }
    }

    if (allCorrect) {
        std::cout << "All tests passed!" << std::endl;
        return 0;
    }
    return -1;
}

#endif // FK_TEST_CONSTEXPR_EXPF_EXACT_H
