/* Copyright 2026 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

// cxp::nearbyint rounds at run time on the host with std::nearbyint where the compiler can tell constant evaluation from run time
// (FK_HOST_RUNTIME_MATH), and with its own constexpr code otherwise. The results must be the same bits as before that change:
// this test keeps the previous implementation (referenceNearbyint) and compares it with cxp::nearbyint, and cxp::saturate_cast with
// the previous composition, over deterministic samples that include the ties, the negative values, the sign of zero, the values
// next to the integers and the half integers, the denormals, the NaN and the infinities. The values from 2^31 (float) or 2^63
// (double) are not compared with the previous implementation: it casts to int or long long, which is not defined there (and
// saturate_cast clamps before it rounds); they must be returned unchanged, and are compared with std::nearbyint. Constant
// evaluation is checked with static_asserts. Built for the CPU only (it is host code).
#define __ONLY_CPU__

#include <tests/main.h>

#include <fused_kernel/core/constexpr_libs/constexpr_cmath.h>
#include <fused_kernel/core/constexpr_libs/constexpr_saturate.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

namespace {

// The implementation of cxp::nearbyint before the run time path, unchanged
template <typename ST>
ST referenceNearbyint(const ST& s) {
    static_assert(std::is_floating_point_v<ST>, "Input must be a floating-point type");
    if (cxp::isnan::BaseFunc::exec(s) || cxp::isinf::BaseFunc::exec(s) || (s == static_cast<ST>(0))) {
        return s;
    }
    ST fl{0};
    if constexpr (std::is_same_v<ST, double>) {
        const long long intPart = static_cast<long long>(s);
        if (s < ST(0) && s != static_cast<ST>(intPart)) {
            fl = static_cast<ST>(intPart - 1);
        } else {
            fl = static_cast<ST>(intPart);
        }
    } else {
        const ST intPart = static_cast<int>(s);
        if (s < ST(0) && s != static_cast<ST>(intPart)) {
            fl = static_cast<ST>(intPart - 1);
        } else {
            fl = static_cast<ST>(intPart);
        }
    }
    const ST frac = s - fl;
    if (frac < static_cast<ST>(0.5)) {
        return fl;
    } else if (frac > static_cast<ST>(0.5)) {
        return fl + static_cast<ST>(1.0);
    } else {
        if constexpr (std::is_same_v<ST, double>) {
            return cxp::is_even::BaseFunc::exec(static_cast<long long>(fl)) ? fl : fl + static_cast<ST>(1.0);
        } else {
            return cxp::is_even::BaseFunc::exec(static_cast<int>(fl)) ? fl : fl + static_cast<ST>(1.0);
        }
    }
}

// cxp::saturate_cast to an integer type before the run time path: the clamp, then the previous nearbyint
template <typename OT, typename ST>
OT referenceSaturateCast(const ST& s) {
    constexpr auto maxVal = cxp::maxValue<OT>;
    constexpr auto minVal = cxp::minValue<OT>;
    if (cxp::cmp_greater::BaseFunc::exec(s, maxVal)) {
        return maxVal;
    } else if (cxp::cmp_less::BaseFunc::exec(s, minVal)) {
        return minVal;
    } else {
        return static_cast<OT>(referenceNearbyint(s));
    }
}

template <typename ST>
struct Bits;
template <>
struct Bits<float> {
    using Type = uint32_t;
};
template <>
struct Bits<double> {
    using Type = uint64_t;
};

template <typename ST>
typename Bits<ST>::Type bitsOf(const ST value) {
    typename Bits<ST>::Type bits;
    std::memcpy(&bits, &value, sizeof(ST));
    return bits;
}
template <typename ST>
ST fromBits(const typename Bits<ST>::Type bits) {
    ST value;
    std::memcpy(&value, &bits, sizeof(ST));
    return value;
}

// The values from where the previous implementation is not defined: 2^31 for float (int), 2^63 for double (long long)
template <typename ST>
constexpr ST undefinedFrom = std::is_same_v<ST, float> ? static_cast<ST>(2147483648.0) : static_cast<ST>(9223372036854775808.0);

template <typename ST>
struct Checker {
    uint64_t compared{0}, mismatches{0}, saturateMismatches{0};

    void check(const ST s) {
        if (std::fabs(s) >= undefinedFrom<ST> && s == s) {
            // Not defined before: it must be returned as it is, as std::nearbyint does
            ++compared;
            const ST got = cxp::nearbyint::f(s);
            if (bitsOf(got) != bitsOf(std::nearbyint(s))) {
                report("beyond the integer range", s, got, std::nearbyint(s));
            }
            return;
        }
        ++compared;
        const ST expected = referenceNearbyint(s);
        const ST got = cxp::nearbyint::f(s);
        if (bitsOf(got) != bitsOf(expected) && !(got != got && expected != expected && bitsOf(got) == bitsOf(expected))) {
            report("nearbyint", s, got, expected);
        }
        if constexpr (std::is_same_v<ST, float>) {
            if (s == s) {
                if (cxp::saturate_cast<uchar>::f(s) != referenceSaturateCast<uchar>(s) ||
                    cxp::saturate_cast<signed char>::f(s) != referenceSaturateCast<signed char>(s) ||
                    cxp::saturate_cast<short>::f(s) != referenceSaturateCast<short>(s) ||
                    cxp::saturate_cast<ushort>::f(s) != referenceSaturateCast<ushort>(s) ||
                    cxp::saturate_cast<int>::f(s) != referenceSaturateCast<int>(s)) {
                    ++saturateMismatches;
                    if (saturateMismatches <= 5) {
                        std::cout << "saturate_cast differs for " << s << std::endl;
                    }
                }
            }
        }
    }
    void report(const char* what, const ST s, const ST got, const ST expected) {
        ++mismatches;
        if (mismatches <= 10) {
            std::cout << what << "(" << s << "): got " << got << " (bits " << std::hex << bitsOf(got) << "), expected " << expected << " (bits "
                      << bitsOf(expected) << ")" << std::dec << std::endl;
        }
    }
};

template <typename ST>
bool testType() {
    Checker<ST> c;
    using B = typename Bits<ST>::Type;
    const ST inf = std::numeric_limits<ST>::infinity(), nan = std::numeric_limits<ST>::quiet_NaN();
    // Special values
    for (const ST s : {ST(0), -ST(0), inf, -inf, nan, -nan, std::numeric_limits<ST>::denorm_min(), -std::numeric_limits<ST>::denorm_min(),
                       std::numeric_limits<ST>::min(), -std::numeric_limits<ST>::min(), std::numeric_limits<ST>::max(),
                       std::numeric_limits<ST>::lowest(), std::numeric_limits<ST>::epsilon(), ST(0.5), ST(-0.5), ST(1), ST(-1)}) {
        c.check(s);
    }
    // A signaling NaN
    c.check(fromBits<ST>(std::is_same_v<ST, float> ? B(0x7fa00000u) : B(0x7ff4000000000000ull)));
    // The integers and the half integers, and their neighbours (the ties and the values next to them), both signs
    for (int k = -6000; k <= 6000; ++k) {
        for (const ST base : {static_cast<ST>(k), static_cast<ST>(k) + static_cast<ST>(0.5)}) {
            ST up = base, down = base;
            c.check(base);
            for (int ulp = 0; ulp < 3; ++ulp) {
                up = std::nextafter(up, inf);
                down = std::nextafter(down, -inf);
                c.check(up);
                c.check(down);
            }
        }
    }
    // Bit patterns: a stride through all of them (a prime stride, so that every exponent and many mantissas are visited)
    {
        const B stride = std::is_same_v<ST, float> ? B(8191u) : B(1000000007ull) * B(131071ull);
        B bits = 0;
        const uint64_t count = std::is_same_v<ST, float> ? (1ull << 19) : (1ull << 20);
        for (uint64_t i = 0; i < count; ++i, bits += stride) {
            c.check(fromBits<ST>(bits));
        }
    }
    // Uniform values in the range where fractions matter and in the range of the integers a float represents exactly
    {
        uint64_t state = 0x9e3779b97f4a7c15ull;
        const auto next = [&state]() {
            state = state * 6364136223846793005ull + 1442695040888963407ull;
            return static_cast<double>(state >> 11) / 9007199254740992.0;
        };
        for (int i = 0; i < (1 << 19); ++i) {
            c.check(static_cast<ST>((next() - 0.5) * 600.0));
            c.check(static_cast<ST>((next() - 0.5) * 33554432.0));
        }
    }
    if (c.mismatches != 0 || c.saturateMismatches != 0) {
        std::cout << "cxp::nearbyint<" << (std::is_same_v<ST, float> ? "float" : "double") << ">: " << c.mismatches << " of " << c.compared
                  << " values differ from the previous implementation, " << c.saturateMismatches << " saturate_cast values" << std::endl;
        return false;
    }
    return true;
}

// Constant evaluation keeps working (it takes the code written in the function)
static_assert(cxp::nearbyint::f(0.5f) == 0.f && cxp::nearbyint::f(1.5f) == 2.f && cxp::nearbyint::f(2.5f) == 2.f &&
                  cxp::nearbyint::f(-0.5f) == 0.f && cxp::nearbyint::f(-1.5f) == -2.f && cxp::nearbyint::f(-2.5f) == -2.f &&
                  cxp::nearbyint::f(2.4f) == 2.f && cxp::nearbyint::f(2.6f) == 3.f && cxp::nearbyint::f(-2.6f) == -3.f,
              "cxp::nearbyint must round to nearest, ties to even, in constant expressions (float)");
static_assert(cxp::nearbyint::f(0.5) == 0. && cxp::nearbyint::f(1.5) == 2. && cxp::nearbyint::f(2.5) == 2. &&
                  cxp::nearbyint::f(-0.5) == 0. && cxp::nearbyint::f(-1.5) == -2. && cxp::nearbyint::f(-2.5) == -2. &&
                  cxp::nearbyint::f(1e15 + 0.5) == 1e15,
              "cxp::nearbyint must round to nearest, ties to even, in constant expressions (double)");
static_assert(cxp::saturate_cast<uchar>::f(254.5f) == 254 && cxp::saturate_cast<uchar>::f(255.5f) == 255 &&
                  cxp::saturate_cast<uchar>::f(300.f) == 255 && cxp::saturate_cast<uchar>::f(-3.f) == 0 &&
                  cxp::saturate_cast<uchar>::f(0.5f) == 0 && cxp::saturate_cast<uchar>::f(1.5f) == 2,
              "cxp::saturate_cast must round and clamp in constant expressions");

} // namespace

int launch() {
    const bool floatCorrect = testType<float>();
    const bool doubleCorrect = testType<double>();
#if defined(FK_HOST_RUNTIME_MATH)
    std::cout << "cxp::nearbyint takes the run time path on this compiler" << std::endl;
#else
    std::cout << "cxp::nearbyint has no run time path on this compiler (FK_HOST_RUNTIME_MATH is not defined)" << std::endl;
#endif
    return floatCorrect && doubleCorrect ? 0 : -1;
}
