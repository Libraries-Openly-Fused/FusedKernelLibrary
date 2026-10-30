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

// At run time on the host (FK_HOST_RUNTIME_MATH), cxp::saturate_cast from a floating point type to an integer type has no branches:
// it selects between the limits and the rounded value. This test keeps the previous body of cxp::saturate_cast (referenceSaturateCast:
// the two comparisons and then the rounding) and compares it with cxp::saturate_cast over deterministic samples (the limits and the
// values around them, the ties, the sign of zero, the denormals, the infinities, strided bit patterns and uniform values). The
// values for which the previous body is not defined are checked against the values they are given now: a NaN of any sign, kind and
// payload gives 0 for every output type (it was undefined: 0 for the narrow types, the smallest int on x86), and 2^31 as a float
// gives the maximum of int (it was cast to int unchanged, which is not defined). Constant evaluation is checked with static_asserts.
// Built for the CPU only (it is host code).
#define __ONLY_CPU__

#include <tests/main.h>

#include <fused_kernel/core/constexpr_libs/constexpr_cmath.h>
#include <fused_kernel/core/constexpr_libs/constexpr_saturate.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <type_traits>

namespace {

// The body of cxp::saturate_cast to an integer type before the branch free path, unchanged
template <typename OT, typename ST>
OT referenceSaturateCast(const ST& s) {
    constexpr auto maxVal = cxp::maxValue<OT>;
    constexpr auto minVal = cxp::minValue<OT>;
    if (cxp::cmp_greater::BaseFunc::exec(s, maxVal)) {
        return maxVal;
    } else if (cxp::cmp_less::BaseFunc::exec(s, minVal)) {
        return minVal;
    } else {
        return static_cast<OT>(cxp::nearbyint::BaseFunc::exec(s));
    }
}

// cxp::saturate_cast of a value the compiler can not see: after a comparison with a constant (s == the limit, s != s) it would
// know the value and fold the conversion of the previous body to the maximum, where the instruction of the CPU gives the minimum
template <typename OT, typename ST>
OT saturateCast(const ST s) {
    const volatile ST hidden = s;
    return cxp::saturate_cast<OT>::f(static_cast<ST>(hidden));
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
ST fromBits(const typename Bits<ST>::Type bits) {
    ST value;
    std::memcpy(&value, &bits, sizeof(ST));
    return value;
}

const char* nameOf(const uchar*) { return "uchar"; }
const char* nameOf(const signed char*) { return "schar"; }
const char* nameOf(const short*) { return "short"; }
const char* nameOf(const ushort*) { return "ushort"; }
const char* nameOf(const int*) { return "int"; }
const char* nameOf(const uint*) { return "uint"; }

template <typename ST>
struct Checker {
    uint64_t compared{0}, mismatches{0}, nanWrong{0}, limitWrong{0};

    template <typename OT>
    void checkOne(const ST s) {
        // Not defined before: a NaN, and the value that is the limit of the output type rounded up in the floating point type
        if (s != s) {
            ++compared;
            if (saturateCast<OT>(s) != static_cast<OT>(0)) {
                if (++nanWrong <= 5) std::cout << "saturate_cast<" << nameOf(static_cast<OT*>(nullptr)) << ">(NaN) is not 0" << std::endl;
            }
            return;
        }
        if (s == static_cast<ST>(cxp::maxValue<OT>) && !std::is_same_v<OT, ST>) {
            ++compared;
            if (saturateCast<OT>(s) != cxp::maxValue<OT>) {
                if (++limitWrong <= 5) std::cout << "saturate_cast<" << nameOf(static_cast<OT*>(nullptr)) << ">(" << s << ") is not the maximum" << std::endl;
            }
            return;
        }
        ++compared;
        const OT got = saturateCast<OT>(s);
        const OT expected = referenceSaturateCast<OT>(s);
        if (got != expected) {
            if (++mismatches <= 10) {
                std::cout << "saturate_cast<" << nameOf(static_cast<OT*>(nullptr)) << ">(" << s << "): got " << +got << ", expected " << +expected
                          << std::endl;
            }
        }
    }
    void check(const ST s) {
        checkOne<uchar>(s);
        checkOne<signed char>(s);
        checkOne<short>(s);
        checkOne<ushort>(s);
        checkOne<int>(s);
        if constexpr (std::is_same_v<ST, double>) {
            checkOne<uint>(s);
        }
    }
};

template <typename ST>
bool testType() {
    Checker<ST> c;
    using B = typename Bits<ST>::Type;
    const ST inf = std::numeric_limits<ST>::infinity(), nan = std::numeric_limits<ST>::quiet_NaN();
    for (const ST s : {ST(0), -ST(0), inf, -inf, std::numeric_limits<ST>::denorm_min(), -std::numeric_limits<ST>::denorm_min(),
                       std::numeric_limits<ST>::min(), -std::numeric_limits<ST>::min(), std::numeric_limits<ST>::max(),
                       std::numeric_limits<ST>::lowest(), std::numeric_limits<ST>::epsilon(), ST(0.5), ST(-0.5), ST(1), ST(-1)}) {
        c.check(s);
    }
    // NaN: both signs, quiet and signaling, and payloads
    c.check(nan);
    c.check(-nan);
    constexpr int mantissaBits = std::is_same_v<ST, float> ? 23 : 52;
    constexpr B exponent = std::is_same_v<ST, float> ? B(0x7f800000u) : B(0x7ff0000000000000ull);
    constexpr B signBit = std::is_same_v<ST, float> ? B(0x80000000u) : B(0x8000000000000000ull);
    for (int bit = 0; bit < mantissaBits; ++bit) {
        c.check(fromBits<ST>(exponent | (B(1) << bit)));
        c.check(fromBits<ST>(exponent | signBit | (B(1) << bit)));
    }
    // The limits of the output types, the values next to them and the ties next to them
    for (const double limit : {255.0, 127.0, -128.0, 32767.0, -32768.0, 65535.0, 2147483647.0, -2147483648.0, 4294967295.0, 0.0}) {
        for (const double delta : {-2.0, -1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0}) {
            ST up = static_cast<ST>(limit + delta), down = up;
            c.check(up);
            for (int ulp = 0; ulp < 4; ++ulp) {
                up = std::nextafter(up, inf);
                down = std::nextafter(down, -inf);
                c.check(up);
                c.check(down);
            }
        }
    }
    // The integers and the half integers around the narrow types, both signs
    for (int k = -70000; k <= 70000; ++k) {
        c.check(static_cast<ST>(k));
        c.check(static_cast<ST>(k) + static_cast<ST>(0.5));
        c.check(static_cast<ST>(k) - static_cast<ST>(0.5));
    }
    // Bit patterns: a stride through all of them
    {
        const B stride = std::is_same_v<ST, float> ? B(8191u) : B(1000000007ull) * B(131071ull);
        B bits = 0;
        const uint64_t count = std::is_same_v<ST, float> ? (1ull << 20) : (1ull << 20);
        for (uint64_t i = 0; i < count; ++i, bits += stride) {
            c.check(fromBits<ST>(bits));
        }
    }
    // Uniform values in the range of the narrow types and in the range of int
    {
        uint64_t state = 0x9e3779b97f4a7c15ull;
        const auto next = [&state]() {
            state = state * 6364136223846793005ull + 1442695040888963407ull;
            return static_cast<double>(state >> 11) / 9007199254740992.0;
        };
        for (int i = 0; i < (1 << 19); ++i) {
            c.check(static_cast<ST>((next() - 0.5) * 600.0));
            c.check(static_cast<ST>((next() - 0.5) * 140000.0));
            c.check(static_cast<ST>((next() - 0.5) * 5000000000.0));
        }
    }
    if (c.mismatches != 0 || c.nanWrong != 0 || c.limitWrong != 0) {
        std::cout << "cxp::saturate_cast<" << (std::is_same_v<ST, float> ? "float" : "double") << ">: " << c.mismatches << " of " << c.compared
                  << " values differ from the previous body, " << c.nanWrong << " NaN are not 0, " << c.limitWrong
                  << " limits are not the maximum" << std::endl;
        return false;
    }
    return true;
}

// Constant evaluation keeps working (it takes the code written in the function)
static_assert(cxp::saturate_cast<uchar>::f(254.5f) == 254 && cxp::saturate_cast<uchar>::f(255.5f) == 255 &&
                  cxp::saturate_cast<uchar>::f(300.f) == 255 && cxp::saturate_cast<uchar>::f(-3.f) == 0 &&
                  cxp::saturate_cast<uchar>::f(0.5f) == 0 && cxp::saturate_cast<uchar>::f(1.5f) == 2,
              "cxp::saturate_cast must round and clamp in constant expressions");
static_assert(cxp::saturate_cast<signed char>::f(-128.5) == -128 && cxp::saturate_cast<signed char>::f(127.4) == 127 &&
                  cxp::saturate_cast<short>::f(-2.5f) == -2 && cxp::saturate_cast<ushort>::f(65535.4f) == 65535 &&
                  cxp::saturate_cast<int>::f(-2147483648.0) == std::numeric_limits<int>::min() &&
                  cxp::saturate_cast<int>::f(2147483647.0) == std::numeric_limits<int>::max(),
              "cxp::saturate_cast must clamp to the limits in constant expressions");

} // namespace

int launch() {
    const bool floatCorrect = testType<float>();
    const bool doubleCorrect = testType<double>();
    // 2^31 as a float is the limit of int rounded up: the maximum of int (it was cast unchanged before, which is not defined)
    bool limitsCorrect = true;
    const volatile float twoPow31 = 2147483648.f, belowTwoPow31 = 2147483520.f, minusTwoPow31 = -2147483648.f;
    const volatile float posInf = std::numeric_limits<float>::infinity(), negInf = -std::numeric_limits<float>::infinity();
    if (cxp::saturate_cast<int>::f(static_cast<float>(twoPow31)) != std::numeric_limits<int>::max() ||
        cxp::saturate_cast<int>::f(static_cast<float>(belowTwoPow31)) != 2147483520 ||
        cxp::saturate_cast<int>::f(static_cast<float>(minusTwoPow31)) != std::numeric_limits<int>::min() ||
        cxp::saturate_cast<int>::f(static_cast<float>(posInf)) != std::numeric_limits<int>::max() ||
        cxp::saturate_cast<int>::f(static_cast<float>(negInf)) != std::numeric_limits<int>::min()) {
        std::cout << "saturate_cast<int> of the float limits is wrong" << std::endl;
        limitsCorrect = false;
    }
    // The same for the other limits that are not exact in the floating point type: 2^32 (unsigned) and 2^63 (long long) and 2^64
    // (unsigned long long) as a float or a double. They gave 0, the smallest long long and 0 or 2^63 before.
    const volatile float f32 = 4294967296.f, f63 = 9223372036854775808.f, f64 = 18446744073709551616.f;
    const volatile double d63 = 9223372036854775808.0, d64 = 18446744073709551616.0;
    if (cxp::saturate_cast<unsigned>::f(static_cast<float>(f32)) != std::numeric_limits<unsigned>::max() ||
        cxp::saturate_cast<long long>::f(static_cast<float>(f63)) != std::numeric_limits<long long>::max() ||
        cxp::saturate_cast<unsigned long long>::f(static_cast<float>(f64)) != std::numeric_limits<unsigned long long>::max() ||
        cxp::saturate_cast<long long>::f(static_cast<double>(d63)) != std::numeric_limits<long long>::max() ||
        cxp::saturate_cast<unsigned long long>::f(static_cast<double>(d64)) != std::numeric_limits<unsigned long long>::max()) {
        std::cout << "saturate_cast to unsigned, long long and unsigned long long of 2^32, 2^63 and 2^64 is wrong" << std::endl;
        limitsCorrect = false;
    }
#if defined(FK_HOST_RUNTIME_MATH)
    std::cout << "cxp::saturate_cast takes the run time path on this compiler" << std::endl;
#else
    std::cout << "cxp::saturate_cast has no run time path on this compiler (FK_HOST_RUNTIME_MATH is not defined)" << std::endl;
#endif
    return floatCorrect && doubleCorrect && limitsCorrect ? 0 : -1;
}
