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

#ifndef FK_TEST_CONSTEXPR_CMATH_H
#define FK_TEST_CONSTEXPR_CMATH_H

#include <fused_kernel/core/constexpr_libs/constexpr_cmath.h>
#include <fused_kernel/core/utils/type_to_string.h>
#include <limits>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <iomanip>

// Test isnan function compile-time
template <typename T>
constexpr bool test_isnan_ct() {
    static_assert(std::is_floating_point_v<T>, "isnan test only for floating point types");

    // Test with normal values
    static_assert(!cxp::isnan::f(static_cast<T>(0.0)), "0.0 should not be NaN");
    static_assert(!cxp::isnan::f(static_cast<T>(1.0)), "1.0 should not be NaN");
    static_assert(!cxp::isnan::f(static_cast<T>(-1.0)), "-1.0 should not be NaN");
    static_assert(!cxp::isnan::f(std::numeric_limits<T>::max()), "max value should not be NaN");
    static_assert(!cxp::isnan::f(std::numeric_limits<T>::min()), "min value should not be NaN");
    static_assert(!cxp::isnan::f(std::numeric_limits<T>::lowest()), "lowest value should not be NaN");
    static_assert(!cxp::isnan::f(std::numeric_limits<T>::infinity()), "infinity should not be NaN");
    static_assert(!cxp::isnan::f(-std::numeric_limits<T>::infinity()), "-infinity should not be NaN");

    // Test with NaN
    static_assert(cxp::isnan::f(std::numeric_limits<T>::quiet_NaN()), "quiet_NaN should be NaN");
    static_assert(cxp::isnan::f(std::numeric_limits<T>::signaling_NaN()), "signaling_NaN should be NaN");

    return true;
}

// Test isnan function at runtime
template <typename T>
bool test_isnan_rt() {
    static_assert(std::is_floating_point_v<T>, "isnan test only for floating point types");
    bool allCorrect{true};
    // Test with normal values
    if (cxp::isnan::f(static_cast<T>(1.0)) != std::isnan(static_cast<T>(1.0f))) {
        std::cout << "Failed: cxp::isnan::f(1.0f) should be the same as std::isnan(1.0f)" << std::endl;
        allCorrect = false;
    }
    if (cxp::isnan::f(static_cast<T>(0.0)) != std::isnan(static_cast<T>(0.0))) {
        std::cout << "Failed: cxp::isnan::f(0.0f) should be the same as std::isnan(0.0f)" << std::endl;
        allCorrect = false;
    }
    if (cxp::isnan::f(static_cast<T>(-1.0)) != std::isnan(static_cast<T>(-1.0))) {
        std::cout << "Failed: cxp::isnan::f(-1.0f) should be the same as std::isnan(-1.0f)" << std::endl;
        allCorrect = false;
    }
    if (cxp::isnan::f(std::numeric_limits<T>::max()) != std::isnan(std::numeric_limits<T>::max())) {
        std::cout << "Failed: cxp::isnan::f(max) should be the same as std::isnan(max)" << std::endl;
        allCorrect = false;
    }
    if (cxp::isnan::f(std::numeric_limits<T>::min()) != std::isnan(std::numeric_limits<T>::min())) {
        std::cout << "Failed: cxp::isnan::f(min) should be the same as std::isnan(min)" << std::endl;
        allCorrect = false;
    }
    if (cxp::isnan::f(std::numeric_limits<T>::lowest()) != std::isnan(std::numeric_limits<T>::lowest())) {
        std::cout << "Failed: cxp::isnan::f(lowest) should be the same as std::isnan(lowest)" << std::endl;
        allCorrect = false;
    }
    if (cxp::isnan::f(std::numeric_limits<T>::infinity()) != std::isnan(std::numeric_limits<T>::infinity())) {
        std::cout << "Failed: cxp::isnan::f(infinity) should be the same as std::isnan(infinity)" << std::endl;
        allCorrect = false;
    }
    if (cxp::isnan::f(-std::numeric_limits<T>::infinity()) != std::isnan(-std::numeric_limits<T>::infinity())) {
        std::cout << "Failed: cxp::isnan::f(-infinity) should be the same as std::isnan(-infinity)" << std::endl;
        allCorrect = false;
    }

    // Test with NaN
    if (cxp::isnan::f(std::numeric_limits<T>::quiet_NaN()) != std::isnan(std::numeric_limits<T>::quiet_NaN())) {
        std::cout << "Failed: cxp::isnan::f(quiet_NaN) should be the same as std::isnan(quiet_NaN)" << std::endl;
        allCorrect = false;
    }
    if (cxp::isnan::f(std::numeric_limits<T>::signaling_NaN()) != std::isnan(std::numeric_limits<T>::signaling_NaN())) {
        std::cout << "Failed: cxp::isnan::f(signaling_NaN) should be the same as std::isnan(signaling_NaN)" << std::endl;
        allCorrect = false;
    }

    return allCorrect;
}

// Test isinf function
template <typename T>
constexpr bool test_isinf_ct() {
    static_assert(std::is_floating_point_v<T>, "isinf test only for floating point types");

    // Test with normal values
    static_assert(!cxp::isinf::f(static_cast<T>(0.0)), "0.0 should not be infinite");
    static_assert(!cxp::isinf::f(static_cast<T>(1.0)), "1.0 should not be infinite");
    static_assert(!cxp::isinf::f(static_cast<T>(-1.0)), "-1.0 should not be infinite");
    static_assert(!cxp::isinf::f(static_cast<T>(1000.0)), "1000.0 should not be infinite");
    static_assert(!cxp::isinf::f(static_cast<T>(-1000.0)), "-1000.0 should not be infinite");
    static_assert(!cxp::isinf::f(std::numeric_limits<T>::quiet_NaN()), "NaN should not be infinite");

    // Test with infinity
    static_assert(cxp::isinf::f(std::numeric_limits<T>::infinity()), "infinity should be infinite");
    static_assert(cxp::isinf::f(-std::numeric_limits<T>::infinity()), "-infinity should be infinite");

    return true;
}

// Test isinf function at runtime
template <typename T>
constexpr bool test_isinf_rt() {
    static_assert(std::is_floating_point_v<T>, "isinf test only for floating point types");
    bool allCorrect{true};
    // Test isinf with runtime values
    T inf_val = std::numeric_limits<T>::infinity();
    if (cxp::isinf::f(inf_val) != std::isinf(inf_val)) {
        std::cout << "Failed: cxp::isinf::f(inf_val) should be the same as std::isinf(inf_val)" << std::endl;
        allCorrect = false;
    }
    if (cxp::isinf::f(-inf_val) != std::isinf(-inf_val)) {
        std::cout << "Failed: cxp::isinf::f(-inf_val) should be the same as std::isinf(-inf_val)" << std::endl;
        allCorrect = false;
    }
    if (cxp::isinf::f(1.0f) != std::isinf(1.0f)) {
        std::cout << "Failed: cxp::isinf::f(1.0f) should be the same as std::isinf(1.0f)" << std::endl;
        allCorrect = false;
    }
    if (cxp::isinf::f(-1.0f) != std::isinf(-1.0f)) {
        std::cout << "Failed: cxp::isinf::f(-1.0f) should be the same as std::isinf(-1.0f)" << std::endl;
        allCorrect = false;
    }
    if (cxp::isinf::f(std::numeric_limits<T>::max()) != std::isinf(std::numeric_limits<T>::max())) {
        std::cout << "Failed: cxp::isinf::f(max) should be the same as std::isinf(max)" << std::endl;
        allCorrect = false;
    }
    if (cxp::isinf::f(std::numeric_limits<T>::min()) != std::isinf(std::numeric_limits<T>::min())) {
        std::cout << "Failed: cxp::isinf::f(min) should be the same as std::isinf(min)" << std::endl;
        allCorrect = false;
    }
    if (cxp::isinf::f(std::numeric_limits<T>::lowest()) != std::isinf(std::numeric_limits<T>::lowest())) {
        std::cout << "Failed: cxp::isinf::f(lowest) should be the same as std::isinf(lowest)" << std::endl;
        allCorrect = false;
    }
    if (cxp::isinf::f(0.0f) != std::isinf(0.0f)) {
        std::cout << "Failed: cxp::isinf::f(0.0f) should be the same as std::isinf(0.0f)" << std::endl;
        allCorrect = false;
    }
    return allCorrect;
}

// Test cmp_equal function
constexpr bool test_cmp_equal() {
    // Same type comparisons
    static_assert(cxp::cmp_equal::f(5, 5), "5 == 5 should be true");
    static_assert(!cxp::cmp_equal::f(5, 4), "5 == 4 should be false");
    static_assert(cxp::cmp_equal::f(5.0, 5.0), "5.0 == 5.0 should be true");
    static_assert(!cxp::cmp_equal::f(5.0, 4.0), "5.0 == 4.0 should be false");
    
    // Mixed signed/unsigned comparisons
    static_assert(cxp::cmp_equal::f(5, 5u), "5 == 5u should be true");
    static_assert(!cxp::cmp_equal::f(-1, 5u), "-1 == 5u should be false");
    static_assert(!cxp::cmp_equal::f(5u, -1), "5u == -1 should be false");
    static_assert(cxp::cmp_equal::f(0, 0u), "0 == 0u should be true");
    
    // Mixed integer/floating point comparisons
    static_assert(cxp::cmp_equal::f(5, 5.0), "5 == 5.0 should be true");
    static_assert(cxp::cmp_equal::f(5.0, 5), "5.0 == 5 should be true");
    static_assert(!cxp::cmp_equal::f(5, 5.1), "5 == 5.1 should be false");

    return true;
}

// Test cmp_not_equal function
constexpr bool test_cmp_not_equal() {
    static_assert(!cxp::cmp_not_equal::f(5, 5), "5 != 5 should be false");
    static_assert(cxp::cmp_not_equal::f(5, 4), "5 != 4 should be true");
    static_assert(cxp::cmp_not_equal::f(-1, 5u), "-1 != 5u should be true");
    static_assert(!cxp::cmp_not_equal::f(5, 5.0), "5 != 5.0 should be false");

    return true;
}

// Test cmp_less function
constexpr bool test_cmp_less() {
    // Same type comparisons
    static_assert(cxp::cmp_less::f(4, 5), "4 < 5 should be true");
    static_assert(!cxp::cmp_less::f(5, 4), "5 < 4 should be false");
    static_assert(!cxp::cmp_less::f(5, 5), "5 < 5 should be false");
    
    // Mixed signed/unsigned comparisons
    static_assert(cxp::cmp_less::f(-1, 5u), "-1 < 5u should be true");
    static_assert(!cxp::cmp_less::f(5u, -1), "5u < -1 should be false");
    static_assert(cxp::cmp_less::f(4u, 5), "4u < 5 should be true");
    static_assert(!cxp::cmp_less::f(5u, 4), "5u < 4 should be false");
    
    // Mixed integer/floating point comparisons
    static_assert(cxp::cmp_less::f(4, 5.0), "4 < 5.0 should be true");
    static_assert(cxp::cmp_less::f(4.0, 5), "4.0 < 5 should be true");
    static_assert(!cxp::cmp_less::f(5.0, 4), "5.0 < 4 should be false");
    
    return true;
}

// Test cmp_greater function
constexpr bool test_cmp_greater() {
    static_assert(cxp::cmp_greater::f(5, 4), "5 > 4 should be true");
    static_assert(!cxp::cmp_greater::f(4, 5), "4 > 5 should be false");
    static_assert(!cxp::cmp_greater::f(5, 5), "5 > 5 should be false");
    static_assert(cxp::cmp_greater::f(5u, -1), "5u > -1 should be true");
    static_assert(!cxp::cmp_greater::f(-1, 5u), "-1 > 5u should be false");
    
    return true;
}

// Test cmp_less_equal function
constexpr bool test_cmp_less_equal() {
    static_assert(cxp::cmp_less_equal::f(4, 5), "4 <= 5 should be true");
    static_assert(cxp::cmp_less_equal::f(5, 5), "5 <= 5 should be true");
    static_assert(!cxp::cmp_less_equal::f(5, 4), "5 <= 4 should be false");
    static_assert(cxp::cmp_less_equal::f(-1, 5u), "-1 <= 5u should be true");
    static_assert(cxp::cmp_less_equal::f(0, 0u), "0 <= 0u should be true");
    
    return true;
}

// Test cmp_greater_equal function
constexpr bool test_cmp_greater_equal() {
    static_assert(cxp::cmp_greater_equal::f(5, 4), "5 >= 4 should be true");
    static_assert(cxp::cmp_greater_equal::f(5, 5), "5 >= 5 should be true");
    static_assert(!cxp::cmp_greater_equal::f(4, 5), "4 >= 5 should be false");
    static_assert(cxp::cmp_greater_equal::f(5u, -1), "5u >= -1 should be true");
    static_assert(cxp::cmp_greater_equal::f(0u, 0), "0u >= 0 should be true");
    
    return true;
}

// Runtime tests for round function
template <typename T>
bool test_round_rt() {
    static_assert(std::is_floating_point_v<T>, "round test only for floating point types");

    // Runtime test
    bool allCorrect{true};
    if (cxp::round::f(static_cast<T>(1.4)) != std::round(static_cast<T>(1.4))) {
        std::cout << "cxp::round::f(1.4) failed" << std::endl;
        allCorrect = false;
    }
    if (cxp::round::f(static_cast<T>(1.5)) != std::round(static_cast<T>(1.5))) {
        std::cout << "cxp::round::f(1.5) failed" << std::endl;
        allCorrect = false;
    }
    if (cxp::round::f(static_cast<T>(1.6)) != std::round(static_cast<T>(1.6))) {
        std::cout << "cxp::round::f(1.6) failed" << std::endl;
        allCorrect = false;
    }
    if (cxp::round::f(static_cast<T>(-1.4)) != std::round(static_cast<T>(-1.4))) {
        std::cout << "cxp::round::f(-1.4) failed" << std::endl;
        allCorrect = false;
    }
    if (cxp::round::f(static_cast<T>(-1.5)) != std::round(static_cast<T>(-1.5))) {
        std::cout << "cxp::round::f(-1.5) failed" << std::endl;
        allCorrect = false;
    }
    if (cxp::round::f(static_cast<T>(-1.6)) != std::round(static_cast<T>(-1.6))) {
        std::cout << "cxp::round::f(-1.6) failed" << std::endl;
        allCorrect = false;
    }
    if (cxp::round::f(static_cast<T>(0.0)) != std::round(static_cast<T>(0.0))) {
        std::cout << "cxp::round::f(0.0) failed" << std::endl;
        allCorrect = false;
    }
    if (cxp::round::f(static_cast<T>(2.0)) != std::round(static_cast<T>(2.0))) {
        std::cout << "cxp::round::f(2.0) failed" << std::endl;
        allCorrect = false;
    }
    
    // Special values runtime
    if (!std::isnan(cxp::round::f(std::numeric_limits<T>::quiet_NaN()))) {
        std::cout << "Failed: cxp::round::f(NaN) should be NaN" << std::endl;
        allCorrect = false;
    }
    if (!std::isinf(cxp::round::f(std::numeric_limits<T>::infinity()))) {
        std::cout << "Failed: cxp::round::f(inf) should be inf" << std::endl;
        allCorrect = false;
    }
    if (!std::isinf(cxp::round::f(-std::numeric_limits<T>::infinity()))) {
        std::cout << "Failed: cxp::round::f(-inf) should be -inf" << std::endl;
        allCorrect = false;
    }
    
    return allCorrect;
}

// Compile time test round function
template <typename T>
constexpr bool test_round_ct() {
    static_assert(std::is_floating_point_v<T>, "round test only for floating point types");

    // Compile-time tests
    static_assert(cxp::round::f(static_cast<T>(1.4)) == static_cast<T>(1.0), "round(1.4) should be 1.0");
    static_assert(cxp::round::f(static_cast<T>(1.5)) == static_cast<T>(2.0), "round(1.5) should be 2.0");
    static_assert(cxp::round::f(static_cast<T>(1.6)) == static_cast<T>(2.0), "round(1.6) should be 2.0");
    static_assert(cxp::round::f(static_cast<T>(-1.4)) == static_cast<T>(-1.0), "round(-1.4) should be -1.0");
    static_assert(cxp::round::f(static_cast<T>(-1.5)) == static_cast<T>(-2.0), "round(-1.5) should be -2.0");
    static_assert(cxp::round::f(static_cast<T>(-1.6)) == static_cast<T>(-2.0), "round(-1.6) should be -2.0");
    static_assert(cxp::round::f(static_cast<T>(0.0)) == static_cast<T>(0.0), "round(0.0) should be 0.0");
    static_assert(cxp::round::f(static_cast<T>(2.0)) == static_cast<T>(2.0), "round(2.0) should be 2.0");
    static_assert(cxp::isnan::f(cxp::round::f(std::numeric_limits<T>::quiet_NaN())), "round(NaN) should be NaN");
    static_assert(cxp::isinf::f(cxp::round::f(std::numeric_limits<T>::infinity())), "round(inf) should be inf");
    static_assert(cxp::isinf::f(cxp::round::f(-std::numeric_limits<T>::infinity())), "round(-inf) should be -inf");

    return true;
}

// Test abs function at compile-time
template <typename T>
constexpr bool test_abs_ct() {
    static_assert(std::is_fundamental_v<T>, "abs test only for fundamental types");
    
    if constexpr (std::is_signed_v<T> && sizeof(T) >= 4) {
        static_assert(cxp::abs::f(static_cast<T>(5)) == static_cast<T>(5), "abs(5) should be 5");
        static_assert(cxp::abs::f(static_cast<T>(-5)) == static_cast<T>(5), "abs(-5) should be 5");
        static_assert(cxp::abs::f(static_cast<T>(0)) == static_cast<T>(0), "abs(0) should be 0");

        // Test edge case: most negative value
        static_assert(cxp::abs::f(cxp::minValue<T> + 1) == -(cxp::minValue<T> + 1), 
                      "abs(min) should be max for signed types");
    } else if constexpr (std::is_signed_v<T> && sizeof(T) < 4) {
        static_assert(cxp::abs::f(static_cast<T>(5)) == static_cast<int>(5), "abs(5) should be 5");
        static_assert(cxp::abs::f(static_cast<T>(-5)) == static_cast<int>(5), "abs(-5) should be 5");
        static_assert(cxp::abs::f(static_cast<T>(0)) == static_cast<int>(0), "abs(0) should be 0");

        // Test edge case: most negative value
        static_assert(cxp::abs::f(cxp::minValue<T>) == -static_cast<int>(cxp::minValue<T>), 
                      "abs(min) should be max for signed types");
    } else {
        // Unsigned types
        static_assert(cxp::abs::f(static_cast<T>(5)) == static_cast<T>(5), "abs(5) should be 5 for unsigned");
        static_assert(cxp::abs::f(static_cast<T>(0)) == static_cast<T>(0), "abs(0) should be 0 for unsigned");
    }
    
    return true;
}

// Test abs function at runtime
template <typename T>
bool test_abs_rt() {
    static_assert(std::is_fundamental_v<T>, "abs test only for fundamental types");
    bool allCorrect{true};
    if constexpr (std::is_signed_v<T>) {
        // Signed types
        if (cxp::abs::f(static_cast<T>(5)) != std::abs(static_cast<T>(5))) {
            std::cout << "Failed: abs(5) should be 5" << std::endl;
            allCorrect = false;
        }
        if (cxp::abs::f(static_cast<T>(-5)) != std::abs(static_cast<T>(5))) {
            std::cout << "Failed: abs(-5) should be 5" << std::endl;
            allCorrect = false;
        }
        if (cxp::abs::f(static_cast<T>(0)) != std::abs(static_cast<T>(0))) {
            std::cout << "Failed: abs(0) should be 0" << std::endl;
            allCorrect = false;
        }
        // Edge case for signed types
        // For integer signed types, abs(minValue) is undefined behavior in C++
        constexpr T extra = std::is_integral_v<T> ? static_cast<T>(1) : static_cast<T>(0);
        if (cxp::abs::f(cxp::minValue<T> + extra) != std::abs(cxp::minValue<T> + extra)) {
            using CxpType = std::decay_t<decltype(cxp::abs::f(cxp::minValue<T>))>;
            using StdType = std::decay_t<decltype(std::abs(cxp::minValue<T>))>;
            static_assert(std::is_same_v<CxpType, StdType>, 
                          "cxp::abs::f(minValue<T>) should have the same type as std::abs(cxp::minValue<T>)");
            std::cout << "Failed: abs(min) should be max for signed types" << std::endl;
            if constexpr (sizeof(T) < 4) {
                std::cout << "T= " + fk::typeToString<T>() + " Expected: " << std::abs(cxp::minValue<T> + extra) << ", got: " << static_cast<int>(cxp::abs::f(cxp::minValue<T> + extra)) << std::endl;
            } else {
                std::cout << "T= " + fk::typeToString<T>() + " Expected: " << std::abs(cxp::minValue<T> + extra) << ", got: " << cxp::abs::f(cxp::minValue<T> + extra) << std::endl;
            }
            allCorrect = false;
        }
    } else {
        // Unsigned types
        if (cxp::abs::f(static_cast<T>(5)) != std::abs(static_cast<T>(5))) {
            std::cout << "Failed: abs(5) should be 5 for unsigned" << std::endl;
            allCorrect = false;
        }
        if (cxp::abs::f(static_cast<T>(0)) != std::abs(static_cast<T>(0))) {
            std::cout << "Failed: abs(0) should be 0 for unsigned" << std::endl;
            allCorrect = false;
        }
    }

    return allCorrect;
}

// Test max function at compile-time
constexpr bool test_max_ct() {
    static_assert(cxp::max::f(5) == 5, "max(5) should be 5");
    static_assert(cxp::max::f(3, 5) == 5, "max(3, 5) should be 5");
    static_assert(cxp::max::f(5, 3) == 5, "max(5, 3) should be 5");
    static_assert(cxp::max::f(1, 3, 5, 2) == 5, "max(1, 3, 5, 2) should be 5");
    static_assert(cxp::max::f(-1, -3, -5, -2) == -1, "max(-1, -3, -5, -2) should be -1");
    static_assert(cxp::max::f(1.0, 3.0, 5.0, 2.0) == 5.0, "max(1.0, 3.0, 5.0, 2.0) should be 5.0");
    return true;
}

// Test max function at runtime
bool test_max_rt() {
    bool allCorrect{true};
    // Test with runtime values
    if (cxp::max::f(3, 5) != std::max(3, 5)) {
        std::cout << "Failed: max(3, 5) should be 5" << std::endl;
        allCorrect = false;
    }
    if (cxp::max::f(5, 3) != std::max(5, 3)) {
        std::cout << "Failed: max(5, 3) should be 5" << std::endl;
        allCorrect = false;
    }
    if (cxp::max::f(1, 3, 5, 2) != std::max(std::max(1, 3), std::max(5, 2))) {
        std::cout << "Failed: max(1, 3, 5, 2) should be 5" << std::endl;
        allCorrect = false;
    }
    if (cxp::max::f(-1, -3, -5, -2) != std::max(std::max(-1, -3), std::max(-5, -2))) {
        std::cout << "Failed: max(-1, -3, -5, -2) should be -1" << std::endl;
        allCorrect = false;
    }
    if (cxp::max::f(1.0, 3.0, 5.0, 2.0) != std::max(std::max(1.0, 3.0), std::max(5.0, 2.0))) {
        std::cout << "Failed: max(1.0, 3.0, 5.0, 2.0) should be 5.0" << std::endl;
        allCorrect = false;
    }

    return allCorrect;
}

// Test min function at compile-time
constexpr bool test_min_ct() {
    static_assert(cxp::min::f(5) == 5, "min(5) should be 5");
    
    // Test with two arguments to see if the bug manifests
    // The bug in min_helper calls max_helper instead of min_helper recursively
    static_assert(cxp::min::f(3, 5) == 3, "min(3, 5) should be 3");
    static_assert(cxp::min::f(5, 3) == 3, "min(5, 3) should be 3");
    static_assert(cxp::min::f(1, 3, 5, 2) == 1, "min(1, 3, 5, 2) should be 1");
    static_assert(cxp::min::f(-1, -3, -5, -2) == -5, "min(-1, -3, -5, -2) should be -5");
    static_assert(cxp::min::f(1.0, 3.0, 5.0, 2.0) == 1.0, "min(1.0, 3.0, 5.0, 2.0) should be 1.0");
    
    return true;
}
// Test min function at runtime
bool test_min_rt() {
    bool allCorrect{true};
    // Test with runtime values
    if (cxp::min::f(3, 5) != std::min(3, 5)) {
        std::cout << "Failed: min(3, 5) should be 3" << std::endl;
        allCorrect = false;
    }
    if (cxp::min::f(5, 3) != std::min(5, 3)) {
        std::cout << "Failed: min(5, 3) should be 3" << std::endl;
        allCorrect = false;
    }
    if (cxp::min::f(1, 3, 5, 2) != std::min(std::min(1, 3), std::min(5, 2))) {
        std::cout << "Failed: min(1, 3, 5, 2) should be 1" << std::endl;
        allCorrect = false;
    }
    if (cxp::min::f(-1, -3, -5, -2) != std::min(std::min(-1, -3), std::min(-5, -2))) {
        std::cout << "Failed: min(-1, -3, -5, -2) should be -5" << std::endl;
        allCorrect = false;
    }
    if (cxp::min::f(1.0, 3.0, 5.0, 2.0) != std::min(std::min(1.0, 3.0), std::min(5.0, 2.0))) {
        std::cout << "Failed: min(1.0, 3.0, 5.0, 2.0) should be 1.0" << std::endl;
        allCorrect = false;
    }

    return allCorrect;
}

// Test floor function at compile-time
template <typename T>
constexpr bool test_floor_ct() {
    static_assert(std::is_floating_point_v<T>, "floor test only for floating point types");

    // Basic positive values
    static_assert(cxp::floor::f(static_cast<T>(3.7)) == static_cast<T>(3.0), "floor(3.7) should be 3.0");
    static_assert(cxp::floor::f(static_cast<T>(3.2)) == static_cast<T>(3.0), "floor(3.2) should be 3.0");
    static_assert(cxp::floor::f(static_cast<T>(3.0)) == static_cast<T>(3.0), "floor(3.0) should be 3.0");
    static_assert(cxp::floor::f(static_cast<T>(0.9)) == static_cast<T>(0.0), "floor(0.9) should be 0.0");
    static_assert(cxp::floor::f(static_cast<T>(0.1)) == static_cast<T>(0.0), "floor(0.1) should be 0.0");
    
    // Basic negative values
    static_assert(cxp::floor::f(static_cast<T>(-3.2)) == static_cast<T>(-4.0), "floor(-3.2) should be -4.0");
    static_assert(cxp::floor::f(static_cast<T>(-3.7)) == static_cast<T>(-4.0), "floor(-3.7) should be -4.0");
    static_assert(cxp::floor::f(static_cast<T>(-3.0)) == static_cast<T>(-3.0), "floor(-3.0) should be -3.0");
    static_assert(cxp::floor::f(static_cast<T>(-0.1)) == static_cast<T>(-1.0), "floor(-0.1) should be -1.0");
    static_assert(cxp::floor::f(static_cast<T>(-0.9)) == static_cast<T>(-1.0), "floor(-0.9) should be -1.0");
    
    // Zero values
    static_assert(cxp::floor::f(static_cast<T>(0.0)) == static_cast<T>(0.0), "floor(0.0) should be 0.0");
    static_assert(cxp::floor::f(static_cast<T>(-0.0)) == static_cast<T>(-0.0), "floor(-0.0) should be -0.0");
    
    // Values near integer boundaries
    static_assert(cxp::floor::f(static_cast<T>(1.0)) == static_cast<T>(1.0), "floor(1.0) should be 1.0");
    static_assert(cxp::floor::f(static_cast<T>(-1.0)) == static_cast<T>(-1.0), "floor(-1.0) should be -1.0");
    static_assert(cxp::floor::f(static_cast<T>(1.9999)) == static_cast<T>(1.0), "floor(1.9999) should be 1.0");
    static_assert(cxp::floor::f(static_cast<T>(-1.0001)) == static_cast<T>(-2.0), "floor(-1.0001) should be -2.0");
    
    // Large values
    static_assert(cxp::floor::f(static_cast<T>(1000.7)) == static_cast<T>(1000.0), "floor(1000.7) should be 1000.0");
    static_assert(cxp::floor::f(static_cast<T>(-1000.3)) == static_cast<T>(-1001.0), "floor(-1000.3) should be -1001.0");
    
    // Special values compile-time
    static_assert(cxp::isnan::f(cxp::floor::f(std::numeric_limits<T>::quiet_NaN())), "floor(NaN) should be NaN");
    static_assert(cxp::isinf::f(cxp::floor::f(std::numeric_limits<T>::infinity())), "floor(inf) should be inf");
    static_assert(cxp::isinf::f(cxp::floor::f(-std::numeric_limits<T>::infinity())), "floor(-inf) should be -inf");

    // Edge cases for different floating point precisions
    if constexpr (std::is_same_v<T, float>) {
        // Float-specific tests
        static_assert(cxp::floor::f(16777215.9f) == 16777216.0f, "floor(16777215.9f) should be 16777216.0f");
        static_assert(cxp::floor::f(-16777215.1f) == -16777215.0f, "floor(-16777215.1f) should be -16777215.0f");
    } else if constexpr (std::is_same_v<T, double>) {
        // Double-specific tests
        static_assert(cxp::floor::f(9007199254740991.9) == 9007199254740992.0, "floor(9007199254740991.9) should be 9007199254740992.0");
        static_assert(cxp::floor::f(-9007199254740991.1) == -9007199254740991.0, "floor(-9007199254740991.1) should be -9007199254740991.0");
    }
    
    return true;
}

// Test floor function at runtime
template <typename T>
bool test_floor_rt() {
    static_assert(std::is_floating_point_v<T>, "floor test only for floating point types");
    
    bool allCorrect{true};
    
    // Basic positive values
    if (cxp::floor::f(static_cast<T>(3.7)) != std::floor(static_cast<T>(3.7))) {
        std::cout << "Failed: cxp::floor::f(3.7) should match std::floor(3.7)" << std::endl;
        allCorrect = false;
    }
    if (cxp::floor::f(static_cast<T>(3.2)) != std::floor(static_cast<T>(3.2))) {
        std::cout << "Failed: cxp::floor::f(3.2) should match std::floor(3.2)" << std::endl;
        allCorrect = false;
    }
    if (cxp::floor::f(static_cast<T>(3.0)) != std::floor(static_cast<T>(3.0))) {
        std::cout << "Failed: cxp::floor::f(3.0) should match std::floor(3.0)" << std::endl;
        allCorrect = false;
    }
    if (cxp::floor::f(static_cast<T>(0.9)) != std::floor(static_cast<T>(0.9))) {
        std::cout << "Failed: cxp::floor::f(0.9) should match std::floor(0.9)" << std::endl;
        allCorrect = false;
    }
    if (cxp::floor::f(static_cast<T>(0.1)) != std::floor(static_cast<T>(0.1))) {
        std::cout << "Failed: cxp::floor::f(0.1) should match std::floor(0.1)" << std::endl;
        allCorrect = false;
    }
    
    // Basic negative values
    if (cxp::floor::f(static_cast<T>(-3.2)) != std::floor(static_cast<T>(-3.2))) {
        std::cout << "Failed: cxp::floor::f(-3.2) should match std::floor(-3.2)" << std::endl;
        allCorrect = false;
    }
    if (cxp::floor::f(static_cast<T>(-3.7)) != std::floor(static_cast<T>(-3.7))) {
        std::cout << "Failed: cxp::floor::f(-3.7) should match std::floor(-3.7)" << std::endl;
        allCorrect = false;
    }
    if (cxp::floor::f(static_cast<T>(-3.0)) != std::floor(static_cast<T>(-3.0))) {
        std::cout << "Failed: cxp::floor::f(-3.0) should match std::floor(-3.0)" << std::endl;
        allCorrect = false;
    }
    if (cxp::floor::f(static_cast<T>(-0.1)) != std::floor(static_cast<T>(-0.1))) {
        std::cout << "Failed: cxp::floor::f(-0.1) should match std::floor(-0.1)" << std::endl;
        allCorrect = false;
    }
    if (cxp::floor::f(static_cast<T>(-0.9)) != std::floor(static_cast<T>(-0.9))) {
        std::cout << "Failed: cxp::floor::f(-0.9) should match std::floor(-0.9)" << std::endl;
        allCorrect = false;
    }
    
    // Zero values
    if (cxp::floor::f(static_cast<T>(0.0)) != std::floor(static_cast<T>(0.0))) {
        std::cout << "Failed: cxp::floor::f(0.0) should match std::floor(0.0)" << std::endl;
        allCorrect = false;
    }
    if (cxp::floor::f(static_cast<T>(-0.0)) != std::floor(static_cast<T>(-0.0))) {
        std::cout << "Failed: cxp::floor::f(-0.0) should match std::floor(-0.0)" << std::endl;
        allCorrect = false;
    }
    
    // Values near integer boundaries
    if (cxp::floor::f(static_cast<T>(1.0)) != std::floor(static_cast<T>(1.0))) {
        std::cout << "Failed: cxp::floor::f(1.0) should match std::floor(1.0)" << std::endl;
        allCorrect = false;
    }
    if (cxp::floor::f(static_cast<T>(-1.0)) != std::floor(static_cast<T>(-1.0))) {
        std::cout << "Failed: cxp::floor::f(-1.0) should match std::floor(-1.0)" << std::endl;
        allCorrect = false;
    }
    if (cxp::floor::f(static_cast<T>(1.9999)) != std::floor(static_cast<T>(1.9999))) {
        std::cout << "Failed: cxp::floor::f(1.9999) should match std::floor(1.9999)" << std::endl;
        allCorrect = false;
    }
    if (cxp::floor::f(static_cast<T>(-1.0001)) != std::floor(static_cast<T>(-1.0001))) {
        std::cout << "Failed: cxp::floor::f(-1.0001) should match std::floor(-1.0001)" << std::endl;
        allCorrect = false;
    }
    
    // Large values
    if (cxp::floor::f(static_cast<T>(1000.7)) != std::floor(static_cast<T>(1000.7))) {
        std::cout << "Failed: cxp::floor::f(1000.7) should match std::floor(1000.7)" << std::endl;
        allCorrect = false;
    }
    if (cxp::floor::f(static_cast<T>(-1000.3)) != std::floor(static_cast<T>(-1000.3))) {
        std::cout << "Failed: cxp::floor::f(-1000.3) should match std::floor(-1000.3)" << std::endl;
        allCorrect = false;
    }
    
    // Special values runtime
    if (!std::isnan(cxp::floor::f(std::numeric_limits<T>::quiet_NaN()))) {
        std::cout << "Failed: cxp::floor::f(NaN) should be NaN" << std::endl;
        allCorrect = false;
    }
    if (!std::isinf(cxp::floor::f(std::numeric_limits<T>::infinity()))) {
        std::cout << "Failed: cxp::floor::f(inf) should be inf" << std::endl;
        allCorrect = false;
    }
    if (!std::isinf(cxp::floor::f(-std::numeric_limits<T>::infinity()))) {
        std::cout << "Failed: cxp::floor::f(-inf) should be -inf" << std::endl;
        allCorrect = false;
    }
    
    // Verify special value signs are preserved
    T result_pos_inf = cxp::floor::f(std::numeric_limits<T>::infinity());
    T result_neg_inf = cxp::floor::f(-std::numeric_limits<T>::infinity());
    if (result_pos_inf < 0) {
        std::cout << "Failed: cxp::floor::f(+inf) should be positive infinity" << std::endl;
        allCorrect = false;
    }
    if (result_neg_inf > 0) {
        std::cout << "Failed: cxp::floor::f(-inf) should be negative infinity" << std::endl;
        allCorrect = false;
    }
    
    // Edge cases for different floating point precisions
    if constexpr (std::is_same_v<T, float>) {
        // Float-specific runtime tests
        if (cxp::floor::f(16777215.9f) != std::floor(16777215.9f)) {
            std::cout << "Failed: cxp::floor::f(16777215.9f) should match std::floor(16777215.9f)" << std::endl;
            allCorrect = false;
        }
    } else if constexpr (std::is_same_v<T, double>) {
        // Double-specific runtime tests
        if (cxp::floor::f(9007199254740991.9) != std::floor(9007199254740991.9)) {
            std::cout << "Failed: cxp::floor::f(9007199254740991.9) should match std::floor(9007199254740991.9)" << std::endl;
            allCorrect = false;
        }
    }
    
    // Stress test with many small values
    for (int i = -100; i <= 100; ++i) {
        T val = static_cast<T>(i) + static_cast<T>(0.1);
        if (cxp::floor::f(val) != std::floor(val)) {
            std::cout << "Failed: cxp::floor::f(" << val << ") should match std::floor(" << val << ")" << std::endl;
            allCorrect = false;
            break;
        }
        
        val = static_cast<T>(i) + static_cast<T>(0.9);
        if (cxp::floor::f(val) != std::floor(val)) {
            std::cout << "Failed: cxp::floor::f(" << val << ") should match std::floor(" << val << ")" << std::endl;
            allCorrect = false;
            break;
        }
    }
    
    return allCorrect;
}

// Test nearbyint function at compile-time
template <typename T>
constexpr bool test_nearbyint_ct() {
    static_assert(std::is_floating_point_v<T>, "nearbyint test only for floating point types");

    // Basic positive values
    static_assert(cxp::nearbyint::f(static_cast<T>(3.2)) == static_cast<T>(3.0), "nearbyint(3.2) should be 3.0");
    static_assert(cxp::nearbyint::f(static_cast<T>(3.7)) == static_cast<T>(4.0), "nearbyint(3.7) should be 4.0");
    static_assert(cxp::nearbyint::f(static_cast<T>(3.0)) == static_cast<T>(3.0), "nearbyint(3.0) should be 3.0");
    static_assert(cxp::nearbyint::f(static_cast<T>(0.1)) == static_cast<T>(0.0), "nearbyint(0.1) should be 0.0");
    static_assert(cxp::nearbyint::f(static_cast<T>(0.9)) == static_cast<T>(1.0), "nearbyint(0.9) should be 1.0");

    // Basic negative values
    static_assert(cxp::nearbyint::f(static_cast<T>(-3.2)) == static_cast<T>(-3.0), "nearbyint(-3.2) should be -3.0");
    static_assert(cxp::nearbyint::f(static_cast<T>(-3.7)) == static_cast<T>(-4.0), "nearbyint(-3.7) should be -4.0");
    static_assert(cxp::nearbyint::f(static_cast<T>(-3.0)) == static_cast<T>(-3.0), "nearbyint(-3.0) should be -3.0");
    static_assert(cxp::nearbyint::f(static_cast<T>(-0.1)) == static_cast<T>(0.0), "nearbyint(-0.1) should be 0.0");
    static_assert(cxp::nearbyint::f(static_cast<T>(-0.9)) == static_cast<T>(-1.0), "nearbyint(-0.9) should be -1.0");

    // Zero values
    static_assert(cxp::nearbyint::f(static_cast<T>(0.0)) == static_cast<T>(0.0), "nearbyint(0.0) should be 0.0");
    static_assert(cxp::nearbyint::f(static_cast<T>(-0.0)) == static_cast<T>(-0.0), "nearbyint(-0.0) should be -0.0");

    // Banker's rounding (round half to even) - positive values
    static_assert(cxp::nearbyint::f(static_cast<T>(0.5)) == static_cast<T>(0.0), "nearbyint(0.5) should be 0.0 (round to even)");
    static_assert(cxp::nearbyint::f(static_cast<T>(1.5)) == static_cast<T>(2.0), "nearbyint(1.5) should be 2.0 (round to even)");
    static_assert(cxp::nearbyint::f(static_cast<T>(2.5)) == static_cast<T>(2.0), "nearbyint(2.5) should be 2.0 (round to even)");
    static_assert(cxp::nearbyint::f(static_cast<T>(3.5)) == static_cast<T>(4.0), "nearbyint(3.5) should be 4.0 (round to even)");
    static_assert(cxp::nearbyint::f(static_cast<T>(4.5)) == static_cast<T>(4.0), "nearbyint(4.5) should be 4.0 (round to even)");

    // Banker's rounding (round half to even) - negative values
    static_assert(cxp::nearbyint::f(static_cast<T>(-0.5)) == static_cast<T>(0.0), "nearbyint(-0.5) should be 0.0 (round to even)");
    static_assert(cxp::nearbyint::f(static_cast<T>(-1.5)) == static_cast<T>(-2.0), "nearbyint(-1.5) should be -2.0 (round to even)");
    static_assert(cxp::nearbyint::f(static_cast<T>(-2.5)) == static_cast<T>(-2.0), "nearbyint(-2.5) should be -2.0 (round to even)");
    static_assert(cxp::nearbyint::f(static_cast<T>(-3.5)) == static_cast<T>(-4.0), "nearbyint(-3.5) should be -4.0 (round to even)");
    static_assert(cxp::nearbyint::f(static_cast<T>(-4.5)) == static_cast<T>(-4.0), "nearbyint(-4.5) should be -4.0 (round to even)");

    // Values near integer boundaries
    static_assert(cxp::nearbyint::f(static_cast<T>(1.0)) == static_cast<T>(1.0), "nearbyint(1.0) should be 1.0");
    static_assert(cxp::nearbyint::f(static_cast<T>(-1.0)) == static_cast<T>(-1.0), "nearbyint(-1.0) should be -1.0");
    static_assert(cxp::nearbyint::f(static_cast<T>(1.4999)) == static_cast<T>(1.0), "nearbyint(1.4999) should be 1.0");
    static_assert(cxp::nearbyint::f(static_cast<T>(1.5001)) == static_cast<T>(2.0), "nearbyint(1.5001) should be 2.0");
    static_assert(cxp::nearbyint::f(static_cast<T>(-1.4999)) == static_cast<T>(-1.0), "nearbyint(-1.4999) should be -1.0");
    static_assert(cxp::nearbyint::f(static_cast<T>(-1.5001)) == static_cast<T>(-2.0), "nearbyint(-1.5001) should be -2.0");

    // Large values
    static_assert(cxp::nearbyint::f(static_cast<T>(1000.2)) == static_cast<T>(1000.0), "nearbyint(1000.2) should be 1000.0");
    static_assert(cxp::nearbyint::f(static_cast<T>(1000.7)) == static_cast<T>(1001.0), "nearbyint(1000.7) should be 1001.0");
    static_assert(cxp::nearbyint::f(static_cast<T>(-1000.3)) == static_cast<T>(-1000.0), "nearbyint(-1000.3) should be -1000.0");
    static_assert(cxp::nearbyint::f(static_cast<T>(-1000.8)) == static_cast<T>(-1001.0), "nearbyint(-1000.8) should be -1001.0");

    // Special values compile-time
    static_assert(cxp::isnan::f(cxp::nearbyint::f(std::numeric_limits<T>::quiet_NaN())), "nearbyint(NaN) should be NaN");
    static_assert(cxp::isinf::f(cxp::nearbyint::f(std::numeric_limits<T>::infinity())), "nearbyint(inf) should be inf");
    static_assert(cxp::isinf::f(cxp::nearbyint::f(-std::numeric_limits<T>::infinity())), "nearbyint(-inf) should be -inf");

    return true;
}

// Test nearbyint function at runtime
template <typename T>
bool test_nearbyint_rt() {
    static_assert(std::is_floating_point_v<T>, "nearbyint test only for floating point types");

    bool allCorrect{ true };

    // Basic positive values
    if (cxp::nearbyint::f(static_cast<T>(3.2)) != std::nearbyint(static_cast<T>(3.2))) {
        std::cout << "Failed: cxp::nearbyint::f(3.2) should match std::nearbyint(3.2)" << std::endl;
        allCorrect = false;
    }
    if (cxp::nearbyint::f(static_cast<T>(3.7)) != std::nearbyint(static_cast<T>(3.7))) {
        std::cout << "Failed: cxp::nearbyint::f(3.7) should match std::nearbyint(3.7)" << std::endl;
        allCorrect = false;
    }
    if (cxp::nearbyint::f(static_cast<T>(3.0)) != std::nearbyint(static_cast<T>(3.0))) {
        std::cout << "Failed: cxp::nearbyint::f(3.0) should match std::nearbyint(3.0)" << std::endl;
        allCorrect = false;
    }
    if (cxp::nearbyint::f(static_cast<T>(0.1)) != std::nearbyint(static_cast<T>(0.1))) {
        std::cout << "Failed: cxp::nearbyint::f(0.1) should match std::nearbyint(0.1)" << std::endl;
        allCorrect = false;
    }
    if (cxp::nearbyint::f(static_cast<T>(0.9)) != std::nearbyint(static_cast<T>(0.9))) {
        std::cout << "Failed: cxp::nearbyint::f(0.9) should match std::nearbyint(0.9)" << std::endl;
        allCorrect = false;
    }

    // Basic negative values
    if (cxp::nearbyint::f(static_cast<T>(-3.2)) != std::nearbyint(static_cast<T>(-3.2))) {
        std::cout << "Failed: cxp::nearbyint::f(-3.2) should match std::nearbyint(-3.2)" << std::endl;
        allCorrect = false;
    }
    if (cxp::nearbyint::f(static_cast<T>(-3.7)) != std::nearbyint(static_cast<T>(-3.7))) {
        std::cout << "Failed: cxp::nearbyint::f(-3.7) should match std::nearbyint(-3.7)" << std::endl;
        allCorrect = false;
    }
    if (cxp::nearbyint::f(static_cast<T>(-3.0)) != std::nearbyint(static_cast<T>(-3.0))) {
        std::cout << "Failed: cxp::nearbyint::f(-3.0) should match std::nearbyint(-3.0)" << std::endl;
        allCorrect = false;
    }
    if (cxp::nearbyint::f(static_cast<T>(-0.1)) != std::nearbyint(static_cast<T>(-0.1))) {
        std::cout << "Failed: cxp::nearbyint::f(-0.1) should match std::nearbyint(-0.1)" << std::endl;
        allCorrect = false;
    }
    if (cxp::nearbyint::f(static_cast<T>(-0.9)) != std::nearbyint(static_cast<T>(-0.9))) {
        std::cout << "Failed: cxp::nearbyint::f(-0.9) should match std::nearbyint(-0.9)" << std::endl;
        allCorrect = false;
    }

    // Zero values
    if (cxp::nearbyint::f(static_cast<T>(0.0)) != std::nearbyint(static_cast<T>(0.0))) {
        std::cout << "Failed: cxp::nearbyint::f(0.0) should match std::nearbyint(0.0)" << std::endl;
        allCorrect = false;
    }
    if (cxp::nearbyint::f(static_cast<T>(-0.0)) != std::nearbyint(static_cast<T>(-0.0))) {
        std::cout << "Failed: cxp::nearbyint::f(-0.0) should match std::nearbyint(-0.0)" << std::endl;
        allCorrect = false;
    }

    // Banker's rounding (round half to even) - positive values
    if (cxp::nearbyint::f(static_cast<T>(0.5)) != std::nearbyint(static_cast<T>(0.5))) {
        std::cout << "Failed: cxp::nearbyint::f(0.5) should match std::nearbyint(0.5)" << std::endl;
        allCorrect = false;
    }
    if (cxp::nearbyint::f(static_cast<T>(1.5)) != std::nearbyint(static_cast<T>(1.5))) {
        std::cout << "Failed: cxp::nearbyint::f(1.5) should match std::nearbyint(1.5)" << std::endl;
        allCorrect = false;
    }
    if (cxp::nearbyint::f(static_cast<T>(2.5)) != std::nearbyint(static_cast<T>(2.5))) {
        std::cout << "Failed: cxp::nearbyint::f(2.5) should match std::nearbyint(2.5)" << std::endl;
        allCorrect = false;
    }
    if (cxp::nearbyint::f(static_cast<T>(3.5)) != std::nearbyint(static_cast<T>(3.5))) {
        std::cout << "Failed: cxp::nearbyint::f(3.5) should match std::nearbyint(3.5)" << std::endl;
        allCorrect = false;
    }
    if (cxp::nearbyint::f(static_cast<T>(4.5)) != std::nearbyint(static_cast<T>(4.5))) {
        std::cout << "Failed: cxp::nearbyint::f(4.5) should match std::nearbyint(4.5)" << std::endl;
        allCorrect = false;
    }

    // Banker's rounding (round half to even) - negative values
    if (cxp::nearbyint::f(static_cast<T>(-0.5)) != std::nearbyint(static_cast<T>(-0.5))) {
        std::cout << "Failed: cxp::nearbyint::f(-0.5) should match std::nearbyint(-0.5)" << std::endl;
        allCorrect = false;
    }
    if (cxp::nearbyint::f(static_cast<T>(-1.5)) != std::nearbyint(static_cast<T>(-1.5))) {
        std::cout << "Failed: cxp::nearbyint::f(-1.5) should match std::nearbyint(-1.5)" << std::endl;
        allCorrect = false;
    }
    if (cxp::nearbyint::f(static_cast<T>(-2.5)) != std::nearbyint(static_cast<T>(-2.5))) {
        std::cout << "Failed: cxp::nearbyint::f(-2.5) should match std::nearbyint(-2.5)" << std::endl;
        allCorrect = false;
    }
    if (cxp::nearbyint::f(static_cast<T>(-3.5)) != std::nearbyint(static_cast<T>(-3.5))) {
        std::cout << "Failed: cxp::nearbyint::f(-3.5) should match std::nearbyint(-3.5)" << std::endl;
        allCorrect = false;
    }
    if (cxp::nearbyint::f(static_cast<T>(-4.5)) != std::nearbyint(static_cast<T>(-4.5))) {
        std::cout << "Failed: cxp::nearbyint::f(-4.5) should match std::nearbyint(-4.5)" << std::endl;
        allCorrect = false;
    }

    // Values near integer boundaries
    if (cxp::nearbyint::f(static_cast<T>(1.0)) != std::nearbyint(static_cast<T>(1.0))) {
        std::cout << "Failed: cxp::nearbyint::f(1.0) should match std::nearbyint(1.0)" << std::endl;
        allCorrect = false;
    }
    if (cxp::nearbyint::f(static_cast<T>(-1.0)) != std::nearbyint(static_cast<T>(-1.0))) {
        std::cout << "Failed: cxp::nearbyint::f(-1.0) should match std::nearbyint(-1.0)" << std::endl;
        allCorrect = false;
    }
    if (cxp::nearbyint::f(static_cast<T>(1.4999)) != std::nearbyint(static_cast<T>(1.4999))) {
        std::cout << "Failed: cxp::nearbyint::f(1.4999) should match std::nearbyint(1.4999)" << std::endl;
        allCorrect = false;
    }
    if (cxp::nearbyint::f(static_cast<T>(1.5001)) != std::nearbyint(static_cast<T>(1.5001))) {
        std::cout << "Failed: cxp::nearbyint::f(1.5001) should match std::nearbyint(1.5001)" << std::endl;
        allCorrect = false;
    }

    // Large values
    if (cxp::nearbyint::f(static_cast<T>(1000.2)) != std::nearbyint(static_cast<T>(1000.2))) {
        std::cout << "Failed: cxp::nearbyint::f(1000.2) should match std::nearbyint(1000.2)" << std::endl;
        allCorrect = false;
    }
    if (cxp::nearbyint::f(static_cast<T>(1000.7)) != std::nearbyint(static_cast<T>(1000.7))) {
        std::cout << "Failed: cxp::nearbyint::f(1000.7) should match std::nearbyint(1000.7)" << std::endl;
        allCorrect = false;
    }
    if (cxp::nearbyint::f(static_cast<T>(-1000.3)) != std::nearbyint(static_cast<T>(-1000.3))) {
        std::cout << "Failed: cxp::nearbyint::f(-1000.3) should match std::nearbyint(-1000.3)" << std::endl;
        allCorrect = false;
    }
    if (cxp::nearbyint::f(static_cast<T>(-1000.8)) != std::nearbyint(static_cast<T>(-1000.8))) {
        std::cout << "Failed: cxp::nearbyint::f(-1000.8) should match std::nearbyint(-1000.8)" << std::endl;
        allCorrect = false;
    }

    // Special values runtime
    if (!std::isnan(cxp::nearbyint::f(std::numeric_limits<T>::quiet_NaN()))) {
        std::cout << "Failed: cxp::nearbyint::f(NaN) should be NaN" << std::endl;
        allCorrect = false;
    }
    if (!std::isinf(cxp::nearbyint::f(std::numeric_limits<T>::infinity()))) {
        std::cout << "Failed: cxp::nearbyint::f(inf) should be inf" << std::endl;
        allCorrect = false;
    }
    if (!std::isinf(cxp::nearbyint::f(-std::numeric_limits<T>::infinity()))) {
        std::cout << "Failed: cxp::nearbyint::f(-inf) should be -inf" << std::endl;
        allCorrect = false;
    }

    // Verify special value signs are preserved
    T result_pos_inf = cxp::nearbyint::f(std::numeric_limits<T>::infinity());
    T result_neg_inf = cxp::nearbyint::f(-std::numeric_limits<T>::infinity());
    if (result_pos_inf < 0) {
        std::cout << "Failed: cxp::nearbyint::f(+inf) should be positive infinity" << std::endl;
        allCorrect = false;
    }
    if (result_neg_inf > 0) {
        std::cout << "Failed: cxp::nearbyint::f(-inf) should be negative infinity" << std::endl;
        allCorrect = false;
    }

    // Stress test with many values around half-integers to verify banker's rounding
    for (int i = -10; i <= 10; ++i) {
        T val = static_cast<T>(i) + static_cast<T>(0.5);
        if (cxp::nearbyint::f(val) != std::nearbyint(val)) {
            std::cout << "Failed: cxp::nearbyint::f(" << val << ") should match std::nearbyint(" << val << ")" << std::endl;
            allCorrect = false;
            break;
        }
    }

    return allCorrect;
}

// ============================================================================
// Bitwise and ULP Helpers for complex transcendental and bitwise functions
// ============================================================================
inline uint32_t get_float_bits(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));
    return bits;
}

inline int32_t get_ulp_distance(float a, float b) {
    if (std::isnan(a) && std::isnan(b))
        return 0;
    if (std::isinf(a) && std::isinf(b) && (std::signbit(a) == std::signbit(b)))
        return 0;
    if (std::isnan(a) || std::isnan(b) || std::isinf(a) || std::isinf(b))
        return -1;

    uint32_t ua = get_float_bits(a);
    uint32_t ub = get_float_bits(b);

    if ((ua >> 31) != (ub >> 31)) {
        if (a == 0.0f && b == 0.0f)
            return 0;
        return -1; // Sign mismatch
    }

    int32_t ia = static_cast<int32_t>(ua & 0x7FFFFFFF);
    int32_t ib = static_cast<int32_t>(ub & 0x7FFFFFFF);
    return std::abs(ia - ib);
}

// ============================================================================
// Tests for custom implementations: ldexpf and expf
// ============================================================================

bool test_ldexpf_rt() {
    bool allCorrect{true};

    auto check = [&](float x, int exp) {
        float expected = std::ldexp(x, exp);
        float actual = cxp::ldexpf::f(x, exp);
        if (std::isnan(expected) && std::isnan(actual))
            return;
        if (get_float_bits(expected) != get_float_bits(actual)) {
            std::cout << "Failed: cxp::ldexpf::f(" << x << ", " << exp << ") expected bits 0x" << std::hex
                      << get_float_bits(expected) << ", got 0x" << get_float_bits(actual) << std::dec << std::endl;
            allCorrect = false;
        }
    };

    check(0.0f, 5);
    check(-0.0f, -10);
    check(std::numeric_limits<float>::infinity(), 10);
    check(-std::numeric_limits<float>::infinity(), -5);
    check(std::numeric_limits<float>::quiet_NaN(), 4);
    check(1.5f, 3);
    check(128.0f, -4);
    check(1.0f, 200);
    check(1.0f, -200);
    check(1.0f, -130);
    check(1.4013e-45f, 10);
    check(1.4013e-45f, -5);

    return allCorrect;
}

bool test_expf_rt() {
    bool allCorrect{true};

    // Edge cases check
    auto check_special = [&](float x, float expected) {
        float actual = cxp::expf::f(x);
        bool match = false;
        if (std::isnan(expected) && std::isnan(actual))
            match = true;
        else if (std::isinf(expected) && std::isinf(actual) && (std::signbit(expected) == std::signbit(actual)))
            match = true;
        else if (expected == actual)
            match = true;

        if (!match) {
            std::cout << std::setprecision(std::numeric_limits<float>::max_digits10) << "Failed: cxp::expf::f(" << x
                      << ") should be " << expected << " but got " << actual << " [Bits Exp: 0x" << std::hex
                      << get_float_bits(expected) << " | Act: 0x" << get_float_bits(actual) << std::dec << "]\n";
            allCorrect = false;
        }
    };

    check_special(0.0f, 1.0f);
    check_special(-0.0f, 1.0f);
    check_special(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity());
    check_special(-std::numeric_limits<float>::infinity(), 0.0f);
    check_special(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN());
    check_special(89.0f, std::numeric_limits<float>::infinity());
    check_special(-105.0f, 0.0f);

    // Deterministic ULP Precision sweep over valid float domain [-103.0f, 88.0f]
    int32_t max_ulp_error = 0;
    const int32_t NORMAL_ULP_THRESHOLD = 3;
    const int32_t SUBNORMAL_ULP_THRESHOLD = 5; // Relaxed for mantissa precision loss

    float step = (88.0f - (-103.0f)) / 10000.0f; // 10,000 deterministic points
    for (float x = -103.0f; x <= 88.0f; x += step) {
        float expected = std::exp(x); // std::exp does double-precision internally
        float actual = cxp::expf::f(x);

        int32_t ulp = get_ulp_distance(expected, actual);
        if (ulp < 0) {
            std::cout << "Failed: cxp::expf::f(" << x << ") sign/domain mismatch. Expected " << expected << ", got "
                      << actual << std::endl;
            allCorrect = false;
            break;
        }

        if (ulp > max_ulp_error)
            max_ulp_error = ulp;

        // Subnormal numbers begin when x falls below ~ -87.33f
        int32_t current_threshold = (x < -87.3f) ? SUBNORMAL_ULP_THRESHOLD : NORMAL_ULP_THRESHOLD;

        if (ulp > current_threshold) {
            std::cout << std::setprecision(std::numeric_limits<float>::max_digits10) << "Failed: cxp::expf::f(" << x
                      << ") exceeded ULP threshold of " << current_threshold << ".\n  Expected : " << expected
                      << "\n  Actual   : " << actual << "\n  ULP dist : " << ulp << "\n  Bits Exp : 0x" << std::hex
                      << get_float_bits(expected) << "\n  Bits Act : 0x" << get_float_bits(actual) << std::dec << "\n";
            allCorrect = false;
            break;
        }
    }

    if (max_ulp_error > SUBNORMAL_ULP_THRESHOLD) {
        allCorrect = false;
    }

    return allCorrect;
}

// Compile-time tests for ldexpf
constexpr bool test_ldexpf_ct() {
    // Special values
    static_assert(cxp::ldexpf::f(0.0f, 5) == 0.0f, "ldexpf(0.0f, 5) should be 0.0f");
    static_assert(cxp::ldexpf::f(-0.0f, -10) == -0.0f, "ldexpf(-0.0f, -10) should be -0.0f");
    static_assert(cxp::isinf::f(cxp::ldexpf::f(std::numeric_limits<float>::infinity(), 10)),
                  "ldexpf(inf) should be inf");
    static_assert(cxp::isinf::f(cxp::ldexpf::f(-std::numeric_limits<float>::infinity(), -5)),
                  "ldexpf(-inf) should be -inf");
    static_assert(cxp::isnan::f(cxp::ldexpf::f(std::numeric_limits<float>::quiet_NaN(), 4)),
                  "ldexpf(NaN) should be NaN");

    // Normal boundaries
    static_assert(cxp::ldexpf::f(1.5f, 3) == 12.0f, "ldexpf(1.5f, 3) should be 12.0f");
    static_assert(cxp::ldexpf::f(128.0f, -4) == 8.0f, "ldexpf(128.0f, -4) should be 8.0f");

    // Overflow / Underflow Bounds
    static_assert(cxp::isinf::f(cxp::ldexpf::f(1.0f, 200)), "ldexpf(1.0f, 200) should overflow to inf");
    static_assert(cxp::ldexpf::f(1.0f, -200) == 0.0f, "ldexpf(1.0f, -200) should underflow to 0.0f");

    return true;
}

// Compile-time tests for expf
constexpr bool test_expf_ct() {
    // Edge and Special Cases
    static_assert(cxp::expf::f(0.0f) == 1.0f, "expf(0.0f) should be 1.0f");
    static_assert(cxp::expf::f(-0.0f) == 1.0f, "expf(-0.0f) should be 1.0f");
    static_assert(cxp::isinf::f(cxp::expf::f(std::numeric_limits<float>::infinity())), "expf(inf) should be inf");
    static_assert(cxp::expf::f(-std::numeric_limits<float>::infinity()) == 0.0f, "expf(-inf) should be 0.0f");
    static_assert(cxp::isnan::f(cxp::expf::f(std::numeric_limits<float>::quiet_NaN())), "expf(NaN) should be NaN");

    // Hard architectural bounds
    static_assert(cxp::isinf::f(cxp::expf::f(89.0f)), "expf(89.0f) should hard overflow to inf");
    static_assert(cxp::expf::f(-105.0f) == 0.0f, "expf(-105.0f) should hard underflow to 0.0f");

    // Precision verification using epsilon bounds
    // e^1 ~= 2.7182818f
    static_assert(cxp::expf::f(1.0f) > 2.71828f && cxp::expf::f(1.0f) < 2.71829f, "expf(1.0f) precision error");
    // e^-1 ~= 0.3678794f
    static_assert(cxp::expf::f(-1.0f) > 0.36787f && cxp::expf::f(-1.0f) < 0.36788f, "expf(-1.0f) precision error");
    // e^2 ~= 7.389056f
    static_assert(cxp::expf::f(2.0f) > 7.38905f && cxp::expf::f(2.0f) < 7.38906f, "expf(2.0f) precision error");

    // Subnormal boundary regression test (Ensures degree-6 polynomial is intact)
    // std::exp(-88.3721771f) is approximately 4.173026e-39f
    static_assert(cxp::expf::f(-88.3721771f) > 4.173e-39f && cxp::expf::f(-88.3721771f) < 4.174e-39f, "expf subnormal regression check failed");

    return true;
}

constexpr bool test_fmaxf_ct() {
    constexpr float pos_zero = +0.0f;
    constexpr float neg_zero = -0.0f;
    constexpr float nan_val = std::numeric_limits<float>::quiet_NaN();

    // --- Scalar Tests ---
    static_assert(cxp::fmaxf::f(1.0f, 2.0f) == 2.0f, "fmaxf scalar normal failed");
    static_assert(cxp::fmaxf::f(nan_val, 5.0f) == 5.0f, "fmaxf scalar NaN failed");
    static_assert(cxp::bit_cast<uint32_t>(cxp::fmaxf::f(neg_zero, pos_zero)) == 0x00000000,
                  "fmaxf scalar signed zero failed");

    // --- Vector Tests (float3) ---
    constexpr float3 fmax_v = cxp::fmaxf::f(float3{1.0f, nan_val, neg_zero}, float3{5.0f, 3.0f, pos_zero});
    static_assert(fmax_v.x == 5.0f, "fmaxf float3.x failed (Normal)");
    static_assert(fmax_v.y == 3.0f, "fmaxf float3.y failed (NaN propagation)");
    static_assert(cxp::bit_cast<uint32_t>(fmax_v.z) == 0x00000000, "fmaxf float3.z failed (Signed Zero: MUST be +0.0)");

    // --- Vector Tests (int2 using standard max) ---
    constexpr int2 max_i = cxp::max::f(int2{100, -50}, int2{200, -10});
    static_assert(max_i.x == 200, "max int2.x failed");
    static_assert(max_i.y == -10, "max int2.y failed");

    return true;
}

bool test_fmaxf_rt() {
    bool allCorrect = true;
    float pos_zero = +0.0f;
    float neg_zero = -0.0f;
    float nan_val = std::numeric_limits<float>::quiet_NaN();

    // --- Scalar Tests ---
    if (cxp::bit_cast<uint32_t>(cxp::fmaxf::f(neg_zero, pos_zero)) != 0x00000000) {
        std::cout << "Runtime Fail: cxp::fmaxf::f(-0.0, +0.0) did not yield +0.0\n";
        allCorrect = false;
    }

    // --- Vector Tests (float3) ---
    float3 v1 = fk::make_<float3>(1.0f, nan_val, neg_zero);
    float3 v2 = fk::make_<float3>(5.0f, 3.0f, pos_zero);
    float3 v_max = cxp::fmaxf::f(v1, v2);

    if (v_max.x != 5.0f || v_max.y != 3.0f || cxp::bit_cast<uint32_t>(v_max.z) != 0x00000000) {
        std::cout << "Runtime Fail: cxp::fmaxf::f(float3, float3) component mapping failed\n";
        allCorrect = false;
    }

    // --- Vector Tests (int2 using standard max) ---
    int2 i1 = fk::make_<int2>(100, -50);
    int2 i2 = fk::make_<int2>(200, -10);
    int2 i_max = cxp::max::f(i1, i2);

    if (i_max.x != 200 || i_max.y != -10) {
        std::cout << "Runtime Fail: cxp::max::f(int2, int2) integer logic failed\n";
        allCorrect = false;
    }

    return allCorrect;
}

constexpr bool test_fminf_ct() {
    constexpr float pos_zero = +0.0f;
    constexpr float neg_zero = -0.0f;
    constexpr float nan_val = std::numeric_limits<float>::quiet_NaN();

    // --- Scalar Tests ---
    static_assert(cxp::fminf::f(1.0f, 2.0f) == 1.0f, "fminf scalar normal failed");
    static_assert(cxp::fminf::f(nan_val, 5.0f) == 5.0f, "fminf scalar NaN failed");
    static_assert(cxp::bit_cast<uint32_t>(cxp::fminf::f(neg_zero, pos_zero)) == 0x80000000,
                  "fminf scalar signed zero failed");

    // --- Vector Tests (float3) ---
    constexpr float3 fmin_v = cxp::fminf::f(float3{1.0f, nan_val, neg_zero}, float3{5.0f, 3.0f, pos_zero});
    static_assert(fmin_v.x == 1.0f, "fminf float3.x failed (Normal)");
    static_assert(fmin_v.y == 3.0f, "fminf float3.y failed (NaN propagation)");
    static_assert(cxp::bit_cast<uint32_t>(fmin_v.z) == 0x80000000, "fminf float3.z failed (Signed Zero: MUST be -0.0)");

    // --- Vector Tests (int2 using standard min) ---
    constexpr int2 min_i = cxp::min::f(int2{100, -50}, int2{200, -10});
    static_assert(min_i.x == 100, "min int2.x failed");
    static_assert(min_i.y == -50, "min int2.y failed");

    return true;
}

bool test_fminf_rt() {
    bool allCorrect = true;
    float pos_zero = +0.0f;
    float neg_zero = -0.0f;
    float nan_val = std::numeric_limits<float>::quiet_NaN();

    // --- Scalar Tests ---
    if (cxp::bit_cast<uint32_t>(cxp::fminf::f(neg_zero, pos_zero)) != 0x80000000) {
        std::cout << "Runtime Fail: cxp::fminf::f(-0.0, +0.0) did not yield -0.0\n";
        allCorrect = false;
    }

    // --- Vector Tests (float3) ---
    float3 v1 = fk::make_<float3>(1.0f, nan_val, neg_zero);
    float3 v2 = fk::make_<float3>(5.0f, 3.0f, pos_zero);
    float3 v_min = cxp::fminf::f(v1, v2);

    if (v_min.x != 1.0f || v_min.y != 3.0f || cxp::bit_cast<uint32_t>(v_min.z) != 0x80000000) {
        std::cout << "Runtime Fail: cxp::fminf::f(float3, float3) component mapping failed\n";
        allCorrect = false;
    }

    // --- Vector Tests (int2 using standard min) ---
    int2 i1 = fk::make_<int2>(100, -50);
    int2 i2 = fk::make_<int2>(200, -10);
    int2 i_min = cxp::min::f(i1, i2);

    if (i_min.x != 100 || i_min.y != -50) {
        std::cout << "Runtime Fail: cxp::min::f(int2, int2) integer logic failed\n";
        allCorrect = false;
    }

    return allCorrect;
}

// Runtime tests to complement compile-time tests
bool runtime_tests() {
    bool allCorrect{true};
    // Test round with runtime values
    allCorrect &= test_round_rt<float>();
    allCorrect &= test_round_rt<double>();

    // Test floor with runtime values
    allCorrect &= test_floor_rt<float>();
    allCorrect &= test_floor_rt<double>();

    // Test isinf with runtime values
    allCorrect &= test_isinf_rt<float>();
    allCorrect &= test_isinf_rt<double>();
    
    // Test isnan with runtime values
    allCorrect &= test_isnan_rt<float>();
    allCorrect &= test_isnan_rt<double>();
    
    // Test nearbyint with runtime values
    allCorrect &= test_nearbyint_rt<float>();
    allCorrect &= test_nearbyint_rt<double>();

    // Test comparison functions with runtime values
    if (!cxp::cmp_equal::f(5, 5)) {
        std::cout << "Failed: cxp::cmp_equal::f(5, 5) should be true" << std::endl;
        allCorrect = false;
    }
    if (cxp::cmp_equal::f(5, 4)) {
        std::cout << "Failed: cxp::cmp_equal::f(5, 4) should be false" << std::endl;
        allCorrect = false;
    }
    if (!cxp::cmp_less::f(4, 5)) {
        std::cout << "Failed: cxp::cmp_less::f(4, 5) should be true" << std::endl;
        allCorrect = false;
    }
    if (cxp::cmp_less::f(5, 4)) {
        std::cout << "Failed: cxp::cmp_less::f(5, 4) should be false" << std::endl;
        allCorrect = false;
    }

    // Test mixed type comparisons
    if (!cxp::cmp_equal::f(5u, 5)) {
        std::cout << "Failed: cxp::cmp_equal::f(5u, 5) should be true" << std::endl;
        allCorrect = false;
    }
    if (cxp::cmp_equal::f(5u, -1)) {
        std::cout << "Failed: cxp::cmp_equal::f(5u, -1) should be false" << std::endl;
        allCorrect = false;
    }
    if (!cxp::cmp_less::f(-1, 5u)) {
        std::cout << "Failed: cxp::cmp_less::f(-1, 5u) should be true" << std::endl;
        allCorrect = false;
    }
    if (cxp::cmp_less::f(5u, -1)) {
        std::cout << "Failed: cxp::cmp_less::f(5u, -1) should be false" << std::endl;
        allCorrect = false;
    }

    // Test abs with runtime values
    allCorrect &= test_abs_rt<char>();
    allCorrect &= test_abs_rt<short>();
    allCorrect &= test_abs_rt<long>();
    allCorrect &= test_abs_rt<long long>();
    allCorrect &= test_abs_rt<int>();
    allCorrect &= test_abs_rt<float>();
    allCorrect &= test_abs_rt<double>();
    
    // Test max with runtime values
    allCorrect &= test_max_rt();
    
    // Test min with runtime values
    allCorrect &= test_min_rt();

    // Test custom transcendental implementations
    allCorrect &= test_ldexpf_rt();
    allCorrect &= test_expf_rt();

    // Test fmaxf and fminf with runtime values
    allCorrect &= test_fmaxf_rt();
    allCorrect &= test_fminf_rt();
    
    return allCorrect;
}

int launch() {
    static_assert(test_round_ct<float>());
    static_assert(test_round_ct<double>());

    static_assert(test_isnan_ct<float>(), "isnan test failed for float");
    static_assert(test_isnan_ct<double>(), "isnan test failed for double");
    
    static_assert(test_isinf_ct<float>(), "isinf test failed for float");
    static_assert(test_isinf_ct<double>(), "isinf test failed for double");
    
    static_assert(test_cmp_equal(), "cmp_equal test failed");
    static_assert(test_cmp_not_equal(), "cmp_not_equal test failed");
    static_assert(test_cmp_less(), "cmp_less test failed");
    static_assert(test_cmp_greater(), "cmp_greater test failed");
    static_assert(test_cmp_less_equal(), "cmp_less_equal test failed");
    static_assert(test_cmp_greater_equal(), "cmp_greater_equal test failed");

    static_assert(test_abs_ct<char>(), "abs test failed for char");
    static_assert(test_abs_ct<short>(), "abs test failed for short");
    static_assert(test_abs_ct<long>(), "abs test failed for long");
    static_assert(test_abs_ct<long long>(), "abs test failed for long long");
    static_assert(test_abs_ct<int>(), "abs test failed for int");
    static_assert(test_abs_ct<float>(), "abs test failed for float");
    static_assert(test_abs_ct<double>(), "abs test failed for double");

    static_assert(test_max_ct(), "max test failed");
    static_assert(test_min_ct(), "min test failed");

    static_assert(test_floor_ct<float>(), "floor test failed for float");
    static_assert(test_floor_ct<double>(), "floor test failed for double");

    static_assert(test_nearbyint_ct<float>(), "nearbyint test failed for float");
    static_assert(test_nearbyint_ct<double>(), "nearbyint test failed for double");

    static_assert(test_ldexpf_ct(), "ldexpf compile-time tests failed");
    static_assert(test_expf_ct(), "expf compile-time tests failed");

    static_assert(test_fmaxf_ct(), "fmaxf compile-time tests failed");
    static_assert(test_fminf_ct(), "fminf compile-time tests failed");

    // Runtime tests
    if (!runtime_tests()) {
        return -1;
    }
    std::cout << "All tests passed!" << std::endl;
    return 0;
}

#endif