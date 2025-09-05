/* Copyright 2025 Oscar Amoros Huguet

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

// Test isnan function compile-time
template <typename T>
constexpr bool test_isnan_ct() {
    static_assert(std::is_floating_point_v<T>, "isnan test only for floating point types");

    // Test with normal values
    static_assert(!cxp::isnan(static_cast<T>(0.0)), "0.0 should not be NaN");
    static_assert(!cxp::isnan(static_cast<T>(1.0)), "1.0 should not be NaN");
    static_assert(!cxp::isnan(static_cast<T>(-1.0)), "-1.0 should not be NaN");
    static_assert(!cxp::isnan(std::numeric_limits<T>::max()), "max value should not be NaN");
    static_assert(!cxp::isnan(std::numeric_limits<T>::min()), "min value should not be NaN");
    static_assert(!cxp::isnan(std::numeric_limits<T>::lowest()), "lowest value should not be NaN");
    static_assert(!cxp::isnan(std::numeric_limits<T>::infinity()), "infinity should not be NaN");
    static_assert(!cxp::isnan(-std::numeric_limits<T>::infinity()), "-infinity should not be NaN");

    // Test with NaN
    static_assert(cxp::isnan(std::numeric_limits<T>::quiet_NaN()), "quiet_NaN should be NaN");
    static_assert(cxp::isnan(std::numeric_limits<T>::signaling_NaN()), "signaling_NaN should be NaN");

    return true;
}

// Test isnan function at runtime
template <typename T>
bool test_isnan_rt() {
    static_assert(std::is_floating_point_v<T>, "isnan test only for floating point types");
    bool allCorrect{true};
    // Test with normal values
    if (cxp::isnan(static_cast<T>(1.0)) != std::isnan(static_cast<T>(1.0f))) {
        std::cout << "Failed: cxp::isnan(1.0f) should be the same as std::isnan(1.0f)" << std::endl;
        allCorrect = false;
    }
    if (cxp::isnan(static_cast<T>(0.0)) != std::isnan(static_cast<T>(0.0))) {
        std::cout << "Failed: cxp::isnan(0.0f) should be the same as std::isnan(0.0f)" << std::endl;
        allCorrect = false;
    }
    if (cxp::isnan(static_cast<T>(-1.0)) != std::isnan(static_cast<T>(-1.0))) {
        std::cout << "Failed: cxp::isnan(-1.0f) should be the same as std::isnan(-1.0f)" << std::endl;
        allCorrect = false;
    }
    if (cxp::isnan(std::numeric_limits<T>::max()) != std::isnan(std::numeric_limits<T>::max())) {
        std::cout << "Failed: cxp::isnan(max) should be the same as std::isnan(max)" << std::endl;
        allCorrect = false;
    }
    if (cxp::isnan(std::numeric_limits<T>::min()) != std::isnan(std::numeric_limits<T>::min())) {
        std::cout << "Failed: cxp::isnan(min) should be the same as std::isnan(min)" << std::endl;
        allCorrect = false;
    }
    if (cxp::isnan(std::numeric_limits<T>::lowest()) != std::isnan(std::numeric_limits<T>::lowest())) {
        std::cout << "Failed: cxp::isnan(lowest) should be the same as std::isnan(lowest)" << std::endl;
        allCorrect = false;
    }
    if (cxp::isnan(std::numeric_limits<T>::infinity()) != std::isnan(std::numeric_limits<T>::infinity())) {
        std::cout << "Failed: cxp::isnan(infinity) should be the same as std::isnan(infinity)" << std::endl;
        allCorrect = false;
    }
    if (cxp::isnan(-std::numeric_limits<T>::infinity()) != std::isnan(-std::numeric_limits<T>::infinity())) {
        std::cout << "Failed: cxp::isnan(-infinity) should be the same as std::isnan(-infinity)" << std::endl;
        allCorrect = false;
    }

    // Test with NaN
    if (cxp::isnan(std::numeric_limits<T>::quiet_NaN()) != std::isnan(std::numeric_limits<T>::quiet_NaN())) {
        std::cout << "Failed: cxp::isnan(quiet_NaN) should be the same as std::isnan(quiet_NaN)" << std::endl;
        allCorrect = false;
    }
    if (cxp::isnan(std::numeric_limits<T>::signaling_NaN()) != std::isnan(std::numeric_limits<T>::signaling_NaN())) {
        std::cout << "Failed: cxp::isnan(signaling_NaN) should be the same as std::isnan(signaling_NaN)" << std::endl;
        allCorrect = false;
    }

    return allCorrect;
}

// Test isinf function
template <typename T>
constexpr bool test_isinf_ct() {
    static_assert(std::is_floating_point_v<T>, "isinf test only for floating point types");
    
    // Test with normal values
    static_assert(!cxp::isinf(static_cast<T>(0.0)), "0.0 should not be infinite");
    static_assert(!cxp::isinf(static_cast<T>(1.0)), "1.0 should not be infinite");
    static_assert(!cxp::isinf(static_cast<T>(-1.0)), "-1.0 should not be infinite");
    static_assert(!cxp::isinf(static_cast<T>(1000.0)), "1000.0 should not be infinite");
    static_assert(!cxp::isinf(static_cast<T>(-1000.0)), "-1000.0 should not be infinite");
    static_assert(!cxp::isinf(std::numeric_limits<T>::quiet_NaN()), "NaN should not be infinite");

    // Test with infinity
    static_assert(cxp::isinf(std::numeric_limits<T>::infinity()), "infinity should be infinite");
    static_assert(cxp::isinf(-std::numeric_limits<T>::infinity()), "-infinity should be infinite");

    return true;
}

// Test isinf function at runtime
template <typename T>
constexpr bool test_isinf_rt() {
    static_assert(std::is_floating_point_v<T>, "isinf test only for floating point types");
    bool allCorrect{true};
    // Test isinf with runtime values
    T inf_val = std::numeric_limits<T>::infinity();
    if (cxp::isinf(inf_val) != std::isinf(inf_val)) {
        std::cout << "Failed: cxp::isinf(inf_val) should be the same as std::isinf(inf_val)" << std::endl;
        allCorrect = false;
    }
    if (cxp::isinf(-inf_val) != std::isinf(-inf_val)) {
        std::cout << "Failed: cxp::isinf(-inf_val) should be the same as std::isinf(-inf_val)" << std::endl;
        allCorrect = false;
    }
    if (cxp::isinf(1.0f) != std::isinf(1.0f)) {
        std::cout << "Failed: cxp::isinf(1.0f) should be the same as std::isinf(1.0f)" << std::endl;
        allCorrect = false;
    }
    if (cxp::isinf(-1.0f) != std::isinf(-1.0f)) {
        std::cout << "Failed: cxp::isinf(-1.0f) should be the same as std::isinf(-1.0f)" << std::endl;
        allCorrect = false;
    }
    if (cxp::isinf(std::numeric_limits<T>::max()) != std::isinf(std::numeric_limits<T>::max())) {
        std::cout << "Failed: cxp::isinf(max) should be the same as std::isinf(max)" << std::endl;
        allCorrect = false;
    }
    if (cxp::isinf(std::numeric_limits<T>::min()) != std::isinf(std::numeric_limits<T>::min())) {
        std::cout << "Failed: cxp::isinf(min) should be the same as std::isinf(min)" << std::endl;
        allCorrect = false;
    }
    if (cxp::isinf(std::numeric_limits<T>::lowest()) != std::isinf(std::numeric_limits<T>::lowest())) {
        std::cout << "Failed: cxp::isinf(lowest) should be the same as std::isinf(lowest)" << std::endl;
        allCorrect = false;
    }
    if (cxp::isinf(0.0f) != std::isinf(0.0f)) {
        std::cout << "Failed: cxp::isinf(0.0f) should be the same as std::isinf(0.0f)" << std::endl;
        allCorrect = false;
    }
    return allCorrect;
}

// Test cmp_equal function
constexpr bool test_cmp_equal() {
    // Same type comparisons
    static_assert(cxp::cmp_equal(5, 5), "5 == 5 should be true");
    static_assert(!cxp::cmp_equal(5, 4), "5 == 4 should be false");
    static_assert(cxp::cmp_equal(5.0, 5.0), "5.0 == 5.0 should be true");
    static_assert(!cxp::cmp_equal(5.0, 4.0), "5.0 == 4.0 should be false");
    
    // Mixed signed/unsigned comparisons
    static_assert(cxp::cmp_equal(5, 5u), "5 == 5u should be true");
    static_assert(!cxp::cmp_equal(-1, 5u), "-1 == 5u should be false");
    static_assert(!cxp::cmp_equal(5u, -1), "5u == -1 should be false");
    static_assert(cxp::cmp_equal(0, 0u), "0 == 0u should be true");
    
    // Mixed integer/floating point comparisons
    static_assert(cxp::cmp_equal(5, 5.0), "5 == 5.0 should be true");
    static_assert(cxp::cmp_equal(5.0, 5), "5.0 == 5 should be true");
    static_assert(!cxp::cmp_equal(5, 5.1), "5 == 5.1 should be false");
    
    return true;
}

// Test cmp_not_equal function
constexpr bool test_cmp_not_equal() {
    static_assert(!cxp::cmp_not_equal(5, 5), "5 != 5 should be false");
    static_assert(cxp::cmp_not_equal(5, 4), "5 != 4 should be true");
    static_assert(cxp::cmp_not_equal(-1, 5u), "-1 != 5u should be true");
    static_assert(!cxp::cmp_not_equal(5, 5.0), "5 != 5.0 should be false");
    
    return true;
}

// Test cmp_less function
constexpr bool test_cmp_less() {
    // Same type comparisons
    static_assert(cxp::cmp_less(4, 5), "4 < 5 should be true");
    static_assert(!cxp::cmp_less(5, 4), "5 < 4 should be false");
    static_assert(!cxp::cmp_less(5, 5), "5 < 5 should be false");
    
    // Mixed signed/unsigned comparisons
    static_assert(cxp::cmp_less(-1, 5u), "-1 < 5u should be true");
    static_assert(!cxp::cmp_less(5u, -1), "5u < -1 should be false");
    static_assert(cxp::cmp_less(4u, 5), "4u < 5 should be true");
    static_assert(!cxp::cmp_less(5u, 4), "5u < 4 should be false");
    
    // Mixed integer/floating point comparisons
    static_assert(cxp::cmp_less(4, 5.0), "4 < 5.0 should be true");
    static_assert(cxp::cmp_less(4.0, 5), "4.0 < 5 should be true");
    static_assert(!cxp::cmp_less(5.0, 4), "5.0 < 4 should be false");
    
    return true;
}

// Test cmp_greater function
constexpr bool test_cmp_greater() {
    static_assert(cxp::cmp_greater(5, 4), "5 > 4 should be true");
    static_assert(!cxp::cmp_greater(4, 5), "4 > 5 should be false");
    static_assert(!cxp::cmp_greater(5, 5), "5 > 5 should be false");
    static_assert(cxp::cmp_greater(5u, -1), "5u > -1 should be true");
    static_assert(!cxp::cmp_greater(-1, 5u), "-1 > 5u should be false");
    
    return true;
}

// Test cmp_less_equal function
constexpr bool test_cmp_less_equal() {
    static_assert(cxp::cmp_less_equal(4, 5), "4 <= 5 should be true");
    static_assert(cxp::cmp_less_equal(5, 5), "5 <= 5 should be true");
    static_assert(!cxp::cmp_less_equal(5, 4), "5 <= 4 should be false");
    static_assert(cxp::cmp_less_equal(-1, 5u), "-1 <= 5u should be true");
    static_assert(cxp::cmp_less_equal(0, 0u), "0 <= 0u should be true");
    
    return true;
}

// Test cmp_greater_equal function
constexpr bool test_cmp_greater_equal() {
    static_assert(cxp::cmp_greater_equal(5, 4), "5 >= 4 should be true");
    static_assert(cxp::cmp_greater_equal(5, 5), "5 >= 5 should be true");
    static_assert(!cxp::cmp_greater_equal(4, 5), "4 >= 5 should be false");
    static_assert(cxp::cmp_greater_equal(5u, -1), "5u >= -1 should be true");
    static_assert(cxp::cmp_greater_equal(0u, 0), "0u >= 0 should be true");
    
    return true;
}

// Runtime tests for round function
template <typename T>
bool test_round_rt() {
    static_assert(std::is_floating_point_v<T>, "round test only for floating point types");

    // Runtime test
    bool allCorrect{true};
    if (cxp::round(static_cast<T>(1.4)) != std::round(static_cast<T>(1.4))) {
        std::cout << "cxp::round(1.4) failed" << std::endl;
        allCorrect = false;
    }
    if (cxp::round(static_cast<T>(1.5)) != std::round(static_cast<T>(1.5))) {
        std::cout << "cxp::round(1.5) failed" << std::endl;
        allCorrect = false;
    }
    if (cxp::round(static_cast<T>(1.6)) != std::round(static_cast<T>(1.6))) {
        std::cout << "cxp::round(1.6) failed" << std::endl;
        allCorrect = false;
    }
    if (cxp::round(static_cast<T>(-1.4)) != std::round(static_cast<T>(-1.4))) {
        std::cout << "cxp::round(-1.4) failed" << std::endl;
        allCorrect = false;
    }
    if (cxp::round(static_cast<T>(-1.5)) != std::round(static_cast<T>(-1.5))) {
        std::cout << "cxp::round(-1.5) failed" << std::endl;
        allCorrect = false;
    }
    if (cxp::round(static_cast<T>(-1.6)) != std::round(static_cast<T>(-1.6))) {
        std::cout << "cxp::round(-1.6) failed" << std::endl;
        allCorrect = false;
    }
    if (cxp::round(static_cast<T>(0.0)) != std::round(static_cast<T>(0.0))) {
        std::cout << "cxp::round(0.0) failed" << std::endl;
        allCorrect = false;
    }
    if (cxp::round(static_cast<T>(2.0)) != std::round(static_cast<T>(2.0))) {
        std::cout << "cxp::round(2.0) failed" << std::endl;
        allCorrect = false;
    }
    
    // Special values runtime
    if (!std::isnan(cxp::round(std::numeric_limits<T>::quiet_NaN()))) {
        std::cout << "Failed: cxp::round(NaN) should be NaN" << std::endl;
        allCorrect = false;
    }
    if (!std::isinf(cxp::round(std::numeric_limits<T>::infinity()))) {
        std::cout << "Failed: cxp::round(inf) should be inf" << std::endl;
        allCorrect = false;
    }
    if (!std::isinf(cxp::round(-std::numeric_limits<T>::infinity()))) {
        std::cout << "Failed: cxp::round(-inf) should be -inf" << std::endl;
        allCorrect = false;
    }
    
    return allCorrect;
}

// Compile time test round function
template <typename T>
constexpr bool test_round_ct() {
    static_assert(std::is_floating_point_v<T>, "round test only for floating point types");

    // Compile-time tests
    static_assert(cxp::round(static_cast<T>(1.4)) == static_cast<T>(1.0), "round(1.4) should be 1.0");
    static_assert(cxp::round(static_cast<T>(1.5)) == static_cast<T>(2.0), "round(1.5) should be 2.0");
    static_assert(cxp::round(static_cast<T>(1.6)) == static_cast<T>(2.0), "round(1.6) should be 2.0");
    static_assert(cxp::round(static_cast<T>(-1.4)) == static_cast<T>(-1.0), "round(-1.4) should be -1.0");
    static_assert(cxp::round(static_cast<T>(-1.5)) == static_cast<T>(-2.0), "round(-1.5) should be -2.0");
    static_assert(cxp::round(static_cast<T>(-1.6)) == static_cast<T>(-2.0), "round(-1.6) should be -2.0");
    static_assert(cxp::round(static_cast<T>(0.0)) == static_cast<T>(0.0), "round(0.0) should be 0.0");
    static_assert(cxp::round(static_cast<T>(2.0)) == static_cast<T>(2.0), "round(2.0) should be 2.0");

    // Special values compile-time
    static_assert(cxp::isnan(cxp::round(std::numeric_limits<T>::quiet_NaN())), "round(NaN) should be NaN");
    static_assert(cxp::isinf(cxp::round(std::numeric_limits<T>::infinity())), "round(inf) should be inf");
    static_assert(cxp::isinf(cxp::round(-std::numeric_limits<T>::infinity())), "round(-inf) should be -inf");
    
    return true;
}

// Test abs function at compile-time
template <typename T>
constexpr bool test_abs_ct() {
    static_assert(std::is_fundamental_v<T>, "abs test only for fundamental types");
    
    if constexpr (std::is_signed_v<T> && sizeof(T) >= 4) {
        static_assert(cxp::abs(static_cast<T>(5)) == static_cast<T>(5), "abs(5) should be 5");
        static_assert(cxp::abs(static_cast<T>(-5)) == static_cast<T>(5), "abs(-5) should be 5");
        static_assert(cxp::abs(static_cast<T>(0)) == static_cast<T>(0), "abs(0) should be 0");

        // Test edge case: most negative value
        static_assert(cxp::abs(cxp::minValue<T> + 1) == -(cxp::minValue<T> + 1), 
                      "abs(min) should be max for signed types");
    } else if constexpr (std::is_signed_v<T> && sizeof(T) < 4) {
        static_assert(cxp::abs(static_cast<T>(5)) == static_cast<int>(5), "abs(5) should be 5");
        static_assert(cxp::abs(static_cast<T>(-5)) == static_cast<int>(5), "abs(-5) should be 5");
        static_assert(cxp::abs(static_cast<T>(0)) == static_cast<int>(0), "abs(0) should be 0");

        // Test edge case: most negative value
        static_assert(cxp::abs(cxp::minValue<T>) == -static_cast<int>(cxp::minValue<T>), 
                      "abs(min) should be max for signed types");
    } else {
        // Unsigned types
        static_assert(cxp::abs(static_cast<T>(5)) == static_cast<T>(5), "abs(5) should be 5 for unsigned");
        static_assert(cxp::abs(static_cast<T>(0)) == static_cast<T>(0), "abs(0) should be 0 for unsigned");
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
        if (cxp::abs(static_cast<T>(5)) != std::abs(static_cast<T>(5))) {
            std::cout << "Failed: abs(5) should be 5" << std::endl;
            allCorrect = false;
        }
        if (cxp::abs(static_cast<T>(-5)) != std::abs(static_cast<T>(5))) {
            std::cout << "Failed: abs(-5) should be 5" << std::endl;
            allCorrect = false;
        }
        if (cxp::abs(static_cast<T>(0)) != std::abs(static_cast<T>(0))) {
            std::cout << "Failed: abs(0) should be 0" << std::endl;
            allCorrect = false;
        }
        // Edge case for signed types
        // For integer signed types, abs(minValue) is undefined behavior in C++
        constexpr T extra = std::is_integral_v<T> ? static_cast<T>(1) : static_cast<T>(0);
        if (cxp::abs(cxp::minValue<T> + extra) != std::abs(cxp::minValue<T> + extra)) {
            using CxpType = std::decay_t<decltype(cxp::abs(cxp::minValue<T>))>;
            using StdType = std::decay_t<decltype(std::abs(cxp::minValue<T>))>;
            static_assert(std::is_same_v<CxpType, StdType>, 
                          "cxp::abs(minValue<T>) should have the same type as std::abs(cxp::minValue<T>)");
            std::cout << "Failed: abs(min) should be max for signed types" << std::endl;
            if constexpr (sizeof(T) < 4) {
                std::cout << "T= " + fk::typeToString<T>() + " Expected: " << std::abs(cxp::minValue<T> + extra) << ", got: " << static_cast<int>(cxp::abs(cxp::minValue<T> + extra)) << std::endl;
            } else {
                std::cout << "T= " + fk::typeToString<T>() + " Expected: " << std::abs(cxp::minValue<T> + extra) << ", got: " << cxp::abs(cxp::minValue<T> + extra) << std::endl;
            }
            allCorrect = false;
        }
    } else {
        // Unsigned types
        if (cxp::abs(static_cast<T>(5)) != std::abs(static_cast<T>(5))) {
            std::cout << "Failed: abs(5) should be 5 for unsigned" << std::endl;
            allCorrect = false;
        }
        if (cxp::abs(static_cast<T>(0)) != std::abs(static_cast<T>(0))) {
            std::cout << "Failed: abs(0) should be 0 for unsigned" << std::endl;
            allCorrect = false;
        }
    }

    return allCorrect;
}

// Test max function at compile-time
constexpr bool test_max_ct() {
    static_assert(cxp::max(5) == 5, "max(5) should be 5");
    static_assert(cxp::max(3, 5) == 5, "max(3, 5) should be 5");
    static_assert(cxp::max(5, 3) == 5, "max(5, 3) should be 5");
    static_assert(cxp::max(1, 3, 5, 2) == 5, "max(1, 3, 5, 2) should be 5");
    static_assert(cxp::max(-1, -3, -5, -2) == -1, "max(-1, -3, -5, -2) should be -1");
    static_assert(cxp::max(1.0, 3.0, 5.0, 2.0) == 5.0, "max(1.0, 3.0, 5.0, 2.0) should be 5.0");
    
    return true;
}

// Test max function at runtime
bool test_max_rt() {
    bool allCorrect{true};
    // Test with runtime values
    if (cxp::max(3, 5) != std::max(3, 5)) {
        std::cout << "Failed: max(3, 5) should be 5" << std::endl;
        allCorrect = false;
    }
    if (cxp::max(5, 3) != std::max(5, 3)) {
        std::cout << "Failed: max(5, 3) should be 5" << std::endl;
        allCorrect = false;
    }
    if (cxp::max(1, 3, 5, 2) != std::max(std::max(1, 3), std::max(5, 2))) {
        std::cout << "Failed: max(1, 3, 5, 2) should be 5" << std::endl;
        allCorrect = false;
    }
    if (cxp::max(-1, -3, -5, -2) != std::max(std::max(-1, -3), std::max(-5, -2))) {
        std::cout << "Failed: max(-1, -3, -5, -2) should be -1" << std::endl;
        allCorrect = false;
    }
    if (cxp::max(1.0, 3.0, 5.0, 2.0) != std::max(std::max(1.0, 3.0), std::max(5.0, 2.0))) {
        std::cout << "Failed: max(1.0, 3.0, 5.0, 2.0) should be 5.0" << std::endl;
        allCorrect = false;
    }

    return allCorrect;
}

// Test min function at compile-time
constexpr bool test_min_ct() {
    static_assert(cxp::min(5) == 5, "min(5) should be 5");
    
    // Test with two arguments to see if the bug manifests
    // The bug in min_helper calls max_helper instead of min_helper recursively
    static_assert(cxp::min(3, 5) == 3, "min(3, 5) should be 3");
    static_assert(cxp::min(5, 3) == 3, "min(5, 3) should be 3");
    static_assert(cxp::min(1, 3, 5, 2) == 1, "min(1, 3, 5, 2) should be 1");
    static_assert(cxp::min(-1, -3, -5, -2) == -5, "min(-1, -3, -5, -2) should be -5");
    static_assert(cxp::min(1.0, 3.0, 5.0, 2.0) == 1.0, "min(1.0, 3.0, 5.0, 2.0) should be 1.0");
    
    return true;
}
// Test min function at runtime
bool test_min_rt() {
    bool allCorrect{true};
    // Test with runtime values
    if (cxp::min(3, 5) != std::min(3, 5)) {
        std::cout << "Failed: min(3, 5) should be 3" << std::endl;
        allCorrect = false;
    }
    if (cxp::min(5, 3) != std::min(5, 3)) {
        std::cout << "Failed: min(5, 3) should be 3" << std::endl;
        allCorrect = false;
    }
    if (cxp::min(1, 3, 5, 2) != std::min(std::min(1, 3), std::min(5, 2))) {
        std::cout << "Failed: min(1, 3, 5, 2) should be 1" << std::endl;
        allCorrect = false;
    }
    if (cxp::min(-1, -3, -5, -2) != std::min(std::min(-1, -3), std::min(-5, -2))) {
        std::cout << "Failed: min(-1, -3, -5, -2) should be -5" << std::endl;
        allCorrect = false;
    }
    if (cxp::min(1.0, 3.0, 5.0, 2.0) != std::min(std::min(1.0, 3.0), std::min(5.0, 2.0))) {
        std::cout << "Failed: min(1.0, 3.0, 5.0, 2.0) should be 1.0" << std::endl;
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

    // Test isinf with runtime values
    allCorrect &= test_isinf_rt<float>();
    allCorrect &= test_isinf_rt<double>();
    
    // Test isnan with runtime values
    allCorrect &= test_isnan_rt<float>();
    allCorrect &= test_isnan_rt<double>();
    
    // Test comparison functions with runtime values
    if (!cxp::cmp_equal(5, 5)) {
        std::cout << "Failed: cxp::cmp_equal(5, 5) should be true" << std::endl;
        allCorrect = false;
    }
    if (cxp::cmp_equal(5, 4)) {
        std::cout << "Failed: cxp::cmp_equal(5, 4) should be false" << std::endl;
        allCorrect = false;
    }
    if (!cxp::cmp_less(4, 5)) {
        std::cout << "Failed: cxp::cmp_less(4, 5) should be true" << std::endl;
        allCorrect = false;
    }
    if (cxp::cmp_less(5, 4)) {
        std::cout << "Failed: cxp::cmp_less(5, 4) should be false" << std::endl;
        allCorrect = false;
    }

    // Test mixed type comparisons
    if (!cxp::cmp_equal(5u, 5)) {
        std::cout << "Failed: cxp::cmp_equal(5u, 5) should be true" << std::endl;
        allCorrect = false;
    }
    if (cxp::cmp_equal(5u, -1)) {
        std::cout << "Failed: cxp::cmp_equal(5u, -1) should be false" << std::endl;
        allCorrect = false;
    }
    if (!cxp::cmp_less(-1, 5u)) {
        std::cout << "Failed: cxp::cmp_less(-1, 5u) should be true" << std::endl;
        allCorrect = false;
    }
    if (cxp::cmp_less(5u, -1)) {
        std::cout << "Failed: cxp::cmp_less(5u, -1) should be false" << std::endl;
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

    // Runtime tests
    if (!runtime_tests()) {
        return -1;
    }
    std::cout << "All tests passed!" << std::endl;
    return 0;
}

#endif