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

#ifndef FK_CONSTEXPR_CMATH
#define FK_CONSTEXPR_CMATH

#include <fused_kernel/core/utils/utils.h>
#include <fused_kernel/core/utils/type_lists.h>

#include <type_traits>
#include <limits>

namespace cxp {
    template <typename T>
    constexpr T minValue = std::numeric_limits<T>::lowest();

    template <typename T>
    constexpr T maxValue = std::numeric_limits<T>::max();

    template <typename T>
    constexpr T smallestPositiveValue = std::is_floating_point_v<T> ? std::numeric_limits<T>::min() : static_cast<T>(1);

    template <typename T>
    FK_HOST_DEVICE_CNST bool isnan(T x) {
        return x != x;
    }

    template <typename T>
    FK_HOST_DEVICE_CNST bool isinf(T x) {
        return x == x && x != T(0) && x + x == x;
    }

    // safe_cmp_equal
    template<typename T, typename U>
    FK_HOST_DEVICE_CNST bool cmp_equal(const T& t, const U& u) {
        static_assert(!std::is_same_v<T, bool> && std::is_fundamental_v<T>,
            "First parameter should be a fundamental type other than bool.");
        static_assert(!std::is_same_v<U, bool> && std::is_fundamental_v<U>,
            "Second parameter should be a fundamental type other than bool.");
        constexpr bool isAnyFloatingPoint = std::is_floating_point_v<T> || std::is_floating_point_v<U>;
        constexpr bool areBothSigned = std::is_signed_v<T> == std::is_signed_v<U>;

        if constexpr (isAnyFloatingPoint || areBothSigned) {
            // Safe comparison cases
            return t == u;
        } else if constexpr (std::is_signed_v<T>) {
            // T is signed, U is unsigned, both are integers
            if (t < 0) return false; // Negative cannot equal any unsigned.
            return std::make_unsigned_t<T>(t) == u;
        } else {
            // T is unsigned, U is signed, both are integers
            if (u < 0) return false; // Negative cannot equal any unsigned.
            return t == std::make_unsigned_t<U>(u);
        }
    }

    // safe_cmp_not_equal
    template<typename T, typename U>
    FK_HOST_DEVICE_CNST bool cmp_not_equal(const T& t, const U& u) {
        return !cmp_equal(t, u);
    }

    // safe_cmp_less
    template<typename T, typename U>
    FK_HOST_DEVICE_CNST bool cmp_less(const T& t, const U& u) {
        static_assert(!std::is_same_v<T, bool> && std::is_fundamental_v<T>,
            "First parameter must be a fundamental type other than bool");
        static_assert(!std::is_same_v<U, bool> && std::is_fundamental_v<U>,
            "Second parameter must be a fundamental type other than bool");
        constexpr bool isAnyFloatingPoint = std::is_floating_point_v<T> || std::is_floating_point_v<U>;
        constexpr bool areBothSigned = std::is_signed_v<T> == std::is_signed_v<U>;

        if constexpr (isAnyFloatingPoint || areBothSigned) {
            // Safe comparison cases
            return t < u;
        } else if constexpr (std::is_signed_v<T>) {
            // T is signed, U is unsigned, both are integers
            if (t < 0) return true; // Signed negative is always less than unsigned.
            return static_cast<std::make_unsigned_t<T>>(t) < u;
        } else {
            // T is unsigned, U is signed, both are integers
            if (u < 0) return false; // Unsigned is never less than a signed negative.
            return t < static_cast<std::make_unsigned_t<U>>(u);
        }
    }

    // safe_cmp_greater
    template<typename T, typename U>
    FK_HOST_DEVICE_CNST bool cmp_greater(const T& t, const U& u) {
        // Re-use the logic from cmp_less by swapping the arguments.
        return cmp_less(u, t);
    }

    // safe_cmp_less_equal
    template<typename T, typename U>
    FK_HOST_DEVICE_CNST bool cmp_less_equal(const T& t, const U& u) {
        // Equivalent to "not greater than".
        return !cmp_greater(t, u);
    }

    // safe_cmp_greater_equal
    template<typename T, typename U>
    FK_HOST_DEVICE_CNST bool cmp_greater_equal(const T& t, const U& u) {
        // Equivalent to "not less than".
        return !cmp_less(t, u);
    }

    template<typename T>
    FK_HOST_DEVICE_CNST T round(T x) {
        static_assert(std::is_floating_point<T>::value, "Input must be a floating-point type");

        if (isnan(x) || isinf(x)) {
            return x;
        }
        // Casted to int instead of long long, because long long is very slow on GPU
        return (x > T(0))
            ? static_cast<T>(static_cast<int>(x + T(0.5)))
            : static_cast<T>(static_cast<int>(x - T(0.5)));
    }

    namespace internal {
        template <typename Type>
        FK_HOST_DEVICE_CNST auto max_helper(const Type& value) {
            return value;
        }
        template <typename FirstType, typename... Types>
        FK_HOST_DEVICE_CNST auto max_helper(const FirstType& firstValue,
                                            const Types&... values) {
            const auto previousMax = max_helper(values...);
            return firstValue >= previousMax ? firstValue : previousMax;
        }
        template <typename Type>
        FK_HOST_DEVICE_CNST auto min_helper(const Type& value) {
            return value;
        }
        template <typename FirstType, typename... Types>
        FK_HOST_DEVICE_CNST auto min_helper(const FirstType& firstValue,
            const Types&... values) {
            const auto previousMin = min_helper(values...);
            return firstValue <= previousMin ? firstValue : previousMin;
        }
    } // namespace internal

    template <typename FirstType, typename... Types>
    FK_HOST_DEVICE_CNST auto max(const FirstType& value, const Types&... values) {
        if constexpr (sizeof...(values) == 0) {
            return value;
        } else {
            static_assert((fk::all_types_are_same<FirstType, Types...>), "All types must be the same");
            return internal::max_helper(value, values...);
        }
    }

    template <typename FirstType, typename... Types>
    FK_HOST_DEVICE_CNST auto min(const FirstType& value, const Types&... values) {
        if constexpr (sizeof...(values) == 0) {
            return value;
        } else {
            static_assert(fk::all_types_are_same<FirstType, Types...>, "All types must be the same");
            return internal::min_helper(value, values...);
        }
    }

    template <typename T>
    FK_HOST_DEVICE_CNST auto abs(const T& x) {
        static_assert(std::is_fundamental_v<T>, "abs does not support non fundamental types");
        if constexpr (std::is_signed_v<T>) {
            // For signed integrals, when x is std::numerical_limits<T>::lowest(),
            // the result is undefined behavior in C++. So, for the sake of performance,
            // we will not do any special treatment for those cases.
            return x < static_cast<T>(0) ? -x : x;
        } else {
            return x;
        }
    }

    // Non std variants
    template <typename FirstType, typename... Types>
    FK_HOST_DEVICE_CNST auto sum(const FirstType& value, const Types&... values) {
        static_assert(fk::all_types_are_same<FirstType, Types...>, "All types must be the same");
        if constexpr (sizeof...(values) == 0) {
            return value;
        } else {
            return value + sum(values...);
        }
    }
} // namespace cxp

#endif
