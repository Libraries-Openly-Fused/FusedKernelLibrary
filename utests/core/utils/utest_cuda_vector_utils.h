/* Copyright 2025 Oscar Amoros Huguet
   Copyright 2025 Albert Andaluz

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#define __ONLY_CPU__

#ifndef FK_UTEST_CUDA_VECTOR_UTILS_H
#define FK_UTEST_CUDA_VECTOR_UTILS_H

#include <fused_kernel/core/utils/cuda_vector_utils.h>
#include <fused_kernel/core/utils/type_to_string.h>
#include <array>
#include <type_traits>
#include <iostream>
#include <vector>
#include <string>


namespace fk::test {
    // Track compilation results
    std::vector<std::string> unexpected_failed_compilations;

    // SFINAE-based test helpers
    template<typename T, typename = void>
    struct can_unary_minus : std::false_type {};

    template<typename T>
    struct can_unary_minus<T, std::void_t<decltype(-std::declval<T>())>> : std::true_type {};

    template<typename T, typename = void>
    struct can_unary_not : std::false_type {};

    template<typename T>
    struct can_unary_not<T, std::void_t<decltype(!std::declval<T>())>> : std::true_type {};

    template<typename T, typename = void>
    struct can_unary_bitwise_not : std::false_type {};

    template<typename T>
    struct can_unary_bitwise_not<T, std::void_t<decltype(~std::declval<T>())>> : std::true_type {};

    template<typename T1, typename T2, typename = void>
    struct can_add : std::false_type {};

    template<typename T1, typename T2>
    struct can_add<T1, T2, std::void_t<decltype(std::declval<T1>() + std::declval<T2>())>> : std::true_type {};

    template<typename T1, typename T2, typename = void>
    struct can_subtract : std::false_type {};

    template<typename T1, typename T2>
    struct can_subtract<T1, T2, std::void_t<decltype(std::declval<T1>() - std::declval<T2>())>> : std::true_type {};

    template<typename T1, typename T2, typename = void>
    struct can_multiply : std::false_type {};

    template<typename T1, typename T2>
    struct can_multiply<T1, T2, std::void_t<decltype(std::declval<T1>() * std::declval<T2>())>> : std::true_type {};

    template<typename T1, typename T2, typename = void>
    struct can_divide : std::false_type {};

    template<typename T1, typename T2>
    struct can_divide<T1, T2, std::void_t<decltype(std::declval<T1>() / std::declval<T2>())>> : std::true_type {};

    template<typename T1, typename T2, typename = void>
    struct can_equal : std::false_type {};

    template<typename T1, typename T2>
    struct can_equal<T1, T2, std::void_t<decltype(std::declval<T1>() == std::declval<T2>())>> : std::true_type {};

    template<typename T1, typename T2, typename = void>
    struct can_not_equal : std::false_type {};

    template<typename T1, typename T2>
    struct can_not_equal<T1, T2, std::void_t<decltype(std::declval<T1>() != std::declval<T2>())>> : std::true_type {};

    template<typename T1, typename T2, typename = void>
    struct can_less : std::false_type {};

    template<typename T1, typename T2>
    struct can_less<T1, T2, std::void_t<decltype(std::declval<T1>() < std::declval<T2>())>> : std::true_type {};

    template<typename T1, typename T2, typename = void>
    struct can_less_equal : std::false_type {};

    template<typename T1, typename T2>
    struct can_less_equal<T1, T2, std::void_t<decltype(std::declval<T1>() <= std::declval<T2>())>> : std::true_type {};

    template<typename T1, typename T2, typename = void>
    struct can_greater : std::false_type {};

    template<typename T1, typename T2>
    struct can_greater<T1, T2, std::void_t<decltype(std::declval<T1>() > std::declval<T2>())>> : std::true_type {};

    template<typename T1, typename T2, typename = void>
    struct can_greater_equal : std::false_type {};

    template<typename T1, typename T2>
    struct can_greater_equal<T1, T2, std::void_t<decltype(std::declval<T1>() >= std::declval<T2>())>> : std::true_type {};

    template<typename T1, typename T2, typename = void>
    struct can_logical_and : std::false_type {};

    template<typename T1, typename T2>
    struct can_logical_and<T1, T2, std::void_t<decltype(std::declval<T1>() && std::declval<T2>())>> : std::true_type {};

    template<typename T1, typename T2, typename = void>
    struct can_logical_or : std::false_type {};

    template<typename T1, typename T2>
    struct can_logical_or<T1, T2, std::void_t<decltype(std::declval<T1>() || std::declval<T2>())>> : std::true_type {};

    template<typename T1, typename T2, typename = void>
    struct can_bitwise_and : std::false_type {};

    template<typename T1, typename T2>
    struct can_bitwise_and<T1, T2, std::void_t<decltype(std::declval<T1>() & std::declval<T2>())>> : std::true_type {};

    template<typename T1, typename T2, typename = void>
    struct can_bitwise_or : std::false_type {};

    template<typename T1, typename T2>
    struct can_bitwise_or<T1, T2, std::void_t<decltype(std::declval<T1>() | std::declval<T2>())>> : std::true_type {};

    template<typename T1, typename T2, typename = void>
    struct can_bitwise_xor : std::false_type {};

    template<typename T1, typename T2>
    struct can_bitwise_xor<T1, T2, std::void_t<decltype(std::declval<T1>() ^ std::declval<T2>())>> : std::true_type {};

    // Compound assignment operators
    template<typename T1, typename T2, typename = void>
    struct can_add_assign : std::false_type {};

    template<typename T1, typename T2>
    struct can_add_assign<T1, T2, std::void_t<decltype(std::declval<T1&>() += std::declval<T2>())>> : std::true_type {};

    template<typename T1, typename T2, typename = void>
    struct can_sub_assign : std::false_type {};

    template<typename T1, typename T2>
    struct can_sub_assign<T1, T2, std::void_t<decltype(std::declval<T1&>() -= std::declval<T2>())>> : std::true_type {};

    template<typename T1, typename T2, typename = void>
    struct can_mul_assign : std::false_type {};

    template<typename T1, typename T2>
    struct can_mul_assign<T1, T2, std::void_t<decltype(std::declval<T1&>() *= std::declval<T2>())>> : std::true_type {};

    template<typename T1, typename T2, typename = void>
    struct can_div_assign : std::false_type {};

    template<typename T1, typename T2>
    struct can_div_assign<T1, T2, std::void_t<decltype(std::declval<T1&>() /= std::declval<T2>())>> : std::true_type {};

    template<typename T1, typename T2, typename = void>
    struct can_and_assign : std::false_type {};

    template<typename T1, typename T2>
    struct can_and_assign<T1, T2, std::void_t<decltype(std::declval<T1&>() &= std::declval<T2>())>> : std::true_type {};

    template<typename T1, typename T2, typename = void>
    struct can_or_assign : std::false_type {};

    template<typename T1, typename T2>
    struct can_or_assign<T1, T2, std::void_t<decltype(std::declval<T1&>() |= std::declval<T2>())>> : std::true_type {};

    using VecAndStdTypes = TypeListCat_t<VAll, StandardTypes>;

    template <typename T>
    void detectUnaryUnexpectedCompilationErrors() {
        if constexpr (can_unary_minus<T>::value != can_unary_minus<VBase<T>>::value) {
            unexpected_failed_compilations.push_back("unaryMinus_" + typeToString<T>());
        }
        if constexpr (can_unary_not<T>::value != can_unary_not<VBase<T>>::value) {
            unexpected_failed_compilations.push_back("unaryNot_" + typeToString<T>());
        }
        if constexpr (can_unary_bitwise_not<T>::value != can_unary_bitwise_not<VBase<T>>::value) {
            unexpected_failed_compilations.push_back("unaryBitwiseNot_" + typeToString<T>());
        }
    }

    template <typename I1,  typename I2>
    void detectBinaryUnexpectedCompilationErrors() {
        if constexpr (cn<I1> == cn<I2>) {
            if constexpr (can_add<I1, I2>::value != can_add<VBase<I1>, VBase<I2>>::value) {
                unexpected_failed_compilations.push_back("binaryAdd_" + typeToString<I1>() + "_" + typeToString<I2>());
            }
            if constexpr (can_subtract<I1, I2>::value != can_subtract<VBase<I1>, VBase<I2>>::value) {
                unexpected_failed_compilations.push_back("binaryMinus_" + typeToString<I1>() + "_" + typeToString<I2>());
            }
            if constexpr (can_multiply<I1, I2>::value != can_multiply<VBase<I1>, VBase<I2>>::value) {
                unexpected_failed_compilations.push_back("binaryMul_" + typeToString<I1>() + "_" + typeToString<I2>());
            }
            if constexpr (can_divide<I1, I2>::value != can_divide<VBase<I1>, VBase<I2>>::value) {
                unexpected_failed_compilations.push_back("binaryDiv_" + typeToString<I1>() + "_" + typeToString<I2>());
            }
            if constexpr (can_equal<I1, I2>::value != can_equal<VBase<I1>, VBase<I2>>::value) {
                unexpected_failed_compilations.push_back("binaryEqual_" + typeToString<I1>() + "_" + typeToString<I2>());
            }
            if constexpr (can_not_equal<I1, I2>::value != can_not_equal<VBase<I1>, VBase<I2>>::value) {
                unexpected_failed_compilations.push_back("binaryNotEqual_" + typeToString<I1>() + "_" + typeToString<I2>());
            }
            if constexpr (can_less<I1, I2>::value != can_less<VBase<I1>, VBase<I2>>::value) {
                unexpected_failed_compilations.push_back("binaryLess_" + typeToString<I1>() + "_" + typeToString<I2>());
            }
            if constexpr (can_less_equal<I1, I2>::value != can_less_equal<VBase<I1>, VBase<I2>>::value) {
                unexpected_failed_compilations.push_back("binaryLessEqual_" + typeToString<I1>() + "_" + typeToString<I2>());
            }
            if constexpr (can_greater<I1, I2>::value != can_greater<VBase<I1>, VBase<I2>>::value) {
                unexpected_failed_compilations.push_back("binaryGreater_" + typeToString<I1>() + "_" + typeToString<I2>());
            }
            if constexpr (can_greater_equal<I1, I2>::value != can_greater_equal<VBase<I1>, VBase<I2>>::value) {
                unexpected_failed_compilations.push_back("binaryGreaterEqual_" + typeToString<I1>() + "_" + typeToString<I2>());
            }
            if constexpr (can_logical_and<I1, I2>::value != can_logical_and<VBase<I1>, VBase<I2>>::value) {
                unexpected_failed_compilations.push_back("logicalAnd_" + typeToString<I1>() + "_" + typeToString<I2>());
            }
            if constexpr (can_logical_or<I1, I2>::value != can_logical_or<VBase<I1>, VBase<I2>>::value) {
                unexpected_failed_compilations.push_back("logicalOr_" + typeToString<I1>() + "_" + typeToString<I2>());
            }
            if constexpr (can_bitwise_and<I1, I2>::value != can_bitwise_and<VBase<I1>, VBase<I2>>::value) {
                unexpected_failed_compilations.push_back("bitwiseAnd_" + typeToString<I1>() + "_" + typeToString<I2>());
            }
            if constexpr (can_bitwise_or<I1, I2>::value != can_bitwise_or<VBase<I1>, VBase<I2>>::value) {
                unexpected_failed_compilations.push_back("bitwiseOr_" + typeToString<I1>() + "_" + typeToString<I2>());
            }
            if constexpr (can_bitwise_xor<I1, I2>::value != can_bitwise_xor<VBase<I1>, VBase<I2>>::value) {
                unexpected_failed_compilations.push_back("bitwiseXor_" + typeToString<I1>() + "_" + typeToString<I2>());
            }
        }
    }

    template <typename I1, typename I2>
    void detectCompoundUnexpectedCompilationErrors() {
        if constexpr (fk::AreVVEqCN<I1, I2>::value || fk::AreVS<I1, I2>::value) {
            // The case scalar += vector is not supported (same for the other compound operators)
            if constexpr (can_add_assign<I1, I2>::value != can_add_assign<VBase<I1>, VBase<I2>>::value) {
                unexpected_failed_compilations.push_back("addAssign_" + typeToString<I1>() + "_" + typeToString<I2>());
            }
            if constexpr (can_sub_assign<I1, I2>::value != can_sub_assign<VBase<I1>, VBase<I2>>::value) {
                unexpected_failed_compilations.push_back("subAssign_" + typeToString<I1>() + "_" + typeToString<I2>());
            }
            if constexpr (can_mul_assign<I1, I2>::value != can_mul_assign<VBase<I1>, VBase<I2>>::value) {
                unexpected_failed_compilations.push_back("mulAssign_" + typeToString<I1>() + "_" + typeToString<I2>());
            }
            if constexpr (can_div_assign<I1, I2>::value != can_div_assign<VBase<I1>, VBase<I2>>::value) {
                unexpected_failed_compilations.push_back("divAssign_" + typeToString<I1>() + "_" + typeToString<I2>());
            }
            if constexpr (can_and_assign<I1, I2>::value != can_and_assign<VBase<I1>, VBase<I2>>::value) {
                unexpected_failed_compilations.push_back("andAssign_" + typeToString<I1>() + "_" + typeToString<I2>());
            }
            if constexpr (can_or_assign<I1, I2>::value != can_or_assign<VBase<I1>, VBase<I2>>::value) {
                unexpected_failed_compilations.push_back("orAssign_" + typeToString<I1>() + "_" + typeToString<I2>());
            }
        }
    }

    template <typename T>
    constexpr inline bool testUnaryMinus() {
        if constexpr (can_unary_minus<T>::value) {
            constexpr VBase<T> base_val{ static_cast<VBase<T>>(5) };
            constexpr T val = make_set<T>(base_val);
            constexpr auto result = -val; // Test unary minus
            constexpr auto expectedBase = -base_val;
            using ExpectedBaseType = std::decay_t<decltype(expectedBase)>;
            using ExpectedType = typename VectorType<ExpectedBaseType, cn<T>>::type_v;
            constexpr ExpectedType expectedResult = make_set<ExpectedType>(expectedBase);
            return expectedResult == result;
        } else {
            return true;
        }
    }

    template <typename T>
    constexpr inline bool testUnaryNot() {
        if constexpr (can_unary_not<T>::value) {
            constexpr VBase<T> base_val{ static_cast<VBase<T>>(5) };
            constexpr T val = make_set<T>(base_val);
            constexpr auto result = !val; // Test unary not
            constexpr auto expectedBase = !base_val;
            using ExpectedBaseType = std::decay_t<decltype(expectedBase)>;
            using ExpectedType = typename VectorType<ExpectedBaseType, cn<T>>::type_v;
            constexpr ExpectedType expectedResult = make_set<ExpectedType>(expectedBase);
            return expectedResult == result;
        } else {
            return true;
        }
    }

    template <typename T>
    constexpr inline bool testUnaryBitwiseNot() {
        if constexpr (can_unary_bitwise_not<T>::value) {
            constexpr VBase<T> base_val{ static_cast<VBase<T>>(5) };
            constexpr T val = make_set<T>(base_val);
            constexpr auto result = ~val; // Test unary bitwise not
            constexpr auto expectedBase = ~base_val;
            using ExpectedBaseType = std::decay_t<decltype(expectedBase)>;
            using ExpectedType = typename VectorType<ExpectedBaseType, cn<T>>::type_v;
            constexpr ExpectedType expectedResult = make_set<ExpectedType>(expectedBase);
            return expectedResult == result;
        } else {
            return true;
        }
    }
    template <typename T>
    bool testUnaryOperators() {
        bool correct{ true };
        if (!testUnaryMinus<T>()) {
            std::cout << "Failed unaryMinus test for type: " << typeToString<T>() << std::endl;
            correct = false;
        }
        if (!testUnaryNot<T>()) {
            std::cout << "Failed unaryNot test for type: " << typeToString<T>() << std::endl;
            correct = false;
        }
        if (!testUnaryBitwiseNot<T>()) {
            std::cout << "Failed unaryBitwiseNot test for type: " << typeToString<T>() << std::endl;
            correct = false;
        }
        return correct;
    }

#define BINARY_OP_TEST(OP_NAME, OP) \
    template <typename I1, typename I2> \
    constexpr inline bool binary ## OP_NAME() { \
        if constexpr (can_ ## OP_NAME<I1, I2>::value) { \
            constexpr VBase<I1> base_val1{ static_cast<VBase<I1>>(5) }; \
            constexpr VBase<I2> base_val2{ static_cast<VBase<I2>>(3) }; \
            constexpr I1 val1 = make_set<I1>(base_val1); \
            constexpr I2 val2 = make_set<I2>(base_val2); \
            constexpr auto result = val1 OP val2; \
            constexpr auto expectedBase = base_val1 OP base_val2; \
            using ExpectedBaseType = std::decay_t<decltype(expectedBase)>; \
            using ExpectedType = typename VectorType<ExpectedBaseType, (cn<I1> > cn<I2> ? cn<I1> : cn<I2>)>::type_v; \
            constexpr ExpectedType expectedResult = make_set<ExpectedType>(expectedBase); \
            return expectedResult == result; \
        } else { \
            return true; \
        } \
    }

BINARY_OP_TEST(add, +)
BINARY_OP_TEST(subtract, -)
BINARY_OP_TEST(multiply, *)
BINARY_OP_TEST(divide, /)
BINARY_OP_TEST(equal, ==)
BINARY_OP_TEST(not_equal, !=)
BINARY_OP_TEST(less, <)
BINARY_OP_TEST(less_equal, <=)
BINARY_OP_TEST(greater, >)
BINARY_OP_TEST(greater_equal, >=)
BINARY_OP_TEST(logical_and, &&)
BINARY_OP_TEST(logical_or, ||)
BINARY_OP_TEST(bitwise_and, &)
BINARY_OP_TEST(bitwise_or, |)
BINARY_OP_TEST(bitwise_xor, ^)

#undef BINARY_OP_TEST

    template <typename I1, typename I2>
    bool testBinaryOperators() {
        constexpr std::array<std::string_view, 15> binaryOperatorTestNames
        { "binaryAdd", "binaryMinus", "binaryMul", "binaryDiv",
          "binaryEqual", "binaryNotEqual", "binaryLess", "binaryLessEqual",
          "binaryGreater", "binaryGreaterEqual", "binaryLogicalAnd", "binaryLogicalOr",
          "binaryBitwiseAnd", "binaryBitwiseOr", "binaryBitwiseXor" };
        constexpr std::array<bool, 15> results = {
            binaryadd<I1, I2>(), binarysubtract<I1, I2>(), binarymultiply<I1, I2>(), binarydivide<I1, I2>(),
            binaryequal<I1, I2>(), binarynot_equal<I1, I2>(), binaryless<I1, I2>(),
            binaryless_equal<I1, I2>(), binarygreater<I1, I2>(), binarygreater_equal<I1, I2>(),
            binarylogical_and<I1, I2>(), binarylogical_or<I1, I2>(), binarybitwise_and<I1, I2>(),
            binarybitwise_or<I1, I2>(), binarybitwise_xor<I1, I2>()
        };
        bool correct{ true };
        for (int i = 0; i < 15; ++i) {
            if (!results[i]) {
                std::cout << "Failed " << binaryOperatorTestNames[i] << " test for types: "
                    << typeToString<I1>() << " and " << typeToString<I2>() << std::endl;
                correct = false;
            }
        }
        return correct;
    }

#define COMPOUND_OP_TEST(OP_NAME, OP) \
    template <typename I1, typename I2> \
    constexpr inline bool compound ## OP_NAME() { \
        if constexpr (can_ ## OP_NAME<I1, I2>::value) { \
            VBase<I1> base_val1{ static_cast<VBase<I1>>(5) }; \
            constexpr VBase<I2> base_val2{ static_cast<VBase<I2>>(3) }; \
            I1 val1 = make_set<I1>(base_val1); \
            constexpr I2 val2 = make_set<I2>(base_val2); \
            val1 OP val2; \
            base_val1 OP base_val2; \
            using ExpectedBaseType = VBase<I1>; \
            using ExpectedType = I1; \
            ExpectedType expectedResult = make_set<ExpectedType>(base_val1); \
            return expectedResult == val1; \
        } else { \
            return true; \
        } \
    }

COMPOUND_OP_TEST(add_assign, +=)
COMPOUND_OP_TEST(sub_assign, -=)
COMPOUND_OP_TEST(mul_assign, *=)
COMPOUND_OP_TEST(div_assign, /=)
COMPOUND_OP_TEST(and_assign, &=)
COMPOUND_OP_TEST(or_assign, |=)

#undef COMPOUND_OP_TEST

    // Test compound operators
    template <typename I1, typename I2>
    bool testCompoundOperators() {
        constexpr std::array<std::string_view, 6> compoundOperatorTestNames
        { "compoundAddAssign", "compoundSubAssign", "compoundMulAssign",
          "compoundDivAssign", "compoundAndAssign", "compoundOrAssign" };
        const std::array<bool, 6> results{ compoundadd_assign<I1, I2>(), compoundsub_assign<I1, I2>(),
                                           compoundmul_assign<I1, I2>(), compounddiv_assign<I1, I2>(),
                                           compoundand_assign<I1, I2>(), compoundor_assign<I1, I2>() };
        bool correct{ true };
        for (int i = 0; i < 6; ++i) {
            if (!results[i]) {
                std::cout << "Failed " << compoundOperatorTestNames[i] << " test for types: "
                    << typeToString<I1>() << " and " << typeToString<I2>() << std::endl;
                correct = false;
            }
        }
        return correct;
    }

    template <typename TypeList_>
    struct UnaryTest;

    template <typename... Types>
    struct UnaryTest<TypeList<Types...>> {
        static bool execute() {
            (detectUnaryUnexpectedCompilationErrors<Types>(), ...);
            return (testUnaryOperators<Types>() && ...);
        }
    };

    template <typename TypeList1, typename TypeList2>
    struct BinaryTests;

    template <typename Type1, typename... Types1, typename... Types2>
    struct BinaryTests<TypeList<Type1, Types1...>, TypeList<Types2...>> {
        static bool execute() {
            if constexpr (sizeof...(Types1) == 0) {
                (detectBinaryUnexpectedCompilationErrors<Type1, Types2>(), ...);
                return (testBinaryOperators<Type1, Types2>() && ...);
            } else {
                (detectBinaryUnexpectedCompilationErrors<Type1, Types2>(), ...);
                const bool result = (testBinaryOperators<Type1, Types2>() && ...);
                return result && BinaryTests<TypeList<Types1...>, TypeList<Types2...>>::execute();
            }
        }
    };

    template <typename TypeList1, typename TypeList2>
    struct CompoundTests;

    template <typename Type1, typename... Types1, typename... Types2>
    struct CompoundTests<TypeList<Type1, Types1...>, TypeList<Types2...>> {
        static bool execute() {
            if constexpr (sizeof...(Types1) == 0) {
                (detectCompoundUnexpectedCompilationErrors<Type1, Types2>(), ...);
                return (testCompoundOperators<Type1, Types2>() && ...);
            } else {
                (detectCompoundUnexpectedCompilationErrors<Type1, Types2>(), ...);
                const bool result = (testCompoundOperators<Type1, Types2>() && ...);
                return result && CompoundTests<TypeList<Types1...>, TypeList<Types2...>>::execute();
            }
        }
    };

} // namespace fk::test

int launch() {
    using namespace fk::test;

    bool passed = UnaryTest<VecAndStdTypes>::execute();
    passed &= BinaryTests<VecAndStdTypes, VecAndStdTypes>::execute();
    passed &= CompoundTests<VecAndStdTypes, VecAndStdTypes>::execute();

    if (!unexpected_failed_compilations.empty()) {
        std::cout << "ERROR: Unexpected compilation failures that did occur:\n";
        for (const auto& failure : unexpected_failed_compilations) {
            std::cout << "- " << failure << std::endl;
        }
        return -1;
    }

    return passed ? 0 : -1;
}

#endif // FK_UTEST_CUDA_VECTOR_UTILS_H