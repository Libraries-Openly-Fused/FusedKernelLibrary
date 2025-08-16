/* Copyright 2025 Oscar Amoros Huguet
   Copyright 2025 Albert Andaluz
   Copyright 2025 Grup Mediapro S.L.U

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_UTEST_CUDA_VECTOR_UTILS_H
#define FK_UTEST_CUDA_VECTOR_UTILS_H

#include <fused_kernel/core/utils/cuda_vector_utils.h>
#include <type_traits>
#include <iostream>
#include <vector>
#include <string>

namespace fk_test {

// Track compilation results
std::vector<std::string> failed_compilations;

// Helper to create test values
template<typename T>
constexpr T create_test_value() {
    if constexpr (std::is_same_v<T, bool>) {
        return true;
    } else if constexpr (std::is_floating_point_v<T>) {
        return T(3.14);
    } else if constexpr (std::is_signed_v<T>) {
        return T(-42);
    } else {
        return T(42);
    }
}

template<typename VecType>
constexpr VecType create_vector_test_value() {
    using BaseType = fk::VBase<VecType>;
    constexpr BaseType val = create_test_value<BaseType>();
    
    if constexpr (fk::cn<VecType> == 1) {
        return fk::make_<VecType>(val);
    } else if constexpr (fk::cn<VecType> == 2) {
        return fk::make_<VecType>(val, BaseType(val + 1));
    } else if constexpr (fk::cn<VecType> == 3) {
        return fk::make_<VecType>(val, BaseType(val + 1), BaseType(val + 2));
    } else {
        return fk::make_<VecType>(val, BaseType(val + 1), BaseType(val + 2), BaseType(val + 3));
    }
}

// SFINAE-based test helpers
template<typename T, typename = void>
struct can_unary_minus : std::false_type {};

template<typename T>
struct can_unary_minus<T, std::void_t<decltype(-std::declval<T>())>> : std::true_type {};

template<typename T, typename = void>
struct can_logical_not : std::false_type {};

template<typename T>
struct can_logical_not<T, std::void_t<decltype(!std::declval<T>())>> : std::true_type {};

template<typename T, typename = void>
struct can_bitwise_not : std::false_type {};

template<typename T>
struct can_bitwise_not<T, std::void_t<decltype(~std::declval<T>())>> : std::true_type {};

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

// Macro to simplify testing
#define TEST_UNARY_OP(OP_NAME, TYPE, TRAIT) \
    if constexpr (!TRAIT<TYPE>::value) { \
        failed_compilations.push_back(OP_NAME + std::string("_") + typeid(TYPE).name()); \
    }

#define TEST_BINARY_OP(OP_NAME, TYPE1, TYPE2, TRAIT) \
    if constexpr (!TRAIT<TYPE1, TYPE2>::value) { \
        failed_compilations.push_back(OP_NAME + std::string("_") + typeid(TYPE1).name() + "_" + typeid(TYPE2).name()); \
    }

// Test specific operators with specific types
void test_operators_bool1() {
    TEST_UNARY_OP("unary_minus", bool1, can_unary_minus);
    TEST_UNARY_OP("logical_not", bool1, can_logical_not);
    TEST_BINARY_OP("add", bool1, bool1, can_add);
    TEST_BINARY_OP("sub", bool1, bool1, can_subtract);
    TEST_BINARY_OP("mul", bool1, bool1, can_multiply);
    TEST_BINARY_OP("div", bool1, bool1, can_divide);
    TEST_BINARY_OP("eq", bool1, bool1, can_equal);
    TEST_BINARY_OP("ne", bool1, bool1, can_not_equal);
    TEST_BINARY_OP("lt", bool1, bool1, can_less);
    TEST_BINARY_OP("le", bool1, bool1, can_less_equal);
    TEST_BINARY_OP("gt", bool1, bool1, can_greater);
    TEST_BINARY_OP("ge", bool1, bool1, can_greater_equal);
    TEST_BINARY_OP("logical_and", bool1, bool1, can_logical_and);
    TEST_BINARY_OP("logical_or", bool1, bool1, can_logical_or);
    TEST_BINARY_OP("add_assign", bool1, bool1, can_add_assign);
    TEST_BINARY_OP("sub_assign", bool1, bool1, can_sub_assign);
    TEST_BINARY_OP("mul_assign", bool1, bool1, can_mul_assign);
    TEST_BINARY_OP("div_assign", bool1, bool1, can_div_assign);
}

void test_operators_char1() {
    TEST_UNARY_OP("unary_minus", char1, can_unary_minus);
    TEST_UNARY_OP("logical_not", char1, can_logical_not);
    TEST_UNARY_OP("bitwise_not", char1, can_bitwise_not);
    TEST_BINARY_OP("add", char1, char1, can_add);
    TEST_BINARY_OP("sub", char1, char1, can_subtract);
    TEST_BINARY_OP("mul", char1, char1, can_multiply);
    TEST_BINARY_OP("div", char1, char1, can_divide);
    TEST_BINARY_OP("eq", char1, char1, can_equal);
    TEST_BINARY_OP("ne", char1, char1, can_not_equal);
    TEST_BINARY_OP("lt", char1, char1, can_less);
    TEST_BINARY_OP("le", char1, char1, can_less_equal);
    TEST_BINARY_OP("gt", char1, char1, can_greater);
    TEST_BINARY_OP("ge", char1, char1, can_greater_equal);
    TEST_BINARY_OP("logical_and", char1, char1, can_logical_and);
    TEST_BINARY_OP("logical_or", char1, char1, can_logical_or);
    TEST_BINARY_OP("bitwise_and", char1, char1, can_bitwise_and);
    TEST_BINARY_OP("bitwise_or", char1, char1, can_bitwise_or);
    TEST_BINARY_OP("bitwise_xor", char1, char1, can_bitwise_xor);
    TEST_BINARY_OP("add_assign", char1, char1, can_add_assign);
    TEST_BINARY_OP("sub_assign", char1, char1, can_sub_assign);
    TEST_BINARY_OP("mul_assign", char1, char1, can_mul_assign);
    TEST_BINARY_OP("div_assign", char1, char1, can_div_assign);
    TEST_BINARY_OP("and_assign", char1, char1, can_and_assign);
    TEST_BINARY_OP("or_assign", char1, char1, can_or_assign);
}

void test_operators_uchar1() {
    TEST_UNARY_OP("unary_minus", uchar1, can_unary_minus);
    TEST_UNARY_OP("logical_not", uchar1, can_logical_not);
    TEST_UNARY_OP("bitwise_not", uchar1, can_bitwise_not);
    TEST_BINARY_OP("add", uchar1, uchar1, can_add);
    TEST_BINARY_OP("sub", uchar1, uchar1, can_subtract);
    TEST_BINARY_OP("mul", uchar1, uchar1, can_multiply);
    TEST_BINARY_OP("div", uchar1, uchar1, can_divide);
    TEST_BINARY_OP("eq", uchar1, uchar1, can_equal);
    TEST_BINARY_OP("ne", uchar1, uchar1, can_not_equal);
    TEST_BINARY_OP("lt", uchar1, uchar1, can_less);
    TEST_BINARY_OP("le", uchar1, uchar1, can_less_equal);
    TEST_BINARY_OP("gt", uchar1, uchar1, can_greater);
    TEST_BINARY_OP("ge", uchar1, uchar1, can_greater_equal);
    TEST_BINARY_OP("logical_and", uchar1, uchar1, can_logical_and);
    TEST_BINARY_OP("logical_or", uchar1, uchar1, can_logical_or);
    TEST_BINARY_OP("bitwise_and", uchar1, uchar1, can_bitwise_and);
    TEST_BINARY_OP("bitwise_or", uchar1, uchar1, can_bitwise_or);
    TEST_BINARY_OP("bitwise_xor", uchar1, uchar1, can_bitwise_xor);
    TEST_BINARY_OP("add_assign", uchar1, uchar1, can_add_assign);
    TEST_BINARY_OP("sub_assign", uchar1, uchar1, can_sub_assign);
    TEST_BINARY_OP("mul_assign", uchar1, uchar1, can_mul_assign);
    TEST_BINARY_OP("div_assign", uchar1, uchar1, can_div_assign);
    TEST_BINARY_OP("and_assign", uchar1, uchar1, can_and_assign);
    TEST_BINARY_OP("or_assign", uchar1, uchar1, can_or_assign);
}

void test_operators_int1() {
    TEST_UNARY_OP("unary_minus", int1, can_unary_minus);
    TEST_UNARY_OP("logical_not", int1, can_logical_not);
    TEST_UNARY_OP("bitwise_not", int1, can_bitwise_not);
    TEST_BINARY_OP("add", int1, int1, can_add);
    TEST_BINARY_OP("sub", int1, int1, can_subtract);
    TEST_BINARY_OP("mul", int1, int1, can_multiply);
    TEST_BINARY_OP("div", int1, int1, can_divide);
    TEST_BINARY_OP("eq", int1, int1, can_equal);
    TEST_BINARY_OP("ne", int1, int1, can_not_equal);
    TEST_BINARY_OP("lt", int1, int1, can_less);
    TEST_BINARY_OP("le", int1, int1, can_less_equal);
    TEST_BINARY_OP("gt", int1, int1, can_greater);
    TEST_BINARY_OP("ge", int1, int1, can_greater_equal);
    TEST_BINARY_OP("logical_and", int1, int1, can_logical_and);
    TEST_BINARY_OP("logical_or", int1, int1, can_logical_or);
    TEST_BINARY_OP("bitwise_and", int1, int1, can_bitwise_and);
    TEST_BINARY_OP("bitwise_or", int1, int1, can_bitwise_or);
    TEST_BINARY_OP("bitwise_xor", int1, int1, can_bitwise_xor);
    TEST_BINARY_OP("add_assign", int1, int1, can_add_assign);
    TEST_BINARY_OP("sub_assign", int1, int1, can_sub_assign);
    TEST_BINARY_OP("mul_assign", int1, int1, can_mul_assign);
    TEST_BINARY_OP("div_assign", int1, int1, can_div_assign);
    TEST_BINARY_OP("and_assign", int1, int1, can_and_assign);
    TEST_BINARY_OP("or_assign", int1, int1, can_or_assign);
}

void test_operators_float1() {
    TEST_UNARY_OP("unary_minus", float1, can_unary_minus);
    TEST_UNARY_OP("logical_not", float1, can_logical_not);
    TEST_BINARY_OP("add", float1, float1, can_add);
    TEST_BINARY_OP("sub", float1, float1, can_subtract);
    TEST_BINARY_OP("mul", float1, float1, can_multiply);
    TEST_BINARY_OP("div", float1, float1, can_divide);
    TEST_BINARY_OP("eq", float1, float1, can_equal);
    TEST_BINARY_OP("ne", float1, float1, can_not_equal);
    TEST_BINARY_OP("lt", float1, float1, can_less);
    TEST_BINARY_OP("le", float1, float1, can_less_equal);
    TEST_BINARY_OP("gt", float1, float1, can_greater);
    TEST_BINARY_OP("ge", float1, float1, can_greater_equal);
    TEST_BINARY_OP("logical_and", float1, float1, can_logical_and);
    TEST_BINARY_OP("logical_or", float1, float1, can_logical_or);
    TEST_BINARY_OP("add_assign", float1, float1, can_add_assign);
    TEST_BINARY_OP("sub_assign", float1, float1, can_sub_assign);
    TEST_BINARY_OP("mul_assign", float1, float1, can_mul_assign);
    TEST_BINARY_OP("div_assign", float1, float1, can_div_assign);
}

void test_operators_int2() {
    TEST_UNARY_OP("unary_minus", int2, can_unary_minus);
    TEST_UNARY_OP("logical_not", int2, can_logical_not);
    TEST_UNARY_OP("bitwise_not", int2, can_bitwise_not);
    TEST_BINARY_OP("add", int2, int2, can_add);
    TEST_BINARY_OP("sub", int2, int2, can_subtract);
    TEST_BINARY_OP("mul", int2, int2, can_multiply);
    TEST_BINARY_OP("div", int2, int2, can_divide);
    TEST_BINARY_OP("eq", int2, int2, can_equal);
    TEST_BINARY_OP("ne", int2, int2, can_not_equal);
    TEST_BINARY_OP("lt", int2, int2, can_less);
    TEST_BINARY_OP("le", int2, int2, can_less_equal);
    TEST_BINARY_OP("gt", int2, int2, can_greater);
    TEST_BINARY_OP("ge", int2, int2, can_greater_equal);
    TEST_BINARY_OP("logical_and", int2, int2, can_logical_and);
    TEST_BINARY_OP("logical_or", int2, int2, can_logical_or);
    TEST_BINARY_OP("bitwise_and", int2, int2, can_bitwise_and);
    TEST_BINARY_OP("bitwise_or", int2, int2, can_bitwise_or);
    TEST_BINARY_OP("bitwise_xor", int2, int2, can_bitwise_xor);
    TEST_BINARY_OP("add_assign", int2, int2, can_add_assign);
    TEST_BINARY_OP("sub_assign", int2, int2, can_sub_assign);
    TEST_BINARY_OP("mul_assign", int2, int2, can_mul_assign);
    TEST_BINARY_OP("div_assign", int2, int2, can_div_assign);
    TEST_BINARY_OP("and_assign", int2, int2, can_and_assign);
    TEST_BINARY_OP("or_assign", int2, int2, can_or_assign);
}

void test_operators_float4() {
    TEST_UNARY_OP("unary_minus", float4, can_unary_minus);
    TEST_UNARY_OP("logical_not", float4, can_logical_not);
    TEST_BINARY_OP("add", float4, float4, can_add);
    TEST_BINARY_OP("sub", float4, float4, can_subtract);
    TEST_BINARY_OP("mul", float4, float4, can_multiply);
    TEST_BINARY_OP("div", float4, float4, can_divide);
    TEST_BINARY_OP("eq", float4, float4, can_equal);
    TEST_BINARY_OP("ne", float4, float4, can_not_equal);
    TEST_BINARY_OP("lt", float4, float4, can_less);
    TEST_BINARY_OP("le", float4, float4, can_less_equal);
    TEST_BINARY_OP("gt", float4, float4, can_greater);
    TEST_BINARY_OP("ge", float4, float4, can_greater_equal);
    TEST_BINARY_OP("logical_and", float4, float4, can_logical_and);
    TEST_BINARY_OP("logical_or", float4, float4, can_logical_or);
    TEST_BINARY_OP("add_assign", float4, float4, can_add_assign);
    TEST_BINARY_OP("sub_assign", float4, float4, can_sub_assign);
    TEST_BINARY_OP("mul_assign", float4, float4, can_mul_assign);
    TEST_BINARY_OP("div_assign", float4, float4, can_div_assign);
}

// Test vector-scalar combinations
void test_vector_scalar_operations() {
    // Vector with scalar
    TEST_BINARY_OP("int2_scalar_add", int2, int, can_add);
    TEST_BINARY_OP("int2_scalar_sub", int2, int, can_subtract);
    TEST_BINARY_OP("int2_scalar_mul", int2, int, can_multiply);
    TEST_BINARY_OP("int2_scalar_div", int2, int, can_divide);
    TEST_BINARY_OP("int2_scalar_eq", int2, int, can_equal);
    TEST_BINARY_OP("int2_scalar_ne", int2, int, can_not_equal);
    TEST_BINARY_OP("int2_scalar_lt", int2, int, can_less);
    
    // Scalar with vector
    TEST_BINARY_OP("scalar_int2_add", int, int2, can_add);
    TEST_BINARY_OP("scalar_int2_sub", int, int2, can_subtract);
    TEST_BINARY_OP("scalar_int2_mul", int, int2, can_multiply);
    TEST_BINARY_OP("scalar_int2_div", int, int2, can_divide);
    TEST_BINARY_OP("scalar_int2_eq", int, int2, can_equal);
    
    // Float vector with float scalar
    TEST_BINARY_OP("float2_scalar_add", float2, float, can_add);
    TEST_BINARY_OP("float2_scalar_sub", float2, float, can_subtract);
    TEST_BINARY_OP("float2_scalar_mul", float2, float, can_multiply);
    TEST_BINARY_OP("float2_scalar_div", float2, float, can_divide);
    
    // Compound assignments
    TEST_BINARY_OP("int2_scalar_add_assign", int2, int, can_add_assign);
    TEST_BINARY_OP("float2_scalar_add_assign", float2, float, can_add_assign);
}

// Test mixed type operations  
void test_mixed_type_operations() {
    // Same channels, different types
    TEST_BINARY_OP("int2_float2_add", int2, float2, can_add);
    TEST_BINARY_OP("int2_bool2_add", int2, bool2, can_add);
    TEST_BINARY_OP("char2_int2_add", char2, int2, can_add);
    TEST_BINARY_OP("float2_bool2_eq", float2, bool2, can_equal);
    
    // Different channels should fail
    TEST_BINARY_OP("int2_int3_add", int2, int3, can_add);
    
    // Test some operations that should definitely fail
    TEST_BINARY_OP("bitwise_and_float_float", float2, float2, can_bitwise_and);  // floats don't support bitwise ops
    TEST_UNARY_OP("bitwise_not", float1, can_bitwise_not);  // floats don't support bitwise ops
    TEST_BINARY_OP("bitwise_and_double_double", double2, double2, can_bitwise_and);  // doubles don't support bitwise ops
    TEST_UNARY_OP("bitwise_not", double1, can_bitwise_not);  // doubles don't support bitwise ops
}

// Test additional vector types
void test_additional_vector_types() {
    // Test more vector types
    TEST_UNARY_OP("unary_minus", short1, can_unary_minus);
    TEST_UNARY_OP("unary_minus", ushort1, can_unary_minus);
    TEST_UNARY_OP("unary_minus", uint1, can_unary_minus);
    TEST_UNARY_OP("unary_minus", long1, can_unary_minus);
    TEST_UNARY_OP("unary_minus", ulong1, can_unary_minus);
    TEST_UNARY_OP("unary_minus", longlong1, can_unary_minus);
    TEST_UNARY_OP("unary_minus", ulonglong1, can_unary_minus);
    TEST_UNARY_OP("unary_minus", double1, can_unary_minus);
    
    // Test 3-channel vectors
    TEST_UNARY_OP("unary_minus", int3, can_unary_minus);
    TEST_UNARY_OP("logical_not", float3, can_logical_not);
    TEST_UNARY_OP("bitwise_not", char3, can_bitwise_not);
    
    // Test 4-channel vectors
    TEST_UNARY_OP("unary_minus", double4, can_unary_minus);
    TEST_UNARY_OP("logical_not", bool4, can_logical_not);
    TEST_UNARY_OP("bitwise_not", uint4, can_bitwise_not);
    
    // Test binary operations with different channel counts (should fail)
    TEST_BINARY_OP("int1_int4_add", int1, int4, can_add);
    TEST_BINARY_OP("float2_float3_add", float2, float3, can_add);
    
    // Test some working combinations
    TEST_BINARY_OP("short3_short3_add", short3, short3, can_add);
    TEST_BINARY_OP("double4_double4_mul", double4, double4, can_multiply);
}

} // namespace fk_test

int launch() {
    using namespace fk_test;
    
    std::cout << "Starting comprehensive CUDA vector utils operator tests...\n";
    
    // Clear previous results
    failed_compilations.clear();
    
    // Test individual types
    std::cout << "Testing bool1 operators...\n";
    test_operators_bool1();
    
    std::cout << "Testing char1 operators...\n";
    test_operators_char1();
    
    std::cout << "Testing uchar1 operators...\n";
    test_operators_uchar1();
    
    std::cout << "Testing int1 operators...\n";
    test_operators_int1();
    
    std::cout << "Testing float1 operators...\n";
    test_operators_float1();
    
    std::cout << "Testing int2 operators...\n";
    test_operators_int2();
    
    std::cout << "Testing float4 operators...\n";
    test_operators_float4();
    
    std::cout << "Testing vector-scalar operations...\n";
    test_vector_scalar_operations();
    
    std::cout << "Testing mixed type operations...\n";
    test_mixed_type_operations();
    
    std::cout << "Testing additional vector types...\n";
    test_additional_vector_types();
    
    // Report results
    std::cout << "\n=== TEST RESULTS ===\n";
    std::cout << "Total failed compilations: " << failed_compilations.size() << "\n";
    
    if (!failed_compilations.empty()) {
        std::cout << "\n=== TESTS THAT DID NOT COMPILE ===\n";
        for (const auto& test_name : failed_compilations) {
            std::cout << "- " << test_name << "\n";
        }
    } else {
        std::cout << "All tests compiled successfully!\n";
    }
    
    return 0;
}

#endif // FK_UTEST_CUDA_VECTOR_UTILS_H