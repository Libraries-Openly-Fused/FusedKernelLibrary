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

#ifndef FK_USTEST_OPERATION_TYPES_H
#define FK_USTEST_OPERATION_TYPES_H

#include <fused_kernel/core/execution_model/operation_model/operation_types.h>

template <typename ExpectedRDT, typename ExpectedPT, typename ExpectedOT, typename ReadOpAlias>
constexpr inline void testReadOp(const ReadOpAlias&) {
    static_assert(std::is_same_v<typename ReadOpAlias::ReadDataType, ExpectedRDT>, "Wrong ReadDataType");
    static_assert(std::is_same_v<typename ReadOpAlias::ParamsType, ExpectedPT>, "Wrong ParamsType");
    static_assert(std::is_same_v<typename ReadOpAlias::OutputType, ExpectedOT>, "Wrong OutputType");
}

template <typename ExpectedRDT, typename ExpectedPT, typename ExpectedBIOpT, typename ExpectedOT, typename ReadBackOpAlias>
constexpr inline void testReadBackOp(const ReadBackOpAlias&) {
    static_assert(std::is_same_v<typename ReadBackOpAlias::ReadDataType, ExpectedRDT>, "Wrong ReadDataType");
    static_assert(std::is_same_v<typename ReadBackOpAlias::ParamsType, ExpectedPT>, "Wrong ParamsType");
    static_assert(std::is_same_v<typename ReadBackOpAlias::OutputType, ExpectedOT>, "Wrong OutputType");
    static_assert(std::is_same_v<typename ReadBackOpAlias::BackIOp, ExpectedBIOpT>, "Wrong BackIOp");
}

template <typename ExpectedIT, typename ExpectedOT, typename UnaryOpAlias>
constexpr inline void testUnaryOp(const UnaryOpAlias&) {
    static_assert(std::is_same_v<typename UnaryOpAlias::InputType, ExpectedIT>, "Wrong InputType");
    static_assert(std::is_same_v<typename UnaryOpAlias::OutputType, ExpectedOT>, "Wrong OutputType");
}

template <typename ExpectedIT, typename ExpectedPT, typename ExpectedOT, typename BinaryOpAlias>
constexpr inline void testBinaryOp(const BinaryOpAlias&) {
    static_assert(std::is_same_v<typename BinaryOpAlias::InputType, ExpectedIT>, "Wrong InputType");
    static_assert(std::is_same_v<typename BinaryOpAlias::ParamsType, ExpectedPT>, "Wrong ParamsType");
    static_assert(std::is_same_v<typename BinaryOpAlias::OutputType, ExpectedOT>, "Wrong OutputType");
}

template <typename ExpectedIT, typename ExpectedPT, typename ExpectedBIOpT, typename ExpectedOT, typename TernaryOpAlias>
constexpr inline void testTernaryOp(const TernaryOpAlias&) {
    static_assert(std::is_same_v<typename TernaryOpAlias::InputType, ExpectedIT>, "Wrong InputType");
    static_assert(std::is_same_v<typename TernaryOpAlias::ParamsType, ExpectedPT>, "Wrong ParamsType");
    static_assert(std::is_same_v<typename TernaryOpAlias::OutputType, ExpectedOT>, "Wrong OutputType");
    static_assert(std::is_same_v<typename TernaryOpAlias::BackIOp, ExpectedBIOpT>, "Wrong BackIOp");
}

template <typename ExpectedIT, typename ExpectedPT, typename ExpectedWDT, typename WriteOpAlias>
constexpr inline void testWriteOp(const WriteOpAlias&) {
    static_assert(std::is_same_v<typename WriteOpAlias::InputType, ExpectedIT>, "Wrong InputType");
    static_assert(std::is_same_v<typename WriteOpAlias::ParamsType, ExpectedPT>, "Wrong ParamsType");
    static_assert(std::is_same_v<typename WriteOpAlias::WriteDataType, ExpectedWDT>, "Wrong WriteDataType");
}

template <typename ExpectedIT, typename ExpectedPT, typename ExpectedWDT, typename ExpectedOT, typename MidWriteOpAlias>
constexpr inline void testMidWriteOp(const MidWriteOpAlias&) {
    static_assert(std::is_same_v<typename MidWriteOpAlias::InputType, ExpectedIT>, "Wrong InputType");
    static_assert(std::is_same_v<typename MidWriteOpAlias::ParamsType, ExpectedPT>, "Wrong ParamsType");
    static_assert(std::is_same_v<typename MidWriteOpAlias::OutputType, ExpectedOT>, "Wrong OutputType");
    static_assert(std::is_same_v<typename MidWriteOpAlias::WriteDataType, ExpectedWDT>, "Wrong WriteDataType");
}

int launch() {
    using namespace fk;

    testReadOp<int, NullType, double>(ReadOp<RDT<int>, OT<double>>());
    testReadOp<float, NullType, NullType>(ReadOp<RDT<float>>());
    testReadOp<NullType, NullType, NullType>(ReadOp<>());
    testReadOp<NullType, uchar, NullType>(ReadOp<PT<uchar>>());
    testReadOp<int, int, int>(ReadOp<PT<int>, OT<int>, RDT<int>>());

    testReadBackOp<int, float, NullType, double>(ReadBackOp<RDT<int>, PT<float>, OT<double>>());
    testReadBackOp<NullType, NullType, NullType, NullType>(ReadBackOp<>());
    testReadBackOp<NullType, NullType, int, NullType>(ReadBackOp<BIOpT<int>>());
    testReadBackOp<int, float, double, uchar>(ReadBackOp<RDT<int>, PT<float>, OT<uchar>, BIOpT<double>>());

    testUnaryOp<int, float>(UnaryOp<IT<int>, OT<float>>());
    testUnaryOp<NullType, NullType>(UnaryOp<>());
    testUnaryOp<uchar, NullType>(UnaryOp<IT<uchar>>());
    testUnaryOp<NullType, double>(UnaryOp<OT<double>>());

    testBinaryOp<int, float, double>(BinaryOp<IT<int>, PT<float>, OT<double>>());
    testBinaryOp<NullType, NullType, NullType>(BinaryOp<>());
    testBinaryOp<uchar, NullType, NullType>(BinaryOp<IT<uchar>>());
    testBinaryOp<NullType, int, NullType>(BinaryOp<PT<int>>());
    testBinaryOp<NullType, NullType, double>(BinaryOp<OT<double>>());

    testTernaryOp<int, float, double, uchar>(TernaryOp<IT<int>, PT<float>, OT<uchar>, BIOpT<double>>());
    testTernaryOp<NullType, NullType, NullType, NullType>(TernaryOp<>());
    testTernaryOp<uchar, NullType, NullType, NullType>(TernaryOp<IT<uchar>>());
    testTernaryOp<NullType, int, NullType, NullType>(TernaryOp<PT<int>>());
    testTernaryOp<NullType, NullType, NullType, double>(TernaryOp<OT<double>>());
    testTernaryOp<NullType, NullType, float, NullType>(TernaryOp<BIOpT<float>>());

    testWriteOp<int, float, double>(WriteOp<IT<int>, PT<float>, WDT<double>>());
    testWriteOp<NullType, NullType, NullType>(WriteOp<>());
    testWriteOp<uchar, NullType, NullType>(WriteOp<IT<uchar>>());
    testWriteOp<NullType, int, NullType>(WriteOp<PT<int>>());
    testWriteOp<NullType, NullType, double>(WriteOp<WDT<double>>());
    testWriteOp<int, int, int>(WriteOp<IT<int>, PT<int>, WDT<int>>());

    testMidWriteOp<int, float, double, int>(MidWriteOp<IT<int>, PT<float>, OT<int>, WDT<double>>());
    testMidWriteOp<NullType, NullType, NullType, NullType>(MidWriteOp<>());
    testMidWriteOp<uchar, NullType, NullType, NullType>(MidWriteOp<IT<uchar>>());
    testMidWriteOp<NullType, int, NullType, NullType>(MidWriteOp<PT<int>>());
    testMidWriteOp<NullType, NullType, double, NullType>(MidWriteOp<WDT<double>>());
    testMidWriteOp<NullType, NullType, NullType, int>(MidWriteOp<OT<int>>());
    testMidWriteOp<int, int, int, int>(MidWriteOp<IT<int>, PT<int>, OT<int>, WDT<int>>());

    return 0;
}

#endif // FK_USTEST_OPERATION_TYPES_H