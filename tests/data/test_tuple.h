/* Copyright 2023-2024 Mediaproduccion S.L.U. (Oscar Amoros Huguet)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include <tests/main.h>

#include <iostream>

#include <fused_kernel/core/execution_model/operation_model/operation_tuple.h>
#include <fused_kernel/algorithms/basic_ops/vector_ops.h>
#include <fused_kernel/core/execution_model/memory_operations.h>

constexpr bool buildTuple() {
    constexpr fk::Tuple<int, float, double, float3> test{1, 4.f, 5.0, {4.f, 3.f, 1.f}};

    constexpr bool result1 = fk::TupleUtil::get<0>(test) == 1;
    constexpr bool result2 = fk::TupleUtil::get<1>(test) == 4.f;
    constexpr bool result3 = fk::TupleUtil::get<2>(test) == 5.0;
    constexpr float3 temp = fk::TupleUtil::get<3>(test);
    constexpr bool result4 = (temp.x == 4.f) && (temp.y == 3.f) && (temp.z == 1.f);

    return fk::and_v<result1, result2, result3, result4>;
}

constexpr bool buildOperationTupleType() {
    using Op1 = fk::PerThreadRead<fk::ND::_2D, uchar3>;
    using Op2 = fk::VectorReorder<uchar3, 0, 1, 2>;
    using Op3 = fk::PerThreadWrite<fk::ND::_2D, uchar3>;

    using TupleType = fk::OperationTuple<Op1, Op2, Op3>;

    constexpr bool result1 = std::is_same_v<fk::get_type_t<0, TupleType>, Op1>;
    constexpr bool result2 = std::is_same_v<fk::get_type_t<1, TupleType>, Op2>;
    constexpr bool result3 = std::is_same_v<fk::get_type_t<2, TupleType>, Op3>;

    return fk::and_v<result1, result2, result3>;
}

constexpr bool tupleCat() {
    using Tuple1 = fk::Tuple<int, float>;
    using Tuple2 = fk::Tuple<char, double>;
    constexpr Tuple1 tuple1{ 1, 1.f };
    constexpr Tuple2 tuple2{ 1u, 1.0 };

    constexpr auto myTuple = fk::cat(tuple1, tuple2);

    return fk::and_v<fk::get<0>(myTuple) == 1,
                     fk::get<1>(myTuple) == 1.f,
                     fk::get<2>(myTuple) == 1u,
                     fk::get<3>(myTuple) == 1.0>;
}

constexpr bool tupleInsert() {
#if defined(_MSC_VER) && _MSC_VER >= 1910 && _MSC_VER <= 1916
    // We will remove support for this MSC compilers soon, so we skip this test
    return true;
#else
    constexpr auto myTuple = fk::TupleUtil::cat(fk::Tuple<int>{1}, fk::Tuple<char>{1u});

    return fk::and_v<fk::get<0>(fk::tuple_insert<0, float>(2.f, myTuple)) == 2.f,
                     fk::get<1>(fk::tuple_insert<1, uchar>(240u, myTuple)) == 240u,
                     fk::get<2>(fk::tuple_insert<2, double>(23.0, myTuple)) == 23.0>;
#endif
}

bool modifyTupleElement() {
    fk::Tuple<int, float, double> myTuple{1, 1.f, 1.0};

    fk::get<0>(myTuple) += 1;
    fk::get<1>(myTuple) -= 0.5f;
    fk::get<2>(myTuple) += 2.0;

    return (fk::get<0>(myTuple) == 2) &&
           (fk::get<1>(myTuple) == 0.5f) &&
           (fk::get<2>(myTuple) == 3.0);
}

int launch() {
    static_assert(buildTuple(), "Failed buildTuple test");
    static_assert(buildOperationTupleType(), "Failed buildOperationTupleType test");
    static_assert(tupleCat(), "Failed tupleCat test");
    static_assert(tupleInsert(), "Failed tupleInsert test");

    if (modifyTupleElement()) {
        std::cout << "test_tuple Passed!!" << std::endl;
        return 0;
    } else {
        std::cout << "test_tuple Failed!!" << std::endl;
        return -1;
    }
}
