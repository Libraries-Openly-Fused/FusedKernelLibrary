/* Copyright 2024 Oscar Amoros Huguet

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

#include <fused_kernel/core/utils/type_lists.h>
#include <fused_kernel/core/utils/template_operations.h>
#include <fused_kernel/core/data/vector_types.h>

template <typename T>
struct DummyTemplateType {};

template <typename T>
using DTT = DummyTemplateType<T>;

int launch() {

    using InitialTL = fk::TypeList<int>;

    static_assert(InitialTL::size == 1, "Wrong TypeList size");

    using TL2 = fk::InsertTypeBack_t<InitialTL, float>;

    {
        static_assert(TL2::size == 2, "Wrong TypeList size");
        constexpr bool firstType = std::is_same_v<int, fk::TypeAt_t<0, TL2>>;
        constexpr bool secondType = std::is_same_v<float, fk::TypeAt_t<1, TL2>>;
        static_assert(firstType && secondType, "Unexpected types in TypeList");
    }

    using TL3 = fk::InsertTypeBack_t<InitialTL, DTT<float>>;

    {
        static_assert(TL3::size == 2, "Wrong TypeList size");
        constexpr bool firstType = std::is_same_v<int, fk::TypeAt_t<0, TL3>>;
        constexpr bool secondType = std::is_same_v<DTT<float>, fk::TypeAt_t<1, TL3>>;
        static_assert(firstType && secondType, "Unexpected types in TypeList");
    }

    using TL4 = fk::TypeListCat_t<TL2, TL3>;

    {
        static_assert(TL4::size == 4, "Wrong TypeList size");
        constexpr bool firstType = std::is_same_v<int, fk::TypeAt_t<0, TL4>>;
        constexpr bool secondType = std::is_same_v<float, fk::TypeAt_t<1, TL4>>;
        constexpr bool thirdType = std::is_same_v<int, fk::TypeAt_t<2, TL4>>;
        constexpr bool fourthType = std::is_same_v<DTT<float>, fk::TypeAt_t<3, TL4>>;
        static_assert(fk::and_v<firstType, secondType, thirdType, fourthType>,
                      "Unexpected types in TypeList");
    }

    // double, uchar, int, float, DTT<DTT<double>>, int, DTT<float>
    using TL5_0 = fk::InsertTypeFront_t<double, TL4>;
    using TL5_1 = fk::InsertType_t<1, uchar, TL5_0>;
    using TL5 = fk::InsertType_t<4, DTT<DTT<double>>, TL5_1>;

    {
        static_assert(TL5::size == 7, "Wrong TypeList size");
        constexpr bool firstType = std::is_same_v<double, fk::TypeAt_t<0, TL5>>;
        constexpr bool secondType = std::is_same_v<uchar, fk::TypeAt_t<1, TL5>>;
        constexpr bool thirdType = std::is_same_v<int, fk::TypeAt_t<2, TL5>>;
        constexpr bool fourthType = std::is_same_v<float, fk::TypeAt_t<3, TL5>>;
        constexpr bool fifthType = std::is_same_v<DTT<DTT<double>>, fk::TypeAt_t<4, TL5>>;
        constexpr bool sixthType = std::is_same_v<int, fk::TypeAt_t<5, TL5>>;
        constexpr bool seventhType = std::is_same_v<DTT<float>, fk::TypeAt_t<6, TL5>>;
        static_assert(fk::and_v<firstType,
                                secondType,
                                thirdType,
                                fourthType,
                                fifthType,
                                sixthType,
                                seventhType>, "Unexpected types in TypeList");
    }

    using TL6 = fk::TypeList<int, char, float, double, uchar3>;
    using TL6_0 = TL6::at<0>;
    using TL6_1 = TL6::at<1>;
    using TL6_2 = TL6::at<2>;
    using TL6_3 = TL6::at<3>;
    using TL6_4 = TL6::at<4>;
    static_assert(std::is_same_v<int, TL6_0>, "Unexpected type");
    static_assert(std::is_same_v<char, TL6_1>, "Unexpected type");
    static_assert(std::is_same_v<float, TL6_2>, "Unexpected type");
    static_assert(std::is_same_v<double, TL6_3>, "Unexpected type");
    static_assert(std::is_same_v<uchar3, TL6_4>, "Unexpected type");

    return 0;
}