/* Copyright 2024-2026 Oscar Amoros Huguet

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

#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/resize.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>

namespace fk {

template <typename Restriction, typename T>
struct CheckCompliance {
    static constexpr bool value = Restriction::template complies<T>();
};

template <typename Restriction, typename TypeList_>
struct AllIOpsComply;

template <typename Restriction, typename... Types>
struct AllIOpsComply<Restriction, TypeList<Types...>> {
    static constexpr bool value = and_v<(CheckCompliance<Restriction, typename Types::InstanceType>::value)...>;
};

int launch_impl() {
    using ReadDummy = PerThreadRead<ND::_2D, int>;
    using ReadBackDummy = ResizeComplete<AspectRatio::IGNORE_AR, Ternary<InterpolateComplete<InterpolationType::INTER_LINEAR, Read<ReadDummy>>>>;
    using BinaryDummy = Add<int>;
    using TernaryDummy = InterpolateComplete<InterpolationType::INTER_LINEAR, Read<ReadDummy>>;
    using WriteDummy = PerThreadWrite<ND::_2D, int>;

    using DFList = TypeList<Read<ReadDummy>, ReadBack<ReadBackDummy>,
                   Read<ReadDummy>, ReadBack<ReadBackDummy>,
                   Binary<BinaryDummy>, Ternary<TernaryDummy>,
                   MidWrite<WriteDummy>, Write<WriteDummy>>;

    constexpr bool correctDFRestrict = AllIOpsComply<NotIsUnaryRestriction, DFList>::value;
    static_assert(correctDFRestrict, "The list of operations does not comply with the restriction");

    using ListToCheck =
        TypeList<Read<PerThreadRead<ND::_2D, uchar>>,
                     Binary<Add<float>>,
                     Unary<Cast<float, int>>,
                     Ternary<InterpolateComplete<InterpolationType::INTER_LINEAR, Read<PerThreadRead<ND::_2D, uchar>>>>,
                     Unary<Cast<float, int>>,
                     MidWrite<PerThreadWrite<ND::_2D, uchar>>,
                     Binary<Mul<float>>,
                     Write<PerThreadWrite<ND::_2D, uchar>>>;

    using IndexList = filtered_index_sequence_t<NotIsUnaryRestriction, ListToCheck>;
    using IntegerList = filtered_integer_sequence_t<int, NotIsUnaryRestriction, ListToCheck>;

    constexpr IndexList indexList{};

    static_assert(IndexList::size() == 6, "The index list does not have the expected size");
    constexpr size_t var = get_index_f<0>(indexList);
    static_assert(var == 0, "Incorrect index");
    static_assert(get_index<1, IndexList> == 1, "Incorrect index");
    static_assert(get_index<2, IndexList> == 3, "Incorrect index");
    static_assert(get_index<3, IndexList> == 5, "Incorrect index");
    static_assert(get_index<4, IndexList> == 6, "Incorrect index");
    static_assert(get_index<5, IndexList> == 7, "Incorrect index");

    static_assert(IntegerList::size() == 6, "The integer list does not have the expected size");
    static_assert(get_integer<int, 0, IntegerList> == 0, "Incorrect integer");
    static_assert(get_integer<int, 1, IntegerList> == 1, "Incorrect integer");
    static_assert(get_integer<int, 2, IntegerList> == 3, "Incorrect integer");
    static_assert(get_integer<int, 3, IntegerList> == 5, "Incorrect integer");
    static_assert(get_integer<int, 4, IntegerList> == 6, "Incorrect integer");
    static_assert(get_integer<int, 5, IntegerList> == 7, "Incorrect integer");

    using FirstUnary =
        TypeList<Unary<Cast<float, int>>,
        Ternary<InterpolateComplete<InterpolationType::INTER_LINEAR, Read<PerThreadRead<ND::_2D, uchar>>>>,
        Unary<Cast<float, int>>,
        MidWrite<PerThreadWrite<ND::_2D, uchar>>,
        Binary<Mul<float>>,
        Write<PerThreadWrite<ND::_2D, uchar>>>;

    using IndexListFirstUnary = filtered_index_sequence_t<NotIsUnaryRestriction, FirstUnary>;

    static_assert(IndexListFirstUnary::size() == 4, "Incorrect sequence size");
    static_assert(get_index<0, IndexListFirstUnary> == 1, "Incorrect index");
    static_assert(get_index<1, IndexListFirstUnary> == 3, "Incorrect index");
    static_assert(get_index<2, IndexListFirstUnary> == 4, "Incorrect index");
    static_assert(get_index<3, IndexListFirstUnary> == 5, "Incorrect index");

    using FirstAndLastUnary =
        TypeList<Unary<Cast<float, int>>,
        Ternary<InterpolateComplete<InterpolationType::INTER_LINEAR, Read<PerThreadRead<ND::_2D, uchar>>>>,
        Unary<Cast<float, int>>>;

    using IndexListFirstAndLastUnary = filtered_index_sequence_t<NotIsUnaryRestriction, FirstAndLastUnary>;

    static_assert(IndexListFirstAndLastUnary::size() == 1, "Incorrect sequence size");
    static_assert(get_index<0, IndexListFirstAndLastUnary> == 1, "Incorrect index");

    using LastUnary = TypeList<Ternary<InterpolateComplete<InterpolationType::INTER_LINEAR, Read<PerThreadRead<ND::_2D, uchar>>>>,
        Unary<Cast<float, int>>,
        MidWrite<PerThreadWrite<ND::_2D, uchar>>,
        Binary<Mul<float>>,
        Unary<Cast<float, int>>>;

    using IndexListLastUnary = filtered_index_sequence_t<NotIsUnaryRestriction, LastUnary>;

    static_assert(IndexListLastUnary::size() == 3, "Incorrect sequence size");
    static_assert(get_index<0, IndexListLastUnary> == 0, "Incorrect index");
    static_assert(get_index<1, IndexListLastUnary> == 2, "Incorrect index");
    static_assert(get_index<2, IndexListLastUnary> == 3, "Incorrect index");

    using AllTypesUnary = TypeList<Unary<Cast<float, int>>,
                                       Unary<Cast<float, int>>,
                                       Unary<Cast<float, int>>,
                                       Unary<Cast<float, int>>>;

    using IndexListAllUnary = filtered_index_sequence_t<NotIsUnaryRestriction, AllTypesUnary>;

    static_assert(IndexListAllUnary::size() == 0, "Incorrect index");

    return 0;
}

} // namespace fk

int launch() {
    return fk::launch_impl();
}
