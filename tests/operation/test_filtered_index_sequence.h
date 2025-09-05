/* Copyright 2024-2025 Oscar Amoros Huguet

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

#include <fused_kernel/core/execution_model/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/resize.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>

template <typename Restriction, typename... ListTypes>
constexpr bool allInstantiableOperationsComplieWith(const fk::TypeList<ListTypes...>& tl) {
    return fk::and_v<(Restriction::template complies<typename ListTypes::InstanceType>())...>;
}

int launch() {
    using ReadDummy = fk::PerThreadRead<fk::ND::_2D, int>;
    using ReadBackDummy = fk::ResizeComplete<fk::AspectRatio::IGNORE_AR, fk::Ternary<fk::InterpolateComplete<fk::InterpolationType::INTER_LINEAR, fk::Read<ReadDummy>>>>;
    using BinaryDummy = fk::Add<int>;
    using TernaryDummy = fk::InterpolateComplete<fk::InterpolationType::INTER_LINEAR, fk::Read<ReadDummy>>;
    using WriteDummy = fk::PerThreadWrite<fk::ND::_2D, int>;

    using DFList = fk::TypeList<fk::Read<ReadDummy>, fk::ReadBack<ReadBackDummy>,
                   fk::Read<ReadDummy>, fk::ReadBack<ReadBackDummy>,
                   fk::Binary<BinaryDummy>, fk::Ternary<TernaryDummy>,
                   fk::MidWrite<WriteDummy>, fk::Write<WriteDummy>>;

    constexpr bool correctDFRestrict = allInstantiableOperationsComplieWith<fk::NotUnaryRestriction>(DFList{});
    static_assert(correctDFRestrict, "The list of operations does not comply with the restriction");

    using ListToCheck =
        fk::TypeList<fk::ReadType, fk::BinaryType, fk::UnaryType, fk::TernaryType,
                     fk::UnaryType, fk::MidWriteType, fk::BinaryType, fk::WriteType>;

    using IndexList = fk::filtered_index_sequence_t<fk::NotUnaryRestriction, ListToCheck>;
    using IntegerList = fk::filtered_integer_sequence_t<int, fk::NotUnaryRestriction, ListToCheck>;

    constexpr IndexList indexList;

    static_assert(IndexList::size() == 6, "The index list does not have the expected size");
    constexpr size_t var = fk::get_index_f<0>(indexList);
    static_assert(var == 0, "Incorrect index");
    static_assert(fk::get_index<1, IndexList> == 1, "Incorrect index");
    static_assert(fk::get_index<2, IndexList> == 3, "Incorrect index");
    static_assert(fk::get_index<3, IndexList> == 5, "Incorrect index");
    static_assert(fk::get_index<4, IndexList> == 6, "Incorrect index");
    static_assert(fk::get_index<5, IndexList> == 7, "Incorrect index");

    static_assert(IntegerList::size() == 6, "The integer list does not have the expected size");
    static_assert(fk::get_integer<int, 0, IntegerList> == 0, "Incorrect integer");
    static_assert(fk::get_integer<int, 1, IntegerList> == 1, "Incorrect integer");
    static_assert(fk::get_integer<int, 2, IntegerList> == 3, "Incorrect integer");
    static_assert(fk::get_integer<int, 3, IntegerList> == 5, "Incorrect integer");
    static_assert(fk::get_integer<int, 4, IntegerList> == 6, "Incorrect integer");
    static_assert(fk::get_integer<int, 5, IntegerList> == 7, "Incorrect integer");

    using FirstUnary =
        fk::TypeList<fk::UnaryType, fk::TernaryType,
        fk::UnaryType, fk::MidWriteType, fk::BinaryType, fk::WriteType>;

    using IndexListFirstUnary = fk::filtered_index_sequence_t<fk::NotUnaryRestriction, FirstUnary>;

    static_assert(IndexListFirstUnary::size() == 4, "Incorrect sequence size");
    static_assert(fk::get_index<0, IndexListFirstUnary> == 1, "Incorrect index");
    static_assert(fk::get_index<1, IndexListFirstUnary> == 3, "Incorrect index");
    static_assert(fk::get_index<2, IndexListFirstUnary> == 4, "Incorrect index");
    static_assert(fk::get_index<3, IndexListFirstUnary> == 5, "Incorrect index");

    using FirstAndLastUnary =
        fk::TypeList<fk::UnaryType, fk::TernaryType,
        fk::UnaryType, fk::MidWriteType, fk::BinaryType, fk::UnaryType>;

    using IndexListFirstAndLastUnary = fk::filtered_index_sequence_t<fk::NotUnaryRestriction, FirstAndLastUnary>;

    static_assert(IndexListFirstAndLastUnary::size() == 3, "Incorrect sequence size");
    static_assert(fk::get_index<0, IndexListFirstAndLastUnary> == 1, "Incorrect index");
    static_assert(fk::get_index<1, IndexListFirstAndLastUnary> == 3, "Incorrect index");
    static_assert(fk::get_index<2, IndexListFirstAndLastUnary> == 4, "Incorrect index");

    using LastUnary = fk::TypeList<fk::TernaryType,
        fk::UnaryType, fk::MidWriteType, fk::BinaryType, fk::UnaryType>;

    using IndexListLastUnary = fk::filtered_index_sequence_t<fk::NotUnaryRestriction, LastUnary>;

    static_assert(IndexListLastUnary::size() == 3, "Incorrect sequence size");
    static_assert(fk::get_index<0, IndexListLastUnary> == 0, "Incorrect index");
    static_assert(fk::get_index<1, IndexListLastUnary> == 2, "Incorrect index");
    static_assert(fk::get_index<2, IndexListLastUnary> == 3, "Incorrect index");

    using AllTypesUnary = fk::TypeList<fk::UnaryType, fk::UnaryType, fk::UnaryType, fk::UnaryType>;

    using IndexListAllUnary = fk::filtered_index_sequence_t<fk::NotUnaryRestriction, AllTypesUnary>;

    static_assert(IndexListAllUnary::size() == 0, "Incorrect index");

    return 0;
}