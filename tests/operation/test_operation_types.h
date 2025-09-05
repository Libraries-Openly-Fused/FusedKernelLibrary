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

#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/cast.h>
#include <fused_kernel/algorithms/image_processing/resize.h>
#include <fused_kernel/algorithms/image_processing/crop.h>
#include <fused_kernel/core/execution_model/memory_operations.h>

// Operation types
// Read
using RPerThrFloat = fk::PerThreadRead<fk::ND::_2D, float>;
// ReadBack
using RBResize = fk::ResizeComplete<fk::AspectRatio::IGNORE_AR, fk::Instantiable<fk::InterpolateComplete<fk::InterpolationType::INTER_LINEAR, fk::Instantiable<RPerThrFloat>>>>;
// Unary
using UIntFloat = fk::Cast<int, float>;
using UFloatInt = fk::Cast<float, int>;
using Unaries = fk::TypeList<UIntFloat, UFloatInt>;
// Binary
using BAddInt = fk::Add<int>;
using BAddFloat = fk::Add<float>;
using Binaries = fk::TypeList<BAddInt, BAddFloat>;
// Ternary
using TInterpFloat = fk::InterpolateComplete<fk::InterpolationType::INTER_LINEAR, fk::Instantiable<RPerThrFloat>>;
// Write
using WPerThrFloat = fk::PerThreadWrite<fk::ND::_2D, float>;
// MidWrite
using MWPerThrFloat = fk::FusedOperation<WPerThrFloat, BAddFloat>;

// Test combination type lists
template <typename... Types>
using TL = fk::TypeList<Types...>;

template <typename TL1, typename TL2>
using TLC = fk::TypeListCat_t<TL1, TL2>;

template <typename TL, typename T>
using ITB = fk::InsertTypeBack_t<TL, T>;

template <typename T, typename TL>
using ITF = fk::InsertTypeFront_t<T, TL>;

// No Read
using NoRead = ITB<ITB<ITB<TLC<TLC<TL<RBResize>, Unaries>, Binaries>, TInterpFloat>, WPerThrFloat>, MWPerThrFloat>;
// No ReadBack
using NoReadBack = ITB<ITB<ITB<TLC<TLC<TL<RPerThrFloat>, Unaries>, Binaries>, TInterpFloat>, WPerThrFloat>, MWPerThrFloat>;
// No Unary
using NoUnary = ITB<ITB<ITB<TLC<TLC<TL<RPerThrFloat>, TL<RBResize>>, Binaries>, TInterpFloat>, WPerThrFloat>, MWPerThrFloat>;
// No Binary
using NoBinary = ITB<ITB<ITB<TLC<TLC<TL<RPerThrFloat>, TL<RBResize>>, Unaries>, TInterpFloat>, WPerThrFloat>, MWPerThrFloat>;
// No Ternary
using NoTernary = ITB<ITB<TLC<TLC<TLC<TL<RPerThrFloat>, TL<RBResize>>, Unaries>, Binaries>, WPerThrFloat>, MWPerThrFloat>;
// No Write
using NoWrite = ITB<ITB<TLC<TLC<TLC<TL<RPerThrFloat>, TL<RBResize>>, Unaries>, Binaries>, TInterpFloat>, MWPerThrFloat>;
// No Midwrite
using NoMidWrite = ITB<ITB<TLC<TLC<TLC<TL<RPerThrFloat>, TL<RBResize>>, Unaries>, Binaries>, TInterpFloat>, WPerThrFloat>;
// No AnyWrite
using NoAnyWrite = ITB<TLC<TLC<ITB<TL<RPerThrFloat>, RBResize>, Unaries>, Binaries>, TInterpFloat>;
// All Compute
using AllCompute = ITB<TLC<Unaries, Binaries>, TInterpFloat>;

template <typename TypeList>
struct IsReadType;
template <typename... Types>
struct IsReadType<fk::TypeList<Types...>> {
    static constexpr bool value = fk::or_v<fk::isReadType<Types>...>;
};

template <typename TypeList>
struct IsReadBackType;
template <typename... Types>
struct IsReadBackType<fk::TypeList<Types...>> {
    static constexpr bool value = fk::or_v<fk::isReadBackType<Types>...>;
};

template <typename TypeList>
struct NoneAnyWriteType;
template <typename... Types>
struct NoneAnyWriteType<fk::TypeList<Types...>> {
    static constexpr bool value = fk::noneAnyWriteType<Types...>;
};

template <typename TypeList_t>
struct Test_allUnaryTypes;

template <typename... OpsOrIOps>
struct Test_allUnaryTypes<fk::TypeList<OpsOrIOps...>> {
    static constexpr bool value = fk::allUnaryTypes<OpsOrIOps...>;
};

constexpr bool test_allUnaryTypes() {
    constexpr bool mustTrue = Test_allUnaryTypes<Unaries>::value;
    constexpr bool mustFalse1 = Test_allUnaryTypes<NoUnary>::value;
    constexpr bool mustFalse2 = Test_allUnaryTypes<NoTernary>::value;
    constexpr bool mustFalse3 = Test_allUnaryTypes<NoWrite>::value;
    constexpr bool mustFalse4 = Test_allUnaryTypes<NoAnyWrite>::value;
    constexpr bool mustFalse5 = Test_allUnaryTypes<NoBinary>::value;
    using ComplexType =
    fk::Read<fk::FusedOperation_<void,
                                 fk::ResizeComplete<fk::AspectRatio::PRESERVE_AR,
                                            fk::Ternary<fk::InterpolateComplete<fk::InterpolationType::INTER_LINEAR, fk::ReadBack<fk::Crop<fk::Read<fk::PerThreadRead<fk::ND::_2D, uchar3>>>>>>>,
                                 fk::Mul<float3, float3, float3>>>;
    constexpr bool mustFalse6 = fk::allUnaryTypes<ComplexType>;

    using ComplexType2 = fk::Read<fk::FusedOperation_<void,
                                                      fk::ResizeComplete<fk::AspectRatio::PRESERVE_AR,
        fk::Ternary<fk::InterpolateComplete<fk::InterpolationType::INTER_LINEAR, fk::ReadBack<fk::Crop<fk::Read<fk::PerThreadRead<fk::ND::_2D, uchar3>>>>>>>,
                                                          fk::Mul<float3, float3, float3>>>;
    constexpr bool mustFalse7 = Test_allUnaryTypes<fk::TypeList<ComplexType2>>::value;

    return mustTrue && !fk::or_v<mustFalse1, mustFalse2, mustFalse3, mustFalse4, mustFalse5, mustFalse6, mustFalse7>;
}

template <typename TypeList_t>
struct Test_notAllUnaryTypes;

template <typename... OpsOrIOps>
struct Test_notAllUnaryTypes<fk::TypeList<OpsOrIOps...>> {
    static constexpr bool value = fk::notAllUnaryTypes<OpsOrIOps...>;
};

constexpr bool test_notAllUnaryTypes() {
    constexpr bool mustFalse = Test_notAllUnaryTypes<Unaries>::value;
    constexpr bool mustTrue1 = Test_notAllUnaryTypes<NoUnary>::value;
    constexpr bool mustTrue2 = Test_notAllUnaryTypes<NoTernary>::value;
    constexpr bool mustTrue3 = Test_notAllUnaryTypes<NoWrite>::value;
    constexpr bool mustTrue4 = Test_notAllUnaryTypes<NoAnyWrite>::value;
    constexpr bool mustTrue5 = Test_notAllUnaryTypes<NoBinary>::value;

    using ComplexType =
        fk::Read<fk::FusedOperation_<void,
        fk::ResizeComplete<fk::AspectRatio::PRESERVE_AR,
        fk::Ternary<fk::InterpolateComplete<fk::InterpolationType::INTER_LINEAR, fk::ReadBack<fk::Crop<fk::Read<fk::PerThreadRead<fk::ND::_2D, uchar3>>>>>>>,
        fk::Mul<float3, float3, float3>>>;
    constexpr bool mustTrue6 = fk::notAllUnaryTypes<ComplexType>;

    using ComplexType2 = fk::Read<fk::FusedOperation_<void,
        fk::ResizeComplete<fk::AspectRatio::PRESERVE_AR, fk::Ternary<fk::InterpolateComplete<fk::InterpolationType::INTER_LINEAR, fk::ReadBack<fk::Crop<fk::Read<fk::PerThreadRead<fk::ND::_2D, uchar3>>>>>>>,
        fk::Mul<float3, float3, float3>>>;
    constexpr bool mustTrue7 = Test_notAllUnaryTypes<fk::TypeList<ComplexType2>>::value;

    return !mustFalse && fk::and_v<mustTrue1, mustTrue2, mustTrue3, mustTrue4, mustTrue5, mustTrue6, mustTrue7>;
}

int launch() {
    // isReadType
    constexpr bool noneRead = !IsReadType<NoRead>::value;
    constexpr bool isRead = fk::isReadType<RPerThrFloat>;
    static_assert(noneRead && isRead, "Something wrong with isReadType");

    // isReadBackType
    constexpr bool noneReadBack = !IsReadBackType<NoReadBack>::value;
    constexpr bool isReadBack = fk::isReadBackType<RBResize>;
    static_assert(noneReadBack && isReadBack, "Something wrong with isReadType");

    // noneAnyWriteType
    constexpr bool noneAnyWriteType_v = NoneAnyWriteType<NoAnyWrite>::value;
    constexpr bool oneIsMidWrite = !NoneAnyWriteType<NoWrite>::value;
    constexpr bool oneIsWrite = !NoneAnyWriteType<NoMidWrite>::value;
    static_assert(fk::and_v<noneAnyWriteType_v, oneIsMidWrite, oneIsWrite>, "Something wrong with isReadType");

    // allUnaryTypes
    constexpr bool allUnaryTypes_v = test_allUnaryTypes();
    static_assert(allUnaryTypes_v, "Something wrong with allUnaryTypes");

    // notAllUnaryTypes
    constexpr bool notAllUnaryTypes_v = test_notAllUnaryTypes();
    static_assert(notAllUnaryTypes_v, "Something wrong with notAllUnaryTypes");

    return 0;
}
