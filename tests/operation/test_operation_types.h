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

#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/cast.h>
#include <fused_kernel/algorithms/image_processing/resize.h>
#include <fused_kernel/algorithms/image_processing/crop.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>

namespace fk {

// Operation types
// Read
using RPerThrFloat = PerThreadRead<ND::_2D, float>;
// ReadBack
using RBResize = ResizeComplete<AspectRatio::IGNORE_AR, Instantiable<InterpolateComplete<InterpolationType::INTER_LINEAR, Instantiable<RPerThrFloat>>>>;
// Unary
using UIntFloat = Cast<int, float>;
using UFloatInt = Cast<float, int>;
using Unaries = TypeList<UIntFloat, UFloatInt>;
// Binary
using BAddInt = Add<int>;
using BAddFloat = Add<float>;
using Binaries = TypeList<BAddInt, BAddFloat>;
// Ternary
using TInterpFloat = InterpolateComplete<InterpolationType::INTER_LINEAR, Instantiable<RPerThrFloat>>;
// Write
using WPerThrFloat = PerThreadWrite<ND::_2D, float>;
// MidWrite
using FusedPerThrFloat = FusedOperation<MidWrite<WPerThrFloat>, Binary<BAddFloat>>;

// Test combination type lists
template <typename... Types>
using TL = TypeList<Types...>;

template <typename TL1, typename TL2>
using TLC = TypeListCat_t<TL1, TL2>;

template <typename TL, typename T>
using ITB = InsertTypeBack_t<TL, T>;

template <typename T, typename TL>
using ITF = InsertTypeFront_t<T, TL>;

// No Read
using NoRead = ITB<ITB<ITB<TLC<TLC<TL<RBResize>, Unaries>, Binaries>, TInterpFloat>, WPerThrFloat>, FusedPerThrFloat>;
// No ReadBack
using NoReadBack = ITB<ITB<ITB<TLC<TLC<TL<RPerThrFloat>, Unaries>, Binaries>, TInterpFloat>, WPerThrFloat>, FusedPerThrFloat>;
// No Unary
using NoUnary = ITB<ITB<ITB<TLC<TLC<TL<RPerThrFloat>, TL<RBResize>>, Binaries>, TInterpFloat>, WPerThrFloat>, FusedPerThrFloat>;
// No Binary
using NoBinary = ITB<ITB<ITB<TLC<TLC<TL<RPerThrFloat>, TL<RBResize>>, Unaries>, TInterpFloat>, WPerThrFloat>, FusedPerThrFloat>;
// No Ternary
using NoTernary = ITB<ITB<TLC<TLC<TLC<TL<RPerThrFloat>, TL<RBResize>>, Unaries>, Binaries>, WPerThrFloat>, FusedPerThrFloat>;
// No Write
using NoWrite = ITB<ITB<TLC<TLC<TLC<TL<RPerThrFloat>, TL<RBResize>>, Unaries>, Binaries>, TInterpFloat>, FusedPerThrFloat>;
// No Midwrite
using NoFused = ITB<ITB<TLC<TLC<TLC<TL<RPerThrFloat>, TL<RBResize>>, Unaries>, Binaries>, TInterpFloat>, WPerThrFloat>;
// No AnyWrite
using NoAnyWrite = ITB<ITB<TLC<TLC<ITB<TL<RPerThrFloat>, RBResize>, Unaries>, Binaries>, TInterpFloat>, FusedPerThrFloat>;
// All Compute
using AllCompute = ITB<TLC<Unaries, Binaries>, TInterpFloat>;

template <typename TypeList>
struct ContainsReadType;
template <typename... Types>
struct ContainsReadType<TypeList<Types...>> {
    static constexpr bool value = or_v<opIs<ReadType, Types>...>;
};

template <typename TypeList>
struct ContainsReadBackType;
template <typename... Types>
struct ContainsReadBackType<TypeList<Types...>> {
    static constexpr bool value = or_v<opIs<ReadBackType, Types>...>;
};

template <typename TypeList>
struct NoneAnyWriteType;
template <typename... Types>
struct NoneAnyWriteType<TypeList<Types...>> {
    static constexpr bool value = noneAnyWriteType<Types...>;
};

template <typename TypeList>
struct NoneFusedType;

template <typename... Types>
struct NoneFusedType<TypeList<Types...>> {
    static constexpr bool value = !or_v<opIs<OpenType, Types>...>;
};

template <typename TypeList_t>
struct Test_allUnaryTypes;

template <typename... OpsOrIOps>
struct Test_allUnaryTypes<TypeList<OpsOrIOps...>> {
    static constexpr bool value = allUnaryTypes<OpsOrIOps...>;
};

constexpr bool test_allUnaryTypes() {
    constexpr bool mustTrue = Test_allUnaryTypes<Unaries>::value;
    constexpr bool mustFalse1 = Test_allUnaryTypes<NoUnary>::value;
    constexpr bool mustFalse2 = Test_allUnaryTypes<NoTernary>::value;
    constexpr bool mustFalse3 = Test_allUnaryTypes<NoWrite>::value;
    constexpr bool mustFalse4 = Test_allUnaryTypes<NoAnyWrite>::value;
    constexpr bool mustFalse5 = Test_allUnaryTypes<NoBinary>::value;
    using ComplexType =
    Read<FusedOperation<typename ResizeComplete<AspectRatio::PRESERVE_AR,
                                Ternary<InterpolateComplete<InterpolationType::INTER_LINEAR, ReadBack<Crop<Read<PerThreadRead<ND::_2D, uchar3>>>>>>>::InstantiableType,
                                typename Mul<float3, float3, float3>::InstantiableType>>;
    constexpr bool mustFalse6 = allUnaryTypes<ComplexType>;

    using ComplexType2 =
        Read<FusedOperation<typename ResizeComplete<AspectRatio::PRESERVE_AR,
                                    Ternary<InterpolateComplete<InterpolationType::INTER_LINEAR, ReadBack<Crop<Read<PerThreadRead<ND::_2D, uchar3>>>>>>>::InstantiableType,
                                    typename Mul<float3, float3, float3>::InstantiableType>>;
    constexpr bool mustFalse7 = Test_allUnaryTypes<TypeList<ComplexType2>>::value;

    return mustTrue && !or_v<mustFalse1, mustFalse2, mustFalse3, mustFalse4, mustFalse5, mustFalse6, mustFalse7>;
}

int launch_impl() {
    // isReadType
    constexpr bool noneRead = !ContainsReadType<NoRead>::value;
    constexpr bool isRead = opIs<ReadType, RPerThrFloat>;
    static_assert(noneRead && isRead, "Something wrong with isReadType");

    // isReadBackType
    constexpr bool noneReadBack = !ContainsReadBackType<NoReadBack>::value;
    constexpr bool isReadBack = opIs<ReadBackType, RBResize>;
    static_assert(noneReadBack && isReadBack, "Something wrong with isReadType");

    // noneAnyWriteType
    constexpr bool noneAnyWriteType_ = NoneAnyWriteType<NoAnyWrite>::value;
    static_assert(noneAnyWriteType_, "Something wrong with isReadType");

    // allUnaryTypes
    constexpr bool allUnaryTypes_v = test_allUnaryTypes();
    static_assert(allUnaryTypes_v, "Something wrong with allUnaryTypes");
    
    return 0;
}

} // namespace fk

int launch() {
    return fk::launch_impl();
}
