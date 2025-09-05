/* Copyright 2025 Grup Mediapro S.L.U (Oscar Amoros Hguet)

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

#include <fused_kernel/core/core.h>
#include <fused_kernel/algorithms/algorithms.h>

using namespace fk;

using ComplexType =
Read<FusedOperation_<void,
    Resize<InterpolationType::INTER_LINEAR, AspectRatio::PRESERVE_AR,
    ReadBack<Crop<Read<PerThreadRead<ND::_2D, uchar3>>>>>,
    Mul<float3, float3, float3>>>;

// Operation types
// Read
using RPerThrFloat = PerThreadRead<ND::_2D, float>;
// ReadBack
using RBResize = Resize<InterpolationType::INTER_LINEAR, AspectRatio::IGNORE_AR, Instantiable<RPerThrFloat>>;
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
using MWPerThrFloat = FusedOperation<WPerThrFloat, BAddFloat>;


constexpr bool test_InstantiableFusedOperationToOperationTuple() {
    constexpr bool mustFalse = isAllUnaryFusedOperation<MWPerThrFloat>;
    static_assert(!mustFalse, "MWPerThrFloat is not an all Unary FusedOperation");
    constexpr bool mustTrue = isAllUnaryFusedOperation<FusedOperation<UIntFloat>>;
    static_assert(mustTrue, "FusedOperation<UIntFloat> is an all Unary FusedOperation");
    constexpr bool mustTrue2 = isAllUnaryFusedOperation<FusedOperation<UIntFloat, UFloatInt>>;
    static_assert(mustTrue2, "FusedOperation<UIntFloat, UFloatInt> is an all Unary FusedOperation");

    constexpr bool mustTrue3 = isNotAllUnaryFusedOperation<MWPerThrFloat>;
    static_assert(mustTrue3, "MWPerThrFloat is not an all Unary FusedOperation");
    constexpr bool mustFalse2 = isNotAllUnaryFusedOperation<FusedOperation<UIntFloat>>;
    static_assert(!mustFalse2, "FusedOperation<UIntFloat> is an all Unary FusedOperation");
    constexpr bool mustFalse3 = isNotAllUnaryFusedOperation<FusedOperation<UIntFloat, UFloatInt>>;
    static_assert(!mustFalse3, "FusedOperation<UIntFloat, UFloatInt> is an all Unary FusedOperation");
    constexpr auto fusedOp0 = Binary<FusedOperation<Add<float>>>{};
    [[maybe_unused]] constexpr auto operationTuple = InstantiableFusedOperationToOperationTuple<Binary<FusedOperation<Add<float>>>>::value(fusedOp0);
    constexpr auto fusedOp = MWPerThrFloat::build(OperationData<MWPerThrFloat>{});
    using FusedOpType = std::decay_t<decltype(fusedOp)>;
    [[maybe_unused]] constexpr auto operationTuple2 = InstantiableFusedOperationToOperationTuple<FusedOpType>::value(fusedOp);

    return true;
}

int launch() {
#if defined(_MSC_VER) && (_MSC_VER >= 1910) && (_MSC_VER < 1920)
    return test_InstantiableFusedOperationToOperationTuple() ? 0 : -1;
#else
    constexpr ComplexType complexVar{};
    [[maybe_unused]] constexpr auto opTuple = InstantiableFusedOperationToOperationTuple<ComplexType>::value(complexVar);
    return test_InstantiableFusedOperationToOperationTuple() ? 0 : -1;
#endif
}
