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
Read<FusedOperation<
    ReadBack<ResizeComplete<AspectRatio::PRESERVE_AR,
                    Ternary<InterpolateComplete<
                        InterpolationType::INTER_LINEAR, ReadBack<Crop<Read<PerThreadRead<ND::_2D, uchar3>>>>>>>>,
             Binary<Mul<float3, float3, float3>>>>;

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

    constexpr auto fusedOp = FusedOperation<>::build(ComplexType{}, Add<float3>::build(make_set<float3>(2.f)));

    constexpr auto opTuple = fusedOp.params;

    static_assert(opTuple.size == 3, "Wrong OperationTuple size");

    return true;
}

int launch() {
    constexpr ComplexType complexVar{};
    return test_InstantiableFusedOperationToOperationTuple() ? 0 : -1;
}
