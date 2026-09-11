/* Copyright 2025 Grup Mediapro S.L.U (Oscar Amoros Huguet)
   Copyright 2026 Oscar Amoros Huguet

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

using ComplexType =
fk::Read<fk::FusedOperation<
    fk::ReadBack<fk::ResizeComplete<fk::AspectRatio::PRESERVE_AR,
                    fk::Ternary<fk::InterpolateComplete<
                        fk::InterpolationType::INTER_LINEAR, fk::ReadBack<fk::Crop<fk::Read<fk::PerThreadRead<fk::ND::_2D, fk::uchar3>>>>>>>>,
             fk::Binary<fk::Mul<fk::float3, fk::float3, fk::float3>>>>;

// Operation types
// Read
using RPerThrFloat = fk::PerThreadRead<fk::ND::_2D, float>;
// ReadBack
using RBResize = fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR, fk::Instantiable<RPerThrFloat>>;
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

constexpr bool test_InstantiableFusedOperationToOperationTuple() {
    using namespace fk;
    constexpr auto fusedOp = FusedOperation<>::build(ComplexType{}, Add<fk::float3>::build(make_set<fk::float3>(2.f)));

    constexpr auto opTuple = fusedOp.params;

    static_assert(opTuple.size == 3, "Wrong OperationTuple size");

    return true;
}

int launch() {
    return test_InstantiableFusedOperationToOperationTuple() ? 0 : -1;
}
