/* Copyright 2023-2025 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_ALGEBRAIC
#define FK_ALGEBRAIC

#include <fused_kernel/core/execution_model/operation_model/operation_model.h>
#include <fused_kernel/core/utils/vector_utils.h>

namespace fk {

    struct M3x3Float {
        const float3 x;
        const float3 y;
        const float3 z;
    };

    template <typename OpInstanceType = BinaryType>
    struct MxVFloat3;

    template <>
    struct MxVFloat3<BinaryType> {
    private:
        using SelfType = MxVFloat3<BinaryType>;
    public:
        FK_STATIC_STRUCT(MxVFloat3, SelfType)
        using Parent = BinaryOperation<float3, M3x3Float, float3, MxVFloat3<BinaryType>>;
        DECLARE_BINARY_PARENT

        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& params) {
            const float3 xOut = input * params.x;
            const float3 yOut = input * params.y;
            const float3 zOut = input * params.z;
            return { cxp::sum::f(xOut), cxp::sum::f(yOut), cxp::sum::f(zOut) };
        }
    };

    template <>
    struct MxVFloat3<UnaryType> {
    private:
        using SelfType = MxVFloat3<UnaryType>;
    public:
        FK_STATIC_STRUCT(MxVFloat3, SelfType)
        using Parent = UnaryOperation<Tuple<float3, M3x3Float>, float3, MxVFloat3<UnaryType>>;
        DECLARE_UNARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            const float3 xOut = get<0>(input) * get<1>(input).x;
            const float3 yOut = get<0>(input) * get<1>(input).y;
            const float3 zOut = get<0>(input) * get<1>(input).z;
            return { cxp::sum::f(xOut), cxp::sum::f(yOut), cxp::sum::f(zOut) };
        }
    };
} //namespace fk

#endif
