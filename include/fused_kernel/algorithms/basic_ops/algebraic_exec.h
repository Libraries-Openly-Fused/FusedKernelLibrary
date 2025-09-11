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

#ifndef FK_ALGEBRAIC_EXEC_H
#define FK_ALGEBRAIC_EXEC_H

#include <fused_kernel/core/utils/cuda_vector_utils.h>
#include <fused_kernel/core/execution_model/operation_model/parent_operations_exec.h>
#include <fused_kernel/core/data/tuple.h>

namespace fk {
    struct M3x3Float {
        const float3 x;
        const float3 y;
        const float3 z;
    };

    template <typename OpInstanceType = BinaryType>
    struct MxVFloat3;

    template <>
    struct MxVFloat3<UnaryType> {
        TEMPLATE_UNARY_OPERATION_EXEC(MxVFloat3, (UnaryType),
                                 (Tuple<float3, M3x3Float>), (float3), (false))
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            const float3 xOut = get<0>(input) * get<1>(input).x;
            const float3 yOut = get<0>(input) * get<1>(input).y;
            const float3 zOut = get<0>(input) * get<1>(input).z;
            return { v_sum(xOut), v_sum(yOut), v_sum(zOut) };
        }
    };

    template <>
    struct MxVFloat3<BinaryType> {
        TEMPLATE_BINARY_OPERATION_EXEC(MxVFloat3, (BinaryType),
                                       (float3), (M3x3Float), (float3), (false))
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& params) {
            const float3 xOut = input * params.x;
            const float3 yOut = input * params.y;
            const float3 zOut = input * params.z;
            return { v_sum(xOut), v_sum(yOut), v_sum(zOut) };
        }
    };
} // namespace fk

#endif // FK_ALGEBRAIC_EXEC_H