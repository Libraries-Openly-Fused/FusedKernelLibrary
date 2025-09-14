/* Copyright 2025 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_ARITHMETIC_EXEC_H
#define FK_ARITHMETIC_EXEC_H

#include <fused_kernel/core/utils/cuda_vector_utils.h>
#include <fused_kernel/core/execution_model/operation_model/parent_operations_exec.h>
#include <fused_kernel/core/data/tuple.h>

namespace fk {

    template <typename I1, typename I2 = I1, typename O = I1, typename IT = BinaryType>
    struct AddExec;

    template <typename I, typename P, typename O>
    struct AddExec<I, P, O, BinaryType> {
        TEMPLATE_BINARY_OPERATION_EXEC(AddExec, (I, P, O, BinaryType), (I), (P), (O), (false))
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& params) {
            return input + params;
        }
    };

    template <typename I1, typename I2, typename O>
    struct AddExec<I1, I2, O, UnaryType> {
        TEMPLATE_UNARY_OPERATION_EXEC(AddExec, (I1, I2, O, UnaryType), (Tuple<I1, I2>), (O), (false))
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            return get<0>(input) + get<1>(input);
        }
    };

} // namespace fk

#endif // FK_ARITHMETIC_EXEC_H
