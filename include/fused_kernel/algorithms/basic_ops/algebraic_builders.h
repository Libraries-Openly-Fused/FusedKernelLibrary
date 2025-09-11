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

#ifndef FK_ALGEBRAIC_BUILDERS_H
#define FK_ALGEBRAIC_BUILDERS_H

#include <fused_kernel/algorithms/basic_ops/algebraic_exec.h>
#include <fused_kernel/core/execution_model/operation_model/parent_operation_builders.h>

namespace fk {

    struct MxVFloat3Builder {
        FK_HOST_FUSE auto build() {
            return Unary<MxVFloat3<UnaryType>>{};
        }

        FK_HOST_FUSE auto build(const M3x3Float& params) {
            return Binary<MxVFloat3<BinaryType>>{{params}};
        }

        constexpr auto operator()() const {
            return build();
        }

        constexpr auto operator()(const M3x3Float& params) const {
            return build(params);
        }
    };

    constexpr MxVFloat3Builder V3xM33{};

} // namespace fk

#endif //  FK_ALGEBRAIC_BUILDERS_H
