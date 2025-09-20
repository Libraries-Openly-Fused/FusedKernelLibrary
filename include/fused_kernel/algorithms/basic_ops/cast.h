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

#ifndef FK_CAST
#define FK_CAST

#include <fused_kernel/core/execution_model/operation_model/operation_model.h>
#include <fused_kernel/core/constexpr_libs/constexpr_vector.h>

namespace fk {
    template <typename I, typename O>
    struct Cast {
    private:
        using SelfType = Cast<I, O>;
    public:
        FK_STATIC_STRUCT(Cast, SelfType)
        using Parent = UnaryOperation<I, O, Cast<I, O>>;
        DECLARE_UNARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            return cxp::cast<OutputType>::f(input);
        }
    };
} // namespace fk

#endif
