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

#include "tests/main.h"

#include <fused_kernel/algorithms/basic_ops/algebraic_builders.h>

int launch() {
    using namespace fk;
    constexpr auto myOp =
        V3xM33().then(V3xM33(M3x3Float{1,1,1,1,1,1,1,1,1}));

    constexpr float3 res = decltype(myOp)::Operation::exec(make_tuple(float3{1,1,1}, M3x3Float{ 1,1,1,1,1,1,1,1,1 }), myOp.params);

    static_assert(res.x == 9.f && res.y == 9.f && res.z == 9.f, "Unexpected results");

    return 0;
}
