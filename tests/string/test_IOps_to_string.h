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

#include "tests/main.h"

#include <fused_kernel/algorithms/algorithms.h>
#include <fused_kernel/core/core.h>
#include <fused_kernel/fused_kernel.h>

namespace fk {

int launch_impl() {
    Ptr2D<float> input(10, 10);
    auto readOp = PerThreadRead<ND::_2D, float>::build(input);

    std::cout << "Read Operation typeToString: " << typeToString<decltype(readOp)>() << std::endl;

    return 0;
}

} // namespace fk

int launch() {
    return fk::launch_impl();
}
