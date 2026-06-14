/* Copyright 2025-2026 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

// Regression test for: Executor<DivergentBatchTransformDPP<GPU_NVIDIA, Selector>>::executeOperations
// does not compile when IOpSequence objects are const lvalues built with buildOperationSequence.
// Root cause: fuseBackSequence passed explicit template args to BackFuser::fuse_back<IOps...>,
// fixing IOps&& as rvalue references that cannot bind to const lvalues.
// Fix: let type deduction happen via a lambda instead of explicit template arguments.

#define __ONLY_CU__
#include <tests/main.h>

#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/fused_kernel.h>

using namespace fk;

struct MySelector {
    FK_HOST_DEVICE_FUSE uint at(const uint& index) { return index == 0 ? 0u : 1u; }
};

int launch() {
    Stream stream;

    constexpr int WIDTH = 8;
    constexpr int HEIGHT = 4;

    Ptr2D<float> imgA(WIDTH, HEIGHT);
    Ptr2D<float> imgB(WIDTH, HEIGHT);
    Tensor<float> output(WIDTH, HEIGHT, 2);

    // Build the const lvalue sequences — this is the value category that triggered the bug
    const auto seq1 = buildOperationSequence(PerThreadRead<ND::_2D, float>::build(imgA),
                                             Mul<float>::build(2.0f),
                                             TensorWrite<float>::build(output));
    const auto seq2 = buildOperationSequence(PerThreadRead<ND::_2D, float>::build(imgB),
                                             Add<float>::build(100.0f),
                                             TensorWrite<float>::build(output));

    // Prior to the fix this failed to compile with:
    //   "qualifiers dropped in binding reference of type ... &&
    //    to initializer of type const ..."
    Executor<DivergentBatchTransformDPP<ParArch::GPU_NVIDIA, MySelector>>::executeOperations(stream, seq1, seq2);
    stream.sync();

    return 0;
}
