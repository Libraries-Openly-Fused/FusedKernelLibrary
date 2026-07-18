/* Copyright 2026 Oscar Amoros Huguet, Johnny Nunez

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

// Tests RowReduceDPP: a DPP defined purely through the InstantiableDPP protocol
// (IO_SPEC + build_details + exec + getLaunchConfig), with both CPU and GPU
// implementations, executed through the generic fk::execute path. It proves that a
// new conforming DPP needs no Executor specialization and no dedicated __global__
// kernel. This test compiles and runs on both backends.

#include <tests/main.h>

#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/reductions/row_reduce.h>
#include <fused_kernel/fused_kernel.h>
#include <iostream>
#include <stdexcept>

using namespace fk;

// ============================================================================
// Compile-time IO contract checks (DPPIOSpec)
// ============================================================================
namespace row_reduce_contract_checks {
    constexpr RawPtr<ND::_2D, int> dummyInput{ nullptr, { 32, 8, 0 } };
    constexpr RawPtr<ND::_1D, int> dummyOutput{ nullptr, { 8, 0 } };
    constexpr auto readIOp = PerThreadRead<ND::_2D, int>::build({ dummyInput });
    constexpr auto mulIOp = Mul<int>::build(2);
    constexpr auto writeIOp = PerThreadWrite<ND::_1D, int>::build({ dummyOutput });

    using ReadT = std::decay_t<decltype(readIOp)>;
    using MulT = std::decay_t<decltype(mulIOp)>;
    using WriteT = std::decay_t<decltype(writeIOp)>;
    using RowSum = RowReduceDPP<Add<int>>;

    static_assert(hasDPPIOSpec_v<RowSum>, "RowReduceDPP must declare its IO contract (IO_SPEC)");
    static_assert(dppIOContractSatisfied<RowSum, ReadT, WriteT>(), "read + write must conform");
    // RowReduceDPP does NOT accept a compute IOp chain: per-element preprocessing has
    // to be fused into the input IOp (readIOp.then(...)) instead.
    static_assert(!dppIOContractSatisfied<RowSum, ReadT, MulT, WriteT>(),
                  "read + compute + write must NOT conform");
    static_assert(!dppIOContractSatisfied<RowSum, ReadT>(), "missing Write must not conform");
    static_assert(!dppIOContractSatisfied<RowSum, MulT, WriteT>(), "missing Read must not conform");

    // Compile-fail documentation (the repo has no compile-fail test infra). If
    // uncommented, this line must fail to compile with the quoted message:
    //
    // constexpr auto bad = RowSum::build(readIOp, mulIOp, writeIOp);
    //   error: static assertion failed: DPP IO contract violation: this DPP does not accept compute
    //   IOps between its input (Read/ReadBack) and output (Write) IOps. Fuse them into the input
    //   IOp instead (readIOp.then(computeIOp...)).
} // namespace row_reduce_contract_checks

template <int WIDTH, int HEIGHT>
bool testRowSum(Stream& stream) {
    Ptr2D<int> input(WIDTH, HEIGHT);
    Ptr1D<int> output(HEIGHT);

    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            input.at(x, y) = (x + y) % 7 + 1;
        }
    }
    input.upload(stream);

    const auto iDPP = RowReduceDPP<Add<int>>::build(PerThreadRead<ND::_2D, int>::build(input),
                                                    PerThreadWrite<ND::_1D, int>::build(output));
    static_assert(isInstantiableDPP_v<decltype(iDPP)>, "build() must return an InstantiableDPP");
    execute(stream, iDPP);

    output.download(stream);
    stream.sync();

    bool correct{ true };
    for (int y = 0; y < HEIGHT; ++y) {
        int expected = 0;
        for (int x = 0; x < WIDTH; ++x) {
            expected += (x + y) % 7 + 1;
        }
        if (output.at(Point{ y }) != expected) {
            std::cout << "testRowSum<" << WIDTH << ", " << HEIGHT << "> mismatch at row " << y
                      << ": got " << output.at(Point{ y }) << ", expected " << expected << std::endl;
            correct = false;
        }
    }
    return correct;
}

bool testRowReduceRejects3DInput() {
    // RowReduceDPP reduces 2D inputs only: both backends fix the Point z coordinate
    // to 0, so a batched/3D input would silently reduce only plane z == 0. build()
    // must fail loudly instead of accepting it.
    constexpr int WIDTH = 16;
    constexpr int HEIGHT = 8;
    constexpr int PLANES = 2;

    Tensor<int> input(WIDTH, HEIGHT, PLANES);
    Ptr1D<int> output(HEIGHT);

    try {
        const auto iDPP = RowReduceDPP<Add<int>>::build(TensorRead<int>::build(input),
                                                        PerThreadWrite<ND::_1D, int>::build(output));
        (void)iDPP;
    } catch (const std::invalid_argument&) {
        return true;
    }
    std::cout << "testRowReduceRejects3DInput: build() did not reject a 3D (batched) input" << std::endl;
    return false;
}

bool testRowSumFusedPrologue(Stream& stream) {
    // Per-element preprocessing is fused into the input Read IOp: the compute chain
    // runs in-register at load time, on every element, before the reduction.
    constexpr int WIDTH = 129;
    constexpr int HEIGHT = 9;

    Ptr2D<int> input(WIDTH, HEIGHT);
    Ptr1D<int> output(HEIGHT);

    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            input.at(x, y) = x % 5;
        }
    }
    input.upload(stream);

    const auto fusedReadIOp = PerThreadRead<ND::_2D, int>::build(input).then(Mul<int>::build(3));
    const auto iDPP = RowReduceDPP<Add<int>>::build(fusedReadIOp, PerThreadWrite<ND::_1D, int>::build(output));
    execute(stream, iDPP);

    output.download(stream);
    stream.sync();

    bool correct{ true };
    for (int y = 0; y < HEIGHT; ++y) {
        int expected = 0;
        for (int x = 0; x < WIDTH; ++x) {
            expected += (x % 5) * 3;
        }
        if (output.at(Point{ y }) != expected) {
            std::cout << "testRowSumFusedPrologue mismatch at row " << y << ": got "
                      << output.at(Point{ y }) << ", expected " << expected << std::endl;
            correct = false;
        }
    }
    return correct;
}

int launch() {
    Stream stream;

    // Integer ReduceOp on purpose: the reduction order is backend-dependent (sequential
    // left fold on CPU, strided per-lane accumulation + tree reduction on GPU), so the
    // ReduceOp must be associative and commutative, and floating-point results may
    // differ between backends by rounding (reassociation).
    bool correct{ true };
    correct &= testRowSum<7, 5>(stream);     // narrower than one block
    correct &= testRowSum<256, 13>(stream);  // exactly one block wide (default BLOCK_SIZE)
    correct &= testRowSum<300, 17>(stream);  // wider than one block: strided accumulation
    correct &= testRowReduceRejects3DInput();
    correct &= testRowSumFusedPrologue(stream);

    return correct ? 0 : -1;
}
