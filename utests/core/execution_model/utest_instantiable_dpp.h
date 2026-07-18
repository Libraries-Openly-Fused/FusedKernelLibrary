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

// Tests for the InstantiableDPP model: DPP::build(iOps...) -> InstantiableDPP value,
// the compile-time DPP IO contract (DPPIOSpec / dppIOContractSatisfied) and the
// generic execution entry point fk::execute(stream, instantiableDPP), which must
// produce the same results as the legacy Executor path on both backends.

#include <tests/main.h>

#include <fused_kernel/algorithms/algorithms.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/fused_kernel.h>
#include <iostream>

using namespace fk;

// ============================================================================
// Compile-time IO contract checks (DPPIOSpec)
// ============================================================================
namespace instantiable_dpp_contract_checks {
    constexpr RawPtr<ND::_2D, float> dummyInput{ nullptr, { 64, 64, 0 } };
    constexpr RawPtr<ND::_2D, float> dummyOutput{ nullptr, { 64, 64, 0 } };
    constexpr auto readIOp = PerThreadRead<ND::_2D, float>::build({ dummyInput });
    constexpr auto mulIOp = Mul<float>::build(2.f);
    constexpr auto writeIOp = PerThreadWrite<ND::_2D, float>::build({ dummyOutput });

    using ReadT = std::decay_t<decltype(readIOp)>;
    using MulT = std::decay_t<decltype(mulIOp)>;
    using WriteT = std::decay_t<decltype(writeIOp)>;
    using TDPP = TransformDPP<>;

    // TransformDPP declares its IO API: 1 Read/ReadBack in, optional compute chain, 1 Write out
    static_assert(hasDPPIOSpec_v<TDPP>, "TransformDPP must declare its IO contract (IO_SPEC)");
    static_assert(TDPP::IO_SPEC.inputIOps == 1 && TDPP::IO_SPEC.outputIOps == 1 &&
                  TDPP::IO_SPEC.acceptsComputeIOps && !TDPP::IO_SPEC.argsAreIOpSequences,
                  "Unexpected TransformDPP IO_SPEC");

    // Conforming packs
    static_assert(dppIOContractSatisfied<TDPP, ReadT, WriteT>(), "read + write must conform");
    static_assert(dppIOContractSatisfied<TDPP, ReadT, MulT, WriteT>(), "read + compute + write must conform");
    static_assert(dppIOContractSatisfied<TDPP, ReadT, MulT, MulT, WriteT>(), "compute chains must conform");

    // Non-conforming packs (these are the same checks that make DPP::build() fail
    // to compile, reported through static_asserts with explicit messages)
    static_assert(!dppIOContractSatisfied<TDPP>(), "empty pack must not conform");
    static_assert(!dppIOContractSatisfied<TDPP, ReadT>(), "missing Write must not conform");
    static_assert(!dppIOContractSatisfied<TDPP, MulT, WriteT>(), "missing Read must not conform");
    static_assert(!dppIOContractSatisfied<TDPP, ReadT, MulT>(), "compute in Write position must not conform");
    static_assert(!dppIOContractSatisfied<TDPP, WriteT, ReadT>(), "swapped Read/Write must not conform");
    static_assert(!dppIOContractSatisfied<TDPP, ReadT, WriteT, MulT>(), "compute after Write must not conform");
    static_assert(!dppIOContractSatisfied<TDPP, ReadT, ReadT, WriteT>(), "Read in compute position must not conform");
    // TransformDPP consumes loose IOps: IOpSequences must not conform (the inverse of
    // passing loose IOps to DivergentBatchTransformDPP, see utest_instantiable_divergent_dpp.h)
    using SeqT = InstantiableOperationSequence<ReadT, MulT, WriteT>;
    static_assert(!dppIOContractSatisfied<TDPP, SeqT>(), "an IOpSequence must not conform");
    static_assert(!dppIOContractSatisfied<TDPP, SeqT, SeqT>(), "IOpSequences must not conform");

    // Compile-fail documentation (the repo has no compile-fail test infra). Each of the
    // following lines, if uncommented, must fail to compile with the quoted message:
    //
    // TransformDPP<>::build(mulIOp, writeIOp);
    //   error: static assertion failed: DPP IO contract violation: the first IO_SPEC.inputIOps IOps
    //   must be complete Read or ReadBack IOps (the data going INTO the DPP).
    //
    // TransformDPP<>::build(buildOperationSequence(readIOp, mulIOp, writeIOp));
    //   error: static assertion failed: DPP IO contract violation: this DPP consumes loose IOps,
    //   not InstantiableOperationSequences. Pass the IOps directly instead of wrapping them with
    //   buildOperationSequence().
    //
    // TransformDPP<>::build(readIOp, mulIOp);
    //   error: static assertion failed: DPP IO contract violation: the last IO_SPEC.outputIOps IOps
    //   must be Write IOps (the data coming OUT of the DPP).
    //
    // TransformDPP<>::build(readIOp);
    //   error: static assertion failed: DPP IO contract violation: not enough IOps. [...]
    //
    // fk::execute(stream, InstantiableDPP<TransformDPP<>, TransformDPPDetails<false, ReadT>, ReadT>{});
    //   error: static assertion failed: fk::execute: the InstantiableDPP does not conform to the
    //   IO contract (IO_SPEC) of its DPP.
} // namespace instantiable_dpp_contract_checks

bool testTransformBuildAndExecute(Stream& stream) {
    constexpr int WIDTH = 67;
    constexpr int HEIGHT = 43;

    Ptr2D<float> input(WIDTH, HEIGHT);
    Ptr2D<float> outputNew(WIDTH, HEIGHT);
    Ptr2D<float> outputLegacy(WIDTH, HEIGHT);

    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            input.at(x, y) = static_cast<float>(x * 100 + y);
        }
    }
    input.upload(stream);

    const auto readIOp = PerThreadRead<ND::_2D, float>::build(input);
    const auto mulIOp = Mul<float>::build(3.f);
    const auto writeNew = PerThreadWrite<ND::_2D, float>::build(outputNew);
    const auto writeLegacy = PerThreadWrite<ND::_2D, float>::build(outputLegacy);

    // New path: DPPs are instantiable values, like Operations
    const auto iDPP = TransformDPP<>::build(readIOp, mulIOp, writeNew);
    static_assert(isInstantiableDPP_v<decltype(iDPP)>, "build() must return an InstantiableDPP");
    execute(stream, iDPP);

    // Legacy path, unchanged
    executeOperations<TransformDPP<>>(stream, readIOp, mulIOp, writeLegacy);

    outputNew.download(stream);
    outputLegacy.download(stream);
    stream.sync();

    bool correct{ true };
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            const float expected = static_cast<float>(x * 100 + y) * 3.f;
            if (outputNew.at(x, y) != expected || outputNew.at(x, y) != outputLegacy.at(x, y)) {
                std::cout << "testTransformBuildAndExecute mismatch at (" << x << ", " << y << "): new = "
                          << outputNew.at(x, y) << ", legacy = " << outputLegacy.at(x, y)
                          << ", expected = " << expected << std::endl;
                correct = false;
            }
        }
    }
    return correct;
}

bool testTransformReadBackStack(Stream& stream) {
    // build() fuses the ReadBack stack (Backwards Vertical Fusion) into canonical form
    // before checking the IO contract, exactly like the legacy Executor path does.
    Ptr2D<float2> input(320, 240);
    Ptr2D<float2> outputNew(16, 16);
    Ptr2D<float2> outputLegacy(16, 16);

    for (int y = 0; y < 240; ++y) {
        for (int x = 0; x < 320; ++x) {
            input.at(x, y) = make_<float2>(static_cast<float>(x), static_cast<float>(y));
        }
    }
    input.upload(stream);

    const auto readIOp = PerThreadRead<ND::_2D, float2>::build(input);
    const auto cropIOp = Crop<>::build(Rect(32, 64, 64, 64));
    const auto resizeIOp = Resize<InterpolationType::INTER_LINEAR>::build(Size(16, 16));
    const auto mulIOp = Mul<float2>::build(make_<float2>(3.f, 5.f));

    const auto iDPP = TransformDPP<>::build(readIOp, cropIOp, resizeIOp, mulIOp,
                                            PerThreadWrite<ND::_2D, float2>::build(outputNew));
    execute(stream, iDPP);

    executeOperations<TransformDPP<>>(stream, readIOp, cropIOp, resizeIOp, mulIOp,
                                      PerThreadWrite<ND::_2D, float2>::build(outputLegacy));

    outputNew.download(stream);
    outputLegacy.download(stream);
    stream.sync();

    bool correct{ true };
    for (int y = 0; y < 16; ++y) {
        for (int x = 0; x < 16; ++x) {
            if (outputNew.at(x, y) != outputLegacy.at(x, y)) {
                std::cout << "testTransformReadBackStack mismatch at (" << x << ", " << y << "): new = ("
                          << outputNew.at(x, y).x << ", " << outputNew.at(x, y).y << "), legacy = ("
                          << outputLegacy.at(x, y).x << ", " << outputLegacy.at(x, y).y << ")" << std::endl;
                correct = false;
            }
        }
    }
    return correct;
}

template <int WIDTH>
bool testTransformThreadFusion(Stream& stream) {
    // Instantiated with a width divisible by the thread fusion factor AND with a
    // non-divisible one, so BOTH threadDivisible dispatches of the InstantiableDPP
    // path (the grid-uniform runtime branch of the front exec) run and are compared
    // against the legacy executeOperations path.
    constexpr int HEIGHT = 17;

    Ptr2D<float> input(WIDTH, HEIGHT);
    Ptr2D<float> outputNew(WIDTH, HEIGHT);
    Ptr2D<float> outputLegacy(WIDTH, HEIGHT);

    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            input.at(x, y) = static_cast<float>(x + y);
        }
    }
    input.upload(stream);

    using TFDPP = TransformDPP<defaultParArch, TF::ENABLED>;

    const auto readIOp = PerThreadRead<ND::_2D, float>::build(input);
    const auto addIOp = Add<float>::build(10.f);

    const auto iDPP = TFDPP::build(readIOp, addIOp, PerThreadWrite<ND::_2D, float>::build(outputNew));

    using DetailsT = typename std::decay_t<decltype(iDPP)>::Details;
    static_assert(DetailsT::TFI::ENABLED, "thread fusion must be enabled for this IOp pack");
    static_assert(DetailsT::TFI::elems_per_thread > 1,
                  "elems_per_thread must be > 1 for the divisibility dispatch to be exercised");
    const bool expectedDivisible = (WIDTH % static_cast<int>(DetailsT::TFI::elems_per_thread)) == 0;
    bool correct{ iDPP.details.threadDivisible == expectedDivisible };
    if (!correct) {
        std::cout << "testTransformThreadFusion<" << WIDTH << ">: threadDivisible = "
                  << iDPP.details.threadDivisible << ", expected " << expectedDivisible << std::endl;
    }

    execute(stream, iDPP);

    executeOperations<TFDPP>(stream, readIOp, addIOp, PerThreadWrite<ND::_2D, float>::build(outputLegacy));

    outputNew.download(stream);
    outputLegacy.download(stream);
    stream.sync();

    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            const float expected = static_cast<float>(x + y) + 10.f;
            if (outputNew.at(x, y) != expected || outputNew.at(x, y) != outputLegacy.at(x, y)) {
                std::cout << "testTransformThreadFusion<" << WIDTH << "> mismatch at (" << x << ", " << y
                          << "): new = " << outputNew.at(x, y) << ", legacy = " << outputLegacy.at(x, y)
                          << ", expected = " << expected << std::endl;
                correct = false;
            }
        }
    }
    return correct;
}

int launch() {
    Stream stream;

    bool correct{ true };
    correct &= testTransformBuildAndExecute(stream);
    correct &= testTransformReadBackStack(stream);
    correct &= testTransformThreadFusion<1023>(stream); // not divisible by the TF factor
    correct &= testTransformThreadFusion<1024>(stream); // divisible: grid-uniform fast branch

    return correct ? 0 : -1;
}
