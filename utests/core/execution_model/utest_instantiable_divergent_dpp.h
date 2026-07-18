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

// Tests DivergentBatchTransformDPP through the InstantiableDPP path:
// DPP::build(iOpSequences...) + fk::execute(stream, instantiableDPP), checking the
// per-sequence IO contract and comparing results against the legacy Executor path.
// The Divergent Horizontal Fusion DPP is GPU-only (its CPU backend is not implemented).

#define __ONLY_CU__
#include <tests/main.h>

#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/fused_kernel.h>
#include <iostream>

using namespace fk;

struct PlaneSelector {
    FK_HOST_DEVICE_FUSE uint at(const uint& index) { return index == 0 ? 0u : 1u; }
};

int launch() {
    Stream stream;

    constexpr int WIDTH = 64;
    constexpr int HEIGHT = 32;

    Ptr2D<float> imgA(WIDTH, HEIGHT);
    Ptr2D<float> imgB(WIDTH, HEIGHT);
    Tensor<float> outputNew(WIDTH, HEIGHT, 2);
    Tensor<float> outputLegacy(WIDTH, HEIGHT, 2);

    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            imgA.at(x, y) = static_cast<float>(x + y);
            imgB.at(x, y) = static_cast<float>(x * y);
        }
    }
    imgA.upload(stream);
    imgB.upload(stream);

    using DivergentDPP = DivergentBatchTransformDPP<ParArch::GPU_NVIDIA, PlaneSelector>;

    // Contract: the divergent DPP consumes IOpSequences (one per divergent branch),
    // each holding 1 Read/ReadBack IOp, an optional compute chain, and 1 Write IOp.
    {
        using ReadT = decltype(PerThreadRead<ND::_2D, float>::build(imgA));
        using MulT = decltype(Mul<float>::build(2.f));
        using WriteT = decltype(TensorWrite<float>::build(outputNew));
        using SeqT = InstantiableOperationSequence<ReadT, MulT, WriteT>;
        using BadSeqT = InstantiableOperationSequence<MulT, WriteT>;
        static_assert(DivergentDPP::IO_SPEC.argsAreIOpSequences, "Divergent DPP consumes IOpSequences");
        static_assert(dppIOContractSatisfied<DivergentDPP, SeqT, SeqT>(), "conforming sequences must pass");
        static_assert(!dppIOContractSatisfied<DivergentDPP, SeqT, BadSeqT>(), "a sequence without a Read must fail");
        static_assert(!dppIOContractSatisfied<DivergentDPP>(), "at least one sequence is required");
        static_assert(!dppIOContractSatisfied<DivergentDPP, ReadT, WriteT>(),
                      "loose IOps must fail: the divergent DPP consumes IOpSequences");
        // Compile-fail documentation: if uncommented, must fail with the quoted message:
        // DivergentDPP::build(PerThreadRead<ND::_2D, float>::build(imgA), Mul<float>::build(2.f),
        //                     TensorWrite<float>::build(outputNew));
        //   error: static assertion failed: DPP IO contract violation: this DPP consumes
        //   InstantiableOperationSequences (see buildOperationSequence()), not loose IOps.
    }

    const auto seq1New = buildOperationSequence(PerThreadRead<ND::_2D, float>::build(imgA),
                                                Mul<float>::build(2.0f),
                                                TensorWrite<float>::build(outputNew));
    const auto seq2New = buildOperationSequence(PerThreadRead<ND::_2D, float>::build(imgB),
                                                Add<float>::build(100.0f),
                                                TensorWrite<float>::build(outputNew));

    const auto iDPP = DivergentDPP::build(seq1New, seq2New);
    static_assert(isInstantiableDPP_v<decltype(iDPP)>, "build() must return an InstantiableDPP");
    execute(stream, iDPP);

    const auto seq1Legacy = buildOperationSequence(PerThreadRead<ND::_2D, float>::build(imgA),
                                                   Mul<float>::build(2.0f),
                                                   TensorWrite<float>::build(outputLegacy));
    const auto seq2Legacy = buildOperationSequence(PerThreadRead<ND::_2D, float>::build(imgB),
                                                   Add<float>::build(100.0f),
                                                   TensorWrite<float>::build(outputLegacy));
    Executor<DivergentDPP>::executeOperations(stream, seq1Legacy, seq2Legacy);

    outputNew.download(stream);
    outputLegacy.download(stream);
    stream.sync();

    bool correct{ true };
    for (int z = 0; z < 2; ++z) {
        for (int y = 0; y < HEIGHT; ++y) {
            for (int x = 0; x < WIDTH; ++x) {
                const float expected = z == 0 ? imgA.at(x, y) * 2.f : imgB.at(x, y) + 100.f;
                const float newVal = outputNew.at(Point{ x, y, z });
                const float legacyVal = outputLegacy.at(Point{ x, y, z });
                if (newVal != expected || newVal != legacyVal) {
                    std::cout << "Divergent mismatch at (" << x << ", " << y << ", " << z << "): new = "
                              << newVal << ", legacy = " << legacyVal << ", expected = " << expected << std::endl;
                    correct = false;
                }
            }
        }
    }

    return correct ? 0 : -1;
}
