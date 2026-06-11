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

#include <tests/main.h>

#include <fused_kernel/algorithms/algorithms.h>
#include <fused_kernel/core/execution_model/executors.h>

#include <iostream>
#include <vector>

using namespace fk;

/* Regression test for issue #250: the DivergentBatchTransformDPP Executor
   must launch max(sequence z extents) planes, not their sum. With two
   sequences that EACH carry a full-batch read of B planes and a selector
   that partitions the same global plane space (plane 0 -> seq1, others ->
   seq2), the launch must have grid.z == B. Summing launches 2*B planes and
   planes B..2*B-1 read/write out of bounds (detected here with a canary
   tensor allocated right after the output). */

#if defined(__NVCC__) || defined(__CUDACC__)

struct PlaneZeroSelector {
    FK_HOST_DEVICE_FUSE uint at(const uint& z) { return z == 0 ? 1u : 2u; }
};

int launch() {
    constexpr int W = 8, H = 4, B = 3;
    constexpr float CANARY = 777.f;

    Stream stream;

    Tensor<float> in(W, H, B, 1, MemType::Device);
    Tensor<float> out(W, H, B, 1, MemType::Device);
    Tensor<float> canary(W, H, B, 1, MemType::Device);  // right after 'out'

    std::vector<float> hostIn(W * H * B);
    for (int i = 0; i < W * H * B; ++i) {
        hostIn[i] = static_cast<float>(i % 100);
    }
    gpuErrchk(cudaMemcpy(in.ptr().data, hostIn.data(),
                         W * H * B * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(out.ptr().data, 0, W * H * B * sizeof(float)));
    const std::vector<float> sentinel(W * H * B, CANARY);
    gpuErrchk(cudaMemcpy(canary.ptr().data, sentinel.data(),
                         W * H * B * sizeof(float), cudaMemcpyHostToDevice));

    const auto seq1 = buildOperationSequence(TensorRead<float>::build(in),
                                             Mul<float>::build(10.f),
                                             TensorWrite<float>::build(out));
    const auto seq2 = buildOperationSequence(TensorRead<float>::build(in),
                                             Add<float>::build(1.f),
                                             TensorWrite<float>::build(out));

    Executor<DivergentBatchTransformDPP<ParArch::GPU_NVIDIA, PlaneZeroSelector>>::
        executeOperations(stream, seq1, seq2);
    stream.sync();

    std::vector<float> got(W * H * B), can(W * H * B);
    gpuErrchk(cudaMemcpy(got.data(), out.ptr().data,
                         W * H * B * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(can.data(), canary.ptr().data,
                         W * H * B * sizeof(float), cudaMemcpyDeviceToHost));

    int wrong = 0;
    for (int z = 0; z < B; ++z) {
        for (int i = 0; i < W * H; ++i) {
            const float x = hostIn[z * W * H + i];
            const float expected = (z == 0) ? x * 10.f : x + 1.f;
            if (got[z * W * H + i] != expected) {
                ++wrong;
            }
        }
    }
    int corrupted = 0;
    for (int i = 0; i < W * H * B; ++i) {
        if (can[i] != CANARY) {
            ++corrupted;
        }
    }

    if (wrong != 0 || corrupted != 0) {
        std::cout << "Divergent executor plane-count regression: wrong="
                  << wrong << " canaryCorrupted=" << corrupted << std::endl;
        return -1;
    }
    std::cout << "Divergent executor launches selector-defined plane count: Success!!"
              << std::endl;
    return 0;
}

#else // CPU-only build: the Divergent Executor has no CPU specialization yet

int launch() {
    std::cout << "Divergent executor plane-count test: skipped on CPU (no "
                 "Executor<DivergentBatchTransformDPP<ParArch::CPU>> yet)"
              << std::endl;
    return 0;
}

#endif
