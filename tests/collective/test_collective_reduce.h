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

#include <fused_kernel/algorithms/collective/reduce.h>
#include "tests/nvtx.h"

#include <vector>
#include <random>
#include <cmath>

/* CPU pass: exercise the associative combine functors and the serial fold
 * oracle (serialReduce). These compile and run on the g++ CPU TU too, so
 * the math of the combine functors is covered without a GPU.
 * CUDA pass: additionally launch warpReduce / blockReduce kernels and
 * cross-check them against the same serial oracle. */

template <typename Combine>
static bool combineMatchesFold(const std::vector<float>& data, float init) {
    Combine c;
    float acc = init;
    for (float x : data) acc = c(acc, x);
    const float ref = fk::serialReduce<Combine>(data.data(), (int)data.size(), init, c);
    return std::fabs(acc - ref) <= 1e-5f * std::fmax(1.f, std::fabs(ref));
}

#if defined(__NVCC__)
#include <cstdio>

template <typename Combine, int BLOCK_SIZE>
__global__ void blockReduceKernel(const float* in, float* out, int n, float init) {
    __shared__ float scratch[BLOCK_SIZE / 32];
    float acc = init;
    for (int i = threadIdx.x; i < n; i += BLOCK_SIZE) acc = Combine{}(acc, in[i]);
    const float r = fk::blockReduce<Combine, BLOCK_SIZE>(acc, scratch, Combine{}, true);
    if (threadIdx.x == 0) *out = r;
}

template <int WARP_WIDTH>
__global__ void warpReduceSumKernel(const float* in, float* out) {
    float v = in[threadIdx.x];
    out[threadIdx.x] = fk::warpReduce<fk::ReduceSum<float>, WARP_WIDTH>(v);
}

template <typename Combine, int BLOCK_SIZE>
static bool checkBlock(const std::vector<float>& h, float init) {
    float *d, *dout; cudaMalloc(&d, h.size()*4); cudaMalloc(&dout, 4);
    cudaMemcpy(d, h.data(), h.size()*4, cudaMemcpyHostToDevice);
    blockReduceKernel<Combine, BLOCK_SIZE><<<1, BLOCK_SIZE>>>(d, dout, (int)h.size(), init);
    cudaDeviceSynchronize();
    float got; cudaMemcpy(&got, dout, 4, cudaMemcpyDeviceToHost);
    cudaFree(d); cudaFree(dout);
    Combine c; float ref = init; for (float x : h) ref = c(ref, x);
    const bool ok = std::fabs(got - ref) <= 1e-5f * std::fmax(1.f, std::fabs(ref));
    if (!ok) printf("  blockReduce<BS=%d> got=%.5f ref=%.5f FAIL\n", BLOCK_SIZE, got, ref);
    return ok;
}

static bool runDeviceChecks() {
    bool ok = true;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-5.f, 5.f);

    // warpReduce sum across 32 lanes, broadcast to every lane
    std::vector<float> w(32); for (auto& x : w) x = dist(rng);
    float *d, *dout; cudaMalloc(&d, 128); cudaMalloc(&dout, 128);
    cudaMemcpy(d, w.data(), 128, cudaMemcpyHostToDevice);
    warpReduceSumKernel<32><<<1, 32>>>(d, dout);
    cudaDeviceSynchronize();
    std::vector<float> got(32); cudaMemcpy(got.data(), dout, 128, cudaMemcpyDeviceToHost);
    cudaFree(d); cudaFree(dout);
    float ref = 0; for (float x : w) ref += x;
    for (int lane : {0, 1, 17, 31})
        ok &= std::fabs(got[lane] - ref) <= 1e-5f * std::fmax(1.f, std::fabs(ref));

    // blockReduce sum + max, several block sizes and a non-divisible count
    std::vector<float> big(4096 + 37); for (auto& x : big) x = dist(rng);
    ok &= checkBlock<fk::ReduceSum<float>, 128>(big, 0.f);
    ok &= checkBlock<fk::ReduceSum<float>, 256>(big, 0.f);
    ok &= checkBlock<fk::ReduceMax<float>, 256>(big, -3.4e38f);
    ok &= checkBlock<fk::ReduceMax<float>, 512>(big, -3.4e38f);
    return ok;
}
#endif // __NVCC__

int launch() {
    bool passed = true;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-5.f, 5.f);
    std::vector<float> data(100); for (auto& x : data) x = dist(rng);

    {
        PUSH_RANGE_RAII p("combine_functors_cpu");
        passed &= combineMatchesFold<fk::ReduceSum<float>>(data, 0.f);
        passed &= combineMatchesFold<fk::ReduceMax<float>>(data, -3.4e38f);
        passed &= combineMatchesFold<fk::ReduceMin<float>>(data, 3.4e38f);
    }
#if defined(__NVCC__)
    {
        PUSH_RANGE_RAII p("collective_reduce_device");
        passed &= runDeviceChecks();
    }
#endif
    return passed ? 0 : -1;
}
