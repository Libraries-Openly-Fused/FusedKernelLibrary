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
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/logical.h>
#include "tests/nvtx.h"

#include <vector>
#include <random>
#include <cmath>

/* The reduction operator is a STANDARD FKL binary Op (Add / Max / Min),
 * passed as a template argument to the Reduce*DPP. No bespoke combine
 * functors, no duplicated reduction code.
 *
 * CPU pass: exercise serialReduce (the oracle) against a hand fold using the
 * same Op. CUDA pass: additionally launch ReduceWarpDPP / ReduceBlockDPP /
 * ReduceGridDPP and cross-check them against the same serial oracle. */

using SumOp = fk::Add<float, float, float, fk::BinaryType>;
using MaxOp = fk::Max<float, float, float, fk::BinaryType>;
using MinOp = fk::Min<float, float, float, fk::BinaryType>;

template <typename ReduceOp>
static bool foldMatchesOracle(const std::vector<float>& data, float init) {
    float acc = init;
    for (float x : data) acc = ReduceOp::exec(acc, x);
    const float ref = fk::serialReduce<ReduceOp>(data.data(), (int)data.size(), init);
    return std::fabs(acc - ref) <= 1e-5f * std::fmax(1.f, std::fabs(ref));
}

#if defined(__NVCC__)
#include <cstdio>

template <typename ReduceOp, int BLOCK_SIZE>
__global__ void blockReduceKernel(const float* in, float* out, int n, float init) {
    __shared__ float scratch[BLOCK_SIZE / 32];
    float acc = init;
    for (int i = threadIdx.x; i < n; i += BLOCK_SIZE) acc = ReduceOp::exec(acc, in[i]);
    const float r = fk::ReduceBlockDPP<ReduceOp, BLOCK_SIZE>::exec(acc, scratch, true);
    if (threadIdx.x == 0) *out = r;
}

template <int WARP_WIDTH>
__global__ void warpReduceSumKernel(const float* in, float* out) {
    float v = in[threadIdx.x];
    out[threadIdx.x] = fk::ReduceWarpDPP<SumOp, WARP_WIDTH>::exec(v);
}

template <typename ReduceOp, int BLOCK_SIZE>
__global__ void gridReduceKernel(const float* in, float* gAccum, int n) {
    __shared__ float scratch[BLOCK_SIZE / 32];
    float acc = (blockIdx.x * BLOCK_SIZE + threadIdx.x < n)
              ? in[blockIdx.x * BLOCK_SIZE + threadIdx.x] : 0.f;
    fk::ReduceGridDPP<ReduceOp, BLOCK_SIZE>::exec(acc, scratch, gAccum);
}

template <typename ReduceOp, int BLOCK_SIZE>
static bool checkBlock(const std::vector<float>& h, float init) {
    float *d, *dout; cudaMalloc(&d, h.size()*4); cudaMalloc(&dout, 4);
    cudaMemcpy(d, h.data(), h.size()*4, cudaMemcpyHostToDevice);
    blockReduceKernel<ReduceOp, BLOCK_SIZE><<<1, BLOCK_SIZE>>>(d, dout, (int)h.size(), init);
    cudaDeviceSynchronize();
    float got; cudaMemcpy(&got, dout, 4, cudaMemcpyDeviceToHost);
    cudaFree(d); cudaFree(dout);
    float ref = init; for (float x : h) ref = ReduceOp::exec(ref, x);
    const bool ok = std::fabs(got - ref) <= 1e-5f * std::fmax(1.f, std::fabs(ref));
    if (!ok) printf("  ReduceBlockDPP<BS=%d> got=%.5f ref=%.5f FAIL\n", BLOCK_SIZE, got, ref);
    return ok;
}

static bool runDeviceChecks() {
    bool ok = true;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-5.f, 5.f);

    // ReduceWarpDPP sum across 32 lanes, broadcast to every lane
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

    // ReduceBlockDPP sum + max, several block sizes and a non-divisible count
    std::vector<float> big(4096 + 37); for (auto& x : big) x = dist(rng);
    ok &= checkBlock<SumOp, 128>(big, 0.f);
    ok &= checkBlock<SumOp, 256>(big, 0.f);
    ok &= checkBlock<MaxOp, 256>(big, -3.4e38f);
    ok &= checkBlock<MaxOp, 512>(big, -3.4e38f);

    // ReduceGridDPP sum across a multi-block grid, atomic combine into gAccum
    {
        constexpr int BS = 256;
        std::vector<float> g(4096 + 37); for (auto& x : g) x = dist(rng);
        const int n = (int)g.size();
        const int blocks = (n + BS - 1) / BS;
        float *dg, *dacc; cudaMalloc(&dg, n*4); cudaMalloc(&dacc, 4);
        cudaMemcpy(dg, g.data(), n*4, cudaMemcpyHostToDevice);
        float zero = 0.f; cudaMemcpy(dacc, &zero, 4, cudaMemcpyHostToDevice);
        gridReduceKernel<SumOp, BS><<<blocks, BS>>>(dg, dacc, n);
        cudaDeviceSynchronize();
        float gotg; cudaMemcpy(&gotg, dacc, 4, cudaMemcpyDeviceToHost);
        cudaFree(dg); cudaFree(dacc);
        float refg = 0.f; for (float x : g) refg += x;
        const bool okg = std::fabs(gotg - refg) <= 1e-3f * std::fmax(1.f, std::fabs(refg));
        if (!okg) printf("  ReduceGridDPP<Sum> got=%.4f ref=%.4f FAIL\n", gotg, refg);
        ok &= okg;
    }
    return ok;
}
#endif // __NVCC__

int launch() {
    bool passed = true;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-5.f, 5.f);
    std::vector<float> data(100); for (auto& x : data) x = dist(rng);

    {
        PUSH_RANGE_RAII p("reduce_ops_cpu");
        passed &= foldMatchesOracle<SumOp>(data, 0.f);
        passed &= foldMatchesOracle<MaxOp>(data, -3.4e38f);
        passed &= foldMatchesOracle<MinOp>(data, 3.4e38f);
    }
#if defined(__NVCC__)
    {
        PUSH_RANGE_RAII p("collective_reduce_device");
        passed &= runDeviceChecks();
    }
#endif
    return passed ? 0 : -1;
}
