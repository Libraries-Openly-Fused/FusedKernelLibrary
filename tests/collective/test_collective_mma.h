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

#define __ONLY_CU__  // tensor-core MMA + cp.async are device-only
#include <tests/main.h>

#include <fused_kernel/algorithms/collective/mma.h>
#include <fused_kernel/algorithms/collective/copy.h>

#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <cuda_bf16.h>

using namespace fk;

/* Verify MmaBf16_16x8x16 against an fp64 oracle. The per-lane fragment
 * mapping for m16n8k16 (PTX ISA), VERIFIED here against the oracle:
 *   group g = lane>>2, tig t = lane&3; each .b32 packs 2 consecutive-k bf16
 *   A (16x16 row-major, m,k): a0=row g   k[2t,2t+1]; a1=row g+8 k[2t,2t+1]
 *                             a2=row g   k[2t+8,..]; a3=row g+8 k[2t+8,..]
 *   B (.col => 8x16 row-major, n,k): b0=col g k[2t,2t+1]; b1=col g k[2t+8,..]
 *   D (16x8 f32): d0,d1 = row g cols 2t,2t+1 ; d2,d3 = row g+8 cols 2t,2t+1
 * This is the bf16 analogue of the fp8 mapping validated in
 * spikes/spike_fp8_qk_mapping.cu. */
__global__ void mmaBf16Kernel(const __nv_bfloat16* A,   // 16x16 row-major (m,k)
                              const __nv_bfloat16* B,   // 8x16  row-major (n,k)
                              float* D) {                // 16x8 row-major (m,n)
    const int lane = threadIdx.x;
    const int g = lane >> 2, t = lane & 3;
    uint32_t a[4], b[2];
    auto pack = [](const __nv_bfloat16* p) {
        uint32_t r; __nv_bfloat16 two[2] = { p[0], p[1] };
        memcpy(&r, two, 4); return r;
    };
    // m16n8k16: each .b32 packs 2 consecutive-k bf16. tig t in [0,4).
    // a0: row g    , k = 2t, 2t+1        a1: row g+8 , k = 2t, 2t+1
    // a2: row g    , k = 2t+8, 2t+9      a3: row g+8 , k = 2t+8, 2t+9
    a[0] = pack(&A[(g)     * 16 + 2 * t]);
    a[1] = pack(&A[(g + 8) * 16 + 2 * t]);
    a[2] = pack(&A[(g)     * 16 + 2 * t + 8]);
    a[3] = pack(&A[(g + 8) * 16 + 2 * t + 8]);
    // B operand .col => stored (n,k) row-major; lane holds col n=g.
    // b0: k = 2t, 2t+1     b1: k = 2t+8, 2t+9
    b[0] = pack(&B[g * 16 + 2 * t]);
    b[1] = pack(&B[g * 16 + 2 * t + 8]);

    float d[4] = {};
    // Through the FKL Op: bundle the per-lane fragments and accumulate.
    MmaFragment<MmaBf16_16x8x16> frag{ a, b, d };
    MmaBf16Op::exec(frag);

    D[(g)     * 8 + 2 * t]     = d[0];
    D[(g)     * 8 + 2 * t + 1] = d[1];
    D[(g + 8) * 8 + 2 * t]     = d[2];
    D[(g + 8) * 8 + 2 * t + 1] = d[3];
}

__global__ void cpAsyncKernel(const float* gsrc, float* gdst, int n4) {
    extern __shared__ float smem[];
    const int tid = threadIdx.x;
    // Stage gmem->smem through the FKL Op (16-byte chunks via int4).
    using Stage = fk::CpAsyncStage16Op<int4>;
    const auto params = Stage::Params{ reinterpret_cast<int4*>(smem),
                                       reinterpret_cast<const int4*>(gsrc) };
    for (int i = tid; i < n4; i += blockDim.x)
        Stage::exec(Point{ i, 0, 0 }, params);
    fk::cpAsyncCommit();
    fk::cpAsyncWaitGroups<0>();
    __syncthreads();
    for (int i = tid; i < n4; i += blockDim.x)
        reinterpret_cast<float4*>(gdst)[i] = reinterpret_cast<float4*>(smem)[i];
}

static int failures = 0;

static void testMma() {
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float> Af(16*16), Bf(8*16);
    for (auto& x : Af) x = dist(rng);
    for (auto& x : Bf) x = dist(rng);
    std::vector<__nv_bfloat16> Ah(16*16), Bh(8*16);
    for (int i = 0; i < 16*16; ++i) Ah[i] = __float2bfloat16(Af[i]);
    for (int i = 0; i < 8*16;  ++i) Bh[i] = __float2bfloat16(Bf[i]);

    __nv_bfloat16 *dA, *dB; float* dD;
    cudaMalloc(&dA, 16*16*2); cudaMalloc(&dB, 8*16*2); cudaMalloc(&dD, 16*8*4);
    cudaMemcpy(dA, Ah.data(), 16*16*2, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, Bh.data(), 8*16*2, cudaMemcpyHostToDevice);
    mmaBf16Kernel<<<1, 32>>>(dA, dB, dD);
    cudaDeviceSynchronize();
    std::vector<float> D(16*8); cudaMemcpy(D.data(), dD, 16*8*4, cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dD);

    // fp64 oracle using the bf16-rounded inputs (mma rounds inputs to bf16).
    // A is (m,k) row-major; B is (n,k) row-major -> D[m,n] = sum_k A[m,k]*B[n,k].
    double maxErr = 0;
    for (int m = 0; m < 16; ++m)
        for (int n = 0; n < 8; ++n) {
            double acc = 0;
            for (int k = 0; k < 16; ++k)
                acc += (double)__bfloat162float(Ah[m*16+k]) * (double)__bfloat162float(Bh[n*16+k]);
            maxErr = std::max(maxErr, std::abs((double)D[m*8+n] - acc));
        }
    // bf16 mantissa ~8 bits; k=16 accumulate in f32 -> small abs error
    if (maxErr > 5e-2) { printf("FAIL MmaBf16 maxErr=%.4e\n", maxErr); ++failures; }
    else printf("MmaBf16_16x8x16 vs fp64 oracle: PASS (maxErr=%.2e)\n", maxErr);
}

static void testCpAsync() {
    const int n4 = 64;          // 64 float4 = 256 floats
    std::vector<float> src(n4*4), dst(n4*4, -1.f);
    for (int i = 0; i < n4*4; ++i) src[i] = (float)i;
    float *dS, *dD; cudaMalloc(&dS, n4*16); cudaMalloc(&dD, n4*16);
    cudaMemcpy(dS, src.data(), n4*16, cudaMemcpyHostToDevice);
    cpAsyncKernel<<<1, 128, n4*16>>>(dS, dD, n4);
    cudaDeviceSynchronize();
    cudaMemcpy(dst.data(), dD, n4*16, cudaMemcpyDeviceToHost);
    cudaFree(dS); cudaFree(dD);
    bool ok = true;
    for (int i = 0; i < n4*4; ++i) ok &= (dst[i] == (float)i);
    if (ok) printf("cpAsync16 global->smem->global round-trip: PASS\n");
    else { printf("FAIL cpAsync16 round-trip\n"); ++failures; }
}

int launch() {
    testMma();
    testCpAsync();
    return failures == 0 ? 0 : -1;
}
