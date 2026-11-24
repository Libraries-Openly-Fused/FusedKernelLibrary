/* Copyright 2026 Oscar Amoros Huguet
   Copyright 2026 Johnny Nunez

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

/* FlashAttentionMmaDPP (mma.sync.m16n8k16 bf16) against the double-precision
 * oracle, through executeFlashAttentionMma with raw bf16 Q/K/V (the cp.async
 * schedule): dense, causal, ragged sequence lengths, and a small grid that
 * takes the split-KV forward and its combine kernel. The oracle sees the
 * bf16-rounded inputs. The FP8 QK^T path (FK_ENABLE_FP8_QK, sm_120a) is not
 * tested here.
 *
 * The kernel's inline PTX (mma.sync bf16, ldmatrix, cp.async) needs sm_80;
 * on older GPUs the test prints SKIPPED and passes. */

#define __ONLY_CU__ // This file is only generated and compiled with nvcc, not with the host compiler
#include <tests/main.h>

#include <fused_kernel/algorithms/attention/flash_attention_mma.h>
#include <utests/algorithm/attention/attention_test_common.h>

#include <random>

namespace fk {

// P is rounded to bf16 (8 significant bits) for the PV mma: each output is a
// convex combination of V values in [-1, 1], so the rounding error is below
// 2^-8 = 3.9e-3. The softmax uses ex2.approx; 1e-2 leaves margin for both.
constexpr double MMA_BF16_TOL = 1e-2;

template <int HEAD_DIM>
static void testMmaBf16(const char* name, const int bh, const int seqQ, const int seqK, const bool causal,
                        const unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    const size_t nQ = (size_t)bh * seqQ * HEAD_DIM, nK = (size_t)bh * seqK * HEAD_DIM;
    std::vector<__nv_bfloat16> bq(nQ), bk(nK), bv(nK);
    std::vector<double> dq(nQ), dk(nK), dv(nK), ref;
    for (size_t i = 0; i < nQ; ++i) { bq[i] = __float2bfloat16(dist(rng)); dq[i] = __bfloat162float(bq[i]); }
    for (size_t i = 0; i < nK; ++i) { bk[i] = __float2bfloat16(dist(rng)); dk[i] = __bfloat162float(bk[i]); }
    for (size_t i = 0; i < nK; ++i) { bv[i] = __float2bfloat16(dist(rng)); dv[i] = __bfloat162float(bv[i]); }
    cpuAttention(dq, dk, dv, ref, bh, seqQ, seqK, HEAD_DIM, 1.0 / std::sqrt((double)HEAD_DIM), causal);

    Stream stream;
    auto q = uploadBuffer(bq, stream);
    auto k = uploadBuffer(bk, stream);
    auto v = uploadBuffer(bv, stream);
    auto o = zeroBuffer<float>(nQ, stream);
    const auto qIOp = makeAttentionRead(static_cast<const __nv_bfloat16*>(q.ptr().data), bh, seqQ, HEAD_DIM);
    const auto kIOp = makeAttentionRead(static_cast<const __nv_bfloat16*>(k.ptr().data), bh, seqK, HEAD_DIM);
    const auto vIOp = makeAttentionRead(static_cast<const __nv_bfloat16*>(v.ptr().data), bh, seqK, HEAD_DIM);
    static_assert(isRawBf16Read<std::decay_t<decltype(kIOp)>>, "bf16 K/V must take the raw (cp.async) schedule");
    executeFlashAttentionMma<HEAD_DIM>(qIOp, kIOp, vIOp, o.ptr().data, bh, seqQ, seqK, causal, stream);
    const std::vector<float> got = downloadBuffer(o, stream);
    report(name, maxAbsErr(got, ref), MMA_BF16_TOL);
}

int launch_impl() {
    int device = 0;
    cudaDeviceProp prop{};
    gpuErrchk(cudaGetDevice(&device));
    gpuErrchk(cudaGetDeviceProperties(&prop, device));
    if (prop.major < 8) {
        std::cout << "SKIPPED (FlashAttentionMmaDPP needs sm_80, this GPU is sm_" << prop.major << prop.minor << ")"
                  << std::endl;
        return 0;
    }
    testMmaBf16<64>("FA-MMA bf16 d64 bh2 s256", 2, 256, 256, false, 1);
    testMmaBf16<64>("FA-MMA bf16 d64 bh2 s256 causal", 2, 256, 256, true, 2);
    testMmaBf16<64>("FA-MMA bf16 d64 ragged s100/s200", 2, 100, 200, false, 3);
    testMmaBf16<128>("FA-MMA bf16 d128 bh2 s128 causal", 2, 128, 128, true, 4);
    testMmaBf16<128>("FA-MMA bf16 d128 ragged s70/s150 causal", 2, 70, 150, true, 5);
    // 16 q-blocks and 16 KV tiles per row: under one wave, so split-KV + combine
    testMmaBf16<128>("FA-MMA bf16 d128 bh1 s1024 (split-KV)", 1, 1024, 1024, false, 6);
    if (failures == 0) { return 0; }
    std::cout << failures << " flash attention mma test(s) FAILED" << std::endl;
    return -1;
}

} // namespace fk

int launch() {
    return fk::launch_impl();
}
