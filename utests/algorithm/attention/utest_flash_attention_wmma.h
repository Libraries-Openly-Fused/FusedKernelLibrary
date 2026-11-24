/* Copyright 2026 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

/* FlashAttentionWmmaDPP (RDNA4 gfx12 WMMA) against the double-precision
 * oracle, through executeFlashAttentionWmma: raw bf16 Q/K/V (dense, causal,
 * ragged, cross-attention, d32/64/128), fp32 and int8 K/V prologues, a fused
 * epilogue, the FP8 QK^T path (fp8 KV cache), and executeFlashAttentionAuto.
 * On a GPU without gfx12 WMMA it prints SKIPPED and passes. */

#define __ONLY_HIP__ // Generated as _hip only: gfx12 WMMA has no CUDA or CPU counterpart here
#include <tests/main.h>

#include <fused_kernel/algorithms/attention/flash_attention_wmma.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <utests/algorithm/attention/attention_test_common.h>

#include <random>

namespace fk {

// bf16 operands, fp32 accumulation: with bf16 inputs the oracle sees the
// same values and only P is rounded (8 significant bits; each output is a
// convex combination of V values in [-1, 1], so below 2^-8 = 3.9e-3); fp32
// or int8 inputs are rounded to bf16 too, which perturbs the scores by a few
// 1e-3. 1e-2 bounds both (the bound of utest_flash_attention_mma).
constexpr double WMMA_BF16_TOL = 1e-2;
// FP8 QK^T: Q is quantized to e4m3 in-kernel (3 mantissa bits, relative
// error up to 2^-4 per element), so the scores move by ~1e-2 at d64: the
// outputs by a few 1e-3, bounded by 3e-2.
constexpr double WMMA_FP8_TOL = 3e-2;

enum class WmmaInput { BF16, FP32, INT8_KV, FP8_KV };

template <int HEAD_DIM, WmmaInput IN, int BLOCK_KV = 32, int NUM_WAVES = 8>
static void testWmma(const char* name, const int bh, const int seqQ, const int seqK, const bool causal,
                     const double tol, const unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    const size_t nQ = (size_t)bh * seqQ * HEAD_DIM, nK = (size_t)bh * seqK * HEAD_DIM;
    const size_t nTok = (size_t)bh * seqK;
    std::vector<float> hq(nQ), hk(nK), hv(nK);
    for (auto& x : hq) x = dist(rng);
    for (auto& x : hk) x = dist(rng);
    for (auto& x : hv) x = dist(rng);
    std::vector<double> dq(nQ), dk(nK), dv(nK), ref;
    const double scale = 1.0 / std::sqrt((double)HEAD_DIM);

    Stream stream;
    auto o = zeroBuffer<float>(nQ, stream);
    if constexpr (IN == WmmaInput::BF16) {
        std::vector<attn_bf16> bq(nQ), bk(nK), bv(nK);
        for (size_t i = 0; i < nQ; ++i) { bq[i] = __float2bfloat16(hq[i]); dq[i] = __bfloat162float(bq[i]); }
        for (size_t i = 0; i < nK; ++i) {
            bk[i] = __float2bfloat16(hk[i]); dk[i] = __bfloat162float(bk[i]);
            bv[i] = __float2bfloat16(hv[i]); dv[i] = __bfloat162float(bv[i]);
        }
        auto q = uploadBuffer(bq, stream);
        auto k = uploadBuffer(bk, stream);
        auto v = uploadBuffer(bv, stream);
        const auto qIOp = makeAttentionRead(static_cast<const attn_bf16*>(q.ptr().data), bh, seqQ, HEAD_DIM);
        const auto kIOp = makeAttentionRead(static_cast<const attn_bf16*>(k.ptr().data), bh, seqK, HEAD_DIM);
        const auto vIOp = makeAttentionRead(static_cast<const attn_bf16*>(v.ptr().data), bh, seqK, HEAD_DIM);
        executeFlashAttentionWmma<HEAD_DIM, BLOCK_KV, NUM_WAVES>(qIOp, kIOp, vIOp, o.ptr().data, bh, seqQ, seqK, causal,
                                                                 stream);
        const std::vector<float> got = downloadBuffer(o, stream);
        cpuAttention(dq, dk, dv, ref, bh, seqQ, seqK, HEAD_DIM, scale, causal);
        report(name, maxAbsErr(got, ref), tol);
    } else if constexpr (IN == WmmaInput::FP32) {
        for (size_t i = 0; i < nQ; ++i) dq[i] = hq[i];
        for (size_t i = 0; i < nK; ++i) { dk[i] = hk[i]; dv[i] = hv[i]; }
        auto q = uploadBuffer(hq, stream);
        auto k = uploadBuffer(hk, stream);
        auto v = uploadBuffer(hv, stream);
        const auto qIOp = makeAttentionRead(q.ptr().data, bh, seqQ, HEAD_DIM);
        const auto kIOp = makeAttentionRead(k.ptr().data, bh, seqK, HEAD_DIM);
        const auto vIOp = makeAttentionRead(v.ptr().data, bh, seqK, HEAD_DIM);
        executeFlashAttentionWmma<HEAD_DIM, BLOCK_KV, NUM_WAVES>(qIOp, kIOp, vIOp, o.ptr().data, bh, seqQ, seqK, causal,
                                                                 stream);
        const std::vector<float> got = downloadBuffer(o, stream);
        cpuAttention(dq, dk, dv, ref, bh, seqQ, seqK, HEAD_DIM, scale, causal);
        report(name, maxAbsErr(got, ref), tol);
    } else {
        // quantized caches: the oracle sees the dequantized values
        std::vector<int8_t> k8(nK), v8(nK);
        std::vector<float> kSc(nTok), vSc(nTok);
        for (size_t i = 0; i < nQ; ++i) dq[i] = hq[i];
        if constexpr (IN == WmmaInput::INT8_KV) {
            quantizeKVCacheHost(hk.data(), k8.data(), kSc.data(), (int)nTok, HEAD_DIM);
            quantizeKVCacheHost(hv.data(), v8.data(), vSc.data(), (int)nTok, HEAD_DIM);
            for (size_t i = 0; i < nK; ++i) {
                dk[i] = (double)k8[i] * kSc[i / HEAD_DIM];
                dv[i] = (double)v8[i] * vSc[i / HEAD_DIM];
            }
        } else {
            quantizeKVCacheFp8Host(hk.data(), k8.data(), kSc.data(), (int)nTok, HEAD_DIM);
            quantizeKVCacheFp8Host(hv.data(), v8.data(), vSc.data(), (int)nTok, HEAD_DIM);
            for (size_t i = 0; i < nK; ++i) {
                dk[i] = (double)attnFp8ToF32(k8[i]) * kSc[i / HEAD_DIM];
                dv[i] = (double)attnFp8ToF32(v8[i]) * vSc[i / HEAD_DIM];
            }
        }
        auto q = uploadBuffer(hq, stream);
        auto kd = uploadBuffer(k8, stream);
        auto vd = uploadBuffer(v8, stream);
        auto ks = uploadBuffer(kSc, stream);
        auto vs = uploadBuffer(vSc, stream);
        const auto qIOp = makeAttentionRead(q.ptr().data, bh, seqQ, HEAD_DIM);
        if constexpr (IN == WmmaInput::INT8_KV) {
            const auto kIOp = makeInt8KVRead(kd.ptr().data, ks.ptr().data, bh, seqK, HEAD_DIM);
            const auto vIOp = makeInt8KVRead(vd.ptr().data, vs.ptr().data, bh, seqK, HEAD_DIM);
            executeFlashAttentionWmma<HEAD_DIM, BLOCK_KV, NUM_WAVES>(qIOp, kIOp, vIOp, o.ptr().data, bh, seqQ, seqK,
                                                                     causal, stream);
        } else {
            const auto kIOp = makeFp8KVRead(kd.ptr().data, ks.ptr().data, bh, seqK, HEAD_DIM);
            const auto vIOp = makeFp8KVRead(vd.ptr().data, vs.ptr().data, bh, seqK, HEAD_DIM);
            using DPP = FlashAttentionWmmaDPP<float, HEAD_DIM, std::decay_t<decltype(qIOp)>,
                                              std::decay_t<decltype(kIOp)>, std::decay_t<decltype(vIOp)>,
                                              AttentionIdentityEpilogue, BLOCK_KV, NUM_WAVES>;
            static_assert(DPP::FP8_QK, "an fp8 K cache must take the FP8 QK^T path");
            executeFlashAttentionWmma<HEAD_DIM, BLOCK_KV, NUM_WAVES>(qIOp, kIOp, vIOp, o.ptr().data, bh, seqQ, seqK,
                                                                     causal, stream);
        }
        const std::vector<float> got = downloadBuffer(o, stream);
        cpuAttention(dq, dk, dv, ref, bh, seqQ, seqK, HEAD_DIM, scale, causal);
        report(name, maxAbsErr(got, ref), tol);
    }
}

// epilogue Mul(2).then(Add(0.5)) in-register, and executeFlashAttentionAuto
// (which takes the WMMA kernel on gfx12): both against the plain WMMA run
static void testWmmaEpilogueAndAuto() {
    constexpr int HEAD_DIM = 64, BH = 2, SQ = 48, SK = 80;
    std::mt19937 rng(77);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    const size_t nQ = (size_t)BH * SQ * HEAD_DIM, nK = (size_t)BH * SK * HEAD_DIM;
    std::vector<attn_bf16> bq(nQ), bk(nK), bv(nK);
    for (auto& x : bq) x = __float2bfloat16(dist(rng));
    for (auto& x : bk) x = __float2bfloat16(dist(rng));
    for (auto& x : bv) x = __float2bfloat16(dist(rng));
    Stream stream;
    auto q = uploadBuffer(bq, stream);
    auto k = uploadBuffer(bk, stream);
    auto v = uploadBuffer(bv, stream);
    auto o1 = zeroBuffer<float>(nQ, stream);
    auto o2 = zeroBuffer<float>(nQ, stream);
    auto o3 = zeroBuffer<float>(nQ, stream);
    const auto qIOp = makeAttentionRead(static_cast<const attn_bf16*>(q.ptr().data), BH, SQ, HEAD_DIM);
    const auto kIOp = makeAttentionRead(static_cast<const attn_bf16*>(k.ptr().data), BH, SK, HEAD_DIM);
    const auto vIOp = makeAttentionRead(static_cast<const attn_bf16*>(v.ptr().data), BH, SK, HEAD_DIM);
    executeFlashAttentionWmma<HEAD_DIM>(qIOp, kIOp, vIOp, o1.ptr().data, BH, SQ, SK, true, stream);
    const auto epilogue = Mul<float>::build(2.f).then(Add<float>::build(0.5f));
    executeFlashAttentionWmma<HEAD_DIM>(qIOp, kIOp, vIOp, o2.ptr().data, BH, SQ, SK, true, stream, -1.f, epilogue);
    executeFlashAttentionAuto<HEAD_DIM>(qIOp, kIOp, vIOp, o3.ptr().data, BH, SQ, SK, true, stream);
    const std::vector<float> g1 = downloadBuffer(o1, stream);
    const std::vector<float> g2 = downloadBuffer(o2, stream);
    const std::vector<float> g3 = downloadBuffer(o3, stream);
    double epiErr = 0.0, autoErr = 0.0;
    for (size_t i = 0; i < nQ; ++i) {
        epiErr = std::max(epiErr, std::abs((double)g2[i] - ((double)g1[i] * 2.0 + 0.5)));
        autoErr = std::max(autoErr, std::abs((double)g3[i] - (double)g1[i]));
    }
    report("FA-WMMA fused epilogue Mul(2).then(Add(0.5))", epiErr, 1e-6);
    report("FA-WMMA executeFlashAttentionAuto == executeFlashAttentionWmma on gfx12", autoErr, 0.0);
}

int launch_impl() {
    if (!attnDeviceHasWmmaGfx12()) {
        std::cout << "SKIPPED (FlashAttentionWmmaDPP needs a gfx12 GPU)" << std::endl;
        return 0;
    }
    testWmma<64, WmmaInput::BF16>("FA-WMMA bf16 d64 bh2 s128", 2, 128, 128, false, WMMA_BF16_TOL, 1);
    testWmma<64, WmmaInput::BF16>("FA-WMMA bf16 d64 bh2 s128 causal", 2, 128, 128, true, WMMA_BF16_TOL, 2);
    testWmma<64, WmmaInput::BF16>("FA-WMMA bf16 d64 ragged s100/s200", 2, 100, 200, false, WMMA_BF16_TOL, 3);
    testWmma<64, WmmaInput::BF16>("FA-WMMA bf16 d64 ragged s67/s131 causal", 2, 67, 131, true, WMMA_BF16_TOL, 4);
    testWmma<32, WmmaInput::BF16>("FA-WMMA bf16 d32 cross s32->s96", 2, 32, 96, false, WMMA_BF16_TOL, 5);
    testWmma<128, WmmaInput::BF16>("FA-WMMA bf16 d128 bh2 s128 causal", 2, 128, 128, true, WMMA_BF16_TOL, 6);
    testWmma<128, WmmaInput::BF16>("FA-WMMA bf16 d128 bh1 s1024", 1, 1024, 1024, false, WMMA_BF16_TOL, 7);
    testWmma<64, WmmaInput::FP32>("FA-WMMA fp32 prologue d64 s64/s100 causal", 2, 64, 100, true, WMMA_BF16_TOL, 8);
    testWmma<64, WmmaInput::INT8_KV>("FA-WMMA int8-KV d64 s50/s100", 2, 50, 100, false, WMMA_BF16_TOL, 9);
    testWmma<64, WmmaInput::FP8_KV>("FA-WMMA fp8-KV (fp8 QK^T) d64 s64 causal", 2, 64, 64, true, WMMA_FP8_TOL, 10);
    testWmma<64, WmmaInput::FP8_KV>("FA-WMMA fp8-KV (fp8 QK^T) d64 ragged s50/s100", 2, 50, 100, false,
                                    WMMA_FP8_TOL, 11);
    testWmma<128, WmmaInput::FP8_KV>("FA-WMMA fp8-KV (fp8 QK^T) d128 s256", 1, 256, 256, false, WMMA_FP8_TOL, 12);
    // the other tile shapes: BLOCK_KV 64, 4 waves (64 queries per block)
    testWmma<64, WmmaInput::BF16, 64, 4>("FA-WMMA bf16 d64 ragged s67/s131 causal BKV64/W4", 2, 67, 131, true,
                                         WMMA_BF16_TOL, 13);
    testWmma<128, WmmaInput::BF16, 64, 4>("FA-WMMA bf16 d128 s200 BKV64/W4", 1, 200, 200, false, WMMA_BF16_TOL, 14);
    testWmma<64, WmmaInput::FP8_KV, 64, 4>("FA-WMMA fp8-KV (fp8 QK^T) d64 s50/s100 BKV64/W4", 2, 50, 100, true,
                                           WMMA_FP8_TOL, 15);
    testWmmaEpilogueAndAuto();
    if (failures == 0) { return 0; }
    std::cout << failures << " flash attention wmma test(s) FAILED" << std::endl;
    return -1;
}

} // namespace fk

int launch() {
    return fk::launch_impl();
}
