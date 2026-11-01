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

#define __ONLY_GPU__ // Generated as _cu and _hip, not as _cpp: the attention DPPs are GPU-only
#include <tests/main.h>

#include <fused_kernel/algorithms/attention/flash_attention.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <utests/algorithm/attention/attention_test_common.h>

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

namespace fk {

template <int HEAD_DIM>
static void testDense(const char* name, const int bh, const int seqQ, const int seqK,
                      const bool causal, const double tol, const unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    const size_t nQ = (size_t)bh * seqQ * HEAD_DIM, nK = (size_t)bh * seqK * HEAD_DIM;
    std::vector<float> hq(nQ), hk(nK), hv(nK);
    std::vector<double> dq(nQ), dk(nK), dv(nK), dref;
    for (size_t i = 0; i < nQ; ++i) { hq[i] = dist(rng); dq[i] = hq[i]; }
    for (size_t i = 0; i < nK; ++i) { hk[i] = dist(rng); dk[i] = hk[i]; }
    for (size_t i = 0; i < nK; ++i) { hv[i] = dist(rng); dv[i] = hv[i]; }
    cpuAttention(dq, dk, dv, dref, bh, seqQ, seqK, HEAD_DIM,
                 1.0 / std::sqrt((double)HEAD_DIM), causal);

    Stream stream;
    auto q = uploadBuffer(hq, stream);
    auto k = uploadBuffer(hk, stream);
    auto v = uploadBuffer(hv, stream);
    auto o = zeroBuffer<float>(nQ, stream);
    executeFlashAttention<float, HEAD_DIM>(q.ptr().data, k.ptr().data, v.ptr().data, o.ptr().data, bh, seqQ,
                                           seqK, causal, stream);
    const std::vector<float> got = downloadBuffer(o, stream);
    report(name, maxAbsErr(got, dref), tol);
}

template <int HEAD_DIM>
static void testInt8KV(const char* name, const int bh, const int seqQ, const int seqK,
                       const bool causal, const double tol, const unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    const size_t nQ = (size_t)bh * seqQ * HEAD_DIM, nK = (size_t)bh * seqK * HEAD_DIM;
    const size_t nTok = (size_t)bh * seqK;
    std::vector<float> hq(nQ), hk(nK), hv(nK);
    for (size_t i = 0; i < nQ; ++i) hq[i] = dist(rng);
    for (size_t i = 0; i < nK; ++i) hk[i] = dist(rng);
    for (size_t i = 0; i < nK; ++i) hv[i] = dist(rng);

    // host-side compression (per-token int8)
    std::vector<int8_t> k8(nK), v8(nK);
    std::vector<float> kSc(nTok), vSc(nTok);
    quantizeKVCacheHost(hk.data(), k8.data(), kSc.data(), (int)nTok, HEAD_DIM);
    quantizeKVCacheHost(hv.data(), v8.data(), vSc.data(), (int)nTok, HEAD_DIM);

    // the oracle sees the DEQUANTIZED values: tests kernel exactness given
    // the compressed cache (quantization error itself is bounded separately)
    std::vector<double> dq(nQ), dkD(nK), dvD(nK), dref;
    for (size_t i = 0; i < nQ; ++i) dq[i] = hq[i];
    for (size_t i = 0; i < nK; ++i) dkD[i] = (double)k8[i] * kSc[i / HEAD_DIM];
    for (size_t i = 0; i < nK; ++i) dvD[i] = (double)v8[i] * vSc[i / HEAD_DIM];
    cpuAttention(dq, dkD, dvD, dref, bh, seqQ, seqK, HEAD_DIM,
                 1.0 / std::sqrt((double)HEAD_DIM), causal);

    Stream stream;
    auto q = uploadBuffer(hq, stream);
    auto dk8 = uploadBuffer(k8, stream);
    auto dv8 = uploadBuffer(v8, stream);
    auto dkS = uploadBuffer(kSc, stream);
    auto dvS = uploadBuffer(vSc, stream);
    auto o = zeroBuffer<float>(nQ, stream);
    executeFlashAttention<float, HEAD_DIM, KVLayout::INT8_PER_TOKEN>(
        q.ptr().data, dk8.ptr().data, dv8.ptr().data, o.ptr().data, bh, seqQ, seqK, causal, stream,
        dkS.ptr().data, dvS.ptr().data);
    const std::vector<float> got = downloadBuffer(o, stream);
    report(name, maxAbsErr(got, dref), tol);
}

#ifdef FK_HAS_FP8
// The oracle decodes with attnFp8ToF32 on the host, so first make sure it is
// OCP e4m3 and not another 8-bit float (FNUZ, e5m2): a wrong format would
// otherwise fool the kernel and the oracle alike.
static void testFp8Format() {
    if (attnFp8IsOcpE4M3()) {
        std::cout << "Running test attn_fp8_e4m3 decodes OCP e4m3 (0x7E=448, 0x38=1, 0x80=-0): Success!!"
                  << std::endl;
    } else {
        std::cout << "FAIL attn_fp8_e4m3 is not OCP e4m3: 0x7E=" << attnFp8ToF32(0x7E)
                  << " 0x38=" << attnFp8ToF32(0x38) << " 0x80=" << attnFp8ToF32(static_cast<int8_t>(0x80))
                  << std::endl;
        ++failures;
    }
}

template <int HEAD_DIM>
static void testFp8KV(const char* name, const int bh, const int seqQ, const int seqK,
                      const bool causal, const double tol, const unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    const size_t nQ = (size_t)bh * seqQ * HEAD_DIM, nK = (size_t)bh * seqK * HEAD_DIM;
    const size_t nTok = (size_t)bh * seqK;
    std::vector<float> hq(nQ), hk(nK), hv(nK);
    for (size_t i = 0; i < nQ; ++i) hq[i] = dist(rng);
    for (size_t i = 0; i < nK; ++i) hk[i] = dist(rng);
    for (size_t i = 0; i < nK; ++i) hv[i] = dist(rng);

    // host-side compression (per-token fp8 e4m3)
    std::vector<int8_t> k8(nK), v8(nK);
    std::vector<float> kSc(nTok), vSc(nTok);
    quantizeKVCacheFp8Host(hk.data(), k8.data(), kSc.data(), (int)nTok, HEAD_DIM);
    quantizeKVCacheFp8Host(hv.data(), v8.data(), vSc.data(), (int)nTok, HEAD_DIM);

    // the oracle sees the DEQUANTIZED values, as in testInt8KV
    std::vector<double> dq(nQ), dkD(nK), dvD(nK), dref;
    for (size_t i = 0; i < nQ; ++i) dq[i] = hq[i];
    for (size_t i = 0; i < nK; ++i) dkD[i] = (double)attnFp8ToF32(k8[i]) * kSc[i / HEAD_DIM];
    for (size_t i = 0; i < nK; ++i) dvD[i] = (double)attnFp8ToF32(v8[i]) * vSc[i / HEAD_DIM];
    cpuAttention(dq, dkD, dvD, dref, bh, seqQ, seqK, HEAD_DIM,
                 1.0 / std::sqrt((double)HEAD_DIM), causal);

    Stream stream;
    auto q = uploadBuffer(hq, stream);
    auto dk8 = uploadBuffer(k8, stream);
    auto dv8 = uploadBuffer(v8, stream);
    auto dkS = uploadBuffer(kSc, stream);
    auto dvS = uploadBuffer(vSc, stream);
    auto o = zeroBuffer<float>(nQ, stream);
    const auto qIOp = makeAttentionRead(q.ptr().data, bh, seqQ, HEAD_DIM);
    const auto kIOp = makeFp8KVRead(dk8.ptr().data, dkS.ptr().data, bh, seqK, HEAD_DIM);
    const auto vIOp = makeFp8KVRead(dv8.ptr().data, dvS.ptr().data, bh, seqK, HEAD_DIM);
    executeFlashAttention<HEAD_DIM>(qIOp, kIOp, vIOp, o.ptr().data, bh, seqQ, seqK, causal, stream);
    const std::vector<float> got = downloadBuffer(o, stream);
    report(name, maxAbsErr(got, dref), tol);
}
#endif // FK_HAS_FP8

static void testFusedEpilogue() {
    // attention output | Mul(2) | Add(0.5) fused in-register: compare against
    // dense run + host-applied epilogue (proves the chain ran inside).
    constexpr int HEAD_DIM = 32, BH = 2, SQ = 16, SK = 16;
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    const size_t n = (size_t)BH * SQ * HEAD_DIM;
    std::vector<float> hq(n), hk(n), hv(n);
    for (auto* vec : { &hq, &hk, &hv })
        for (auto& x : *vec) x = dist(rng);

    Stream stream;
    auto q = uploadBuffer(hq, stream);
    auto k = uploadBuffer(hk, stream);
    auto v = uploadBuffer(hv, stream);
    auto o1 = zeroBuffer<float>(n, stream);
    auto o2 = zeroBuffer<float>(n, stream);
    executeFlashAttention<float, HEAD_DIM>(q.ptr().data, k.ptr().data, v.ptr().data, o1.ptr().data, BH, SQ, SK,
                                           false, stream);
    const auto epilogue = Mul<float>::build(2.f).then(Add<float>::build(0.5f));
    executeFlashAttention<float, HEAD_DIM, KVLayout::DENSE, 32, 4>(
        q.ptr().data, k.ptr().data, v.ptr().data, o2.ptr().data, BH, SQ, SK, false, stream, nullptr, nullptr,
        -1.f, epilogue);
    const std::vector<float> g1 = downloadBuffer(o1, stream);
    const std::vector<float> g2 = downloadBuffer(o2, stream);

    double maxErr = 0.0;
    for (size_t i = 0; i < n; ++i)
        maxErr = std::max(maxErr, std::abs((double)g2[i] - ((double)g1[i] * 2.0 + 0.5)));
    report("FlashAttention fused epilogue Mul(2).then(Add(0.5))", maxErr, 1e-6);
}

static void testFusedPrologue() {
    /* PROLOGUE = a Read IOp (possibly fused with .then chains); the DPP
       reads every element through it. Verifiable algebra:
       Q prologue read.then(Mul(2)): compare against oracle on 2*Q.
       V prologue read.then(Mul(3)).then(Add(1)): out = 3*(sum p_j v_j) + 1
         since sum p_j = 1 - compare against 3*oracle + 1. */
    constexpr int HEAD_DIM = 32, BH = 2, SQ = 24, SK = 48;
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    const size_t nQ = (size_t)BH * SQ * HEAD_DIM, nK = (size_t)BH * SK * HEAD_DIM;
    std::vector<float> hq(nQ), hk(nK), hv(nK);
    for (auto& x : hq) x = dist(rng);
    for (auto& x : hk) x = dist(rng);
    for (auto& x : hv) x = dist(rng);

    Stream stream;
    auto qBuf = uploadBuffer(hq, stream);
    auto kBuf = uploadBuffer(hk, stream);
    auto vBuf = uploadBuffer(hv, stream);
    auto oBuf = zeroBuffer<float>(nQ, stream);
    const float* q = qBuf.ptr().data;
    const float* k = kBuf.ptr().data;
    const float* v = vBuf.ptr().data;
    float* o = oBuf.ptr().data;
    const double scl = 1.0 / std::sqrt((double)HEAD_DIM);
    std::vector<double> dq(nQ), dk(nK), dv(nK), ref;
    for (size_t i = 0; i < nK; ++i) dk[i] = hk[i];
    for (size_t i = 0; i < nK; ++i) dv[i] = hv[i];

    // --- Q prologue: Read IOp fused with Mul(2) ---
    {
        const auto qIOp = makeAttentionRead(q, BH, SQ, HEAD_DIM)
                              .then(Mul<float>::build(2.f));
        const auto kIOp = makeAttentionRead(k, BH, SK, HEAD_DIM);
        const auto vIOp = makeAttentionRead(v, BH, SK, HEAD_DIM);
        executeFlashAttention<HEAD_DIM>(qIOp, kIOp, vIOp, o, BH, SQ, SK,
                                        false, stream);
        const std::vector<float> got = downloadBuffer(oBuf, stream);
        for (size_t i = 0; i < nQ; ++i) dq[i] = 2.0 * hq[i];   // host-applied
        cpuAttention(dq, dk, dv, ref, BH, SQ, SK, HEAD_DIM, scl, false);
        double maxErr = 0.0;
        for (size_t i = 0; i < nQ; ++i)
            maxErr = std::max(maxErr, std::abs((double)got[i] - ref[i]));
        report("FlashAttention Q-prologue ReadIOp.then(Mul(2))", maxErr, 5e-6);
    }

    // --- V prologue: Read IOp fused with Mul(3).then(Add(1)) => 3*out + 1 ---
    {
        const auto qIOp = makeAttentionRead(q, BH, SQ, HEAD_DIM);
        const auto kIOp = makeAttentionRead(k, BH, SK, HEAD_DIM);
        const auto vIOp = makeAttentionRead(v, BH, SK, HEAD_DIM)
                              .then(Mul<float>::build(3.f))
                              .then(Add<float>::build(1.f));
        executeFlashAttention<HEAD_DIM>(qIOp, kIOp, vIOp, o, BH, SQ, SK,
                                        false, stream);
        const std::vector<float> got = downloadBuffer(oBuf, stream);
        for (size_t i = 0; i < nQ; ++i) dq[i] = hq[i];
        cpuAttention(dq, dk, dv, ref, BH, SQ, SK, HEAD_DIM, scl, false);
        double maxErr = 0.0;
        for (size_t i = 0; i < nQ; ++i)
            maxErr = std::max(maxErr, std::abs((double)got[i] - (3.0 * ref[i] + 1.0)));
        report("FlashAttention V-prologue ReadIOp.then(Mul(3)).then(Add(1))", maxErr, 5e-6);
    }

}

int launch_impl() {
    testDense<64>("FA dense d64 b2 s64 causal", 2, 64, 64, true, 5e-6, 1);
    testDense<64>("FA dense d64 ragged s67/s131", 2, 67, 131, false, 5e-6, 2);
    testDense<32>("FA dense d32 cross s32->s96", 2, 32, 96, false, 5e-6, 3);
    testDense<128>("FA dense d128 b1 s64", 1, 64, 64, false, 5e-6, 4);
    testInt8KV<64>("FA int8-KV d64 b2 s64 causal", 2, 64, 64, true, 5e-6, 5);
    testInt8KV<64>("FA int8-KV d64 ragged s50/s100", 2, 50, 100, false, 5e-6, 6);
#ifdef FK_HAS_FP8
    testFp8Format();
    testFp8KV<64>("FA fp8-KV d64 b2 s64 causal", 2, 64, 64, true, 5e-6, 7);
    testFp8KV<64>("FA fp8-KV d64 ragged s50/s100", 2, 50, 100, false, 5e-6, 8);
    testFp8KV<128>("FA fp8-KV d128 b1 s64", 1, 64, 64, false, 5e-6, 9);
#endif
    testFusedEpilogue();
    testFusedPrologue();
    if (failures == 0) { return 0; }
    std::cout << failures << " attention test(s) FAILED" << std::endl;
    return -1;
}

} // namespace fk

int launch() {
    return fk::launch_impl();
}
