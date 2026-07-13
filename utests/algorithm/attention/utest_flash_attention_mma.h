/* Copyright 2026 the Fused Kernel Library authors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#define __ONLY_CU__ // This file is only generated and compiled with nvcc, not with the host compiler
#include <tests/main.h>

#include <fused_kernel/algorithms/attention/flash_attention_mma.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using namespace fk;

/* The mma.sync kernel stages Q/K/V (post-prologue) and the online-softmax P
   tiles as bf16 fragments, so the fp64 oracle gets the bf16-ROUNDED inputs
   and the tolerances account for the P->bf16 quantization (~2^-8 relative)
   that the oracle cannot model. */
static float bf16Round(const float x) {
    return __bfloat162float(__float2bfloat16(x));
}

#ifdef FK_HAS_FP8
static float fp8DequantHost(const int8_t raw, const float scale) {
    __nv_fp8_e4m3 f8;
    f8.__x = static_cast<__nv_fp8_storage_t>(raw);
    return static_cast<float>(f8) * scale;
}
#endif

// double-precision CPU oracle: O = softmax(scale * Q K^T [causal]) V
static void cpuAttention(const std::vector<double>& q, const std::vector<double>& k,
                         const std::vector<double>& v, std::vector<double>& o,
                         const int bh, const int seqQ, const int seqK, const int d,
                         const double scale, const bool causal) {
    o.assign((size_t)bh * seqQ * d, 0.0);
    std::vector<double> s(seqK);
    for (int b = 0; b < bh; ++b) {
        const double* Q = q.data() + (size_t)b * seqQ * d;
        const double* K = k.data() + (size_t)b * seqK * d;
        const double* V = v.data() + (size_t)b * seqK * d;
        double* O = o.data() + (size_t)b * seqQ * d;
        for (int i = 0; i < seqQ; ++i) {
            const int kEnd = causal ? std::min(seqK, i + 1) : seqK;
            double m = -1e300;
            for (int j = 0; j < kEnd; ++j) {
                double dot = 0.0;
                for (int c = 0; c < d; ++c) dot += Q[(size_t)i*d+c] * K[(size_t)j*d+c];
                s[j] = dot * scale;
                m = std::max(m, s[j]);
            }
            double l = 0.0;
            for (int j = 0; j < kEnd; ++j) { s[j] = std::exp(s[j] - m); l += s[j]; }
            for (int j = 0; j < kEnd; ++j) {
                const double pj = s[j] / l;
                for (int c = 0; c < d; ++c) O[(size_t)i*d+c] += pj * V[(size_t)j*d+c];
            }
        }
    }
}

static int failures = 0;

static void report(const char* name, const double maxErr, const double tol) {
    if (maxErr > tol) {
        std::cout << "FAIL " << name << ": maxErr=" << maxErr << " tol=" << tol << std::endl;
        ++failures;
    } else {
        std::cout << "Running test " << name << ": Success!! (maxErr=" << maxErr << ")" << std::endl;
    }
}

template <int HEAD_DIM>
static void testDense(const char* name, const int bh, const int seqQ, const int seqK,
                      const bool causal, const double tol, const unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    const size_t nQ = (size_t)bh * seqQ * HEAD_DIM, nK = (size_t)bh * seqK * HEAD_DIM;
    std::vector<float> hq(nQ), hk(nK), hv(nK);
    std::vector<double> dq(nQ), dk(nK), dv(nK), dref;
    for (size_t i = 0; i < nQ; ++i) { hq[i] = bf16Round(dist(rng)); dq[i] = hq[i]; }
    for (size_t i = 0; i < nK; ++i) { hk[i] = bf16Round(dist(rng)); dk[i] = hk[i]; }
    for (size_t i = 0; i < nK; ++i) { hv[i] = bf16Round(dist(rng)); dv[i] = hv[i]; }
    cpuAttention(dq, dk, dv, dref, bh, seqQ, seqK, HEAD_DIM,
                 1.0 / std::sqrt((double)HEAD_DIM), causal);

    float *q, *k, *v, *o;
    gpuErrchk(cudaMalloc(&q, nQ * sizeof(float)));
    gpuErrchk(cudaMalloc(&k, nK * sizeof(float)));
    gpuErrchk(cudaMalloc(&v, nK * sizeof(float)));
    gpuErrchk(cudaMalloc(&o, nQ * sizeof(float)));
    gpuErrchk(cudaMemcpy(q, hq.data(), nQ * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(k, hk.data(), nK * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(v, hv.data(), nK * sizeof(float), cudaMemcpyHostToDevice));

    Stream stream;
    const auto qIOp = makeAttentionRead(q, bh, seqQ, HEAD_DIM);
    const auto kIOp = makeAttentionRead(k, bh, seqK, HEAD_DIM);
    const auto vIOp = makeAttentionRead(v, bh, seqK, HEAD_DIM);
    const auto oIOp = makeAttentionWrite(o, bh, seqQ, HEAD_DIM);
    executeFlashAttentionMma<HEAD_DIM>(qIOp, kIOp, vIOp, oIOp, bh, seqQ, seqK,
                                       causal, stream);
    stream.sync();

    std::vector<float> got(nQ);
    gpuErrchk(cudaMemcpy(got.data(), o, nQ * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(o);

    double maxErr = 0.0;
    for (size_t i = 0; i < nQ; ++i) maxErr = std::max(maxErr, std::abs((double)got[i] - dref[i]));
    report(name, maxErr, tol);
}

// RAW bf16 Read IOps.
template <int HEAD_DIM>
static void testDenseBf16Raw(const char* name, const int bh, const int seqQ,
                             const int seqK, const bool causal, const double tol,
                             const unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    const size_t nQ = (size_t)bh * seqQ * HEAD_DIM, nK = (size_t)bh * seqK * HEAD_DIM;
    std::vector<__nv_bfloat16> hq(nQ), hk(nK), hv(nK);
    std::vector<double> dq(nQ), dk(nK), dv(nK), dref;
    for (size_t i = 0; i < nQ; ++i) { hq[i] = __float2bfloat16(dist(rng)); dq[i] = __bfloat162float(hq[i]); }
    for (size_t i = 0; i < nK; ++i) { hk[i] = __float2bfloat16(dist(rng)); dk[i] = __bfloat162float(hk[i]); }
    for (size_t i = 0; i < nK; ++i) { hv[i] = __float2bfloat16(dist(rng)); dv[i] = __bfloat162float(hv[i]); }
    cpuAttention(dq, dk, dv, dref, bh, seqQ, seqK, HEAD_DIM,
                 1.0 / std::sqrt((double)HEAD_DIM), causal);

    __nv_bfloat16 *q, *k, *v;
    float* o;
    gpuErrchk(cudaMalloc(&q, nQ * sizeof(__nv_bfloat16)));
    gpuErrchk(cudaMalloc(&k, nK * sizeof(__nv_bfloat16)));
    gpuErrchk(cudaMalloc(&v, nK * sizeof(__nv_bfloat16)));
    gpuErrchk(cudaMalloc(&o, nQ * sizeof(float)));
    gpuErrchk(cudaMemcpy(q, hq.data(), nQ * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(k, hk.data(), nK * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(v, hv.data(), nK * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    Stream stream;
    const auto qIOp = makeAttentionRead(q, bh, seqQ, HEAD_DIM);
    const auto kIOp = makeAttentionRead(k, bh, seqK, HEAD_DIM);
    const auto vIOp = makeAttentionRead(v, bh, seqK, HEAD_DIM);
    const auto oIOp = makeAttentionWrite(o, bh, seqQ, HEAD_DIM);
    executeFlashAttentionMma<HEAD_DIM>(qIOp, kIOp, vIOp, oIOp, bh, seqQ, seqK,
                                       causal, stream);
    stream.sync();

    std::vector<float> got(nQ);
    gpuErrchk(cudaMemcpy(got.data(), o, nQ * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(o);

    double maxErr = 0.0;
    for (size_t i = 0; i < nQ; ++i) maxErr = std::max(maxErr, std::abs((double)got[i] - dref[i]));
    report(name, maxErr, tol);
}

template <int HEAD_DIM>
static void testInt8KV(const char* name, const int bh, const int seqQ, const int seqK,
                       const bool causal, const double tol, const unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    const size_t nQ = (size_t)bh * seqQ * HEAD_DIM, nK = (size_t)bh * seqK * HEAD_DIM;
    const size_t nTok = (size_t)bh * seqK;
    std::vector<float> hq(nQ), hk(nK), hv(nK);
    std::vector<double> dq(nQ), dk(nK), dv(nK), dref;
    for (size_t i = 0; i < nQ; ++i) { hq[i] = bf16Round(dist(rng)); dq[i] = hq[i]; }
    for (size_t i = 0; i < nK; ++i) hk[i] = dist(rng);
    for (size_t i = 0; i < nK; ++i) hv[i] = dist(rng);

    std::vector<int8_t> k8(nK), v8(nK);
    std::vector<float> kSc(nTok), vSc(nTok);
    quantizeKVCacheHost(hk.data(), k8.data(), kSc.data(), (int)nTok, HEAD_DIM);
    quantizeKVCacheHost(hv.data(), v8.data(), vSc.data(), (int)nTok, HEAD_DIM);

    for (size_t i = 0; i < nK; ++i) dk[i] = bf16Round((float)k8[i] * kSc[i / HEAD_DIM]);
    for (size_t i = 0; i < nK; ++i) dv[i] = bf16Round((float)v8[i] * vSc[i / HEAD_DIM]);
    cpuAttention(dq, dk, dv, dref, bh, seqQ, seqK, HEAD_DIM,
                 1.0 / std::sqrt((double)HEAD_DIM), causal);

    float *q, *o, *dkS, *dvS;
    int8_t *dk8, *dv8;
    gpuErrchk(cudaMalloc(&q, nQ * sizeof(float)));
    gpuErrchk(cudaMalloc(&o, nQ * sizeof(float)));
    gpuErrchk(cudaMalloc(&dk8, nK));
    gpuErrchk(cudaMalloc(&dv8, nK));
    gpuErrchk(cudaMalloc(&dkS, nTok * sizeof(float)));
    gpuErrchk(cudaMalloc(&dvS, nTok * sizeof(float)));
    gpuErrchk(cudaMemcpy(q, hq.data(), nQ * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dk8, k8.data(), nK, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dv8, v8.data(), nK, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dkS, kSc.data(), nTok * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dvS, vSc.data(), nTok * sizeof(float), cudaMemcpyHostToDevice));

    Stream stream;
    const auto qIOp = makeAttentionRead(q, bh, seqQ, HEAD_DIM);
    const auto kIOp = makeInt8KVRead(dk8, dkS, bh, seqK, HEAD_DIM);
    const auto vIOp = makeInt8KVRead(dv8, dvS, bh, seqK, HEAD_DIM);
    const auto oIOp = makeAttentionWrite(o, bh, seqQ, HEAD_DIM);
    executeFlashAttentionMma<HEAD_DIM>(qIOp, kIOp, vIOp, oIOp, bh, seqQ, seqK,
                                       causal, stream);
    stream.sync();

    std::vector<float> got(nQ);
    gpuErrchk(cudaMemcpy(got.data(), o, nQ * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(q); cudaFree(o); cudaFree(dk8); cudaFree(dv8); cudaFree(dkS); cudaFree(dvS);

    double maxErr = 0.0;
    for (size_t i = 0; i < nQ; ++i)
        maxErr = std::max(maxErr, std::abs((double)got[i] - dref[i]));
    report(name, maxErr, tol);
}

#ifdef FK_HAS_FP8
template <int HEAD_DIM>
static void testFp8KV(const char* name, const int bh, const int seqQ, const int seqK,
                      const bool causal, const double tol, const unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    const size_t nQ = (size_t)bh * seqQ * HEAD_DIM, nK = (size_t)bh * seqK * HEAD_DIM;
    const size_t nTok = (size_t)bh * seqK;
    std::vector<float> hq(nQ), hk(nK), hv(nK);
    std::vector<double> dq(nQ), dk(nK), dv(nK), dref;
    for (size_t i = 0; i < nQ; ++i) { hq[i] = bf16Round(dist(rng)); dq[i] = hq[i]; }
    for (size_t i = 0; i < nK; ++i) hk[i] = dist(rng);
    for (size_t i = 0; i < nK; ++i) hv[i] = dist(rng);

    std::vector<int8_t> k8(nK), v8(nK);
    std::vector<float> kSc(nTok), vSc(nTok);
    quantizeKVCacheFp8Host(hk.data(), k8.data(), kSc.data(), (int)nTok, HEAD_DIM);
    quantizeKVCacheFp8Host(hv.data(), v8.data(), vSc.data(), (int)nTok, HEAD_DIM);

    for (size_t i = 0; i < nK; ++i) dk[i] = bf16Round(fp8DequantHost(k8[i], kSc[i / HEAD_DIM]));
    for (size_t i = 0; i < nK; ++i) dv[i] = bf16Round(fp8DequantHost(v8[i], vSc[i / HEAD_DIM]));
    cpuAttention(dq, dk, dv, dref, bh, seqQ, seqK, HEAD_DIM,
                 1.0 / std::sqrt((double)HEAD_DIM), causal);

    float *q, *o, *dkS, *dvS;
    int8_t *dk8, *dv8;
    gpuErrchk(cudaMalloc(&q, nQ * sizeof(float)));
    gpuErrchk(cudaMalloc(&o, nQ * sizeof(float)));
    gpuErrchk(cudaMalloc(&dk8, nK));
    gpuErrchk(cudaMalloc(&dv8, nK));
    gpuErrchk(cudaMalloc(&dkS, nTok * sizeof(float)));
    gpuErrchk(cudaMalloc(&dvS, nTok * sizeof(float)));
    gpuErrchk(cudaMemcpy(q, hq.data(), nQ * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dk8, k8.data(), nK, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dv8, v8.data(), nK, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dkS, kSc.data(), nTok * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dvS, vSc.data(), nTok * sizeof(float), cudaMemcpyHostToDevice));

    Stream stream;
    const auto qIOp = makeAttentionRead(q, bh, seqQ, HEAD_DIM);
    const auto kIOp = makeFp8KVRead(dk8, dkS, bh, seqK, HEAD_DIM);
    const auto vIOp = makeFp8KVRead(dv8, dvS, bh, seqK, HEAD_DIM);
    const auto oIOp = makeAttentionWrite(o, bh, seqQ, HEAD_DIM);
    executeFlashAttentionMma<HEAD_DIM>(qIOp, kIOp, vIOp, oIOp, bh, seqQ, seqK,
                                       causal, stream);
    stream.sync();

    std::vector<float> got(nQ);
    gpuErrchk(cudaMemcpy(got.data(), o, nQ * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(q); cudaFree(o); cudaFree(dk8); cudaFree(dv8); cudaFree(dkS); cudaFree(dvS);

    double maxErr = 0.0;
    for (size_t i = 0; i < nQ; ++i)
        maxErr = std::max(maxErr, std::abs((double)got[i] - dref[i]));
    report(name, maxErr, tol);
}
#endif

static void testFusedEpilogue() {
    // attention output | Mul(2) | Add(0.5) fused in-register: compare against
    // a plain run + host-applied epilogue (proves the chain ran inside).
    constexpr int HEAD_DIM = 32, BH = 2, SQ = 64, SK = 64;
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    const size_t n = (size_t)BH * SQ * HEAD_DIM;
    std::vector<float> hq(n), hk(n), hv(n);
    for (auto* vec : { &hq, &hk, &hv })
        for (auto& x : *vec) x = bf16Round(dist(rng));

    float *q, *k, *v, *o1, *o2;
    gpuErrchk(cudaMalloc(&q, n * sizeof(float)));
    gpuErrchk(cudaMalloc(&k, n * sizeof(float)));
    gpuErrchk(cudaMalloc(&v, n * sizeof(float)));
    gpuErrchk(cudaMalloc(&o1, n * sizeof(float)));
    gpuErrchk(cudaMalloc(&o2, n * sizeof(float)));
    gpuErrchk(cudaMemcpy(q, hq.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(k, hk.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(v, hv.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    Stream stream;
    const auto qIOp = makeAttentionRead(q, BH, SQ, HEAD_DIM);
    const auto kIOp = makeAttentionRead(k, BH, SK, HEAD_DIM);
    const auto vIOp = makeAttentionRead(v, BH, SK, HEAD_DIM);
    const auto o1IOp = makeAttentionWrite(o1, BH, SQ, HEAD_DIM);
    executeFlashAttentionMma<HEAD_DIM>(qIOp, kIOp, vIOp, o1IOp, BH, SQ, SK,
                                       false, stream);
    const auto epilogue = Mul<float>::build(2.f).then(Add<float>::build(0.5f));
    const auto o2IOp = makeAttentionOutput(o2, BH, SQ, HEAD_DIM, epilogue);
    executeFlashAttentionMma<HEAD_DIM>(qIOp, kIOp, vIOp, o2IOp, BH, SQ, SK,
                                       false, stream);
    stream.sync();

    std::vector<float> g1(n), g2(n);
    gpuErrchk(cudaMemcpy(g1.data(), o1, n * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(g2.data(), o2, n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(o1); cudaFree(o2);

    double maxErr = 0.0;
    for (size_t i = 0; i < n; ++i)
        maxErr = std::max(maxErr, std::abs((double)g2[i] - ((double)g1[i] * 2.0 + 0.5)));
    report("FA-mma fused epilogue Mul(2).then(Add(0.5))", maxErr, 1e-6);
}

static void testFusedPrologue() {
    /* Q prologue read.then(Mul(2)): compare against the oracle on 2*Q.
       V prologue read.then(Mul(3)).then(Add(1)): out = 3*attn + 1 since
       sum p_j = 1 — compare against 3*oracle + 1. */
    constexpr int HEAD_DIM = 32, BH = 2, SQ = 96, SK = 160;
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    const size_t nQ = (size_t)BH * SQ * HEAD_DIM, nK = (size_t)BH * SK * HEAD_DIM;
    std::vector<float> hq(nQ), hk(nK), hv(nK);
    for (auto& x : hq) x = bf16Round(dist(rng));
    for (auto& x : hk) x = bf16Round(dist(rng));
    for (auto& x : hv) x = bf16Round(dist(rng));

    float *q, *k, *v, *o;
    gpuErrchk(cudaMalloc(&q, nQ * sizeof(float)));
    gpuErrchk(cudaMalloc(&k, nK * sizeof(float)));
    gpuErrchk(cudaMalloc(&v, nK * sizeof(float)));
    gpuErrchk(cudaMalloc(&o, nQ * sizeof(float)));
    gpuErrchk(cudaMemcpy(q, hq.data(), nQ * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(k, hk.data(), nK * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(v, hv.data(), nK * sizeof(float), cudaMemcpyHostToDevice));

    Stream stream;
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
        const auto oIOp = makeAttentionWrite(o, BH, SQ, HEAD_DIM);
        executeFlashAttentionMma<HEAD_DIM>(qIOp, kIOp, vIOp, oIOp, BH, SQ, SK,
                                           false, stream);
        stream.sync();
        std::vector<float> got(nQ);
        gpuErrchk(cudaMemcpy(got.data(), o, nQ * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < nQ; ++i) dq[i] = 2.0 * hq[i];   // host-applied
        cpuAttention(dq, dk, dv, ref, BH, SQ, SK, HEAD_DIM, scl, false);
        double maxErr = 0.0;
        for (size_t i = 0; i < nQ; ++i)
            maxErr = std::max(maxErr, std::abs((double)got[i] - ref[i]));
        report("FA-mma Q-prologue ReadIOp.then(Mul(2))", maxErr, 2.5e-2);
    }

    // --- V prologue: Read IOp fused with Mul(3).then(Add(1)) => 3*out + 1 ---
    {
        const auto qIOp = makeAttentionRead(q, BH, SQ, HEAD_DIM);
        const auto kIOp = makeAttentionRead(k, BH, SK, HEAD_DIM);
        const auto vIOp = makeAttentionRead(v, BH, SK, HEAD_DIM)
                              .then(Mul<float>::build(3.f))
                              .then(Add<float>::build(1.f));
        const auto oIOp = makeAttentionWrite(o, BH, SQ, HEAD_DIM);
        executeFlashAttentionMma<HEAD_DIM>(qIOp, kIOp, vIOp, oIOp, BH, SQ, SK,
                                           false, stream);
        stream.sync();
        std::vector<float> got(nQ);
        gpuErrchk(cudaMemcpy(got.data(), o, nQ * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < nQ; ++i) dq[i] = hq[i];
        cpuAttention(dq, dk, dv, ref, BH, SQ, SK, HEAD_DIM, scl, false);
        double maxErr = 0.0;
        for (size_t i = 0; i < nQ; ++i)
            maxErr = std::max(maxErr, std::abs((double)got[i] - (3.0 * ref[i] + 1.0)));
        report("FA-mma V-prologue ReadIOp.then(Mul(3)).then(Add(1))", maxErr, 5e-2);
    }

    cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(o);
}

/* FUSED READ IOps + FUSED WRITE IOp in the same run (see
   utest_flash_attention.h): out = 2*(3*attn(2Q,K,V) + 1) + 0.5
   = 6*oracle(2Q,K,V) + 2.5. Big shapes span many CUDA blocks; the small-grid
   long-KV shape triggers the SPLIT-KV path, exercising the fused write in
   the combine kernel. */
template <int HEAD_DIM>
static void testFusedReadWrite(const char* name, const int bh, const int seqQ,
                               const int seqK, const bool causal,
                               const double tol, const unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    const size_t nQ = (size_t)bh * seqQ * HEAD_DIM, nK = (size_t)bh * seqK * HEAD_DIM;
    std::vector<float> hq(nQ), hk(nK), hv(nK);
    std::vector<double> dq(nQ), dk(nK), dv(nK), ref;
    for (size_t i = 0; i < nQ; ++i) { hq[i] = bf16Round(dist(rng)); dq[i] = 2.0 * hq[i]; }
    for (size_t i = 0; i < nK; ++i) { hk[i] = bf16Round(dist(rng)); dk[i] = hk[i]; }
    for (size_t i = 0; i < nK; ++i) { hv[i] = bf16Round(dist(rng)); dv[i] = hv[i]; }
    cpuAttention(dq, dk, dv, ref, bh, seqQ, seqK, HEAD_DIM,
                 1.0 / std::sqrt((double)HEAD_DIM), causal);

    float *q, *k, *v, *o;
    gpuErrchk(cudaMalloc(&q, nQ * sizeof(float)));
    gpuErrchk(cudaMalloc(&k, nK * sizeof(float)));
    gpuErrchk(cudaMalloc(&v, nK * sizeof(float)));
    gpuErrchk(cudaMalloc(&o, nQ * sizeof(float)));
    gpuErrchk(cudaMemcpy(q, hq.data(), nQ * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(k, hk.data(), nK * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(v, hv.data(), nK * sizeof(float), cudaMemcpyHostToDevice));

    Stream stream;
    const auto qIOp = makeAttentionRead(q, bh, seqQ, HEAD_DIM)
                          .then(Mul<float>::build(2.f));
    const auto kIOp = makeAttentionRead(k, bh, seqK, HEAD_DIM);
    const auto vIOp = makeAttentionRead(v, bh, seqK, HEAD_DIM)
                          .then(Mul<float>::build(3.f))
                          .then(Add<float>::build(1.f));
    const auto oIOp = Mul<float>::build(2.f)
                          .then(Add<float>::build(0.5f))
                          .then(makeAttentionWrite(o, bh, seqQ, HEAD_DIM));
    executeFlashAttentionMma<HEAD_DIM>(qIOp, kIOp, vIOp, oIOp, bh, seqQ, seqK,
                                       causal, stream);
    stream.sync();

    std::vector<float> got(nQ);
    gpuErrchk(cudaMemcpy(got.data(), o, nQ * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(o);

    double maxErr = 0.0;
    for (size_t i = 0; i < nQ; ++i)
        maxErr = std::max(maxErr, std::abs((double)got[i] - (6.0 * ref[i] + 2.5)));
    report(name, maxErr, tol);
}

template <int HEAD_DIM>
static void testBf16WriteIOp() {
    constexpr int BH = 2, SQ = 64, SK = 96;
    std::mt19937 rng(654);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    const size_t nQ = (size_t)BH * SQ * HEAD_DIM, nK = (size_t)BH * SK * HEAD_DIM;
    std::vector<float> hq(nQ), hk(nK), hv(nK);
    std::vector<double> dq(nQ), dk(nK), dv(nK), ref;
    for (size_t i = 0; i < nQ; ++i) { hq[i] = bf16Round(dist(rng)); dq[i] = hq[i]; }
    for (size_t i = 0; i < nK; ++i) { hk[i] = bf16Round(dist(rng)); dk[i] = hk[i]; }
    for (size_t i = 0; i < nK; ++i) { hv[i] = bf16Round(dist(rng)); dv[i] = hv[i]; }
    cpuAttention(dq, dk, dv, ref, BH, SQ, SK, HEAD_DIM,
                 1.0 / std::sqrt((double)HEAD_DIM), false);

    float *q, *k, *v;
    __nv_bfloat16* o;
    gpuErrchk(cudaMalloc(&q, nQ * sizeof(float)));
    gpuErrchk(cudaMalloc(&k, nK * sizeof(float)));
    gpuErrchk(cudaMalloc(&v, nK * sizeof(float)));
    gpuErrchk(cudaMalloc(&o, nQ * sizeof(__nv_bfloat16)));
    gpuErrchk(cudaMemcpy(q, hq.data(), nQ * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(k, hk.data(), nK * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(v, hv.data(), nK * sizeof(float), cudaMemcpyHostToDevice));

    Stream stream;
    const auto qIOp = makeAttentionRead(q, BH, SQ, HEAD_DIM);
    const auto kIOp = makeAttentionRead(k, BH, SK, HEAD_DIM);
    const auto vIOp = makeAttentionRead(v, BH, SK, HEAD_DIM);
    executeFlashAttentionMma<HEAD_DIM>(qIOp, kIOp, vIOp,
                                       makeAttentionWrite(o, BH, SQ, HEAD_DIM),
                                       BH, SQ, SK, false, stream);
    stream.sync();

    std::vector<__nv_bfloat16> got(nQ);
    gpuErrchk(cudaMemcpy(got.data(), o, nQ * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(o);

    double maxErr = 0.0;
    for (size_t i = 0; i < nQ; ++i)
        maxErr = std::max(maxErr, std::abs((double)__bfloat162float(got[i]) - ref[i]));
    report("FA-mma canonical WriteIOp fp32->bf16", maxErr, 3e-2);
}

int launch() {
    testDense<64>("FA-mma dense d64 b2 s128 causal", 2, 128, 128, true, 2.5e-2, 1);
    testDense<64>("FA-mma dense d64 ragged s131/s259", 2, 131, 259, false, 2.5e-2, 2);
    testDense<32>("FA-mma dense d32 cross s96->s192", 2, 96, 192, false, 2.5e-2, 3);
    testDense<128>("FA-mma dense d128 b1 s128", 1, 128, 128, false, 2.5e-2, 4);
    testDenseBf16Raw<64>("FA-mma RAW bf16 d64 b2 s128 causal", 2, 128, 128, true, 2.5e-2, 5);
    testDenseBf16Raw<128>("FA-mma RAW bf16 d128 b2 s256", 2, 256, 256, false, 2.5e-2, 6);
    testInt8KV<64>("FA-mma int8-KV d64 b2 s128 causal", 2, 128, 128, true, 3e-2, 11);
#ifdef FK_HAS_FP8
    testFp8KV<64>("FA-mma fp8-KV d64 ragged s96/s160", 2, 96, 160, false, 4e-2, 12);
#endif
    testFusedEpilogue();
    testFusedPrologue();
    // fused Read IOps + fused Write IOp, small and BIG shapes
    testFusedReadWrite<32>("FA-mma fused R/W IOps d32 b2 s64/s96", 2, 64, 96, false, 1e-1, 7);
    testFusedReadWrite<64>("FA-mma fused R/W IOps d64 b4 s512 causal", 4, 512, 512, true, 1e-1, 8);
    testFusedReadWrite<128>("FA-mma fused R/W IOps d128 b2 s384/s333", 2, 384, 333, false, 1e-1, 9);
    // small grid + long KV -> SPLIT-KV path: fused write runs in the combine
    testFusedReadWrite<64>("FA-mma fused R/W IOps split-KV d64 b2 s64/s2048", 2, 64, 2048, false, 1e-1, 10);
    testBf16WriteIOp<64>();
    if (failures == 0) { return 0; }
    std::cout << failures << " attention-mma test(s) FAILED" << std::endl;
    return -1;
}
