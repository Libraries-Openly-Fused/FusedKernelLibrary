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

#define __ONLY_CU__ // This file is only generated and compiled with nvcc, not with the host compiler
#include <tests/main.h>

#include <fused_kernel/algorithms/attention/flash_attention.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using namespace fk;

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
    for (size_t i = 0; i < nQ; ++i) { hq[i] = dist(rng); dq[i] = hq[i]; }
    for (size_t i = 0; i < nK; ++i) { hk[i] = dist(rng); dk[i] = hk[i]; }
    for (size_t i = 0; i < nK; ++i) { hv[i] = dist(rng); dv[i] = hv[i]; }
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
    executeFlashAttention<float, HEAD_DIM>(q, k, v, o, bh, seqQ, seqK, causal, stream);
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
    executeFlashAttention<float, HEAD_DIM, KVLayout::INT8_PER_TOKEN>(
        q, dk8, dv8, o, bh, seqQ, seqK, causal, stream, dkS, dvS);
    stream.sync();

    std::vector<float> got(nQ);
    gpuErrchk(cudaMemcpy(got.data(), o, nQ * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(q); cudaFree(o); cudaFree(dk8); cudaFree(dv8); cudaFree(dkS); cudaFree(dvS);

    double maxErr = 0.0;
    for (size_t i = 0; i < nQ; ++i) maxErr = std::max(maxErr, std::abs((double)got[i] - dref[i]));
    report(name, maxErr, tol);
}

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
    executeFlashAttention<float, HEAD_DIM>(q, k, v, o1, BH, SQ, SK, false, stream);
    const auto epilogue = Mul<float>::build(2.f).then(Add<float>::build(0.5f));
    executeFlashAttention<float, HEAD_DIM, KVLayout::DENSE, 32, 4>(
        q, k, v, o2, BH, SQ, SK, false, stream, nullptr, nullptr, -1.f, epilogue);
    stream.sync();

    std::vector<float> g1(n), g2(n);
    gpuErrchk(cudaMemcpy(g1.data(), o1, n * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(g2.data(), o2, n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(o1); cudaFree(o2);

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
       since sum p_j = 1 — compare against 3*oracle + 1. */
    constexpr int HEAD_DIM = 32, BH = 2, SQ = 24, SK = 48;
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    const size_t nQ = (size_t)BH * SQ * HEAD_DIM, nK = (size_t)BH * SK * HEAD_DIM;
    std::vector<float> hq(nQ), hk(nK), hv(nK);
    for (auto& x : hq) x = dist(rng);
    for (auto& x : hk) x = dist(rng);
    for (auto& x : hv) x = dist(rng);

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
        executeFlashAttention<HEAD_DIM>(qIOp, kIOp, vIOp, o, BH, SQ, SK,
                                        false, stream);
        stream.sync();
        std::vector<float> got(nQ);
        gpuErrchk(cudaMemcpy(got.data(), o, nQ * sizeof(float), cudaMemcpyDeviceToHost));
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
        stream.sync();
        std::vector<float> got(nQ);
        gpuErrchk(cudaMemcpy(got.data(), o, nQ * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < nQ; ++i) dq[i] = hq[i];
        cpuAttention(dq, dk, dv, ref, BH, SQ, SK, HEAD_DIM, scl, false);
        double maxErr = 0.0;
        for (size_t i = 0; i < nQ; ++i)
            maxErr = std::max(maxErr, std::abs((double)got[i] - (3.0 * ref[i] + 1.0)));
        report("FlashAttention V-prologue ReadIOp.then(Mul(3)).then(Add(1))", maxErr, 5e-6);
    }

    cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(o);
}

int launch() {
    testDense<64>("FA dense d64 b2 s64 causal", 2, 64, 64, true, 5e-6, 1);
    testDense<64>("FA dense d64 ragged s67/s131", 2, 67, 131, false, 5e-6, 2);
    testDense<32>("FA dense d32 cross s32->s96", 2, 32, 96, false, 5e-6, 3);
    testDense<128>("FA dense d128 b1 s64", 1, 64, 64, false, 5e-6, 4);
    testInt8KV<64>("FA int8-KV d64 b2 s64 causal", 2, 64, 64, true, 5e-6, 5);
    testInt8KV<64>("FA int8-KV d64 ragged s50/s100", 2, 50, 100, false, 5e-6, 6);
    testFusedEpilogue();
    testFusedPrologue();
    if (failures == 0) { return 0; }
    std::cout << failures << " attention test(s) FAILED" << std::endl;
    return -1;
}
