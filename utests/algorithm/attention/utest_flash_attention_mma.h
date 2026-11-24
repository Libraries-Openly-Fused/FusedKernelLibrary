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

#define __ONLY_CU__ // This file is only generated and compiled with nvcc, not with the host compiler
#include <tests/main.h>

#include <fused_kernel/algorithms/attention/flash_attention_mma.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>

#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace fk {

// double-precision CPU oracle: O = softmax(scale * Q K^T [causal]) V
static void cpuAttentionMma(const std::vector<double>& q, const std::vector<double>& k,
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

// Random bf16 data plus its exact value in double (the oracle sees the
// bf16-rounded inputs, so only the kernel's own rounding is measured).
static void randomBf16(std::mt19937& rng, const size_t n, std::vector<__nv_bfloat16>& h,
                       std::vector<double>& exact) {
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    h.resize(n);
    exact.resize(n);
    for (size_t i = 0; i < n; ++i) {
        h[i] = __float2bfloat16(dist(rng));
        exact[i] = (double)__bfloat162float(h[i]);
    }
}

static __nv_bfloat16* toDevice(const std::vector<__nv_bfloat16>& h) {
    __nv_bfloat16* d;
    gpuErrchk(cudaMalloc(&d, h.size() * sizeof(__nv_bfloat16)));
    gpuErrchk(cudaMemcpy(d, h.data(), h.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    return d;
}

/* V_FUSED=false: raw bf16 reads on Q/K/V -> the cp.async schedule.
   V_FUSED=true: V is read through .then(Mul(3)).then(Add(1)) -> the
   register-prefetch schedule; since sum p_j = 1 the result is 3*oracle + 1. */
template <int HEAD_DIM, bool V_FUSED = false>
static void testMma(const char* name, const int bh, const int seqQ, const int seqK,
                    const bool causal, const double tol, const unsigned seed) {
    std::mt19937 rng(seed);
    const size_t nQ = (size_t)bh * seqQ * HEAD_DIM, nK = (size_t)bh * seqK * HEAD_DIM;
    std::vector<__nv_bfloat16> hq, hk, hv;
    std::vector<double> dq, dk, dv, ref;
    randomBf16(rng, nQ, hq, dq);
    randomBf16(rng, nK, hk, dk);
    randomBf16(rng, nK, hv, dv);
    cpuAttentionMma(dq, dk, dv, ref, bh, seqQ, seqK, HEAD_DIM,
                    1.0 / std::sqrt((double)HEAD_DIM), causal);

    __nv_bfloat16* q = toDevice(hq);
    __nv_bfloat16* k = toDevice(hk);
    __nv_bfloat16* v = toDevice(hv);
    float* o;
    gpuErrchk(cudaMalloc(&o, nQ * sizeof(float)));

    Stream stream;
    const auto qIOp = makeAttentionRead(q, bh, seqQ, HEAD_DIM);
    const auto kIOp = makeAttentionRead(k, bh, seqK, HEAD_DIM);
    if constexpr (V_FUSED) {
        const auto vIOp = makeAttentionRead(v, bh, seqK, HEAD_DIM)
                              .then(Mul<float>::build(3.f))
                              .then(Add<float>::build(1.f));
        executeFlashAttentionMma<HEAD_DIM>(qIOp, kIOp, vIOp, o, bh, seqQ, seqK, causal, stream);
    } else {
        const auto vIOp = makeAttentionRead(v, bh, seqK, HEAD_DIM);
        executeFlashAttentionMma<HEAD_DIM>(qIOp, kIOp, vIOp, o, bh, seqQ, seqK, causal, stream);
    }
    stream.sync();

    std::vector<float> got(nQ);
    gpuErrchk(cudaMemcpy(got.data(), o, nQ * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(o);

    double maxErr = 0.0;
    for (size_t i = 0; i < nQ; ++i) {
        const double expected = V_FUSED ? 3.0 * ref[i] + 1.0 : ref[i];
        maxErr = std::max(maxErr, std::abs((double)got[i] - expected));
    }
    report(name, maxErr, tol);
}

// Below compute capability 8.0 the launcher must refuse instead of launching.
static void testUnsupportedDeviceThrows() {
    constexpr int HEAD_DIM = 64, BH = 1, S = 64;
    std::vector<__nv_bfloat16> h((size_t)BH * S * HEAD_DIM, __float2bfloat16(0.f));
    __nv_bfloat16* x = toDevice(h);
    float* o;
    gpuErrchk(cudaMalloc(&o, h.size() * sizeof(float)));
    Stream stream;
    const auto io = makeAttentionRead(x, BH, S, HEAD_DIM);
    bool threw = false;
    try {
        executeFlashAttentionMma<HEAD_DIM>(io, io, io, o, BH, S, S, false, stream);
        stream.sync();
    } catch (const std::runtime_error&) {
        threw = true;
    }
    cudaFree(x); cudaFree(o);
    report("FA-MMA throws below compute capability 8.0", threw ? 0.0 : 1.0, 0.0);
}

int launch_impl() {
    int device = 0, ccMajor = 0;
    gpuErrchk(cudaGetDevice(&device));
    gpuErrchk(cudaDeviceGetAttribute(&ccMajor, cudaDevAttrComputeCapabilityMajor, device));
    if (ccMajor < 8) {
        testUnsupportedDeviceThrows();
    } else {
        // P is rounded to bf16 for the PV mma, hence the bf16-level tolerances.
        testMma<64>("FA-MMA bf16 d64 b2 s128 causal", 2, 128, 128, true, 1e-2, 1);
        testMma<64>("FA-MMA bf16 d64 ragged s67/s131", 2, 67, 131, false, 1e-2, 2);
        testMma<64>("FA-MMA bf16 d64 cross s32->s96 causal", 2, 32, 96, true, 1e-2, 3);
        testMma<128>("FA-MMA bf16 d128 b1 s64", 1, 64, 64, false, 1e-2, 4);
        // one q-block against a long KV range -> split-KV + combine kernel
        testMma<64>("FA-MMA bf16 d64 split-KV s64/s1024", 1, 64, 1024, false, 1e-2, 5);
        testMma<64, true>("FA-MMA bf16 d64 V-prologue .then(Mul(3)).then(Add(1))", 2, 64, 96, false,
                          3e-2, 6);
    }
    if (failures == 0) { return 0; }
    std::cout << failures << " attention MMA test(s) FAILED" << std::endl;
    return -1;
}

} // namespace fk

int launch() {
    return fk::launch_impl();
}
