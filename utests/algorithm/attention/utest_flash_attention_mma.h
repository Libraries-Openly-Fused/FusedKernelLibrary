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

#include <fused_kernel/algorithms/attention/flash_attention.h>
#include <fused_kernel/algorithms/attention/flash_attention_mma.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>

using namespace fk;

/* FlashAttentionMmaDPP (mma.sync m16n8k16 on bf16 tensor cores) against
 *   1. a double-precision CPU oracle, O = softmax(scale * Q K^T [causal]) V, evaluated on the bf16-ROUNDED inputs (so the
 *      oracle and the kernels read the same values), and
 *   2. the SIMT FlashAttention DPP (FlashAttentionDPP<ParArch::GPU_NVIDIA, float, HEAD_DIM, ...>) on the same values.
 * Head dimensions 64 and 128, dense and causal (causal only with seqQ == seqK), aligned and ragged lengths, cross-attention.
 * The MMA kernel converts the probabilities to bf16 before the second matrix product and accumulates in fp32, so its error against the
 * oracle is that of a bf16 product (up to about 1.2e-3 in these cases), not the 1e-7 of the fp32 SIMT kernel; the tolerance below is a bound that a real
 * indexing, masking or softmax bug exceeds by a wide margin (those give errors of order 1e-1 to 1). The kernels need sm_80 or newer
 * (ldmatrix, cp.async, bf16 mma): on an older device the test prints a message and succeeds. */

static void cpuAttention(const std::vector<double>& q, const std::vector<double>& k, const std::vector<double>& v,
                         std::vector<double>& o, const int bh, const int seqQ, const int seqK, const int d,
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
                for (int c = 0; c < d; ++c) dot += Q[(size_t)i * d + c] * K[(size_t)j * d + c];
                s[j] = dot * scale;
                m = std::max(m, s[j]);
            }
            double l = 0.0;
            for (int j = 0; j < kEnd; ++j) { s[j] = std::exp(s[j] - m); l += s[j]; }
            for (int j = 0; j < kEnd; ++j) {
                const double pj = s[j] / l;
                for (int c = 0; c < d; ++c) O[(size_t)i * d + c] += pj * V[(size_t)j * d + c];
            }
        }
    }
}

static int failures = 0;

static void report(const char* name, const double maxErr, const double tol) {
    if (!(maxErr <= tol)) {   // also fails on NaN
        std::cout << "FAIL " << name << ": maxErr=" << maxErr << " tol=" << tol << std::endl;
        ++failures;
    } else {
        std::cout << "Running test " << name << ": Success!! (maxErr=" << maxErr << ")" << std::endl;
    }
}

template <int HEAD_DIM>
static void testMma(const char* name, const int bh, const int seqQ, const int seqK, const bool causal, const double tolOracle,
                    const double tolSimt, const unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    const size_t nQ = (size_t)bh * seqQ * HEAD_DIM, nK = (size_t)bh * seqK * HEAD_DIM;
    // bf16-rounded inputs: hq/hk/hv hold the float value of the rounded number, hbq/hbk/hbv the bf16 bits for the MMA kernel
    std::vector<float> hq(nQ), hk(nK), hv(nK);
    std::vector<__nv_bfloat16> hbq(nQ), hbk(nK), hbv(nK);
    std::vector<double> dq(nQ), dk(nK), dv(nK), dref;
    for (size_t i = 0; i < nQ; ++i) { hbq[i] = __float2bfloat16(dist(rng)); hq[i] = __bfloat162float(hbq[i]); dq[i] = hq[i]; }
    for (size_t i = 0; i < nK; ++i) { hbk[i] = __float2bfloat16(dist(rng)); hk[i] = __bfloat162float(hbk[i]); dk[i] = hk[i]; }
    for (size_t i = 0; i < nK; ++i) { hbv[i] = __float2bfloat16(dist(rng)); hv[i] = __bfloat162float(hbv[i]); dv[i] = hv[i]; }
    cpuAttention(dq, dk, dv, dref, bh, seqQ, seqK, HEAD_DIM, 1.0 / std::sqrt((double)HEAD_DIM), causal);

    __nv_bfloat16 *bq, *bk, *bv;
    float *fq, *fk_, *fv, *oMma, *oSimt;
    gpuErrchk(cudaMalloc(&bq, nQ * sizeof(__nv_bfloat16)));
    gpuErrchk(cudaMalloc(&bk, nK * sizeof(__nv_bfloat16)));
    gpuErrchk(cudaMalloc(&bv, nK * sizeof(__nv_bfloat16)));
    gpuErrchk(cudaMalloc(&fq, nQ * sizeof(float)));
    gpuErrchk(cudaMalloc(&fk_, nK * sizeof(float)));
    gpuErrchk(cudaMalloc(&fv, nK * sizeof(float)));
    gpuErrchk(cudaMalloc(&oMma, nQ * sizeof(float)));
    gpuErrchk(cudaMalloc(&oSimt, nQ * sizeof(float)));
    gpuErrchk(cudaMemset(oMma, 0xff, nQ * sizeof(float)));   // NaN pattern: an element the kernel does not write fails the comparison
    gpuErrchk(cudaMemcpy(bq, hbq.data(), nQ * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(bk, hbk.data(), nK * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(bv, hbv.data(), nK * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(fq, hq.data(), nQ * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(fk_, hk.data(), nK * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(fv, hv.data(), nK * sizeof(float), cudaMemcpyHostToDevice));

    Stream stream;
    // The library's automatic choice for these shapes: BQ64 / BKV64 with 4 warps, and no split (fewer than 8 KV tiles per
    // q-block), so the test runs that DPP
    const auto qRaw = makeAttentionRead(bq, bh, seqQ, HEAD_DIM);
    const auto kRaw = makeAttentionRead(bk, bh, seqK, HEAD_DIM);
    const auto vRaw = makeAttentionRead(bv, bh, seqK, HEAD_DIM);
    using Raw = std::decay_t<decltype(kRaw)>;
    using MmaDPP = FlashAttentionMmaDPP<ParArch::GPU_NVIDIA, float, HEAD_DIM, Raw, Raw, Raw, AttentionIdentityEpilogue,
                                        64, 64, 4>;
    const FlashAttentionMmaTiles tiles = flashAttentionMmaTiles<HEAD_DIM, Raw, Raw>(bh, seqQ, causal);
    const int splits = flashAttentionMmaSplits<MmaDPP>(bh, seqQ, seqK, causal);
    if (tiles.blockQ != 64 || tiles.blockKV != 64 || tiles.numWarps != 4 || splits != 1) {
        std::cout << "FAIL " << name << ": automatic choice " << tiles.blockQ << "x" << tiles.blockKV << "x" << tiles.numWarps
                  << ", " << splits << " splits, expected 64x64x4 and 1" << std::endl;
        ++failures;
    }
    executeOperations<MmaDPP>(stream, qRaw, kRaw, vRaw, oMma, bh, seqQ, seqK, causal);
    const auto qRead = makeAttentionRead(fq, bh, seqQ, HEAD_DIM);
    const auto kRead = makeAttentionRead(fk_, bh, seqK, HEAD_DIM);
    const auto vRead = makeAttentionRead(fv, bh, seqK, HEAD_DIM);
    executeOperations<FlashAttentionDPP<ParArch::GPU_NVIDIA, float, HEAD_DIM, std::decay_t<decltype(qRead)>,
                                        std::decay_t<decltype(kRead)>, std::decay_t<decltype(vRead)>>>(
        stream, qRead, kRead, vRead, oSimt, bh, seqQ, seqK, causal);
    stream.sync();
    gpuErrchk(cudaGetLastError());

    std::vector<float> gotMma(nQ), gotSimt(nQ);
    gpuErrchk(cudaMemcpy(gotMma.data(), oMma, nQ * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(gotSimt.data(), oSimt, nQ * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(bq); cudaFree(bk); cudaFree(bv); cudaFree(fq); cudaFree(fk_); cudaFree(fv); cudaFree(oMma); cudaFree(oSimt);

    double errOracle = 0.0, errSimtOracle = 0.0, errMmaSimt = 0.0;
    for (size_t i = 0; i < nQ; ++i) {
        const double a = gotMma[i], b = gotSimt[i];
        errOracle = std::max(errOracle, std::isnan(a) ? 1e300 : std::abs(a - dref[i]));
        errSimtOracle = std::max(errSimtOracle, std::isnan(b) ? 1e300 : std::abs(b - dref[i]));
        errMmaSimt = std::max(errMmaSimt, (std::isnan(a) || std::isnan(b)) ? 1e300 : std::abs(a - b));
    }
    const std::string base = name;
    report((base + " MMA vs double oracle").c_str(), errOracle, tolOracle);
    report((base + " SIMT vs double oracle").c_str(), errSimtOracle, 5e-6);
    report((base + " MMA vs SIMT").c_str(), errMmaSimt, tolSimt);
}

/* The split-KV pair: FlashAttentionMmaSplitKVDPP writes the unnormalized partials of `splits` KV chunks and its CombineDPP
 * merges them and applies the epilogue, as the unsplit FlashAttentionMmaDPP does in its single write. On the bf16-rounded inputs,
 * against the double oracle, with the bound of the unsplit kernel (tol): the split only changes the maximum each probability is
 * computed against before its bf16 rounding, so its error is of the same order. With the epilogue o * 2 + 0.5 the output must be
 * within 2 * tol of 2 * oracle + 0.5 (the unsplit DPP's too): an epilogue the combine dropped would miss it by about |o + 0.5|, far more.
 * The shapes are under-filled (bh * numQ is a few blocks and every q-block has at least 8 KV tiles), so flashAttentionMmaSplits must
 * split them. */
template <int HEAD_DIM>
static void testMmaSplit(const char* name, const int bh, const int seqQ, const int seqK, const bool causal, const double tol,
                         const unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    const size_t nQ = (size_t)bh * seqQ * HEAD_DIM, nK = (size_t)bh * seqK * HEAD_DIM;
    std::vector<__nv_bfloat16> hbq(nQ), hbk(nK), hbv(nK);
    std::vector<double> dq(nQ), dk(nK), dv(nK), dref;
    for (size_t i = 0; i < nQ; ++i) { hbq[i] = __float2bfloat16(dist(rng)); dq[i] = __bfloat162float(hbq[i]); }
    for (size_t i = 0; i < nK; ++i) { hbk[i] = __float2bfloat16(dist(rng)); dk[i] = __bfloat162float(hbk[i]); }
    for (size_t i = 0; i < nK; ++i) { hbv[i] = __float2bfloat16(dist(rng)); dv[i] = __bfloat162float(hbv[i]); }
    cpuAttention(dq, dk, dv, dref, bh, seqQ, seqK, HEAD_DIM, 1.0 / std::sqrt((double)HEAD_DIM), causal);
    __nv_bfloat16 *bq, *bk, *bv;
    float *oRef, *oSplit, *oSplitId;
    gpuErrchk(cudaMalloc(&bq, nQ * sizeof(__nv_bfloat16)));
    gpuErrchk(cudaMalloc(&bk, nK * sizeof(__nv_bfloat16)));
    gpuErrchk(cudaMalloc(&bv, nK * sizeof(__nv_bfloat16)));
    for (float** p : {&oRef, &oSplit, &oSplitId}) {
        gpuErrchk(cudaMalloc(p, nQ * sizeof(float)));
        gpuErrchk(cudaMemset(*p, 0xff, nQ * sizeof(float)));   // NaN pattern: an element not written fails the comparison
    }
    gpuErrchk(cudaMemcpy(bq, hbq.data(), nQ * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(bk, hbk.data(), nK * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(bv, hbv.data(), nK * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    const auto qRaw = makeAttentionRead(bq, bh, seqQ, HEAD_DIM);
    const auto kRaw = makeAttentionRead(bk, bh, seqK, HEAD_DIM);
    const auto vRaw = makeAttentionRead(bv, bh, seqK, HEAD_DIM);
    using Raw = std::decay_t<decltype(kRaw)>;
    const auto epilogue = Mul<float>::build(2.f).then(Add<float>::build(0.5f));
    using Epi = std::decay_t<decltype(epilogue)>;
    using MmaDPP = FlashAttentionMmaDPP<ParArch::GPU_NVIDIA, float, HEAD_DIM, Raw, Raw, Raw, Epi, 64, 64, 4>;
    using SplitDPP = FlashAttentionMmaSplitKVDPP<ParArch::GPU_NVIDIA, float, HEAD_DIM, Raw, Raw, Raw, Epi, 64, 64, 4>;
    using SplitIdDPP = FlashAttentionMmaSplitKVDPP<ParArch::GPU_NVIDIA, float, HEAD_DIM, Raw, Raw, Raw,
                                                   AttentionIdentityEpilogue, 64, 64, 4>;
    static_assert(std::is_same_v<typename SplitDPP::CombineDPP,
                                 FlashAttentionSplitCombineDPP<ParArch::GPU_NVIDIA, float, HEAD_DIM,
                                                               MmaDPP::LOG2_DOMAIN, Epi>>,
                  "the combine of a split DPP has its log domain and its epilogue");
    const int splits = flashAttentionMmaSplits<MmaDPP>(bh, seqQ, seqK, causal);
    if (splits < 2) {
        std::cout << "FAIL " << name << ": flashAttentionMmaSplits gave " << splits << " for an under-filled grid" << std::endl;
        ++failures;
    }
    const int s = splits < 2 ? 2 : splits;
    float* partial = nullptr;
    gpuErrchk(cudaMalloc(&partial, flashAttentionMmaPartialFloats(bh, seqQ, s, HEAD_DIM) * sizeof(float)));

    Stream stream;
    executeOperations<MmaDPP>(stream, qRaw, kRaw, vRaw, oRef, bh, seqQ, seqK, causal, -1.f, epilogue);
    executeOperations<SplitDPP>(stream, qRaw, kRaw, vRaw, partial, s, bh, seqQ, seqK, causal, -1.f, epilogue);
    executeOperations<typename SplitDPP::CombineDPP>(stream, partial, oSplit, bh, seqQ, s, epilogue);
    executeOperations<SplitIdDPP>(stream, qRaw, kRaw, vRaw, partial, s, bh, seqQ, seqK, causal);
    executeOperations<typename SplitIdDPP::CombineDPP>(stream, partial, oSplitId, bh, seqQ, s);
    stream.sync();
    gpuErrchk(cudaGetLastError());

    std::vector<float> ref(nQ), split(nQ), splitId(nQ);
    gpuErrchk(cudaMemcpy(ref.data(), oRef, nQ * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(split.data(), oSplit, nQ * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(splitId.data(), oSplitId, nQ * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(bq); cudaFree(bk); cudaFree(bv); cudaFree(oRef); cudaFree(oSplit); cudaFree(oSplitId); cudaFree(partial);
    double errSplitId = 0.0, errSplit = 0.0, errRef = 0.0;
    for (size_t i = 0; i < nQ; ++i) {
        const double withEpilogue = 2.0 * dref[i] + 0.5;
        errSplitId = std::max(errSplitId, std::isnan(splitId[i]) ? 1e300 : std::abs((double)splitId[i] - dref[i]));
        errSplit = std::max(errSplit, std::isnan(split[i]) ? 1e300 : std::abs((double)split[i] - withEpilogue));
        errRef = std::max(errRef, std::isnan(ref[i]) ? 1e300 : std::abs((double)ref[i] - withEpilogue));
    }
    const std::string base = name;
    report((base + " split + combine vs double oracle").c_str(), errSplitId, tol);
    report((base + " split + combine, epilogue o*2+0.5, vs 2*oracle+0.5").c_str(), errSplit, 2 * tol);
    report((base + " unsplit, epilogue o*2+0.5, vs 2*oracle+0.5").c_str(), errRef, 2 * tol);
}

int launch() {
    int device = 0, major = 0;
    gpuErrchk(cudaGetDevice(&device));
    gpuErrchk(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    if (major < 8) {
        std::cout << "FlashAttentionMmaDPP needs sm_80 or newer (ldmatrix, cp.async, bf16 mma): skipped on this device" << std::endl;
        return 0;
    }
    // Measured on an RTX 4090 (sm_89) and an RTX 5050 (sm_120): the largest error of the MMA kernel against the double oracle over these seven cases is 1.21e-3 (causal, d128)
    // and its largest difference to the SIMT kernel 1.21e-3; the bound keeps a margin of 2.5x.
    constexpr double TOL = 3e-3;
    testMma<64>("FA-MMA dense d64 b2 s64", 2, 64, 64, false, TOL, TOL, 11);
    testMma<64>("FA-MMA causal d64 b2 s256", 2, 256, 256, true, TOL, TOL, 12);
    testMma<64>("FA-MMA dense d64 ragged s67/s131", 2, 67, 131, false, TOL, TOL, 13);
    testMma<64>("FA-MMA dense d64 cross s32->s96", 2, 32, 96, false, TOL, TOL, 14);
    testMma<128>("FA-MMA dense d128 b1 s128", 1, 128, 128, false, TOL, TOL, 15);
    testMma<128>("FA-MMA causal d128 b2 s200", 2, 200, 200, true, TOL, TOL, 16);
    testMma<128>("FA-MMA dense d128 ragged s70/s300", 2, 70, 300, false, TOL, TOL, 17);
    testMmaSplit<64>("FA-MMA split-KV d64 b2 s64->s1024", 2, 64, 1024, false, TOL, 21);
    testMmaSplit<64>("FA-MMA split-KV causal d64 b1 s1024", 1, 1024, 1024, true, TOL, 22);
    testMmaSplit<128>("FA-MMA split-KV d128 ragged b2 s70->s1500", 2, 70, 1500, false, TOL, 23);
    if (failures == 0) { return 0; }
    std::cout << failures << " FlashAttentionMmaDPP check(s) FAILED" << std::endl;
    return -1;
}
