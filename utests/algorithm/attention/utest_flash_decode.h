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
#include <fused_kernel/algorithms/attention/flash_decode.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

namespace fk {

/* FlashDecodeDPP (split-KV decode: one query row per batch*head, its KV range split over the CTAs of the row, the last CTA
 * to arrive merging the partials) against a double-precision CPU oracle, o = softmax(scale * q K^T) V, scale = 1 / sqrt(d).
 * fp32 inputs in [-1, 1]; the bound is the fp32 attention tests' (utest_flash_attention.h, 5e-6). The workspace is the
 * caller's: flashDecodeWorkspaceFloats(batchHeads, splits, HEAD_DIM) floats, zeroed once after its allocation, as the
 * executor requires, with splits the library's automatic choice (flashDecodeSplits). The output is pre-filled with NaN, so
 * an element the kernel does not write fails the comparison. */

static void cpuDecode(const std::vector<double>& q, const std::vector<double>& k, const std::vector<double>& v,
                      std::vector<double>& o, const int bh, const int seqK, const int d, const double scale) {
    o.assign((size_t)bh * d, 0.0);
    std::vector<double> s(seqK);
    for (int b = 0; b < bh; ++b) {
        const double* Q = q.data() + (size_t)b * d;
        const double* K = k.data() + (size_t)b * seqK * d;
        const double* V = v.data() + (size_t)b * seqK * d;
        double* O = o.data() + (size_t)b * d;
        double m = -1e300;
        for (int j = 0; j < seqK; ++j) {
            double dot = 0.0;
            for (int c = 0; c < d; ++c) dot += Q[c] * K[(size_t)j * d + c];
            s[j] = dot * scale;
            m = std::max(m, s[j]);
        }
        double l = 0.0;
        for (int j = 0; j < seqK; ++j) { s[j] = std::exp(s[j] - m); l += s[j]; }
        for (int j = 0; j < seqK; ++j) {
            const double pj = s[j] / l;
            for (int c = 0; c < d; ++c) O[c] += pj * V[(size_t)j * d + c];
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
static void testDecode(const char* name, const int bh, const int seqK, const double tol, const unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    const size_t nQ = (size_t)bh * HEAD_DIM, nK = (size_t)bh * seqK * HEAD_DIM;
    std::vector<float> hq(nQ), hk(nK), hv(nK);
    std::vector<double> dq(nQ), dk(nK), dv(nK), dref;
    for (size_t i = 0; i < nQ; ++i) { hq[i] = dist(rng); dq[i] = hq[i]; }
    for (size_t i = 0; i < nK; ++i) { hk[i] = dist(rng); dk[i] = hk[i]; }
    for (size_t i = 0; i < nK; ++i) { hv[i] = dist(rng); dv[i] = hv[i]; }
    cpuDecode(dq, dk, dv, dref, bh, seqK, HEAD_DIM, 1.0 / std::sqrt((double)HEAD_DIM));

    // the caller's workspace, sized for the library's automatic number of splits and zeroed once
    const int splits = flashDecodeSplits(bh, seqK);
    const size_t workspaceFloats = flashDecodeWorkspaceFloats(bh, splits, HEAD_DIM);
    float *fq, *fk_, *fv, *o, *workspace;
    gpuErrchk(cudaMalloc(&fq, nQ * sizeof(float)));
    gpuErrchk(cudaMalloc(&fk_, nK * sizeof(float)));
    gpuErrchk(cudaMalloc(&fv, nK * sizeof(float)));
    gpuErrchk(cudaMalloc(&o, nQ * sizeof(float)));
    gpuErrchk(cudaMalloc(&workspace, workspaceFloats * sizeof(float)));
    gpuErrchk(cudaMemset(workspace, 0, workspaceFloats * sizeof(float)));
    gpuErrchk(cudaMemset(o, 0xff, nQ * sizeof(float)));   // NaN pattern
    gpuErrchk(cudaMemcpy(fq, hq.data(), nQ * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(fk_, hk.data(), nK * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(fv, hv.data(), nK * sizeof(float), cudaMemcpyHostToDevice));

    Stream stream;
    const auto q = makeAttentionRead(fq, bh, 1, HEAD_DIM);
    const auto k = makeAttentionRead(fk_, bh, seqK, HEAD_DIM);
    const auto v = makeAttentionRead(fv, bh, seqK, HEAD_DIM);
    using DPP = FlashDecodeDPP<ParArch::GPU_NVIDIA, float, HEAD_DIM, std::decay_t<decltype(q)>, std::decay_t<decltype(k)>,
                               std::decay_t<decltype(v)>>;
    executeOperations<DPP>(stream, q, k, v, o, workspace, bh, seqK);
    stream.sync();

    std::vector<float> result(nQ);
    gpuErrchk(cudaMemcpy(result.data(), o, nQ * sizeof(float), cudaMemcpyDeviceToHost));
    double maxErr = 0.0;
    for (size_t i = 0; i < nQ; ++i) {
        const double err = std::fabs((double)result[i] - dref[i]);
        maxErr = std::isnan(err) ? err : std::max(maxErr, err);
        if (std::isnan(maxErr)) break;
    }
    const std::string label = std::string(name) + " (" + std::to_string(splits) + " splits)";
    report(label.c_str(), maxErr, tol);
    if (splits < 2) {
        std::cout << "FAIL " << name << ": flashDecodeSplits chose " << splits << " split, the merge is not exercised" << std::endl;
        ++failures;
    }
    gpuErrchk(cudaFree(fq));
    gpuErrchk(cudaFree(fk_));
    gpuErrchk(cudaFree(fv));
    gpuErrchk(cudaFree(o));
    gpuErrchk(cudaFree(workspace));
}

int launch_impl() {
    // bh 4, seqK 1001: flashDecodeSplits gives 4 splits of 251 keys, the last one ragged (248)
    testDecode<64>("FlashDecode d64 bh4 s1001", 4, 1001, 5e-6, 41);
    if (failures == 0) { return 0; }
    std::cout << failures << " FlashDecodeDPP check(s) FAILED" << std::endl;
    return -1;
}

} // namespace fk

int launch() {
    return fk::launch_impl();
}
