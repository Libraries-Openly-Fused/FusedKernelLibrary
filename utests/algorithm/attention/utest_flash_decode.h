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

/* FlashDecodeDPP (seq_q == 1, split-KV) against the double-precision oracle,
 * through executeFlashDecode, for every K/V read path of loadRow: generic
 * (fp32 PerThreadRead), raw bf16 (vectorized bfloat162 loads), and the
 * int8 / fp8 per-token caches (packed byte loads). The quantized cases give
 * the oracle the dequantized values, as utest_flash_attention does.
 * Forced split counts larger than seqK / 4 leave some splits empty, which
 * must still arrive at the fused last-block merge; every case runs twice on
 * the same workspace to check that the arrival counters reset themselves. */

#define __ONLY_CU__ // This file is only generated and compiled with nvcc, not with the host compiler
#include <tests/main.h>

#include <fused_kernel/algorithms/attention/flash_decode.h>
#include <utests/algorithm/attention/attention_test_common.h>

#include <random>

namespace fk {

// Decode accumulates in fp32 like FlashAttentionDPP, but its softmax uses
// __expf (fast exp, a few ulp) instead of expf, hence a looser bound than the
// 5e-6 of utest_flash_attention.
constexpr double DECODE_TOL = 1e-5;

enum class DecodeKV { FP32, BF16, INT8, FP8 };

#ifdef FK_HAS_FP8
// host decode of one e4m3 byte, for the oracle
static double fp8ToDouble(const int8_t byte) {
    __nv_fp8_e4m3 f8;
    f8.__x = static_cast<__nv_fp8_storage_t>(byte);
    return static_cast<double>(static_cast<float>(f8));
}
#endif

template <int HEAD_DIM, typename QIOp, typename KIOp, typename VIOp>
static void runDecode(const char* name, const QIOp& q, const KIOp& k, const VIOp& v,
                      const std::vector<double>& ref, const int bh, const int seqK,
                      const int splitsOverride, Stream& stream) {
    const int splits = splitsOverride > 0 ? splitsOverride : flashDecodeSplits(bh, seqK);
    auto workspace = zeroBuffer<float>(flashDecodeWorkspaceFloats(bh, splits, HEAD_DIM), stream);
    auto o = zeroBuffer<float>((size_t)bh * HEAD_DIM, stream);
    for (int run = 0; run < 2; ++run) {
        executeFlashDecode<HEAD_DIM>(q, k, v, o.ptr().data, workspace.ptr().data, bh, seqK, stream,
                                     -1.f, splitsOverride);
        const std::vector<float> got = downloadBuffer(o, stream);
        const std::string runName = std::string(name) + " splits=" + std::to_string(splits) +
                                    (run == 0 ? "" : " (2nd call, same workspace)");
        report(runName.c_str(), maxAbsErr(got, ref), DECODE_TOL);
    }
}

template <int HEAD_DIM, DecodeKV KV>
static void testDecode(const char* name, const int bh, const int seqK, const int splitsOverride,
                       const unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    const size_t nQ = (size_t)bh * HEAD_DIM, nK = (size_t)bh * seqK * HEAD_DIM;
    const size_t nTok = (size_t)bh * seqK;
    std::vector<float> hq(nQ), hk(nK), hv(nK);
    for (auto& x : hq) x = dist(rng);
    for (auto& x : hk) x = dist(rng);
    for (auto& x : hv) x = dist(rng);

    std::vector<double> dq(nQ), dk(nK), dv(nK), ref;
    Stream stream;
    const double scale = 1.0 / std::sqrt((double)HEAD_DIM);
    if constexpr (KV == DecodeKV::FP32) {
        for (size_t i = 0; i < nQ; ++i) dq[i] = hq[i];
        for (size_t i = 0; i < nK; ++i) { dk[i] = hk[i]; dv[i] = hv[i]; }
        cpuAttention(dq, dk, dv, ref, bh, 1, seqK, HEAD_DIM, scale, false);
        auto q = uploadBuffer(hq, stream);
        auto kd = uploadBuffer(hk, stream);
        auto vd = uploadBuffer(hv, stream);
        runDecode<HEAD_DIM>(name, makeAttentionRead(q.ptr().data, bh, 1, HEAD_DIM),
                            makeAttentionRead(kd.ptr().data, bh, seqK, HEAD_DIM),
                            makeAttentionRead(vd.ptr().data, bh, seqK, HEAD_DIM), ref, bh, seqK,
                            splitsOverride, stream);
    } else if constexpr (KV == DecodeKV::BF16) {
        // Q, K and V are all bf16 (the raw prologue); the oracle sees the
        // bf16-rounded values
        std::vector<__nv_bfloat16> bq(nQ), bk(nK), bv(nK);
        for (size_t i = 0; i < nQ; ++i) { bq[i] = __float2bfloat16(hq[i]); dq[i] = __bfloat162float(bq[i]); }
        for (size_t i = 0; i < nK; ++i) {
            bk[i] = __float2bfloat16(hk[i]); dk[i] = __bfloat162float(bk[i]);
            bv[i] = __float2bfloat16(hv[i]); dv[i] = __bfloat162float(bv[i]);
        }
        cpuAttention(dq, dk, dv, ref, bh, 1, seqK, HEAD_DIM, scale, false);
        auto q = uploadBuffer(bq, stream);
        auto kd = uploadBuffer(bk, stream);
        auto vd = uploadBuffer(bv, stream);
        const auto kIOp = makeAttentionRead(static_cast<const __nv_bfloat16*>(kd.ptr().data), bh, seqK, HEAD_DIM);
        static_assert(isRawBf16Read<std::decay_t<decltype(kIOp)>>, "bf16 K must take the raw (vectorized) path");
        runDecode<HEAD_DIM>(name, makeAttentionRead(static_cast<const __nv_bfloat16*>(q.ptr().data), bh, 1, HEAD_DIM),
                            kIOp,
                            makeAttentionRead(static_cast<const __nv_bfloat16*>(vd.ptr().data), bh, seqK, HEAD_DIM),
                            ref, bh, seqK, splitsOverride, stream);
    } else {
        std::vector<int8_t> k8(nK), v8(nK);
        std::vector<float> kSc(nTok), vSc(nTok);
        for (size_t i = 0; i < nQ; ++i) dq[i] = hq[i];
        if constexpr (KV == DecodeKV::INT8) {
            quantizeKVCacheHost(hk.data(), k8.data(), kSc.data(), (int)nTok, HEAD_DIM);
            quantizeKVCacheHost(hv.data(), v8.data(), vSc.data(), (int)nTok, HEAD_DIM);
            for (size_t i = 0; i < nK; ++i) {
                dk[i] = (double)k8[i] * kSc[i / HEAD_DIM];
                dv[i] = (double)v8[i] * vSc[i / HEAD_DIM];
            }
        } else {
#ifdef FK_HAS_FP8
            quantizeKVCacheFp8Host(hk.data(), k8.data(), kSc.data(), (int)nTok, HEAD_DIM);
            quantizeKVCacheFp8Host(hv.data(), v8.data(), vSc.data(), (int)nTok, HEAD_DIM);
            for (size_t i = 0; i < nK; ++i) {
                dk[i] = fp8ToDouble(k8[i]) * kSc[i / HEAD_DIM];
                dv[i] = fp8ToDouble(v8[i]) * vSc[i / HEAD_DIM];
            }
#endif
        }
        cpuAttention(dq, dk, dv, ref, bh, 1, seqK, HEAD_DIM, scale, false);
        auto q = uploadBuffer(hq, stream);
        auto kd = uploadBuffer(k8, stream);
        auto vd = uploadBuffer(v8, stream);
        auto ks = uploadBuffer(kSc, stream);
        auto vs = uploadBuffer(vSc, stream);
        const auto qIOp = makeAttentionRead(q.ptr().data, bh, 1, HEAD_DIM);
        if constexpr (KV == DecodeKV::INT8) {
            runDecode<HEAD_DIM>(name, qIOp, makeInt8KVRead(kd.ptr().data, ks.ptr().data, bh, seqK, HEAD_DIM),
                                makeInt8KVRead(vd.ptr().data, vs.ptr().data, bh, seqK, HEAD_DIM), ref, bh, seqK,
                                splitsOverride, stream);
        } else {
#ifdef FK_HAS_FP8
            runDecode<HEAD_DIM>(name, qIOp, makeFp8KVRead(kd.ptr().data, ks.ptr().data, bh, seqK, HEAD_DIM),
                                makeFp8KVRead(vd.ptr().data, vs.ptr().data, bh, seqK, HEAD_DIM), ref, bh, seqK,
                                splitsOverride, stream);
#endif
        }
    }
}

int launch_impl() {
    // splits=0: flashDecodeSplits picks them (4 for bh4 s1000, 3 for bh2 s513)
    testDecode<64, DecodeKV::FP32>("FlashDecode fp32 d64 bh4 s1000", 4, 1000, 0, 1);
    testDecode<32, DecodeKV::FP32>("FlashDecode fp32 d32 bh2 s300", 2, 300, 1, 2);
    testDecode<128, DecodeKV::FP32>("FlashDecode fp32 d128 bh2 s513", 2, 513, 0, 3);
    testDecode<64, DecodeKV::FP32>("FlashDecode fp32 d64 bh2 s100 empty splits", 2, 100, 30, 4);
    testDecode<64, DecodeKV::BF16>("FlashDecode bf16 d64 bh4 s1000", 4, 1000, 0, 5);
    testDecode<128, DecodeKV::BF16>("FlashDecode bf16 d128 bh2 s100 empty splits", 2, 100, 30, 6);
    testDecode<64, DecodeKV::INT8>("FlashDecode int8-KV d64 bh4 s1000", 4, 1000, 0, 7);
    testDecode<128, DecodeKV::INT8>("FlashDecode int8-KV d128 bh2 s100 empty splits", 2, 100, 30, 8);
#ifdef FK_HAS_FP8
    testDecode<64, DecodeKV::FP8>("FlashDecode fp8-KV d64 bh4 s1000", 4, 1000, 0, 9);
    testDecode<128, DecodeKV::FP8>("FlashDecode fp8-KV d128 bh2 s100 empty splits", 2, 100, 30, 10);
#endif
    // d32 (1 element per lane) and d256 (8): the packed loads don't cover
    // these widths, loadRow falls back to reading each element
    testDecode<32, DecodeKV::BF16>("FlashDecode bf16 d32 bh2 s300", 2, 300, 0, 11);
    testDecode<256, DecodeKV::BF16>("FlashDecode bf16 d256 bh2 s300", 2, 300, 0, 12);
    testDecode<32, DecodeKV::INT8>("FlashDecode int8-KV d32 bh2 s300", 2, 300, 0, 13);
    testDecode<256, DecodeKV::INT8>("FlashDecode int8-KV d256 bh2 s300", 2, 300, 0, 14);
#ifdef FK_HAS_FP8
    testDecode<32, DecodeKV::FP8>("FlashDecode fp8-KV d32 bh2 s300", 2, 300, 0, 15);
    testDecode<256, DecodeKV::FP8>("FlashDecode fp8-KV d256 bh2 s300", 2, 300, 0, 16);
#endif
    if (failures == 0) { return 0; }
    std::cout << failures << " flash decode test(s) FAILED" << std::endl;
    return -1;
}

} // namespace fk

int launch() {
    return fk::launch_impl();
}
