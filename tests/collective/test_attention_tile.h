/* Copyright 2026 Johnny Nunez

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include <tests/main.h>

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/collective/attention_tile.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace fk;

namespace {

constexpr float Q_SCALE = 1.125f;
constexpr float K_BIAS = -0.0625f;
constexpr float V_SCALE = 0.875f;
constexpr float O_SCALE = 1.25f;
constexpr float O_BIAS = -0.03125f;

float qSource(const int bh, const int row, const int d) {
    return ((bh * 17 + row * 11 + d * 7) % 29 - 14) * 0.03125f;
}

float kSource(const int bh, const int row, const int d) {
    if (row == 32) return 16.f * qSource(bh, 0, d);
    return ((bh * 13 + row * 5 + d * 3) % 23 - 11) * 0.0390625f;
}

float vSource(const int bh, const int row, const int d) {
    return ((bh * 19 + row * 7 + d * 2) % 31 - 15) * 0.02734375f;
}

struct CaseResult {
    bool oraclePassed;
    std::vector<float> output;
    double maxError;
};

template <ParArch PA, int HEAD_DIM>
CaseResult runCase(const int batchHeads, const int seqQ, const int seqK,
                   const bool causal) {
    constexpr int QUERY_ROWS = 32;
    using Details = AttentionDPPDetails<HEAD_DIM, QUERY_ROWS>;
    using DPP = AttentionTileDPP<PA, Details>;
    constexpr bool GPU = PA == ParArch::GPU_NVIDIA;
    const MemType memoryType = GPU ? MemType::DeviceAndPinned : MemType::Host;

    Ptr2D<float> q(HEAD_DIM, batchHeads * seqQ, 0, memoryType);
    Ptr2D<float> k(HEAD_DIM, batchHeads * seqK, 0, memoryType);
    Ptr2D<float> v(HEAD_DIM, batchHeads * seqK, 0, memoryType);
    Ptr2D<float2> output(HEAD_DIM / 2, batchHeads * seqQ, 0, memoryType);
    for (int bh = 0; bh < batchHeads; ++bh) {
        for (int row = 0; row < seqQ; ++row)
            for (int d = 0; d < HEAD_DIM; ++d)
                q.at(Point{d, bh * seqQ + row, 0}) = qSource(bh, row, d);
        for (int row = 0; row < seqK; ++row) {
            for (int d = 0; d < HEAD_DIM; ++d) {
                k.at(Point{d, bh * seqK + row, 0}) = kSource(bh, row, d);
                v.at(Point{d, bh * seqK + row, 0}) = vSource(bh, row, d);
            }
        }
    }
    for (int row = 0; row < batchHeads * seqQ; ++row)
        for (int pair = 0; pair < HEAD_DIM / 2; ++pair)
            output.at(Point{pair, row, 0}) = float2{-999.f, -999.f};

    Stream_<PA> stream;
#if defined(__NVCC__)
    if constexpr (GPU) {
        q.upload(stream);
        k.upload(stream);
        v.upload(stream);
        output.upload(stream);
    }
#endif
    const auto inputs = make_tuple(
        PerThreadRead<ND::_2D, float>::build(q)
            .then(Mul<float>::build(Q_SCALE)),
        PerThreadRead<ND::_2D, float>::build(k)
            .then(Add<float>::build(K_BIAS)),
        PerThreadRead<ND::_2D, float>::build(v)
            .then(Mul<float>::build(V_SCALE)));
    const auto outputIOp =
        Mul<float2>::build(float2{O_SCALE, O_SCALE})
            .then(Add<float2>::build(float2{O_BIAS, O_BIAS}))
            .then(PerThreadWrite<ND::_2D, float2>::build(output));
    const float scale = 1.f / std::sqrt(static_cast<float>(HEAD_DIM));
    executeOperations<DPP>(
        stream,
        Details{batchHeads, seqQ, seqK, scale, causal, -77, -91},
        inputs, outputIOp);
#if defined(__NVCC__)
    if constexpr (GPU) output.download(stream);
#endif
    stream.sync();

    std::vector<float> actual(
        static_cast<std::size_t>(batchHeads * seqQ * HEAD_DIM));
    for (int bh = 0; bh < batchHeads; ++bh) {
        for (int row = 0; row < seqQ; ++row) {
            for (int d = 0; d < HEAD_DIM; ++d) {
                const float2 packed = output.at(
                    Point{d / 2, bh * seqQ + row, 0});
                actual[static_cast<std::size_t>(
                    (bh * seqQ + row) * HEAD_DIM + d)] =
                    d & 1 ? packed.y : packed.x;
            }
        }
    }

    double maxError = 0.0;
    int maxBh = 0;
    int maxRow = 0;
    int maxD = 0;
    double maxExpected = 0.0;
    double maxActual = 0.0;
    std::vector<double> logits(static_cast<std::size_t>(seqK));
    for (int bh = 0; bh < batchHeads; ++bh) {
        for (int row = 0; row < seqQ; ++row) {
            const int keyEnd = causal ? std::min(seqK, row + 1) : seqK;
            double rowMax = -1e300;
            for (int keyRow = 0; keyRow < keyEnd; ++keyRow) {
                double dot = 0.0;
                for (int d = 0; d < HEAD_DIM; ++d) {
                    const double qv = qSource(bh, row, d) * Q_SCALE;
                    const double kv = kSource(bh, keyRow, d) + K_BIAS;
                    dot += qv * kv;
                }
                logits[static_cast<std::size_t>(keyRow)] = dot * scale;
                rowMax = std::max(rowMax, dot * scale);
            }
            double denominator = 0.0;
            for (int keyRow = 0; keyRow < keyEnd; ++keyRow) {
                auto& probability = logits[static_cast<std::size_t>(keyRow)];
                probability = std::exp(probability - rowMax);
                denominator += probability;
            }
            for (int d = 0; d < HEAD_DIM; ++d) {
                double expected = 0.0;
                for (int keyRow = 0; keyRow < keyEnd; ++keyRow) {
                    expected += logits[static_cast<std::size_t>(keyRow)] *
                                vSource(bh, keyRow, d) * V_SCALE;
                }
                expected = (expected / denominator) * O_SCALE + O_BIAS;
                const double value = actual[static_cast<std::size_t>(
                    (bh * seqQ + row) * HEAD_DIM + d)];
                const double error = std::abs(value - expected);
                if (error > maxError) {
                    maxError = error;
                    maxBh = bh;
                    maxRow = row;
                    maxD = d;
                    maxExpected = expected;
                    maxActual = value;
                }
            }
        }
    }
    const double tolerance = GPU ? 3.0e-2 : 2.0e-5;
    if constexpr (GPU) {
        if (maxError > tolerance) {
            std::printf("max mismatch bh=%d row=%d d=%d expected=%g actual=%g\n",
                        maxBh, maxRow, maxD, maxExpected, maxActual);
        }
    }
    return {maxError <= tolerance, std::move(actual), maxError};
}

template <int HEAD_DIM>
bool verifyShape(const int batchHeads, const int seqQ, const int seqK,
                 const bool causal) {
    const auto cpu = runCase<ParArch::CPU, HEAD_DIM>(
        batchHeads, seqQ, seqK, causal);
    bool ok = cpu.oraclePassed;
#if defined(__NVCC__)
    const auto gpu = runCase<ParArch::GPU_NVIDIA, HEAD_DIM>(
        batchHeads, seqQ, seqK, causal);
    double directError = 0.0;
    for (std::size_t i = 0; i < cpu.output.size(); ++i)
        directError = std::max(
            directError,
            std::abs(static_cast<double>(cpu.output[i]) - gpu.output[i]));
    ok = gpu.oraclePassed && directError <= 3.0e-2 && ok;
    std::printf(
        "Attention D=%d BH=%d Q=%d K=%d causal=%d cpu=%g gpu=%g direct=%g %s\n",
        HEAD_DIM, batchHeads, seqQ, seqK, causal,
        cpu.maxError, gpu.maxError, directError, ok ? "PASS" : "FAIL");
#else
    std::printf(
        "Attention CPU D=%d BH=%d Q=%d K=%d causal=%d error=%g %s\n",
        HEAD_DIM, batchHeads, seqQ, seqK, causal,
        cpu.maxError, ok ? "PASS" : "FAIL");
#endif
    return ok;
}

} // namespace

int launch() {
    bool ok = true;
    ok = verifyShape<32>(1, 7, 5, false) && ok;
    ok = verifyShape<32>(2, 35, 19, true) && ok;
    ok = verifyShape<64>(2, 33, 41, false) && ok;
    ok = verifyShape<64>(1, 17, 17, true) && ok;
    return ok ? 0 : -1;
}
