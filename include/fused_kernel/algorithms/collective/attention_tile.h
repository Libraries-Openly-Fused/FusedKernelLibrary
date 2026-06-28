/* Copyright 2026 Oscar Amoros Huguet, Johnny Nunez

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_COLLECTIVE_ATTENTION_TILE_H
#define FK_COLLECTIVE_ATTENTION_TILE_H

#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/logical.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/collective/copy.h>
#include <fused_kernel/algorithms/collective/cta_raster.h>
#include <fused_kernel/algorithms/collective/mma.h>
#include <fused_kernel/algorithms/collective/reduce.h>
#include <fused_kernel/core/execution_model/executor_details/dpp_launch_config.h>

#include <cmath>
#include <limits>
#include <type_traits>

#if defined(__NVCC__)
#include <cuda_bf16.h>
#include <cfloat>
#endif

namespace fk {

/**
 * Static attention policy plus runtime launch state.
 *
 * Q/K/V are supplied as one Tuple of Read IOps. O is supplied only as the
 * final Write IOp. qTile and bh are per-block fields updated internally by
 * AttentionTileDPP after applying SchedulerOperation.
 */
template <
    int HEAD_DIMENSION,
    int QUERIES_PER_TILE,
    typename MmaDetailsType = MmaDPPDetails<MmaBf16_16x8x16>,
    typename QStageLayoutType =
        RowMajorLayout<QUERIES_PER_TILE, HEAD_DIMENSION>,
    typename KStageLayoutType = RowMajorLayout<16, HEAD_DIMENSION>,
    typename VStageLayoutType = ColumnMajorLayout<16, HEAD_DIMENSION>,
    typename PStageLayoutType = RowMajorLayout<16, 16>,
    typename SchedulerOperationT = CtaTileScheduler<RowMajorCtaTileRaster>,
    template <ParArch, typename> class MmaPolicyTemplate = MmaWarpDPP,
    template <ParArch, typename> class CopyPolicyTemplate = CopyTileDPP,
    template <ParArch, typename> class ReducePolicyTemplate = ReduceWarpDPP>
struct AttentionDPPDetails {
    static_assert(HEAD_DIMENSION == 32 || HEAD_DIMENSION == 64,
                  "AttentionTileDPP supports head dimensions 32 and 64");
    static_assert(QUERIES_PER_TILE > 0 && QUERIES_PER_TILE % 16 == 0,
                  "query tile must contain whole m16 MMA atoms");

    static constexpr int HEAD_DIM = HEAD_DIMENSION;
    static constexpr int QUERY_ROWS = QUERIES_PER_TILE;
    static constexpr int KEY_ROWS = 16;
    static constexpr int BLOCK_THREADS = 32;
    static constexpr int QUERY_ATOMS = QUERY_ROWS / 16;
    static constexpr int OUTPUT_ATOMS = HEAD_DIM / 8;

    using MmaDetails = MmaDetailsType;
    using QStageLayout = QStageLayoutType;
    using KStageLayout = KStageLayoutType;
    using VStageLayout = VStageLayoutType;
    using PStageLayout = PStageLayoutType;
    using SchedulerOperation = SchedulerOperationT;
    using SchedulerIOp = decltype(SchedulerOperation::build());
    using ReduceDetails = ReduceDPPDetails<float, BLOCK_THREADS>;

    template <typename Layout>
    using CopyDetails = CopyTileDPPDetails<float, Layout, BLOCK_THREADS>;
    template <ParArch PA, typename Details>
    using MmaPolicy = MmaPolicyTemplate<PA, Details>;
    template <ParArch PA, typename Details>
    using CopyPolicy = CopyPolicyTemplate<PA, Details>;
    template <ParArch PA, typename Details>
    using ReducePolicy = ReducePolicyTemplate<PA, Details>;

    int batchHeads;
    int seqQ;
    int seqK;
    float scale;
    bool causal;
    int qTile;
    int bh;

    FK_HOST_DEVICE_CNST bool valid() const {
        return batchHeads > 0 && seqQ > 0 && seqK > 0 && scale > 0.f;
    }

    FK_HOST_DEVICE_CNST int queryTileCount() const {
        return (seqQ + QUERY_ROWS - 1) / QUERY_ROWS;
    }
};

template <ParArch PA, typename AttentionDetailsType>
struct AttentionTileDPP;

template <typename AttentionDetailsType>
struct AttentionTileDPP<ParArch::CPU, AttentionDetailsType> {
private:
    using SelfType = AttentionTileDPP<ParArch::CPU, AttentionDetailsType>;
    using Details = AttentionDetailsType;

    template <typename Inputs, typename Output, typename... ComputeIOps>
    FK_HOST_STATIC void executeBlock(const Details& details,
                                     const Inputs& inputs,
                                     const Output& output,
                                     const ComputeIOps&...) {
        const auto& q = get<0>(inputs);
        const auto& k = get<1>(inputs);
        const auto& v = get<2>(inputs);
        using Q = std::decay_t<decltype(q)>;
        using K = std::decay_t<decltype(k)>;
        using V = std::decay_t<decltype(v)>;

        const int qRow0 = details.qTile * Details::QUERY_ROWS;
        const int qHead = details.bh * details.seqQ;
        const int kHead = details.bh * details.seqK;
        for (int localRow = 0;
             localRow < Details::QUERY_ROWS && qRow0 + localRow < details.seqQ;
             ++localRow) {
            const int queryRow = qRow0 + localRow;
            const int keyEnd = details.causal
                ? std::min(details.seqK, queryRow + 1) : details.seqK;
            double rowMax = -std::numeric_limits<double>::infinity();
            for (int keyRow = 0; keyRow < keyEnd; ++keyRow) {
                double dot = 0.0;
                for (int d = 0; d < Details::HEAD_DIM; ++d) {
                    dot += static_cast<double>(Q::Operation::exec(
                               Point{d, qHead + queryRow, 0}, q)) *
                           static_cast<double>(K::Operation::exec(
                               Point{d, kHead + keyRow, 0}, k));
                }
                rowMax = std::max(rowMax, dot * details.scale);
            }
            double denominator = 0.0;
            for (int keyRow = 0; keyRow < keyEnd; ++keyRow) {
                double dot = 0.0;
                for (int d = 0; d < Details::HEAD_DIM; ++d) {
                    dot += static_cast<double>(Q::Operation::exec(
                               Point{d, qHead + queryRow, 0}, q)) *
                           static_cast<double>(K::Operation::exec(
                               Point{d, kHead + keyRow, 0}, k));
                }
                denominator += std::exp(dot * details.scale - rowMax);
            }
            const double inverse = denominator > 0.0 ? 1.0 / denominator : 0.0;
            for (int d = 0; d < Details::HEAD_DIM; d += 2) {
                double first = 0.0;
                double second = 0.0;
                for (int keyRow = 0; keyRow < keyEnd; ++keyRow) {
                    double dot = 0.0;
                    for (int inner = 0; inner < Details::HEAD_DIM; ++inner) {
                        dot += static_cast<double>(Q::Operation::exec(
                                   Point{inner, qHead + queryRow, 0}, q)) *
                               static_cast<double>(K::Operation::exec(
                                   Point{inner, kHead + keyRow, 0}, k));
                    }
                    const double probability =
                        std::exp(dot * details.scale - rowMax) * inverse;
                    first += probability * static_cast<double>(V::Operation::exec(
                        Point{d, kHead + keyRow, 0}, v));
                    second += probability * static_cast<double>(V::Operation::exec(
                        Point{d + 1, kHead + keyRow, 0}, v));
                }
                Output::Operation::exec(
                    Point{d / 2, qHead + queryRow, 0},
                    float2{static_cast<float>(first),
                           static_cast<float>(second)}, output);
            }
        }
    }

public:
    FK_STATIC_STRUCT(AttentionTileDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::CPU;

    FK_HOST_FUSE DPPLaunchConfig launchConfig(const Details&) { return {}; }

    template <typename Inputs, typename Output, typename... ComputeIOps>
    FK_HOST_STATIC void exec(const Details& launchDetails,
                             const Inputs& inputs,
                             const Output& output,
                             const ComputeIOps&... computeIOps) {
        if (!launchDetails.valid()) return;
        const int qTiles = launchDetails.queryTileCount();
        const typename Details::SchedulerIOp scheduler{};
        for (int block = 0; block < launchDetails.batchHeads * qTiles; ++block) {
            const CtaTileAssignment tile =
                make_tuple(block, launchDetails.batchHeads, qTiles) | scheduler;
            if (!tile.valid) continue;
            Details details = launchDetails;
            details.bh = tile.mTile;
            details.qTile = tile.nTile;
            executeBlock(details, inputs, output, computeIOps...);
        }
    }
};

#if defined(__NVCC__)
template <typename AttentionDetailsType>
struct AttentionTileDPP<ParArch::GPU_NVIDIA, AttentionDetailsType> {
private:
    using SelfType = AttentionTileDPP<ParArch::GPU_NVIDIA, AttentionDetailsType>;
    using Details = AttentionDetailsType;
    using Mma = typename Details::template MmaPolicy<
        ParArch::GPU_NVIDIA, typename Details::MmaDetails>;
    using Reduce = typename Details::template ReducePolicy<
        ParArch::GPU_NVIDIA, typename Details::ReduceDetails>;
    using AddOp = Add<float, float, float, UnaryType>;
    using MaxOp = Max<float, float, float, UnaryType>;

    template <typename T>
    FK_DEVICE_FUSE RawPtr<ND::_2D, T> sharedView(
            T* storage, const uint width, const uint height) {
        RawPtr<ND::_2D, T> result;
        result.data = storage;
        result.dims.width = width;
        result.dims.height = height;
        result.dims.pitch = width * sizeof(T);
        return result;
    }

    FK_DEVICE_FUSE float rowScore(
            const typename Mma::AccumulatorFragment& low,
            const typename Mma::AccumulatorFragment& high,
            const int row, const int lane) {
        const int ownerGroup = row & 7;
        const bool upperRow = row >= 8;
        const int sourceLane = ownerGroup * 4 + ((lane & 7) >> 1);
        const float lowEven = upperRow ? low.values[2] : low.values[0];
        const float lowOdd = upperRow ? low.values[3] : low.values[1];
        const float highEven = upperRow ? high.values[2] : high.values[0];
        const float highOdd = upperRow ? high.values[3] : high.values[1];
        const bool highKey = lane >= 8;
        const float shuffledLowEven = __shfl_sync(
            0xffffffffu, lowEven, sourceLane);
        const float shuffledLowOdd = __shfl_sync(
            0xffffffffu, lowOdd, sourceLane);
        const float shuffledHighEven = __shfl_sync(
            0xffffffffu, highEven, sourceLane);
        const float shuffledHighOdd = __shfl_sync(
            0xffffffffu, highOdd, sourceLane);
        const float even = highKey ? shuffledHighEven : shuffledLowEven;
        const float odd = highKey ? shuffledHighOdd : shuffledLowOdd;
        return lane & 1 ? odd : even;
    }

public:
    FK_STATIC_STRUCT(AttentionTileDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::GPU_NVIDIA;

    FK_HOST_FUSE DPPLaunchConfig launchConfig(const Details& details) {
        const unsigned int blocks = details.valid()
            ? static_cast<unsigned int>(
                  details.batchHeads * details.queryTileCount())
            : 1u;
        return {blocks, 1, 1,
                static_cast<unsigned int>(Details::BLOCK_THREADS), 1, 1, 0};
    }

    template <typename Inputs, typename Output, typename... ComputeIOps>
    static __device__ __forceinline__ void exec(
            const Details& launchDetails,
            const Inputs& inputs,
            const Output& output,
            const ComputeIOps&...) {
        if (!launchDetails.valid()) return;
        const int qTiles = launchDetails.queryTileCount();
        const typename Details::SchedulerIOp scheduler{};
        const CtaTileAssignment tile =
            make_tuple(static_cast<int>(blockIdx.x),
                       launchDetails.batchHeads, qTiles) | scheduler;
        if (!tile.valid) return;
        Details details = launchDetails;
        details.bh = tile.mTile;
        details.qTile = tile.nTile;

        const auto& qInput = get<0>(inputs);
        const auto& kInput = get<1>(inputs);
        const auto& vInput = get<2>(inputs);
        const int qRow0 = details.qTile * Details::QUERY_ROWS;
        const int qHead = details.bh * details.seqQ;
        const int kHead = details.bh * details.seqK;
        const int lane = static_cast<int>(threadIdx.x) & 31;

        __shared__ float qStorage[Details::QStageLayout::size()];
        __shared__ float kStorage[Details::KStageLayout::size()];
        __shared__ float vStorage[Details::VStageLayout::size()];
        __shared__ float probabilities[Details::PStageLayout::size()];
        __shared__ float outputStorage[
            Details::QUERY_ROWS * Details::HEAD_DIM];
        float runningMax[Details::QUERY_ROWS];
        float runningSum[Details::QUERY_ROWS];

        for (int index = lane;
             index < Details::QUERY_ROWS * Details::HEAD_DIM;
             index += Details::BLOCK_THREADS) {
            outputStorage[index] = 0.f;
        }
        #pragma unroll
        for (int row = 0; row < Details::QUERY_ROWS; ++row) {
            runningMax[row] = -FLT_MAX;
            runningSum[row] = 0.f;
        }
        __syncthreads();

        Tile<float, typename Details::QStageLayout> qTile(qStorage);
        Tile<float, typename Details::KStageLayout> kTile(kStorage);
        Tile<float, typename Details::VStageLayout> vTile(vStorage);
        Tile<float, typename Details::PStageLayout> pTile(probabilities);

        using QCopyDetails = typename Details::template CopyDetails<
            typename Details::QStageLayout>;
        using KCopyDetails = typename Details::template CopyDetails<
            typename Details::KStageLayout>;
        using VCopyDetails = typename Details::template CopyDetails<
            typename Details::VStageLayout>;
        using QCopy = typename Details::template CopyPolicy<
            ParArch::GPU_NVIDIA, QCopyDetails>;
        using KCopy = typename Details::template CopyPolicy<
            ParArch::GPU_NVIDIA, KCopyDetails>;
        using VCopy = typename Details::template CopyPolicy<
            ParArch::GPU_NVIDIA, VCopyDetails>;

        QCopy::load(QCopyDetails{
                        0, qHead + qRow0,
                        Details::HEAD_DIM, qHead + details.seqQ, 0.f},
                    qInput, qTile);

        const auto qRead = typename PerThreadRead<ND::_2D, float>::InstantiableType{
            {sharedView(qStorage, Details::HEAD_DIM, Details::QUERY_ROWS)}};
        const auto kRead = typename PerThreadRead<ND::_2D, float>::InstantiableType{
            {sharedView(kStorage, Details::HEAD_DIM, Details::KEY_ROWS)}};

        const typename MaxOp::InstantiableType maxOp{};
        const typename AddOp::InstantiableType addOp{};
        const typename Details::ReduceDetails maxReduction{
            1, 16, -FLT_MAX, 0};
        const typename Details::ReduceDetails sumReduction{1, 16, 0.f, 0};

        for (int key0 = 0; key0 < details.seqK;
             key0 += Details::KEY_ROWS) {
            KCopy::load(KCopyDetails{
                            0, kHead + key0,
                            Details::HEAD_DIM, kHead + details.seqK, 0.f},
                        kInput, kTile);
            VCopy::load(VCopyDetails{
                            0, kHead + key0,
                            Details::HEAD_DIM, kHead + details.seqK, 0.f},
                        vInput, vTile);

            #pragma unroll
            for (int queryAtom = 0;
                 queryAtom < Details::QUERY_ATOMS; ++queryAtom) {
                typename Mma::AccumulatorFragment scoresLow;
                typename Mma::AccumulatorFragment scoresHigh;
                Mma::clear(scoresLow);
                Mma::clear(scoresHigh);
                #pragma unroll
                for (int d = 0; d < Details::HEAD_DIM;
                     d += Details::MmaDetails::K) {
                    Mma::accumulate(
                        typename Details::MmaDetails{
                            d, queryAtom * 16, d, 0, 0, 0},
                        qRead, kRead, scoresLow);
                    Mma::accumulate(
                        typename Details::MmaDetails{
                            d, queryAtom * 16, d, 8, 0, 0},
                        qRead, kRead, scoresHigh);
                }

                #pragma unroll
                for (int row = 0; row < 16; ++row) {
                    const int queryRow = qRow0 + queryAtom * 16 + row;
                    const float rawScore =
                        rowScore(scoresLow, scoresHigh, row, lane) *
                        details.scale;
                    float score = lane < 16 ? rawScore : -FLT_MAX;
                    const int keyRow = key0 + lane;
                    if (lane >= 16 || keyRow >= details.seqK ||
                        queryRow >= details.seqQ ||
                        (details.causal && keyRow > queryRow)) {
                        score = -FLT_MAX;
                    }
                    float tileMax = Reduce::exec(maxReduction, score, maxOp);
                    tileMax = __shfl_sync(0xffffffffu, tileMax, 0);

                    const int localRow = queryAtom * 16 + row;
                    const float oldMax = runningMax[localRow];
                    const float oldSum = runningSum[localRow];
                    const float newMax = fmaxf(oldMax, tileMax);
                    const float probability = lane < 16 && score > -FLT_MAX
                        ? __expf(score - newMax) : 0.f;
                    float tileSum = Reduce::exec(
                        sumReduction, probability, addOp);
                    tileSum = __shfl_sync(0xffffffffu, tileSum, 0);
                    const float correction = oldMax > -FLT_MAX
                        ? __expf(oldMax - newMax) : 0.f;

                    for (int d = lane; d < Details::HEAD_DIM;
                         d += Details::BLOCK_THREADS) {
                        outputStorage[localRow * Details::HEAD_DIM + d] *=
                            correction;
                    }
                    if (lane < 16)
                        pTile.at(static_cast<uint>(row),
                                 static_cast<uint>(lane)) = probability;
                    runningMax[localRow] = newMax;
                    runningSum[localRow] = oldSum * correction + tileSum;
                    __syncwarp();

                    for (int d = lane; d < Details::HEAD_DIM;
                         d += Details::BLOCK_THREADS) {
                        float accumulator =
                            outputStorage[localRow * Details::HEAD_DIM + d];
                        #pragma unroll
                        for (int key = 0; key < 16; ++key) {
                            accumulator += pTile.at(
                                               static_cast<uint>(row),
                                               static_cast<uint>(key)) *
                                           vTile.at(
                                               static_cast<uint>(key),
                                               static_cast<uint>(d));
                        }
                        outputStorage[localRow * Details::HEAD_DIM + d] =
                            accumulator;
                    }
                    __syncwarp();
                }
            }
        }

        for (int localRow = 0; localRow < Details::QUERY_ROWS; ++localRow) {
            const int queryRow = qRow0 + localRow;
            if (queryRow >= details.seqQ) continue;
            const float inverse = runningSum[localRow] > 0.f
                ? 1.f / runningSum[localRow] : 0.f;
            for (int pair = lane; pair < Details::HEAD_DIM / 2;
                 pair += Details::BLOCK_THREADS) {
                const int d = pair * 2;
                Output::Operation::exec(
                    Point{pair, qHead + queryRow, 0},
                    float2{
                        outputStorage[localRow * Details::HEAD_DIM + d] * inverse,
                        outputStorage[localRow * Details::HEAD_DIM + d + 1] * inverse},
                    output);
            }
        }
    }
};
#endif // defined(__NVCC__)

} // namespace fk

#endif // FK_COLLECTIVE_ATTENTION_TILE_H
