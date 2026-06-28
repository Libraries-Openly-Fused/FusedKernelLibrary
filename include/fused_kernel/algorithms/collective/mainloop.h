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

#ifndef FK_COLLECTIVE_MAINLOOP_H
#define FK_COLLECTIVE_MAINLOOP_H

#include <fused_kernel/algorithms/collective/async_copy.h>
#include <fused_kernel/algorithms/collective/mma.h>
#include <fused_kernel/algorithms/collective/tile.h>
#include <fused_kernel/core/data/tuple.h>
#include <fused_kernel/core/execution_model/operation_model/operation_model.h>
#include <fused_kernel/core/utils/utils.h>

#include <type_traits>

namespace fk {

/**
 * Static mainloop policy plus runtime matrix origins and K extent.
 *
 * The MMA policy is part of Details so the DPP itself has only the canonical
 * <ParArch, Details> template shape. A/B and D remain real exec-time IOps.
 */
template <typename MmaDetails,
          template <ParArch, typename> class MmaPolicyTemplate = MmaWarpDPP>
struct TileMmaMainloopDPPDetails {
    using MmaDetailsType = MmaDetails;
    template <ParArch PA>
    using MmaPolicy = MmaPolicyTemplate<PA, MmaDetails>;

    static constexpr int M = MmaDetails::M;
    static constexpr int N = MmaDetails::N;
    static constexpr int K_TILE = MmaDetails::K;

    int K;
    int aOriginX;
    int aOriginY;
    int bOriginX;
    int bOriginY;
    int dOriginX;
    int dOriginY;
};

template <ParArch PA, typename DPPDetails>
struct TileMmaMainloopDPP;

template <typename DPPDetails>
struct TileMmaMainloopDPP<ParArch::CPU, DPPDetails> {
private:
    using SelfType = TileMmaMainloopDPP<ParArch::CPU, DPPDetails>;
    using MmaDetails = typename DPPDetails::MmaDetailsType;
    using MmaPolicy = typename DPPDetails::template MmaPolicy<ParArch::CPU>;

public:
    FK_STATIC_STRUCT(TileMmaMainloopDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::CPU;

    template <typename ReadIOps, typename WriteIOp>
    FK_HOST_STATIC void exec(const DPPDetails& details,
                             const ReadIOps& reads,
                             const WriteIOp& output) {
        static_assert(isTuple_v<ReadIOps> && std::decay_t<ReadIOps>::size == 2,
                      "TileMmaMainloopDPP requires Tuple<AReadIOp, BReadIOp>");
        static_assert(isAnyWriteType<WriteIOp>,
                      "TileMmaMainloopDPP requires a Write IOp");
        if (details.K <= 0 || details.K % DPPDetails::K_TILE != 0) return;

        const auto& a = get<0>(reads);
        const auto& b = get<1>(reads);
        typename MmaPolicy::AccumulatorFragment fragment;
        MmaPolicy::clear(fragment);
        for (int kBase = 0; kBase < details.K;
             kBase += DPPDetails::K_TILE) {
            const MmaDetails mmaDetails{
                details.aOriginX + kBase, details.aOriginY,
                details.bOriginX + kBase, details.bOriginY,
                details.dOriginX, details.dOriginY};
            MmaPolicy::accumulate(mmaDetails, a, b, fragment);
        }
        const MmaDetails outputDetails{
            details.aOriginX, details.aOriginY,
            details.bOriginX, details.bOriginY,
            details.dOriginX, details.dOriginY};
        MmaPolicy::store(outputDetails, fragment, output);
    }
};

#if defined(__NVCC__)
template <typename DPPDetails>
struct TileMmaMainloopDPP<ParArch::GPU_NVIDIA, DPPDetails> {
private:
    using SelfType = TileMmaMainloopDPP<ParArch::GPU_NVIDIA, DPPDetails>;
    using MmaDetails = typename DPPDetails::MmaDetailsType;
    using MmaPolicy =
        typename DPPDetails::template MmaPolicy<ParArch::GPU_NVIDIA>;

public:
    FK_STATIC_STRUCT(TileMmaMainloopDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::GPU_NVIDIA;

    template <typename ReadIOps, typename WriteIOp>
    FK_DEVICE_STATIC void exec(const DPPDetails& details,
                               const ReadIOps& reads,
                               const WriteIOp& output) {
        static_assert(isTuple_v<ReadIOps> && std::decay_t<ReadIOps>::size == 2,
                      "TileMmaMainloopDPP requires Tuple<AReadIOp, BReadIOp>");
        static_assert(isAnyWriteType<WriteIOp>,
                      "TileMmaMainloopDPP requires a Write IOp");
        if (details.K <= 0 || details.K % DPPDetails::K_TILE != 0) return;

        const auto& a = get<0>(reads);
        const auto& b = get<1>(reads);
        typename MmaPolicy::AccumulatorFragment fragment;
        MmaPolicy::clear(fragment);
        for (int kBase = 0; kBase < details.K;
             kBase += DPPDetails::K_TILE) {
            const MmaDetails mmaDetails{
                details.aOriginX + kBase, details.aOriginY,
                details.bOriginX + kBase, details.bOriginY,
                details.dOriginX, details.dOriginY};
            MmaPolicy::accumulate(mmaDetails, a, b, fragment);
        }
        const MmaDetails outputDetails{
            details.aOriginX, details.aOriginY,
            details.bOriginX, details.bOriginY,
            details.dOriginX, details.dOriginY};
        MmaPolicy::store(outputDetails, fragment, output);
    }
};
#endif // defined(__NVCC__)

/**
 * Fixed two-buffer software pipeline policy. Copy and MMA policies are owned by
 * Details; exec still receives only real A/B Read IOps and the D Write IOp.
 */
template <typename MmaDetails, typename ALayout, typename BLayout,
          typename StageT = float,
          template <ParArch, typename> class MmaPolicyTemplate = MmaWarpDPP,
          template <ParArch, typename> class CopyPolicyTemplate = AsyncCopyDPP>
struct TileMmaMainloopPipelinedDPPDetails {
    using MmaDetailsType = MmaDetails;
    using ALayoutType = ALayout;
    using BLayoutType = BLayout;
    using StageType = StageT;
    using ACopyDetails = AsyncCopy2DDPPDetails<StageT, ALayout, 32>;
    using BCopyDetails = AsyncCopy2DDPPDetails<StageT, BLayout, 32>;

    template <ParArch PA>
    using MmaPolicy = MmaPolicyTemplate<PA, MmaDetails>;
    template <ParArch PA>
    using ACopyPolicy = CopyPolicyTemplate<PA, ACopyDetails>;
    template <ParArch PA>
    using BCopyPolicy = CopyPolicyTemplate<PA, BCopyDetails>;

    static constexpr int M = MmaDetails::M;
    static constexpr int N = MmaDetails::N;
    static constexpr int K_TILE = MmaDetails::K;
    static constexpr int BUFFER_COUNT = 2;

    static_assert(ALayout::rows == M && ALayout::cols == K_TILE,
                  "A staging layout must be M x K_TILE");
    static_assert(BLayout::rows == N && BLayout::cols == K_TILE,
                  "B staging layout must be N x K_TILE");

    int K;
    int aOriginX;
    int aOriginY;
    int bOriginX;
    int bOriginY;
    int dOriginX;
    int dOriginY;
};

namespace mainloop_detail {

template <typename T, typename Layout>
struct SharedTileRead {
private:
    using Parent = ReadOperation<T, Tile<T, Layout>, T, TF::ENABLED,
                                 SharedTileRead<T, Layout>>;
    using SelfType = SharedTileRead<T, Layout>;

public:
    FK_STATIC_STRUCT(SharedTileRead, SelfType)
    DECLARE_READ_PARENT

    template <uint ELEMS_PER_THREAD = 1>
    FK_HOST_DEVICE_FUSE auto exec(const Point point,
                                  const ParamsType& tile)
        -> ThreadFusionType<T, ELEMS_PER_THREAD, T> {
        static_assert(ELEMS_PER_THREAD == 1,
                      "Shared MMA fragments read one scalar per access");
        return tile.at(static_cast<uint>(point.y),
                       static_cast<uint>(point.x));
    }
};

} // namespace mainloop_detail

template <ParArch PA, typename DPPDetails>
struct TileMmaMainloopPipelinedDPP;

template <typename DPPDetails>
struct TileMmaMainloopPipelinedDPP<ParArch::CPU, DPPDetails> {
private:
    using SelfType =
        TileMmaMainloopPipelinedDPP<ParArch::CPU, DPPDetails>;
    using T = typename DPPDetails::StageType;

public:
    FK_STATIC_STRUCT(TileMmaMainloopPipelinedDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::CPU;

    template <typename ReadIOps, typename WriteIOp>
    FK_HOST_STATIC void exec(const DPPDetails& details,
                             const ReadIOps& reads,
                             const WriteIOp& output) {
        static_assert(isTuple_v<ReadIOps> &&
                      std::decay_t<ReadIOps>::size == 2,
                      "Pipelined mainloop requires Tuple<AReadIOp, BReadIOp>");
        static_assert(isAnyWriteType<WriteIOp>,
                      "Pipelined mainloop requires a Write IOp");
        if (details.K <= 0) return;
        const auto& a = get<0>(reads);
        const auto& b = get<1>(reads);
        for (int row = 0; row < DPPDetails::M; ++row) {
            for (int pair = 0; pair < DPPDetails::N / 2; ++pair) {
                T values[2] = {T{}, T{}};
                for (int item = 0; item < 2; ++item) {
                    const int col = 2 * pair + item;
                    T accumulator{};
                    for (int k = 0; k < details.K; ++k) {
                        const T av = static_cast<T>(
                            std::decay_t<decltype(a)>::Operation::exec(
                                Point{details.aOriginX + k,
                                      details.aOriginY + row, 0}, a));
                        const T bv = static_cast<T>(
                            std::decay_t<decltype(b)>::Operation::exec(
                                Point{details.bOriginX + k,
                                      details.bOriginY + col, 0}, b));
                        accumulator += av * bv;
                    }
                    values[item] = accumulator;
                }
                WriteIOp::Operation::exec(
                    Point{details.dOriginX + pair,
                          details.dOriginY + row, 0},
                    float2{static_cast<float>(values[0]),
                           static_cast<float>(values[1])}, output);
            }
        }
    }
};

#if defined(__NVCC__)
template <typename DPPDetails>
struct TileMmaMainloopPipelinedDPP<ParArch::GPU_NVIDIA, DPPDetails> {
private:
    using SelfType = TileMmaMainloopPipelinedDPP<
        ParArch::GPU_NVIDIA, DPPDetails>;
    using T = typename DPPDetails::StageType;
    using MmaDetails = typename DPPDetails::MmaDetailsType;
    using ALayout = typename DPPDetails::ALayoutType;
    using BLayout = typename DPPDetails::BLayoutType;
    using MmaPolicy =
        typename DPPDetails::template MmaPolicy<ParArch::GPU_NVIDIA>;
    using ACopyPolicy =
        typename DPPDetails::template ACopyPolicy<ParArch::GPU_NVIDIA>;
    using BCopyPolicy =
        typename DPPDetails::template BCopyPolicy<ParArch::GPU_NVIDIA>;
    using ACopyDetails = typename DPPDetails::ACopyDetails;
    using BCopyDetails = typename DPPDetails::BCopyDetails;
    using ASharedOperation = mainloop_detail::SharedTileRead<T, ALayout>;
    using BSharedOperation = mainloop_detail::SharedTileRead<T, BLayout>;
    using ASharedRead = typename ASharedOperation::InstantiableType;
    using BSharedRead = typename BSharedOperation::InstantiableType;

public:
    FK_STATIC_STRUCT(TileMmaMainloopPipelinedDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::GPU_NVIDIA;

    template <typename ReadIOps, typename WriteIOp>
    FK_DEVICE_STATIC void exec(const DPPDetails& details,
                               const ReadIOps& reads,
                               const WriteIOp& output) {
        static_assert(isTuple_v<ReadIOps> &&
                      std::decay_t<ReadIOps>::size == 2,
                      "Pipelined mainloop requires Tuple<AReadIOp, BReadIOp>");
        static_assert(isAnyWriteType<WriteIOp>,
                      "Pipelined mainloop requires a Write IOp");
        if (details.K <= 0) return;

        __shared__ T aStorage[DPPDetails::BUFFER_COUNT][ALayout::size()];
        __shared__ T bStorage[DPPDetails::BUFFER_COUNT][BLayout::size()];
        const auto& aInput = get<0>(reads);
        const auto& bInput = get<1>(reads);
        const int tileCount =
            (details.K + DPPDetails::K_TILE - 1) / DPPDetails::K_TILE;

        typename MmaPolicy::AccumulatorFragment fragment;
        MmaPolicy::clear(fragment);

        auto issueTile = [&](const int tileIndex, const int buffer) {
            const int kBase = tileIndex * DPPDetails::K_TILE;
            int validK = details.K - kBase;
            validK = validK > DPPDetails::K_TILE
                ? DPPDetails::K_TILE : validK;
            const ACopyDetails aCopy{
                details.aOriginX + kBase, details.aOriginY,
                validK, DPPDetails::M, T{}};
            const BCopyDetails bCopy{
                details.bOriginX + kBase, details.bOriginY,
                validK, DPPDetails::N, T{}};
            ACopyPolicy::issue(aCopy, aInput,
                               Tile<T, ALayout>{aStorage[buffer]});
            BCopyPolicy::issue(bCopy, bInput,
                               Tile<T, BLayout>{bStorage[buffer]});
            ACopyPolicy::commit();
        };

        issueTile(0, 0);
        for (int tileIndex = 0; tileIndex < tileCount; ++tileIndex) {
            const int current = tileIndex & 1;
            const int next = current ^ 1;
            const bool hasNext = tileIndex + 1 < tileCount;

            // The current stage is the only committed group at loop entry.
            // Wait for it, make its tail visible, then launch the next stage.
            // That next cp.async group overlaps the MMA on the current buffer.
            ACopyPolicy::template wait<0>();

            const int kBase = tileIndex * DPPDetails::K_TILE;
            int validK = details.K - kBase;
            validK = validK > DPPDetails::K_TILE
                ? DPPDetails::K_TILE : validK;
            const ACopyDetails aCopy{
                details.aOriginX + kBase, details.aOriginY,
                validK, DPPDetails::M, T{}};
            const BCopyDetails bCopy{
                details.bOriginX + kBase, details.bOriginY,
                validK, DPPDetails::N, T{}};
            const Tile<T, ALayout> aTile{aStorage[current]};
            const Tile<T, BLayout> bTile{bStorage[current]};
            ACopyPolicy::complete(aCopy, aTile);
            BCopyPolicy::complete(bCopy, bTile);
            __syncwarp();
            if (hasNext) issueTile(tileIndex + 1, next);

            const ASharedRead aShared{{aTile}};
            const BSharedRead bShared{{bTile}};
            const MmaDetails mmaDetails{
                0, 0, 0, 0, details.dOriginX, details.dOriginY};
            MmaPolicy::accumulate(
                mmaDetails, aShared, bShared, fragment);
        }
        const MmaDetails outputDetails{
            0, 0, 0, 0, details.dOriginX, details.dOriginY};
        MmaPolicy::store(outputDetails, fragment, output);
    }
};
#endif // defined(__NVCC__)

} // namespace fk

#endif // FK_COLLECTIVE_MAINLOOP_H
