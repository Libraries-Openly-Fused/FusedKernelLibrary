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

#ifndef FK_COLLECTIVE_MULTISTAGE_H
#define FK_COLLECTIVE_MULTISTAGE_H

#include <fused_kernel/algorithms/collective/async_copy.h>
#include <fused_kernel/algorithms/collective/mma.h>
#include <fused_kernel/algorithms/collective/tile.h>
#include <fused_kernel/core/data/tuple.h>
#include <fused_kernel/core/execution_model/operation_model/operation_model.h>
#include <fused_kernel/core/utils/utils.h>

#include <type_traits>

namespace fk {

template <int MAX_STAGE_COUNT, typename MmaDetails,
          typename ALayout, typename BLayout, typename StageT = float,
          template <ParArch, typename> class MmaPolicyTemplate = MmaWarpDPP,
          template <ParArch, typename> class CopyPolicyTemplate = AsyncCopyDPP>
struct MultiStageMainloopDPPDetails {
    static_assert(MAX_STAGE_COUNT >= 2 && MAX_STAGE_COUNT <= 8,
                  "MAX_STAGES must be in [2, 8]");
    static_assert(ALayout::rows == MmaDetails::M &&
                  ALayout::cols == MmaDetails::K,
                  "A stage layout must be M x K_TILE");
    static_assert(BLayout::rows == MmaDetails::N &&
                  BLayout::cols == MmaDetails::K,
                  "B stage layout must be N x K_TILE");

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

    static constexpr int MAX_STAGES = MAX_STAGE_COUNT;
    static constexpr int M = MmaDetails::M;
    static constexpr int N = MmaDetails::N;
    static constexpr int K_TILE = MmaDetails::K;
    static constexpr uint A_STAGE_ELEMENTS = ALayout::size();
    static constexpr uint B_STAGE_ELEMENTS = BLayout::size();
    static constexpr uint SHARED_ELEMENTS =
        MAX_STAGES * (A_STAGE_ELEMENTS + B_STAGE_ELEMENTS);
    static constexpr uint SHARED_BYTES = SHARED_ELEMENTS * sizeof(StageT);

    int stages;
    int K;
    int aOriginX;
    int aOriginY;
    int bOriginX;
    int bOriginY;
    int dOriginX;
    int dOriginY;

    FK_HOST_DEVICE_CNST bool valid() const {
        return stages >= 2 && stages <= MAX_STAGES && K > 0;
    }
};

namespace multistage_detail {

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
                      "MMA shared fragments read one scalar per access");
        return tile.at(static_cast<uint>(point.y),
                       static_cast<uint>(point.x));
    }
};

} // namespace multistage_detail

template <ParArch PA, typename DPPDetails>
struct MultiStageMainloopDPP;

template <typename DPPDetails>
struct MultiStageMainloopDPP<ParArch::CPU, DPPDetails> {
private:
    using SelfType = MultiStageMainloopDPP<ParArch::CPU, DPPDetails>;
    using T = typename DPPDetails::StageType;
    using MmaDetails = typename DPPDetails::MmaDetailsType;
    using ALayout = typename DPPDetails::ALayoutType;
    using BLayout = typename DPPDetails::BLayoutType;
    using MmaPolicy = typename DPPDetails::template MmaPolicy<ParArch::CPU>;
    using ACopyPolicy =
        typename DPPDetails::template ACopyPolicy<ParArch::CPU>;
    using BCopyPolicy =
        typename DPPDetails::template BCopyPolicy<ParArch::CPU>;
    using ACopyDetails = typename DPPDetails::ACopyDetails;
    using BCopyDetails = typename DPPDetails::BCopyDetails;
    using ASharedOperation = multistage_detail::SharedTileRead<T, ALayout>;
    using BSharedOperation = multistage_detail::SharedTileRead<T, BLayout>;
    using ASharedRead = typename ASharedOperation::InstantiableType;
    using BSharedRead = typename BSharedOperation::InstantiableType;

public:
    FK_STATIC_STRUCT(MultiStageMainloopDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::CPU;

    template <typename ReadIOps, typename WriteIOp>
    FK_HOST_STATIC void exec(const DPPDetails& details,
                             const ReadIOps& reads,
                             const WriteIOp& output) {
        static_assert(isTuple_v<ReadIOps> &&
                      std::decay_t<ReadIOps>::size == 2,
                      "MultiStageMainloopDPP requires Tuple<A, B> reads");
        static_assert(isAnyWriteType<WriteIOp>,
                      "MultiStageMainloopDPP requires a Write IOp");
        if (!details.valid()) return;

        T aStorage[DPPDetails::MAX_STAGES][ALayout::size()];
        T bStorage[DPPDetails::MAX_STAGES][BLayout::size()];
        const auto& aInput = get<0>(reads);
        const auto& bInput = get<1>(reads);
        const int tileCount =
            (details.K + DPPDetails::K_TILE - 1) / DPPDetails::K_TILE;
        typename MmaPolicy::AccumulatorFragment accumulator;
        MmaPolicy::clear(accumulator);

        auto stageTile = [&](const int tileIndex, const int slot) {
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
            ACopyPolicy::load(
                aCopy, aInput, Tile<T, ALayout>{aStorage[slot]});
            BCopyPolicy::load(
                bCopy, bInput, Tile<T, BLayout>{bStorage[slot]});
        };

        const int prefetch = tileCount < details.stages - 1
            ? tileCount : details.stages - 1;
        for (int tile = 0; tile < prefetch; ++tile)
            stageTile(tile, tile % details.stages);
        int nextTile = prefetch;
        for (int tile = 0; tile < tileCount; ++tile) {
            const int slot = tile % details.stages;
            const Tile<T, ALayout> aTile{aStorage[slot]};
            const Tile<T, BLayout> bTile{bStorage[slot]};
            const ASharedRead aShared{{aTile}};
            const BSharedRead bShared{{bTile}};
            const MmaDetails mma{0, 0, 0, 0,
                                 details.dOriginX, details.dOriginY};
            MmaPolicy::accumulate(mma, aShared, bShared, accumulator);
            if (nextTile < tileCount) {
                stageTile(nextTile, nextTile % details.stages);
                ++nextTile;
            }
        }
        const MmaDetails outputDetails{
            0, 0, 0, 0, details.dOriginX, details.dOriginY};
        MmaPolicy::store(outputDetails, accumulator, output);
    }
};

#if defined(__NVCC__)
template <typename DPPDetails>
struct MultiStageMainloopDPP<ParArch::GPU_NVIDIA, DPPDetails> {
private:
    using SelfType =
        MultiStageMainloopDPP<ParArch::GPU_NVIDIA, DPPDetails>;
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
    using ASharedOperation = multistage_detail::SharedTileRead<T, ALayout>;
    using BSharedOperation = multistage_detail::SharedTileRead<T, BLayout>;
    using ASharedRead = typename ASharedOperation::InstantiableType;
    using BSharedRead = typename BSharedOperation::InstantiableType;

    FK_DEVICE_STATIC void waitForPending(const int pending) {
        switch (pending) {
            case 0: ACopyPolicy::template wait<0>(); break;
            case 1: ACopyPolicy::template wait<1>(); break;
            case 2: ACopyPolicy::template wait<2>(); break;
            case 3: ACopyPolicy::template wait<3>(); break;
            case 4: ACopyPolicy::template wait<4>(); break;
            case 5: ACopyPolicy::template wait<5>(); break;
            case 6: ACopyPolicy::template wait<6>(); break;
            default: ACopyPolicy::template wait<7>(); break;
        }
    }

public:
    FK_STATIC_STRUCT(MultiStageMainloopDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::GPU_NVIDIA;

    template <typename ReadIOps, typename WriteIOp>
    FK_DEVICE_STATIC void exec(const DPPDetails& details,
                               const ReadIOps& reads,
                               const WriteIOp& output) {
        static_assert(isTuple_v<ReadIOps> &&
                      std::decay_t<ReadIOps>::size == 2,
                      "MultiStageMainloopDPP requires Tuple<A, B> reads");
        static_assert(isAnyWriteType<WriteIOp>,
                      "MultiStageMainloopDPP requires a Write IOp");
        if (!details.valid() || blockDim.x != 32) return;

        __shared__ T aStorage[DPPDetails::MAX_STAGES][ALayout::size()];
        __shared__ T bStorage[DPPDetails::MAX_STAGES][BLayout::size()];
        const auto& aInput = get<0>(reads);
        const auto& bInput = get<1>(reads);
        const int tileCount =
            (details.K + DPPDetails::K_TILE - 1) / DPPDetails::K_TILE;
        typename MmaPolicy::AccumulatorFragment accumulator;
        MmaPolicy::clear(accumulator);

        auto issueTile = [&](const int tileIndex, const int slot) {
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
            ACopyPolicy::issue(
                aCopy, aInput, Tile<T, ALayout>{aStorage[slot]});
            BCopyPolicy::issue(
                bCopy, bInput, Tile<T, BLayout>{bStorage[slot]});
            ACopyPolicy::commit();
        };

        const int prefetch = tileCount < details.stages - 1
            ? tileCount : details.stages - 1;
        for (int tile = 0; tile < prefetch; ++tile)
            issueTile(tile, tile % details.stages);
        int nextTile = prefetch;

        for (int tile = 0; tile < tileCount; ++tile) {
            const int slot = tile % details.stages;
            const int pendingAfterCurrent = nextTile - tile - 1;
            waitForPending(pendingAfterCurrent);

            const int kBase = tile * DPPDetails::K_TILE;
            int validK = details.K - kBase;
            validK = validK > DPPDetails::K_TILE
                ? DPPDetails::K_TILE : validK;
            const ACopyDetails aCopy{
                details.aOriginX + kBase, details.aOriginY,
                validK, DPPDetails::M, T{}};
            const BCopyDetails bCopy{
                details.bOriginX + kBase, details.bOriginY,
                validK, DPPDetails::N, T{}};
            const Tile<T, ALayout> aTile{aStorage[slot]};
            const Tile<T, BLayout> bTile{bStorage[slot]};
            ACopyPolicy::complete(aCopy, aTile);
            BCopyPolicy::complete(bCopy, bTile);
            __syncwarp();

            if (nextTile < tileCount) {
                issueTile(nextTile, nextTile % details.stages);
                ++nextTile;
            }

            const ASharedRead aShared{{aTile}};
            const BSharedRead bShared{{bTile}};
            const MmaDetails mma{0, 0, 0, 0,
                                 details.dOriginX, details.dOriginY};
            MmaPolicy::accumulate(mma, aShared, bShared, accumulator);
        }

        const MmaDetails outputDetails{
            0, 0, 0, 0, details.dOriginX, details.dOriginY};
        MmaPolicy::store(outputDetails, accumulator, output);
    }
};
#endif

} // namespace fk

#endif // FK_COLLECTIVE_MULTISTAGE_H
