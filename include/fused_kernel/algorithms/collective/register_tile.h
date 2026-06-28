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

#ifndef FK_COLLECTIVE_REGISTER_TILE_H
#define FK_COLLECTIVE_REGISTER_TILE_H

#include <fused_kernel/algorithms/collective/mma.h>
#include <fused_kernel/algorithms/collective/tile_scheduler.h>
#include <fused_kernel/core/data/tuple.h>
#include <fused_kernel/core/execution_model/operation_model/operation_model.h>
#include <fused_kernel/core/utils/utils.h>

#include <type_traits>

namespace fk {

template <ParArch PA, typename MmaPolicy, typename MmaDetails>
struct RegisterFragmentPolicy;

struct DefaultRegisterFragmentLayout {
    template <ParArch PA, typename MmaPolicy, typename MmaDetails>
    using Policy = RegisterFragmentPolicy<PA, MmaPolicy, MmaDetails>;
};

template <int WM_ATOMS, int WN_ATOMS, typename MmaDetails,
          template <ParArch, typename> class MmaPolicyTemplate = MmaWarpDPP,
          typename FragmentLayout = DefaultRegisterFragmentLayout,
          typename SchedulerOperation =
              WarpTileScheduler<RowMajorWarpTileRaster>>
struct RegisterTileMainloopDPPDetails {
    static_assert(WM_ATOMS > 0 && WN_ATOMS > 0,
                  "Register tile atom counts must be positive");

    using MmaDetailsType = MmaDetails;
    using FragmentLayoutType = FragmentLayout;
    using SchedulerOperationType = SchedulerOperation;
    template <ParArch PA>
    using MmaPolicy = MmaPolicyTemplate<PA, MmaDetails>;
    template <ParArch PA>
    using FragmentPolicy = typename FragmentLayout::template Policy<
        PA, MmaPolicy<PA>, MmaDetails>;

    static constexpr int WM = WM_ATOMS;
    static constexpr int WN = WN_ATOMS;
    static constexpr int ATOM_M = MmaDetails::M;
    static constexpr int ATOM_N = MmaDetails::N;
    static constexpr int K_TILE = MmaDetails::K;
    static constexpr int REGISTER_M = WM * ATOM_M;
    static constexpr int REGISTER_N = WN * ATOM_N;

    int K;
    int mTileExtent;
    int nTileExtent;
    int outputRows;
    int outputCols;
    int aOriginX;
    int aOriginY;
    int bOriginX;
    int bOriginY;
    int dOriginX;
    int dOriginY;
};

template <typename MmaPolicy, typename MmaDetails>
struct RegisterFragmentPolicy<ParArch::CPU, MmaPolicy, MmaDetails> {
    using AFragment = typename MmaPolicy::AFragment;
    using BFragment = typename MmaPolicy::BFragment;
    using AccumulatorFragment = typename MmaPolicy::AccumulatorFragment;

    FK_HOST_STATIC void clear(AccumulatorFragment& fragment) {
        MmaPolicy::clear(fragment);
    }

    template <typename ReadIOp>
    FK_HOST_STATIC void loadA(const MmaDetails& details,
                              const ReadIOp& input,
                              AFragment& fragment) {
        MmaPolicy::loadAFragment(details, input, fragment);
    }

    template <typename ReadIOp>
    FK_HOST_STATIC void loadB(const MmaDetails& details,
                              const ReadIOp& input,
                              BFragment& fragment) {
        MmaPolicy::loadBFragment(details, input, fragment);
    }

    FK_HOST_STATIC void multiply(const AFragment& a,
                                 const BFragment& b,
                                 AccumulatorFragment& accumulator) {
        MmaPolicy::multiply(a, b, accumulator);
    }

    template <typename WriteIOp>
    FK_HOST_STATIC void store(const AccumulatorFragment& fragment,
                              const int rowBase, const int colBase,
                              const int outputRows, const int outputCols,
                              const int dOriginX, const int dOriginY,
                              const WriteIOp& output) {
        for (int row = 0; row < MmaDetails::M; ++row)
            for (int col = 0; col < MmaDetails::N; ++col)
                if (rowBase + row < outputRows &&
                    colBase + col < outputCols)
                    WriteIOp::Operation::exec(
                        Point{dOriginX + colBase + col,
                              dOriginY + rowBase + row, 0},
                        fragment.values[row][col], output);
    }
};

#if defined(__NVCC__)
template <typename MmaPolicy, typename MmaDetails>
struct RegisterFragmentPolicy<ParArch::GPU_NVIDIA, MmaPolicy, MmaDetails> {
    using AFragment = typename MmaPolicy::AFragment;
    using BFragment = typename MmaPolicy::BFragment;
    using AccumulatorFragment = typename MmaPolicy::AccumulatorFragment;

    FK_DEVICE_STATIC void clear(AccumulatorFragment& fragment) {
        MmaPolicy::clear(fragment);
    }

    template <typename ReadIOp>
    FK_DEVICE_STATIC void loadA(const MmaDetails& details,
                                const ReadIOp& input,
                                AFragment& fragment) {
        MmaPolicy::loadAFragment(details, input, fragment);
    }

    template <typename ReadIOp>
    FK_DEVICE_STATIC void loadB(const MmaDetails& details,
                                const ReadIOp& input,
                                BFragment& fragment) {
        MmaPolicy::loadBFragment(details, input, fragment);
    }

    FK_DEVICE_STATIC void multiply(const AFragment& a,
                                   const BFragment& b,
                                   AccumulatorFragment& accumulator) {
        MmaPolicy::multiply(a, b, accumulator);
    }

    template <typename WriteIOp>
    FK_DEVICE_STATIC void store(const AccumulatorFragment& fragment,
                                const int rowBase, const int colBase,
                                const int outputRows, const int outputCols,
                                const int dOriginX, const int dOriginY,
                                const WriteIOp& output) {
        static_assert(MmaDetails::M == 16 && MmaDetails::N == 8,
                      "Default GPU fragment layout supports m16n8 atoms");
        const int lane = threadIdx.x & 31;
        const int group = lane >> 2;
        const int threadInGroup = lane & 3;
        const int rows[4]{group, group, group + 8, group + 8};
        const int cols[4]{2 * threadInGroup, 2 * threadInGroup + 1,
                          2 * threadInGroup, 2 * threadInGroup + 1};
        #pragma unroll
        for (int value = 0; value < 4; ++value) {
            if (rowBase + rows[value] < outputRows &&
                colBase + cols[value] < outputCols)
                WriteIOp::Operation::exec(
                    Point{dOriginX + colBase + cols[value],
                          dOriginY + rowBase + rows[value], 0},
                    fragment.values[value], output);
        }
    }
};
#endif

template <ParArch PA, typename DPPDetails>
struct RegisterTileMainloopDPP;

template <typename DPPDetails>
struct RegisterTileMainloopDPP<ParArch::CPU, DPPDetails> {
private:
    using SelfType = RegisterTileMainloopDPP<ParArch::CPU, DPPDetails>;
    using MmaDetails = typename DPPDetails::MmaDetailsType;
    using FragmentPolicy = typename DPPDetails::template FragmentPolicy<
        ParArch::CPU>;
    using AFragment = typename FragmentPolicy::AFragment;
    using BFragment = typename FragmentPolicy::BFragment;
    using AccumulatorFragment = typename FragmentPolicy::AccumulatorFragment;

    template <typename ReadIOps, typename WriteIOp>
    FK_HOST_STATIC void executeAssignment(
            const DPPDetails& details,
            const WarpTileAssignment assignment,
            const ReadIOps& reads, const WriteIOp& output) {
        const auto& a = get<0>(reads);
        const auto& b = get<1>(reads);
        const int registerRow = assignment.mTile * DPPDetails::REGISTER_M;
        const int registerCol = assignment.nTile * DPPDetails::REGISTER_N;
        AccumulatorFragment accumulators[DPPDetails::WM][DPPDetails::WN];
        for (int i = 0; i < DPPDetails::WM; ++i)
            for (int j = 0; j < DPPDetails::WN; ++j)
                FragmentPolicy::clear(accumulators[i][j]);

        for (int kBase = 0; kBase < details.K;
             kBase += DPPDetails::K_TILE) {
            AFragment aFragments[DPPDetails::WM];
            BFragment bFragments[DPPDetails::WN];
            for (int i = 0; i < DPPDetails::WM; ++i) {
                const MmaDetails mma{
                    details.aOriginX + kBase,
                    details.aOriginY + registerRow + i * DPPDetails::ATOM_M,
                    details.bOriginX + kBase, details.bOriginY,
                    0, 0};
                FragmentPolicy::loadA(mma, a, aFragments[i]);
            }
            for (int j = 0; j < DPPDetails::WN; ++j) {
                const MmaDetails mma{
                    details.aOriginX + kBase, details.aOriginY,
                    details.bOriginX + kBase,
                    details.bOriginY + registerCol + j * DPPDetails::ATOM_N,
                    0, 0};
                FragmentPolicy::loadB(mma, b, bFragments[j]);
            }
            for (int i = 0; i < DPPDetails::WM; ++i)
                for (int j = 0; j < DPPDetails::WN; ++j)
                    FragmentPolicy::multiply(
                        aFragments[i], bFragments[j], accumulators[i][j]);
        }
        for (int i = 0; i < DPPDetails::WM; ++i)
            for (int j = 0; j < DPPDetails::WN; ++j)
                FragmentPolicy::store(
                    accumulators[i][j],
                    registerRow + i * DPPDetails::ATOM_M,
                    registerCol + j * DPPDetails::ATOM_N,
                    details.outputRows, details.outputCols,
                    details.dOriginX, details.dOriginY, output);
    }

public:
    FK_STATIC_STRUCT(RegisterTileMainloopDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::CPU;

    template <typename ReadIOps, typename WriteIOp, typename SchedulerIOp>
    FK_HOST_STATIC void exec(const DPPDetails& details,
                             const ReadIOps& reads,
                             const WriteIOp& output,
                             const SchedulerIOp& scheduler) {
        static_assert(isTuple_v<ReadIOps> &&
                      std::decay_t<ReadIOps>::size == 2);
        static_assert(isAnyWriteType<WriteIOp>);
        static_assert(std::is_same_v<
            typename std::decay_t<SchedulerIOp>::Operation,
            typename DPPDetails::SchedulerOperationType>);
        if (details.K <= 0 || details.K % DPPDetails::K_TILE != 0 ||
            details.mTileExtent <= 0 || details.nTileExtent <= 0 ||
            details.outputRows <= 0 || details.outputCols <= 0) return;
        const int warpCount = details.mTileExtent * details.nTileExtent;
        for (int warpId = 0; warpId < warpCount; ++warpId) {
            const WarpTileAssignment assignment =
                make_tuple(warpId, details.mTileExtent,
                           details.nTileExtent) | scheduler;
            if (assignment.valid)
                executeAssignment(details, assignment, reads, output);
        }
    }
};

#if defined(__NVCC__)
template <typename DPPDetails>
struct RegisterTileMainloopDPP<ParArch::GPU_NVIDIA, DPPDetails> {
private:
    using SelfType = RegisterTileMainloopDPP<ParArch::GPU_NVIDIA, DPPDetails>;
    using MmaDetails = typename DPPDetails::MmaDetailsType;
    using FragmentPolicy = typename DPPDetails::template FragmentPolicy<
        ParArch::GPU_NVIDIA>;
    using AFragment = typename FragmentPolicy::AFragment;
    using BFragment = typename FragmentPolicy::BFragment;
    using AccumulatorFragment = typename FragmentPolicy::AccumulatorFragment;

public:
    FK_STATIC_STRUCT(RegisterTileMainloopDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::GPU_NVIDIA;

    template <typename ReadIOps, typename WriteIOp, typename SchedulerIOp>
    FK_DEVICE_STATIC void exec(const DPPDetails& details,
                               const ReadIOps& reads,
                               const WriteIOp& output,
                               const SchedulerIOp& scheduler) {
        static_assert(isTuple_v<ReadIOps> &&
                      std::decay_t<ReadIOps>::size == 2);
        static_assert(isAnyWriteType<WriteIOp>);
        static_assert(std::is_same_v<
            typename std::decay_t<SchedulerIOp>::Operation,
            typename DPPDetails::SchedulerOperationType>);
        if (details.K <= 0 || details.K % DPPDetails::K_TILE != 0 ||
            details.mTileExtent <= 0 || details.nTileExtent <= 0 ||
            details.outputRows <= 0 || details.outputCols <= 0 ||
            blockDim.x % 32 != 0) return;

        const int warpsPerBlock = blockDim.x / 32;
        const int warpId = blockIdx.x * warpsPerBlock + threadIdx.x / 32;
        const WarpTileAssignment assignment =
            make_tuple(warpId, details.mTileExtent,
                       details.nTileExtent) | scheduler;
        if (!assignment.valid) return;

        const auto& a = get<0>(reads);
        const auto& b = get<1>(reads);
        const int registerRow = assignment.mTile * DPPDetails::REGISTER_M;
        const int registerCol = assignment.nTile * DPPDetails::REGISTER_N;
        AccumulatorFragment accumulators[DPPDetails::WM][DPPDetails::WN];
        #pragma unroll
        for (int i = 0; i < DPPDetails::WM; ++i)
            #pragma unroll
            for (int j = 0; j < DPPDetails::WN; ++j)
                FragmentPolicy::clear(accumulators[i][j]);

        for (int kBase = 0; kBase < details.K;
             kBase += DPPDetails::K_TILE) {
            AFragment aFragments[DPPDetails::WM];
            BFragment bFragments[DPPDetails::WN];
            #pragma unroll
            for (int i = 0; i < DPPDetails::WM; ++i) {
                const MmaDetails mma{
                    details.aOriginX + kBase,
                    details.aOriginY + registerRow + i * DPPDetails::ATOM_M,
                    details.bOriginX + kBase, details.bOriginY,
                    0, 0};
                FragmentPolicy::loadA(mma, a, aFragments[i]);
            }
            #pragma unroll
            for (int j = 0; j < DPPDetails::WN; ++j) {
                const MmaDetails mma{
                    details.aOriginX + kBase, details.aOriginY,
                    details.bOriginX + kBase,
                    details.bOriginY + registerCol + j * DPPDetails::ATOM_N,
                    0, 0};
                FragmentPolicy::loadB(mma, b, bFragments[j]);
            }
            #pragma unroll
            for (int i = 0; i < DPPDetails::WM; ++i)
                #pragma unroll
                for (int j = 0; j < DPPDetails::WN; ++j)
                    FragmentPolicy::multiply(
                        aFragments[i], bFragments[j], accumulators[i][j]);
        }
        #pragma unroll
        for (int i = 0; i < DPPDetails::WM; ++i)
            #pragma unroll
            for (int j = 0; j < DPPDetails::WN; ++j)
                FragmentPolicy::store(
                    accumulators[i][j],
                    registerRow + i * DPPDetails::ATOM_M,
                    registerCol + j * DPPDetails::ATOM_N,
                    details.outputRows, details.outputCols,
                    details.dOriginX, details.dOriginY, output);
    }
};
#endif

} // namespace fk

#endif // FK_COLLECTIVE_REGISTER_TILE_H
