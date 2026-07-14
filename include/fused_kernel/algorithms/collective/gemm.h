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

#ifndef FK_COLLECTIVE_GEMM_H
#define FK_COLLECTIVE_GEMM_H

#include <fused_kernel/algorithms/collective/multistage.h>
#include <fused_kernel/algorithms/collective/tile_scheduler.h>
#include <fused_kernel/core/execution_model/executor_details/dpp_launch_config.h>
#include <fused_kernel/core/execution_model/parallel_architectures.h>

#include <type_traits>
#include <utility>

namespace fk {

namespace gemm_detail {

template <typename Input, typename T>
struct BoundedReadOperation {
private:
    using SelfType = BoundedReadOperation<Input, T>;

public:
    using InputParams = std::decay_t<decltype(std::declval<Input>().params)>;
    struct BoundedParams {
        InputParams input;
        int rows;
        int columns;
    };
    using Parent = ReadOperation<T, BoundedParams, T,
                                 TF::DISABLED, SelfType>;
    FK_STATIC_STRUCT(BoundedReadOperation, SelfType)
    DECLARE_READ_PARENT

    FK_HOST_DEVICE_FUSE InstantiableType instantiate(
            const ParamsType& params) {
        return {{params}};
    }

    FK_HOST_DEVICE_FUSE OutputType exec(const Point point,
                                        const ParamsType& params) {
        if (point.x < 0 || point.y < 0 ||
            point.x >= params.columns || point.y >= params.rows)
            return T{};
        return Input::Operation::exec(point, params.input);
    }
};

template <typename Output>
struct BoundedWriteOperation {
private:
    using SelfType = BoundedWriteOperation<Output>;

public:
    using OutputParams = std::decay_t<decltype(std::declval<Output>().params)>;
    struct BoundedParams {
        OutputParams output;
        int rows;
        int columns;
    };
    using Parent = WriteOperation<float2, BoundedParams, float2,
                                  TF::DISABLED, SelfType>;
    FK_STATIC_STRUCT(BoundedWriteOperation, SelfType)
    DECLARE_WRITE_PARENT

    FK_HOST_DEVICE_FUSE InstantiableType instantiate(
            const ParamsType& params) {
        return {{params}};
    }

    FK_HOST_DEVICE_FUSE void exec(const Point point, const InputType value,
                                  const ParamsType& params) {
        const int firstColumn = point.x * 2;
        if (point.y < 0 || point.y >= params.rows ||
            firstColumn < 0 || firstColumn >= params.columns)
            return;
        Output::Operation::exec(point, value, params.output);
    }
};

} // namespace gemm_detail

/**
 * Static GEMM policy plus runtime M/N/K.
 *
 * A and B are supplied together as Tuple<ReadIOp, ReadIOp>. D is never part
 * of this details object: it exists only in the final (possibly fused) Write
 * IOp passed to GemmDPP::exec.
 */
template <
    int STAGE_COUNT,
    typename MmaDetailsType,
    typename AStageLayoutType,
    typename BStageLayoutType,
    typename StageTypeT = float,
    typename SchedulerOperationT = WarpTileScheduler<RowMajorWarpTileRaster>,
    template <ParArch, typename> class MmaPolicyTemplate = MmaWarpDPP,
    template <ParArch, typename> class CopyPolicyTemplate = AsyncCopyDPP>
struct GemmDPPDetails {
    static_assert(STAGE_COUNT >= 2,
                  "GemmDPP requires at least double buffering");

    static constexpr int STAGES = STAGE_COUNT;
    using MmaDetails = MmaDetailsType;
    using AStageLayout = AStageLayoutType;
    using BStageLayout = BStageLayoutType;
    using StageType = StageTypeT;
    using SchedulerOperation = SchedulerOperationT;
    using SchedulerIOp = decltype(SchedulerOperation::build());
    template <ParArch PA, typename Details>
    using MmaPolicy = MmaPolicyTemplate<PA, Details>;
    template <ParArch PA, typename Details>
    using CopyPolicy = CopyPolicyTemplate<PA, Details>;
    using MainloopDetails = MultiStageMainloopDPPDetails<
        STAGES, MmaDetails, AStageLayout, BStageLayout, StageType,
        MmaPolicyTemplate, CopyPolicyTemplate>;

    static constexpr int TILE_M = MmaDetails::M;
    static constexpr int TILE_N = MmaDetails::N;

    int M;
    int N;
    int K;

    FK_HOST_DEVICE_CNST bool valid() const {
        return M > 0 && N > 0 && K > 0;
    }
};

template <ParArch PA, typename GemmDetails>
struct GemmDPP;

template <typename GemmDetails>
struct GemmDPP<ParArch::CPU, GemmDetails> {
private:
    using SelfType = GemmDPP<ParArch::CPU, GemmDetails>;
    using Mainloop = MultiStageMainloopDPP<
        ParArch::CPU, typename GemmDetails::MainloopDetails>;

    template <typename Inputs, typename Output>
    FK_HOST_STATIC void executeTile(const GemmDetails& details,
                                    const WarpTileAssignment tile,
                                    const Inputs& inputs,
                                    const Output& output) {
        const auto& a = get<0>(inputs);
        const auto& b = get<1>(inputs);
        using A = std::decay_t<decltype(a)>;
        using B = std::decay_t<decltype(b)>;
        using Out = std::decay_t<Output>;
        using MmaDetails = typename GemmDetails::MmaDetails;
        const int row = tile.mTile * GemmDetails::TILE_M;
        const int column = tile.nTile * GemmDetails::TILE_N;
        const typename GemmDetails::MainloopDetails mainloopDetails{
            GemmDetails::STAGES, details.K,
            0, row, 0, column, column / 2, row};
        const bool fullTile = row + GemmDetails::TILE_M <= details.M &&
                              column + GemmDetails::TILE_N <= details.N;
        const bool fullK = details.K % MmaDetails::K == 0;
        if (fullTile && fullK) {
            Mainloop::exec(mainloopDetails, inputs, output);
            return;
        }
        const auto boundedInputs = make_tuple(
            gemm_detail::BoundedReadOperation<
                A, typename GemmDetails::StageType>::instantiate(
                    {a.params, details.M, details.K}),
            gemm_detail::BoundedReadOperation<
                B, typename GemmDetails::StageType>::instantiate(
                    {b.params, details.N, details.K}));
        const auto boundedOutput =
            gemm_detail::BoundedWriteOperation<Out>::instantiate(
                {output.params, details.M, details.N});
        Mainloop::exec(mainloopDetails, boundedInputs, boundedOutput);
    }

public:
    FK_STATIC_STRUCT(GemmDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::CPU;

    FK_HOST_FUSE DPPLaunchConfig launchConfig(const GemmDetails&) {
        return {};
    }

    template <typename Inputs, typename Output>
    FK_HOST_STATIC void exec(const GemmDetails& details,
                             const Inputs& inputs, const Output& output) {
        if (!details.valid()) return;
        const int mTiles =
            (details.M + GemmDetails::TILE_M - 1) / GemmDetails::TILE_M;
        const int nTiles =
            (details.N + GemmDetails::TILE_N - 1) / GemmDetails::TILE_N;
        const typename GemmDetails::SchedulerIOp scheduler{};
        for (int tileId = 0; tileId < mTiles * nTiles; ++tileId) {
            const WarpTileAssignment tile =
                make_tuple(tileId, mTiles, nTiles) | scheduler;
            if (tile.valid) executeTile(details, tile, inputs, output);
        }
    }
};

#if defined(__NVCC__)
template <typename GemmDetails>
struct GemmDPP<ParArch::GPU_NVIDIA, GemmDetails> {
private:
    using SelfType = GemmDPP<ParArch::GPU_NVIDIA, GemmDetails>;
    using Mainloop = MultiStageMainloopDPP<
        ParArch::GPU_NVIDIA, typename GemmDetails::MainloopDetails>;

public:
    FK_STATIC_STRUCT(GemmDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::GPU_NVIDIA;

    FK_HOST_FUSE DPPLaunchConfig launchConfig(const GemmDetails& details) {
        const unsigned int mTiles = static_cast<unsigned int>(
            (details.M + GemmDetails::TILE_M - 1) / GemmDetails::TILE_M);
        const unsigned int nTiles = static_cast<unsigned int>(
            (details.N + GemmDetails::TILE_N - 1) / GemmDetails::TILE_N);
        // MultiStageMainloopDPP owns static shared storage internally.
        return {mTiles * nTiles, 1, 1, 32, 1, 1, 0};
    }

    template <typename Inputs, typename Output>
    FK_DEVICE_STATIC void exec(const GemmDetails& details,
                               const Inputs& inputs, const Output& output) {
        if (!details.valid()) return;
        const int mTiles =
            (details.M + GemmDetails::TILE_M - 1) / GemmDetails::TILE_M;
        const int nTiles =
            (details.N + GemmDetails::TILE_N - 1) / GemmDetails::TILE_N;
        const typename GemmDetails::SchedulerIOp scheduler{};
        const WarpTileAssignment tile =
            make_tuple(static_cast<int>(blockIdx.x), mTiles, nTiles) |
            scheduler;
        if (!tile.valid) return;

        const auto& a = get<0>(inputs);
        const auto& b = get<1>(inputs);
        using A = std::decay_t<decltype(a)>;
        using B = std::decay_t<decltype(b)>;
        using Out = std::decay_t<Output>;
        using MmaDetails = typename GemmDetails::MmaDetails;
        const int row = tile.mTile * GemmDetails::TILE_M;
        const int column = tile.nTile * GemmDetails::TILE_N;
        const typename GemmDetails::MainloopDetails mainloopDetails{
            GemmDetails::STAGES, details.K,
            0, row, 0, column, column / 2, row};
        const bool fullTile = row + GemmDetails::TILE_M <= details.M &&
                              column + GemmDetails::TILE_N <= details.N;
        const bool fullK = details.K % MmaDetails::K == 0;
        if (fullTile && fullK) {
            Mainloop::exec(mainloopDetails, inputs, output);
            return;
        }
        const auto boundedInputs = make_tuple(
            gemm_detail::BoundedReadOperation<
                A, typename GemmDetails::StageType>::instantiate(
                    {a.params, details.M, details.K}),
            gemm_detail::BoundedReadOperation<
                B, typename GemmDetails::StageType>::instantiate(
                    {b.params, details.N, details.K}));
        const auto boundedOutput =
            gemm_detail::BoundedWriteOperation<Out>::instantiate(
                {output.params, details.M, details.N});
        Mainloop::exec(mainloopDetails, boundedInputs, boundedOutput);
    }
};
#endif

} // namespace fk

#endif // FK_COLLECTIVE_GEMM_H
