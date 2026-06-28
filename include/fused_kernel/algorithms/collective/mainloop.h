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

#include <fused_kernel/algorithms/collective/mma.h>
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

} // namespace fk

#endif // FK_COLLECTIVE_MAINLOOP_H
