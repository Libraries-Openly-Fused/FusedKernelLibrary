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

#include <fused_kernel/algorithms/collective/cta_raster.h>
#include <fused_kernel/fused_kernel.h>

#include <vector>

using namespace fk;

namespace {

struct MappingDetails {
    int mTiles;
    int nTiles;
};

template <ParArch PA>
struct MappingDPP;

template <>
struct MappingDPP<ParArch::CPU> {
    static constexpr ParArch PAR_ARCH = ParArch::CPU;

    FK_HOST_FUSE DPPLaunchConfig launchConfig(const MappingDetails&) {
        return {};
    }

    template <typename Scheduler, typename Output>
    FK_HOST_STATIC void exec(const MappingDetails& details,
                             const Scheduler& scheduler,
                             const Output& output) {
        const int total = details.mTiles * details.nTiles;
        for (int cta = 0; cta < total; ++cta) {
            const CtaTileAssignment tile =
                make_tuple(cta, details.mTiles, details.nTiles) | scheduler;
            Output::Operation::exec(
                Point{cta, 0, 0}, int2{tile.mTile, tile.nTile},
                output.params);
        }
    }
};

#if defined(__NVCC__)
template <>
struct MappingDPP<ParArch::GPU_NVIDIA> {
    static constexpr ParArch PAR_ARCH = ParArch::GPU_NVIDIA;

    FK_HOST_FUSE DPPLaunchConfig launchConfig(const MappingDetails& details) {
        const unsigned int total = static_cast<unsigned int>(
            details.mTiles * details.nTiles);
        return {(total + 31) / 32, 1, 1, 32, 1, 1, 0};
    }

    template <typename Scheduler, typename Output>
    FK_DEVICE_STATIC void exec(const MappingDetails& details,
                               const Scheduler& scheduler,
                               const Output& output) {
        const int cta = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        if (cta >= details.mTiles * details.nTiles) return;
        const CtaTileAssignment tile =
            make_tuple(cta, details.mTiles, details.nTiles) | scheduler;
        Output::Operation::exec(
            Point{cta, 0, 0}, int2{tile.mTile, tile.nTile},
            output.params);
    }
};
#endif

template <typename RasterOperation>
bool mappingIsBijective(const int mTiles, const int nTiles) {
    const auto scheduler = RasterOperation::build();
    const int total = mTiles * nTiles;
    std::vector<int> seen(static_cast<std::size_t>(total), -1);
    for (int cta = 0; cta < total; ++cta) {
        const CtaTileAssignment tile =
            make_tuple(cta, mTiles, nTiles) | scheduler;
        if (!tile.valid || tile.mTile < 0 || tile.mTile >= mTiles ||
            tile.nTile < 0 || tile.nTile >= nTiles)
            return false;
        const int index = tile.mTile * nTiles + tile.nTile;
        if (seen[static_cast<std::size_t>(index)] != -1) return false;
        seen[static_cast<std::size_t>(index)] = cta;
    }
    for (const int owner : seen)
        if (owner < 0) return false;
    return true;
}

template <typename RasterOperation>
bool mappingBackendsAgree(const int mTiles, const int nTiles) {
    const int total = mTiles * nTiles;
    Ptr1D<int2> cpu(static_cast<uint>(total), 0, MemType::Host);
    Stream_<ParArch::CPU> cpuStream;
    executeOperations<MappingDPP<ParArch::CPU>>(
        cpuStream, MappingDetails{mTiles, nTiles}, RasterOperation::build(),
        PerThreadWrite<ND::_1D, int2>::build(cpu));

    const auto scheduler = RasterOperation::build();
    for (int cta = 0; cta < total; ++cta) {
        const CtaTileAssignment expected =
            make_tuple(cta, mTiles, nTiles) | scheduler;
        const int2 actual = cpu.at(Point{cta, 0, 0});
        if (actual.x != expected.mTile || actual.y != expected.nTile)
            return false;
    }

#if defined(__NVCC__)
    Ptr1D<int2> gpu(static_cast<uint>(total));
    Stream gpuStream;
    executeOperations<MappingDPP<ParArch::GPU_NVIDIA>>(
        gpuStream, MappingDetails{mTiles, nTiles}, RasterOperation::build(),
        PerThreadWrite<ND::_1D, int2>::build(gpu));
    gpu.download(gpuStream);
    gpuStream.sync();
    for (int cta = 0; cta < total; ++cta) {
        const int2 expected = cpu.at(Point{cta, 0, 0});
        const int2 actual = gpu.at(Point{cta, 0, 0});
        if (actual.x != expected.x || actual.y != expected.y) return false;
    }
#endif
    return true;
}

template <typename RasterOperation>
bool mappingSuite() {
    bool ok = true;
    ok = mappingIsBijective<RasterOperation>(1, 1) && ok;
    ok = mappingIsBijective<RasterOperation>(1, 9) && ok;
    ok = mappingIsBijective<RasterOperation>(9, 1) && ok;
    ok = mappingIsBijective<RasterOperation>(13, 7) && ok;
    ok = mappingIsBijective<RasterOperation>(7, 13) && ok;
    ok = mappingIsBijective<RasterOperation>(16, 16) && ok;
    ok = mappingBackendsAgree<RasterOperation>(13, 7) && ok;
    return ok;
}

} // namespace

int launch() {
    bool ok = true;
    ok = mappingSuite<CtaTileScheduler<RowMajorCtaTileRaster>>() && ok;
    ok = mappingSuite<CtaTileScheduler<ColumnMajorCtaTileRaster>>() && ok;
    ok = mappingSuite<CtaTileScheduler<GroupedCtaTileRaster<1>>>() && ok;
    ok = mappingSuite<CtaTileScheduler<GroupedCtaTileRaster<2>>>() && ok;
    ok = mappingSuite<CtaTileScheduler<GroupedCtaTileRaster<3>>>() && ok;
    ok = mappingSuite<CtaTileScheduler<GroupedCtaTileRaster<4>>>() && ok;
    ok = mappingSuite<CtaTileScheduler<GroupedCtaTileRaster<8>>>() && ok;
    ok = mappingSuite<CtaTileScheduler<GroupedCtaTileRaster<16>>>() && ok;
    std::printf("CTA raster Operation coverage + CPU/GPU parity: %s\n",
                ok ? "PASS" : "FAIL");
    return ok ? 0 : -1;
}
