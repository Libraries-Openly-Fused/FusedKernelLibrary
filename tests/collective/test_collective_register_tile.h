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

#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/collective/register_tile.h>
#include <fused_kernel/algorithms/collective/tile_scheduler.h>

#include <cmath>
#include <cstdio>
#include <vector>

#if defined(__NVCC__)
#include <cuda_bf16.h>
#endif

using namespace fk;

namespace {

using AtomDetails = MmaDPPDetails<MmaBf16_16x8x16>;
using SchedulerOperation = WarpTileScheduler<RowMajorWarpTileRaster>;

float rawA(const int row, const int k) {
    return ((row * 7 + k * 3) % 19 - 9) * 0.0625f;
}

float rawB(const int col, const int k) {
    return ((col * 5 + k * 2) % 17 - 8) * 0.078125f;
}

float effective(const float value, const bool gpuBf16) {
#if defined(__NVCC__)
    if (gpuBf16)
        return __bfloat162float(__float2bfloat16(value));
#else
    (void)gpuBf16;
#endif
    return value;
}

double reference(const int row, const int col, const int K,
                 const bool gpuBf16) {
    double result = 0.0;
    for (int k = 0; k < K; ++k) {
        const float a = effective(rawA(row, k) * 1.5f, gpuBf16);
        const float b = effective(rawB(col, k) + 0.25f, gpuBf16);
        result += static_cast<double>(a) * static_cast<double>(b);
    }
    return result * 0.5;
}

bool checkSchedulerOrder() {
    const auto scheduler = SchedulerOperation::build();
    for (int warp = 0; warp < 6; ++warp) {
        const WarpTileAssignment assignment =
            make_tuple(warp, 2, 3) | scheduler;
        if (!assignment.valid || assignment.mTile != warp / 3 ||
            assignment.nTile != warp % 3) return false;
    }
    return true;
}

template <int WM, int WN>
using Details = RegisterTileMainloopDPPDetails<
    WM, WN, AtomDetails, MmaWarpDPP,
    DefaultRegisterFragmentLayout, SchedulerOperation>;

template <int WM, int WN, typename OutputGetter>
bool checkOutput(const int K, const int rows, const int cols,
                 const bool gpuBf16, const OutputGetter& output) {
    const double tolerance = gpuBf16 ? 0.05 * (K / 16.0) + 1e-3 : 1e-5;
    double maxError = 0.0;
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            maxError = std::max(maxError, std::abs(
                static_cast<double>(output(row, col)) -
                reference(row, col, K, gpuBf16)));
        }
    }
    if (maxError > tolerance) {
        std::printf("RegisterTile %dx%d K=%d maxError=%g tolerance=%g\n",
                    WM, WN, K, maxError, tolerance);
        return false;
    }
    return true;
}

template <int WM, int WN>
bool runCpu(const int K) {
    using D = Details<WM, WN>;
    constexpr int REGISTER_M = D::REGISTER_M;
    constexpr int REGISTER_N = D::REGISTER_N;
    constexpr int M_TILES = 2;
    constexpr int N_TILES = 3;
    const int rows = REGISTER_M * M_TILES - 3;
    const int cols = REGISTER_N * N_TILES - 1;
    const int paddedRows = REGISTER_M * M_TILES;
    const int paddedCols = REGISTER_N * N_TILES;
    std::vector<float> a(paddedRows * K), b(paddedCols * K);
    std::vector<float> output(rows * cols, -99.f);
    for (int row = 0; row < paddedRows; ++row)
        for (int k = 0; k < K; ++k) a[row * K + k] = rawA(row, k);
    for (int col = 0; col < paddedCols; ++col)
        for (int k = 0; k < K; ++k) b[col * K + k] = rawB(col, k);
    const RawPtr<ND::_2D, float> aPtr{
        a.data(), PtrDims<ND::_2D>(K, paddedRows, K * sizeof(float))};
    const RawPtr<ND::_2D, float> bPtr{
        b.data(), PtrDims<ND::_2D>(K, paddedCols, K * sizeof(float))};
    const RawPtr<ND::_2D, float> dPtr{
        output.data(), PtrDims<ND::_2D>(cols, rows, cols * sizeof(float))};
    const auto reads = make_tuple(
        PerThreadRead<ND::_2D, float>::build(aPtr)
            .then(Mul<float>::build(1.5f)),
        PerThreadRead<ND::_2D, float>::build(bPtr)
            .then(Add<float>::build(0.25f)));
    const auto write = Mul<float>::build(0.5f)
        .then(PerThreadWrite<ND::_2D, float>::build(dPtr));
    const auto scheduler = SchedulerOperation::build();
    const D details{K, M_TILES, N_TILES, rows, cols,
                    0, 0, 0, 0, 0, 0};
    RegisterTileMainloopDPP<ParArch::CPU, D>::exec(
        details, reads, write, scheduler);
    return checkOutput<WM, WN>(K, rows, cols, false,
        [&](const int row, const int col) {
            return output[row * cols + col];
        });
}

#if defined(__NVCC__)
template <typename D, typename Reads, typename Write, typename Scheduler>
__global__ void registerTileKernel(const D details, const Reads reads,
                                   const Write output,
                                   const Scheduler scheduler) {
    RegisterTileMainloopDPP<ParArch::GPU_NVIDIA, D>::exec(
        details, reads, output, scheduler);
}

template <int WM, int WN>
bool runGpu(const int K) {
    using D = Details<WM, WN>;
    constexpr int REGISTER_M = D::REGISTER_M;
    constexpr int REGISTER_N = D::REGISTER_N;
    constexpr int M_TILES = 2;
    constexpr int N_TILES = 3;
    const int rows = REGISTER_M * M_TILES - 3;
    const int cols = REGISTER_N * N_TILES - 1;
    const int paddedRows = REGISTER_M * M_TILES;
    const int paddedCols = REGISTER_N * N_TILES;
    Ptr2D<float> a(K, paddedRows), b(K, paddedCols), output(cols, rows);
    for (int row = 0; row < paddedRows; ++row)
        for (int k = 0; k < K; ++k)
            a.at(Point{k, row, 0}) = rawA(row, k);
    for (int col = 0; col < paddedCols; ++col)
        for (int k = 0; k < K; ++k)
            b.at(Point{k, col, 0}) = rawB(col, k);
    Stream stream;
    a.upload(stream);
    b.upload(stream);
    const auto reads = make_tuple(
        PerThreadRead<ND::_2D, float>::build(a)
            .then(Mul<float>::build(1.5f)),
        PerThreadRead<ND::_2D, float>::build(b)
            .then(Add<float>::build(0.25f)));
    const auto write = Mul<float>::build(0.5f)
        .then(PerThreadWrite<ND::_2D, float>::build(output));
    const auto scheduler = SchedulerOperation::build();
    const D details{K, M_TILES, N_TILES, rows, cols,
                    0, 0, 0, 0, 0, 0};
    constexpr int WARPS_PER_BLOCK = 4;
    const int blocks = (M_TILES * N_TILES + WARPS_PER_BLOCK - 1) /
                       WARPS_PER_BLOCK;
    registerTileKernel<<<blocks, WARPS_PER_BLOCK * 32, 0,
                         stream.getCUDAStream()>>>(
        details, reads, write, scheduler);
    gpuErrchk(cudaGetLastError());
    output.download(stream);
    stream.sync();
    return checkOutput<WM, WN>(K, rows, cols, true,
        [&](const int row, const int col) {
            return output.at(Point{col, row, 0});
        });
}
#endif

} // namespace

int launch() {
    bool ok = checkSchedulerOrder();
    ok = runCpu<1, 1>(16) && ok;
    ok = runCpu<1, 1>(48) && ok;
    ok = runCpu<2, 2>(16) && ok;
    ok = runCpu<2, 2>(48) && ok;
    ok = runCpu<2, 4>(16) && ok;
    ok = runCpu<2, 4>(48) && ok;
    ok = runCpu<4, 2>(16) && ok;
    ok = runCpu<4, 2>(48) && ok;
#if defined(__NVCC__)
    ok = runGpu<1, 1>(16) && ok;
    ok = runGpu<1, 1>(48) && ok;
    ok = runGpu<2, 2>(16) && ok;
    ok = runGpu<2, 2>(48) && ok;
    ok = runGpu<2, 4>(16) && ok;
    ok = runGpu<2, 4>(48) && ok;
    ok = runGpu<4, 2>(16) && ok;
    ok = runGpu<4, 2>(48) && ok;
#endif
    if (ok) std::printf("RegisterTileMainloopDPP contracts: PASS\n");
    return ok ? 0 : -1;
}
