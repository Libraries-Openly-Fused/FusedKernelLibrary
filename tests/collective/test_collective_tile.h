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
#include <fused_kernel/algorithms/collective/copy.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <type_traits>

using namespace fk;

namespace {

using RowLayout = RowMajorLayout<3, 4>;
using ColumnLayout = ColumnMajorLayout<3, 4>;
using Details = CopyTileDPPDetails<float, RowLayout, 64>;

static_assert(RowLayout::rows == 3 && RowLayout::cols == 4);
static_assert(RowLayout::size() == 12);
static_assert(ColumnLayout::rows == 3 && ColumnLayout::cols == 4);
static_assert(ColumnLayout::size() == 12);
static_assert(std::is_trivially_copyable_v<Details>);
static_assert(std::is_trivially_copyable_v<Tile<float, RowLayout>>);
static_assert(CopyTileDPP<ParArch::CPU, Details>::PAR_ARCH == ParArch::CPU);
#if defined(__NVCC__)
static_assert(CopyTileDPP<ParArch::GPU_NVIDIA, Details>::PAR_ARCH ==
              ParArch::GPU_NVIDIA);
#endif

bool testLayoutMappings() {
    constexpr std::array<unsigned, 12> rowExpected{
        0, 1, 2, 3,
        4, 5, 6, 7,
        8, 9, 10, 11};
    constexpr std::array<unsigned, 12> columnExpected{
        0, 3, 6, 9,
        1, 4, 7, 10,
        2, 5, 8, 11};
    for (unsigned row = 0; row < 3; ++row) {
        for (unsigned col = 0; col < 4; ++col) {
            const unsigned logical = row * 4 + col;
            if (RowLayout::offset(row, col) != rowExpected[logical] ||
                ColumnLayout::offset(row, col) != columnExpected[logical]) {
                std::printf("layout mismatch row=%u col=%u rowOff=%u colOff=%u\n",
                            row, col, RowLayout::offset(row, col),
                            ColumnLayout::offset(row, col));
                return false;
            }
        }
    }
    return true;
}

template <typename Layout>
bool runCpuRoundTrip(const char* name) {
    constexpr int WIDTH = 7;
    constexpr int HEIGHT = 5;
    std::array<float, WIDTH * HEIGHT> input{};
    std::array<float, WIDTH * HEIGHT> output{};
    output.fill(-99.f);
    for (int row = 0; row < HEIGHT; ++row) {
        for (int col = 0; col < WIDTH; ++col) {
            input[row * WIDTH + col] = static_cast<float>(row * 10 + col);
        }
    }

    const RawPtr<ND::_2D, float> inputPtr{
        input.data(), PtrDims<ND::_2D>(WIDTH, HEIGHT, WIDTH * sizeof(float))};
    const RawPtr<ND::_2D, float> outputPtr{
        output.data(), PtrDims<ND::_2D>(WIDTH, HEIGHT, WIDTH * sizeof(float))};
    const auto read = PerThreadRead<ND::_2D, float>::build(inputPtr)
        .then(Mul<float>::build(2.f))
        .then(Add<float>::build(1.f));
    const auto write = Mul<float>::build(3.f)
        .then(PerThreadWrite<ND::_2D, float>::build(outputPtr));

    using D = CopyTileDPPDetails<float, Layout, 64>;
    const D details{5, 3, WIDTH, HEIGHT, -7.f};
    std::array<float, Layout::size()> local{};
    Tile<float, Layout> tile{local.data()};
    CopyTileDPP<ParArch::CPU, D>::load(details, read, tile);

    for (unsigned row = 0; row < Layout::rows; ++row) {
        for (unsigned col = 0; col < Layout::cols; ++col) {
            const int globalX = details.originX + static_cast<int>(col);
            const int globalY = details.originY + static_cast<int>(row);
            const bool valid = globalX < WIDTH && globalY < HEIGHT;
            const float expected = valid
                ? input[globalY * WIDTH + globalX] * 2.f + 1.f
                : -7.f;
            if (std::fabs(tile.at(row, col) - expected) > 1e-6f) {
                std::printf("%s CPU load row=%u col=%u got=%f expected=%f\n",
                            name, row, col, tile.at(row, col), expected);
                return false;
            }
        }
    }

    CopyTileDPP<ParArch::CPU, D>::store(details, tile, write);
    for (int row = 0; row < HEIGHT; ++row) {
        for (int col = 0; col < WIDTH; ++col) {
            const bool inside = row >= details.originY && col >= details.originX;
            const float expected = inside
                ? (input[row * WIDTH + col] * 2.f + 1.f) * 3.f
                : -99.f;
            if (std::fabs(output[row * WIDTH + col] - expected) > 1e-6f) {
                std::printf("%s CPU store row=%d col=%d got=%f expected=%f\n",
                            name, row, col, output[row * WIDTH + col], expected);
                return false;
            }
        }
    }
    return true;
}

#if defined(__NVCC__)
template <typename Layout, typename D, typename ReadIOp, typename WriteIOp>
__global__ void copyTileRoundTripKernel(
        const __grid_constant__ D details,
        const __grid_constant__ ReadIOp input,
        const __grid_constant__ WriteIOp output) {
    __shared__ float storage[Layout::size()];
    Tile<float, Layout> tile{storage};
    CopyTileDPP<ParArch::GPU_NVIDIA, D>::load(details, input, tile);
    CopyTileDPP<ParArch::GPU_NVIDIA, D>::store(details, tile, output);
}

template <typename Layout>
bool runGpuRoundTrip(const char* name) {
    constexpr int WIDTH = 7;
    constexpr int HEIGHT = 5;
    Ptr2D<float> input(WIDTH, HEIGHT);
    Ptr2D<float> output(WIDTH, HEIGHT);
    for (int row = 0; row < HEIGHT; ++row) {
        for (int col = 0; col < WIDTH; ++col) {
            input.at(Point{col, row, 0}) = static_cast<float>(row * 10 + col);
            output.at(Point{col, row, 0}) = -99.f;
        }
    }

    Stream stream;
    input.upload(stream);
    output.upload(stream);
    const auto read = PerThreadRead<ND::_2D, float>::build(input)
        .then(Mul<float>::build(2.f))
        .then(Add<float>::build(1.f));
    const auto write = Mul<float>::build(3.f)
        .then(PerThreadWrite<ND::_2D, float>::build(output));
    using D = CopyTileDPPDetails<float, Layout, 64>;
    const D details{5, 3, WIDTH, HEIGHT, -7.f};
    copyTileRoundTripKernel<Layout><<<1, D::BLOCK_THREADS, 0,
                                      stream.getCUDAStream()>>>(
        details, read, write);
    gpuErrchk(cudaGetLastError());
    output.download(stream);
    stream.sync();

    for (int row = 0; row < HEIGHT; ++row) {
        for (int col = 0; col < WIDTH; ++col) {
            const bool inside = row >= details.originY && col >= details.originX;
            const float inputValue = static_cast<float>(row * 10 + col);
            const float expected = inside
                ? (inputValue * 2.f + 1.f) * 3.f
                : -99.f;
            const float got = output.at(Point{col, row, 0});
            if (std::fabs(got - expected) > 1e-6f) {
                std::printf("%s GPU row=%d col=%d got=%f expected=%f\n",
                            name, row, col, got, expected);
                return false;
            }
        }
    }
    return true;
}
#endif

} // namespace

int launch() {
    bool ok = testLayoutMappings();
    ok = runCpuRoundTrip<RowLayout>("row-major") && ok;
    ok = runCpuRoundTrip<ColumnLayout>("column-major") && ok;
#if defined(__NVCC__)
    ok = runGpuRoundTrip<RowLayout>("row-major") && ok;
    ok = runGpuRoundTrip<ColumnLayout>("column-major") && ok;
#endif
    if (ok) std::printf("Tile layouts + CopyTileDPP IOp contract: PASS\n");
    return ok ? 0 : -1;
}
