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
#include <fused_kernel/algorithms/collective/async_copy.h>
#include <fused_kernel/algorithms/collective/copy.h>
#include <fused_kernel/algorithms/collective/mma.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>

#if defined(__NVCC__)
#include <cuda_bf16.h>
#endif

using namespace fk;

namespace {

using MmaDetails = MmaDPPDetails<MmaBf16_16x8x16>;
using AsyncLayout = RowMajorLayout<1, 20>;
using AsyncDetails = AsyncCopyDPPDetails<float, AsyncLayout, 128>;

static_assert(MmaDetails::M == 16 && MmaDetails::N == 8 &&
              MmaDetails::K == 16);
static_assert(std::is_trivially_copyable_v<MmaDetails>);
static_assert(std::is_trivially_copyable_v<AsyncDetails>);
static_assert(MmaWarpDPP<ParArch::CPU, MmaDetails>::PAR_ARCH == ParArch::CPU);
static_assert(AsyncCopyDPP<ParArch::CPU, AsyncDetails>::PAR_ARCH ==
              ParArch::CPU);
#if defined(__NVCC__)
static_assert(MmaWarpDPP<ParArch::GPU_NVIDIA, MmaDetails>::PAR_ARCH ==
              ParArch::GPU_NVIDIA);
static_assert(AsyncCopyDPP<ParArch::GPU_NVIDIA, AsyncDetails>::PAR_ARCH ==
              ParArch::GPU_NVIDIA);
#endif

float valueA(const int row, const int k) {
    return static_cast<float>((row + 2 * k) % 7 - 3) * 0.25f;
}

float valueB(const int col, const int k) {
    return static_cast<float>((3 * col + k) % 5 - 2) * 0.5f;
}

double oracleMma(const int row, const int col) {
    double sum = 0.0;
    for (int k = 0; k < 16; ++k) {
        sum += static_cast<double>(valueA(row, k)) *
               static_cast<double>(valueB(col, k));
    }
    return sum;
}

template <typename Output>
bool checkMmaOutput(const Output& output, const char* label) {
    double maxError = 0.0;
    for (int row = 0; row < 16; ++row) {
        for (int pair = 0; pair < 4; ++pair) {
            const float2 got = output(row, pair);
            maxError = std::max(
                maxError,
                std::fabs(static_cast<double>(got.x) - oracleMma(row, 2 * pair)));
            maxError = std::max(
                maxError,
                std::fabs(static_cast<double>(got.y) - oracleMma(row, 2 * pair + 1)));
        }
    }
    if (maxError > 1e-4) {
        std::printf("%s MMA max error=%g\n", label, maxError);
        return false;
    }
    return true;
}

bool testCpuMma() {
    std::array<float, 16 * 16> a{};
    std::array<float, 8 * 16> b{};
    std::array<float2, 16 * 4> d{};
    for (int row = 0; row < 16; ++row)
        for (int k = 0; k < 16; ++k)
            a[row * 16 + k] = valueA(row, k);
    for (int col = 0; col < 8; ++col)
        for (int k = 0; k < 16; ++k)
            b[col * 16 + k] = valueB(col, k);

    const RawPtr<ND::_2D, float> aPtr{
        a.data(), PtrDims<ND::_2D>(16, 16, 16 * sizeof(float))};
    const RawPtr<ND::_2D, float> bPtr{
        b.data(), PtrDims<ND::_2D>(16, 8, 16 * sizeof(float))};
    const RawPtr<ND::_2D, float2> dPtr{
        d.data(), PtrDims<ND::_2D>(4, 16, 4 * sizeof(float2))};
    const auto readA = PerThreadRead<ND::_2D, float>::build(aPtr);
    const auto readB = PerThreadRead<ND::_2D, float>::build(bPtr);
    const auto writeD = PerThreadWrite<ND::_2D, float2>::build(dPtr);
    const MmaDetails details{0, 0, 0, 0, 0, 0};
    MmaWarpDPP<ParArch::CPU, MmaDetails>::exec(
        details, readA, readB, writeD);
    return checkMmaOutput(
        [&](const int row, const int pair) { return d[row * 4 + pair]; },
        "CPU");
}

template <typename ReadIOp>
bool testCpuAsyncRead(const ReadIOp& read, const float scale,
                      const float offset, const char* label) {
    std::array<float, AsyncLayout::size()> storage{};
    Tile<float, AsyncLayout> tile{storage.data()};
    const AsyncDetails details{4, 18, -7.f};
    AsyncCopyDPP<ParArch::CPU, AsyncDetails>::load(details, read, tile);
    for (int index = 0; index < static_cast<int>(AsyncLayout::size()); ++index) {
        const float expected = index < details.elementCount
            ? static_cast<float>(details.origin + index) * scale + offset
            : details.boundaryValue;
        if (std::fabs(tile.at(0, index) - expected) > 1e-6f) {
            std::printf("%s CPU async index=%d got=%f expected=%f\n",
                        label, index, tile.at(0, index), expected);
            return false;
        }
    }
    return true;
}

bool testCpuAsyncCopy() {
    std::array<float, 24> source{};
    for (int i = 0; i < 24; ++i) source[i] = static_cast<float>(i);
    const RawPtr<ND::_1D, float> sourcePtr{
        source.data(), PtrDims<ND::_1D>(24, 24 * sizeof(float))};
    const auto plain = PerThreadRead<ND::_1D, float>::build(sourcePtr);
    const auto fused = plain.then(Mul<float>::build(2.f))
                            .then(Add<float>::build(1.f));
    return testCpuAsyncRead(plain, 1.f, 0.f, "plain") &&
           testCpuAsyncRead(fused, 2.f, 1.f, "fused");
}

#if defined(__NVCC__)
template <typename D, typename AReadIOp, typename BReadIOp, typename WriteIOp>
__global__ void mmaKernel(const __grid_constant__ D details,
                          const __grid_constant__ AReadIOp a,
                          const __grid_constant__ BReadIOp b,
                          const __grid_constant__ WriteIOp d) {
    MmaWarpDPP<ParArch::GPU_NVIDIA, D>::exec(details, a, b, d);
}

template <typename D, typename ReadIOp, typename WriteIOp>
__global__ void asyncCopyKernel(const __grid_constant__ D details,
                                const __grid_constant__ ReadIOp input,
                                const __grid_constant__ WriteIOp output) {
    __shared__ __align__(16) float storage[AsyncLayout::size()];
    Tile<float, AsyncLayout> tile{storage};
    AsyncCopyDPP<ParArch::GPU_NVIDIA, D>::load(details, input, tile);
    using StoreDetails = CopyTileDPPDetails<float, AsyncLayout, 128>;
    const StoreDetails storeDetails{0, 0, 20, 1, 0.f};
    CopyTileDPP<ParArch::GPU_NVIDIA, StoreDetails>::store(
        storeDetails, tile, output);
}

bool testGpuMma() {
    Ptr2D<uint16_t> a(16, 16);
    Ptr2D<uint16_t> b(16, 8);
    Ptr2D<float2> d(4, 16);
    for (int row = 0; row < 16; ++row) {
        for (int k = 0; k < 16; ++k) {
            const __nv_bfloat16 value = __float2bfloat16(valueA(row, k));
            uint16_t bits;
            std::memcpy(&bits, &value, sizeof(bits));
            a.at(Point{k, row, 0}) = bits;
        }
    }
    for (int col = 0; col < 8; ++col) {
        for (int k = 0; k < 16; ++k) {
            const __nv_bfloat16 value = __float2bfloat16(valueB(col, k));
            uint16_t bits;
            std::memcpy(&bits, &value, sizeof(bits));
            b.at(Point{k, col, 0}) = bits;
        }
    }

    Stream stream;
    a.upload(stream);
    b.upload(stream);
    const auto readA = PerThreadRead<ND::_2D, uint16_t>::build(a);
    const auto readB = PerThreadRead<ND::_2D, uint16_t>::build(b);
    const auto writeD = PerThreadWrite<ND::_2D, float2>::build(d);
    const MmaDetails details{0, 0, 0, 0, 0, 0};
    mmaKernel<<<1, 32, 0, stream.getCUDAStream()>>>(
        details, readA, readB, writeD);
    gpuErrchk(cudaGetLastError());
    d.download(stream);
    stream.sync();
    return checkMmaOutput(
        [&](const int row, const int pair) {
            return d.at(Point{pair, row, 0});
        },
        "GPU");
}

template <typename ReadIOp>
bool testGpuAsyncRead(const ReadIOp& read, Ptr2D<float>& output,
                      const float scale, const float offset,
                      const char* label, Stream& stream) {
    for (int index = 0; index < 20; ++index)
        output.at(Point{index, 0, 0}) = -99.f;
    output.upload(stream);
    const auto write = PerThreadWrite<ND::_2D, float>::build(output);
    const AsyncDetails details{4, 18, -7.f};
    asyncCopyKernel<<<1, AsyncDetails::BLOCK_THREADS, 0,
                      stream.getCUDAStream()>>>(details, read, write);
    gpuErrchk(cudaGetLastError());
    output.download(stream);
    stream.sync();
    for (int index = 0; index < 20; ++index) {
        const float expected = index < details.elementCount
            ? static_cast<float>(details.origin + index) * scale + offset
            : details.boundaryValue;
        const float got = output.at(Point{index, 0, 0});
        if (std::fabs(got - expected) > 1e-6f) {
            std::printf("%s GPU async index=%d got=%f expected=%f\n",
                        label, index, got, expected);
            return false;
        }
    }
    return true;
}

bool testGpuAsyncCopy() {
    Ptr1D<float> source(24);
    Ptr2D<float> output(20, 1);
    for (int i = 0; i < 24; ++i) source.at(Point{i, 0, 0}) = static_cast<float>(i);
    Stream stream;
    source.upload(stream);
    const auto plain = PerThreadRead<ND::_1D, float>::build(source);
    const auto fused = plain.then(Mul<float>::build(2.f))
                            .then(Add<float>::build(1.f));
    return testGpuAsyncRead(plain, output, 1.f, 0.f, "plain", stream) &&
           testGpuAsyncRead(fused, output, 2.f, 1.f, "fused", stream);
}
#endif

} // namespace

int launch() {
    bool ok = testCpuMma() && testCpuAsyncCopy();
#if defined(__NVCC__)
    ok = testGpuMma() && ok;
    ok = testGpuAsyncCopy() && ok;
#endif
    if (ok) std::printf("MMA + AsyncCopy DPP contracts: PASS\n");
    return ok ? 0 : -1;
}
