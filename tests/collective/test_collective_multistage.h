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
#include <fused_kernel/algorithms/collective/mainloop.h>
#include <fused_kernel/algorithms/collective/multistage.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

#if defined(__NVCC__)
#include <cuda_bf16.h>
#endif

using namespace fk;

namespace {

using AtomDetails = MmaDPPDetails<MmaBf16_16x8x16>;
using ALayout = RowMajorLayout<16, 16>;
using BLayout = RowMajorLayout<8, 16>;
using Details = MultiStageMainloopDPPDetails<
    4, AtomDetails, ALayout, BLayout, float,
    MmaWarpDPP, AsyncCopyDPP>;

static_assert(Details::MAX_STAGES == 4);
static_assert(Details::SHARED_ELEMENTS ==
              4 * (ALayout::size() + BLayout::size()));

float rawA(const int row, const int k) {
    return ((row * 5 + k * 3) % 17 - 8) * 0.0625f;
}

float rawB(const int col, const int k) {
    return ((col * 7 + k * 2) % 13 - 6) * 0.078125f;
}

float effective(const float value, const bool gpu) {
#if defined(__NVCC__)
    if (gpu) return __bfloat162float(__float2bfloat16(value));
#else
    (void)gpu;
#endif
    return value;
}

double oracle(const int row, const int col, const int K,
              const bool fused, const bool gpu) {
    double result = 0.0;
    for (int k = 0; k < K; ++k) {
        const float a = effective(
            fused ? rawA(row, k) * 1.25f : rawA(row, k), gpu);
        const float b = effective(
            fused ? rawB(col, k) - 0.125f : rawB(col, k), gpu);
        result += static_cast<double>(a) * static_cast<double>(b);
    }
    return fused ? result * 0.5 : result;
}

template <typename Get>
bool check(const int K, const bool fused, const bool gpu, const Get& get) {
    const double tolerance = gpu ? 0.05 * ((K + 15) / 16.0) + 1e-3 : 1e-5;
    double maxError = 0.0;
    for (int row = 0; row < 16; ++row)
        for (int col = 0; col < 8; ++col)
            maxError = std::max(maxError, std::abs(
                static_cast<double>(get(row, col)) -
                oracle(row, col, K, fused, gpu)));
    if (maxError > tolerance) {
        std::printf("multistage K=%d fused=%d gpu=%d maxError=%g tol=%g\n",
                    K, fused, gpu, maxError, tolerance);
        return false;
    }
    return true;
}

template <bool FUSED>
bool runCpu(const int stages, const int K) {
    std::vector<float> a(16 * K), b(8 * K);
    std::array<float2, 16 * 4> output{};
    for (int row = 0; row < 16; ++row)
        for (int k = 0; k < K; ++k) a[row * K + k] = rawA(row, k);
    for (int col = 0; col < 8; ++col)
        for (int k = 0; k < K; ++k) b[col * K + k] = rawB(col, k);
    const RawPtr<ND::_2D, float> aPtr{
        a.data(), PtrDims<ND::_2D>(K, 16, K * sizeof(float))};
    const RawPtr<ND::_2D, float> bPtr{
        b.data(), PtrDims<ND::_2D>(K, 8, K * sizeof(float))};
    const RawPtr<ND::_2D, float2> dPtr{
        output.data(), PtrDims<ND::_2D>(4, 16, 4 * sizeof(float2))};
    const Details details{stages, K, 0, 0, 0, 0, 0, 0};
    const auto baseA = PerThreadRead<ND::_2D, float>::build(aPtr);
    const auto baseB = PerThreadRead<ND::_2D, float>::build(bPtr);
    const auto baseD = PerThreadWrite<ND::_2D, float2>::build(dPtr);
    if constexpr (FUSED) {
        const auto reads = make_tuple(
            baseA.then(Mul<float>::build(1.25f)),
            baseB.then(Add<float>::build(-0.125f)));
        const auto write = Mul<float2>::build(float2{0.5f, 0.5f})
            .then(baseD);
        MultiStageMainloopDPP<ParArch::CPU, Details>::exec(
            details, reads, write);
    } else {
        MultiStageMainloopDPP<ParArch::CPU, Details>::exec(
            details, make_tuple(baseA, baseB), baseD);
    }
    return check(K, FUSED, false,
        [&](const int row, const int col) {
            const float2 value = output[row * 4 + col / 2];
            return col & 1 ? value.y : value.x;
        });
}

bool invalidStagesRejected() {
    const Details tooFew{1, 17, 0, 0, 0, 0, 0, 0};
    const Details tooMany{5, 17, 0, 0, 0, 0, 0, 0};
    if (tooFew.valid() || tooMany.valid()) return false;
    std::vector<float> a(16 * 17, 1.f), b(8 * 17, 1.f);
    std::array<float2, 64> output;
    for (auto& value : output) value = float2{-77.f, -77.f};
    const RawPtr<ND::_2D, float> aPtr{
        a.data(), PtrDims<ND::_2D>(17, 16, 17 * sizeof(float))};
    const RawPtr<ND::_2D, float> bPtr{
        b.data(), PtrDims<ND::_2D>(17, 8, 17 * sizeof(float))};
    const RawPtr<ND::_2D, float2> dPtr{
        output.data(), PtrDims<ND::_2D>(4, 16, 4 * sizeof(float2))};
    const auto reads = make_tuple(
        PerThreadRead<ND::_2D, float>::build(aPtr),
        PerThreadRead<ND::_2D, float>::build(bPtr));
    const auto write = PerThreadWrite<ND::_2D, float2>::build(dPtr);
    MultiStageMainloopDPP<ParArch::CPU, Details>::exec(
        tooFew, reads, write);
    MultiStageMainloopDPP<ParArch::CPU, Details>::exec(
        tooMany, reads, write);
    for (const auto value : output)
        if (value.x != -77.f || value.y != -77.f) return false;
    return true;
}

bool singleStageCpuParity() {
    constexpr int K = 32;
    std::vector<float> a(16 * K), b(8 * K);
    std::array<float2, 64> output{};
    for (int row = 0; row < 16; ++row)
        for (int k = 0; k < K; ++k) a[row * K + k] = rawA(row, k);
    for (int col = 0; col < 8; ++col)
        for (int k = 0; k < K; ++k) b[col * K + k] = rawB(col, k);
    const RawPtr<ND::_2D, float> aPtr{
        a.data(), PtrDims<ND::_2D>(K, 16, K * sizeof(float))};
    const RawPtr<ND::_2D, float> bPtr{
        b.data(), PtrDims<ND::_2D>(K, 8, K * sizeof(float))};
    const RawPtr<ND::_2D, float2> dPtr{
        output.data(), PtrDims<ND::_2D>(4, 16, 4 * sizeof(float2))};
    using SingleDetails = TileMmaMainloopDPPDetails<AtomDetails>;
    TileMmaMainloopDPP<ParArch::CPU, SingleDetails>::exec(
        SingleDetails{K, 0, 0, 0, 0, 0, 0},
        make_tuple(PerThreadRead<ND::_2D, float>::build(aPtr),
                   PerThreadRead<ND::_2D, float>::build(bPtr)),
        PerThreadWrite<ND::_2D, float2>::build(dPtr));
    return check(K, false, false,
        [&](const int row, const int col) {
            const float2 value = output[row * 4 + col / 2];
            return col & 1 ? value.y : value.x;
        });
}

#if defined(__NVCC__)
template <typename Reads, typename Write>
__global__ void multistageKernel(const Details details,
                                 const Reads reads, const Write output) {
    MultiStageMainloopDPP<ParArch::GPU_NVIDIA, Details>::exec(
        details, reads, output);
}

template <bool FUSED>
bool runGpu(const int stages, const int K) {
    Ptr2D<float> a(K, 16), b(K, 8);
    Ptr2D<float2> output(4, 16);
    for (int row = 0; row < 16; ++row)
        for (int k = 0; k < K; ++k)
            a.at(Point{k, row, 0}) = rawA(row, k);
    for (int col = 0; col < 8; ++col)
        for (int k = 0; k < K; ++k)
            b.at(Point{k, col, 0}) = rawB(col, k);
    Stream stream;
    a.upload(stream);
    b.upload(stream);
    const Details details{stages, K, 0, 0, 0, 0, 0, 0};
    const auto baseA = PerThreadRead<ND::_2D, float>::build(a);
    const auto baseB = PerThreadRead<ND::_2D, float>::build(b);
    const auto baseD = PerThreadWrite<ND::_2D, float2>::build(output);
    if constexpr (FUSED) {
        const auto reads = make_tuple(
            baseA.then(Mul<float>::build(1.25f)),
            baseB.then(Add<float>::build(-0.125f)));
        const auto write = Mul<float2>::build(float2{0.5f, 0.5f})
            .then(baseD);
        multistageKernel<<<1, 32, 0, stream.getCUDAStream()>>>(
            details, reads, write);
    } else {
        const auto reads = make_tuple(baseA, baseB);
        multistageKernel<<<1, 32, 0, stream.getCUDAStream()>>>(
            details, reads, baseD);
    }
    gpuErrchk(cudaGetLastError());
    output.download(stream);
    stream.sync();
    return check(K, FUSED, true,
        [&](const int row, const int col) {
            const float2 value = output.at(Point{col / 2, row, 0});
            return col & 1 ? value.y : value.x;
        });
}
#endif

} // namespace

int launch() {
    bool ok = invalidStagesRejected();
    ok = singleStageCpuParity() && ok;
    for (const int stages : {2, 3, 4}) {
        ok = runCpu<false>(stages, 1) && ok;
        ok = runCpu<true>(stages, 31) && ok;
        ok = runCpu<false>(stages, 160) && ok;
#if defined(__NVCC__)
        ok = runGpu<false>(stages, 1) && ok;
        ok = runGpu<true>(stages, 31) && ok;
        ok = runGpu<false>(stages, 160) && ok;
#endif
    }
    if (ok) std::printf("MultiStageMainloopDPP contracts: PASS\n");
    return ok ? 0 : -1;
}
