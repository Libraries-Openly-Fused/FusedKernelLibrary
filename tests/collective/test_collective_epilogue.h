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
#include <fused_kernel/algorithms/basic_ops/logical.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/collective/epilogue.h>
#include <fused_kernel/algorithms/collective/mainloop.h>

#include <cmath>
#include <cstdio>
#include <vector>

#if defined(__NVCC__)
#include <cuda_bf16.h>
#endif

using namespace fk;

namespace {

using MmaDetails = MmaDPPDetails<MmaBf16_16x8x16>;
using MainloopDetails = TileMmaMainloopDPPDetails<MmaDetails>;

enum class EpilogueCase { BARE, AFFINE, RELU, LONG };

float transform(const EpilogueCase testCase, const float value) {
    if (testCase == EpilogueCase::AFFINE) return 2.f * value - 0.5f;
    if (testCase == EpilogueCase::RELU) return value > 0.f ? value : 0.f;
    if (testCase == EpilogueCase::LONG) {
        const float affine = 2.f * value - 0.5f;
        return 0.25f * (affine > 0.f ? affine : 0.f);
    }
    return value;
}

template <EpilogueCase CASE, typename DPP, typename Reads, typename Write>
void executeMainloop(const MainloopDetails& details,
                     const Reads& reads, const Write& write) {
    if constexpr (CASE == EpilogueCase::BARE) {
        DPP::exec(details, reads, write);
    } else if constexpr (CASE == EpilogueCase::AFFINE) {
        DPP::exec(details, reads,
            scaleBiasEpilogue<float2>(
                float2{2.f, 2.f}, float2{-0.5f, -0.5f}, write));
    } else if constexpr (CASE == EpilogueCase::RELU) {
        DPP::exec(details, reads, reluEpilogue<float2>(write));
    } else {
        const auto chain = Mul<float2>::build(float2{2.f, 2.f})
            .then(Add<float2>::build(float2{-0.5f, -0.5f}))
            .then(Max<float2>::build(float2{0.f, 0.f}))
            .then(Mul<float2>::build(float2{0.25f, 0.25f}))
            .then(write);
        DPP::exec(details, reads, chain);
    }
}

#if defined(__NVCC__)
template <typename Reads, typename Write>
__global__ void mainloopKernel(const MainloopDetails details,
                               const Reads reads, const Write write) {
    TileMmaMainloopDPP<ParArch::GPU_NVIDIA, MainloopDetails>::exec(
        details, reads, write);
}
#endif

void initialize(std::vector<float>& a, std::vector<float>& b, const int k) {
    for (int row = 0; row < 16; ++row)
        for (int x = 0; x < k; ++x)
            a[row * k + x] = ((row * 7 + x * 3) % 17 - 8) * 0.0625f;
    for (int col = 0; col < 8; ++col)
        for (int x = 0; x < k; ++x)
            b[col * k + x] = ((col * 5 + x * 2) % 13 - 6) * 0.078125f;
}

double oracle(const std::vector<float>& a, const std::vector<float>& b,
              const int k, const int row, const int col,
              const bool gpuBf16) {
    double sum = 0.0;
    for (int x = 0; x < k; ++x) {
        float av = a[row * k + x];
        float bv = b[col * k + x];
#if defined(__NVCC__)
        if (gpuBf16) {
            av = __bfloat162float(__float2bfloat16(av));
            bv = __bfloat162float(__float2bfloat16(bv));
        }
#else
        (void)gpuBf16;
#endif
        sum += static_cast<double>(av) * static_cast<double>(bv);
    }
    return sum;
}

template <EpilogueCase CASE>
bool checkCpuMainloop() {
    constexpr int K = 32;
    std::vector<float> a(16 * K), b(8 * K), output(16 * 8, -99.f);
    initialize(a, b, K);
    const RawPtr<ND::_2D, float> aPtr{
        a.data(), PtrDims<ND::_2D>(K, 16, K * sizeof(float))};
    const RawPtr<ND::_2D, float> bPtr{
        b.data(), PtrDims<ND::_2D>(K, 8, K * sizeof(float))};
    const RawPtr<ND::_2D, float2> dPtr{
        reinterpret_cast<float2*>(output.data()),
        PtrDims<ND::_2D>(4, 16, 8 * sizeof(float))};
    const auto reads = make_tuple(
        PerThreadRead<ND::_2D, float>::build(aPtr),
        PerThreadRead<ND::_2D, float>::build(bPtr));
    const auto write = PerThreadWrite<ND::_2D, float2>::build(dPtr);
    const MainloopDetails details{K, 0, 0, 0, 0, 0, 0};
    executeMainloop<CASE,
        TileMmaMainloopDPP<ParArch::CPU, MainloopDetails>>(
            details, reads, write);
    for (int row = 0; row < 16; ++row)
        for (int col = 0; col < 8; ++col) {
            const double expected = transform(
                CASE, static_cast<float>(oracle(a, b, K, row, col, false)));
            if (std::abs(output[row * 8 + col] - expected) > 1e-5)
                return false;
        }
    return true;
}

#if defined(__NVCC__)
template <EpilogueCase CASE>
bool checkGpuMainloop() {
    constexpr int K = 32;
    std::vector<float> a(16 * K), b(8 * K);
    initialize(a, b, K);
    Ptr2D<float> aGpu(K, 16), bGpu(K, 8);
    Ptr2D<float> outputGpu(8, 16);
    for (int row = 0; row < 16; ++row)
        for (int x = 0; x < K; ++x)
            aGpu.at(Point{x, row, 0}) = a[row * K + x];
    for (int row = 0; row < 8; ++row)
        for (int x = 0; x < K; ++x)
            bGpu.at(Point{x, row, 0}) = b[row * K + x];
    Stream stream;
    aGpu.upload(stream);
    bGpu.upload(stream);
    const auto reads = make_tuple(
        PerThreadRead<ND::_2D, float>::build(aGpu),
        PerThreadRead<ND::_2D, float>::build(bGpu));
    RawPtr<ND::_2D, float2> dPtr{
        reinterpret_cast<float2*>(outputGpu.ptr().data),
        PtrDims<ND::_2D>(4, 16, outputGpu.ptr().dims.pitch)};
    const auto write = PerThreadWrite<ND::_2D, float2>::build(dPtr);
    const MainloopDetails details{K, 0, 0, 0, 0, 0, 0};
    if constexpr (CASE == EpilogueCase::BARE) {
        mainloopKernel<<<1, 32>>>(details, reads, write);
    } else if constexpr (CASE == EpilogueCase::AFFINE) {
        const auto output = scaleBiasEpilogue<float2>(
            float2{2.f, 2.f}, float2{-0.5f, -0.5f}, write);
        mainloopKernel<<<1, 32>>>(details, reads, output);
    } else if constexpr (CASE == EpilogueCase::RELU) {
        const auto output = reluEpilogue<float2>(write);
        mainloopKernel<<<1, 32>>>(details, reads, output);
    } else {
        const auto output = Mul<float2>::build(float2{2.f, 2.f})
            .then(Add<float2>::build(float2{-0.5f, -0.5f}))
            .then(Max<float2>::build(float2{0.f, 0.f}))
            .then(Mul<float2>::build(float2{0.25f, 0.25f}))
            .then(write);
        mainloopKernel<<<1, 32>>>(details, reads, output);
    }
    outputGpu.download(stream);
    stream.sync();
    for (int row = 0; row < 16; ++row)
        for (int col = 0; col < 8; ++col) {
            const float expected = transform(
                CASE, static_cast<float>(oracle(a, b, K, row, col, true)));
            if (std::abs(outputGpu.at(Point{col, row, 0}) - expected) > 0.02f)
                return false;
        }
    return true;
}
#endif

struct PairTailDetails { int width; };

template <ParArch PA, typename Details>
struct PairTailStoreDPP;

template <typename Details>
struct PairTailStoreDPP<ParArch::CPU, Details> {
    template <typename Read, typename PairWrite, typename TailWrite>
    FK_HOST_STATIC void exec(const Details& details, const Read& input,
                             const PairWrite& pairs, const TailWrite& tail) {
        for (int pair = 0; pair < details.width / 2; ++pair) {
            const float2 value{
                Read::Operation::exec(Point{2 * pair, 0, 0}, input),
                Read::Operation::exec(Point{2 * pair + 1, 0, 0}, input)};
            PairWrite::Operation::exec(Point{pair, 0, 0}, value, pairs);
        }
        if (details.width & 1) {
            TailWrite::Operation::exec(
                Point{0, 0, 0},
                Read::Operation::exec(Point{details.width - 1, 0, 0}, input),
                tail);
        }
    }
};

#if defined(__NVCC__)
template <typename Details>
struct PairTailStoreDPP<ParArch::GPU_NVIDIA, Details> {
    template <typename Read, typename PairWrite, typename TailWrite>
    FK_DEVICE_STATIC void exec(const Details& details, const Read& input,
                               const PairWrite& pairs, const TailWrite& tail) {
        const int pair = threadIdx.x;
        if (pair < details.width / 2) {
            const float2 value{
                Read::Operation::exec(Point{2 * pair, 0, 0}, input),
                Read::Operation::exec(Point{2 * pair + 1, 0, 0}, input)};
            PairWrite::Operation::exec(Point{pair, 0, 0}, value, pairs);
        }
        if (threadIdx.x == 0 && (details.width & 1)) {
            TailWrite::Operation::exec(
                Point{0, 0, 0},
                Read::Operation::exec(Point{details.width - 1, 0, 0}, input),
                tail);
        }
    }
};

template <typename Read, typename PairWrite, typename TailWrite>
__global__ void pairTailKernel(const PairTailDetails details,
                               const Read input,
                               const PairWrite pairs,
                               const TailWrite tail) {
    PairTailStoreDPP<ParArch::GPU_NVIDIA, PairTailDetails>::exec(
        details, input, pairs, tail);
}
#endif

bool checkPairTail() {
    constexpr int WIDTH = 9;
    std::vector<float> input(WIDTH), output(WIDTH, -99.f);
    for (int i = 0; i < WIDTH; ++i) input[i] = i * 0.5f - 1.f;
    const RawPtr<ND::_2D, float> inPtr{
        input.data(), PtrDims<ND::_2D>(WIDTH, 1, WIDTH * sizeof(float))};
    const RawPtr<ND::_2D, float2> pairPtr{
        reinterpret_cast<float2*>(output.data()),
        PtrDims<ND::_2D>(WIDTH / 2, 1, WIDTH * sizeof(float))};
    const RawPtr<ND::_2D, float> tailPtr{
        output.data() + WIDTH - 1,
        PtrDims<ND::_2D>(1, 1, sizeof(float))};
    const auto read = PerThreadRead<ND::_2D, float>::build(inPtr);
    const auto pairWrite = scaleBiasEpilogue<float2>(
        float2{2.f, 2.f}, float2{1.f, 1.f},
        PerThreadWrite<ND::_2D, float2>::build(pairPtr));
    const auto tailWrite = scaleBiasEpilogue<float>(
        2.f, 1.f, PerThreadWrite<ND::_2D, float>::build(tailPtr));
    PairTailStoreDPP<ParArch::CPU, PairTailDetails>::exec(
        PairTailDetails{WIDTH}, read, pairWrite, tailWrite);
    for (int i = 0; i < WIDTH; ++i)
        if (output[i] != 2.f * input[i] + 1.f) return false;

#if defined(__NVCC__)
    Ptr2D<float> gpuInput(WIDTH, 1), gpuOutput(WIDTH, 1);
    for (int i = 0; i < WIDTH; ++i)
        gpuInput.at(Point{i, 0, 0}) = input[i];
    Stream stream;
    gpuInput.upload(stream);
    const auto gpuRead = PerThreadRead<ND::_2D, float>::build(gpuInput);
    RawPtr<ND::_2D, float2> gpuPairPtr{
        reinterpret_cast<float2*>(gpuOutput.ptr().data),
        PtrDims<ND::_2D>(WIDTH / 2, 1, gpuOutput.ptr().dims.pitch)};
    RawPtr<ND::_2D, float> gpuTailPtr{
        gpuOutput.ptr().data + WIDTH - 1,
        PtrDims<ND::_2D>(1, 1, sizeof(float))};
    const auto gpuPairs = scaleBiasEpilogue<float2>(
        float2{2.f, 2.f}, float2{1.f, 1.f},
        PerThreadWrite<ND::_2D, float2>::build(gpuPairPtr));
    const auto gpuTail = scaleBiasEpilogue<float>(
        2.f, 1.f, PerThreadWrite<ND::_2D, float>::build(gpuTailPtr));
    pairTailKernel<<<1, 32>>>(
        PairTailDetails{WIDTH}, gpuRead, gpuPairs, gpuTail);
    gpuOutput.download(stream);
    stream.sync();
    for (int i = 0; i < WIDTH; ++i)
        if (gpuOutput.at(Point{i, 0, 0}) != 2.f * input[i] + 1.f)
            return false;
#endif
    return true;
}

} // namespace

int launch() {
    bool ok = checkCpuMainloop<EpilogueCase::BARE>();
    ok = checkCpuMainloop<EpilogueCase::AFFINE>() && ok;
    ok = checkCpuMainloop<EpilogueCase::RELU>() && ok;
    ok = checkCpuMainloop<EpilogueCase::LONG>() && ok;
    ok = checkPairTail() && ok;
#if defined(__NVCC__)
    ok = checkGpuMainloop<EpilogueCase::BARE>() && ok;
    ok = checkGpuMainloop<EpilogueCase::AFFINE>() && ok;
    ok = checkGpuMainloop<EpilogueCase::RELU>() && ok;
    ok = checkGpuMainloop<EpilogueCase::LONG>() && ok;
#endif
    if (ok) std::printf("Fused Write-IOp epilogues: PASS\n");
    return ok ? 0 : -1;
}
