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

#include <cmath>
#include <cstdio>
#include <type_traits>
#include <utility>
#include <vector>

using namespace fk;

namespace {

using AtomDetails = MmaDPPDetails<MmaBf16_16x8x16>;
using ALayout = RowMajorLayout<16, 16>;
using BLayout = RowMajorLayout<8, 16>;
using PipelineDetails = TileMmaMainloopPipelinedDPPDetails<
    AtomDetails, ALayout, BLayout>;
using PlainDetails = TileMmaMainloopDPPDetails<AtomDetails>;

struct ScaleFloat2 {
private:
    using SelfType = ScaleFloat2;

public:
    FK_STATIC_STRUCT(ScaleFloat2, SelfType)
    using Parent = BinaryOperation<float2, float, float2, SelfType>;
    DECLARE_BINARY_PARENT

    FK_HOST_DEVICE_FUSE OutputType exec(const InputType input,
                                        const ParamsType& scale) {
        return float2{input.x * scale, input.y * scale};
    }
};

template <typename DPP, typename Details, typename Reads,
          typename Write, typename = void>
struct HasThreeArgExec : std::false_type {};

template <typename DPP, typename Details, typename Reads, typename Write>
struct HasThreeArgExec<DPP, Details, Reads, Write,
    std::void_t<decltype(DPP::exec(std::declval<const Details&>(),
                                  std::declval<const Reads&>(),
                                  std::declval<const Write&>()))>>
    : std::true_type {};

using HostRead = decltype(PerThreadRead<ND::_2D, float>::build(
    std::declval<RawPtr<ND::_2D, float>>()));
using HostWrite = decltype(PerThreadWrite<ND::_2D, float2>::build(
    std::declval<RawPtr<ND::_2D, float2>>()));
using HostReads = Tuple<HostRead, HostRead>;
static_assert(HasThreeArgExec<
    TileMmaMainloopPipelinedDPP<ParArch::CPU, PipelineDetails>,
    PipelineDetails, HostReads, HostWrite>::value);
static_assert(PipelineDetails::BUFFER_COUNT == 2);

float aValue(const int row, const int k) {
    return static_cast<float>((row * 3 + k * 5) % 9 - 4) * 0.25f;
}

float bValue(const int col, const int k) {
    return static_cast<float>((col * 7 + k * 2) % 7 - 3) * 0.25f;
}

std::vector<double> oracle(const int K, const bool fused) {
    std::vector<double> output(16 * 8, 0.0);
    for (int row = 0; row < 16; ++row) {
        for (int col = 0; col < 8; ++col) {
            double value = 0.0;
            for (int k = 0; k < K; ++k) {
                const double a = aValue(row, k) * (fused ? 2.0 : 1.0);
                const double b = bValue(col, k) + (fused ? 0.25 : 0.0);
                value += a * b;
            }
            output[row * 8 + col] = value * (fused ? 0.5 : 1.0);
        }
    }
    return output;
}

template <typename ReadA, typename ReadB, typename Write>
bool runCpu(const int K, const bool fused,
            const ReadA& a, const ReadB& b, const Write& d,
            const std::vector<double>& expected) {
    const PipelineDetails details{K, 0, 0, 0, 0, 0, 0};
    TileMmaMainloopPipelinedDPP<ParArch::CPU, PipelineDetails>::exec(
        details, make_tuple(a, b), d);
    (void)fused;
    return true;
}

#if defined(__NVCC__)
template <typename ReadA, typename ReadB, typename Write>
__global__ void pipelinedKernel(const PipelineDetails details,
                                const ReadA a, const ReadB b,
                                const Write d) {
    TileMmaMainloopPipelinedDPP<ParArch::GPU_NVIDIA,
                                PipelineDetails>::exec(
        details, make_tuple(a, b), d);
}

template <typename ReadA, typename ReadB, typename Write>
__global__ void plainKernel(const PlainDetails details,
                            const ReadA a, const ReadB b,
                            const Write d) {
    TileMmaMainloopDPP<ParArch::GPU_NVIDIA, PlainDetails>::exec(
        details, make_tuple(a, b), d);
}
#endif

bool runCase(const int K, const bool fused) {
    std::vector<float> hostA(16 * K);
    std::vector<float> hostB(8 * K);
    std::vector<float2> hostD(16 * 4, float2{-999.f, -999.f});
    for (int row = 0; row < 16; ++row)
        for (int k = 0; k < K; ++k)
            hostA[row * K + k] = aValue(row, k);
    for (int col = 0; col < 8; ++col)
        for (int k = 0; k < K; ++k)
            hostB[col * K + k] = bValue(col, k);
    const auto expected = oracle(K, fused);

    const RawPtr<ND::_2D, float> aPtr{
        hostA.data(), PtrDims<ND::_2D>(K, 16, K * sizeof(float))};
    const RawPtr<ND::_2D, float> bPtr{
        hostB.data(), PtrDims<ND::_2D>(K, 8, K * sizeof(float))};
    const RawPtr<ND::_2D, float2> dPtr{
        hostD.data(), PtrDims<ND::_2D>(4, 16, 4 * sizeof(float2))};
    const auto aBase = PerThreadRead<ND::_2D, float>::build(aPtr);
    const auto bBase = PerThreadRead<ND::_2D, float>::build(bPtr);
    const auto dBase = PerThreadWrite<ND::_2D, float2>::build(dPtr);

    if (fused) {
        const auto aRead = aBase.then(Mul<float>::build(2.f));
        const auto bRead = bBase.then(Add<float>::build(0.25f));
        const auto dWrite = ScaleFloat2::build(0.5f).then(dBase);
        runCpu(K, fused, aRead, bRead, dWrite, expected);
    } else {
        runCpu(K, fused, aBase, bBase, dBase, expected);
    }
    for (int row = 0; row < 16; ++row) {
        for (int pair = 0; pair < 4; ++pair) {
            const float2 got = hostD[row * 4 + pair];
            const double e0 = expected[row * 8 + 2 * pair];
            const double e1 = expected[row * 8 + 2 * pair + 1];
            if (std::fabs(got.x - e0) > 1e-5 ||
                std::fabs(got.y - e1) > 1e-5) {
                std::printf("CPU pipeline K=%d fused=%d row=%d pair=%d "
                            "got=(%g,%g) expected=(%g,%g)\n",
                            K, fused, row, pair, got.x, got.y, e0, e1);
                return false;
            }
        }
    }

#if defined(__NVCC__)
    Ptr2D<float> deviceA(K, 16);
    Ptr2D<float> deviceB(K, 8);
    Ptr2D<float2> pipelineD(4, 16);
    Ptr2D<float2> plainD(4, 16);
    for (int row = 0; row < 16; ++row)
        for (int k = 0; k < K; ++k)
            deviceA.at(Point{k, row, 0}) = hostA[row * K + k];
    for (int col = 0; col < 8; ++col)
        for (int k = 0; k < K; ++k)
            deviceB.at(Point{k, col, 0}) = hostB[col * K + k];
    Stream stream;
    deviceA.upload(stream);
    deviceB.upload(stream);
    const auto gpuABase = PerThreadRead<ND::_2D, float>::build(deviceA);
    const auto gpuBBase = PerThreadRead<ND::_2D, float>::build(deviceB);
    const auto gpuPipelineBase =
        PerThreadWrite<ND::_2D, float2>::build(pipelineD);
    const auto gpuPlainBase = PerThreadWrite<ND::_2D, float2>::build(plainD);
    const PipelineDetails pipelineDetails{K, 0, 0, 0, 0, 0, 0};
    const PlainDetails plainDetails{K, 0, 0, 0, 0, 0, 0};

    if (fused) {
        const auto aRead = gpuABase.then(Mul<float>::build(2.f));
        const auto bRead = gpuBBase.then(Add<float>::build(0.25f));
        const auto pipelineWrite =
            ScaleFloat2::build(0.5f).then(gpuPipelineBase);
        pipelinedKernel<<<1, 32, 0, stream.getCUDAStream()>>>(
            pipelineDetails, aRead, bRead, pipelineWrite);
        if (K % 16 == 0) {
            const auto plainWrite =
                ScaleFloat2::build(0.5f).then(gpuPlainBase);
            plainKernel<<<1, 32, 0, stream.getCUDAStream()>>>(
                plainDetails, aRead, bRead, plainWrite);
        }
    } else {
        pipelinedKernel<<<1, 32, 0, stream.getCUDAStream()>>>(
            pipelineDetails, gpuABase, gpuBBase, gpuPipelineBase);
        if (K % 16 == 0)
            plainKernel<<<1, 32, 0, stream.getCUDAStream()>>>(
                plainDetails, gpuABase, gpuBBase, gpuPlainBase);
    }
    gpuErrchk(cudaGetLastError());
    pipelineD.download(stream);
    if (K % 16 == 0) plainD.download(stream);
    stream.sync();

    for (int row = 0; row < 16; ++row) {
        for (int pair = 0; pair < 4; ++pair) {
            const float2 got = pipelineD.at(Point{pair, row, 0});
            const double e0 = expected[row * 8 + 2 * pair];
            const double e1 = expected[row * 8 + 2 * pair + 1];
            if (std::fabs(got.x - e0) > 2e-4 ||
                std::fabs(got.y - e1) > 2e-4) {
                std::printf("GPU pipeline K=%d fused=%d row=%d pair=%d "
                            "got=(%g,%g) expected=(%g,%g)\n",
                            K, fused, row, pair, got.x, got.y, e0, e1);
                return false;
            }
            if (K % 16 == 0) {
                const float2 plain = plainD.at(Point{pair, row, 0});
                if (std::fabs(plain.x - e0) > 2e-4 ||
                    std::fabs(plain.y - e1) > 2e-4) {
                    std::printf("GPU plain K=%d fused=%d mismatch\n", K, fused);
                    return false;
                }
            }
        }
    }
#endif
    return true;
}

} // namespace

int launch() {
    bool ok = true;
    ok = runCase(16, false) && ok;
    ok = runCase(32, false) && ok;
    ok = runCase(48, false) && ok;
    ok = runCase(128, false) && ok;
    ok = runCase(48, true) && ok;
    ok = runCase(20, false) && ok;
    if (ok) std::printf("Double-buffer MMA mainloop contracts: PASS\n");
    return ok ? 0 : -1;
}
