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

#include <array>
#include <cmath>
#include <cstdio>
#include <type_traits>
#include <utility>
#include <vector>

using namespace fk;

namespace {

struct ScaleFloat2 {
private:
    using SelfType = ScaleFloat2;
public:
    FK_STATIC_STRUCT(ScaleFloat2, SelfType)
    using Parent = BinaryOperation<float2, float, float2, ScaleFloat2>;
    DECLARE_BINARY_PARENT
    FK_HOST_DEVICE_FUSE OutputType exec(const InputType input,
                                        const ParamsType& scale) {
        return float2{input.x * scale, input.y * scale};
    }
};

using AtomDetails = MmaDPPDetails<MmaBf16_16x8x16>;

template <ParArch PA, typename D>
struct DeterministicWrongMmaDPP;

template <typename D>
struct DeterministicWrongMmaDPP<ParArch::CPU, D> {
    using Good = MmaWarpDPP<ParArch::CPU, D>;
    using AccumulatorFragment = typename Good::AccumulatorFragment;
    FK_HOST_STATIC void clear(AccumulatorFragment& fragment) {
        Good::clear(fragment);
    }
    template <typename A, typename B>
    FK_HOST_STATIC void accumulate(const D&, const A&, const B&,
                                   AccumulatorFragment&) {}
    template <typename W>
    FK_HOST_STATIC void store(const D& details,
                              const AccumulatorFragment& fragment,
                              const W& output) {
        Good::store(details, fragment, output);
    }
};

#if defined(__NVCC__)
template <typename D>
struct DeterministicWrongMmaDPP<ParArch::GPU_NVIDIA, D> {
    using Good = MmaWarpDPP<ParArch::GPU_NVIDIA, D>;
    using AccumulatorFragment = typename Good::AccumulatorFragment;
    FK_DEVICE_STATIC void clear(AccumulatorFragment& fragment) {
        Good::clear(fragment);
    }
    template <typename A, typename B>
    FK_DEVICE_STATIC void accumulate(const D&, const A&, const B&,
                                     AccumulatorFragment&) {}
    template <typename W>
    FK_DEVICE_STATIC void store(const D& details,
                                const AccumulatorFragment& fragment,
                                const W& output) {
        Good::store(details, fragment, output);
    }
};
#endif

using MainloopDetails = TileMmaMainloopDPPDetails<AtomDetails, MmaWarpDPP>;

static_assert(MainloopDetails::M == 16 && MainloopDetails::N == 8 &&
              MainloopDetails::K_TILE == 16);
static_assert(std::is_trivially_copyable_v<MainloopDetails>);

template <typename DPP, typename Details, typename Reads, typename Write,
          typename = void>
struct HasThreeArgumentExec : std::false_type {};

template <typename DPP, typename Details, typename Reads, typename Write>
struct HasThreeArgumentExec<
    DPP, Details, Reads, Write,
    std::void_t<decltype(DPP::exec(
        std::declval<const Details&>(),
        std::declval<const Reads&>(),
        std::declval<const Write&>()))>> : std::true_type {};

float valueA(const int row, const int k) {
    return static_cast<float>((row + k) % 9 - 4) * 0.125f;
}

float valueB(const int col, const int k) {
    return static_cast<float>((2 * col + k) % 7 - 3) * 0.25f;
}

double oracle(const int row, const int col, const int K) {
    double sum = 0.0;
    for (int k = 0; k < K; ++k) {
        const double a = static_cast<double>(valueA(row, k) * 2.f);
        const double b = static_cast<double>(valueB(col, k) + 0.25f);
        sum += a * b;
    }
    return sum * 0.5;
}

template <typename GetOutput>
bool checkOutput(const GetOutput& getOutput, const int K, const char* label) {
    double maxError = 0.0;
    for (int row = 0; row < 16; ++row) {
        for (int pair = 0; pair < 4; ++pair) {
            const float2 got = getOutput(row, pair);
            maxError = std::max(maxError,
                std::fabs(static_cast<double>(got.x) - oracle(row, 2 * pair, K)));
            maxError = std::max(maxError,
                std::fabs(static_cast<double>(got.y) - oracle(row, 2 * pair + 1, K)));
        }
    }
    if (maxError > 1e-4) {
        std::printf("%s K=%d max error=%g\n", label, K, maxError);
        return false;
    }
    return true;
}

template <typename DetailsPolicy = MainloopDetails>
bool runCpu(const int K) {
    std::vector<float> a(16 * K);
    std::vector<float> b(8 * K);
    std::array<float2, 16 * 4> d{};
    for (int row = 0; row < 16; ++row)
        for (int k = 0; k < K; ++k)
            a[row * K + k] = valueA(row, k);
    for (int col = 0; col < 8; ++col)
        for (int k = 0; k < K; ++k)
            b[col * K + k] = valueB(col, k);

    const RawPtr<ND::_2D, float> aPtr{
        a.data(), PtrDims<ND::_2D>(K, 16, K * sizeof(float))};
    const RawPtr<ND::_2D, float> bPtr{
        b.data(), PtrDims<ND::_2D>(K, 8, K * sizeof(float))};
    const RawPtr<ND::_2D, float2> dPtr{
        d.data(), PtrDims<ND::_2D>(4, 16, 4 * sizeof(float2))};
    const auto readA = PerThreadRead<ND::_2D, float>::build(aPtr)
        .then(Mul<float>::build(2.f));
    const auto readB = PerThreadRead<ND::_2D, float>::build(bPtr)
        .then(Add<float>::build(0.25f));
    const auto reads = make_tuple(readA, readB);
    const auto write = ScaleFloat2::build(0.5f)
        .then(PerThreadWrite<ND::_2D, float2>::build(dPtr));
    const DetailsPolicy details{K, 0, 0, 0, 0, 0, 0};

    using DPP = TileMmaMainloopDPP<ParArch::CPU, DetailsPolicy>;
    static_assert(HasThreeArgumentExec<DPP, DetailsPolicy,
                  decltype(reads), decltype(write)>::value);
    DPP::exec(details, reads, write);
    return checkOutput(
        [&](const int row, const int pair) { return d[row * 4 + pair]; },
        K, "CPU");
}

#if defined(__NVCC__)
template <typename DetailsPolicy, typename Reads, typename Write>
__global__ void mainloopKernel(const __grid_constant__ DetailsPolicy details,
                               const __grid_constant__ Reads reads,
                               const __grid_constant__ Write output) {
    TileMmaMainloopDPP<ParArch::GPU_NVIDIA, DetailsPolicy>::exec(
        details, reads, output);
}

template <typename DetailsPolicy = MainloopDetails>
bool runGpu(const int K) {
    Ptr2D<float> a(K, 16);
    Ptr2D<float> b(K, 8);
    Ptr2D<float2> d(4, 16);
    for (int row = 0; row < 16; ++row)
        for (int k = 0; k < K; ++k)
            a.at(Point{k, row, 0}) = valueA(row, k);
    for (int col = 0; col < 8; ++col)
        for (int k = 0; k < K; ++k)
            b.at(Point{k, col, 0}) = valueB(col, k);

    Stream stream;
    a.upload(stream);
    b.upload(stream);
    const auto readA = PerThreadRead<ND::_2D, float>::build(a)
        .then(Mul<float>::build(2.f));
    const auto readB = PerThreadRead<ND::_2D, float>::build(b)
        .then(Add<float>::build(0.25f));
    const auto reads = make_tuple(readA, readB);
    const auto write = ScaleFloat2::build(0.5f)
        .then(PerThreadWrite<ND::_2D, float2>::build(d));
    const DetailsPolicy details{K, 0, 0, 0, 0, 0, 0};

    using DPP = TileMmaMainloopDPP<ParArch::GPU_NVIDIA, DetailsPolicy>;
    static_assert(HasThreeArgumentExec<DPP, DetailsPolicy,
                  decltype(reads), decltype(write)>::value);
    mainloopKernel<<<1, 32, 0, stream.getCUDAStream()>>>(
        details, reads, write);
    gpuErrchk(cudaGetLastError());
    d.download(stream);
    stream.sync();
    return checkOutput(
        [&](const int row, const int pair) {
            return d.at(Point{pair, row, 0});
        }, K, "GPU");
}
#endif

} // namespace

int launch() {
    bool ok = true;
    for (const int K : {16, 48, 128}) ok = runCpu(K) && ok;
#if defined(__NVCC__)
    for (const int K : {16, 48, 128}) ok = runGpu(K) && ok;
#endif
    if (ok) std::printf("TileMmaMainloopDPP contracts: PASS\n");
    return ok ? 0 : -1;
}
