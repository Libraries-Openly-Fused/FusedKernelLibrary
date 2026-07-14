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

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/logical.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/collective/epilogue.h>
#include <fused_kernel/algorithms/collective/gemm.h>
#include <fused_kernel/algorithms/image_processing/crop.h>
#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/execution_model/stream.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <type_traits>
#include <vector>

using namespace fk;

namespace {

struct MockDetails {
    int* calls;
};

struct MockDPP {
    static constexpr ParArch PAR_ARCH = ParArch::CPU;

    template <typename... IOps>
    FK_HOST_STATIC void exec(const MockDetails details, const IOps&... iOps) {
        static_assert(sizeof...(IOps) == 3,
                      "details must stay out of BackFuser; Read+Crop must fuse");
        ++*details.calls;
        const auto args = forward_as_tuple(iOps...);
        const auto& read = get<0>(args);
        const auto& scale = get<1>(args);
        const auto& output = get<2>(args);
        using Output = std::decay_t<decltype(output)>;
        const float value = (Point{0, 0, 0} | read).input;
        const float scaled = value | scale;
        Output::Operation::exec(Point{0, 0, 0}, scaled, output.params);
    }
};

bool checkGenericCpuExecutor() {
    Ptr2D<float> input(1, 1, 0, MemType::Host);
    Ptr2D<float> output(1, 1, 0, MemType::Host);
    input.at(Point{0, 0, 0}) = 2.f;
    output.at(Point{0, 0, 0}) = -1.f;
    int calls = 0;
    Stream_<ParArch::CPU> stream;
    executeOperations<MockDPP>(
        stream, MockDetails{&calls},
        PerThreadRead<ND::_2D, float>::build(input),
        Crop<>::build(Rect{0, 0, 1, 1}),
        Mul<float>::build(3.f),
        PerThreadWrite<ND::_2D, float>::build(output));
    return calls == 1 && std::abs(output.at(Point{0, 0, 0}) - 6.f) < 1e-6f;
}

#if defined(__NVCC__)
struct MockGpuDetails {
    int* calls;
};

struct MockGpuDPP {
    static constexpr ParArch PAR_ARCH = ParArch::GPU_NVIDIA;

    FK_HOST_FUSE DPPLaunchConfig launchConfig(const MockGpuDetails&) {
        return {1, 1, 1, 1, 1, 1, 0};
    }

    template <typename... IOps>
    FK_DEVICE_STATIC void exec(const MockGpuDetails details,
                               const IOps&...) {
        static_assert(sizeof...(IOps) == 1,
                      "the dummy IOp must cross BackFuser exactly once");
        atomicAdd(details.calls, 1);
    }
};

bool checkGenericGpuExecutor() {
    int* deviceCalls = nullptr;
    gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&deviceCalls), sizeof(int)));
    gpuErrchk(cudaMemset(deviceCalls, 0, sizeof(int)));
    Ptr1D<int> dummy(1, 0, MemType::Device);
    Stream stream;
    executeOperations<MockGpuDPP>(
        stream, MockGpuDetails{deviceCalls},
        PerThreadWrite<ND::_1D, int>::build(dummy));
    gpuErrchk(cudaStreamSynchronize(stream.getCUDAStream()));
    int calls = 0;
    gpuErrchk(cudaMemcpy(&calls, deviceCalls, sizeof(int),
                        cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(deviceCalls));
    return calls == 1;
}
#endif

enum class OutputCase { BARE, AFFINE, RELU };

using DefaultCtaScheduler =
    CtaTileScheduler<RowMajorCtaTileRaster>;

template <typename SchedulerOperation = DefaultCtaScheduler>
using ScheduledGemmConfig = GemmDPPDetails<
    3, MmaDPPDetails<MmaBf16_16x8x16>,
    RowMajorLayout<16, 16>, RowMajorLayout<8, 16>, float,
    SchedulerOperation, MmaWarpDPP, AsyncCopyDPP>;

using GemmConfig = ScheduledGemmConfig<>;

float sourceA(const int row, const int k) {
    return ((row * 11 + k * 7) % 23 - 11) * 0.0625f;
}

float sourceB(const int col, const int k) {
    return ((col * 13 + k * 5) % 19 - 9) * 0.0546875f;
}

template <OutputCase CASE>
double transformOutput(double value) {
    if constexpr (CASE == OutputCase::AFFINE) return value * 1.5 - 0.25;
    if constexpr (CASE == OutputCase::RELU) return std::max(value, 0.0);
    return value;
}

template <OutputCase CASE, bool GPU, bool FUSED_INPUTS = true>
bool checkOutput(const Ptr2D<float2>& output, const int M, const int N,
                 const int K) {
    double maxError = 0.0;
    int maxRow = -1, maxCol = -1;
    double maxActual = 0.0, maxExpected = 0.0;
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            double expected = 0.0;
            for (int k = 0; k < K; ++k) {
                float a = sourceA(row, k);
                float b = sourceB(col, k);
                if constexpr (FUSED_INPUTS) {
                    a *= 1.25f;
                    b -= 0.125f;
                }
#if defined(__NVCC__)
                if constexpr (GPU) {
                    a = __bfloat162float(__float2bfloat16(a));
                    b = __bfloat162float(__float2bfloat16(b));
                }
#endif
                expected += static_cast<double>(a) * static_cast<double>(b);
            }
            expected = transformOutput<CASE>(expected);
            const float2 packed = output.at(Point{col / 2, row, 0});
            const double actual = col & 1 ? packed.y : packed.x;
            const double error = std::abs(actual - expected);
            if (error > maxError) {
                maxError = error;
                maxRow = row;
                maxCol = col;
                maxActual = actual;
                maxExpected = expected;
            }
        }
    }
    const double tolerance = GPU ? 0.08 * ((K + 15) / 16.0) + 1e-3 : 1e-4;
    if (maxError > tolerance)
        std::printf("verify_fail case=%d M=%d N=%d K=%d row=%d col=%d actual=%g expected=%g max_error=%g tolerance=%g\n",
                    static_cast<int>(CASE), M, N, K, maxRow, maxCol,
                    maxActual, maxExpected, maxError, tolerance);
    return maxError <= tolerance;
}

void captureOutput(const Ptr2D<float2>& output, const int M, const int N,
                   std::vector<float>& captured) {
    captured.resize(static_cast<std::size_t>(M * N));
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            const float2 packed = output.at(Point{col / 2, row, 0});
            captured[static_cast<std::size_t>(row * N + col)] =
                col & 1 ? packed.y : packed.x;
        }
    }
}

template <OutputCase CASE, typename DPP, typename StreamType,
          typename Details, typename Inputs, typename Write>
void dispatchGemm(StreamType& stream, const Details& details,
                  const Inputs& inputs, const Write& write) {
    if constexpr (CASE == OutputCase::BARE) {
        executeOperations<DPP>(stream, details, inputs, write);
    } else if constexpr (CASE == OutputCase::AFFINE) {
        const auto output = scaleBiasEpilogue<float2>(
            float2{1.5f, 1.5f}, float2{-0.25f, -0.25f}, write);
        executeOperations<DPP>(stream, details, inputs, output);
    } else {
        const auto output = reluEpilogue<float2>(write);
        executeOperations<DPP>(stream, details, inputs, output);
    }
}

template <OutputCase CASE, typename SchedulerOperation = DefaultCtaScheduler>
bool runCpu(const int M, const int N, const int K,
            std::vector<float>* captured = nullptr) {
    Ptr2D<float> a(static_cast<uint>(K), static_cast<uint>(M),
                   0, MemType::Host);
    Ptr2D<float> b(static_cast<uint>(K), static_cast<uint>(N),
                   0, MemType::Host);
    Ptr2D<float2> output(static_cast<uint>((N + 1) / 2),
                         static_cast<uint>(M), 0, MemType::Host);
    for (int row = 0; row < M; ++row)
        for (int k = 0; k < K; ++k)
            a.at(Point{k, row, 0}) = sourceA(row, k);
    for (int col = 0; col < N; ++col)
        for (int k = 0; k < K; ++k)
            b.at(Point{k, col, 0}) = sourceB(col, k);
    for (int row = 0; row < M; ++row)
        for (int pair = 0; pair < (N + 1) / 2; ++pair)
            output.at(Point{pair, row, 0}) = float2{-999.f, -999.f};

    const auto inputs = make_tuple(
        PerThreadRead<ND::_2D, float>::build(a)
            .then(Mul<float>::build(1.25f)),
        PerThreadRead<ND::_2D, float>::build(b)
            .then(Add<float>::build(-0.125f)));
    const auto write = PerThreadWrite<ND::_2D, float2>::build(output);
    Stream_<ParArch::CPU> stream;
    using Config = ScheduledGemmConfig<SchedulerOperation>;
    using DPP = GemmDPP<ParArch::CPU, Config>;
    dispatchGemm<CASE, DPP>(stream, Config{M, N, K}, inputs, write);
    if (captured != nullptr) captureOutput(output, M, N, *captured);
    return checkOutput<CASE, false>(output, M, N, K);
}

#if defined(__NVCC__)
template <OutputCase CASE, bool FUSED_INPUTS = true,
          typename SchedulerOperation = DefaultCtaScheduler>
bool runGpu(const int M, const int N, const int K,
            std::vector<float>* captured = nullptr) {
    Ptr2D<float> a(K, M), b(K, N);
    Ptr2D<float2> output((N + 1) / 2, M);
    for (int row = 0; row < M; ++row)
        for (int k = 0; k < K; ++k)
            a.at(Point{k, row, 0}) = sourceA(row, k);
    for (int col = 0; col < N; ++col)
        for (int k = 0; k < K; ++k)
            b.at(Point{k, col, 0}) = sourceB(col, k);
    for (int row = 0; row < M; ++row)
        for (int pair = 0; pair < (N + 1) / 2; ++pair)
            output.at(Point{pair, row, 0}) = float2{-999.f, -999.f};

    Stream stream;
    a.upload(stream);
    b.upload(stream);
    output.upload(stream);
    const auto write = PerThreadWrite<ND::_2D, float2>::build(output);
    using Config = ScheduledGemmConfig<SchedulerOperation>;
    using DPP = GemmDPP<ParArch::GPU_NVIDIA, Config>;
    if constexpr (FUSED_INPUTS) {
        const auto inputs = make_tuple(
            PerThreadRead<ND::_2D, float>::build(a)
                .then(Mul<float>::build(1.25f)),
            PerThreadRead<ND::_2D, float>::build(b)
                .then(Add<float>::build(-0.125f)));
        dispatchGemm<CASE, DPP>(stream, Config{M, N, K}, inputs, write);
    } else {
        static_assert(CASE == OutputCase::BARE,
                      "raw aligned regression uses the bare epilogue");
        const auto inputs = make_tuple(
            PerThreadRead<ND::_2D, float>::build(a),
            PerThreadRead<ND::_2D, float>::build(b));
        executeOperations<DPP>(stream, Config{M, N, K}, inputs, write);
    }
    output.download(stream);
    stream.sync();
    if (captured != nullptr) captureOutput(output, M, N, *captured);
    return checkOutput<CASE, true, FUSED_INPUTS>(output, M, N, K);
}
#endif

template <OutputCase CASE>
bool runCases() {
    bool ok = true;
    ok = runCpu<CASE>(16, 8, 32) && ok;
    ok = runCpu<CASE>(13, 7, 17) && ok;
    ok = runCpu<CASE>(17, 9, 31) && ok;
    ok = runCpu<CASE>(33, 19, 16) && ok;
#if defined(__NVCC__)
    ok = runGpu<CASE>(16, 8, 32) && ok;
    ok = runGpu<CASE>(13, 7, 17) && ok;
    ok = runGpu<CASE>(17, 9, 31) && ok;
    ok = runGpu<CASE>(33, 19, 16) && ok;
#endif
    return ok;
}

template <typename SchedulerOperation>
bool runRasterCpuParityCase(const std::vector<float>& reference) {
    std::vector<float> actual;
    const bool valid = runCpu<OutputCase::BARE, SchedulerOperation>(
        33, 19, 16, &actual);
    return valid && actual == reference;
}

#if defined(__NVCC__)
template <typename SchedulerOperation>
bool runRasterGpuParityCase(const std::vector<float>& reference) {
    std::vector<float> actual;
    const bool valid = runGpu<OutputCase::BARE, true, SchedulerOperation>(
        33, 19, 16, &actual);
    return valid && actual == reference;
}
#endif

bool runRasterGemmCases() {
    using RowScheduler = CtaTileScheduler<RowMajorCtaTileRaster>;
    using ColumnScheduler = CtaTileScheduler<ColumnMajorCtaTileRaster>;
    using Group1Scheduler = CtaTileScheduler<GroupedCtaTileRaster<1>>;
    using Group2Scheduler = CtaTileScheduler<GroupedCtaTileRaster<2>>;
    using Group4Scheduler = CtaTileScheduler<GroupedCtaTileRaster<4>>;
    using Group16Scheduler = CtaTileScheduler<GroupedCtaTileRaster<16>>;

    std::vector<float> cpuReference;
    bool ok = runCpu<OutputCase::BARE, RowScheduler>(
        33, 19, 16, &cpuReference);
    ok = runRasterCpuParityCase<ColumnScheduler>(cpuReference) && ok;
    ok = runRasterCpuParityCase<Group1Scheduler>(cpuReference) && ok;
    ok = runRasterCpuParityCase<Group2Scheduler>(cpuReference) && ok;
    ok = runRasterCpuParityCase<Group4Scheduler>(cpuReference) && ok;
    ok = runRasterCpuParityCase<Group16Scheduler>(cpuReference) && ok;

#if defined(__NVCC__)
    std::vector<float> gpuReference;
    ok = runGpu<OutputCase::BARE, true, RowScheduler>(
        33, 19, 16, &gpuReference) && ok;
    ok = runRasterGpuParityCase<ColumnScheduler>(gpuReference) && ok;
    ok = runRasterGpuParityCase<Group1Scheduler>(gpuReference) && ok;
    ok = runRasterGpuParityCase<Group2Scheduler>(gpuReference) && ok;
    ok = runRasterGpuParityCase<Group4Scheduler>(gpuReference) && ok;
    ok = runRasterGpuParityCase<Group16Scheduler>(gpuReference) && ok;
#endif
    return ok;
}

} // namespace

int launch() {
    const bool mock = checkGenericCpuExecutor();
#if defined(__NVCC__)
    const bool mockGpu = checkGenericGpuExecutor();
    const bool rawAligned =
        runGpu<OutputCase::BARE, false>(16, 8, 32);
#else
    const bool mockGpu = true;
    const bool rawAligned = true;
#endif
    const bool bare = runCases<OutputCase::BARE>();
    const bool affine = runCases<OutputCase::AFFINE>();
    const bool relu = runCases<OutputCase::RELU>();
    const bool rasters = runRasterGemmCases();
    std::printf("mock_cpu=%d mock_gpu=%d raw_aligned=%d "
                "bare=%d affine=%d relu=%d rasters=%d\n",
                mock, mockGpu, rawAligned, bare, affine, relu, rasters);
    const bool ok = mock && mockGpu && rawAligned &&
                    bare && affine && relu && rasters;
    std::printf("Generic DPP executor + GemmDPP contracts: %s\n",
                ok ? "PASS" : "FAIL");
    return ok ? 0 : -1;
}
