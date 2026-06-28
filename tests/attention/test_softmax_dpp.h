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

#include <fused_kernel/algorithms/attention/softmax.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <type_traits>
#include <vector>

using namespace fk;

namespace {

constexpr int BLOCK_SIZE = 128;
using Details = SoftmaxDPPDetails<float, BLOCK_SIZE>;
using MergeIOp = decltype(MergeSoftmaxState<>::build());

static_assert(SoftmaxDPP<ParArch::CPU, Details>::PAR_ARCH == ParArch::CPU);
#if defined(__NVCC__)
static_assert(SoftmaxDPP<ParArch::GPU_NVIDIA, Details>::PAR_ARCH ==
              ParArch::GPU_NVIDIA);
#endif
static_assert(std::is_trivially_copyable_v<Details>);
static_assert(opIs<UnaryType, MergeIOp>);

template <typename Transform>
std::vector<double> softmaxOracle(const std::vector<float>& input,
                                  const int rows, const int width,
                                  const Transform& transform,
                                  const double outputScale) {
    std::vector<double> expected(input.size());
    for (int row = 0; row < rows; ++row) {
        double maximum = -std::numeric_limits<double>::infinity();
        for (int x = 0; x < width; ++x) {
            maximum = std::max(maximum,
                transform(static_cast<double>(input[row * width + x])));
        }
        double denominator = 0.0;
        for (int x = 0; x < width; ++x) {
            denominator += std::exp(
                transform(static_cast<double>(input[row * width + x])) - maximum);
        }
        for (int x = 0; x < width; ++x) {
            expected[row * width + x] = outputScale * std::exp(
                transform(static_cast<double>(input[row * width + x])) - maximum) /
                denominator;
        }
    }
    return expected;
}

bool checkResult(const char* name, const std::vector<float>& got,
                 const std::vector<double>& expected,
                 const int rows, const int width,
                 const double expectedRowSum, const double tolerance) {
    double maxError = 0.0;
    for (int row = 0; row < rows; ++row) {
        double rowSum = 0.0;
        for (int x = 0; x < width; ++x) {
            const float value = got[row * width + x];
            if (!std::isfinite(value)) {
                std::printf("%s non-finite row=%d x=%d value=%f\n",
                            name, row, x, value);
                return false;
            }
            rowSum += value;
            maxError = std::max(maxError,
                std::fabs(static_cast<double>(value) - expected[row * width + x]));
        }
        if (std::fabs(rowSum - expectedRowSum) > tolerance * width) {
            std::printf("%s row=%d sum=%g expected=%g\n",
                        name, row, rowSum, expectedRowSum);
            return false;
        }
    }
    if (maxError > tolerance) {
        std::printf("%s maxError=%g tolerance=%g\n", name, maxError, tolerance);
        return false;
    }
    return true;
}

std::vector<float> makeInput(const int rows, const int width, const float scale) {
    std::vector<float> input(rows * width);
    for (int row = 0; row < rows; ++row) {
        for (int x = 0; x < width; ++x) {
            const int code = ((row + 3) * 29 + (x + 5) * 17) % 101;
            input[row * width + x] = scale * static_cast<float>(code - 50);
        }
    }
    return input;
}

bool testCpuBaseline(const int rows, const int width, const float inputScale) {
    const std::vector<float> input = makeInput(rows, width, inputScale);
    std::vector<float> output(input.size(), -1.f);
    const RawPtr<ND::_2D, float> inputPtr{
        const_cast<float*>(input.data()),
        PtrDims<ND::_2D>(width, rows, width * sizeof(float))};
    const RawPtr<ND::_2D, float> outputPtr{
        output.data(), PtrDims<ND::_2D>(width, rows, width * sizeof(float))};

    const auto read = PerThreadRead<ND::_2D, float>::build(inputPtr);
    const auto write = PerThreadWrite<ND::_2D, float>::build(outputPtr);
    const auto merge = MergeSoftmaxState<>::build();
    const Details details{rows, width};
    SoftmaxDPP<ParArch::CPU, Details>::exec(details, read, write, merge);

    const auto expected = softmaxOracle(
        input, rows, width, [](const double x) { return x; }, 1.0);
    return checkResult("CPU baseline", output, expected,
                       rows, width, 1.0, 2e-6);
}

bool testCpuFusedIOps() {
    constexpr int rows = 3;
    constexpr int width = 17;
    const std::vector<float> input = makeInput(rows, width, 0.125f);
    std::vector<float> output(input.size(), -1.f);
    const RawPtr<ND::_2D, float> inputPtr{
        const_cast<float*>(input.data()),
        PtrDims<ND::_2D>(width, rows, width * sizeof(float))};
    const RawPtr<ND::_2D, float> outputPtr{
        output.data(), PtrDims<ND::_2D>(width, rows, width * sizeof(float))};

    const auto read = PerThreadRead<ND::_2D, float>::build(inputPtr)
        .then(Mul<float>::build(1.5f))
        .then(Add<float>::build(-0.25f));
    const auto write = Mul<float>::build(3.f)
        .then(PerThreadWrite<ND::_2D, float>::build(outputPtr));
    const auto merge = MergeSoftmaxState<>::build();
    const Details details{rows, width};
    SoftmaxDPP<ParArch::CPU, Details>::exec(details, read, write, merge);

    const auto expected = softmaxOracle(
        input, rows, width, [](const double x) { return 1.5 * x - 0.25; }, 3.0);
    return checkResult("CPU fused IOps", output, expected,
                       rows, width, 3.0, 3e-6);
}

#if defined(__NVCC__)
template <bool FUSED>
bool testGpuCase(const char* name, const int rows, const int width,
                 const float inputScale) {
    const std::vector<float> hostInput = makeInput(rows, width, inputScale);
    Ptr2D<float> input(width, rows);
    Ptr2D<float> output(width, rows);
    for (int row = 0; row < rows; ++row) {
        for (int x = 0; x < width; ++x) {
            input.at(Point{x, row, 0}) = hostInput[row * width + x];
            output.at(Point{x, row, 0}) = -1.f;
        }
    }

    Stream stream;
    input.upload(stream);
    output.upload(stream);
    const auto merge = MergeSoftmaxState<>::build();
    const Details details{rows, width};
    if constexpr (FUSED) {
        const auto read = PerThreadRead<ND::_2D, float>::build(input)
            .then(Mul<float>::build(1.5f))
            .then(Add<float>::build(-0.25f));
        const auto write = Mul<float>::build(3.f)
            .then(PerThreadWrite<ND::_2D, float>::build(output));
        executeSoftmax(details, read, write, merge, stream);
    } else {
        const auto read = PerThreadRead<ND::_2D, float>::build(input);
        const auto write = PerThreadWrite<ND::_2D, float>::build(output);
        executeSoftmax(details, read, write, merge, stream);
    }
    output.download(stream);
    stream.sync();

    std::vector<float> got(hostInput.size());
    for (int row = 0; row < rows; ++row) {
        for (int x = 0; x < width; ++x) {
            got[row * width + x] = output.at(Point{x, row, 0});
        }
    }
    const auto expected = FUSED
        ? softmaxOracle(hostInput, rows, width,
              [](const double x) { return 1.5 * x - 0.25; }, 3.0)
        : softmaxOracle(hostInput, rows, width,
              [](const double x) { return x; }, 1.0);
    return checkResult(name, got, expected, rows, width,
                       FUSED ? 3.0 : 1.0, 8e-6);
}
#endif

} // namespace

int launch() {
    bool ok = testCpuBaseline(3, 7, 0.25f);
    ok = testCpuBaseline(2, BLOCK_SIZE, 0.125f) && ok;
    ok = testCpuBaseline(4, BLOCK_SIZE * 2 + 1, 20.f) && ok;
    ok = testCpuFusedIOps() && ok;
#if defined(__NVCC__)
    ok = testGpuCase<false>("GPU width<block", 3, 7, 0.25f) && ok;
    ok = testGpuCase<false>("GPU width==block", 2, BLOCK_SIZE, 0.125f) && ok;
    ok = testGpuCase<false>("GPU non-divisible extreme", 4,
                            BLOCK_SIZE * 2 + 1, 20.f) && ok;
    ok = testGpuCase<true>("GPU fused IOps", 3, 17, 0.125f) && ok;
#endif
    if (ok) std::printf("SoftmaxDPP CPU/GPU IOp contract: PASS\n");
    return ok ? 0 : -1;
}
