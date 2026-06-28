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

#include <fused_kernel/algorithms/algorithms.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/logical.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <limits>

using namespace fk;

namespace {

template <typename ComputeIOp>
bool runCpuCase(const ComputeIOp& compute,
                const std::array<float, 3>& identity,
                const std::array<float, 3>& expected) {
    constexpr int ROWS = 3;
    constexpr int WIDTH = 7;
    constexpr int BLOCK_SIZE = 64;

    std::array<float, ROWS * WIDTH> input{
         1.f, -2.f,  3.f,  4.f, -5.f,  6.f,  7.f,
        -9.f, -8.f, -7.f, -6.f, -5.f, -4.f, -3.f,
         0.5f, 1.5f, -2.f, 8.f,  3.f, -1.f, 2.f
    };
    std::array<float, ROWS> output{123.f, 123.f, 123.f};

    const RawPtr<ND::_2D, float> inputPtr{
        input.data(), PtrDims<ND::_2D>(WIDTH, ROWS, WIDTH * sizeof(float))};
    const RawPtr<ND::_1D, float> outputPtr{
        output.data(), PtrDims<ND::_1D>(ROWS, ROWS * sizeof(float))};
    const auto read = PerThreadRead<ND::_2D, float>::build(inputPtr);
    const auto write = PerThreadWrite<ND::_1D, float>::build(outputPtr);

    using Details = ReduceDPPDetails<float, BLOCK_SIZE>;
    for (int row = 0; row < ROWS; ++row) {
        const Details details{1, WIDTH, identity[row], row};
        ReduceDPP<ParArch::CPU, Details>::exec(details, read, compute, write);
    }

    for (int row = 0; row < ROWS; ++row) {
        if (std::fabs(output[row] - expected[row]) > 1e-6f) {
            std::printf("CPU reduction mismatch row=%d got=%f expected=%f\n",
                        row, output[row], expected[row]);
            return false;
        }
    }
    return true;
}

bool testCpuReductionIOps() {
    const auto sum = Add<float, float, float, UnaryType>::build();
    const auto min = Min<float, float, float, UnaryType>::build();
    const auto max = Max<float, float, float, UnaryType>::build();

    const bool sumOk = runCpuCase(sum, {0.f, 0.f, 0.f}, {14.f, -42.f, 12.f});
    const bool minOk = runCpuCase(min,
        {std::numeric_limits<float>::infinity(),
         std::numeric_limits<float>::infinity(),
         std::numeric_limits<float>::infinity()},
        {-5.f, -9.f, -2.f});
    const bool maxOk = runCpuCase(max,
        {-std::numeric_limits<float>::infinity(),
         -std::numeric_limits<float>::infinity(),
         -std::numeric_limits<float>::infinity()},
        {7.f, -3.f, 8.f});
    return sumOk && minOk && maxOk;
}

bool testCpuGridTierAndFusion() {
    std::array<float, 3> input{1.f, 2.f, 3.f};
    std::array<float, 1> output{0.f};
    const RawPtr<ND::_2D, float> inputPtr{
        input.data(), PtrDims<ND::_2D>(3, 1, 3 * sizeof(float))};
    const RawPtr<ND::_1D, float> outputPtr{
        output.data(), PtrDims<ND::_1D>(1, sizeof(float))};

    const auto read = PerThreadRead<ND::_2D, float>::build(inputPtr)
        .then(Add<float>::build(1.f));
    const auto sum = Add<float, float, float, UnaryType>::build();
    const auto write = Mul<float>::build(2.f)
        .then(PerThreadWrite<ND::_1D, float>::build(outputPtr));

    using Details = ReduceDPPDetails<float, 32>;
    const Details details{1, 3, 0.f, 0};
    ReduceGridDPP<ParArch::CPU, Details>::exec(
        details, read, sum, write);
    if (std::fabs(output[0] - 18.f) > 1e-6f) {
        std::printf("CPU fused grid reduction got=%f expected=18\n", output[0]);
        return false;
    }
    return true;
}

#if defined(__NVCC__)
template <int BLOCK_SIZE, typename ComputeIOp>
bool runGpuVariant(const char* name, const int width, const float identity,
                   const ComputeIOp& compute) {
    constexpr int ROWS = 3;
    const int allocationWidth = width > 0 ? width : 1;
    Ptr2D<float> input(allocationWidth, ROWS);
    Ptr1D<float> output(ROWS);
    std::array<float, ROWS> expected{};

    for (int row = 0; row < ROWS; ++row) {
        expected[row] = identity;
        for (int x = 0; x < allocationWidth; ++x) {
            const float value = static_cast<float>(((x * 5 + row * 13) % 37) - 18);
            input.at(Point{x, row, 0}) = value;
            if (x < width) expected[row] = make_tuple(expected[row], value) | compute;
        }
        output.at(Point{row, 0, 0}) = 999.f;
    }

    Stream stream;
    input.upload(stream);
    output.upload(stream);
    const auto read = PerThreadRead<ND::_2D, float>::build(input);
    const auto write = PerThreadWrite<ND::_1D, float>::build(output);
    const ReduceDPPDetails<float, BLOCK_SIZE> details{ROWS, width, identity, 0};
    executeReduce(details, read, compute, write, stream);
    output.download(stream);
    stream.sync();

    for (int row = 0; row < ROWS; ++row) {
        const float got = output.at(Point{row, 0, 0});
        if (std::fabs(got - expected[row]) > 1e-4f) {
            std::printf("GPU %s B=%d W=%d row=%d got=%f expected=%f\n",
                        name, BLOCK_SIZE, width, row, got, expected[row]);
            return false;
        }
    }
    return true;
}

bool testGpuVariants() {
    const auto sum = Add<float, float, float, UnaryType>::build();
    const auto min = Min<float, float, float, UnaryType>::build();
    const auto max = Max<float, float, float, UnaryType>::build();
    return runGpuVariant<32>("sum-empty", 0, 0.f, sum) &&
           runGpuVariant<32>("sum", 33, 0.f, sum) &&
           runGpuVariant<64>("max", 129,
               -std::numeric_limits<float>::infinity(), max) &&
           runGpuVariant<128>("sum", 257, 0.f, sum) &&
           runGpuVariant<256>("min", 257,
               std::numeric_limits<float>::infinity(), min);
}

bool testGpuRowReductionIOps() {
    constexpr int ROWS = 4;
    constexpr int WIDTH = 257;
    constexpr int BLOCK_SIZE = 128;

    Ptr2D<float> input(WIDTH, ROWS);
    Ptr1D<float> output(ROWS);
    std::array<float, ROWS> expected{};
    for (int row = 0; row < ROWS; ++row) {
        for (int x = 0; x < WIDTH; ++x) {
            const float value = static_cast<float>(((x * 7 + row * 11) % 29) - 14);
            input.at(Point{x, row, 0}) = value;
            expected[row] += value;
        }
        output.at(Point{row, 0, 0}) = 999.f;
    }

    Stream stream;
    input.upload(stream);
    output.upload(stream);

    const auto read = PerThreadRead<ND::_2D, float>::build(input);
    const auto sum = Add<float, float, float, UnaryType>::build();
    const auto write = PerThreadWrite<ND::_1D, float>::build(output);
    const ReduceDPPDetails<float, BLOCK_SIZE> details{ROWS, WIDTH, 0.f, 0};
    executeReduce(details, read, sum, write, stream);

    output.download(stream);
    stream.sync();
    for (int row = 0; row < ROWS; ++row) {
        const float got = output.at(Point{row, 0, 0});
        if (std::fabs(got - expected[row]) > 1e-4f) {
            std::printf("GPU reduction mismatch row=%d got=%f expected=%f\n",
                        row, got, expected[row]);
            return false;
        }
    }

    for (int row = 0; row < ROWS; ++row) {
        output.at(Point{row, 0, 0}) = 999.f;
    }
    output.upload(stream);
    const auto fusedRead = read.then(Add<float>::build(1.f));
    const auto fusedWrite = Mul<float>::build(2.f).then(write);
    executeReduce(details, fusedRead, sum, fusedWrite, stream);
    output.download(stream);
    stream.sync();
    for (int row = 0; row < ROWS; ++row) {
        const float fusedExpected = (expected[row] + WIDTH) * 2.f;
        const float got = output.at(Point{row, 0, 0});
        if (std::fabs(got - fusedExpected) > 1e-4f) {
            std::printf("GPU fused reduction mismatch row=%d got=%f expected=%f\n",
                        row, got, fusedExpected);
            return false;
        }
    }
    return true;
}
#endif

} // namespace

int launch() {
    bool ok = testCpuReductionIOps();
    ok = testCpuGridTierAndFusion() && ok;
#if defined(__NVCC__)
    ok = testGpuVariants() && ok;
    ok = testGpuRowReductionIOps() && ok;
#endif
    if (ok) std::printf("ReduceDPP CPU/GPU IOp contract: PASS\n");
    return ok ? 0 : -1;
}
