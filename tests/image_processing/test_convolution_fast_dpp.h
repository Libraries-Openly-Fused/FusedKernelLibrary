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
#include <fused_kernel/algorithms/image_processing/convolution_fast.h>
#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/execution_model/stream.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace fk;

namespace {
constexpr float READ_BIAS = 0.25f;
constexpr float WRITE_BIAS = -0.5f;

float sourceValue(const int x, const int y) {
    return static_cast<float>(((x * 19 + y * 23) % 97) - 48) / 17.f;
}

std::vector<float> makeCoefficients(const int width, const int height) {
    std::vector<float> values(static_cast<std::size_t>(width * height));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            values[static_cast<std::size_t>(y * width + x)] =
                static_cast<float>(((x + 1) * 7 + (y + 2) * 11) % 29 - 14) /
                static_cast<float>(width * height * 5);
        }
    }
    return values;
}

float oracleAt(const int width, const int height,
               const int kernelWidth, const int kernelHeight,
               const int anchorX, const int anchorY,
               const std::vector<float>& coefficients,
               const int x, const int y) {
    float result = 0.f;
    for (int ky = 0; ky < kernelHeight; ++ky) {
        for (int kx = 0; kx < kernelWidth; ++kx) {
            const int sx = std::max(0, std::min(
                width - 1, x + kx - anchorX));
            const int sy = std::max(0, std::min(
                height - 1, y + ky - anchorY));
            result += coefficients[static_cast<std::size_t>(
                          ky * kernelWidth + kx)] *
                      (sourceValue(sx, sy) + READ_BIAS);
        }
    }
    return result + WRITE_BIAS;
}

struct Result {
    bool passed;
    std::vector<float> output;
};

template <ParArch PA, int EX, int EY, int KW, int KH>
Result runCase(const int width, const int height,
               const int runtimeKW, const int runtimeKH,
               const int anchorX, const int anchorY) {
    constexpr bool GPU = PA == ParArch::GPU_NVIDIA;
    const auto memoryType = GPU ? MemType::DeviceAndPinned : MemType::Host;
    Ptr2D<float> input(width, height, 0, memoryType);
    Ptr2D<float> output(width, height, 0, memoryType);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            input.at(Point{x, y, 0}) = sourceValue(x, y);
            output.at(Point{x, y, 0}) = 1234.f;
        }
    }

    const int kernelWidth = KW > 0 ? KW : runtimeKW;
    const int kernelHeight = KH > 0 ? KH : runtimeKH;
    const auto coefficients = makeCoefficients(kernelWidth, kernelHeight);
    ConvQuadDetails details{
        width, height, runtimeKW, runtimeKH, anchorX, anchorY, {}};
    std::copy(coefficients.begin(), coefficients.end(), details.coefficients);

    Stream_<PA> stream;
#if defined(__NVCC__)
    if constexpr (GPU) {
        input.upload(stream);
        output.upload(stream);
    }
#endif
    const auto read = PerThreadRead<ND::_2D, float>::build(input)
        .then(Add<float>::build(READ_BIAS));
    const auto compute = make_tuple(
        Mul<float, float, float, UnaryType>::build(),
        Add<float, float, float, UnaryType>::build());
    const auto write = Add<float>::build(WRITE_BIAS)
        .then(PerThreadWrite<ND::_2D, float>::build(output));
    using DPP = ConvQuadDPP<PA, float, EX, EY, KW, KH>;
    executeConvQuad<DPP>(stream, details, read, compute, write);
#if defined(__NVCC__)
    if constexpr (GPU) output.download(stream);
#endif
    stream.sync();

    bool passed = true;
    std::vector<float> actual(static_cast<std::size_t>(width * height));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float value = output.at(Point{x, y, 0});
            actual[static_cast<std::size_t>(y * width + x)] = value;
            passed = std::fabs(value - oracleAt(
                width, height, kernelWidth, kernelHeight,
                anchorX, anchorY, coefficients, x, y)) <= 2e-5f && passed;
        }
    }
    return {passed, std::move(actual)};
}

template <int EX, int EY, int KW, int KH>
bool verifyCase(const int width, const int height,
                const int runtimeKW, const int runtimeKH,
                const int anchorX, const int anchorY,
                const char* name) {
    const auto cpu = runCase<ParArch::CPU, EX, EY, KW, KH>(
        width, height, runtimeKW, runtimeKH, anchorX, anchorY);
    bool ok = cpu.passed;
#if defined(__NVCC__)
    const auto gpu = runCase<ParArch::GPU_NVIDIA, EX, EY, KW, KH>(
        width, height, runtimeKW, runtimeKH, anchorX, anchorY);
    bool parity = gpu.output.size() == cpu.output.size();
    for (std::size_t i = 0; i < gpu.output.size() && parity; ++i) {
        parity = std::fabs(gpu.output[i] - cpu.output[i]) <= 2e-5f;
    }
    ok = ok && gpu.passed && parity;
    std::printf("ConvQuad %-18s %dx%d k%dx%d CPU/GPU %s\n",
                name, width, height,
                KW > 0 ? KW : runtimeKW, KH > 0 ? KH : runtimeKH,
                ok ? "PASS" : "FAIL");
#else
    std::printf("ConvQuad CPU %-14s %dx%d k%dx%d %s\n",
                name, width, height,
                KW > 0 ? KW : runtimeKW, KH > 0 ? KH : runtimeKH,
                ok ? "PASS" : "FAIL");
#endif
    return ok;
}
} // namespace

int launch() {
    bool ok = true;
    ok = verifyCase<4, 4, 3, 3>(
        257, 129, 3, 3, 1, 1, "3x3-odd") && ok;
    ok = verifyCase<4, 4, 5, 5>(
        192, 108, 5, 5, 2, 2, "5x5") && ok;
    ok = verifyCase<2, 3, 7, 7>(
        65, 37, 7, 7, 3, 3, "7x7-ragged") && ok;
    ok = verifyCase<4, 4, 7, 7>(
        39, 27, 1, 1, 3, 3, "static-details") && ok;
    ok = verifyCase<4, 4, 0, 0>(
        73, 41, 3, 5, 0, 3, "runtime-3x5") && ok;
    return ok ? 0 : -1;
}
