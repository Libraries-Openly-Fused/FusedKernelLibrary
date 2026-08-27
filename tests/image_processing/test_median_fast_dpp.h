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
#include <fused_kernel/algorithms/image_processing/median_fast.h>
#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/execution_model/stream.h>

#include <algorithm>
#include <cstdio>
#include <vector>

using namespace fk;

namespace {
constexpr unsigned char READ_BIAS = 2;
constexpr unsigned char WRITE_BIAS = 1;

unsigned char sourceValue(const int x, const int y) {
    return static_cast<unsigned char>((x * 37 + y * 13 + (x ^ (y * 3))) % 241);
}

unsigned char oracleAt(const int width, const int height,
                       const int kernelWidth, const int kernelHeight,
                       const int anchorX, const int anchorY,
                       const int x, const int y) {
    std::vector<unsigned char> values;
    values.reserve(static_cast<std::size_t>(kernelWidth * kernelHeight));
    for (int ky = 0; ky < kernelHeight; ++ky) {
        for (int kx = 0; kx < kernelWidth; ++kx) {
            const int sx = std::max(0, std::min(
                width - 1, x + kx - anchorX));
            const int sy = std::max(0, std::min(
                height - 1, y + ky - anchorY));
            values.push_back(static_cast<unsigned char>(
                sourceValue(sx, sy) + READ_BIAS));
        }
    }
    std::sort(values.begin(), values.end());
    return static_cast<unsigned char>(
        values[values.size() / 2] + WRITE_BIAS);
}

struct Result {
    bool passed;
    std::vector<unsigned char> output;
};

template <ParArch PA, int EX, int EY, int KW, int KH>
Result runCase(const int width, const int height,
               const int runtimeKW, const int runtimeKH,
               const int anchorX, const int anchorY) {
    constexpr bool GPU = PA == ParArch::GPU_NVIDIA;
    const auto memoryType = GPU ? MemType::DeviceAndPinned : MemType::Host;
    Ptr2D<unsigned char> input(width, height, 0, memoryType);
    Ptr2D<unsigned char> output(width, height, 0, memoryType);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            input.at(Point{x, y, 0}) = sourceValue(x, y);
            output.at(Point{x, y, 0}) = 0xA5;
        }
    }

    Stream_<PA> stream;
#if defined(__NVCC__)
    if constexpr (GPU) {
        input.upload(stream);
        output.upload(stream);
    }
#endif
    const auto read = PerThreadRead<ND::_2D, unsigned char>::build(input)
        .then(Add<unsigned char>::build(READ_BIAS));
    const auto compare = make_tuple(
        Min<unsigned char, unsigned char,
            unsigned char, UnaryType>::build(),
        Max<unsigned char, unsigned char,
            unsigned char, UnaryType>::build());
    const auto write = Add<unsigned char>::build(WRITE_BIAS)
        .then(PerThreadWrite<ND::_2D, unsigned char>::build(output));
    using DPP = MedianQuadDPP<PA, unsigned char, EX, EY, KW, KH>;
    const MedianQuadDetails details{
        width, height, runtimeKW, runtimeKH, anchorX, anchorY};
    executeMedianQuad<DPP>(stream, details, read, compare, write);
#if defined(__NVCC__)
    if constexpr (GPU) output.download(stream);
#endif
    stream.sync();

    const int kernelWidth = KW > 0 ? KW : runtimeKW;
    const int kernelHeight = KH > 0 ? KH : runtimeKH;
    bool passed = true;
    std::vector<unsigned char> actual(
        static_cast<std::size_t>(width * height));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const auto value = output.at(Point{x, y, 0});
            actual[static_cast<std::size_t>(y * width + x)] = value;
            passed = value == oracleAt(
                width, height, kernelWidth, kernelHeight,
                anchorX, anchorY, x, y) && passed;
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
    ok = ok && gpu.passed && gpu.output == cpu.output;
    std::printf("MedianQuad %-16s %dx%d k%dx%d CPU/GPU %s\n",
                name, width, height,
                KW > 0 ? KW : runtimeKW, KH > 0 ? KH : runtimeKH,
                ok ? "PASS" : "FAIL");
#else
    std::printf("MedianQuad CPU %-12s %dx%d k%dx%d %s\n",
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
