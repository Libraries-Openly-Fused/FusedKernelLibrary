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
#include <fused_kernel/algorithms/basic_ops/cast.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/box_filter_fast.h>
#include <fused_kernel/core/data/ptr_nd.h>

#include <cstdio>
#include <vector>

using namespace fk;

namespace {

constexpr unsigned char READ_BIAS = 1;
constexpr unsigned char WRITE_BIAS = 2;

unsigned char sourceValue(const int x, const int y) {
    return static_cast<unsigned char>((x * 17 + y * 29 + 3) % 251);
}

unsigned char oracleAt(const int width, const int height,
                       const int kernelWidth, const int kernelHeight,
                       const int anchorX, const int anchorY,
                       const int x, const int y) {
    float sum = 0.f;
    for (int ky = 0; ky < kernelHeight; ++ky) {
        for (int kx = 0; kx < kernelWidth; ++kx) {
            int sx = x + kx - anchorX;
            int sy = y + ky - anchorY;
            sx = sx < 0 ? 0 : (sx >= width ? width - 1 : sx);
            sy = sy < 0 ? 0 : (sy >= height ? height - 1 : sy);
            sum += static_cast<float>(sourceValue(sx, sy) + READ_BIAS);
        }
    }
    const auto filtered = static_cast<unsigned char>(
        sum / static_cast<float>(kernelWidth * kernelHeight));
    return static_cast<unsigned char>(filtered + WRITE_BIAS);
}

struct CaseResult {
    bool passed;
    std::vector<unsigned char> output;
};

template <ParArch PA, int EX, int EY, int KW, int KH>
CaseResult runCase(const int width, const int height,
                   const int runtimeKW, const int runtimeKH,
                   const int anchorX, const int anchorY) {
    using DPP = BoxFilterQuadDPP<PA, unsigned char, EX, EY, KW, KH>;
    constexpr bool GPU = PA == ParArch::GPU_NVIDIA;
    const MemType memoryType = GPU ? MemType::DeviceAndPinned : MemType::Host;

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
    const int kernelWidth = KW > 0 ? KW : runtimeKW;
    const int kernelHeight = KH > 0 ? KH : runtimeKH;
    const auto read = PerThreadRead<ND::_2D, unsigned char>::build(input)
        .then(Add<unsigned char>::build(READ_BIAS))
        .then(Cast<unsigned char, float>::build());
    const auto compute = make_tuple(
        Add<float, float, float, UnaryType>::build(),
        Sub<float, float, float, UnaryType>::build(),
        Div<float>::build(
            static_cast<float>(kernelWidth * kernelHeight)));
    const auto write = Cast<float, unsigned char>::build()
        .then(Add<unsigned char>::build(WRITE_BIAS))
        .then(PerThreadWrite<ND::_2D, unsigned char>::build(output));
    const BoxFilterQuadDetails details{
        width, height, runtimeKW, runtimeKH, anchorX, anchorY};
    executeBoxFilterQuad<DPP>(stream, details, read, compute, write);
#if defined(__NVCC__)
    if constexpr (GPU) output.download(stream);
#endif
    stream.sync();

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
    ok = gpu.passed && gpu.output == cpu.output && ok;
    std::printf("BoxFilterQuad %-14s %dx%d k%dx%d CPU/GPU %s\n",
                name, width, height,
                KW > 0 ? KW : runtimeKW,
                KH > 0 ? KH : runtimeKH,
                ok ? "PASS" : "FAIL");
#else
    std::printf("BoxFilterQuad CPU %-10s %dx%d k%dx%d %s\n",
                name, width, height,
                KW > 0 ? KW : runtimeKW,
                KH > 0 ? KH : runtimeKH,
                ok ? "PASS" : "FAIL");
#endif
    return ok;
}

} // namespace

int launch() {
    bool ok = true;
    ok = verifyCase<4, 4, 3, 3>(257, 129, 3, 3, 1, 1, "3x3-odd") && ok;
    ok = verifyCase<4, 4, 5, 5>(192, 108, 5, 5, 2, 2, "5x5") && ok;
    ok = verifyCase<2, 3, 7, 7>(65, 37, 7, 7, 3, 3, "7x7-ragged") && ok;
    ok = verifyCase<4, 4, 7, 7>(39, 27, 1, 1, 3, 3, "static-details") && ok;
    ok = verifyCase<4, 4, 9, 9>(120, 80, 9, 9, 4, 4, "9x9") && ok;
    ok = verifyCase<4, 4, 0, 0>(73, 41, 11, 5, 2, 3, "runtime-11x5") && ok;
    return ok ? 0 : -1;
}
