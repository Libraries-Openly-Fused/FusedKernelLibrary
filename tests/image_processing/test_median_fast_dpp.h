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

namespace fk {

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
    constexpr bool GPU = PA == ParArch::GPU_NVIDIA || PA == ParArch::GPU_AMD;
    const auto memoryType = GPU ? MemType::DeviceAndPinned : MemType::Host;
    Ptr2D<unsigned char> input(static_cast<uint>(width),
                               static_cast<uint>(height), 0, memoryType);
    Ptr2D<unsigned char> output(static_cast<uint>(width),
                                static_cast<uint>(height), 0, memoryType);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            input.at(Point{x, y, 0}) = sourceValue(x, y);
            output.at(Point{x, y, 0}) = 0xA5;
        }
    }

    Stream_<PA> stream;
#if defined(__NVCC__) || defined(__HIPCC__)
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
#if defined(__NVCC__) || defined(__HIPCC__)
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
#if defined(__NVCC__) || defined(__HIPCC__)
    const auto gpu = runCase<defaultParArch, EX, EY, KW, KH>(
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
#ifdef FK_CPU_MULTITHREAD_ENABLED
    const auto mt = runCase<ParArch::CPU_MULTITHREAD, EX, EY, KW, KH>(
        width, height, runtimeKW, runtimeKH, anchorX, anchorY);
    const bool mtOk = cpu.passed && mt.passed && mt.output == cpu.output;
    std::printf("MedianQuad %-16s %dx%d k%dx%d CPU/CPU_MULTITHREAD %s\n",
                name, width, height,
                KW > 0 ? KW : runtimeKW, KH > 0 ? KH : runtimeKH,
                mtOk ? "PASS" : "FAIL");
    ok = ok && mtOk;
#endif
    return ok;
}

#ifdef FK_CPU_MULTITHREAD_ENABLED
// Details the DPP does not accept (a 9x9 kernel has more than 64 elements)
// must leave the output untouched, both through the entry point and when the
// DPP is started with them, and the stream must go on with the next execution
bool verifyRejectedMultithread() {
    constexpr int width = 23;
    constexpr int height = 11;
    Ptr2D<unsigned char> input(width, height, 0, MemType::Host);
    Ptr2D<unsigned char> output(width, height, 0, MemType::Host);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            input.at(Point{x, y, 0}) = sourceValue(x, y);
            output.at(Point{x, y, 0}) = 0xA5;
        }
    }
    const auto read = PerThreadRead<ND::_2D, unsigned char>::build(input)
        .then(Add<unsigned char>::build(READ_BIAS));
    const auto compare = make_tuple(
        Min<unsigned char, unsigned char,
            unsigned char, UnaryType>::build(),
        Max<unsigned char, unsigned char,
            unsigned char, UnaryType>::build());
    const auto write = Add<unsigned char>::build(WRITE_BIAS)
        .then(PerThreadWrite<ND::_2D, unsigned char>::build(output));
    using DPP = MedianQuadDPP<ParArch::CPU_MULTITHREAD, unsigned char>;
    const MedianQuadDetails rejected{width, height, 9, 9, 4, 4};
    Stream_<ParArch::CPU_MULTITHREAD> stream;
    executeMedianQuad<DPP>(stream, rejected, read, compare, write);
    stream.enqueue([=](const cpu_multithread::Completion& done) {
        DPP::exec(done, rejected, read, compare, write);
    });
    stream.sync();
    bool untouched = !DPP::accepts(rejected);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            untouched = output.at(Point{x, y, 0}) == 0xA5 && untouched;
        }
    }

    const MedianQuadDetails accepted{width, height, 3, 3, 1, 1};
    stream.enqueue([=](const cpu_multithread::Completion& done) {
        DPP::exec(done, rejected, read, compare, write);
    });
    executeMedianQuad<DPP>(stream, accepted, read, compare, write);
    stream.sync();
    bool computed = true;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            computed = output.at(Point{x, y, 0}) == oracleAt(
                width, height, 3, 3, 1, 1, x, y) && computed;
        }
    }
    const bool ok = untouched && computed;
    std::printf("MedianQuad CPU_MULTITHREAD rejected details %s\n",
                ok ? "PASS" : "FAIL");
    return ok;
}
#endif
} // namespace

int launch_impl() {
    bool ok = true;
#ifdef FK_CPU_MULTITHREAD_ENABLED
    ok = verifyRejectedMultithread() && ok;
#endif
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

} // namespace fk

int launch() {
    return fk::launch_impl();
}
