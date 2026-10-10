/* Copyright 2026 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

// ParArch::CPU_MULTITHREAD is a host backend, so this test only has a _cpp target
#define __ONLY_CPU__

#include <tests/main.h>

#include <fused_kernel/fused_kernel.h>

#if __has_include(<ThreadPool.h>)
#include <fused_kernel/core/execution_model/cpu_multithread.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/add_border.h>
#include <fused_kernel/algorithms/image_processing/color_conversion.h>
#include <fused_kernel/algorithms/image_processing/crop.h>
#include <fused_kernel/algorithms/image_processing/image.h>
#include <fused_kernel/algorithms/image_processing/resize.h>

#include <array>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

namespace fk {

using MTStream = Stream_<ParArch::CPU_MULTITHREAD>;
using MTDPP = TransformDPP<ParArch::CPU_MULTITHREAD>;
using CPUDPP = TransformDPP<ParArch::CPU>;

// The Figure 1 chain of the BVF paper (NV12 crops resized, padded and normalized) gives exactly the same result on
// the single and the multithreaded CPU backends, since both run the same per thread code
bool testSameResultAsCPU() {
    constexpr int W = 640, H = 360;
    constexpr size_t N = 16;
    Image<PixelFormat::NV12> frame(W, H, MemType::Host);
    auto data = frame.getData();
    std::mt19937 gen(1);
    std::uniform_int_distribution<int> byte(0, 255);
    for (int y = 0; y < H * 3 / 2; ++y) {
        for (int x = 0; x < W; ++x) {
            data.at(x, y) = static_cast<uchar>(byte(gen));
        }
    }
    std::array<Rect, N> crops{};
    std::uniform_int_distribution<int> distX(0, W - 20), distY(0, H - 10);
    for (auto& crop : crops) {
        const int x = distX(gen);
        const int y = distY(gen);
        crop = Rect(x, y, 20, 10);
    }
    Tensor<float3> single(64, 64, N, 1, MemType::Host);
    Tensor<float3> multi(64, 64, N, 1, MemType::Host);
    const auto run = [&](auto dpp, auto& stream, Tensor<float3>& out) {
        executeOperations<decltype(dpp)>(stream,
            ReadYUV<PixelFormat::NV12>::build(frame),
            ConvertYUVToRGB<ColorDepth::p8bit, ColorRange::Full, ColorPrimitives::bt2020>::build(),
            Crop<>::build(crops),
            Resize<InterpolationType::INTER_LINEAR, AspectRatio::IGNORE_AR>::build(Size(64, 32)),
            AddBorder<>::build(16, 16, 0, 0, float3{0.f, 0.f, 0.f}),
            Mul<float3>::build(float3{1.f / 255.f, 1.f / 255.f, 1.f / 255.f}),
            Sub<float3>::build(float3{0.485f, 0.456f, 0.406f}),
            Div<float3>::build(float3{0.229f, 0.224f, 0.225f}),
            TensorWrite<float3>::build(out.ptr()));
        stream.sync();
    };
    Stream_<ParArch::CPU> cpuStream;
    MTStream mtStream;
    run(CPUDPP{}, cpuStream, single);
    run(MTDPP{}, mtStream, multi);
    const size_t bytes = N * 64 * 64 * sizeof(float3);
    return std::memcmp(single.ptr().data, multi.ptr().data, bytes) == 0;
}

// Executions on one stream run in the order they were enqueued: a chain of dependent executions, enqueued without
// synchronizing, gives the expected result
bool testStreamOrder() {
    constexpr int W = 257, H = 129, STEPS = 200;
    Ptr2D<float> a(W, H, 0, MemType::Host);
    Ptr2D<float> b(W, H, 0, MemType::Host);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            a.at(x, y) = static_cast<float>(x + y);
        }
    }
    MTStream stream;
    for (int i = 0; i < STEPS; ++i) {
        // a -> b: +1, then b -> a: +1
        executeOperations<MTDPP>(stream, PerThreadRead<ND::_2D, float>::build(a.ptr()), Add<float>::build(1.f),
                                 PerThreadWrite<ND::_2D, float>::build(b.ptr()));
        executeOperations<MTDPP>(stream, PerThreadRead<ND::_2D, float>::build(b.ptr()), Add<float>::build(1.f),
                                 PerThreadWrite<ND::_2D, float>::build(a.ptr()));
    }
    stream.sync();
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            if (a.at(x, y) != static_cast<float>(x + y + 2 * STEPS)) {
                return false;
            }
        }
    }
    return true;
}

// Several streams in flight at the same time on the shared pool, a copied stream sharing its queue, and a stream
// destroyed with work still enqueued (its destructor waits for it)
bool testConcurrentStreams() {
    constexpr int W = 512, H = 256, STREAMS = 4, STEPS = 50;
    std::vector<Ptr2D<int>> data;
    for (int s = 0; s < STREAMS; ++s) {
        data.emplace_back(W, H, 0, MemType::Host);
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                data[s].at(x, y) = s;
            }
        }
    }
    {
        std::vector<MTStream> streams(STREAMS);
        for (int i = 0; i < STEPS; ++i) {
            for (int s = 0; s < STREAMS; ++s) {
                MTStream copy = streams[s];
                executeOperations<MTDPP>(copy, PerThreadRead<ND::_2D, int>::build(data[s].ptr()), Add<int>::build(1),
                                         PerThreadWrite<ND::_2D, int>::build(data[s].ptr()));
            }
        }
        // No sync: the streams are destroyed here and their destructors wait for the enqueued executions
    }
    for (int s = 0; s < STREAMS; ++s) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                if (data[s].at(x, y) != s + STEPS) {
                    return false;
                }
            }
        }
    }
    return true;
}

int launch_impl() {
    const bool same = testSameResultAsCPU();
    const bool order = testStreamOrder();
    const bool concurrent = testConcurrentStreams();
    std::cout << "CPU_MULTITHREAD (" << cpuThreadPool().size() << " threads): same result as CPU " << same
              << ", stream order " << order << ", concurrent streams " << concurrent << std::endl;
    return same && order && concurrent ? 0 : -1;
}

} // namespace fk

int launch() {
    return fk::launch_impl();
}

#else

#include <iostream>

int launch() {
    std::cout << "ThreadPool.h not available (ENABLE_CPU_MULTITHREAD is OFF): CPU_MULTITHREAD not tested" << std::endl;
    return 0;
}

#endif
