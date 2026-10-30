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

// ParArch::CPU_MULTITHREAD runs the per row code of ParArch::CPU on a thread pool, so the same chain of operations has
// to give the same output on both. This test compares the bytes of the outputs of several chains, executed on a
// Stream_<ParArch::CPU> and on a Stream_<ParArch::CPU_MULTITHREAD>: integer arithmetic, float arithmetic, thread
// fusion with a width that the vectors do not divide, a batch of planes and an NV12 crop, resize and normalize chain.
// Only the bytes of the pixels are compared, not the padding at the end of the rows. A ParArch::CPU_MULTITHREAD only
// test, so it only has a _cpp target.
#define __ONLY_CPU__

#include <tests/main.h>

#include <fused_kernel/fused_kernel.h>

#ifdef FK_CPU_MULTITHREAD_ENABLED
#include <fused_kernel/core/execution_model/cpu_multithread.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/cast.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/color_conversion.h>
#include <fused_kernel/algorithms/image_processing/crop.h>
#include <fused_kernel/algorithms/image_processing/image.h>
#include <fused_kernel/algorithms/image_processing/resize.h>
#include <fused_kernel/algorithms/image_processing/saturate.h>

#include <array>
#include <cstring>
#include <iostream>
#include <random>

namespace fk {

using MTStream = Stream_<ParArch::CPU_MULTITHREAD>;

// Whether the pixels of a and b are the same bytes
template <typename T>
bool samePixels(const Ptr2D<T>& a, const Ptr2D<T>& b) {
    const RawPtr<ND::_2D, T> ra = a.ptr();
    const RawPtr<ND::_2D, T> rb = b.ptr();
    bool same = ra.dims.width == rb.dims.width && ra.dims.height == rb.dims.height;
    for (uint y = 0; same && y < ra.dims.height; ++y) {
        const uchar* const rowA = reinterpret_cast<const uchar*>(ra.data) + static_cast<size_t>(y) * ra.dims.pitch;
        const uchar* const rowB = reinterpret_cast<const uchar*>(rb.data) + static_cast<size_t>(y) * rb.dims.pitch;
        same = std::memcmp(rowA, rowB, ra.dims.width * sizeof(T)) == 0;
    }
    return same;
}

template <typename T>
bool sameTensor(const Tensor<T>& a, const Tensor<T>& b) {
    const RawPtr<ND::_3D, T> ra = a.ptr();
    const RawPtr<ND::_3D, T> rb = b.ptr();
    bool same = ra.dims.width == rb.dims.width && ra.dims.height == rb.dims.height && ra.dims.planes == rb.dims.planes;
    for (uint z = 0; same && z < ra.dims.planes; ++z) {
        for (uint y = 0; same && y < ra.dims.height; ++y) {
            const uchar* const rowA = reinterpret_cast<const uchar*>(ra.data) +
                                      static_cast<size_t>(z) * ra.dims.plane_pitch + static_cast<size_t>(y) * ra.dims.pitch;
            const uchar* const rowB = reinterpret_cast<const uchar*>(rb.data) +
                                      static_cast<size_t>(z) * rb.dims.plane_pitch + static_cast<size_t>(y) * rb.dims.pitch;
            same = std::memcmp(rowA, rowB, ra.dims.width * sizeof(T)) == 0;
        }
    }
    return same;
}

template <typename T>
Ptr2D<T> randomImage(const uint width, const uint height, const uint seed) {
    Ptr2D<T> image(width, height, 0, MemType::Host);
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> byte(0, 255);
    for (int y = 0; y < static_cast<int>(height); ++y) {
        for (int x = 0; x < static_cast<int>(width); ++x) {
            image.at(x, y) = make_set<T>(static_cast<VBase<T>>(byte(gen)));
            if constexpr (cn<T> > 1) {
                image.at(x, y).y = static_cast<VBase<T>>(byte(gen));
            }
        }
    }
    return image;
}

// Runs the same chain on both backends: chain(dpp, stream, output) enqueues it with the DPP type and the stream
template <typename Output, typename Chain>
bool bothGiveTheSameBytes(Output& single, Output& multi, const Chain& chain, bool (*same)(const Output&, const Output&)) {
    Stream_<ParArch::CPU> cpuStream;
    MTStream mtStream;
    chain(TransformDPP<ParArch::CPU>{}, cpuStream, single);
    chain(TransformDPP<ParArch::CPU_MULTITHREAD>{}, mtStream, multi);
    mtStream.sync();
    return same(single, multi);
}

// Integer arithmetic and saturation
bool testIntegerChain() {
    constexpr uint W = 997, H = 61;
    const Ptr2D<uchar3> input = randomImage<uchar3>(W, H, 1);
    Ptr2D<uchar3> single(W, H, 0, MemType::Host);
    Ptr2D<uchar3> multi(W, H, 0, MemType::Host);
    const auto chain = [&](auto dpp, auto& stream, Ptr2D<uchar3>& out) {
        executeOperations<decltype(dpp)>(input, out, stream, Cast<uchar3, int3>::build(),
                                         Mul<int3>::build(make_<int3>(3, -2, 5)), Add<int3>::build(make_<int3>(-100, 300, 7)),
                                         SaturateCast<int3, uchar3>::build());
    };
    return bothGiveTheSameBytes<Ptr2D<uchar3>>(single, multi, chain, samePixels<uchar3>);
}

// Float arithmetic: multiplications followed by additions and subtractions, which a compiler may contract into fused
// multiply-adds; both backends run the same code, so they contract the same way
bool testFloatChain() {
    constexpr uint W = 997, H = 61;
    const Ptr2D<uchar3> input = randomImage<uchar3>(W, H, 2);
    Ptr2D<float3> single(W, H, 0, MemType::Host);
    Ptr2D<float3> multi(W, H, 0, MemType::Host);
    const auto chain = [&](auto dpp, auto& stream, Ptr2D<float3>& out) {
        executeOperations<decltype(dpp)>(input, out, stream, Cast<uchar3, float3>::build(),
                                         Mul<float3>::build(make_<float3>(1.f / 255.f, 0.37f, 1.1f)),
                                         Sub<float3>::build(make_<float3>(0.485f, 0.456f, 0.406f)),
                                         Div<float3>::build(make_<float3>(0.229f, 0.224f, 0.225f)),
                                         Mul<float3>::build(make_<float3>(1.7f, 0.3f, 2.9f)),
                                         Add<float3>::build(make_<float3>(0.25f, -0.5f, 1.25f)));
    };
    return bothGiveTheSameBytes<Ptr2D<float3>>(single, multi, chain, samePixels<float3>);
}

// Thread fusion, with a width that the elements per thread do not divide, and with one that they divide
bool testThreadFusionChain() {
    bool correct = true;
    for (const uint W : {1024u, 997u}) {
        constexpr uint H = 61;
        const Ptr2D<uchar> input = randomImage<uchar>(W, H, 3);
        Ptr2D<float> single(W, H, 0, MemType::Host);
        Ptr2D<float> multi(W, H, 0, MemType::Host);
        Stream_<ParArch::CPU> cpuStream;
        MTStream mtStream;
        const auto ops = [&](auto dpp, auto& stream, Ptr2D<float>& out) {
            executeOperations<decltype(dpp)>(input, out, stream, Cast<uchar, float>::build(), Mul<float>::build(0.37f),
                                             Add<float>::build(-3.5f));
        };
        ops(TransformDPP<ParArch::CPU, TF::ENABLED>{}, cpuStream, single);
        ops(TransformDPP<ParArch::CPU_MULTITHREAD, TF::ENABLED>{}, mtStream, multi);
        mtStream.sync();
        correct &= samePixels(single, multi);
    }
    return correct;
}

// A batch of images into the planes of a Tensor
bool testBatchChain() {
    constexpr uint W = 211, H = 53;
    constexpr size_t BATCH = 5;
    std::array<Ptr2D<uchar3>, BATCH> inputs{randomImage<uchar3>(W, H, 4), randomImage<uchar3>(W, H, 5),
                                            randomImage<uchar3>(W, H, 6), randomImage<uchar3>(W, H, 7),
                                            randomImage<uchar3>(W, H, 8)};
    Tensor<float3> single(W, H, BATCH, 1, MemType::Host);
    Tensor<float3> multi(W, H, BATCH, 1, MemType::Host);
    Stream_<ParArch::CPU> cpuStream;
    MTStream mtStream;
    executeOperations<TransformDPP<ParArch::CPU>>(inputs, single, cpuStream, Cast<uchar3, float3>::build(),
                                                  Mul<float3>::build(make_<float3>(0.5f, 1.5f, 2.5f)));
    executeOperations<TransformDPP<ParArch::CPU_MULTITHREAD>>(inputs, multi, mtStream, Cast<uchar3, float3>::build(),
                                                              Mul<float3>::build(make_<float3>(0.5f, 1.5f, 2.5f)));
    mtStream.sync();
    return sameTensor(single, multi);
}

// A chain of the BVF paper: NV12 crops, resized, normalized
bool testNV12Chain() {
    constexpr int W = 640, H = 360;
    constexpr size_t N = 16;
    Image<PixelFormat::NV12> frame(W, H, MemType::Host);
    auto data = frame.getData();
    std::mt19937 gen(9);
    std::uniform_int_distribution<int> byte(0, 255);
    for (int y = 0; y < H * 3 / 2; ++y) {
        for (int x = 0; x < W; ++x) {
            data.at(x, y) = static_cast<uchar>(byte(gen));
        }
    }
    std::array<Rect, N> crops{};
    std::uniform_int_distribution<int> distX(0, W - 20), distY(0, H - 10);
    for (auto& crop : crops) {
        crop = Rect(static_cast<uint>(distX(gen)), static_cast<uint>(distY(gen)), 20, 10);
    }
    Tensor<float3> single(64, 32, N, 1, MemType::Host);
    Tensor<float3> multi(64, 32, N, 1, MemType::Host);
    const auto run = [&](auto dpp, auto& stream, Tensor<float3>& out) {
        executeOperations<decltype(dpp)>(stream, ReadYUV<PixelFormat::NV12>::build(frame),
            ConvertYUVToRGB<PixelFormat::NV12, ColorRange::Full, ColorPrimitives::bt2020, false, float3>::build(),
            Crop<>::build(crops), Resize<InterpolationType::INTER_LINEAR, AspectRatio::IGNORE_AR>::build(Size(64, 32)),
            Mul<float3>::build(float3{1.f / 255.f, 1.f / 255.f, 1.f / 255.f}),
            Sub<float3>::build(float3{0.485f, 0.456f, 0.406f}), Div<float3>::build(float3{0.229f, 0.224f, 0.225f}),
            TensorWrite<float3>::build(out.ptr()));
    };
    return bothGiveTheSameBytes<Tensor<float3>>(single, multi, run, sameTensor<float3>);
}

int launch_impl() {
    const bool integer = testIntegerChain();
    const bool floating = testFloatChain();
    const bool threadFusion = testThreadFusionChain();
    const bool batch = testBatchChain();
    const bool nv12 = testNV12Chain();
    std::cout << "CPU_MULTITHREAD output is the same bytes as CPU: integer chain " << integer << ", float chain "
              << floating << ", thread fusion chain " << threadFusion << ", batch chain " << batch
              << ", NV12 crop resize chain " << nv12 << std::endl;
    return integer && floating && threadFusion && batch && nv12 ? 0 : -1;
}

} // namespace fk

int launch() {
    return fk::launch_impl();
}

#else

#include <iostream>

int launch() {
    std::cout << "ENABLE_CPU_MULTITHREAD is OFF (thread-pool submodule not checked out): CPU_MULTITHREAD not tested"
              << std::endl;
    return 0;
}

#endif
