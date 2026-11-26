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

// The four Quad DPPs (MorphQuadDPP, MedianQuadDPP, BoxFilterQuadDPP,
// ConvQuadDPP) with batched reads and with vector types:
// - a BatchRead of crops of different sizes (two of them touch the frame's
//   edges), written by a TensorWrite and by a BatchWrite, gives bit for bit
//   the outputs of one single image call per crop, and writes nothing outside
//   each image;
// - vector images give, in every channel, the outputs of the scalar DPP of
//   that channel;
// - a write whose planes differ from the read's images is rejected: the
//   executors throw std::invalid_argument and nothing is written.
// Every case runs on ParArch::CPU, on CPU_MULTITHREAD when it is enabled, and
// on the GPU of the build.

#include <tests/main.h>

#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/logical.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/box_filter_fast.h>
#include <fused_kernel/algorithms/image_processing/convolution_fast.h>
#include <fused_kernel/algorithms/image_processing/median_fast.h>
#include <fused_kernel/algorithms/image_processing/morphology_fast.h>
#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/execution_model/stream.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace fk {

namespace {

template <ParArch PA>
constexpr bool IS_GPU = PA == ParArch::GPU_NVIDIA || PA == ParArch::GPU_AMD;

template <ParArch PA>
constexpr MemType MEMORY = IS_GPU<PA> ? MemType::DeviceAndPinned : MemType::Host;

template <ParArch PA>
constexpr const char* NAME = PA == ParArch::CPU ? "CPU"
    : (PA == ParArch::CPU_MULTITHREAD ? "CPU_MULTITHREAD" : "GPU");

template <ParArch PA>
struct Backend {
    Stream_<PA> stream;

    template <typename P>
    void upload(P& ptr) {
#if defined(__NVCC__) || defined(FK_HIP_BACKEND)
        if constexpr (IS_GPU<PA>) ptr.upload(stream);
#endif
        (void)ptr;
    }
    template <typename P>
    void download(P& ptr) {
#if defined(__NVCC__) || defined(FK_HIP_BACKEND)
        if constexpr (IS_GPU<PA>) ptr.download(stream);
#endif
        (void)ptr;
    }
    void sync() {
        if constexpr (PA != ParArch::CPU) stream.sync();
    }
};

enum class Quad { Morph, Median, Box, Conv };

constexpr const char* quadName(const Quad quad) {
    return quad == Quad::Morph ? "MorphQuad"
        : (quad == Quad::Median ? "MedianQuad"
        : (quad == Quad::Box ? "BoxFilterQuad" : "ConvQuad"));
}

// What each Quad DPP takes for pixels of type T, with a runtime kernel of
// KW x KH anchored at its centre: its compute IOps, its details and the type
// it writes (T for the selections, float with the channels of T for the sums)
template <Quad Q, ParArch PA, typename T>
struct QuadOf;

// Runs DPP through executeOperations: true if it ran (or was enqueued or
// launched), false if the executor rejected the arguments with
// std::invalid_argument
template <typename DPP, ParArch PA, typename Details, typename... IOps>
bool runQuad(Stream_<PA>& stream, const Details& details, const IOps&... iOps) {
    try {
        executeOperations<DPP>(stream, details, iOps...);
        return true;
    } catch (const std::invalid_argument&) {
        return false;
    }
}

template <ParArch PA, typename T>
struct QuadOf<Quad::Morph, PA, T> {
    using DPP = MorphQuadDPP<PA, T>;
    using Out = T;
    static auto compute(const int, const int) {
        return Min<T, T, T, UnaryType>::build();
    }
    template <typename... IOps>
    static bool execute(Stream_<PA>& stream, const int kw, const int kh,
                        const IOps&... iOps) {
        return runQuad<DPP>(stream, MorphQuadDetails{kw, kh, kw / 2, kh / 2},
                            iOps...);
    }
};

template <ParArch PA, typename T>
struct QuadOf<Quad::Median, PA, T> {
    using DPP = MedianQuadDPP<PA, T>;
    using Out = T;
    static auto compute(const int, const int) {
        return make_tuple(Min<T, T, T, UnaryType>::build(),
                          Max<T, T, T, UnaryType>::build());
    }
    template <typename... IOps>
    static bool execute(Stream_<PA>& stream, const int kw, const int kh,
                        const IOps&... iOps) {
        return runQuad<DPP>(stream, MedianQuadDetails{kw, kh, kw / 2, kh / 2},
                            iOps...);
    }
};

template <ParArch PA, typename T>
struct QuadOf<Quad::Box, PA, T> {
    using DPP = BoxFilterQuadDPP<PA, T>;
    using Out = VectorType_t<float, cn<T>>;
    static_assert(std::is_same_v<T, Out>,
                  "These cases read the floatN values the box filter sums");
    static auto compute(const int kw, const int kh) {
        return make_tuple(Add<Out, Out, Out, UnaryType>::build(),
                          Sub<Out, Out, Out, UnaryType>::build(),
                          Div<Out>::build(make_set<Out>(
                              static_cast<float>(kw * kh))));
    }
    template <typename... IOps>
    static bool execute(Stream_<PA>& stream, const int kw, const int kh,
                        const IOps&... iOps) {
        return runQuad<DPP>(
            stream, BoxFilterQuadDetails{kw, kh, kw / 2, kh / 2}, iOps...);
    }
};

template <ParArch PA, typename T>
struct QuadOf<Quad::Conv, PA, T> {
    using DPP = ConvQuadDPP<PA, T>;
    using Out = VectorType_t<float, cn<T>>;
    static auto compute(const int, const int) {
        return make_tuple(Mul<float, T, Out, UnaryType>::build(),
                          Add<Out, Out, Out, UnaryType>::build());
    }
    template <typename... IOps>
    static bool execute(Stream_<PA>& stream, const int kw, const int kh,
                        const IOps&... iOps) {
        ConvQuadDetails details{kw, kh, kw / 2, kh / 2, {}};
        for (int i = 0; i < kw * kh; ++i)
            details.coefficients[i] =
                static_cast<float>((i * 7) % 11 - 5) * 0.0625f;
        return runQuad<DPP>(stream, details, iOps...);
    }
};

constexpr int FRAME_W = 53;
constexpr int FRAME_H = 37;
constexpr int CROPS = 3;
struct Crop {
    int x, y, w, h;
};
constexpr std::array<Crop, CROPS> CROP_RECTS{{{0, 0, 21, 13},
                                              {30, 25, 23, 12},
                                              {10, 8, 17, 19}}};
constexpr int MAX_W = 23;
constexpr int MAX_H = 19;

template <typename S>
S scalarValue(const int x, const int y, const int c) {
    if constexpr (std::is_same_v<S, float>) {
        return static_cast<float>((x * 5 + y * 7 + c * 3 + (x ^ y)) % 23 - 11) *
               0.125f;
    } else {
        return static_cast<S>((x * 31 + y * 17 + c * 7 + (x ^ y)) % 251);
    }
}

// Channel c of a vector frame is the scalar frame of salt c
template <typename T>
T pixel(const int x, const int y) {
    using S = VBase<T>;
    if constexpr (cn<T> == 1) {
        return scalarValue<S>(x, y, 0);
    } else if constexpr (cn<T> == 3) {
        return make_<T>(scalarValue<S>(x, y, 0), scalarValue<S>(x, y, 1),
                        scalarValue<S>(x, y, 2));
    } else {
        return make_<T>(scalarValue<S>(x, y, 0), scalarValue<S>(x, y, 1),
                        scalarValue<S>(x, y, 2), scalarValue<S>(x, y, 3));
    }
}

template <typename V>
VBase<V> channelOf(const V& value, const int c) {
    if constexpr (cn<V> == 1) {
        (void)c;
        return value;
    } else if constexpr (cn<V> == 3) {
        return c == 0 ? value.x : (c == 1 ? value.y : value.z);
    } else {
        return c == 0 ? value.x : (c == 1 ? value.y : (c == 2 ? value.z : value.w));
    }
}

template <typename V>
bool sameBits(const V& a, const V& b) {
    return std::memcmp(&a, &b, sizeof(V)) == 0;
}

// A value no case produces: the uchar values are below 251
template <typename T>
T sentinel() {
    return make_set<T>(static_cast<VBase<T>>(
        std::is_floating_point_v<VBase<T>> ? -999.f : 255.f));
}

template <typename T, ParArch PA>
Ptr2D<T> makeFrame(const int salt) {
    Ptr2D<T> frame(FRAME_W, FRAME_H, 0, MEMORY<PA>);
    for (int y = 0; y < FRAME_H; ++y)
        for (int x = 0; x < FRAME_W; ++x)
            if constexpr (cn<T> == 1) {
                frame.at(Point{x, y, 0}) =
                    salt < 0 ? pixel<T>(x, y) : scalarValue<T>(x, y, salt);
            } else {
                frame.at(Point{x, y, 0}) = pixel<T>(x, y);
            }
    return frame;
}

template <typename T, ParArch PA>
Ptr2D<T> makeFilled(const int width, const int height) {
    Ptr2D<T> image(static_cast<uint>(width), static_cast<uint>(height), 0,
                   MEMORY<PA>);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            image.at(Point{x, y, 0}) = sentinel<T>();
    return image;
}

template <typename T, ParArch PA>
Tensor<T> makeTensor(const int planes) {
    Tensor<T> tensor(MAX_W, MAX_H, static_cast<uint>(planes), 1, MEMORY<PA>);
    for (int z = 0; z < planes; ++z)
        for (int y = 0; y < MAX_H; ++y)
            for (int x = 0; x < MAX_W; ++x)
                tensor.at(Point{x, y, z}) = sentinel<T>();
    return tensor;
}

template <typename T>
std::array<Ptr2D<T>, CROPS> cropsOf(const Ptr2D<T>& frame) {
    std::array<Ptr2D<T>, CROPS> crops;
    for (int i = 0; i < CROPS; ++i) {
        const Crop& r = CROP_RECTS[static_cast<size_t>(i)];
        crops[static_cast<size_t>(i)] = frame.crop2D(
            Point{r.x, r.y, 0}, PtrDims<ND::_2D>(static_cast<uint>(r.w),
                                                 static_cast<uint>(r.h)));
    }
    return crops;
}

// 1. A batch of crops, written by a TensorWrite and by a BatchWrite, against
//    one single image call per crop
template <Quad Q, ParArch PA, typename T>
bool batchOfCrops(const int kw, const int kh, const char* type) {
    using Of = QuadOf<Q, PA, T>;
    using Out = typename Of::Out;
    Backend<PA> backend;
    auto frame = makeFrame<T, PA>(-1);
    backend.upload(frame);
    const auto crops = cropsOf(frame);
    std::array<Ptr2D<Out>, CROPS> batchOutputs;
    std::array<Ptr2D<Out>, CROPS> singleOutputs;
    for (int i = 0; i < CROPS; ++i) {
        const Crop& r = CROP_RECTS[static_cast<size_t>(i)];
        batchOutputs[static_cast<size_t>(i)] = makeFilled<Out, PA>(r.w, r.h);
        singleOutputs[static_cast<size_t>(i)] = makeFilled<Out, PA>(r.w, r.h);
        backend.upload(batchOutputs[static_cast<size_t>(i)]);
        backend.upload(singleOutputs[static_cast<size_t>(i)]);
    }
    auto tensor = makeTensor<Out, PA>(CROPS);
    backend.upload(tensor);
    const auto compute = Of::compute(kw, kh);
    const auto batchRead = PerThreadRead<ND::_2D, T>::build(crops);
    bool ok = Of::execute(backend.stream, kw, kh, batchRead, compute,
                          TensorWrite<Out>::build(tensor.ptr()));
    ok = Of::execute(backend.stream, kw, kh, batchRead, compute,
                     PerThreadWrite<ND::_2D, Out>::build(batchOutputs)) && ok;
    for (int i = 0; i < CROPS; ++i) {
        ok = Of::execute(backend.stream, kw, kh,
                         PerThreadRead<ND::_2D, T>::build(
                             crops[static_cast<size_t>(i)]),
                         compute,
                         PerThreadWrite<ND::_2D, Out>::build(
                             singleOutputs[static_cast<size_t>(i)])) && ok;
    }
    backend.download(tensor);
    for (auto& output : batchOutputs) backend.download(output);
    for (auto& output : singleOutputs) backend.download(output);
    backend.sync();
    for (int z = 0; z < CROPS && ok; ++z) {
        const Crop& r = CROP_RECTS[static_cast<size_t>(z)];
        const auto& single = singleOutputs[static_cast<size_t>(z)];
        const auto& batchOutput = batchOutputs[static_cast<size_t>(z)];
        for (int y = 0; y < MAX_H && ok; ++y) {
            for (int x = 0; x < MAX_W && ok; ++x) {
                const bool inside = x < r.w && y < r.h;
                const Out expected = inside ? single.at(Point{x, y, 0})
                                            : sentinel<Out>();
                ok = sameBits(tensor.at(Point{x, y, z}), expected) &&
                     (!inside || (sameBits(batchOutput.at(Point{x, y, 0}),
                                           expected) &&
                                  !sameBits(expected, sentinel<Out>())));
                if (!ok)
                    std::printf("%s image %d (%d, %d) differs\n", quadName(Q),
                                z, x, y);
            }
        }
    }
    std::printf("%s %s %dx%d batch of 3 crops of different sizes %s %s\n",
                quadName(Q), type, kw, kh, NAME<PA>, ok ? "PASS" : "FAIL");
    return ok;
}

// 2. A vector frame against the scalar DPP of every channel, on batches of
//    the same crops
template <Quad Q, ParArch PA, typename T>
bool perChannel(const int kw, const int kh, const char* type) {
    using S = VBase<T>;
    using VOf = QuadOf<Q, PA, T>;
    using SOf = QuadOf<Q, PA, S>;
    using VOut = typename VOf::Out;
    using SOut = typename SOf::Out;
    Backend<PA> backend;
    auto frame = makeFrame<T, PA>(-1);
    auto vectorOutput = makeTensor<VOut, PA>(CROPS);
    backend.upload(frame);
    backend.upload(vectorOutput);
    bool ok = VOf::execute(backend.stream, kw, kh,
                           PerThreadRead<ND::_2D, T>::build(cropsOf(frame)),
                           VOf::compute(kw, kh),
                           TensorWrite<VOut>::build(vectorOutput.ptr()));
    std::array<Ptr2D<S>, cn<T>> scalarFrames;
    std::array<Tensor<SOut>, cn<T>> scalarOutputs;
    for (int c = 0; c < cn<T>; ++c) {
        scalarFrames[static_cast<size_t>(c)] = makeFrame<S, PA>(c);
        scalarOutputs[static_cast<size_t>(c)] = makeTensor<SOut, PA>(CROPS);
        backend.upload(scalarFrames[static_cast<size_t>(c)]);
        backend.upload(scalarOutputs[static_cast<size_t>(c)]);
        ok = SOf::execute(backend.stream, kw, kh,
                          PerThreadRead<ND::_2D, S>::build(
                              cropsOf(scalarFrames[static_cast<size_t>(c)])),
                          SOf::compute(kw, kh),
                          TensorWrite<SOut>::build(
                              scalarOutputs[static_cast<size_t>(c)].ptr())) && ok;
    }
    backend.download(vectorOutput);
    for (auto& output : scalarOutputs) backend.download(output);
    backend.sync();
    for (int z = 0; z < CROPS && ok; ++z) {
        const Crop& r = CROP_RECTS[static_cast<size_t>(z)];
        for (int y = 0; y < r.h && ok; ++y) {
            for (int x = 0; x < r.w && ok; ++x) {
                const VOut value = vectorOutput.at(Point{x, y, z});
                for (int c = 0; c < cn<T> && ok; ++c) {
                    const SOut expected =
                        scalarOutputs[static_cast<size_t>(c)].at(Point{x, y, z});
                    ok = sameBits(channelOf(value, c), expected) &&
                         !sameBits(expected, sentinel<SOut>());
                    if (!ok)
                        std::printf("%s image %d (%d, %d) channel %d differs\n",
                                    quadName(Q), z, x, y, c);
                }
            }
        }
    }
    std::printf("%s %s %dx%d per channel, batch of 3 crops %s %s\n",
                quadName(Q), type, kw, kh, NAME<PA>, ok ? "PASS" : "FAIL");
    return ok;
}

// 3. A batch of 3 crops with a write of 2 planes: rejected, nothing written
template <Quad Q, ParArch PA, typename T>
bool planeMismatch() {
    using Of = QuadOf<Q, PA, T>;
    using Out = typename Of::Out;
    Backend<PA> backend;
    auto frame = makeFrame<T, PA>(-1);
    auto twoPlanes = makeTensor<Out, PA>(2);
    backend.upload(frame);
    backend.upload(twoPlanes);
    const bool ran = Of::execute(backend.stream, 3, 3,
                                 PerThreadRead<ND::_2D, T>::build(cropsOf(frame)),
                                 Of::compute(3, 3),
                                 TensorWrite<Out>::build(twoPlanes.ptr()));
    backend.download(twoPlanes);
    backend.sync();
    bool ok = !ran;
    for (int z = 0; z < 2; ++z)
        for (int y = 0; y < MAX_H; ++y)
            for (int x = 0; x < MAX_W; ++x)
                ok = ok && sameBits(twoPlanes.at(Point{x, y, z}), sentinel<Out>());
    std::printf("%s rejects a write of other planes %s %s\n", quadName(Q),
                NAME<PA>, ok ? "PASS" : "FAIL");
    return ok;
}

template <ParArch PA>
bool allCases() {
    bool ok = batchOfCrops<Quad::Morph, PA, uchar>(3, 3, "uchar");
    ok = batchOfCrops<Quad::Median, PA, uchar>(3, 3, "uchar") && ok;
    ok = batchOfCrops<Quad::Median, PA, uchar>(5, 3, "uchar") && ok;
    ok = batchOfCrops<Quad::Box, PA, float>(5, 3, "float") && ok;
    ok = batchOfCrops<Quad::Conv, PA, float>(3, 5, "float") && ok;

    ok = perChannel<Quad::Morph, PA, uchar3>(3, 3, "uchar3") && ok;
    ok = perChannel<Quad::Morph, PA, float4>(5, 3, "float4") && ok;
    ok = perChannel<Quad::Median, PA, uchar3>(3, 3, "uchar3") && ok;
    ok = perChannel<Quad::Median, PA, float3>(5, 5, "float3") && ok;
    ok = perChannel<Quad::Median, PA, float4>(5, 3, "float4") && ok;
    ok = perChannel<Quad::Box, PA, float3>(3, 3, "float3") && ok;
    ok = perChannel<Quad::Box, PA, float4>(5, 3, "float4") && ok;
    ok = perChannel<Quad::Conv, PA, uchar3>(3, 3, "uchar3") && ok;
    ok = perChannel<Quad::Conv, PA, float4>(5, 3, "float4") && ok;

    ok = planeMismatch<Quad::Morph, PA, uchar>() && ok;
    ok = planeMismatch<Quad::Median, PA, uchar>() && ok;
    ok = planeMismatch<Quad::Box, PA, float>() && ok;
    ok = planeMismatch<Quad::Conv, PA, float>() && ok;
    return ok;
}

} // namespace

int launch_impl() {
    bool ok = allCases<ParArch::CPU>();
#ifdef FK_CPU_MULTITHREAD_ENABLED
    ok = allCases<ParArch::CPU_MULTITHREAD>() && ok;
#endif
#if defined(__NVCC__) || defined(FK_HIP_BACKEND)
    ok = allCases<defaultParArch>() && ok;
#endif
    std::printf("Quad DPPs batches and vector types: %s\n",
                ok ? "PASS" : "FAIL");
    return ok ? 0 : -1;
}

} // namespace fk

int launch() {
    return fk::launch_impl();
}
