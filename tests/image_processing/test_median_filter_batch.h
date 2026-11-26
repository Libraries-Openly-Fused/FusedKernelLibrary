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

// MedianFilterDPP with batched reads and with vector types:
// - a BatchRead of crops of different sizes (two of them touch the frame's
//   edges), with unary operations fused after it, written by a TensorWrite
//   and by a BatchWrite, gives bit for bit the outputs of one single image
//   call per crop, for MedianWindowSelect and MinimumWindowSelect, and
//   writes nothing outside each image;
// - uchar3 and float3 images give, in every channel, the outputs of the
//   scalar filter of that channel (the selections work per channel);
// - a write whose planes differ from the read's images is rejected: the
//   executors throw std::invalid_argument and nothing is written.
// Every case runs on ParArch::CPU, on CPU_MULTITHREAD when it is enabled, and
// on the GPU of the build.

#include <tests/main.h>

#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/median_filter.h>
#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/execution_model/stream.h>

#include <algorithm>
#include <array>
#include <cmath>
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
    template <typename Details, typename Read, typename Write,
              typename Selection>
    bool median(const Details& details, const Read& read, const Write& write,
                const Selection& selection) {
        // true if the filter ran (or was enqueued or launched), false if the
        // executor rejected the arguments with std::invalid_argument
        try {
            executeOperations<MedianFilterDPP<PA, Details>>(
                stream, details, read, write, selection);
            return true;
        } catch (const std::invalid_argument&) {
            return false;
        }
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

float floatValue(const int x, const int y, const int salt) {
    return static_cast<float>((x * 11 + y * 7 + salt * 5) % 29 - 14) * 0.125f;
}
uchar ucharValue(const int x, const int y, const int salt) {
    return static_cast<uchar>((x * 31 + y * 17 + salt * 7 + (x ^ y)) % 251);
}

template <typename T>
struct Values;
template <>
struct Values<float> {
    static float at(const int x, const int y, const int c) {
        return floatValue(x, y, c);
    }
};
template <>
struct Values<uchar> {
    static uchar at(const int x, const int y, const int c) {
        return ucharValue(x, y, c);
    }
};
template <>
struct Values<uchar3> {
    static uchar3 at(const int x, const int y, const int) {
        return make_<uchar3>(ucharValue(x, y, 0), ucharValue(x, y, 1),
                             ucharValue(x, y, 2));
    }
};
template <>
struct Values<float3> {
    static float3 at(const int x, const int y, const int) {
        return make_<float3>(floatValue(x, y, 0), floatValue(x, y, 1),
                             floatValue(x, y, 2));
    }
};

template <typename V>
VBase<V> channelOf(const V& value, const int c) {
    if constexpr (cn<V> == 1) {
        (void)c;
        return value;
    } else {
        return c == 0 ? value.x : (c == 1 ? value.y : value.z);
    }
}

template <typename V>
bool sameBits(const V& a, const V& b) {
    return std::memcmp(&a, &b, sizeof(V)) == 0;
}

constexpr float SENTINEL = -999.f;
// A value no case produces: the uchar values are below 251
template <typename T>
T sentinel() {
    return make_set<T>(static_cast<VBase<T>>(std::is_same_v<VBase<T>, uchar>
                                                 ? 255.f : SENTINEL));
}

// The frame (channel salt of the vector values, for the scalar frames)
template <typename T, ParArch PA>
Ptr2D<T> makeFrame(const int salt) {
    Ptr2D<T> frame(FRAME_W, FRAME_H, 0, MEMORY<PA>);
    for (int y = 0; y < FRAME_H; ++y)
        for (int x = 0; x < FRAME_W; ++x)
            frame.at(Point{x, y, 0}) = Values<T>::at(x, y, salt);
    return frame;
}

template <typename T, ParArch PA>
Ptr2D<T> makeFilled(const int width, const int height, const T value) {
    Ptr2D<T> image(static_cast<uint>(width), static_cast<uint>(height), 0,
                   MEMORY<PA>);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            image.at(Point{x, y, 0}) = value;
    return image;
}

template <typename T, ParArch PA>
Tensor<T> makeTensor(const int planes, const T value) {
    Tensor<T> tensor(MAX_W, MAX_H, static_cast<uint>(planes), 1, MEMORY<PA>);
    for (int z = 0; z < planes; ++z)
        for (int y = 0; y < MAX_H; ++y)
            for (int x = 0; x < MAX_W; ++x)
                tensor.at(Point{x, y, z}) = value;
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

// The host reference of the 3x3 median (or minimum), with anchor (1, 1) and a
// replicate border, of crop r of the float frame after * 2 + 1, then * 0.5
bool nearReference(const Ptr2D<float>& output, const Crop& r,
                   const bool minimum, const char* label) {
    for (int oy = 0; oy < r.h; ++oy) {
        for (int ox = 0; ox < r.w; ++ox) {
            std::vector<float> values;
            for (int wy = 0; wy < 3; ++wy) {
                for (int wx = 0; wx < 3; ++wx) {
                    const int sx = std::max(0, std::min(r.w - 1, ox + wx - 1));
                    const int sy = std::max(0, std::min(r.h - 1, oy + wy - 1));
                    values.push_back(floatValue(r.x + sx, r.y + sy, 0) * 2.f +
                                     1.f);
                }
            }
            std::sort(values.begin(), values.end());
            const float expected =
                (minimum ? values.front() : values[values.size() / 2]) * 0.5f;
            const float got = output.at(Point{ox, oy, 0});
            if (std::fabs(got - expected) > 1e-6f) {
                std::printf("%s (%d, %d): got %g expected %g\n", label, ox, oy,
                            got, expected);
                return false;
            }
        }
    }
    return true;
}

// Image z of a batched result against the single image call of crop z, bit
// for bit, with the rest of the plane untouched
template <typename T, typename Plane, typename Single>
bool samePlane(const Plane& plane, const int z, const bool planeIsTensor,
               const Single& single, const int width, const int height,
               const char* label) {
    for (int y = 0; y < (planeIsTensor ? MAX_H : height); ++y) {
        for (int x = 0; x < (planeIsTensor ? MAX_W : width); ++x) {
            const T got = planeIsTensor ? plane.at(Point{x, y, z})
                                        : plane.at(Point{x, y, 0});
            const bool inside = x < width && y < height;
            const T expected = inside ? single.at(Point{x, y, 0}) : sentinel<T>();
            if (!sameBits(got, expected)) {
                std::printf("%s image %d (%d, %d): %s\n", label, z, x, y,
                            inside ? "differs from its single call"
                                   : "written outside its image");
                return false;
            }
        }
    }
    return true;
}

using Details = MedianFilterDPPDetails<float, 16, 8, 7, 7>;

// 1. A BatchRead of three crops (* 2 + 1 fused), a 3x3 selection and * 0.5,
//    written by a TensorWrite and by a BatchWrite, against single calls
template <ParArch PA, typename Selection>
bool batchOfCrops(const Selection& selection, const bool minimum,
                  const char* name) {
    const Details details{3, 3, 1, 1};
    Backend<PA> backend;
    auto frame = makeFrame<float, PA>(0);
    backend.upload(frame);
    const auto crops = cropsOf(frame);
    std::array<Ptr2D<float>, CROPS> batchOutputs;
    std::array<Ptr2D<float>, CROPS> singleOutputs;
    for (int i = 0; i < CROPS; ++i) {
        const Crop& r = CROP_RECTS[static_cast<size_t>(i)];
        batchOutputs[static_cast<size_t>(i)] =
            makeFilled<float, PA>(r.w, r.h, SENTINEL);
        singleOutputs[static_cast<size_t>(i)] =
            makeFilled<float, PA>(r.w, r.h, SENTINEL);
        backend.upload(batchOutputs[static_cast<size_t>(i)]);
        backend.upload(singleOutputs[static_cast<size_t>(i)]);
    }
    auto tensor = makeTensor<float, PA>(CROPS, SENTINEL);
    backend.upload(tensor);
    const auto read = PerThreadRead<ND::_2D, float>::build(crops)
                          .then(Mul<float>::build(2.f))
                          .then(Add<float>::build(1.f));
    bool ok = backend.median(details, read,
                             Mul<float>::build(0.5f).then(
                                 TensorWrite<float>::build(tensor.ptr())),
                             selection);
    ok = backend.median(details, read,
                        Mul<float>::build(0.5f).then(
                            PerThreadWrite<ND::_2D, float>::build(batchOutputs)),
                        selection) && ok;
    for (int i = 0; i < CROPS; ++i) {
        ok = backend.median(
            details,
            PerThreadRead<ND::_2D, float>::build(crops[static_cast<size_t>(i)])
                .then(Mul<float>::build(2.f))
                .then(Add<float>::build(1.f)),
            Mul<float>::build(0.5f).then(PerThreadWrite<ND::_2D, float>::build(
                singleOutputs[static_cast<size_t>(i)])),
            selection) && ok;
    }
    backend.download(tensor);
    for (auto& output : batchOutputs) backend.download(output);
    for (auto& output : singleOutputs) backend.download(output);
    backend.sync();
    for (int i = 0; i < CROPS && ok; ++i) {
        const Crop& r = CROP_RECTS[static_cast<size_t>(i)];
        const auto& single = singleOutputs[static_cast<size_t>(i)];
        ok = nearReference(single, r, minimum, name) &&
             samePlane<float>(tensor, i, true, single, r.w, r.h, name) &&
             samePlane<float>(batchOutputs[static_cast<size_t>(i)], i, false,
                              single, r.w, r.h, name);
    }
    std::printf("MedianFilter %s batch of 3 crops of different sizes %s %s\n",
                name, NAME<PA>, ok ? "PASS" : "FAIL");
    return ok;
}

// 2. A batch of crops of a vector frame against the scalar filter of each
//    channel (the crops of a frame of that channel's values)
template <ParArch PA, typename T>
bool perChannel(const int windowWidth, const int windowHeight,
                const char* name) {
    using S = VBase<T>;
    using VDetails = MedianFilterDPPDetails<T, 16, 8, 7, 7>;
    using SDetails = MedianFilterDPPDetails<S, 16, 8, 7, 7>;
    const VDetails vectorDetails{windowWidth, windowHeight, windowWidth / 2,
                                 windowHeight / 2};
    const SDetails scalarDetails{windowWidth, windowHeight, windowWidth / 2,
                                 windowHeight / 2};
    Backend<PA> backend;
    auto frame = makeFrame<T, PA>(0);
    auto vectorOutput = makeTensor<T, PA>(CROPS, sentinel<T>());
    backend.upload(frame);
    backend.upload(vectorOutput);
    bool ok = backend.median(
        vectorDetails, PerThreadRead<ND::_2D, T>::build(cropsOf(frame)),
        TensorWrite<T>::build(vectorOutput.ptr()),
        MedianWindowSelect<T, VDetails::WINDOW_CAPACITY>::build());
    std::array<Tensor<S>, cn<T>> scalarOutputs;
    std::array<Ptr2D<S>, cn<T>> scalarFrames;
    for (int c = 0; c < cn<T>; ++c) {
        scalarFrames[static_cast<size_t>(c)] = makeFrame<S, PA>(c);
        scalarOutputs[static_cast<size_t>(c)] = makeTensor<S, PA>(CROPS, sentinel<S>());
        backend.upload(scalarFrames[static_cast<size_t>(c)]);
        backend.upload(scalarOutputs[static_cast<size_t>(c)]);
        ok = backend.median(
            scalarDetails,
            PerThreadRead<ND::_2D, S>::build(
                cropsOf(scalarFrames[static_cast<size_t>(c)])),
            TensorWrite<S>::build(scalarOutputs[static_cast<size_t>(c)].ptr()),
            MedianWindowSelect<S, SDetails::WINDOW_CAPACITY>::build()) && ok;
    }
    backend.download(vectorOutput);
    for (auto& output : scalarOutputs) backend.download(output);
    backend.sync();
    for (int z = 0; z < CROPS && ok; ++z) {
        const Crop& r = CROP_RECTS[static_cast<size_t>(z)];
        for (int y = 0; y < r.h && ok; ++y) {
            for (int x = 0; x < r.w && ok; ++x) {
                const T value = vectorOutput.at(Point{x, y, z});
                for (int c = 0; c < cn<T> && ok; ++c) {
                    const S expected =
                        scalarOutputs[static_cast<size_t>(c)].at(Point{x, y, z});
                    ok = sameBits(channelOf(value, c), expected) &&
                         !sameBits(expected, sentinel<S>());
                }
            }
        }
    }
    std::printf("MedianFilter %s %dx%d per channel, batch of 3 crops %s %s\n",
                name, windowWidth, windowHeight, NAME<PA>, ok ? "PASS" : "FAIL");
    return ok;
}

// 3. A batch of 3 images with a write of 2 planes: rejected, nothing written
template <ParArch PA>
bool planeMismatch() {
    const Details details{3, 3, 1, 1};
    Backend<PA> backend;
    auto frame = makeFrame<float, PA>(0);
    auto twoPlanes = makeTensor<float, PA>(2, SENTINEL);
    backend.upload(frame);
    backend.upload(twoPlanes);
    const bool ran = backend.median(
        details, PerThreadRead<ND::_2D, float>::build(cropsOf(frame)),
        TensorWrite<float>::build(twoPlanes.ptr()),
        MedianWindowSelect<float, Details::WINDOW_CAPACITY>::build());
    backend.download(twoPlanes);
    backend.sync();
    // every back-end's executor must reject it (std::invalid_argument)
    bool ok = !ran;
    for (int z = 0; z < 2; ++z)
        for (int y = 0; y < MAX_H; ++y)
            for (int x = 0; x < MAX_W; ++x)
                ok = ok && twoPlanes.at(Point{x, y, z}) == SENTINEL;
    std::printf("MedianFilter rejects a write of other planes %s %s\n",
                NAME<PA>, ok ? "PASS" : "FAIL");
    return ok;
}

template <ParArch PA>
bool allCases() {
    bool ok = batchOfCrops<PA>(
        MedianWindowSelect<float, Details::WINDOW_CAPACITY>::build(), false,
        "median");
    ok = batchOfCrops<PA>(
        MinimumWindowSelect<float, Details::WINDOW_CAPACITY>::build(), true,
        "minimum") && ok;
    ok = perChannel<PA, uchar3>(3, 3, "uchar3") && ok;
    ok = perChannel<PA, uchar3>(5, 7, "uchar3") && ok;
    ok = perChannel<PA, float3>(3, 3, "float3") && ok;
    ok = perChannel<PA, float3>(7, 5, "float3") && ok;
    ok = planeMismatch<PA>() && ok;
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
    std::printf("MedianFilterDPP batches and vector types: %s\n",
                ok ? "PASS" : "FAIL");
    return ok ? 0 : -1;
}

} // namespace fk

int launch() {
    return fk::launch_impl();
}
