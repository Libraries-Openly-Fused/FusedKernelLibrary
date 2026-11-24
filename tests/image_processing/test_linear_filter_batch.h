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

// LinearFilterDPP with batched and 3D reads, and with vector types:
// - a BatchRead of crops of different sizes (two of them touch the frame's
//   edges), with unary operations fused after it, written by a TensorWrite
//   and by a BatchWrite, gives bit for bit the outputs of one single image
//   call per crop, and writes nothing outside each image;
// - a 3D read with a 3D write gives the outputs of one call per plane;
// - uchar and uchar3 images with float and float3 coefficients accumulate in
//   float and float3, against a host reference;
// - executeBoxFilter for float, float3, float4 and uchar3 gives the outputs of
//   executeLinearFilter with explicit 1 / 9 coefficients;
// - a write whose planes differ from the read's images is rejected: the
//   execute functions return false and nothing is written.
// Every case runs on ParArch::CPU, on CPU_MULTITHREAD when it is enabled, and
// on the GPU of the build.

#include <tests/main.h>

#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/linear_filter.h>
#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/execution_model/stream.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
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

// Runs the filter on one back-end: ParArch::CPU through the DPP's exec, the
// others through executeLinearFilter / executeBoxFilter
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
    template <typename Details, typename Reads, typename Multiply,
              typename Accumulate, typename Write>
    bool linear(const Details& details, const Reads& reads,
                const Multiply& multiply, const Accumulate& accumulate,
                const Write& write) {
        if constexpr (PA == ParArch::CPU) {
            LinearFilterDPP<ParArch::CPU, Details>::exec(
                details, reads, multiply, accumulate, write);
            return true;
        } else {
            return executeLinearFilter(details, reads, multiply, accumulate,
                                       write, stream);
        }
    }
};

// The frame and its crops: different sizes, the first at the top left
// corner, the second at the bottom right one
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

template <typename T>
struct Values;
template <>
struct Values<float> {
    static float at(const int x, const int y, const int salt) {
        return static_cast<float>((x * 5 + y * 7 + salt * 3) % 19 - 9) *
               0.125f;
    }
};
template <>
struct Values<uchar> {
    static uchar at(const int x, const int y, const int salt) {
        return static_cast<uchar>((x * 31 + y * 17 + salt * 7) % 251);
    }
};
template <>
struct Values<uchar3> {
    static uchar3 at(const int x, const int y, const int salt) {
        return make_<uchar3>(Values<uchar>::at(x, y, salt),
                             Values<uchar>::at(x + 3, y, salt + 1),
                             Values<uchar>::at(x, y + 5, salt + 2));
    }
};
template <>
struct Values<float3> {
    static float3 at(const int x, const int y, const int salt) {
        return make_<float3>(Values<float>::at(x, y, salt),
                             Values<float>::at(x + 3, y, salt + 1),
                             Values<float>::at(x, y + 5, salt + 2));
    }
};
template <>
struct Values<float4> {
    static float4 at(const int x, const int y, const int salt) {
        return make_<float4>(Values<float>::at(x, y, salt),
                             Values<float>::at(x + 3, y, salt + 1),
                             Values<float>::at(x, y + 5, salt + 2),
                             Values<float>::at(x + 1, y + 1, salt + 3));
    }
};

// Channel c of a scalar or a vector, as float
template <typename V>
float channel(const V& value, const int c) {
    if constexpr (cn<V> == 1) {
        (void)c;
        return static_cast<float>(value);
    } else if constexpr (cn<V> == 2) {
        return static_cast<float>(c == 0 ? value.x : value.y);
    } else if constexpr (cn<V> == 3) {
        return static_cast<float>(c == 0 ? value.x : (c == 1 ? value.y : value.z));
    } else {
        return static_cast<float>(c == 0 ? value.x : (c == 1 ? value.y
                                  : (c == 2 ? value.z : value.w)));
    }
}

template <typename V>
bool sameBits(const V& a, const V& b) {
    return std::memcmp(&a, &b, sizeof(V)) == 0;
}

template <typename T, ParArch PA>
Ptr2D<T> makeImage(const int width, const int height, const int salt) {
    Ptr2D<T> image(static_cast<uint>(width), static_cast<uint>(height), 0,
                   MEMORY<PA>);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            image.at(Point{x, y, 0}) = Values<T>::at(x, y, salt);
    return image;
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
Tensor<T> makeTensor(const int width, const int height, const int planes,
                     const T value) {
    Tensor<T> tensor(static_cast<uint>(width), static_cast<uint>(height),
                     static_cast<uint>(planes), 1, MEMORY<PA>);
    for (int z = 0; z < planes; ++z)
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                tensor.at(Point{x, y, z}) = value;
    return tensor;
}

// The 3x3 weights, as type C
template <typename C, ParArch PA>
Ptr2D<C> makeKernel() {
    Ptr2D<C> kernel(3, 3, 0, MEMORY<PA>);
    for (int y = 0; y < 3; ++y)
        for (int x = 0; x < 3; ++x)
            kernel.at(Point{x, y, 0}) = Values<C>::at(x, y, 11);
    return kernel;
}

constexpr float SENTINEL = -999.f;

template <typename A>
A sentinel() {
    return make_set<A>(SENTINEL);
}

// Compares image z of a batched result (planes or images of their own size)
// with the single image call of crop z, bit for bit, and requires the rest
// of the plane to be untouched
template <typename A, typename Plane, typename Single>
bool samePlane(const Plane& plane, const int z, const bool planeIsTensor,
               const Single& single, const int width, const int height,
               const char* label) {
    for (int y = 0; y < (planeIsTensor ? MAX_H : height); ++y) {
        for (int x = 0; x < (planeIsTensor ? MAX_W : width); ++x) {
            const A got = planeIsTensor ? plane.at(Point{x, y, z})
                                        : plane.at(Point{x, y, 0});
            const bool inside = x < width && y < height;
            const A expected = inside ? single.at(Point{x, y, 0})
                                      : sentinel<A>();
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

// The host reference of a 3x3 filter with anchor (1, 1) and a replicate
// border: sum of image * coefficient per channel, in float, after the read's
// prologue (value * readScale + readOffset) and before the write's epilogue
// (* writeScale)
template <typename T, typename C>
bool nearReference(const Ptr2D<VectorType_t<float, cn<T>>>& output,
                   const std::vector<T>& image, const int width,
                   const int height, const Ptr2D<C>& kernel,
                   const float readScale, const float readOffset,
                   const float writeScale, const char* label) {
    for (int oy = 0; oy < height; ++oy) {
        for (int ox = 0; ox < width; ++ox) {
            for (int c = 0; c < cn<T>; ++c) {
                float sum = 0.f;
                for (int ky = 0; ky < 3; ++ky) {
                    for (int kx = 0; kx < 3; ++kx) {
                        int sx = ox + kx - 1;
                        int sy = oy + ky - 1;
                        sx = sx < 0 ? 0 : (sx >= width ? width - 1 : sx);
                        sy = sy < 0 ? 0 : (sy >= height ? height - 1 : sy);
                        const float value = channel(
                            image[static_cast<size_t>(sy * width + sx)], c) *
                            readScale + readOffset;
                        sum += value * channel(kernel.at(Point{kx, ky, 0}),
                                               cn<C> == 1 ? 0 : c);
                    }
                }
                const float got = channel(output.at(Point{ox, oy, 0}), c);
                const float expected = sum * writeScale;
                if (std::fabs(got - expected) >
                    1e-4f * std::max(1.f, std::fabs(expected))) {
                    std::printf("%s (%d, %d) channel %d: got %g expected %g\n",
                                label, ox, oy, c, got, expected);
                    return false;
                }
            }
        }
    }
    return true;
}

template <typename T>
std::vector<T> cropValues(const Crop& crop, const int salt) {
    std::vector<T> values(static_cast<size_t>(crop.w * crop.h));
    for (int y = 0; y < crop.h; ++y)
        for (int x = 0; x < crop.w; ++x)
            values[static_cast<size_t>(y * crop.w + x)] =
                Values<T>::at(crop.x + x, crop.y + y, salt);
    return values;
}

// 1. A BatchRead of three crops, with a fused prologue (* 2 + 1), a user 3x3
//    filter and an epilogue (* 0.25), written by a TensorWrite and by a
//    BatchWrite, against three single image calls
template <ParArch PA>
bool batchOfCrops() {
    using Details = LinearFilterDPPDetails<float>;
    const Details details{3, 3, 1, 1};
    Backend<PA> backend;
    auto frame = makeImage<float, PA>(FRAME_W, FRAME_H, 0);
    auto kernel = makeKernel<float, PA>();
    backend.upload(frame);
    backend.upload(kernel);
    std::array<Ptr2D<float>, CROPS> crops;
    std::array<Ptr2D<float>, CROPS> batchOutputs;
    std::array<Ptr2D<float>, CROPS> singleOutputs;
    for (int i = 0; i < CROPS; ++i) {
        const Crop& r = CROP_RECTS[static_cast<size_t>(i)];
        crops[static_cast<size_t>(i)] = frame.crop2D(
            Point{r.x, r.y, 0}, PtrDims<ND::_2D>(static_cast<uint>(r.w),
                                                 static_cast<uint>(r.h)));
        batchOutputs[static_cast<size_t>(i)] =
            makeFilled<float, PA>(r.w, r.h, SENTINEL);
        singleOutputs[static_cast<size_t>(i)] =
            makeFilled<float, PA>(r.w, r.h, SENTINEL);
        backend.upload(batchOutputs[static_cast<size_t>(i)]);
        backend.upload(singleOutputs[static_cast<size_t>(i)]);
    }
    auto tensor = makeTensor<float, PA>(MAX_W, MAX_H, CROPS, SENTINEL);
    backend.upload(tensor);

    const auto multiply = Mul<float, float, float, UnaryType>::build();
    const auto accumulate = Add<float, float, float, UnaryType>::build();
    const auto coefficients = PerThreadRead<ND::_2D, float>::build(kernel);
    const auto batchRead = PerThreadRead<ND::_2D, float>::build(crops)
                               .then(Mul<float>::build(2.f))
                               .then(Add<float>::build(1.f));
    const auto tensorWrite = Mul<float>::build(0.25f)
        .then(TensorWrite<float>::build(tensor.ptr()));
    const auto batchWrite = Mul<float>::build(0.25f)
        .then(PerThreadWrite<ND::_2D, float>::build(batchOutputs));
    bool ok = backend.linear(details, make_tuple(batchRead, coefficients),
                             multiply, accumulate, tensorWrite);
    ok = backend.linear(details, make_tuple(batchRead, coefficients),
                        multiply, accumulate, batchWrite) && ok;
    for (int i = 0; i < CROPS; ++i) {
        const auto read =
            PerThreadRead<ND::_2D, float>::build(crops[static_cast<size_t>(i)])
                .then(Mul<float>::build(2.f))
                .then(Add<float>::build(1.f));
        const auto write = Mul<float>::build(0.25f).then(
            PerThreadWrite<ND::_2D, float>::build(
                singleOutputs[static_cast<size_t>(i)]));
        ok = backend.linear(details, make_tuple(read, coefficients),
                            multiply, accumulate, write) && ok;
    }
    backend.download(tensor);
    for (auto& output : batchOutputs) backend.download(output);
    for (auto& output : singleOutputs) backend.download(output);
    backend.sync();
    if (!ok) {
        std::printf("batch of crops %s: an execute function returned false\n",
                    NAME<PA>);
        return false;
    }
    for (int i = 0; i < CROPS && ok; ++i) {
        const Crop& r = CROP_RECTS[static_cast<size_t>(i)];
        const auto& single = singleOutputs[static_cast<size_t>(i)];
        ok = nearReference<float>(single, cropValues<float>(r, 0), r.w, r.h,
                                  kernel, 2.f, 1.f, 0.25f, "single crop") &&
             samePlane<float>(tensor, i, true, single, r.w, r.h,
                              "TensorWrite") &&
             samePlane<float>(batchOutputs[static_cast<size_t>(i)], i, false,
                              single, r.w, r.h, "BatchWrite");
    }
    std::printf("LinearFilter batch of 3 crops of different sizes %s %s\n",
                NAME<PA>, ok ? "PASS" : "FAIL");
    return ok;
}

// 2. A 3D read (a Tensor of 3 planes) with a 3D write, against one call per
//    plane
template <ParArch PA>
bool tensorPlanes() {
    using Details = LinearFilterDPPDetails<float>;
    const Details details{3, 3, 1, 1};
    constexpr int W = 19, H = 13, P = 3;
    Backend<PA> backend;
    auto input = makeTensor<float, PA>(W, H, P, 0.f);
    std::array<Ptr2D<float>, P> planes;
    std::array<Ptr2D<float>, P> singleOutputs;
    for (int z = 0; z < P; ++z) {
        planes[static_cast<size_t>(z)] = makeImage<float, PA>(W, H, z + 1);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                input.at(Point{x, y, z}) = Values<float>::at(x, y, z + 1);
        singleOutputs[static_cast<size_t>(z)] =
            makeFilled<float, PA>(W, H, SENTINEL);
        backend.upload(planes[static_cast<size_t>(z)]);
        backend.upload(singleOutputs[static_cast<size_t>(z)]);
    }
    auto output = makeTensor<float, PA>(W, H, P, SENTINEL);
    auto kernel = makeKernel<float, PA>();
    backend.upload(input);
    backend.upload(output);
    backend.upload(kernel);
    const auto multiply = Mul<float, float, float, UnaryType>::build();
    const auto accumulate = Add<float, float, float, UnaryType>::build();
    const auto coefficients = PerThreadRead<ND::_2D, float>::build(kernel);
    bool ok = backend.linear(
        details,
        make_tuple(TensorRead<float>::build(input.ptr()), coefficients),
        multiply, accumulate, TensorWrite<float>::build(output.ptr()));
    for (int z = 0; z < P; ++z) {
        ok = backend.linear(
            details,
            make_tuple(PerThreadRead<ND::_2D, float>::build(
                           planes[static_cast<size_t>(z)]),
                       coefficients),
            multiply, accumulate,
            PerThreadWrite<ND::_2D, float>::build(
                singleOutputs[static_cast<size_t>(z)])) && ok;
    }
    backend.download(output);
    for (auto& single : singleOutputs) backend.download(single);
    backend.sync();
    for (int z = 0; z < P && ok; ++z) {
        const auto& single = singleOutputs[static_cast<size_t>(z)];
        for (int y = 0; y < H && ok; ++y)
            for (int x = 0; x < W && ok; ++x)
                ok = sameBits(output.at(Point{x, y, z}),
                              single.at(Point{x, y, 0}));
    }
    std::printf("LinearFilter 3D read of 3 planes, 3D write %s %s\n",
                NAME<PA>, ok ? "PASS" : "FAIL");
    return ok;
}

// 3. uchar and uchar3 crops, with float and float3 coefficients: the
//    accumulator is float / float3, written by a TensorWrite and by single
//    image calls, against the host reference
template <ParArch PA, typename T>
bool vectorBatch(const char* name) {
    using Acc = VectorType_t<float, cn<T>>;
    using C = Acc;
    using Details = LinearFilterDPPDetails<T>;
    static_assert(std::is_same_v<typename Details::AccumulatorType, Acc>,
                  "The accumulator is float with the channels of T");
    const Details details{3, 3, 1, 1};
    Backend<PA> backend;
    auto frame = makeImage<T, PA>(FRAME_W, FRAME_H, 0);
    auto kernel = makeKernel<C, PA>();
    backend.upload(frame);
    backend.upload(kernel);
    std::array<Ptr2D<T>, CROPS> crops;
    std::array<Ptr2D<Acc>, CROPS> singleOutputs;
    for (int i = 0; i < CROPS; ++i) {
        const Crop& r = CROP_RECTS[static_cast<size_t>(i)];
        crops[static_cast<size_t>(i)] = frame.crop2D(
            Point{r.x, r.y, 0}, PtrDims<ND::_2D>(static_cast<uint>(r.w),
                                                 static_cast<uint>(r.h)));
        singleOutputs[static_cast<size_t>(i)] =
            makeFilled<Acc, PA>(r.w, r.h, sentinel<Acc>());
        backend.upload(singleOutputs[static_cast<size_t>(i)]);
    }
    auto tensor = makeTensor<Acc, PA>(MAX_W, MAX_H, CROPS, sentinel<Acc>());
    backend.upload(tensor);
    const auto multiply = Mul<T, C, Acc, UnaryType>::build();
    const auto accumulate = Add<Acc, Acc, Acc, UnaryType>::build();
    const auto coefficients = PerThreadRead<ND::_2D, C>::build(kernel);
    bool ok = backend.linear(
        details,
        make_tuple(PerThreadRead<ND::_2D, T>::build(crops), coefficients),
        multiply, accumulate, TensorWrite<Acc>::build(tensor.ptr()));
    for (int i = 0; i < CROPS; ++i) {
        ok = backend.linear(
            details,
            make_tuple(PerThreadRead<ND::_2D, T>::build(
                           crops[static_cast<size_t>(i)]),
                       coefficients),
            multiply, accumulate,
            PerThreadWrite<ND::_2D, Acc>::build(
                singleOutputs[static_cast<size_t>(i)])) && ok;
    }
    backend.download(tensor);
    for (auto& single : singleOutputs) backend.download(single);
    backend.sync();
    for (int i = 0; i < CROPS && ok; ++i) {
        const Crop& r = CROP_RECTS[static_cast<size_t>(i)];
        const auto& single = singleOutputs[static_cast<size_t>(i)];
        ok = nearReference<T>(single, cropValues<T>(r, 0), r.w, r.h, kernel,
                              1.f, 0.f, 1.f, name) &&
             samePlane<Acc>(tensor, i, true, single, r.w, r.h, name);
    }
    std::printf("LinearFilter %s batch, %s accumulator %s %s\n", name,
                cn<T> == 1 ? "float" : "floatN", NAME<PA>,
                ok ? "PASS" : "FAIL");
    return ok;
}

// 4. executeBoxFilter (one float scale for every channel) against
//    executeLinearFilter with an explicit kernel of scale / 9 in every channel
template <ParArch PA, typename T>
bool boxFilter(const char* name) {
    if constexpr (PA == ParArch::CPU) {
        return true; // there is no executeBoxFilter for ParArch::CPU
    } else {
        using Details = LinearFilterDPPDetails<T>;
        using Acc = typename Details::AccumulatorType;
        const Details details{3, 3, 1, 1};
        constexpr int W = 29, H = 17;
        Backend<PA> backend;
        auto input = makeImage<T, PA>(W, H, 4);
        auto boxOutput = makeFilled<Acc, PA>(W, H, sentinel<Acc>());
        auto linearOutput = makeFilled<Acc, PA>(W, H, sentinel<Acc>());
        auto kernel = makeFilled<Acc, PA>(
            3, 3, make_set<Acc>(1.f / static_cast<float>(3 * 3)));
        backend.upload(input);
        backend.upload(boxOutput);
        backend.upload(linearOutput);
        backend.upload(kernel);
        const auto read = PerThreadRead<ND::_2D, T>::build(input);
        bool ok = executeBoxFilter(details, read,
                                   PerThreadWrite<ND::_2D, Acc>::build(boxOutput),
                                   1.f, backend.stream);
        ok = backend.linear(details,
                            make_tuple(read, PerThreadRead<ND::_2D, Acc>::build(kernel)),
                            Mul<T, Acc, Acc, UnaryType>::build(),
                            Add<Acc, Acc, Acc, UnaryType>::build(),
                            PerThreadWrite<ND::_2D, Acc>::build(linearOutput)) && ok;
        backend.download(boxOutput);
        backend.download(linearOutput);
        backend.sync();
        for (int y = 0; y < H && ok; ++y)
            for (int x = 0; x < W && ok; ++x)
                ok = sameBits(boxOutput.at(Point{x, y, 0}),
                              linearOutput.at(Point{x, y, 0})) &&
                     !sameBits(boxOutput.at(Point{x, y, 0}), sentinel<Acc>());
        std::printf("executeBoxFilter %s %s %s\n", name, NAME<PA>,
                    ok ? "PASS" : "FAIL");
        return ok;
    }
}

// 5. A write whose planes differ from the read's images: nothing is written,
//    and the execute functions return false
template <ParArch PA>
bool planeMismatch() {
    using Details = LinearFilterDPPDetails<float>;
    const Details details{3, 3, 1, 1};
    constexpr int W = 11, H = 7;
    Backend<PA> backend;
    auto input = makeTensor<float, PA>(W, H, 2, 1.f);
    auto threePlanes = makeTensor<float, PA>(W, H, 3, SENTINEL);
    auto image = makeFilled<float, PA>(W, H, SENTINEL);
    std::array<Ptr2D<float>, 3> sources{makeFilled<float, PA>(W, H, 1.f),
                                        makeFilled<float, PA>(W, H, 1.f),
                                        makeFilled<float, PA>(W, H, 1.f)};
    std::array<Ptr2D<float>, 2> twoOutputs{
        makeFilled<float, PA>(W, H, SENTINEL),
        makeFilled<float, PA>(W, H, SENTINEL)};
    backend.upload(input);
    backend.upload(threePlanes);
    backend.upload(image);
    for (auto& source : sources) backend.upload(source);
    for (auto& output : twoOutputs) backend.upload(output);
    const auto coefficients = ConstantFilterRead<float>::build(1.f);
    const auto multiply = Mul<float, float, float, UnaryType>::build();
    const auto accumulate = Add<float, float, float, UnaryType>::build();
    const auto planes2 = make_tuple(TensorRead<float>::build(input.ptr()),
                                    coefficients);
    // a 3D read of 2 planes with a 3D write of 3, and with a 2D write
    bool ran = backend.linear(details, planes2, multiply, accumulate,
                              TensorWrite<float>::build(threePlanes.ptr()));
    ran = backend.linear(details, planes2, multiply, accumulate,
                         PerThreadWrite<ND::_2D, float>::build(image)) || ran;
    // a batch of 3 images with a BatchWrite of 2
    ran = backend.linear(details,
                         make_tuple(PerThreadRead<ND::_2D, float>::build(sources),
                                    coefficients),
                         multiply, accumulate,
                         PerThreadWrite<ND::_2D, float>::build(twoOutputs)) || ran;
    backend.download(threePlanes);
    backend.download(image);
    for (auto& output : twoOutputs) backend.download(output);
    backend.sync();
    // ParArch::CPU's exec returns nothing; the others must return false
    bool ok = PA == ParArch::CPU || !ran;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            ok = ok && image.at(Point{x, y, 0}) == SENTINEL &&
                 twoOutputs[0].at(Point{x, y, 0}) == SENTINEL &&
                 twoOutputs[1].at(Point{x, y, 0}) == SENTINEL;
            for (int z = 0; z < 3; ++z)
                ok = ok && threePlanes.at(Point{x, y, z}) == SENTINEL;
        }
    }
    std::printf("LinearFilter rejects a write of other planes %s %s\n",
                NAME<PA>, ok ? "PASS" : "FAIL");
    return ok;
}

template <ParArch PA>
bool allCases() {
    bool ok = batchOfCrops<PA>();
    ok = tensorPlanes<PA>() && ok;
    ok = vectorBatch<PA, uchar>("uchar") && ok;
    ok = vectorBatch<PA, uchar3>("uchar3") && ok;
    ok = boxFilter<PA, float>("float") && ok;
    ok = boxFilter<PA, float3>("float3") && ok;
    ok = boxFilter<PA, float4>("float4") && ok;
    ok = boxFilter<PA, uchar3>("uchar3") && ok;
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
    std::printf("LinearFilterDPP batches and vector types: %s\n",
                ok ? "PASS" : "FAIL");
    return ok ? 0 : -1;
}

} // namespace fk

int launch() {
    return fk::launch_impl();
}
