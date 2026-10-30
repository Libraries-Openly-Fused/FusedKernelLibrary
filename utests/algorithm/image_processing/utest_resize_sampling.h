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

// Fixes the two sampling conventions of Resize<INTER_LINEAR> (ResizeSampling):
//   ASYMMETRIC (the default)  the destination pixel x reads the source position x * (src / dst), as cv::cuda::resize does
//   HALF_PIXEL                the centre of the destination pixel, x + 0.5, reads (x + 0.5) * (src / dst) - 0.5, as
//                             cv::resize and NPP do
// with the first pixel replicated before the start and the last one after the end. The conventions only show on images
// that are not constant, so the input is a gradient plus seeded noise, resized by factors that are not 1 and are different
// in x and y. The output of every back-end (this test is built for CPU, CUDA and HIP) is compared with the same scalar
// double precision reference of each convention, with a tolerance of 1e-5 of the value range, so the back-ends also
// agree with each other. The default convention must be the one Resize had before ResizeSampling existed.
#include <tests/main.h>

#include <fused_kernel/algorithms/algorithms.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/core/execution_model/executors.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>

namespace {

template <typename T>
float getComponent(const T& value, const int c) {
    if constexpr (fk::cn<T> == 1) {
        return static_cast<float>(value);
    } else {
        return c == 0 ? value.x : (c == 1 ? value.y : value.z);
    }
}

template <typename T>
void setComponent(T& value, const int c, const float v) {
    if constexpr (fk::cn<T> == 1) {
        value = v;
    } else if (c == 0) {
        value.x = v;
    } else if (c == 1) {
        value.y = v;
    } else {
        value.z = v;
    }
}

// Scalar bilinear resize in double precision with the border replicated
std::vector<double> referenceResize(const std::vector<double>& src, const int srcW, const int srcH, const int cn,
                                    const int dstW, const int dstH, const fk::ResizeSampling sampling) {
    const bool halfPixel = sampling == fk::ResizeSampling::HALF_PIXEL;
    std::vector<double> dst(static_cast<size_t>(dstW) * dstH * cn);
    const double scaleX = static_cast<double>(srcW) / dstW, scaleY = static_cast<double>(srcH) / dstH;
    for (int y = 0; y < dstH; ++y) {
        const double fy = halfPixel ? std::max((y + 0.5) * scaleY - 0.5, 0.0) : y * scaleY;
        int y1 = static_cast<int>(std::floor(fy));
        double ay = fy - y1;
        if (y1 >= srcH - 1) {
            y1 = srcH - 1;
            ay = 0.0;
        }
        const int y2 = std::min(y1 + 1, srcH - 1);
        for (int x = 0; x < dstW; ++x) {
            const double fx = halfPixel ? std::max((x + 0.5) * scaleX - 0.5, 0.0) : x * scaleX;
            int x1 = static_cast<int>(std::floor(fx));
            double ax = fx - x1;
            if (x1 >= srcW - 1) {
                x1 = srcW - 1;
                ax = 0.0;
            }
            const int x2 = std::min(x1 + 1, srcW - 1);
            for (int c = 0; c < cn; ++c) {
                const auto at = [&](const int px, const int py) { return src[(static_cast<size_t>(py) * srcW + px) * cn + c]; };
                dst[(static_cast<size_t>(y) * dstW + x) * cn + c] =
                    (at(x1, y1) * (1.0 - ax) + at(x2, y1) * ax) * (1.0 - ay) + (at(x1, y2) * (1.0 - ax) + at(x2, y2) * ax) * ay;
            }
        }
    }
    return dst;
}

template <typename T, fk::ResizeSampling SAMPLING>
bool testResizeSampling(const int srcW, const int srcH, const int dstW, const int dstH, const unsigned seed) {
    constexpr int CN = fk::cn<T>;
    fk::Stream stream;
    fk::Ptr2D<T> input(srcW, srcH);
    fk::Ptr2D<T> output(dstW, dstH);

    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> noise(0.f, 40.f);
    std::vector<double> reference(static_cast<size_t>(srcW) * srcH * CN);
    double lowest = 1e30, highest = -1e30;
    for (int y = 0; y < srcH; ++y) {
        for (int x = 0; x < srcW; ++x) {
            T pixel{};
            for (int c = 0; c < CN; ++c) {
                const float v = 0.31f * static_cast<float>(x) + 0.17f * static_cast<float>(y) + 20.f * static_cast<float>(c) + noise(gen);
                setComponent(pixel, c, v);
                reference[(static_cast<size_t>(y) * srcW + x) * CN + c] = static_cast<double>(v);
                lowest = std::min(lowest, static_cast<double>(v));
                highest = std::max(highest, static_cast<double>(v));
            }
            input.at(x, y) = pixel;
        }
    }
    input.upload(stream);

    fk::Executor<fk::TransformDPP<>>::executeOperations(
        stream, fk::PerThreadRead<fk::ND::_2D, T>::build(input),
        fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR, NullType, SAMPLING>::build(
            fk::Size(dstW, dstH)),
        fk::PerThreadWrite<fk::ND::_2D, T>::build(output));
    output.download(stream);
    stream.sync();

    const std::vector<double> expected = referenceResize(reference, srcW, srcH, CN, dstW, dstH, SAMPLING);
    const double tolerance = 1e-5 * (highest - lowest);
    double worst = 0.0;
    for (int y = 0; y < dstH; ++y) {
        for (int x = 0; x < dstW; ++x) {
            for (int c = 0; c < CN; ++c) {
                const double diff = std::abs(static_cast<double>(getComponent(output.at(x, y), c)) -
                                             expected[(static_cast<size_t>(y) * dstW + x) * CN + c]);
                worst = std::max(worst, diff);
            }
        }
    }
    // The other convention must give a different image, or this test would not tell them apart
    const fk::ResizeSampling other = SAMPLING == fk::ResizeSampling::HALF_PIXEL ? fk::ResizeSampling::ASYMMETRIC : fk::ResizeSampling::HALF_PIXEL;
    const std::vector<double> otherExpected = referenceResize(reference, srcW, srcH, CN, dstW, dstH, other);
    double apart = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
        apart = std::max(apart, std::abs(expected[i] - otherExpected[i]));
    }
    const bool distinguishes = (srcW == dstW && srcH == dstH) || apart > 100.0 * tolerance;
    const bool correct = worst <= tolerance && distinguishes;
    if (!distinguishes) {
        std::cout << "The two conventions differ by only " << apart << " on " << srcW << "x" << srcH << " -> " << dstW << "x" << dstH << std::endl;
    }
    if (worst > tolerance) {
        std::cout << "Resize<INTER_LINEAR> " << srcW << "x" << srcH << " -> " << dstW << "x" << dstH << " with " << CN
                  << " channels: max abs diff " << worst << " > " << tolerance << " (1e-5 of the range)" << std::endl;
    }
    return correct;
}

template <typename T, fk::ResizeSampling SAMPLING>
bool testAllSizes() {
    bool correct = true;
    correct &= testResizeSampling<T, SAMPLING>(97, 61, 64, 128, 1);    // down in x, up in y
    correct &= testResizeSampling<T, SAMPLING>(640, 480, 100, 75, 2);  // down in both
    correct &= testResizeSampling<T, SAMPLING>(40, 30, 160, 90, 3);    // up in both
    correct &= testResizeSampling<T, SAMPLING>(64, 64, 64, 64, 4);     // same size: a copy
    return correct;
}

// The default is the convention of before ResizeSampling existed, and ResizeHalfPixel selects the other one
using DefaultResize = fk::Resize<fk::InterpolationType::INTER_LINEAR>;
static_assert(std::is_same_v<DefaultResize, fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR,
                                                       NullType, fk::ResizeSampling::ASYMMETRIC>>,
              "The default sampling of Resize must be ASYMMETRIC");
static_assert(std::is_same_v<fk::ResizeHalfPixel<>, fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR,
                                                               NullType, fk::ResizeSampling::HALF_PIXEL>>,
              "ResizeHalfPixel must select HALF_PIXEL");

} // namespace

int launch() {
    bool correct = true;
    correct &= testAllSizes<float, fk::ResizeSampling::ASYMMETRIC>();
    correct &= testAllSizes<fk::float3, fk::ResizeSampling::ASYMMETRIC>();
    correct &= testAllSizes<float, fk::ResizeSampling::HALF_PIXEL>();
    correct &= testAllSizes<fk::float3, fk::ResizeSampling::HALF_PIXEL>();
    return correct ? 0 : -1;
}
