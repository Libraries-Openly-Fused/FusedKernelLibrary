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

// ConvertYUVToRGB for every combination of ColorRange and ColorPrimitives that has a matrix (ccMatrix): the output must
// be the YCbCr to RGB conversion of that matrix, on every back-end (this test is built for CPU, CUDA and HIP). A
// combination without a matrix (for instance ColorRange::Limited with ColorPrimitives::bt601) does not compile: ccMatrix has
// a static_assert for it; it used to be a matrix of zeros, so the conversion gave black without a message.
//
// ColorRange::Full with ColorPrimitives::bt601 is the BT.601 studio swing (limited range) input, Y in 16..235 and chroma in
// 16..240, converted to full range RGB, the conversion of cv::COLOR_YUV2RGB_NV12: Y = 150, U = V = 128 gives 156, 156, 156.
#include <tests/main.h>

#include <fused_kernel/algorithms/algorithms.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/color_conversion.h>
#include <fused_kernel/core/execution_model/executors.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

struct Coefficients {
    double yGain, yOffset;           // luma: yGain * (Y - yOffset)
    double rCr, gCb, gCr, bCb;       // chroma terms
};

// The conversion of a matrix, in double precision (the rows of the matrices of color_conversion.h)
std::array<double, 3> referenceRGB(const Coefficients& k, const double y, const double cb, const double cr) {
    const double luma = k.yGain * (y - k.yOffset);
    return {luma + k.rCr * (cr - 128.0), luma + k.gCb * (cb - 128.0) + k.gCr * (cr - 128.0), luma + k.bCb * (cb - 128.0)};
}

template <fk::ColorRange CR, fk::ColorPrimitives CP>
bool testCombination(const char* name, const Coefficients& k) {
    constexpr int YS = 4, CBS = 5, CRS = 5;
    const double ys[YS] = {16, 82, 150, 235}, cs[CBS] = {16, 90, 128, 200, 240};
    constexpr int W = CBS * CRS, H = YS;
    fk::Stream stream;
    fk::Ptr2D<fk::uchar3> input(W, H);
    fk::Ptr2D<fk::uchar3> output(W, H);
    for (int y = 0; y < YS; ++y) {
        for (int cb = 0; cb < CBS; ++cb) {
            for (int cr = 0; cr < CRS; ++cr) {
                input.at(cb * CRS + cr, y) = fk::make_<fk::uchar3>(static_cast<uchar>(ys[y]), static_cast<uchar>(cs[cb]), static_cast<uchar>(cs[cr]));
            }
        }
    }
    input.upload(stream);
    fk::Executor<fk::TransformDPP<>>::executeOperations(
        stream, fk::PerThreadRead<fk::ND::_2D, fk::uchar3>::build(input),
        fk::ConvertYUVToRGB<fk::PixelFormat::NV12, CR, CP, false, fk::uchar3>::build(),
        fk::PerThreadWrite<fk::ND::_2D, fk::uchar3>::build(output));
    output.download(stream);
    stream.sync();

    bool correct = true;
    int worst = 0;
    for (int y = 0; y < YS; ++y) {
        for (int cb = 0; cb < CBS; ++cb) {
            for (int cr = 0; cr < CRS; ++cr) {
                const auto expected = referenceRGB(k, ys[y], cs[cb], cs[cr]);
                const fk::uchar3 got = output.at(cb * CRS + cr, y);
                const int gotRGB[3] = {got.x, got.y, got.z};
                for (int c = 0; c < 3; ++c) {
                    const int want = static_cast<int>(std::min(255.0, std::max(0.0, std::floor(expected[c] + 0.5))));
                    // The matrices are float and the result is rounded, so one level of difference is allowed
                    worst = std::max(worst, std::abs(gotRGB[c] - want));
                }
            }
        }
    }
    if (worst > 1) {
        correct = false;
        std::cout << name << ": the conversion differs from the reference by " << worst << " levels" << std::endl;
    }
    return correct;
}

// The case that gave black: BT.601 studio swing gray
bool testGray601() {
    fk::Stream stream;
    fk::Ptr2D<fk::uchar3> input(3, 1);
    fk::Ptr2D<fk::uchar3> output(3, 1);
    input.at(0, 0) = fk::make_<fk::uchar3>(16, 128, 128);
    input.at(1, 0) = fk::make_<fk::uchar3>(150, 128, 128);
    input.at(2, 0) = fk::make_<fk::uchar3>(235, 128, 128);
    input.upload(stream);
    fk::Executor<fk::TransformDPP<>>::executeOperations(
        stream, fk::PerThreadRead<fk::ND::_2D, fk::uchar3>::build(input),
        fk::ConvertYUVToRGB<fk::PixelFormat::NV12, fk::ColorRange::Full, fk::ColorPrimitives::bt601, false, fk::uchar3>::build(),
        fk::PerThreadWrite<fk::ND::_2D, fk::uchar3>::build(output));
    output.download(stream);
    stream.sync();
    const int expected[3] = {0, 156, 255};
    bool correct = true;
    for (int x = 0; x < 3; ++x) {
        const fk::uchar3 got = output.at(x, 0);
        if (got.x != expected[x] || got.y != expected[x] || got.z != expected[x]) {
            std::cout << "BT.601 gray Y=" << (x == 0 ? 16 : (x == 1 ? 150 : 235)) << ": expected " << expected[x] << " got " << int(got.x) << ","
                      << int(got.y) << "," << int(got.z) << std::endl;
            correct = false;
        }
    }
    return correct;
}

} // namespace

int launch() {
    bool correct = true;
    correct &= testGray601();
    // The rows of ccMatrix<Full, bt601>: a gain of 1.164383562 on (Y - 16), then the chroma terms
    correct &= testCombination<fk::ColorRange::Full, fk::ColorPrimitives::bt601>("Full + bt601", {1.164383562, 16.0, 1.596026786, -0.39176229, -0.812967647, 2.017232143});
    correct &= testCombination<fk::ColorRange::Full, fk::ColorPrimitives::bt709>("Full + bt709", {1.0, 0.0, 1.5748, -0.1873, -0.4681, 1.8556});
    correct &= testCombination<fk::ColorRange::Limited, fk::ColorPrimitives::bt709>("Limited + bt709", {1.0, 0.0, 1.402, -0.34414, -0.71414, 1.772});
    correct &= testCombination<fk::ColorRange::Full, fk::ColorPrimitives::bt2020>("Full + bt2020", {1.0, 0.0, 1.4746, -0.16455312684366, -0.57135312684366, 1.8814});
    return correct ? 0 : -1;
}
