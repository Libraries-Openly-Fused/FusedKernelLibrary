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

#include <tests/main.h>

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/add_border.h>
#include <fused_kernel/algorithms/image_processing/color_conversion.h>
#include <fused_kernel/algorithms/image_processing/crop.h>
#include <fused_kernel/algorithms/image_processing/image.h>
#include <fused_kernel/algorithms/image_processing/resize.h>

#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

// Backwards Vertical Fusion chain with AddBorder and 100 crops, all in a single kernel:
// ReadNV12 -> YUVtoRGB -> Crop -> Resize -> AddBorder -> Mul -> Sub -> Div -> Write
// The frame is random, the border is different on every side and every output plane is compared with a reference
// computed on the host: the crop read from the NV12 data, bilinear resize with FKL's mapping (src = dst * scale),
// border and normalization.
namespace fk {

using YUVToRGB = ConvertYUVToRGB<ColorDepth::p8bit, ColorRange::Full, ColorPrimitives::bt2020>;

int launch_impl() {
    constexpr size_t BATCH = 100;
    constexpr uint FRAME_W = 3840;
    constexpr uint FRAME_H = 2160;
    constexpr Size CROP_SIZE(20, 10);
    constexpr Size RESIZED(64, 32);
    constexpr int TOP = 16, BOTTOM = 12, LEFT = 3, RIGHT = 5;
    constexpr int OUT_W = RESIZED.width + LEFT + RIGHT;
    constexpr int OUT_H = RESIZED.height + TOP + BOTTOM;

    Stream stream;

    Image<PixelFormat::NV12> frame(FRAME_W, FRAME_H);
    auto frameData = frame.getData();
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> byte(0, 255);
    for (uint y = 0; y < FRAME_H * 3 / 2; ++y) {
        for (uint x = 0; x < FRAME_W; ++x) {
            frameData.at(static_cast<int>(x), static_cast<int>(y)) = static_cast<uchar>(byte(gen));
        }
    }
    frame.upload(stream);

    std::uniform_int_distribution<int> distX(0, static_cast<int>(FRAME_W) - CROP_SIZE.width);
    std::uniform_int_distribution<int> distY(0, static_cast<int>(FRAME_H) - CROP_SIZE.height);
    std::array<Rect, BATCH> crops{};
    for (auto& crop : crops) {
        // Drawn one after the other: the evaluation order of function arguments is not specified
        const uint x = static_cast<uint>(distX(gen));
        const uint y = static_cast<uint>(distY(gen));
        crop = Rect(x, y, CROP_SIZE.width, CROP_SIZE.height);
    }

    const float3 mulValue{1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
    const float3 subValue{0.485f, 0.456f, 0.406f};
    const float3 divValue{0.229f, 0.224f, 0.225f};
    const float3 borderValue{32.f, 64.f, 128.f};

    Tensor<float3> output(OUT_W, OUT_H, BATCH);

    executeOperations<TransformDPP<>>(stream,
        ReadYUV<PixelFormat::NV12>::build(frame),
        YUVToRGB::build(),
        Crop<>::build(crops),
        Resize<InterpolationType::INTER_LINEAR, AspectRatio::IGNORE_AR>::build(RESIZED),
        AddBorder<>::build(TOP, BOTTOM, LEFT, RIGHT, borderValue),
        Mul<float3>::build(mulValue),
        Sub<float3>::build(subValue),
        Div<float3>::build(divValue),
        TensorWrite<float3>::build(output.ptr()));
    output.download(stream);
    stream.sync();

    // Host reference
    const auto rgbAt = [&](const int x, const int y) {
        const int uvX = x & ~1;
        const int uvY = static_cast<int>(FRAME_H) + y / 2;
        const uchar3 yuv{frameData.at(x, y), frameData.at(uvX, uvY), frameData.at(uvX + 1, uvY)};
        return YUVToRGB::exec(yuv);
    };
    const auto normalize = [&](const float3 v) { return ((v * mulValue) - subValue) / divValue; };
    const float scaleX = static_cast<float>(CROP_SIZE.width) / static_cast<float>(RESIZED.width);
    const float scaleY = static_cast<float>(CROP_SIZE.height) / static_cast<float>(RESIZED.height);
    constexpr float TOLERANCE = 1e-3f;

    size_t mismatches = 0;
    float maxError = 0.f;
    for (size_t z = 0; z < BATCH; ++z) {
        const int cropX = static_cast<int>(crops[z].x);
        const int cropY = static_cast<int>(crops[z].y);
        for (int y = 0; y < OUT_H; ++y) {
            for (int x = 0; x < OUT_W; ++x) {
                const int rx = x - LEFT;
                const int ry = y - TOP;
                float3 expected;
                if (rx >= 0 && rx < RESIZED.width && ry >= 0 && ry < RESIZED.height) {
                    const float srcX = static_cast<float>(rx) * scaleX;
                    const float srcY = static_cast<float>(ry) * scaleY;
                    const int x1 = static_cast<int>(std::floor(srcX));
                    const int y1 = static_cast<int>(std::floor(srcY));
                    const int x2 = std::min(x1 + 1, CROP_SIZE.width - 1);
                    const int y2 = std::min(y1 + 1, CROP_SIZE.height - 1);
                    const float ax = srcX - static_cast<float>(x1);
                    const float ay = srcY - static_cast<float>(y1);
                    const float3 top = rgbAt(cropX + x1, cropY + y1) * (1.f - ax) + rgbAt(cropX + x2, cropY + y1) * ax;
                    const float3 bottom = rgbAt(cropX + x1, cropY + y2) * (1.f - ax) + rgbAt(cropX + x2, cropY + y2) * ax;
                    expected = normalize(top * (1.f - ay) + bottom * ay);
                } else {
                    expected = normalize(borderValue);
                }
                const float3 value = output.at(x, y, static_cast<int>(z));
                for (const float error : {std::abs(value.x - expected.x), std::abs(value.y - expected.y),
                                          std::abs(value.z - expected.z)}) {
                    // NaN safe: a NaN error is not <= TOLERANCE
                    if (!(error <= TOLERANCE)) {
                        ++mismatches;
                    }
                    if (std::isnan(error) || error > maxError) {
                        maxError = std::isnan(maxError) ? maxError : error;
                    }
                }
            }
        }
    }
    const bool correct = mismatches == 0;
    std::cout << "AddBorder BVF chain, " << BATCH << " crops, max abs error " << maxError << ", " << mismatches
              << " values out of tolerance" << (correct ? " PASSED" : " FAILED") << std::endl;
    return correct ? 0 : -1;
}

} // namespace fk

int launch() {
    return fk::launch_impl();
}
