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

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <random>

// Backwards Vertical Fusion chain with AddBorder and 100 crops, all in a single kernel:
// ReadNV12 -> YUVtoRGB -> Crop -> Resize -> AddBorder -> Mul -> Sub -> Div -> Write
// With a constant color frame, every output plane has known values, both inside the
// resized crop and in the border rows.
namespace fk {

int launch_impl() {
    constexpr int BATCH = 100;
    constexpr uint FRAME_W = 3840;
    constexpr uint FRAME_H = 2160;
    constexpr Size CROP_SIZE(20, 10);
    constexpr Size RESIZED(64, 32);
    constexpr int BORDER = 16;
    constexpr uchar Y_VALUE = 180;

    Stream stream;

    Image<PixelFormat::NV12> frame(FRAME_W, FRAME_H);
    auto frameData = frame.getData();
    for (uint y = 0; y < FRAME_H * 3 / 2; ++y) {
        for (uint x = 0; x < FRAME_W; ++x) {
            frameData.at(static_cast<int>(x), static_cast<int>(y)) = y < FRAME_H ? Y_VALUE : uchar{128};
        }
    }
    frame.upload(stream);

    std::mt19937 gen(42);
    std::uniform_int_distribution<int> distX(0, FRAME_W - CROP_SIZE.width);
    std::uniform_int_distribution<int> distY(0, FRAME_H - CROP_SIZE.height);
    std::array<Rect, BATCH> crops{};
    for (auto& crop : crops) {
        crop = Rect(distX(gen), distY(gen), CROP_SIZE.width, CROP_SIZE.height);
    }

    const float3 mulValue{1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
    const float3 subValue{0.485f, 0.456f, 0.406f};
    const float3 divValue{0.229f, 0.224f, 0.225f};
    const float3 black{0.f, 0.f, 0.f};

    Tensor<float3> output(RESIZED.width, RESIZED.height + 2 * BORDER, BATCH);

    executeOperations<TransformDPP<>>(stream,
        ReadYUV<PixelFormat::NV12>::build(frame),
        ConvertYUVToRGB<ColorDepth::p8bit, ColorRange::Full, ColorPrimitives::bt2020>::build(),
        Crop<>::build(crops),
        Resize<InterpolationType::INTER_LINEAR, AspectRatio::IGNORE_AR>::build(RESIZED),
        AddBorder<>::build(BORDER, BORDER, 0, 0, black),
        Mul<float3>::build(mulValue),
        Sub<float3>::build(subValue),
        Div<float3>::build(divValue),
        TensorWrite<float3>::build(output.ptr()));
    output.download(stream);
    stream.sync();

    const float3 rgb = ConvertYUVToRGB<ColorDepth::p8bit, ColorRange::Full, ColorPrimitives::bt2020>::exec(
        uchar3{Y_VALUE, 128, 128});
    const float3 expectedInside = ((rgb * mulValue) - subValue) / divValue;
    const float3 expectedBorder = ((black * mulValue) - subValue) / divValue;

    float maxError{0.f};
    for (int z = 0; z < BATCH; ++z) {
        for (int y = 0; y < RESIZED.height + 2 * BORDER; ++y) {
            const bool inside = y >= BORDER && y < BORDER + RESIZED.height;
            const float3 expected = inside ? expectedInside : expectedBorder;
            for (int x = 0; x < RESIZED.width; ++x) {
                const float3 value = output.at(x, y, z);
                maxError = std::max({maxError, std::abs(value.x - expected.x), std::abs(value.y - expected.y),
                                     std::abs(value.z - expected.z)});
            }
        }
    }
    const bool correct = maxError <= 1e-3f;
    std::cout << "AddBorder BVF chain, " << BATCH << " crops, max abs error " << maxError
              << (correct ? " PASSED" : " FAILED") << std::endl;
    return correct ? 0 : -1;
}

} // namespace fk

int launch() {
    return fk::launch_impl();
}
