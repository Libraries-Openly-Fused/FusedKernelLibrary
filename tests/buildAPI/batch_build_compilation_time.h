/* Copyright 2025 Oscar Amoros Huguet

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

#include <fused_kernel/core/execution_model/memory_operations.h>
#include <fused_kernel/core/utils/utils.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/image_processing/crop.h>
#include <fused_kernel/algorithms/image_processing/resize.h>
#include <fused_kernel/algorithms/image_processing/color_conversion.h>
#include <fused_kernel/algorithms/image_processing/image.h>
#include <fused_kernel/algorithms/image_processing/border_reader.h>
#include <fused_kernel/fused_kernel.h>

using namespace fk;

void testCompareReferenceVSValueVSInstantiableDPP() {
    Stream stream;

    // We set all outputs to the same size
    const Size outputSize(60, 60);

    // We perform 5 crops on the image
    constexpr int BATCH_10 = 10;
    constexpr int BATCH = 100;

    // We have a 4K source image
    Image<PixelFormat::NV12> inputImage(3840, 2160);

    // Intermediate RGB image after YUV to RGB conversion
    Ptr2D<float3> rgbImg(3840, 2160);

    // Crops can be of different sizes
    constexpr std::array<Rect, BATCH_10> crops_10{ Rect(0, 0, 34, 25),      Rect(40, 40, 70, 15),     Rect(100, 200, 60, 59),
                                         Rect(300, 1000, 20, 23), Rect(3000, 2000, 12, 11), Rect(0, 0, 34, 25),
                                         Rect(40, 40, 70, 15),    Rect(100, 200, 60, 59),   Rect(300, 1000, 20, 23),
                                         Rect(3000, 2000, 12, 11) };
    std::array<Rect, BATCH> crops{};
    std::array<Ptr2D<float3>, BATCH> cropedPtrs;

    for (int i = 0; i < BATCH_10; ++i) {
        int j{ 0 };
        for (auto&& elem : crops_10) {
            crops[i + j] = elem;
            j++;
        }
    }

    const float3 backgroundColor{ 0.f, 0.f, 0.f };

    // Create the operation instances once, and use them multiple times
    const auto readIOp = ReadYUV<PixelFormat::NV12>::build(inputImage);
    const auto yuvToRGB = ConvertYUVToRGB<PixelFormat::NV12,
        ColorRange::Full,
        ColorPrimitives::bt2020,
        false, float3>::build();
    const auto borderReader = BorderReader<BorderType::REPLICATE>::build();
    const auto cropIOp = Crop<>::build(crops);
    const auto resizeIOp =
        Resize<InterpolationType::INTER_LINEAR, AspectRatio::PRESERVE_AR>::build(outputSize, backgroundColor);

    const auto fusedPipeline = readIOp & yuvToRGB & borderReader & cropIOp & resizeIOp;

    stream.sync();
}

int launch() {
    testCompareReferenceVSValueVSInstantiableDPP();

    return 0;
}