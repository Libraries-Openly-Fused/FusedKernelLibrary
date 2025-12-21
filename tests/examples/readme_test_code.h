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
#include <fused_kernel/algorithms/image_processing/color_conversion.h>
#include <fused_kernel/algorithms/image_processing/resize.h>
#include <fused_kernel/fused_kernel.h>

using namespace fk;

int launch() {
    Stream stream;

    // We set all outputs to the same size
    const Size outputSize(60, 60);
    // We perform 5 crops on the image
    constexpr int BATCH = 5;

    // We have a 4K source image
    Ptr2D<uchar3> inputImage(3840, 2160);

    // We want a Tensor of contiguous memory for all images
    Tensor<float3> output(outputSize.width, outputSize.height, BATCH);

    // Crops can be of different sizes
    std::array<Rect, BATCH> crops{
        Rect(0, 0, 34, 25),
        Rect(10, 10, 70, 15),
        Rect(20, 20, 60, 59),
        Rect(30, 30, 20, 23),
        Rect(40, 40, 12, 11)
    };

    //initImageValues(inputImage);
    const float3 backgroundColor{ 0.f, 0.f, 0.f };

    const float3 mulValue = make_set<float3>(1.4f);
    const float3 subValue = make_set<float3>(0.5f);
    const float3 divValue = make_set<float3>(255.f);

    // Create a fused operation that reads the input image,
    // crops it, resizes it, and applies arithmetic operations
    const auto mySender = PerThreadRead<ND::_2D, uchar3>::build(inputImage)
        .then(Crop<>::build(crops))
        .then(Resize<InterpolationType::INTER_LINEAR, AspectRatio::PRESERVE_AR>::build(outputSize, backgroundColor))
        .then(Mul<float3>::build(mulValue))
        .then(Sub<float3>::build(subValue))
        .then(Div<float3>::build(divValue))
        .then(ColorConversion<ColorConversionCodes::COLOR_RGB2BGR, float3, float3>::build());

    // Define the last operation that will write the results to the output pointer
    const auto myReceiver = TensorWrite<float3>::build(output);

    // Execute the operations in a single kernel
    // At compile time, the types are used to define the kernel code
    // At runtime, the kernel is executed with the provided parameters
    executeOperations<fk::TransformDPP<>>(stream, mySender, myReceiver);
    stream.sync();

    // Use the Tensor for inference

    // Now in CPU
    Stream_<ParArch::CPU> stream_cpu;
    // We have a 4K source image
    Ptr2D<uchar3> cpu_inputImage(3840, 2160, 0, MemType::Host);

    // We want a Tensor of contiguous memory for all images
    Tensor<float3> cpu_output(outputSize.width, outputSize.height, BATCH, 1, MemType::Host);


    // Create a fused operation that reads the input image,
    // crops it, resizes it, and applies arithmetic operations
    const auto mySender_cpu = PerThreadRead<ND::_2D, uchar3>::build(cpu_inputImage)
        .then(Crop<>::build(crops))
        .then(Resize<InterpolationType::INTER_LINEAR, AspectRatio::PRESERVE_AR>::build(outputSize, backgroundColor))
        .then(Mul<float3>::build(mulValue))
        .then(Sub<float3>::build(subValue))
        .then(Div<float3>::build(divValue))
        .then(ColorConversion<ColorConversionCodes::COLOR_RGB2BGR, float3, float3>::build());

    // Define the last operation that will write the results to the output pointer
    const auto myReceiver_cpu = TensorWrite<float3>::build(cpu_output);
    // Execute the operations in a single kernel
    // At compile time, the types are used to define the kernel code
    // At runtime, the kernel is executed with the provided parameters
    executeOperations<TransformDPP<ParArch::CPU>>(stream_cpu, mySender_cpu, myReceiver_cpu);
    stream_cpu.sync();

    return 0;
}
