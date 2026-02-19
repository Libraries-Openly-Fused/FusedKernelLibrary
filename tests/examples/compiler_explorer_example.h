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

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/core/execution_model/execution_model.h>
#include <fused_kernel/algorithms/algorithms.h>

using namespace fk;

void testLTS0013() {
    // Define input and output data
    Ptr2D<uchar4> input(1920, 1080);
    std::array<Rect, 5> crops{ Rect(0, 0, 120, 40),
                               Rect(100, 200, 60, 40),
                               Rect(400, 20, 30, 50),
                               Rect(1000, 800, 30, 30),
                               Rect(40, 40, 40, 40)}; 
    Tensor<uchar4> output(64, 64, 5);
    Stream stream;

    // Define and execute operations over the data
    executeOperations<TransformDPP<>>(input, stream,
                                      Crop<>::build(crops),
                                      Resize<InterpolationType::INTER_LINEAR>::build(Size(64,64)),
                                      Mul<float4>::build(make_set<float4>(1.f/255.f)),
                                      Mul<float4>::build(make_set<float4>(0.33f)),
                                      Add<float4>::build(make_set<float4>(0.5f)),
                                      Mul<float4>::build(make_set<float4>(255.f)),
                                      SaturateCast<float4, uchar4>::build(),
                                      TensorWrite<uchar4>::build(output));

    stream.sync();
}

int launch() {
    testLTS0013();

    return 0;
}