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

#ifndef FK_UTEST_CONVOLUTION2D_H
#define FK_UTEST_CONVOLUTION2D_H

#include <fused_kernel/algorithms/convolutions/convolution2d.h>
#include <fused_kernel/algorithms/image_processing/image_processing.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/data/rect.h>
#include <fused_kernel/fused_kernel.h>

using namespace fk;

int launch() {

    Stream stream;
    Ptr2D<uchar3> input(3840, 2160);
    Tensor<float3> output(256, 64, 8);

    std::array<Rect, 8> crops{Rect(0, 0, 23, 44), Rect(33, 133, 30, 50),
                              Rect(100, 200, 23, 44), Rect(314, 500, 30, 50),
                              Rect(10, 500, 23, 44), Rect(800, 10, 30, 50),
                              Rect(1000, 1000, 23, 44), Rect(2000, 2000, 30, 50)};

    executeOperations<TransformDPP<>>(input, stream, 
                                      Crop<>::build(crops),
                                      Resize<>::build(Size(256, 64)),
                                      Conv2D<>::build(Dilate<3,3>{}),
                                      Conv2D<>::build(Erode<3,3>{}),
                                      Conv2D<>::build(Dilate<3,3>{}),
                                      Conv2D<>::build(Erode<3,3>{}),
                                      TensorWrite<float3>::build(output));

    stream.sync();

    return 0;
}

#endif // FK_UTEST_CONVOLUTION2D_H