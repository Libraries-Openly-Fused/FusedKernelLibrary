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

#include <fused_kernel/algorithms/algorithms.h>
#include <fused_kernel/core/execution_model/memory_operations.h>
#include <fused_kernel/core/execution_model/executors.h>
#include <iostream>

using namespace fk;

int launch() {
    Stream stream;

    Ptr2D<float2> input(1920, 1080);
    Ptr2D<float2> output1(16, 16);
    Ptr2D<float2> output2(16, 16);

    for (int y = 0; y < 1080; ++y) {
        for (int x = 0; x < 1920; ++x) {
            input.at(x, y) = make_<float2>(static_cast<float>(x), static_cast<float>(y));
        }
    }

    input.upload(stream);

    const auto readOp = PerThreadRead<ND::_2D, float2>::build(input);
    const auto cropOp = Crop<>::build(Rect(128, 256, 64, 64));
    const auto resizeOp = Resize<InterpolationType::INTER_LINEAR>::build(Size(16, 16));
    const auto mulOp = Mul<float2>::build(make_<float2>(3.f, 5.f));
    const auto writeOp1 = PerThreadWrite<ND::_2D, float2>::build(output1);
    const auto writeOp2 = PerThreadWrite<ND::_2D, float2>::build(output2);

    Executor<TransformDPP<>>::executeOperations(stream, readOp, cropOp, resizeOp, mulOp, writeOp1);
    Executor<TransformDPP<>>::executeOperations(stream, readOp.then(cropOp).then(resizeOp), mulOp, writeOp2);

    output1.download(stream);
    output2.download(stream);

    stream.sync();

    bool correct{ true };
    for (int y = 0; y < 16; ++y) {
        for (int x = 0; x < 16; ++x) {
            bool currentCorrect = output1.at(x, y) == output2.at(x, y);
            if (!currentCorrect) {
                std::cout << "Mismatch at (" << x << ", " << y << "): output1 = ("
                    << output1.at(x, y).x << ", " << output1.at(x, y).y << "), output2 = ("
                    << output2.at(x, y).x << ", " << output2.at(x, y).y << ")" << std::endl;
                correct = false;
            }
        }
    }

    return correct ? 0 : -1;
}