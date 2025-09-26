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

template <typename... IOps>
constexpr size_t testIdxFirstNonBack(const IOps&...) {
    return Back::idxFirstNonBack<IOps...>();
}

bool testBack() {

    // Inputs
    constexpr RawPtr<ND::_2D, uchar3> input{nullptr, {128, 128, 0}};

    // Read Operations
    constexpr auto readOp = PerThreadRead<ND::_2D, uchar3>::build({ input });

    // ReadBack Operations
    constexpr auto cropOp = Crop<>::build(Rect(0, 0, 16, 16));
    constexpr auto resizeOp = Resize<InterpolationType::INTER_LINEAR>::build(Size(1024, 1024));

    // Compute Operations
    constexpr auto castU3F3 = Cast<uchar3, float3>::build();
    constexpr auto mulOpU3 = Mul<uchar3>::build(uchar3{ 2, 2, 2 });
    constexpr auto mulOpF3 = Mul<float3>::build(float3{ 2, 2, 2 });
    //constexpr auto mulOpF = Mul<float>::build( 2.f );
    //constexpr auto addLastF2 = AddLast<float2, float3>::build(45.f);
    //constexpr auto addLastF3 = AddLast<float3, float4>::build(45.f);
    constexpr auto vecReduceF3 = VectorReduce<float3, Add<float>>::build();
    //constexpr auto vecReduceF4 = VectorReduce<float4, Add<float>>::build();
    //constexpr auto addF = Add<float, float, float>::build(10.f);

    // Outputs
    constexpr RawPtr<ND::_2D, float> outputF{nullptr, {1024, 1024, 0}};
    constexpr RawPtr<ND::_2D, float3> outputF3{ nullptr, {1024, 1024, 0} };
    //constexpr RawPtr<ND::_2D, float4> outputF4{ nullptr, {1024, 1024, 0} };

    // Write Operations
    constexpr auto writeF = PerThreadWrite<ND::_2D, float>::build({ outputF });
    constexpr auto writeF3 = PerThreadWrite<ND::_2D, float3>::build({ outputF3 });
    //constexpr auto writeF4 = PerThreadWrite<ND::_2D, float4>::build({ outputF4 });

    // Test no read back
    {
        // Test idxFirstNonBack
        constexpr size_t idx1 = testIdxFirstNonBack(readOp, castU3F3, mulOpF3, writeF3);
        static_assert(idx1 == 0, "idx1 should be 0");
        constexpr auto fusedIOp1 = Back::fuse(readOp, castU3F3, mulOpF3, writeF3);
        static_assert(std::is_same_v<decltype(fusedIOp1), decltype(readOp)>, "Returned operation must be readOp");

        constexpr size_t idx2 = testIdxFirstNonBack(readOp, cropOp, castU3F3, writeF3);
        static_assert(idx2 == 2, "idx2 should be 2");
        constexpr auto fusedIOp2 = Back::fuse(readOp, cropOp, castU3F3, writeF3);
        static_assert(std::is_same_v<ReadBack<Crop<std::decay_t<decltype(readOp)>>>, std::decay_t<decltype(fusedIOp2)>>,
            "fusedIOps must have type ReadBack<Crop<PerThreadRead<ND::_2D, uchar3>>>");

        constexpr size_t idx3 = testIdxFirstNonBack(readOp, cropOp, mulOpU3, resizeOp, mulOpF3, vecReduceF3, writeF);
        static_assert(idx3 == 4, "idx3 should be 4");
        constexpr auto fusedIOp3 = Back::fuse(readOp, cropOp, mulOpU3, resizeOp, mulOpF3, vecReduceF3, writeF);
        using GenerateFuseBack = std::decay_t<decltype(fusedIOp3)>;
        static_assert(std::is_same_v<typename GenerateFuseBack::Operation::InstanceType, ReadBackType>);
        static_assert(std::is_same_v<typename GenerateFuseBack::Operation::BackIOp::Operation::InstanceType, TernaryType>, "Expecting a ternary operation");
        static_assert(GenerateFuseBack::Operation::BackIOp::Operation::BackIOp::Operation::IS_FUSED_OP, "Expecting a fused operation");
        using FusedBackType = ReadBack<ResizeComplete<AspectRatio::IGNORE_AR, Ternary<InterpolateComplete<InterpolationType::INTER_LINEAR, Read<FusedOperation<Crop<Read<PerThreadRead<ND::_2D, uchar3>>>, Mul<uchar3>>>>>>>;
        static_assert(std::is_same_v<std::decay_t<decltype(fusedIOp3)>, FusedBackType>, "Expecting ReadBack<ResizeComplete<AspectRatio::IGNORE_AR, Ternary<InterpolateComplete<InterpolationType::INTER_LINEAR, Read<FusedOperation<Crop<Read<PerThreadRead<ND::_2D, uchar3>>>, Mul<uchar3>>>>>>>");
    }

    return true;
}

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

    return (correct && testBack()) ? 0 : -1;
}