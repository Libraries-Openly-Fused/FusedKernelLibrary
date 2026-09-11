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

#include <tests/operation_test_utils.h>

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/image_processing/deinterlace.h>

int launch() {
    fk::Stream stream;

    // Input and expected values
    constexpr fk::Size res(8, 8);
    constexpr fk::uchar3 ptr[] =
    {{ 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8},
     { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1},
     { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8},
     { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1},
     { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8},
     { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1},
     { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8},
     { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1} };

    constexpr fk::float3 ptrExpectedBlend[] =
    {{2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f},
     {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f},
     {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f},
     {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f},
     {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f},
     {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f},
     {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f},
     {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f} };

    constexpr fk::float3 ptrExpectedLinearEvenLines[] =
    {{2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f},
     {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f},
     {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f},
     {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f},
     {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f},
     {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f},
     {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f},
     {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f} };

    constexpr fk::float3 ptrExpectedLinearOddLines[] =
    {{2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f},
     {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f},
     {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f},
     {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f},
     {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f},
     {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f},
     {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f},
     {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f} };

    fk::Ptr2D<fk::uchar3> inputPtr(8, 8);
    fk::Ptr<fk::ND::_2D, fk::float3> expectedPtrBlend(8, 8, 0, fk::MemType::Host);
    fk::Ptr<fk::ND::_2D, fk::float3> expectedPtrLinearEven(8, 8, 0, fk::MemType::Host);
    fk::Ptr<fk::ND::_2D, fk::float3> expectedPtrLinearOdd(8, 8, 0, fk::MemType::Host);

    // Fill inputPtr with the test data
    for (int y = 0; y < res.height; ++y) {
        for (int x = 0; x < res.width; ++x) {
            inputPtr.at(x, y) = ptr[y * res.width + x];
            expectedPtrBlend.at(x, y) = ptrExpectedBlend[y * res.width + x];
            expectedPtrLinearEven.at(x, y) = ptrExpectedLinearEvenLines[y * res.width + x];
            expectedPtrLinearOdd.at(x, y) = ptrExpectedLinearOddLines[y * res.width + x];
        }
    }

    // Upload inputPtr to device
    inputPtr.upload(stream);
    
    const auto readIOp = fk::PerThreadRead<fk::ND::_2D, fk::uchar3>::build(inputPtr.ptr());

    const fk::DeinterlaceParameters<fk::DeinterlaceType::INTER_LINEAR> paramsLinearEven{ true };
    const fk::DeinterlaceParameters<fk::DeinterlaceType::INTER_LINEAR> paramsLinearOdd{ false };

    const auto blendTest = readIOp.then(fk::Deinterlace<fk::DeinterlaceType::BLEND>::build());
    const auto linearEvenTest = readIOp.then(fk::Deinterlace<fk::DeinterlaceType::INTER_LINEAR>::build(paramsLinearEven));
    const auto linearOddTest = readIOp.then(fk::Deinterlace<fk::DeinterlaceType::INTER_LINEAR>::build(paramsLinearOdd));

    using DBlend = typename decltype(blendTest)::Operation;
    using DLinear = typename decltype(linearEvenTest)::Operation;

    TestCaseBuilder<DBlend>::addTest(testCases, stream, blendTest, expectedPtrBlend);
    TestCaseBuilder<DLinear>::addTest(testCases, stream, linearEvenTest, expectedPtrLinearEven);
    TestCaseBuilder<DLinear>::addTest(testCases, stream, linearOddTest, expectedPtrLinearOdd);

    return 0;
}