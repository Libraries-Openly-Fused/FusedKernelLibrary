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

namespace fk {

int launch_impl() {
    Stream stream;

    // Input and expected values
    constexpr Size res(8, 8);
    constexpr uchar3 ptr[] =
    {{ 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8},
     { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1},
     { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8},
     { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1},
     { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8},
     { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1},
     { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8},
     { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1} };

    constexpr float3 ptrExpectedBlend[] =
    {{2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f},
     {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f},
     {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f},
     {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f},
     {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f},
     {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f},
     {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f},
     {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f}, {2.f, 3.f, 5.f} };

    constexpr float3 ptrExpectedLinearEvenLines[] =
    {{2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f},
     {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f},
     {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f},
     {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f},
     {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f},
     {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f},
     {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f},
     {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f} };

    constexpr float3 ptrExpectedLinearOddLines[] =
    {{2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f}, {2.f, 4.f, 8.f},
     {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f},
     {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f},
     {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f},
     {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f},
     {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f},
     {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f},
     {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f} };

    Ptr2D<uchar3> inputPtr(8, 8);
    Ptr<ND::_2D, float3> expectedPtrBlend(8, 8, 0, MemType::Host);
    Ptr<ND::_2D, float3> expectedPtrLinearEven(8, 8, 0, MemType::Host);
    Ptr<ND::_2D, float3> expectedPtrLinearOdd(8, 8, 0, MemType::Host);

    // Fill inputPtr with the test data
    for (int y = 0; y < res.height; ++y) {
        for (int x = 0; x < res.width; ++x) {
            inputPtr.at(x, y) = ptr[y * res.width + x];
            expectedPtrBlend.at(x, y) = ptrExpectedBlend[y * res.width + x];
            expectedPtrLinearEven.at(x, y) = ptrExpectedLinearEvenLines[y * res.width + x] +
                                           ((y % 2 == 1 && y < 7) ? 0.5f : 0.f);
            expectedPtrLinearOdd.at(x, y) = ptrExpectedLinearOddLines[y * res.width + x] +
                                          ((y % 2 == 0 && y > 0) ? 0.5f : 0.f);
        }
    }

    // Upload inputPtr to device
    inputPtr.upload(stream);

    const auto readIOp = PerThreadRead<ND::_2D, uchar3>::build(inputPtr.ptr());

    const DeinterlaceParameters<DeinterlaceType::INTER_LINEAR> paramsLinearEven{ true };
    const DeinterlaceParameters<DeinterlaceType::INTER_LINEAR> paramsLinearOdd{ false };

    const auto blendTest = readIOp.then(Deinterlace<DeinterlaceType::BLEND>::build());
    const auto linearEvenTest = readIOp.then(Deinterlace<DeinterlaceType::INTER_LINEAR>::build(paramsLinearEven));
    const auto linearOddTest = readIOp.then(Deinterlace<DeinterlaceType::INTER_LINEAR>::build(paramsLinearOdd));

    using DBlend = typename decltype(blendTest)::Operation;
    using DLinear = typename decltype(linearEvenTest)::Operation;

    TestCaseBuilder<DBlend>::addTest(testCases, stream, blendTest, expectedPtrBlend);
    TestCaseBuilder<DLinear>::addTest(testCases, stream,
                                     std::array{linearEvenTest, linearOddTest},
                                     std::array{expectedPtrLinearEven, expectedPtrLinearOdd});

    bool correct{true};
    for (const auto& testCase : testCases) {
        correct &= testCase.second();
}
    testCases.clear();
    return correct ? 0 : -1;
}

} // namespace fk

int launch() {
    return fk::launch_impl();
}
