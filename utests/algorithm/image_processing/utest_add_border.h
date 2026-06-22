/* Copyright 2025 Oscar Amoros Huguet
   Copyright 2025 Grup Mediapro S.L.U (Oscar Amoros Huguet)

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

#include <fused_kernel/algorithms/image_processing/add_border.h>
#include <fused_kernel/algorithms/image_processing/border_reader.h>

void testAddBorderConstant() {
    fk::Stream stream;

    // Input and expected values
    constexpr fk::Size inputRes(8, 8);
    constexpr fk::Size outputRes{12, 12};
    constexpr uchar3 ptr[] =
    {{ 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8},
     { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1},
     { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8},
     { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1},
     { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8},
     { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1},
     { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8},
     { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1} };

    // For both  add border constant and border reader constant
    constexpr uchar3 ptrExpectedConstant[] =
    {{ 0,  0,  0}, { 0,  0,  0}, { 0,  0,  0}, { 0,  0,  0}, { 0,  0,  0}, { 0,  0,  0}, { 0,  0,  0}, { 0,  0,  0}, { 0,  0,  0}, { 0,  0,  0}, { 0,  0,  0}, { 0,  0,  0},
     { 0,  0,  0}, { 0,  0,  0}, { 0,  0,  0}, { 0,  0,  0}, { 0,  0,  0}, { 0,  0,  0}, { 0,  0,  0}, { 0,  0,  0}, { 0,  0,  0}, { 0,  0,  0}, { 0,  0,  0}, { 0,  0,  0},
     { 0,  0,  0}, { 0,  0,  0}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 0,  0,  0}, { 0,  0,  0},
     { 0,  0,  0}, { 0,  0,  0}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 0,  0,  0}, { 0,  0,  0},
     { 0,  0,  0}, { 0,  0,  0}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 0,  0,  0}, { 0,  0,  0},
     { 0,  0,  0}, { 0,  0,  0}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 0,  0,  0}, { 0,  0,  0},
     { 0,  0,  0}, { 0,  0,  0}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 0,  0,  0}, { 0,  0,  0},
     { 0,  0,  0}, { 0,  0,  0}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 0,  0,  0}, { 0,  0,  0},
     { 0,  0,  0}, { 0,  0,  0}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 2,  4,  8}, { 0,  0,  0}, { 0,  0,  0},
     { 0,  0,  0}, { 0,  0,  0}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}, { 0,  0,  0}, { 0,  0,  0},
     { 0,  0,  0}, { 0,  0,  0}, { 0,  0,  0}, { 0,  0,  0}, { 0,  0,  0}, { 0,  0,  0}, { 0,  0,  0}, { 0,  0,  0}, { 0,  0,  0}, { 0,  0,  0}, { 0,  0,  0}, { 0,  0,  0},
     { 0,  0,  0}, { 0,  0,  0}, { 0,  0,  0}, { 0,  0,  0}, { 0,  0,  0}, { 0,  0,  0}, { 0,  0,  0}, { 0,  0,  0}, { 0,  0,  0}, { 0,  0,  0}, { 0,  0,  0}, { 0,  0,  0}};

    fk::Ptr2D<uchar3> inputPtr(8, 8);
    fk::Ptr<fk::ND::_2D, uchar3> expectedPtrConstant(12, 12, 0, fk::MemType::Host);

    // Fill inputPtr with the test data
    for (int y = 0; y < inputRes.height; ++y) {
        for (int x = 0; x < inputRes.width; ++x) {
            inputPtr.at(x, y) = ptr[y * inputRes.width + x];
        }
    }

    for (int y = 0; y < outputRes.height; ++y) {
        for (int x = 0; x < outputRes.width; ++x) {
            expectedPtrConstant.at(x, y) = ptr[y * outputRes.width + x];
        }
    }

    // Upload inputPtr to device
    inputPtr.upload(stream);
    
    const auto readIOp = fk::PerThreadRead<fk::ND::_2D, uchar3>::build(inputPtr.ptr());

    const auto addBorderConstThen = readIOp.then(fk::AddBorder::build(2, 2, 2, 2, uchar3{0,0,0}));
    const auto addBorderBReaderThen = readIOp.then(fk::BorderReader<fk::BorderType::CONSTANT>::build(uchar3{0,0,0})).then(fk::AddBorder::build(2, 2, 2, 2));
    const auto addBorderConst = fk::AddBorder::build(readIOp, 2, 2, 2, 2, uchar3{0,0,0});
    const auto addBorderBReader = fk::AddBorder::build(readIOp.then(fk::BorderReader<fk::BorderType::CONSTANT>::build(uchar3{0,0,0})), 2, 2, 2, 2);

    /*using DBlend = typename decltype(blendTest)::Operation;
    using DLinear = typename decltype(linearEvenTest)::Operation;

    TestCaseBuilder<DBlend>::addTest(testCases, stream, blendTest, expectedPtrBlend);
    TestCaseBuilder<DLinear>::addTest(testCases, stream, linearEvenTest, expectedPtrLinearEven);
    TestCaseBuilder<DLinear>::addTest(testCases, stream, linearOddTest, expectedPtrLinearOdd);*/
}

START_ADDING_TESTS
STOP_ADDING_TESTS

int launch() {
    RUN_ALL_TESTS
}