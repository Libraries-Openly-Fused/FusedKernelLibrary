/* Copyright 2025 Oscar Amoros Huguet
   Copyright 2025 Grup Mediapro S.L.U

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include <fused_kernel/algorithms/image_processing/color_conversion.h>
#include <fused_kernel/core/utils/vector_utils.h>
#include <fused_kernel/core/utils/type_to_string.h>
#include <tests/operation_test_utils.h>
#include <fused_kernel/algorithms/image_processing/image.h>

// Test PixelFormatTraits for UYVY
void testUYVYPixelFormatTraits() {
    constexpr fk::ColorSpace expectedSpace = fk::ColorSpace::YUV422;
    constexpr fk::ColorDepth expectedDepth = fk::ColorDepth::p8bit;
    constexpr size_t expectedCn = 3;
    
    static_assert(fk::PixelFormatTraits<fk::PixelFormat::UYVY>::space == expectedSpace, "UYVY space should be YUV422");
    static_assert(fk::PixelFormatTraits<fk::PixelFormat::UYVY>::depth == expectedDepth, "UYVY depth should be p8bit");
    static_assert(fk::PixelFormatTraits<fk::PixelFormat::UYVY>::cn == expectedCn, "UYVY cn should be 3");
}

// Test IsEven function used in UYVY processing
void testIsEvenFunction() {
    constexpr std::array<uint, 4> inputVals{ 0, 1, 2, 3 };
    constexpr std::array<bool, 4> expectedVals{ true, false, true, false };

    TestCaseBuilder<fk::IsEven<uint>>::addTest(testCases, inputVals, expectedVals);
}

// Test RGB2Gray conversion functionality
void testRGB2GrayConversion() {
    // Test with known RGB values
    constexpr std::array<uchar3, 3> inputVals{
        uchar3{255, 0, 0},    // Pure red -> expected ~77 (0.299*255)
        uchar3{0, 255, 0},    // Pure green -> expected ~150 (0.587*255)  
        uchar3{0, 0, 255}     // Pure blue -> expected ~29 (0.114*255)
    };

    constexpr std::array<uchar, 3> expectedVals{
        76,    // 0.299*255 ≈ 76.245 -> 76
        150,   // 0.587*255 ≈ 149.685 -> 150 (rounded)
        29     // 0.114*255 ≈ 29.07 -> 29
    };

    TestCaseBuilder<fk::RGB2Gray<uchar3, uchar>>::addTest(testCases, inputVals, expectedVals);
}

// Test AddOpaqueAlpha functionality
void testAddOpaqueAlpha() {
    constexpr std::array<uchar3, 2> inputVals{
        uchar3{100, 150, 200},
        uchar3{50, 75, 125}
    };

    constexpr std::array<uchar4, 2> expectedVals{
        uchar4{100, 150, 200, 255},  // Alpha = maxDepthValue<p8bit> = 255
        uchar4{50, 75, 125, 255}
    };

    TestCaseBuilder<fk::AddOpaqueAlpha<uchar3, fk::ColorDepth::p8bit>>::addTest(testCases, inputVals, expectedVals);
}

// Test ColorConversion operations
void testColorConversionOperations() {
    // Test BGR2BGRA conversion (adds alpha)
    constexpr std::array<uchar3, 2> inputVals1{
        uchar3{100, 150, 200},
        uchar3{50, 75, 125}
    };

    constexpr std::array<uchar4, 2> expectedVals1{
        uchar4{100, 150, 200, 255},
        uchar4{50, 75, 125, 255}
    };

    TestCaseBuilder<fk::ColorConversion<fk::ColorConversionCodes::COLOR_BGR2BGRA, uchar3, uchar4>>::addTest(testCases, inputVals1, expectedVals1);

    // Test BGR2RGB conversion (channel reorder)  
    constexpr std::array<uchar3, 2> inputVals2{
        uchar3{100, 150, 200},  // BGR
        uchar3{50, 75, 125}     // BGR
    };

    constexpr std::array<uchar3, 2> expectedVals2{
        uchar3{200, 150, 100},  // RGB (channels 2,1,0)
        uchar3{125, 75, 50}     // RGB (channels 2,1,0)
    };

    TestCaseBuilder<fk::ColorConversion<fk::ColorConversionCodes::COLOR_BGR2RGB, uchar3, uchar3>>::addTest(testCases, inputVals2, expectedVals2);
}

void testStaticAddAlpha() {
    // Test StaticAddAlpha with alpha value 255
    using StaticAddAlphaTest = fk::StaticAddAlpha<uchar3, 255>;

    std::array<uchar3, 2> inputVals = {
        uchar3{100, 150, 200},
        uchar3{50, 75, 125}
    };

    std::array<uchar4, 2> expectedVals = {
        uchar4{100, 150, 200, 255},
        uchar4{50, 75, 125, 255}
    };

    TestCaseBuilder<StaticAddAlphaTest>::addTest(testCases, inputVals, expectedVals);
}

void testBGR2Gray() {
    // Test BGR2Gray with CCIR_601 formula  
    // Formula uses input.x * 0.299 + input.y * 0.587 + input.z * 0.114
    using BGR2GrayTest = fk::RGB2Gray<uchar3, uchar, fk::GrayFormula::CCIR_601>;

    std::array<uchar3, 2> inputVals = {
        uchar3{50, 100, 150},   // x=50, y=100, z=150
        uchar3{75, 125, 200}    // x=75, y=125, z=200
    };

    // Expected gray values using formula: x * 0.299 + y * 0.587 + z * 0.114
    std::array<uchar, 2> expectedVals = {
        static_cast<uchar>(std::nearbyint(50 * 0.299f + 100 * 0.587f + 150 * 0.114f)), // ~91
        static_cast<uchar>(std::nearbyint(75 * 0.299f + 125 * 0.587f + 200 * 0.114f))  // ~119
    };

    TestCaseBuilder<BGR2GrayTest>::addTest(testCases, inputVals, expectedVals);
}

void testAddOpaqueAlphaStruct() {
    // Test AddOpaqueAlpha struct with 8-bit depth
    using AddOpaqueAlphaTest = fk::AddOpaqueAlpha<uchar3, fk::ColorDepth::p8bit>;

    std::array<uchar3, 2> inputVals = {
        uchar3{100, 150, 200},
        uchar3{50, 75, 125}
    };

    std::array<uchar4, 2> expectedVals = {
        uchar4{100, 150, 200, 255},  // Alpha = 255 for 8-bit
        uchar4{50, 75, 125, 255}
    };

    TestCaseBuilder<AddOpaqueAlphaTest>::addTest(testCases, inputVals, expectedVals);
}

void testDenormalizePixel() {
    // Test DenormalizePixel with 8-bit depth
    using DenormalizePixelTest = fk::DenormalizePixel<float3, fk::ColorDepth::p8bit>;
    
    std::array<float3, 2> inputVals = {
        float3{0.0f, 0.5f, 1.0f},      // Normalized values [0, 1]
        float3{0.25f, 0.75f, 0.9f}
    };
    
    std::array<float3, 2> expectedVals = {
        float3{0.0f, 127.5f, 255.0f},     // Denormalized to [0, 255]
        float3{63.75f, 191.25f, 229.5f}
    };
    
    TestCaseBuilder<DenormalizePixelTest>::addTest(testCases, inputVals, expectedVals);
}

void testNormalizePixel() {
    // Test NormalizePixel with 8-bit depth
    using NormalizePixelTest = fk::NormalizePixel<uchar3, fk::ColorDepth::p8bit>;
    
    std::array<uchar3, 2> inputVals = {
        uchar3{0, 128, 255},
        uchar3{64, 192, 32}
    };
    
    std::array<float3, 2> expectedVals = {
        float3{0.0f, 128.0f/255.0f, 1.0f},        // Normalized to [0, 1]
        float3{64.0f/255.0f, 192.0f/255.0f, 32.0f/255.0f}
    };
    
    TestCaseBuilder<NormalizePixelTest>::addTest(testCases, inputVals, expectedVals);
}

void testSaturateDenormalizePixel() {
    // Test SaturateDenormalizePixel with 8-bit depth
    using SaturateDenormalizePixelTest = fk::SaturateDenormalizePixel<float3, uchar3, fk::ColorDepth::p8bit>;
    
    std::array<float3, 2> inputVals = {
        float3{-0.5f, 0.5f, 1.5f},     // Values that need saturation and denormalization
        float3{0.25f, 0.75f, 0.9f}
    };
    
    std::array<uchar3, 2> expectedVals = {
        uchar3{0, 128, 255},            // Saturated to [0,1] then denormalized to [0,255]
        uchar3{63, 191, 229}            // 0.25*255=63.75≈63, 0.75*255=191.25≈191, 0.9*255=229.5≈229
    };
    
    TestCaseBuilder<SaturateDenormalizePixelTest>::addTest(testCases, inputVals, expectedVals);
}

void testNormalizeColorRangeDepth() {
    // Test NormalizeColorRangeDepth with 8-bit depth
    // For 8-bit, floatShiftFactor is 1.0f, so input * 1.0f = input (unchanged)
    using NormalizeColorRangeDepthTest = fk::NormalizeColorRangeDepth<float3, fk::ColorDepth::p8bit>;
    
    std::array<float3, 2> inputVals = {
        float3{0.0f, 128.0f, 255.0f},    
        float3{64.0f, 192.0f, 100.0f}
    };
    
    // For 8-bit depth, floatShiftFactor = 1.0f, so output = input * 1.0f = input
    std::array<float3, 2> expectedVals = {
        float3{0.0f, 128.0f, 255.0f},    
        float3{64.0f, 192.0f, 100.0f}
    };

    TestCaseBuilder<NormalizeColorRangeDepthTest>::addTest(testCases, inputVals, expectedVals);
}

void testReadYUV() {
    fk::Stream stream;

    // Test ReadYUV with UYVY format
    using ReadYUVTest = fk::ReadYUV<fk::PixelFormat::UYVY>;

    // Input and expected values
    constexpr fk::Size res1_2(4, 2);
    constexpr fk::Size res3(8, 8);
    // Test 1
    constexpr uchar ptr1[] =
    { 128, 254, 129, 255, 128, 254, 129, 255,
      128, 254, 129, 255, 128, 254, 129, 255 }; // UYVY pixel data
    
    constexpr uchar3 ptr1Expected[] =
    { {254, 128, 129}, {255, 128, 129}, {254, 128, 129}, {255, 128, 129},
      {254, 128, 129}, {255, 128, 129}, {254, 128, 129}, {255, 128, 129} }; // Expected YUV values
    
    // Test 2
    constexpr uchar ptr2[] =
    { 0,  2, 1,  3,  4,  6,  5,  7,
      8, 10, 9, 11, 12, 14, 13, 15 }; // UYVY pixel data
    
    constexpr uchar3 ptr2Expected[] =
    { { 2, 0, 1}, { 3, 0, 1}, { 6,  4,  5}, { 7,  4,  5},
      {10, 8, 9}, {11, 8, 9}, {14, 12, 13}, {15, 12, 13}  }; // Expected YUV values

    // Test 3
    constexpr uchar ptr3[] =
    { 0,  2,  1,  3,  4,  6,  5,  7,  8, 10,  9, 11, 12, 14, 13, 15,
     16, 18, 17, 19, 20, 22, 21, 23, 24, 26, 25, 27, 28, 30, 29, 31,
      0,  2,  1,  3,  4,  6,  5,  7,  8, 10,  9, 11, 12, 14, 13, 15,
     16, 18, 17, 19, 20, 22, 21, 23, 24, 26, 25, 27, 28, 30, 29, 31,
      0,  2,  1,  3,  4,  6,  5,  7,  8, 10,  9, 11, 12, 14, 13, 15,
     16, 18, 17, 19, 20, 22, 21, 23, 24, 26, 25, 27, 28, 30, 29, 31,
      0,  2,  1,  3,  4,  6,  5,  7,  8, 10,  9, 11, 12, 14, 13, 15,
     16, 18, 17, 19, 20, 22, 21, 23, 24, 26, 25, 27, 28, 30, 29, 31 }; // UYVY pixel data

    constexpr uchar3 ptr3Expected[] =
    {{ 2,  0,  1}, { 3,  0,  1}, { 6,  4,  5}, { 7,  4,  5}, {10,  8,  9}, {11,  8,  9}, {14, 12, 13}, {15, 12, 13},
     {18, 16, 17}, {19, 16, 17}, {22, 20, 21}, {23, 20, 21}, {26, 24, 25}, {27, 24, 25}, {30, 28, 29}, {31, 28, 29},
     { 2,  0,  1}, { 3,  0,  1}, { 6,  4,  5}, { 7,  4,  5}, {10,  8,  9}, {11,  8,  9}, {14, 12, 13}, {15, 12, 13},
     {18, 16, 17}, {19, 16, 17}, {22, 20, 21}, {23, 20, 21}, {26, 24, 25}, {27, 24, 25}, {30, 28, 29}, {31, 28, 29},
     { 2,  0,  1}, { 3,  0,  1}, { 6,  4,  5}, { 7,  4,  5}, {10,  8,  9}, {11,  8,  9}, {14, 12, 13}, {15, 12, 13},
     {18, 16, 17}, {19, 16, 17}, {22, 20, 21}, {23, 20, 21}, {26, 24, 25}, {27, 24, 25}, {30, 28, 29}, {31, 28, 29},
     { 2,  0,  1}, { 3,  0,  1}, { 6,  4,  5}, { 7,  4,  5}, {10,  8,  9}, {11,  8,  9}, {14, 12, 13}, {15, 12, 13},
     {18, 16, 17}, {19, 16, 17}, {22, 20, 21}, {23, 20, 21}, {26, 24, 25}, {27, 24, 25}, {30, 28, 29}, {31, 28, 29}, };

    // Allocate Ptr's
    // We need to allocate a pitch that is at least width * 2 bytes for UYVY format
    std::array<fk::Image<fk::PixelFormat::UYVY>, 3> inputVals = {
      // Fix: make pitch to be respected when different than 0, even in host
      fk::Image<fk::PixelFormat::UYVY>(res1_2.width, res1_2.height),
      fk::Image<fk::PixelFormat::UYVY>(res1_2.width, res1_2.height),
      fk::Image<fk::PixelFormat::UYVY>(res3.width,   res3.height)
    };

    std::array<fk::Ptr<fk::ND::_2D, uchar3>, 3> expectedVals = {
      fk::Ptr<fk::ND::_2D, uchar3>(res1_2.width, res1_2.height, 0, fk::MemType::Host),
      fk::Ptr<fk::ND::_2D, uchar3>(res1_2.width, res1_2.height, 0, fk::MemType::Host),
      fk::Ptr<fk::ND::_2D, uchar3>(  res3.width,   res3.height, 0, fk::MemType::Host)
    };

    // Copy values test 1 and 2
    for (int y = 0; y < inputVals[0].getData().dims().height; ++y) {
        for (int x = 0; x < inputVals[0].getData().dims().width; ++x) {
            inputVals[0].getData().at(x, y) = ptr1[y * (res1_2.width * 2) + x];
            inputVals[1].getData().at(x, y) = ptr2[y * (res1_2.width * 2) + x];
        }
    }

    // Copy expected values for test 1 and 2
    for (int y = 0; y < res1_2.height; ++y) {
        for (int x = 0; x < res1_2.width; ++x) {
            expectedVals[0].at(x, y) = ptr1Expected[y * res1_2.width + x];
            expectedVals[1].at(x, y) = ptr2Expected[y * res1_2.width + x];
        }
    }

    // Copy values test 3
    for (int y = 0; y < inputVals[2].getData().dims().height; ++y) {
        for (int x = 0; x < inputVals[2].getData().dims().width; ++x) {
            inputVals[2].getData().at(x, y) = ptr3[y * (res3.width * 2) + x];
        }
    }

    // Copy expected values for test 3
    for (int y = 0; y < res3.height; ++y) {
        for (int x = 0; x < res3.width; ++x) {
            expectedVals[2].at(x, y) = ptr3Expected[y * res3.width + x];
        }
    }

    inputVals[0].upload(stream);
    inputVals[1].upload(stream);
    inputVals[2].upload(stream);

    TestCaseBuilder<ReadYUVTest>::addTest(testCases, stream, inputVals, expectedVals);
}

START_ADDING_TESTS
// Test UYVY pixel format traits
testUYVYPixelFormatTraits();

// Test IsEven function
testIsEvenFunction();

// Test RGB2Gray conversion
testRGB2GrayConversion();

// Test AddOpaqueAlpha
testAddOpaqueAlpha();

// Test ColorConversion operations
testColorConversionOperations();

// Test additional structs
testStaticAddAlpha();
testBGR2Gray();
testAddOpaqueAlphaStruct();
testDenormalizePixel();
testNormalizePixel();
testSaturateDenormalizePixel();
testNormalizeColorRangeDepth();

// Test ReadYUV operation
testReadYUV();
STOP_ADDING_TESTS

int launch() {
    RUN_ALL_TESTS
}