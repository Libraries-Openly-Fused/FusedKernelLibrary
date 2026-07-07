/* Copyright 2025-2026 Oscar Amoros Huguet
   Copyright 2025-2026 Grup Mediapro S.L.U

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
#include <fused_kernel/algorithms/basic_ops/logical.h>
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

// Test ColorConversion for the four specializations that use FusedOperation chains:
// BGR2GRAY, BGRA2GRAY, BGR2RGBA, BGRA2RGB
void testColorConversionAffectedCodes() {
    // Test COLOR_BGR2GRAY: reorder(2,1,0) then RGB2Gray
    // Input BGR {B, G, R} -> reorder -> {R, G, B} -> gray
    constexpr std::array<uchar3, 2> bgrInputVals{
        uchar3{100, 150, 200},  // B=100, G=150, R=200 -> gray = round(0.299*200 + 0.587*150 + 0.114*100) = 159
        uchar3{50, 75, 125}     // B=50,  G=75,  R=125 -> gray = round(0.299*125 + 0.587*75  + 0.114*50)  = 87
    };
    constexpr std::array<uchar, 2> bgrGrayExpected{159, 87};
    TestCaseBuilder<fk::ColorConversion<fk::ColorConversionCodes::COLOR_BGR2GRAY, uchar3, uchar>>::addTest(
        testCases, bgrInputVals, bgrGrayExpected);

    // Test COLOR_BGRA2GRAY: reorder(2,1,0,3) then RGB2Gray (alpha is discarded by RGB2Gray)
    constexpr std::array<uchar4, 2> bgraInputVals{
        uchar4{100, 150, 200, 255},  // B=100, G=150, R=200, A=255 -> gray = 159
        uchar4{50, 75, 125, 200}     // B=50,  G=75,  R=125, A=200 -> gray = 87
    };
    constexpr std::array<uchar, 2> bgraGrayExpected{159, 87};
    TestCaseBuilder<fk::ColorConversion<fk::ColorConversionCodes::COLOR_BGRA2GRAY, uchar4, uchar>>::addTest(
        testCases, bgraInputVals, bgraGrayExpected);

    // Test COLOR_BGR2RGBA: reorder(2,1,0) then AddOpaqueAlpha
    // Input BGR {B, G, R} -> reorder -> {R, G, B} -> {R, G, B, 255}
    constexpr std::array<uchar3, 2> bgr2rgbaInputVals{
        uchar3{100, 150, 200},
        uchar3{50, 75, 125}
    };
    constexpr std::array<uchar4, 2> bgr2rgbaExpected{
        uchar4{200, 150, 100, 255},
        uchar4{125, 75, 50, 255}
    };
    TestCaseBuilder<fk::ColorConversion<fk::ColorConversionCodes::COLOR_BGR2RGBA, uchar3, uchar4>>::addTest(
        testCases, bgr2rgbaInputVals, bgr2rgbaExpected);

    // Test COLOR_BGRA2RGB: reorder(2,1,0,3) then Discard (drop alpha)
    // Input BGRA {B, G, R, A} -> reorder -> {R, G, B, A} -> {R, G, B}
    constexpr std::array<uchar4, 2> bgra2rgbInputVals{
        uchar4{100, 150, 200, 255},
        uchar4{50, 75, 125, 200}
    };
    constexpr std::array<uchar3, 2> bgra2rgbExpected{
        uchar3{200, 150, 100},
        uchar3{125, 75, 50}
    };
    TestCaseBuilder<fk::ColorConversion<fk::ColorConversionCodes::COLOR_BGRA2RGB, uchar4, uchar3>>::addTest(
        testCases, bgra2rgbInputVals, bgra2rgbExpected);
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

// Tests for the ColorConversion aliases that expand to FusedOperation
// (regression test for the raw-Op vs IOp template arguments bug):
// COLOR_BGR2GRAY, COLOR_BGRA2GRAY, COLOR_BGR2RGBA, COLOR_BGRA2RGB
void testFusedColorConversionAliases() {
    // COLOR_BGR2GRAY: reorder(2,1,0) then RGB2Gray
    // input is BGR -> luma = 0.299*R(z) + 0.587*G(y) + 0.114*B(x)
    std::array<uchar3, 2> inBGR = {
        uchar3{50, 100, 150},
        uchar3{75, 125, 200}
    };
    std::array<uchar, 2> expGray = {
        static_cast<uchar>(std::nearbyint(150 * 0.299f + 100 * 0.587f + 50 * 0.114f)),
        static_cast<uchar>(std::nearbyint(200 * 0.299f + 125 * 0.587f + 75 * 0.114f))
    };
    TestCaseBuilder<fk::ColorConversion<fk::ColorConversionCodes::COLOR_BGR2GRAY, uchar3, uchar>>::
        addTest(testCases, inBGR, expGray);

    // COLOR_BGRA2GRAY: reorder(2,1,0,3) then RGB2Gray (alpha discarded by formula)
    std::array<uchar4, 2> inBGRA = {
        uchar4{50, 100, 150, 255},
        uchar4{75, 125, 200, 128}
    };
    TestCaseBuilder<fk::ColorConversion<fk::ColorConversionCodes::COLOR_BGRA2GRAY, uchar4, uchar>>::
        addTest(testCases, inBGRA, expGray);

    // COLOR_BGR2RGBA: reorder(2,1,0) then AddOpaqueAlpha
    std::array<uchar4, 2> expRGBA = {
        uchar4{150, 100, 50, 255},
        uchar4{200, 125, 75, 255}
    };
    TestCaseBuilder<fk::ColorConversion<fk::ColorConversionCodes::COLOR_BGR2RGBA, uchar3, uchar4>>::
        addTest(testCases, inBGR, expRGBA);

    // COLOR_BGRA2RGB: reorder(2,1,0,3) then Discard -> 3 channels
    std::array<uchar3, 2> expRGB = {
        uchar3{150, 100, 50},
        uchar3{200, 125, 75}
    };
    TestCaseBuilder<fk::ColorConversion<fk::ColorConversionCodes::COLOR_BGRA2RGB, uchar4, uchar3>>::
        addTest(testCases, inBGRA, expRGB);
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

namespace fk {
template <ColorRange CR, ColorPrimitives CP, ColorConversionDir CCD>
constexpr M3x3Float ccMatrixTest{};

// =========================================================================
// BT.601
// =========================================================================
template <>
constexpr M3x3Float ccMatrixTest<ColorRange::Full, ColorPrimitives::bt601, ColorConversionDir::YCbCr2RGB>{
    {  1.000000000f,  0.000000000f,  1.401999950f },
    {  1.000000000f, -0.344136298f, -0.714136302f },
    {  1.000000000f,  1.771999955f,  0.000000000f }
};

template <>
constexpr M3x3Float ccMatrixTest<ColorRange::Full, ColorPrimitives::bt601, ColorConversionDir::RGB2YCbCr>{
    {  0.298999995f,  0.586999953f,  0.114000000f },
    { -0.168735892f, -0.331264079f,  0.500000000f },
    {  0.500000000f, -0.418687582f, -0.081312411f }
};

template <>
constexpr M3x3Float ccMatrixTest<ColorRange::Limited, ColorPrimitives::bt601, ColorConversionDir::YCbCr2RGB>{
    {  1.164383531f,  0.000000000f,  1.596026659f },
    {  1.164383531f, -0.391762286f, -0.812967658f },
    {  1.164383531f,  2.017231941f,  0.000000000f }
};

template <>
constexpr M3x3Float ccMatrixTest<ColorRange::Limited, ColorPrimitives::bt601, ColorConversionDir::RGB2YCbCr>{
    {  0.256788224f,  0.504129350f,  0.097905882f },
    { -0.148222908f, -0.290992767f,  0.439215690f },
    {  0.439215690f, -0.367788315f, -0.071427375f }
};

// =========================================================================
// BT.709
// =========================================================================
template <>
constexpr M3x3Float ccMatrixTest<ColorRange::Full, ColorPrimitives::bt709, ColorConversionDir::YCbCr2RGB>{
    {  1.000000000f,  0.000000000f,  1.574800014f },
    {  1.000000000f, -0.187324256f, -0.468124270f },
    {  1.000000000f,  1.855599999f,  0.000000000f }
};

template <>
constexpr M3x3Float ccMatrixTest<ColorRange::Full, ColorPrimitives::bt709, ColorConversionDir::RGB2YCbCr>{
    {  0.212599993f,  0.715200007f,  0.072200000f },
    { -0.114572100f, -0.385427892f,  0.500000000f },
    {  0.500000000f, -0.454152912f, -0.045847092f }
};

template <>
constexpr M3x3Float ccMatrixTest<ColorRange::Limited, ColorPrimitives::bt709, ColorConversionDir::YCbCr2RGB>{
    {  1.164383531f,  0.000000000f,  1.792741060f },
    {  1.164383531f, -0.213248581f, -0.532909274f },
    {  1.164383531f,  2.112401724f,  0.000000000f }
};

template <>
constexpr M3x3Float ccMatrixTest<ColorRange::Limited, ColorPrimitives::bt709, ColorConversionDir::RGB2YCbCr>{
    {  0.182585880f,  0.614230573f,  0.062007058f },
    { -0.100643732f, -0.338571966f,  0.439215690f },
    {  0.439215690f, -0.398942173f, -0.040273525f }
};

// =========================================================================
// BT.2020
// =========================================================================
template <>
constexpr M3x3Float ccMatrixTest<ColorRange::Full, ColorPrimitives::bt2020, ColorConversionDir::YCbCr2RGB>{
    {  1.000000000f,  0.000000000f,  1.474600077f },
    {  1.000000000f, -0.164553121f, -0.571353137f },
    {  1.000000000f,  1.881399989f,  0.000000000f }
};

template <>
constexpr M3x3Float ccMatrixTest<ColorRange::Full, ColorPrimitives::bt2020, ColorConversionDir::RGB2YCbCr>{
    {  0.262699991f,  0.678000033f,  0.059300002f },
    { -0.139630064f, -0.360369951f,  0.500000000f },
    {  0.500000000f, -0.459785700f, -0.040214293f }
};

template <>
constexpr M3x3Float ccMatrixTest<ColorRange::Limited, ColorPrimitives::bt2020, ColorConversionDir::YCbCr2RGB>{
    {  1.164383531f,  0.000000000f,  1.678674102f },
    {  1.164383531f, -0.187326089f, -0.650424302f },
    {  1.164383531f,  2.141772270f,  0.000000000f }
};

template <>
constexpr M3x3Float ccMatrixTest<ColorRange::Limited, ColorPrimitives::bt2020, ColorConversionDir::RGB2YCbCr>{
    {  0.225612938f,  0.582282364f,  0.050928239f },
    { -0.122655429f, -0.316560268f,  0.439215690f },
    {  0.439215690f, -0.403890193f, -0.035325497f }
};

constexpr bool equalM3x3(const M3x3Float& first, const M3x3Float& second) {
    return (first.x == second.x) && (first.y == second.y) && (first.z == second.z);
}

template <ColorPrimitives CP>
constexpr bool testTransformationMatrixValues_helper() {
    using CR_t = ColorRange;
    using CCD_t = ColorConversionDir;

    static_assert(equalM3x3(ccMatrixTest<CR_t::Full, CP, CCD_t::RGB2YCbCr>, ccMatrix<CR_t::Full, CP, CCD_t::RGB2YCbCr>),
        "Something wrong in color conversion matrix Full and RGB2YCbCr");
    static_assert(equalM3x3(ccMatrixTest<CR_t::Full, CP, CCD_t::YCbCr2RGB>, ccMatrix<CR_t::Full, CP, CCD_t::YCbCr2RGB>),
        "Something wrong in color conversion matrix Full and YCbCr2RGB");
    static_assert(equalM3x3(ccMatrixTest<CR_t::Limited, CP, CCD_t::RGB2YCbCr>, ccMatrix<CR_t::Limited, CP, CCD_t::RGB2YCbCr>),
        "Something wrong in color conversion matrix Limited and RGB2YCbCr");
    static_assert(equalM3x3(ccMatrixTest<CR_t::Limited, CP, CCD_t::YCbCr2RGB>, ccMatrix<CR_t::Limited, CP, CCD_t::YCbCr2RGB>),
        "Something wrong in color conversion matrix Limited and YCbCr2RGB");

    return true;
}
} // namespace fk

constexpr bool testTransformationMatrixValues() {
    using namespace fk;
    constexpr bool resbt601 = testTransformationMatrixValues_helper<ColorPrimitives::bt601>();
    constexpr bool resbt709 = testTransformationMatrixValues_helper<ColorPrimitives::bt709>();
    constexpr bool resbt2020 = testTransformationMatrixValues_helper<ColorPrimitives::bt2020>();

    return and_v<resbt601, resbt709, resbt2020>;
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

// Test ColorConversion for the FusedOperation-based specializations
testColorConversionAffectedCodes();

// Test additional structs
testStaticAddAlpha();
testBGR2Gray();
testFusedColorConversionAliases();
testAddOpaqueAlphaStruct();
testDenormalizePixel();
testNormalizePixel();
testSaturateDenormalizePixel();
testNormalizeColorRangeDepth();

// Test ReadYUV operation
testReadYUV();
testTransformationMatrixValues();
STOP_ADDING_TESTS

int launch() {
    RUN_ALL_TESTS
}