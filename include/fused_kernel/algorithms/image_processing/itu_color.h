/* Copyright 2023-2026 Oscar Amoros Huguet
   Copyright 2026 Grup Mediapro S.L.U

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_ITU_COLOR
#define FK_ITU_COLOR

#include <fused_kernel/core/data/static_matrix.h>

namespace fk {
    enum class ColorSpace { YUV420, YUV422, YUV444, RGB, RGBA };
    template <ColorSpace CS> struct CS_t {
        ColorSpace value{CS};
    };

    enum class ColorRange { Limited, Full };
    enum class ColorPrimitives { bt601, bt709, bt2020 };

    enum class ColorDepth { p8bit, p10bit, p12bit, fn8bit, fn10bit, fn12bit };
    template <ColorDepth CD>
    struct CD_t {
        ColorDepth value{CD};
    };

    using ColorDepthTypes =
        TypeList<CD_t<ColorDepth::p8bit>, CD_t<ColorDepth::p10bit>, CD_t<ColorDepth::p12bit>,
                 CD_t<ColorDepth::fn8bit>, CD_t<ColorDepth::fn10bit>, CD_t<ColorDepth::fn12bit>>;
    using ColorDepthPixelBaseTypes = TypeList<uchar, ushort, ushort, float, float, float>;
    using ColorDepthPixelTypes = TypeList<uchar3, ushort3, ushort3, float3, float3, float3>;
    template <ColorDepth CD>
    using ColorDepthPixelBaseType = EquivalentType_t<CD_t<CD>, ColorDepthTypes, ColorDepthPixelBaseTypes>;
    template <ColorDepth CD>
    using ColorDepthPixelType = EquivalentType_t<CD_t<CD>, ColorDepthTypes, ColorDepthPixelTypes>;

    // Taking into account the color depth, the pixel base type is uchar, ushort or float
    // ResolutionFactors therefore are used to compute the number of pixel base type elements on width and height
    struct ResolutionFactors {
        float width_f;
        float height_f;
    };

    enum class PixelFormat { NV12, NV21, YV12, P010, P016, P216, P210, Y216, Y210, Y416, UYVY };
    template <PixelFormat PF> struct PixelFormatTraits;
    template <> struct PixelFormatTraits<PixelFormat::NV12> {
        static constexpr ColorSpace space = ColorSpace::YUV420;
        static constexpr ColorDepth depth = ColorDepth::p8bit;
        static constexpr size_t cn = 3;
        static constexpr ResolutionFactors rf{1.f, 1.5f};
    };
    template <> struct PixelFormatTraits<PixelFormat::NV21> {
        static constexpr ColorSpace space = ColorSpace::YUV420;
        static constexpr ColorDepth depth = ColorDepth::p8bit;
        static constexpr size_t cn = 3;
        static constexpr ResolutionFactors rf{1.f, 1.5f};
    };
    template <> struct PixelFormatTraits<PixelFormat::YV12> {
        static constexpr ColorSpace space = ColorSpace::YUV420;
        static constexpr ColorDepth depth = ColorDepth::p8bit;
        static constexpr size_t cn = 3;
        static constexpr ResolutionFactors rf{1.f, 1.5f};
    };
    template <> struct PixelFormatTraits<PixelFormat::P010> {
        static constexpr ColorSpace space = ColorSpace::YUV420;
        static constexpr ColorDepth depth = ColorDepth::p10bit;
        static constexpr size_t cn = 3;
        static constexpr ResolutionFactors rf{1.f, 1.5f};
    };
    template <> struct PixelFormatTraits<PixelFormat::P210> {
        static constexpr ColorSpace space = ColorSpace::YUV422;
        static constexpr ColorDepth depth = ColorDepth::p10bit;
        static constexpr size_t cn = 3;
        static constexpr ResolutionFactors rf{1.f, 2.f};
    };
    template <> struct PixelFormatTraits<PixelFormat::Y210> {
        static constexpr ColorSpace space = ColorSpace::YUV422;
        static constexpr ColorDepth depth = ColorDepth::p10bit;
        static constexpr size_t cn = 3;
        static constexpr ResolutionFactors rf{2.f, 1.f};
    };
    template <> struct PixelFormatTraits<PixelFormat::Y416> {
        static constexpr ColorSpace space = ColorSpace::YUV444;
        static constexpr ColorDepth depth = ColorDepth::p12bit;
        static constexpr size_t cn = 4;
        static constexpr ResolutionFactors rf{4.f, 1.f};
    };
    template <> struct PixelFormatTraits<PixelFormat::UYVY> {
        static constexpr ColorSpace space = ColorSpace::YUV422;
        static constexpr ColorDepth depth = ColorDepth::p8bit;
        static constexpr size_t cn = 3;
        static constexpr ResolutionFactors rf{2.f, 1.f};
    };

    template <PixelFormat PF>
    using PackedPixelType = VectorType_t<ColorDepthPixelBaseType<static_cast<ColorDepth>(PixelFormatTraits<PF>::depth)>,
                                         PixelFormatTraits<PF>::cn>;

    template <PixelFormat PF, bool ALPHA>
    using YUVOutputPixelType =
        VectorType_t<ColorDepthPixelBaseType<PixelFormatTraits<PF>::depth>, ALPHA ? 4 : PixelFormatTraits<PF>::cn>;

    template <ColorDepth CD> constexpr ColorDepthPixelBaseType<CD> maxDepthValue{};
    template <> constexpr ColorDepthPixelBaseType<ColorDepth::p8bit> maxDepthValue<ColorDepth::p8bit>{255u};
    template <> constexpr ColorDepthPixelBaseType<ColorDepth::p10bit> maxDepthValue<ColorDepth::p10bit>{1023u};
    template <> constexpr ColorDepthPixelBaseType<ColorDepth::p12bit> maxDepthValue<ColorDepth::p12bit>{4095u};
    template <> constexpr ColorDepthPixelBaseType<ColorDepth::fn8bit> maxDepthValue<ColorDepth::fn8bit>{1.f};
    template <> constexpr ColorDepthPixelBaseType<ColorDepth::fn10bit> maxDepthValue<ColorDepth::fn10bit>{1.f};
    template <> constexpr ColorDepthPixelBaseType<ColorDepth::fn12bit> maxDepthValue<ColorDepth::fn12bit>{1.f};

    struct SubCoefficients {
        const float luma;
        const float chroma;
    };

    template <ColorDepth CD>
    constexpr SubCoefficients subCoefficients{};
    // Integer Offsets
    template <>
    constexpr SubCoefficients subCoefficients<ColorDepth::p8bit>{16.f, 128.f};
    template <>
    constexpr SubCoefficients subCoefficients<ColorDepth::p10bit>{64.f, 512.f};
    template <>
    constexpr SubCoefficients subCoefficients<ColorDepth::p12bit>{256.f, 2048.f};

    // Normalized Float Offsets (Fractional center of the integer ranges)
    template <>
    constexpr SubCoefficients subCoefficients<ColorDepth::fn8bit>{
        subCoefficients<ColorDepth::p8bit>.luma   / maxDepthValue<ColorDepth::p8bit>,
        subCoefficients<ColorDepth::p8bit>.chroma / maxDepthValue<ColorDepth::p8bit>};
    template <>
    constexpr SubCoefficients subCoefficients<ColorDepth::fn10bit>{
        subCoefficients<ColorDepth::p10bit>.luma   / maxDepthValue<ColorDepth::p10bit>,
        subCoefficients<ColorDepth::p10bit>.chroma / maxDepthValue<ColorDepth::p10bit>};
    template <>
    constexpr SubCoefficients subCoefficients<ColorDepth::fn12bit>{
        subCoefficients<ColorDepth::p12bit>.luma   / maxDepthValue<ColorDepth::p12bit>,
        subCoefficients<ColorDepth::p12bit>.chroma / maxDepthValue<ColorDepth::p12bit>};

    enum class ColorConversionDir { YCbCr2RGB, RGB2YCbCr };

    struct ITUWeights {
        float Kr{0.f};
        float Kb{0.f};
    };

    template <ColorPrimitives CP> constexpr ITUWeights iTUWeights{};
    // For all cases: Kg = 1.f - Kr - Kb
    template <> // Values extracted from https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.601-7-201103-I!!PDF-E.pdf
    constexpr ITUWeights iTUWeights<ColorPrimitives::bt601>{.Kr = 0.299f, .Kb = 0.114f};
    template <> // Values extracted from https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.709-6-201506-I!!PDF-E.pdf
    constexpr ITUWeights iTUWeights<ColorPrimitives::bt709>{.Kr = 0.2126f, .Kb = 0.0722f};
    template <> // Values extracted from https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.2020-2-201510-I!!PDF-E.pdf
    constexpr ITUWeights iTUWeights<ColorPrimitives::bt2020>{.Kr = 0.2627f, .Kb = 0.0593f};

    // --- Range Limits ---
    struct RangeLimits {
        const float limitY;
        const float limitC;
    };

    template <ColorDepth CD> constexpr RangeLimits rangeLimits{};

    // Integer Limits
    template <>
    constexpr RangeLimits rangeLimits<ColorDepth::p8bit>{219.f, 224.f};
    template <>
    constexpr RangeLimits rangeLimits<ColorDepth::p10bit>{876.f, 896.f};
    template <>
    constexpr RangeLimits rangeLimits<ColorDepth::p12bit>{3504.f, 3584.f};

    // Normalized Float Limits (Amplitude ratios exactly match their integer counterparts)
    template <>
    constexpr RangeLimits rangeLimits<ColorDepth::fn8bit>{219.f / maxDepthValue<ColorDepth::p8bit>,
                                                          224.f / maxDepthValue<ColorDepth::p8bit>};
    template <>
    constexpr RangeLimits rangeLimits<ColorDepth::fn10bit>{876.f / maxDepthValue<ColorDepth::p10bit>,
                                                           896.f / maxDepthValue<ColorDepth::p10bit>};
    template <>
    constexpr RangeLimits rangeLimits<ColorDepth::fn12bit>{3504.f / maxDepthValue<ColorDepth::p12bit>,
                                                           3584.f / maxDepthValue<ColorDepth::p12bit>};

    // The pure mathematical generator straight from the ITU equations
    template <ColorRange CR, ColorPrimitives CP, ColorConversionDir CDir, ColorDepth CD>
    constexpr M3x3Float computeMatrix() {
        constexpr float Kr = iTUWeights<CP>.Kr;
        constexpr float Kb = iTUWeights<CP>.Kb;
        constexpr float Kg = 1.0f - Kr - Kb;

        float m[3][3] = {};

        if constexpr (CDir == ColorConversionDir::RGB2YCbCr) {
            // Step 1: Full Range RGB2YCbCr
            m[0][0] = Kr;
            m[0][1] = Kg;
            m[0][2] = Kb;

            m[1][0] = -Kr / (2.0f * (1.0f - Kb));
            m[1][1] = -Kg / (2.0f * (1.0f - Kb));
            m[1][2] = 0.5f;

            m[2][0] = 0.5f;
            m[2][1] = -Kg / (2.0f * (1.0f - Kr));
            m[2][2] = -Kb / (2.0f * (1.0f - Kr));

            // Step 2: Limited Range RGB2YCbCr Scaling
            if constexpr (CR == ColorRange::Limited) {
                constexpr float maxVal = maxDepthValue<CD>;
                constexpr float limitY = rangeLimits<CD>.limitY;
                constexpr float limitC = rangeLimits<CD>.limitC;

                // This ratio dynamically adapts to integer or normalized float spans
                constexpr float scaleY = limitY / maxVal;
                constexpr float scaleC = limitC / maxVal;

                for (int i = 0; i < 3; ++i) {
                    m[0][i] *= scaleY;
                    m[1][i] *= scaleC;
                    m[2][i] *= scaleC;
                }
            }
        } else {
            // Step 1: Full Range YCbCr2RGB
            m[0][0] = 1.0f;
            m[0][1] = 0.0f;
            m[0][2] = 2.0f * (1.0f - Kr);

            m[1][0] = 1.0f;
            m[1][1] = -2.0f * Kb * (1.0f - Kb) / Kg;
            m[1][2] = -2.0f * Kr * (1.0f - Kr) / Kg;

            m[2][0] = 1.0f;
            m[2][1] = 2.0f * (1.0f - Kb);
            m[2][2] = 0.0f;

            // Step 2: Limited Range YCbCr2RGB Scaling
            if constexpr (CR == ColorRange::Limited) {
                constexpr float maxVal = maxDepthValue<CD>;
                constexpr float limitY = rangeLimits<CD>.limitY;
                constexpr float limitC = rangeLimits<CD>.limitC;

                // The exact inverse scale, matching the limits
                constexpr float scaleY = maxVal / limitY;
                constexpr float scaleC = maxVal / limitC;

                for (int i = 0; i < 3; ++i) {
                    m[i][0] *= scaleY;
                    m[i][1] *= scaleC;
                    m[i][2] *= scaleC;
                }
            }
        }

        return M3x3Float{{m[0][0], m[0][1], m[0][2]},
                         {m[1][0], m[1][1], m[1][2]},
                         {m[2][0], m[2][1], m[2][2]}};
    }

    // Getting the transformation matrices directly from computing them using
    // the official ITU recommendations.
    // To see the actual values, check the variable ccMatrixTest at
    // utests/algorithm/image_processing/utest_color_conversion.h
    template <ColorRange CR, ColorPrimitives CP, ColorConversionDir CCD, ColorDepth CD>
    constexpr M3x3Float ccMatrix = computeMatrix<CR, CP, CCD, CD>();
} // namespace fk

#endif