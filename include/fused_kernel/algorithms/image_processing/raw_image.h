/* Copyright 2023-2025 Oscar Amoros Huguet
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

#ifndef FK_RAW_IMAGE_H
#define FK_RAW_IMAGE_H

#include <fused_kernel/core/utils/type_lists.h>
#include <fused_kernel/core/data/rawptr.h>
#include <fused_kernel/core/data/vector_types.h>

namespace fk {
    enum class ColorSpace { YUV420, YUV422, YUV444, RGB, RGBA };
    template <ColorSpace CS>
    struct CS_t { ColorSpace value{ CS }; };

    enum class ColorRange { Limited, Full };
    enum class ColorPrimitives { bt601, bt709, bt2020 };

    enum class ColorDepth { p8bit, p10bit, p12bit, f24bit };
    template <ColorDepth CD>
    struct CD_t { ColorDepth value{ CD }; };
    using ColorDepthTypes = TypeList<CD_t<ColorDepth::p8bit>, CD_t<ColorDepth::p10bit>, CD_t<ColorDepth::p12bit>, CD_t<ColorDepth::f24bit>>;
    using ColorDepthPixelBaseTypes = TypeList<uchar, ushort, ushort, float>;
    using ColorDepthPixelTypes = TypeList<uchar3, ushort3, ushort3, float3>;
    template <ColorDepth CD>
    using ColorDepthPixelBaseType = EquivalentType_t<CD_t<CD>, ColorDepthTypes, ColorDepthPixelBaseTypes>;
    template <ColorDepth CD>
    using ColorDepthPixelType = EquivalentType_t<CD_t<CD>, ColorDepthTypes, ColorDepthPixelTypes>;

    // Taking into account the color depth, the pixel base type is uchar, ushort or float
    // ResolutionFactors therefore are use to compute the number of pixel base type elements on widht and height
    struct ResolutionFactors {
        float width_f;
        float height_f;
    };

    enum class PixelFormat { NV12, NV21, YV12, P010, P016, P216, P210, Y216, Y210, Y416, UYVY };
    template <PixelFormat PF>
    struct PixelFormatTraits;
    template <>
    struct PixelFormatTraits<PixelFormat::NV12> {
        static constexpr ColorSpace space = ColorSpace::YUV420;
        static constexpr ColorDepth depth = ColorDepth::p8bit;
        static constexpr size_t cn = 3;
        static constexpr ResolutionFactors rf{ 1.f, 1.5f };
    };
    template <>
    struct PixelFormatTraits<PixelFormat::NV21> {
        static constexpr ColorSpace space = ColorSpace::YUV420;
        static constexpr ColorDepth depth = ColorDepth::p8bit;
        static constexpr size_t cn = 3;
        static constexpr ResolutionFactors rf{ 1.f, 1.5f };
    };
    template <>
    struct PixelFormatTraits<PixelFormat::YV12> {
        static constexpr ColorSpace space = ColorSpace::YUV420;
        static constexpr ColorDepth depth = ColorDepth::p8bit;
        static constexpr size_t cn = 3;
        static constexpr ResolutionFactors rf{ 1.f, 1.5f };
    };
    template <>
    struct PixelFormatTraits<PixelFormat::P010> {
        static constexpr ColorSpace space = ColorSpace::YUV420;
        static constexpr ColorDepth depth = ColorDepth::p10bit;
        static constexpr size_t cn = 3;
        static constexpr ResolutionFactors rf{ 1.f, 1.5f };
    };
    template <>
    struct PixelFormatTraits<PixelFormat::P210> {
        static constexpr ColorSpace space = ColorSpace::YUV422;
        static constexpr ColorDepth depth = ColorDepth::p10bit;
        static constexpr size_t cn = 3;
        static constexpr ResolutionFactors rf{ 1.f, 2.f };
    };
    template <>
    struct PixelFormatTraits<PixelFormat::Y210> {
        static constexpr ColorSpace space = ColorSpace::YUV422;
        static constexpr ColorDepth depth = ColorDepth::p10bit;
        static constexpr size_t cn = 3;
        static constexpr ResolutionFactors rf{ 2.f, 1.f };
    };
    template <>
    struct PixelFormatTraits<PixelFormat::Y416> {
        static constexpr ColorSpace space = ColorSpace::YUV444;
        static constexpr ColorDepth depth = ColorDepth::p12bit;
        static constexpr size_t cn = 4;
        static constexpr ResolutionFactors rf{ 4.f, 1.f };
    };
    template <>
    struct PixelFormatTraits<PixelFormat::UYVY> {
        static constexpr ColorSpace space = ColorSpace::YUV422;
        static constexpr ColorDepth depth = ColorDepth::p8bit;
        static constexpr size_t cn = 3;
        static constexpr ResolutionFactors rf{ 2.f, 1.f };
    };

    template <PixelFormat PF>
    struct RawImage {
        using BaseType = ColorDepthPixelBaseType<PixelFormatTraits<PF>::depth>;
        RawPtr<ND::_2D, BaseType> data; // Raw image data
        uint width;
        uint height;
    };

} // namespace fk

#endif // FK_RAW_IMAGE_H