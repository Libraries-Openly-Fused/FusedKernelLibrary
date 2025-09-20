/* Copyright 2023-2025 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_COLOR_CONVERSION
#define FK_COLOR_CONVERSION

#include <fused_kernel/core/execution_model/operation_model/operation_model.h>
#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/algorithms/basic_ops/algebraic.h>
#include <fused_kernel/algorithms/image_processing/saturate.h>
//#include <fused_kernel/algorithms/basic_ops/cast.h>
#include <fused_kernel/algorithms/image_processing/raw_image.h>
#include <fused_kernel/algorithms/basic_ops/vector_ops.h>

namespace fk {
    template <typename I>
    using VOneMore = VectorType_t<VBase<I>, (cn<I> + 1)>;

    template <typename I, VBase<I> alpha>
    struct StaticAddAlpha {
    private:
        using SelfType = StaticAddAlpha<I, alpha>;
    public:
        FK_STATIC_STRUCT(StaticAddAlpha, SelfType)
        using Parent = UnaryOperation<I, VOneMore<I>, StaticAddAlpha<I, alpha>>;
        DECLARE_UNARY_PARENT
        FK_DEVICE_FUSE OutputType exec(const InputType& input) {
            return AddLast<InputType, OutputType>::exec(input, { alpha });
        }
    };

    enum class GrayFormula { CCIR_601 };

    template <typename I, typename O = VBase<I>, GrayFormula GF = GrayFormula::CCIR_601>
    struct RGB2Gray {};

    template <typename I, typename O>
    struct RGB2Gray<I, O, GrayFormula::CCIR_601> {
    private:
        using SelfType = RGB2Gray<I, O, GrayFormula::CCIR_601>;
    public:
        FK_STATIC_STRUCT(RGB2Gray, SelfType)
        using Parent = UnaryOperation<I, O, RGB2Gray<I, O, GrayFormula::CCIR_601>>;
        DECLARE_UNARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            // 0.299*R + 0.587*G + 0.114*B
            if constexpr (std::is_unsigned_v<OutputType>) {
#ifdef __CUDA_ARCH__
                return __float2uint_rn(compute_luminance(input));
#else
                return static_cast<OutputType>(std::nearbyint(compute_luminance(input)));
#endif
            } else if constexpr (std::is_signed_v<OutputType>) {
#ifdef __CUDA_ARCH__
                return __float2int_rn(compute_luminance(input));
#else
                return static_cast<OutputType>(std::nearbyint(compute_luminance(input)));
#endif
            } else if constexpr (std::is_floating_point_v<OutputType>) {
                return compute_luminance(input);
            }
        }
    private:
        FK_HOST_DEVICE_FUSE float compute_luminance(const InputType& input) {
            return (input.x * 0.299f) + (input.y * 0.587f) + (input.z * 0.114f);
        }
    };

    

    template <PixelFormat PF>
    using PackedPixelType = VectorType_t<ColorDepthPixelBaseType<static_cast<ColorDepth>(PixelFormatTraits<PF>::depth)>, PixelFormatTraits<PF>::cn>;

    template <PixelFormat PF, bool ALPHA>
    using YUVOutputPixelType = VectorType_t<ColorDepthPixelBaseType<static_cast<ColorDepth>(PixelFormatTraits<PF>::depth)>, ALPHA ? 4 : PixelFormatTraits<PF>::cn>;

    struct SubCoefficients {
        const float luma;
        const float chroma;
    };

    template <ColorDepth CD>
    constexpr SubCoefficients subCoefficients{};
    template <> constexpr SubCoefficients subCoefficients<ColorDepth::p8bit>{ 16.f, 128.f };
    template <> constexpr SubCoefficients subCoefficients<ColorDepth::p10bit>{ 64.f, 512.f };
    template <> constexpr SubCoefficients subCoefficients<ColorDepth::p12bit>{ 64.f, 2048.f };

    template <ColorDepth CD>
    constexpr ColorDepthPixelBaseType<CD> maxDepthValue{};
    template <> constexpr ColorDepthPixelBaseType<ColorDepth::p8bit>  maxDepthValue<ColorDepth::p8bit> { 255u };
    template <> constexpr ColorDepthPixelBaseType<ColorDepth::p10bit> maxDepthValue<ColorDepth::p10bit> { 1023u };
    template <> constexpr ColorDepthPixelBaseType<ColorDepth::p12bit> maxDepthValue<ColorDepth::p12bit> { 4095u };
    template <> constexpr ColorDepthPixelBaseType<ColorDepth::f24bit> maxDepthValue<ColorDepth::f24bit> { 1.f };

    template <typename I, ColorDepth CD>
    struct AddOpaqueAlpha {
    private:
        using SelfType = AddOpaqueAlpha<I, CD>;
    public:
        FK_STATIC_STRUCT(AddOpaqueAlpha, SelfType)
        using Parent = UnaryOperation<I, VOneMore<I>, AddOpaqueAlpha<I, CD>>;
        DECLARE_UNARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            constexpr auto alpha = maxDepthValue<CD>;
            return AddLast<InputType, OutputType>::exec(input, { alpha });
        }
    };

    template <typename T, ColorDepth CD>
    struct SaturateDepth {
    private:
        using SelfType = SaturateDepth<T, CD>;
    public:
        FK_STATIC_STRUCT(SaturateDepth, SelfType)
        using Parent = UnaryOperation<T, T, SaturateDepth<T, CD>>;
        DECLARE_UNARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            return Saturate<float>::exec(input, { { 0.f, static_cast<float>(maxDepthValue<CD>) } });
        }
    };

    enum class ColorConversionDir { YCbCr2RGB, RGB2YCbCr };

    template <ColorRange CR, ColorPrimitives CP, ColorConversionDir CCD>
    constexpr M3x3Float ccMatrix{};
    // Source: https://en.wikipedia.org/wiki/YCbCr
    template <> constexpr M3x3Float ccMatrix<ColorRange::Full, ColorPrimitives::bt601, ColorConversionDir::YCbCr2RGB>{
        { 1.164383562f,           0.f,       1.596026786f  },
        { 1.164383562f,  -0.39176229f,       -0.812967647f },
        { 1.164383562f,  2.017232143f,       0.f           }};

    // Source: https://en.wikipedia.org/wiki/YCbCr
    template <> constexpr M3x3Float ccMatrix<ColorRange::Full, ColorPrimitives::bt709, ColorConversionDir::YCbCr2RGB>{
        { 1.f,               0.f,            1.5748f },
        { 1.f,          -0.1873f,           -0.4681f },
        { 1.f,           1.8556f,                0.f }};

    // To be verified
    template <> constexpr M3x3Float ccMatrix<ColorRange::Limited, ColorPrimitives::bt709, ColorConversionDir::YCbCr2RGB>{
        { 1.f,               0.f,            1.402f },
        { 1.f,         -0.34414f,         -0.71414f },
        { 1.f,            1.772f,               0.f }};
        /*{  1.f,              0.f,            1.4746f },
        { 1.f,          -0.1646f,           -0.5713f },
        { 1.f,           1.8814f,            0.f     }};*/

    // Source: https://en.wikipedia.org/wiki/YCbCr
    template <> constexpr M3x3Float ccMatrix<ColorRange::Full, ColorPrimitives::bt709, ColorConversionDir::RGB2YCbCr>{
        {  0.2126f, 0.7152f, 0.0722f           },
        { -0.1146f,      -0.3854f,            0.5f },
        { 0.5f,         -0.4542f,           -0.0458f }};

    // Source: https://en.wikipedia.org/wiki/YCbCr
    template <> constexpr M3x3Float ccMatrix<ColorRange::Full, ColorPrimitives::bt2020, ColorConversionDir::YCbCr2RGB>{
        {  1.f, 0.f, 1.4746f           },
        { 1.f,          -0.16455312684366f, -0.57135312684366f },
        { 1.f,           1.8814f,            0.f }};

    // Computed from ccMatrix<ColorRange::Full, ColorPrimitives::bt2020, ColorConversionDir::YCbCr2RGB>
    template <> constexpr M3x3Float ccMatrix<ColorRange::Full, ColorPrimitives::bt2020, ColorConversionDir::RGB2YCbCr>{
        { -0.73792134831461f, 1.90449438202248f, -0.16657303370787f },
        { 0.39221927730127f, -1.01227510472121f,  0.62005582741994f },
        { 1.17857137414527f, -1.29153287808387f,  0.11296150393861f }};

    template <ColorDepth CD> constexpr ColorDepthPixelBaseType<CD> shiftFactor{};
    template <> constexpr ColorDepthPixelBaseType<ColorDepth::p8bit>  shiftFactor<ColorDepth::p8bit>{ 0u };
    template <> constexpr ColorDepthPixelBaseType<ColorDepth::p10bit> shiftFactor<ColorDepth::p10bit>{ 6u };
    template <> constexpr ColorDepthPixelBaseType<ColorDepth::p12bit> shiftFactor<ColorDepth::p12bit>{ 4u };

    template <ColorDepth CD>
    constexpr float floatShiftFactor{};
    template <> constexpr float floatShiftFactor<ColorDepth::p8bit>{ 1.f };
    template <> constexpr float floatShiftFactor<ColorDepth::p10bit>{ 64.f };
    template <> constexpr float floatShiftFactor<ColorDepth::p12bit>{ 16.f };


    template <typename O, ColorDepth CD>
    struct DenormalizePixel {
    private:
        using SelfType = DenormalizePixel<O, CD>;
    public:
        FK_STATIC_STRUCT(DenormalizePixel, SelfType)
        using Parent = UnaryOperation<VectorType_t<float, cn<O>>, O, DenormalizePixel<O, CD>>;
        DECLARE_UNARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            constexpr auto maxDepth = maxDepthValue<CD>;
            return cxp::cast<OutputType>::f(input * maxDepth);
        }
    };

    template <typename I, ColorDepth CD>
    struct NormalizePixel {
    private:
        using SelfType = NormalizePixel<I, CD>;
    public:
        FK_STATIC_STRUCT(NormalizePixel, SelfType)
        using Parent = UnaryOperation<I, VectorType_t<float, cn<I>>, NormalizePixel<I, CD>>;
        DECLARE_UNARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            return input / static_cast<float>(maxDepthValue<CD>);
        }
    };

    template <typename I, typename O, ColorDepth CD>
    struct SaturateDenormalizePixel {
    private:
        using SelfType = SaturateDenormalizePixel<I, O, CD>;
    public:
        FK_STATIC_STRUCT(SaturateDenormalizePixel, SelfType)
        using Parent = UnaryOperation<I, O, SaturateDenormalizePixel<I, O, CD>>;
        DECLARE_UNARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            static_assert(std::is_same_v<VBase<I>, float>, "SaturateDenormalizePixel only works with float base types.");
            const InputType saturatedFloat = SaturateFloat<InputType>::exec(input);
            return DenormalizePixel<OutputType, CD>::exec(saturatedFloat);
        }
    };

    template <typename T, ColorDepth CD>
    struct NormalizeColorRangeDepth {
    private:
        using SelfType = NormalizeColorRangeDepth<T, CD>;
    public:
        FK_STATIC_STRUCT(NormalizeColorRangeDepth, SelfType)
        using Parent = UnaryOperation<T, T, NormalizeColorRangeDepth<T, CD>>;
        DECLARE_UNARY_PARENT
        using Base = typename VectorTraits<T>::base;
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            static_assert(std::is_floating_point_v<VBase<T>>, "NormalizeColorRangeDepth only works for floating point values");
            // The nvcc compiler will only be able to use the global constexpr floatShiftFactor<CD> variable if it is stored in 
            // a local variable.
            // By storing it into a local variable, you are forcing the value to exist in private memory, so that each thread has
            // a copy of the value on registers.
            // In a later stage, since the variable is constexpr, the compiler will be able to inline the value in the
            // multiplication instruction, and won't be stored in private memory.
            constexpr auto shiftFactor = floatShiftFactor<CD>;
            return input * shiftFactor;
        }
    };

    template <PixelFormat PF, ColorRange CR, ColorPrimitives CP, bool ALPHA, typename ReturnType = YUVOutputPixelType<PF, ALPHA>>
    struct ConvertYUVToRGB {
    private:
        using SelfType = ConvertYUVToRGB<PF, CR, CP, ALPHA, ReturnType>;
    public:
        FK_STATIC_STRUCT(ConvertYUVToRGB, SelfType)
        static constexpr ColorDepth CD = (ColorDepth)PixelFormatTraits<PF>::depth;
        using Parent = UnaryOperation<PackedPixelType<PF>, ReturnType, ConvertYUVToRGB<PF, CR, CP, ALPHA, ReturnType>>;
        DECLARE_UNARY_PARENT

        private:
        // Y     -> input.x
        // Cb(U) -> input.y
        // Cr(V) -> input.z
        FK_HOST_DEVICE_FUSE float3 computeRGB(const InputType& pixel) {
            constexpr M3x3Float coefficients = ccMatrix<CR, CP, ColorConversionDir::YCbCr2RGB>;
            constexpr float CSub = subCoefficients<CD>.chroma;
            if constexpr (CP == ColorPrimitives::bt601) {
                constexpr float YSub = subCoefficients<CD>.luma;
                return MxVFloat3<UnaryType>::exec({ make_<float3>(pixel.x - YSub, pixel.y - CSub, pixel.z - CSub), coefficients });
            } else {
                return MxVFloat3<UnaryType>::exec({ make_<float3>(pixel.x, pixel.y - CSub, pixel.z - CSub), coefficients });
            }
        }

        FK_HOST_DEVICE_FUSE OutputType computePixel(const InputType& pixel) {
            const float3 pixelRGBFloat = computeRGB(pixel);
            if constexpr (std::is_same_v<VBase<OutputType>, float>) {
                if constexpr (ALPHA) {
                    return { pixelRGBFloat.x, pixelRGBFloat.y, pixelRGBFloat.z, (float)maxDepthValue<CD> };
                } else {
                    return pixelRGBFloat;
                }
            } else {
                const InputType pixelRGB = SaturateCast<float3, InputType>::exec(pixelRGBFloat);
                if constexpr (ALPHA) {
                    return { pixelRGB.x, pixelRGB.y, pixelRGB.z, maxDepthValue<CD> };
                } else {
                    return pixelRGB;
                }
            }

        }

        public:
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            // Pixel data shifted to the right to it's color depth numerical range
            const InputType shiftedPixel = input >> shiftFactor<CD>;

            // Using color depth numerical range to compute the RGB pixel
            const OutputType computedPixel = computePixel(shiftedPixel);
            if constexpr (std::is_same_v<VBase<OutputType>, float>) {
                // Moving back the pixel channels to data type numerical range, either 8bit or 16bit
                return NormalizeColorRangeDepth<OutputType, CD>::exec(computedPixel);
            } else {
                // Moving back the pixel channels to data type numerical range, either 8bit or 16bit
                return computedPixel << shiftFactor<CD>;
            }
        }
    };

    template <PixelFormat PF>
    class Image;

    template <PixelFormat PF>
    struct ReadYUV {
    private:
        using SelfType = ReadYUV<PF>;
    public:
        FK_STATIC_STRUCT(ReadYUV, SelfType)
        using PixelBaseType = ColorDepthPixelBaseType<PixelFormatTraits<PF>::depth>;
        using Parent = ReadOperation<PixelBaseType,
                                     RawImage<PF>,
                                     ColorDepthPixelType<(ColorDepth)PixelFormatTraits<PF>::depth>,
                                     TF::DISABLED,
                                     ReadYUV<PF>>;
        DECLARE_READ_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const ParamsType& params) {
            const auto rawPtr = params.data;
            if constexpr (PF == PixelFormat::NV12 || PF == PixelFormat::P010 ||
                          PF == PixelFormat::P016 || PF == PixelFormat::P210 ||
                          PF == PixelFormat::P216) {
                // Planar luma
                const PixelBaseType Y = *PtrAccessor<ND::_2D>::cr_point(thread, rawPtr);

                // Packed chroma
                const PtrDims<ND::_2D> dims = rawPtr.dims;
                using VectorType2 = VectorType_t<PixelBaseType, 2>;
                const RawPtr<ND::_2D, VectorType2> chromaPlane{
                    reinterpret_cast<VectorType2*>(reinterpret_cast<uchar*>(rawPtr.data) + dims.pitch * dims.height),
                    { dims.width >> 1, dims.height >> 1, dims.pitch }
                };
                const ColorSpace CS = static_cast<ColorSpace>(PixelFormatTraits<PF>::space);
                const VectorType2 UV =
                    *PtrAccessor<ND::_2D>::cr_point({ thread.x >> 1, CS == ColorSpace::YUV420 ? thread.y >> 1 : thread.y, thread.z }, chromaPlane);

                return { Y, UV.x, UV.y };
            } else if constexpr (PF == PixelFormat::NV21) {
                // Planar luma
                const uchar Y = *PtrAccessor<ND::_2D>::cr_point(thread, rawPtr);

                // Packed chroma
                const PtrDims<ND::_2D> dims = rawPtr.dims;
                const RawPtr<ND::_2D, uchar2> chromaPlane{
                    reinterpret_cast<uchar2*>(reinterpret_cast<uchar*>(rawPtr.data) + dims.pitch * dims.height),
                                              { dims.width >> 1, dims.height >> 1, dims.pitch }
                };
                const uchar2 VU = *PtrAccessor<ND::_2D>::cr_point({ thread.x >> 1, thread.y >> 1, thread.z }, chromaPlane);

                return { Y, VU.y, VU.x };
            } else if constexpr (PF == PixelFormat::Y216 || PF == PixelFormat::Y210 || PF == PixelFormat::UYVY) {
                const PtrDims<ND::_2D> dims = rawPtr.dims;
                using VectorType4 = VectorType_t<PixelBaseType, 4>;
                const RawPtr<ND::_2D, VectorType4> image{ reinterpret_cast<VectorType4*>(rawPtr.data), {dims.width >> 2, dims.height, dims.pitch} };
                const VectorType4 pixel = *PtrAccessor<ND::_2D>::cr_point({ thread.x >> 1, thread.y, thread.z }, image);

                const bool isEvenThread = cxp::is_even::f(thread.x);
                if constexpr (PF == PixelFormat::UYVY) {
                    return { isEvenThread ? pixel.y : pixel.w, pixel.x, pixel.z };
                } else {
                    return { isEvenThread ? pixel.x : pixel.z, pixel.y, pixel.w };
                }
            } else if constexpr (PF == PixelFormat::Y416) {
                // AVYU
                // We use ushort as the type, to be compatible with the rest of the cases
                const RawPtr<ND::_2D, ushort4> readImage{ reinterpret_cast<ushort4*>(rawPtr.data), rawPtr.dims };
                const ushort4 pixel = *PtrAccessor<ND::_2D>::cr_point(thread, readImage);
                return { pixel.z, pixel.w, pixel.y, pixel.x };
            }
        }

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return opData.params.width;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const OperationDataType& opData) {
            return opData.params.height;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const OperationDataType& opData) {
            return 1;
        }

        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
            return { num_elems_x(Point(), opData), num_elems_y(Point(), opData), num_elems_z(Point(), opData) };
        }

        FK_HOST_FUSE InstantiableType build(const Image<PF>& data) {
            return { data.ptr() };
        }
    };

    template <PixelFormat PF>
    struct WriteYUV {
    private:
        using SelfType = WriteYUV<PF>;
    public:
        FK_STATIC_STRUCT(WriteYUV, SelfType)
        using PixelBaseType = ColorDepthPixelBaseType<PixelFormatTraits<PF>::depth>;
        using Parent = WriteOperation<ColorDepthPixelType<(ColorDepth)PixelFormatTraits<PF>::depth>,
                                      RawImage<PF>,
                                      PixelBaseType, 
                                      TF::DISABLED,
                                      WriteYUV<PF>>;
        DECLARE_WRITE_PARENT
        FK_HOST_DEVICE_FUSE void exec(const Point& thread, const InputType& input, const ParamsType& params) {
            const auto rawPtr = params.data;
            if constexpr (PF == PixelFormat::NV12 || PF == PixelFormat::P010 ||
                          PF == PixelFormat::P016 || PF == PixelFormat::P210 ||
                          PF == PixelFormat::P216) {
                // Planar luma
                *PtrAccessor<ND::_2D>::point(thread, rawPtr) = input.x;

                // Packed chroma
                using VectorType2 = VectorType_t<PixelBaseType, 2>;
                const PtrDims<ND::_2D> dims = rawPtr.dims;
                const RawPtr<ND::_2D, VectorType2> chromaPlane{
                    reinterpret_cast<VectorType2*>(reinterpret_cast<uchar*>(rawPtr.data) + dims.pitch * dims.height),
                    { dims.width >> 1, dims.height >> 1, dims.pitch }
                };
                constexpr ColorSpace CS = PixelFormatTraits<PF>::space;
                if constexpr (CS == ColorSpace::YUV420) {
                    // Chroma subsampling 4:2:0
                    if (cxp::is_even::f(thread.x) && cxp::is_even::f(thread.y)) {
                        *PtrAccessor<ND::_2D>::point({ thread.x >> 1, thread.y >> 1, thread.z }, chromaPlane) = make_<VectorType2>(input.y, input.z);
                    }
                } else {
                    // Chroma subsampling 4:2:2
                    if (cxp::is_even::f(thread.x)) {
                        *PtrAccessor<ND::_2D>::point({ thread.x >> 1, thread.y, thread.z }, chromaPlane) = make_<VectorType2>(input.y, input.z);
                    }
                }
            } else if constexpr (PF == PixelFormat::NV21) {
                // Planar luma
                *PtrAccessor<ND::_2D>::point(thread, rawPtr) = input.x;

                // Packed chroma
                const PtrDims<ND::_2D> dims = rawPtr.dims;
                const RawPtr<ND::_2D, uchar2> chromaPlane{
                    reinterpret_cast<uchar2*>(reinterpret_cast<uchar*>(rawPtr.data) + dims.pitch * dims.height),
                                              { dims.width >> 1, dims.height >> 1, dims.pitch }
                };
                *PtrAccessor<ND::_2D>::point({ thread.x >> 1, thread.y >> 1, thread.z }, chromaPlane) = make_<uchar2>(input.z, input.y);
            } else if constexpr (PF == PixelFormat::Y216 || PF == PixelFormat::Y210 || PF == PixelFormat::UYVY) {
                const PtrDims<ND::_2D> dims = rawPtr.dims;
                using VectorType2 = VectorType_t<PixelBaseType, 2>;
                const RawPtr<ND::_2D, VectorType2> imageV2{ reinterpret_cast<VectorType2*>(rawPtr.data), {dims.width >> 1, dims.height, dims.pitch} };
                //*PtrAccessor<ND::_2D>::point({ (thread.x >> 1) * 2, thread.y, thread.z }, imageV2);
                //*PtrAccessor<ND::_2D>::point({ (thread.x >> 1) * 4, thread.y, thread.z }, rawPtr);

                const bool isEvenThread = cxp::is_even::f(thread.x);
                // input = { Y, U, V }
                if constexpr (PF == PixelFormat::UYVY) {
                    // Writting UYVY from two threads
                    // Optimizable with shuffle instructions
                    if (isEvenThread) {
                        // Write UY__
                        *PtrAccessor<ND::_2D>::point({ (thread.x >> 1) * 2, thread.y, thread.z }, imageV2) = make_<VectorType2>(input.y, input.x);
                        // Write __V_
                        *PtrAccessor<ND::_2D>::point({ ((thread.x >> 1) * 4) + 2, thread.y, thread.z }, rawPtr) = input.z;
                    } else {
                        // Write ___Y
                        *PtrAccessor<ND::_2D>::point({ ((thread.x >> 1) * 4) + 3, thread.y, thread.z }, rawPtr) = input.x;
                    }
                } else {
                    // Writing YUYV from two threads
                    // Optimizable with shuffle instructions
                    if (isEvenThread) {
                        // Write YU__
                        *PtrAccessor<ND::_2D>::point({ (thread.x >> 1) * 2, thread.y, thread.z }, imageV2) = make_<VectorType2>(input.x, input.y);
                        // Write ___V
                        *PtrAccessor<ND::_2D>::point({ ((thread.x >> 1) * 4) + 3, thread.y, thread.z }, rawPtr) = input.z;
                    } else {
                        // Write __Y_
                        *PtrAccessor<ND::_2D>::point({ ((thread.x >> 1) * 4) + 2, thread.y, thread.z }, rawPtr) = input.x;
                    }
                }
            } else if constexpr (PF == PixelFormat::Y416) {
                // AVYU
                // We use ushort as the type, to be compatible with the rest of the cases
                const RawPtr<ND::_2D, ushort4> readImage{ reinterpret_cast<ushort4*>(rawPtr.data), rawPtr.dims };
                *PtrAccessor<ND::_2D>::point(thread, readImage) = make_<ushort4>(input.w, input.z, input.x, input.y);
            }
        }

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return opData.params.width;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const OperationDataType& opData) {
            return opData.params.height;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const OperationDataType& opData) {
            return 1;
        }

        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
            return { num_elems_x(Point(), opData), num_elems_y(Point(), opData), num_elems_z(Point(), opData) };
        }

        FK_HOST_FUSE InstantiableType build(const Image<PF>& data) {
            return { data.ptr() };
        }
    };

    enum class ColorConversionCodes {
        COLOR_BGR2BGRA = 0,
        COLOR_RGB2RGBA = 0,  // COLOR_BGR2BGRA
        COLOR_BGRA2BGR = 1,
        COLOR_RGBA2RGB = 1,  // COLOR_BGRA2BGR
        COLOR_BGR2RGBA = 2,
        COLOR_RGB2BGRA = 2,  // COLOR_BGR2RGBA
        COLOR_RGBA2BGR = 3,
        COLOR_BGRA2RGB = 3,  // COLOR_RGBA2BGR
        COLOR_BGR2RGB = 4,
        COLOR_RGB2BGR = 4,   // COLOR_BGR2RGB
        COLOR_BGRA2RGBA = 5,
        COLOR_RGBA2BGRA = 5, // COLOR_BGRA2RGBA
        COLOR_BGR2GRAY = 6,
        COLOR_RGB2GRAY = 7,
        COLOR_BGRA2GRAY = 10,
        COLOR_RGBA2GRAY = 11
    };

    template <ColorConversionCodes value>
    using CCC_t = E_t<ColorConversionCodes, value>;

    using SupportedCCC = TypeList<CCC_t<ColorConversionCodes::COLOR_BGR2BGRA>,  CCC_t<ColorConversionCodes::COLOR_RGB2RGBA>,
                                  CCC_t<ColorConversionCodes::COLOR_BGRA2BGR>,  CCC_t<ColorConversionCodes::COLOR_RGBA2RGB>,
                                  CCC_t<ColorConversionCodes::COLOR_BGR2RGBA>,  CCC_t<ColorConversionCodes::COLOR_RGB2BGRA>,
                                  CCC_t<ColorConversionCodes::COLOR_BGRA2RGB>,  CCC_t<ColorConversionCodes::COLOR_RGBA2BGR>,
                                  CCC_t<ColorConversionCodes::COLOR_BGR2RGB>,   CCC_t<ColorConversionCodes::COLOR_RGB2BGR>,
                                  CCC_t<ColorConversionCodes::COLOR_BGRA2RGBA>, CCC_t<ColorConversionCodes::COLOR_RGBA2BGRA>,
                                  CCC_t<ColorConversionCodes::COLOR_RGB2GRAY>,  CCC_t<ColorConversionCodes::COLOR_RGBA2GRAY>,
                                  CCC_t<ColorConversionCodes::COLOR_BGR2GRAY>,  CCC_t<ColorConversionCodes::COLOR_BGRA2GRAY>>;

    template <ColorConversionCodes CODE>
    static constexpr bool isSuportedCCC = one_of_v<CCC_t<CODE>, SupportedCCC>;

    template <ColorConversionCodes CODE, typename I, typename O, ColorDepth CD = ColorDepth::p8bit>
    struct ColorConversionType{
        static_assert(isSuportedCCC<CODE>, "Color conversion code not supported");
    };

    // Will work for COLOR_RGB2RGBA too
    template <typename I, typename O, ColorDepth CD>
    struct ColorConversionType<ColorConversionCodes::COLOR_BGR2BGRA, I, O, CD> {
        using type = AddOpaqueAlpha<I, CD>;
    };

    // Will work for COLOR_RGBA2RGB too
    template <typename I, typename O, ColorDepth CD>
    struct ColorConversionType<ColorConversionCodes::COLOR_BGRA2BGR, I, O, CD> {
        using type = Discard<I, VectorType_t<VBase<I>, 3>>;
    };

    // Will work for ColorConversionCodes::COLOR_RGB2BGRA too
    template <typename I, typename O, ColorDepth CD>
    struct ColorConversionType<ColorConversionCodes::COLOR_BGR2RGBA, I, O, CD> {
        using type = FusedOperation<VectorReorder<I, 2, 1, 0>, AddOpaqueAlpha<I, CD>>;
    };

    // Will work for COLOR_RGBA2BGR too
    template <typename I, typename O, ColorDepth CD>
    struct ColorConversionType<ColorConversionCodes::COLOR_BGRA2RGB, I, O, CD> {
        using type = FusedOperation<VectorReorder<I, 2, 1, 0, 3>,
                           Discard<I, VectorType_t<VBase<I>, 3>>>;
    };

    // Will work for ColorConversionCodes::COLOR_RGB2BGR too
    template <typename I, typename O, ColorDepth CD>
    struct ColorConversionType<ColorConversionCodes::COLOR_BGR2RGB, I, O, CD> {
        using type = VectorReorder<I, 2, 1, 0>;
    };

    // Will work for COLOR_RGBA2BGRA too
    template <typename I, typename O, ColorDepth CD>
    struct ColorConversionType<ColorConversionCodes::COLOR_BGRA2RGBA, I, O, CD> {
        using type = VectorReorder<I, 2, 1, 0, 3>;
    };

    template <typename I, typename O, ColorDepth CD>
    struct ColorConversionType<ColorConversionCodes::COLOR_RGB2GRAY, I, O, CD> {
        using type = RGB2Gray<I, O>;
    };

    template <typename I, typename O, ColorDepth CD>
    struct ColorConversionType<ColorConversionCodes::COLOR_BGR2GRAY, I, O, CD> {
        using type = FusedOperation<VectorReorder<I, 2, 1, 0>, RGB2Gray<I, O>>;
    };

    template <typename I, typename O, ColorDepth CD>
    struct ColorConversionType<ColorConversionCodes::COLOR_RGBA2GRAY, I, O, CD> {
        using type = RGB2Gray<I, O>;
    };

    template <typename I, typename O, ColorDepth CD>
    struct ColorConversionType<ColorConversionCodes::COLOR_BGRA2GRAY, I, O, CD> {
        using type = FusedOperation<VectorReorder<I, 2, 1, 0, 3>, RGB2Gray<I, O>>;
    };

    template <ColorConversionCodes code, typename I, typename O, ColorDepth CD = ColorDepth::p8bit>
    using ColorConversion = typename ColorConversionType<code, I, O, CD>::type;

} // namespace fk

#endif
