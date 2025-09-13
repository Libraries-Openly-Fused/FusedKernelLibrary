/* Copyright 2025 Grup Mediapro S.L.U

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_IMAGE_H
#define FK_IMAGE_H

#include <fused_kernel/algorithms/image_processing/raw_image.h>
#include <fused_kernel/core/data/ptr_nd.h>

namespace fk {
    template <PixelFormat PF>
    class Image {
    public:
        using BaseType = ColorDepthPixelBaseType<PixelFormatTraits<PF>::depth>;
        static constexpr PixelFormat pixelFormat = PF;
    private:
        Ptr<ND::_2D, BaseType> data;
        uint width;
        uint height;
    public:
        uint imageWidth() const { return width; }
        uint imageHeight() const { return height; }

        FK_HOST_CNST Image(const Ptr<ND::_2D, BaseType>& data,
                           const uint& width,
                           const uint& height)
            : data(data), width(width), height(height) {}

        FK_HOST_CNST Image(const uint width, const uint height,
                           const MemType& memType = defaultMemType, const uint deviceID = 0)
            : width(width), height(height) {
            const uint dataWidth = width * PixelFormatTraits<PF>::rf.width_f;
            const uint dataHeight = height * PixelFormatTraits<PF>::rf.height_f;
            data = Ptr<ND::_2D, BaseType>(dataWidth, dataHeight, 0, memType, deviceID);
        }

        FK_HOST_CNST RawImage<PF> ptr() const {
            return RawImage<PF>{ data, width, height };
        }

        FK_HOST_CNST Ptr<ND::_2D, BaseType> getData() const {
            return data;
        }

        FK_HOST_CNST RawImage<PF> operator()() const {
            return ptr();
        }

        FK_HOST_CNST Image<PF> crop(const Point& p, const uint& newWidth, const uint& newHeight) {
            const uint newDataWidth = newWidth * PixelFormatTraits<PF>::rf.width_f;
            const uint newDataHeight = newHeight * PixelFormatTraits<PF>::rf.height_f;
            const Point dataPoint(p.x * PixelFormatTraits<PF>::rf.width_f, p.y * PixelFormatTraits<PF>::rf.height_f);
            PtrDims<ND::_2D> newDataDims{ newDataWidth, newDataHeight, data.dims().pitch };
            return Image<PF>(data.crop(dataPoint, newDataDims), newWidth, newHeight);
        }
#if !defined(NVRTC_COMPILER)
#if defined(__NVCC__) || CLANG_HOST || defined(__HIP__) || defined(NVRTC_ENABLED)
        inline void uploadTo(Image& other, cudaStream_t stream = 0) {
            data.uploadTo(other.data, stream);
        }

        inline void downloadTo(Image& other, cudaStream_t stream = 0) {
            data.downloadTo(other.data, stream);
        }

        inline void upload(Stream_<ParArch::GPU_NVIDIA>& stream) {
            data.upload(stream);
        }
        inline void download(Stream_<ParArch::GPU_NVIDIA>& stream) {
            data.download(stream);
        }
#else
        inline void upload(Stream& stream) {}
        inline void download(Stream& stream) {}
#endif // defined(__NVCC__) || defined(__HIP__) || defined(NVRTC_ENABLED)
#endif // defined(NVRTC_COMPILER)

        FK_HOST_CNST VectorType_t<BaseType, PixelFormatTraits<PF>::cn> readAt(const Point& p) const {
            return ReadYUV<PF>::exec(p, ptr());
        }
    };
} // namespace fk

#endif // FK_IMAGE_H