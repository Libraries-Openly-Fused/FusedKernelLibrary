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

#include <fused_kernel/algorithms/image_processing/itu_color.h>
#include <fused_kernel/core/data/rawptr.h>

namespace fk {
    template <PixelFormat PF>
    struct RawImage {
        using BaseType = ColorDepthPixelBaseType<PixelFormatTraits<PF>::depth>;
        RawPtr<ND::_2D, BaseType> data; // Raw image data
        uint width;
        uint height;
    };
} // namespace fk

#endif // FK_RAW_IMAGE_H