/* Copyright 2026 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include <tests/main.h>

// raw_image.h as the first FKL header: it brings the FKL vector types itself, so its pixel types are fk::uchar3...,
// not HIP's global uchar3... that hip_runtime.h declares
#include <fused_kernel/algorithms/image_processing/raw_image.h>

#include <type_traits>

static_assert(std::is_same_v<fk::ColorDepthPixelType<fk::ColorDepth::p8bit>, fk::uchar3>,
              "the 8 bit pixel type must be fk::uchar3");
static_assert(std::is_same_v<fk::ColorDepthPixelType<fk::ColorDepth::p10bit>, fk::ushort3>,
              "the 10 bit pixel type must be fk::ushort3");
static_assert(std::is_same_v<fk::ColorDepthPixelType<fk::ColorDepth::f24bit>, fk::float3>,
              "the f24 pixel type must be fk::float3");

int launch() {
    return 0;
}
