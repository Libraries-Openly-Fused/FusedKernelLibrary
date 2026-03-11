/* Copyright 2024 Oscar Amoros Huguet

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifndef FK_RECT
#define FK_RECT

#include <fused_kernel/core/data/size.h>
#include <fused_kernel/core/data/point.h>

namespace fk {

    template <typename P, typename S=P>
    struct Rect_ {
        P x{ 0 }, y{0};
        S width{ 0 }, height{0};
        FK_HOST_DEVICE_CNST Rect_(const Point point, const Size size) : x(point.x), y(point.y), width(size.width), height(size.height) {}
        FK_HOST_DEVICE_CNST Rect_(const P x_, const P y_, const S width_, const S height_) : x(x_), y(y_), width(width_), height(height_) {}
        FK_HOST_DEVICE_CNST Rect_(){}
    };

    using Rect = Rect_<uint, int>;

} // namespace fk

#endif
