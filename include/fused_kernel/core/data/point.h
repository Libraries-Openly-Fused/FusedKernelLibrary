/* Copyright 2023-2026 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_POINT
#define FK_POINT

#include <fused_kernel/core/utils/utils.h>
#include <fused_kernel/core/data/nd.h>

namespace fk {
    template <typename T, enum ND D>
    struct Point_;

    template <typename T>
    struct Point_<T, ND::_1D> {
        T x;
    };

    template <typename T>
    struct Point_<T, ND::_2D> {
        T x;
        T y;
    };

    template <typename T>
    struct Point_<T, ND::_3D> {
        T x;
        T y;
        T z;
    };

    using Point = Point_<int, ND::_3D>;
} // namespace fk

#endif
