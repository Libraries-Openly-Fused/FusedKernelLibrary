/* Copyright 2024-2026 Oscar Amoros Huguet

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

#include <fused_kernel/core/data/rect.h>

int launch() {

    constexpr fk::Rect test(fk::Point{16, 32, 0}, fk::Size(32, 64));
    static_assert(test.x == 16, "Something wrong");
    static_assert(test.y == 32, "Something wrong");
    static_assert(test.width == 32, "Something wrong");
    static_assert(test.height == 64, "Something wrong");

    constexpr fk::Rect test2;
    static_assert(test2.x == 0, "Something wrong");
    static_assert(test2.y == 0, "Something wrong");
    static_assert(test2.width == 0, "Something wrong");
    static_assert(test2.height == 0, "Something wrong");

    constexpr fk::Rect_<float, double> test3(3.f, 5.f, 10.0, 20.0);
    static_assert(test3.x == 3.f, "Something wrong");
    static_assert(test3.y == 5.f, "Something wrong");
    static_assert(test3.width == 10.0, "Something wrong");
    static_assert(test3.height == 20.0, "Something wrong");

    return 0;
}
