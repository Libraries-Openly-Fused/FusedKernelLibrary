/* Copyright 2024 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#define __ONLY_CU__ // This file is only generated and compiled with nvcc, not with the host compiler

#include <tests/main.h>

#include <type_traits>

// Why does this code not compile?
// There are 4 ways of making it compile
// 1) Compile with nvcc 12.3 or older, or 12.8 or newer

template <bool results>
constexpr bool variable = results;

template <typename T>
constexpr bool getValue(const T& value) {
    return true;
}

struct SomeType {
    int member;
};

constexpr bool function1() {
    constexpr SomeType test{ 1 };

    // 2) Replace this return with: return and_v<getValue<SomeType>(test)>;
    // 3) Replace this return with: return variable<getValue(5)>;
    return variable<getValue(test)>;
}

// 4) Comment out this function
/*constexpr bool function2() {
    return variable<true>;
}*/

int launch() {
    return 0;
}
