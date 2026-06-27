/* Copyright 2025 Grup Mediapro S.L.U (Oscar Amoros Huguet)

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

#include <array>

/* This code does not compile with gcc 13.3.0 with:
*   - CUDA SDK 12.8.1 (nvcc 12.8.93)
*   - CUDA SDK 12.9.0 (nvcc 12.9.41)
*
* It compiles fine with nvcc 12.8.93 or 12.9.41 + MSVC 19.42.34438.0 (MSVC 2022) on Windows
*/

#ifdef __NVCC__
#define NVCC_VERSION_CALCULATED (__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 + __CUDACC_VER_BUILD__)
#define NVCC_VERSION_12_8_93 120893

// Condition 1: we are compiling with MSVC + nvcc OR other compilers + nvcc versions lower than 12.8.93
#if (defined(_MSC_VER)) || \
        (!defined(_MSC_VER) && (NVCC_VERSION_CALCULATED < NVCC_VERSION_12_8_93))
#define WILL_COMPILE 1
#endif

// Condition 2: we are compiling with gcc/clan AND we are compiling with nvcc versions 12.8.93 or higher
#if (!defined(_MSC_VER)) && \
        (NVCC_VERSION_CALCULATED >= NVCC_VERSION_12_8_93)
#define WILL_NOT_COMPILE 1
#pragma message("nvcc version 12.8.93 or higher detected! Test will be skipped.")
#endif

// Undefine helper macros to avoid polluting the global macro namespace
#undef NVCC_VERSION_CALCULATED
#undef NVCC_VERSION_12_8_94
#else
#define WILL_COMPILE 1
#endif // __NVCC__

// To check the compilation issue, make sure you are compiling with gcc or clang + nvcc 12.8.93 or higuer
// and uncomment the line avove
// #define WILL_COMPILE 1
#ifdef WILL_COMPILE
void test1() {
    // Remove the constexpr std::size_t NUM, and use 5 directly, and it will compile
    constexpr std::size_t NUM = 5;
    [[maybe_unused]] const std::array<int, NUM> d_imgs{ 1, 2, 3, 4, 5 };
}
void test2() {
    // Change the std::array size to something different from 5 and it will compile
    [[maybe_unused]] const std::array<int, 5> d_imgs2{ 6, 7, 8, 9, 10 };
}
#endif

int launch() {
    return 0;
}
