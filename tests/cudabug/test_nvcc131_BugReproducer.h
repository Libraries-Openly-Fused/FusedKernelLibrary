/* Copyright 2023-2025 Mediaproduccion S.L.U. (Oscar Amoros Huguet)
   Copyright 2025 Albert Andaluz
   
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. 
   */

/* NVCC Bug reproducer
   Affects versions 12.4 until 13.1
 
   This code does not compile starting on CUDA version 12.4.

   We add comments in the code, indicating the different
   modifications that make the code compile, to help with the
   bug investigation
*/

#ifdef __NVCC__
#define NVCC_VERSION_CALCULATED (__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 + __CUDACC_VER_BUILD__)
#define NVCC_VERSION_12_4_99 120499

// Condition 1: we are compiling with MSVC + nvcc OR other compilers + nvcc versions lower than 12.4.93
#if ( NVCC_VERSION_CALCULATED < NVCC_VERSION_12_4_99)
#define WILL_COMPILE 1
#else
#define WILL_NOT_COMPILE 1
#endif

// Undefine helper macros to avoid polluting the global macro namespace
#undef NVCC_VERSION_CALCULATED
#undef NVCC_VERSION_12_4_99
#else
#define WILL_COMPILE 1
#endif // __NVCC__

#ifdef WILL_COMPILE
template <bool results> constexpr bool and_v = results;

template <bool results> constexpr bool and_v2 = results;

struct MyInt {
    int instance;
    // Removing this constexpr constructor makes the code compile
    constexpr MyInt(int val) : instance(val) {}
};

constexpr bool firstFunc() {
    constexpr MyInt myint{1};
    constexpr bool result1 = myint.instance == 1;

    // Using and_v2 here and keeping and_v in line 67
    // or the other way arround, compiles
    // Changin result1 to the literal true, compiles
    return and_v<result1>;
}

constexpr bool secondFunc() { 
    return and_v<true>; 
}
#endif
int launch() {
    return 0;
}
