/* Copyright 2025 Oscar Amoros Huguet
   Copyright 2025 Albert Andaluz Gonzalez

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef COMPILER_MACROS_H
#define COMPILER_MACROS_H

#if defined(_MSC_VER)
#define _MSC_VER_EXISTS 1
#else
#define _MSC_VER_EXISTS 0
#endif

// SYCL backend (ParArch::GPU_INTEL). A compiler that compiles the translation unit as SYCL predefines
// SYCL_LANGUAGE_VERSION (DPC++: icx, icpx or clang++ with -fsycl), as nvcc predefines __NVCC__ and hipcc __HIPCC__.
// FK_SYCL_ENABLED is what FKL's code checks. Define FK_SYCL_DISABLED to compile a SYCL translation unit without the
// backend (ParArch::CPU then is the default one).
#if defined(SYCL_LANGUAGE_VERSION) && !defined(__NVCC__) && !defined(__HIPCC__) && !defined(NVRTC_COMPILER) && \
    !defined(FK_SYCL_DISABLED)
#define FK_SYCL_ENABLED 1
#endif


#endif // COMPILER_MACROS_H