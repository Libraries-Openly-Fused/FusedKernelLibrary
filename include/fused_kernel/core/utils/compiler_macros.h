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

#if defined(__clang__) && defined(__CUDA__) 
// clang compiling CUDA code, both host and device mode
#define CLANG_HOST_DEVICE 1
#else
#define CLANG_HOST_DEVICE 0
#endif

#endif // COMPILER_MACROS_H