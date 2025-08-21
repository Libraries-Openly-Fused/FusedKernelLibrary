/* Copyright 2025 Oscar Amoros Huguet
   Copyright 2025 Albert Andaluz

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#define __ONLY_CPU__

#ifndef FK_UTEST_CUDA_VECTOR_UTILS_H
#define FK_UTEST_CUDA_VECTOR_UTILS_H

#include "utest_cuda_vector_utils_char.h"
#include "utest_cuda_vector_utils_double.h"
#include "utest_cuda_vector_utils_float.h"
#include "utest_cuda_vector_utils_int.h"
#include "utest_cuda_vector_utils_long.h"
#include "utest_cuda_vector_utils_longlong.h"
#include "utest_cuda_vector_utils_short.h"
#include "utest_cuda_vector_utils_uchar.h"
#include "utest_cuda_vector_utils_uint.h"
#include "utest_cuda_vector_utils_ulong.h"
#include "utest_cuda_vector_utils_ulonglong.h"
#include "utest_cuda_vector_utils_ushort.h"

int launch() { return launchchar(); }

#endif // FK_UTEST_CUDA_VECTOR_UTILS_H