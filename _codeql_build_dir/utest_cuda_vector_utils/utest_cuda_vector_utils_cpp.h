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

#ifndef FK_UTEST_CUDA_VECTOR_UTILS_cpp_H
#define FK_UTEST_CUDA_VECTOR_UTILS_cpp_H

#include "utest_cuda_vector_utils_char_cpp.h"
#include "utest_cuda_vector_utils_double_cpp.h"
#include "utest_cuda_vector_utils_float_cpp.h"
#include "utest_cuda_vector_utils_int_cpp.h"
#include "utest_cuda_vector_utils_long_cpp.h"
#include "utest_cuda_vector_utils_longlong_cpp.h"
#include "utest_cuda_vector_utils_short_cpp.h"
#include "utest_cuda_vector_utils_uchar_cpp.h"
#include "utest_cuda_vector_utils_uint_cpp.h"
#include "utest_cuda_vector_utils_ulong_cpp.h"
#include "utest_cuda_vector_utils_ulonglong_cpp.h"
#include "utest_cuda_vector_utils_ushort_cpp.h"

#include <vector>
#include <algorithm>

int launch() {
    const std::vector<int> v{
        fk::utest_cuda_vector_utils_char_cpp::launch(),      fk::utest_cuda_vector_utils_double_cpp::launch(),
        fk::utest_cuda_vector_utils_float_cpp::launch(),     fk::utest_cuda_vector_utils_int_cpp::launch(),
        fk::utest_cuda_vector_utils_long_cpp::launch(),      fk::utest_cuda_vector_utils_longlong_cpp::launch(),
        fk::utest_cuda_vector_utils_short_cpp::launch(),     fk::utest_cuda_vector_utils_uchar_cpp::launch(),
        fk::utest_cuda_vector_utils_uint_cpp::launch(),      fk::utest_cuda_vector_utils_ulong_cpp::launch(),
        fk::utest_cuda_vector_utils_ulonglong_cpp::launch(), fk::utest_cuda_vector_utils_ushort_cpp::launch(),
    };
    bool zeros = std::all_of(v.begin(), v.end(), [](int i) { return i == 0; });
 
    return zeros ? 0 : -1;
}

#endif // FK_UTEST_CUDA_VECTOR_UTILS_H
