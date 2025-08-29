/* Copyright 2025 Oscar Amoros Huguet
   Copyright 2025 Grup Mediapro S.L.U

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_UTEST_SATURATE_UCHAR_H
#define FK_UTEST_SATURATE_UCHAR_H

#include "utest_saturate_common.h"

namespace fk::testuchar {
#ifdef WIN32
#include "utest_saturate_uchar_export.h"
#endif

#if defined(__GNUC__) && !defined(_WIN32)
#define  EXPORT_FN_UCHAR  __attribute__((visibility("default")))
#else
#define  EXPORT_FN_UCHAR 
#endif 
#ifdef WIN32
int UTEST_SATURATE_UCHAR_EXPORT launch();
#else
int  EXPORT_FN_UCHAR launch();
#endif
#endif
} //namespace fk::test
