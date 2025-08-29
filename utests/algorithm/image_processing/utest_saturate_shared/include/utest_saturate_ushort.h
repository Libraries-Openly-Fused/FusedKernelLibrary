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

#ifndef FK_UTEST_SATURATE_USHORT_H
#define FK_UTEST_SATURATE_USHORT_H
#include "utest_saturate_common.h"

namespace fk::testushort {
#ifdef WIN32
#include "utest_saturate_ushort_export.h"
#endif
 
#ifdef WIN32
int UTEST_SATURATE_USHORT_EXPORT launch();
#else
int  EXPORT_FN_USHORT launch();
#endif
#endif
} //namespace fk::test
