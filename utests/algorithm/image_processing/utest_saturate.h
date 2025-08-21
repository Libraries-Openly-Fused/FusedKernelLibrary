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
#ifndef FK_UTEST_SATURATE_H
#define FK_UTEST_SATURATE_H
#include "utest_saturate_char.h"
#include "utest_saturate_double.h"
#include "utest_saturate_float.h"
#include "utest_saturate_int.h"
#include "utest_saturate_long.h"
#include "utest_saturate_longlong.h"
#include "utest_saturate_short.h"
#include "utest_saturate_uchar.h"
#include "utest_saturate_uint.h"
#include "utest_saturate_ulong.h"
#include "utest_saturate_ulonglong.h"
#include "utest_saturate_ushort.h"

int launch() { 
     launchchar();    
     launchdouble();    
     launchfloat();
     launchint();
     launchlong();
     launchlonglong();
     launchshort();
     launchuchar();
     launchuint();
     launchulong();
     launchulonglong();
     launchushort();
    return 0;
}

#endif