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

 


#include "utest_saturate_uint.h"
 
using namespace fk::testuint;
START_ADDING_TESTS
using Fundamental = fk::RemoveType_t<0, fk::StandardTypes>;
addAllOutputTestsForInput<Fundamental,uint>(std::make_index_sequence<Fundamental::size>{});
STOP_ADDING_TESTS
 
#ifdef WIN32
int  UTEST_SATURATE_UINT_EXPORT fk::testuint::launch() {
   RUN_ALL_TESTS
   return 0;
}

#else
 
int EXPORT_FN_UINT fk::testuint::launch() {
   RUN_ALL_TESTS
   return 0;
}
#endif

 
