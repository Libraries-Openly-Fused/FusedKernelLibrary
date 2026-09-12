/* Copyright 2025 Grup Mediapro S.L.U (Oscar Amoros Hguet)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "tests/main.h"

#include <fused_kernel/core/data/string.h>
#include <sstream>

namespace fk {

int launch_impl() {
    bool result{ true };
    {
        String str1("");
        String str2("Hello");
        String str3(" ");
        String str4("World!");

        std::stringstream ss;

        ss << str1 << str2 << str3 << str4;
        if (ss.str() != std::string("Hello World!")) {
            std::cout << "String operator<< failed: result " << ss.str() <<
                " expected: " << "Hello World!" << std::endl;
            result &= false;
        }
    }

    {
        String str1("");
        String str2("Hello");
        String str4("World!");

        auto str5 = str1 + " Hi " + str2 + " brave " + " new " + str4;

        if (!(str5 == String(" Hi Hello brave  new World!"))) {
            std::cout << "String operator+ with const char* elements failed" << std::endl;
            result &= false;
        }
    }

    {
        constexpr String str1("");
        constexpr String str2("Hello");
        constexpr String str3(" ");
        constexpr String str4("World!");
        constexpr String str5 = str1 + str2 + str3 + str4;

        static_assert(str5 == String("Hello World!"), "Error in operator== in constexpr context");
        result &= true;
    }

    return result ? 0 : -1;
}

} // namespace fk

int launch() {
    return fk::launch_impl();
}
