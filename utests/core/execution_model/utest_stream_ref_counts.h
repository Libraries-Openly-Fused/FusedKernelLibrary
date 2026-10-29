/* Copyright 2026 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

// The reference counts of Ref, the base of every Stream_, here through Stream_<ParArch::CPU> in every backend: copies
// share one counter, and an assignment releases the counter of the object before sharing the other one (it kept it, so
// the counts grew and were never released).

#include <tests/main.h>

#include <fused_kernel/core/execution_model/stream.h>

#include <iostream>

int launch() {
    using CPUStream = fk::Stream_<fk::ParArch::CPU>;
    bool correct = true;

    CPUStream a;
    CPUStream b;
    const CPUStream c = b;
    correct &= a.getRefCount() == 1 && b.getRefCount() == 2 && c.getRefCount() == 2;

    // b leaves the counter it shares with c, and shares the one of a
    b = a;
    correct &= a.getRefCount() == 2 && b.getRefCount() == 2 && c.getRefCount() == 1;

    // Assigning the counter an object already shares, or the object itself, changes nothing
    b = a;
    CPUStream& alias = b;
    b = alias;
    correct &= a.getRefCount() == 2 && c.getRefCount() == 1;

    // A copy that goes away releases its count
    {
        const CPUStream d = a;
        correct &= a.getRefCount() == 3;
    }
    correct &= a.getRefCount() == 2;

    std::cout << "Stream reference counts: a " << a.getRefCount() << ", c " << c.getRefCount() << " "
              << (correct ? "OK" : "WRONG") << std::endl;
    return correct ? 0 : -1;
}
