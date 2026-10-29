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

#include <tests/main.h>

#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/data/ptr_utils.h>
#include <fused_kernel/fused_kernel.h>

#include <iostream>

// The grid and the thread fusion thread space are sized with integer divisions. ceil(n / static_cast<float>(d)) first
// rounds n to a float, so above 2^24 the last block, or with thread fusion the last group of elements, was missing and
// the last elements were never written: 1 of 2^24 + 1 elements without thread fusion, and 4 of 2^26 + 4 with it
// (2^24 + 1 threads that process an uchar4 each).
template <fk::TF TFEN>
bool everyElementWritten(const uint width) {
    using T = unsigned char;
    fk::Stream stream;
    fk::Ptr1D<T> input(width);
    fk::Ptr1D<T> output(width);
    fk::setTo(static_cast<T>(7), input, stream);
    fk::setTo(static_cast<T>(0), output, stream);
    fk::executeOperations<fk::TransformDPP<fk::defaultParArch, TFEN>>(
        stream, fk::PerThreadRead<fk::ND::_1D, T>::build(input.ptr()), fk::PerThreadWrite<fk::ND::_1D, T>::build(output.ptr()));
    output.download(stream);
    stream.sync();

    uint missing = 0;
    for (uint x = 0; x < width; ++x) {
        missing += output.at(fk::Point{ static_cast<int>(x), 0, 0 }) != 7 ? 1 : 0;
    }
    std::cout << "Width " << width << (TFEN == fk::TF::ENABLED ? " with" : " without") << " thread fusion: " << missing
              << " elements not written" << std::endl;
    return missing == 0;
}

int launch() {
    constexpr uint twoPow24 = 1u << 24;
    bool correct = everyElementWritten<fk::TF::DISABLED>(twoPow24 + 1);
    correct &= everyElementWritten<fk::TF::ENABLED>(4 * twoPow24 + 4);
    correct &= everyElementWritten<fk::TF::ENABLED>(twoPow24);
    return correct ? 0 : -1;
}
