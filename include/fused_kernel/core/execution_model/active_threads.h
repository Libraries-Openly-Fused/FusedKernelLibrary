/* Copyright 2023-2025 Oscar Amoros Huguet
   Copyright 2025 Grup Mediapro S.L.U (Oscar Amoros Huguet)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */


#ifndef FK_ACTIVE_THREADS_H
#define FK_ACTIVE_THREADS_H

namespace fk { // namespace FusedKernel
    /**
     * @brief ActiveThreads: represents the number of active threads in a kernel.
     * It is used to determine how many threads are currently executing in a kernel.
     */
    struct ActiveThreads {
        uint x, y, z;
        FK_HOST_DEVICE_CNST
            ActiveThreads(const uint& vx = 1,
                          const uint& vy = 1,
                          const uint& vz = 1)
            : x(vx), y(vy), z(vz) {
        }
    };

    /**
     * @brief ceilDiv: the number of groups of b elements (threads, blocks...) that cover a elements, in integer
     * arithmetic. ceil(a / static_cast<float>(b)) rounds a to a float first, which above 2^24 can give one group less.
     * Groups of 0 elements (the blocks of an empty thread space) give 0 groups, so that a launch with them fails with an
     * invalid configuration error instead of dividing by 0.
     */
    FK_HOST_DEVICE_CNST uint ceilDiv(const uint& a, const uint& b) {
        return b == 0 ? 0u : a / b + (a % b != 0 ? 1u : 0u);
    }
} // namespace fk

#endif // FK_ACTIVE_THREADS_H