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
} // namespace fk

#endif // FK_ACTIVE_THREADS_H