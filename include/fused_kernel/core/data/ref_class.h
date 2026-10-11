/* Copyright 2025 Grup Mediapro S.L.U (Oscar Amoros Huguet)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_REF_CLASS_H
#define FK_REF_CLASS_H

#ifndef NVRTC_COMPILER
#include <atomic>
#include <stdexcept>

namespace fk {

    // Reference counted handle. Copies share the counter, which is atomic: copies may be made and destroyed
    // concurrently from different threads.
    class Ref {
        struct RefPtr {
            std::atomic<int> cnt{ 1 };
        };
        RefPtr* ref{ nullptr };
        inline void acquire(const Ref& other) {
            ref = other.ref;
            if (ref) {
                ref->cnt.fetch_add(1, std::memory_order_relaxed);
            }
        }
        inline void release() {
            if (ref && ref->cnt.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                delete ref;
            }
            ref = nullptr;
        }
    public:
        inline Ref() : ref(new RefPtr) {}

        inline Ref(const Ref& other) {
            acquire(other);
        }

        Ref(Ref&&) = delete;
        Ref& operator=(Ref&&) = delete;

        Ref& operator=(const Ref& other) {
            if (this != &other && ref != other.ref) {
                release();
                acquire(other);
            }
            return *this;
        }

        virtual inline ~Ref() {
            release();
        }
        inline int getRefCount() const {
            return ref ? ref->cnt.load(std::memory_order_relaxed) : 0;
        }

    };
} // namespace fk
#endif // NVRTC_COMPILER
#endif // !FK_REF_CLASS_H
