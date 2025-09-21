/* Copyright 2025 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_STATIC_GET_H
#define FK_STATIC_GET_H

#include <fused_kernel/core/utils/utils.h>

namespace fk {
    template <typename VT>
    class IsCudaVector;
    template <typename VT>
    class VectorTraits;

    template <size_t Idx>
    struct static_get {
        template <typename VT>
        FK_HOST_DEVICE_FUSE auto f(const VT& v) -> std::enable_if_t<IsCudaVector<VT>::value,
            typename VectorTraits<VT>::base>;
    };
} // namespace fk


#endif // FK_STATIC_GET_H