/* Copyright 2026 Oscar Amoros Huguet, Johnny Nunez

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_COLLECTIVE_EPILOGUE_H
#define FK_COLLECTIVE_EPILOGUE_H

#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/logical.h>
#include <fused_kernel/core/utils/utils.h>

namespace fk {

/** Build out = alpha * accumulator + beta as a fused Write IOp. */
template <typename T, typename WriteIOp>
FK_HOST_CNST auto scaleBiasEpilogue(const T alpha, const T beta,
                                    const WriteIOp& write) {
    return Mul<T>::build(alpha)
        .then(Add<T>::build(beta))
        .then(write);
}

/** Build out = max(accumulator, 0) as a fused Write IOp. */
template <typename T, typename WriteIOp>
FK_HOST_CNST auto reluEpilogue(const WriteIOp& write) {
    return Max<T>::build(T{}).then(write);
}

} // namespace fk

#endif // FK_COLLECTIVE_EPILOGUE_H
