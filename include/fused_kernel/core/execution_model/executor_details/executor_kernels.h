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

#ifndef FK_EXECUTOR_KERNELS_H
#define FK_EXECUTOR_KERNELS_H

#if defined(__NVCC__) || CLANG_HOST_DEVICE
namespace fk {
template <enum ParArch PA, typename SequenceSelector, typename... IOpSequences>
__global__ void launchDivergentBatchTransformDPP_Kernel(const __grid_constant__ IOpSequences... iOpSequences) {
    DivergentBatchTransformDPP<PA, SequenceSelector>::exec(iOpSequences...);
}

template <enum ParArch PA, enum TF TFEN, bool THREAD_DIVISIBLE, typename TDPPDetails, typename... IOps>
__global__ void launchTransformDPP_Kernel(const __grid_constant__ TDPPDetails tDPPDetails,
                                          const __grid_constant__ IOps... operations) {
    TransformDPP<PA, TFEN, TDPPDetails, THREAD_DIVISIBLE>::exec(tDPPDetails, operations...);
}

} // namespace fk
#endif

#endif // FK_EXECUTOR_KERNELS_H