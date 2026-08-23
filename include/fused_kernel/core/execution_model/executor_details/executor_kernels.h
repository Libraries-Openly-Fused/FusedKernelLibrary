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

#if defined(__NVCC__) || defined(__HIPCC__)
namespace fk {
#if defined(__NVCC__)
#define FK_GRID_CONSTANT __grid_constant__
#else
#define FK_GRID_CONSTANT
#endif
template <ParArch PA, typename SequenceSelector, typename DPPDetails, typename... IOpSequences>
__global__ void launchDivergentBatchTransformDPP_Kernel(const FK_GRID_CONSTANT DPPDetails details,
                                                        const FK_GRID_CONSTANT IOpSequences... iOpSequences) {
    DivergentBatchTransformDPP<PA, SequenceSelector>::exec(details, iOpSequences...);
}

template <ParArch PA, TF TFEN, bool THREAD_DIVISIBLE, typename TDPPDetails, typename... IOps>
__global__ void launchTransformDPP_Kernel(const FK_GRID_CONSTANT TDPPDetails tDPPDetails,
                                          const FK_GRID_CONSTANT IOps... operations) {
    TransformDPP<PA, TFEN, TDPPDetails, THREAD_DIVISIBLE>::exec(tDPPDetails, operations...);
}

#undef FK_GRID_CONSTANT
} // namespace fk
#endif

#endif // FK_EXECUTOR_KERNELS_H