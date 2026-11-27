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
// The kernel of every DPP: executeOperations<DPP>(stream, ...) launches launchDPP_Kernel<DPP> with the DPP's details (or
// Params) and IOps, and the kernel only calls DPP::exec with them. The DPP is the first template argument, so the kernel's
// name says which DPP it runs; DPPDetails and IOps are deduced from the launch arguments. Every parameter is
// __grid_constant__: read in place from the parameter space, never copied to a local (Oscar's decision, 2026-11-26).
template <typename DPP, typename DPPDetails, typename... IOps>
__global__ void launchDPP_Kernel(const __grid_constant__ DPPDetails details, const __grid_constant__ IOps... iOps) {
    DPP::exec(details, iOps...);
}

} // namespace fk
#endif

#endif // FK_EXECUTOR_KERNELS_H