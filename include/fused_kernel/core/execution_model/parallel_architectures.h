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

#ifndef PARALLEL_ARCHITECTURES_H
#define PARALLEL_ARCHITECTURES_H

#include <string_view>
#include <fused_kernel/core/utils/compiler_macros.h>

namespace fk {

#define PARALLEL_ARCHITECTURES \
    PAR_ARCH_VALUE(CPU) \
    PAR_ARCH_VALUE(GPU_NVIDIA) \
    PAR_ARCH_VALUE(GPU_NVIDIA_JIT) \
    PAR_ARCH_VALUE(GPU_AMD) \
    PAR_ARCH_VALUE(CPU_OMP) \
    PAR_ARCH_VALUE(CPU_OMPSS) \
    PAR_ARCH_VALUE(MULTI_GPU_NVIDIA) \
    PAR_ARCH_VALUE(CLUSTER_GPU_NVIDIA) \
    PAR_ARCH_VALUE(None)

    enum class ParArch {
#define PAR_ARCH_VALUE(name) name,
        PARALLEL_ARCHITECTURES
#undef PAR_ARCH_VALUE
    };

    constexpr inline std::string_view toStrView(const ParArch& arch) {
        switch (arch) {
#define PAR_ARCH_VALUE(name) case ParArch::name: { return std::string_view{#name}; }
        PARALLEL_ARCHITECTURES
#undef PAR_ARCH_VALUE
        default: return "Unknown";
    }
}
#undef PARALLEL_ARCHITECTURES

#if defined(__NVCC__)
    constexpr ParArch defaultParArch = ParArch::GPU_NVIDIA;
#else
    constexpr ParArch defaultParArch = ParArch::CPU;
#endif

} // namespace fk

#endif