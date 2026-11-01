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

#ifndef FK_ATTENTION_ATTENTION_WAVE_H
#define FK_ATTENTION_ATTENTION_WAVE_H

/* Wave (warp) size of the attention DPPs: they map one row, or one fragment
 * row group, onto exactly 32 lanes (NVIDIA warps, and RDNA waves compiled as
 * wave32, the HIP default on gfx12). Wave64 is not supported.
 *
 * ROCm's clang no longer exposes the wave size at compile time
 * (__AMDGCN_WAVEFRONT_SIZE__ was removed, and neither warpSize nor
 * __builtin_amdgcn_wavefrontsize() is a constant expression), so:
 *  - with a compiler that still defines __AMDGCN_WAVEFRONT_SIZE__, a wave64
 *    device compilation fails here with #error;
 *  - otherwise the kernels call attnRequireWave32(), which the optimizer folds
 *    away in a wave32 build and turns into a trap in a wave64 one. */

#include <fused_kernel/core/utils/utils.h>

namespace fk {

constexpr int ATTN_WAVE_SIZE = 32;

#if defined(__HIPCC__) && defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN_WAVEFRONT_SIZE__)
#if __AMDGCN_WAVEFRONT_SIZE__ != 32
#error "The attention DPPs need wave32: don't compile them with -mwavefrontsize64"
#endif
#endif

#if defined(__NVCC__) || defined(__HIPCC__)

// Traps when the kernel runs as wave64. Compiles to nothing in a wave32 (and in
// any CUDA) build.
__device__ __forceinline__ void attnRequireWave32() {
#if defined(__HIPCC__) && defined(__HIP_DEVICE_COMPILE__)
    if (__builtin_amdgcn_wavefrontsize() != ATTN_WAVE_SIZE) {
        __builtin_trap();
    }
#endif
}

// Butterfly exchange across the 32 lanes of the wave: returns the value of
// lane (lane ^ laneMask).
template <typename T>
__device__ __forceinline__ T attnShflXor(const T value, const int laneMask) {
#if defined(__HIPCC__)
    return __shfl_xor(value, laneMask, ATTN_WAVE_SIZE);
#else
    return __shfl_xor_sync(0xffffffffu, value, laneMask);
#endif
}

#endif // defined(__NVCC__) || defined(__HIPCC__)

} // namespace fk

#endif // FK_ATTENTION_ATTENTION_WAVE_H
