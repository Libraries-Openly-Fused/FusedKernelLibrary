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

#ifndef FK_ATTENTION_SOFTMAX_H
#define FK_ATTENTION_SOFTMAX_H

/* SoftmaxDPP: numerically-stable row-wise softmax as a single cooperative
 * DPP (one DPP per kernel, per the current FKL execution model).
 *
 * Not a TransformDPP: softmax needs a row REDUCTION (cross-thread
 * cooperation), so this is a cooperative DPP kind with block-level
 * shared-memory state. Online softmax (Milakov & Gimelshein, 2018):
 * each thread scans a strided slice keeping a running (max m, sum l);
 * states merge associatively; a second pass writes exp(x-m)/l.
 * Two reads + one write per element, no global intermediates, immune to
 * overflow for any input range.
 */

#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/execution_model/stream.h>
#include <fused_kernel/core/utils/utils.h>

#include <cfloat>
#include <cmath>
#if defined(__NVCC__) || CLANG_HOST_DEVICE
#include <cuda_fp16.h>
#endif

namespace fk {

#if !defined(__CUDA_ARCH__) && !defined(__NVCC__) && !CLANG_HOST_DEVICE
using std::fmaxf;
using std::expf;
#endif

// Cooperative-DPP exec bodies use __shared__ + barriers: they cannot be
// constexpr (FK_DEVICE_FUSE). Plain static device inline qualifier:
#define FK_COOP_DEVICE_FUSE static __device__ __forceinline__ void

template <typename T>
FK_HOST_DEVICE_CNST float attnToF32(const T& v) {
#if defined(__NVCC__) || CLANG_HOST_DEVICE
    if constexpr (std::is_same_v<T, __half>) { return __half2float(v); } else
#endif
    { return static_cast<float>(v); }
}

template <typename T>
FK_HOST_DEVICE_CNST T attnFromF32(const float& v) {
#if defined(__NVCC__) || CLANG_HOST_DEVICE
    if constexpr (std::is_same_v<T, __half>) { return __float2half(v); } else
#endif
    { return static_cast<T>(v); }
}

struct OnlineSoftmaxState {
    float m;  // running max
    float l;  // running sum of exp(x - m)
};

FK_HOST_DEVICE_CNST OnlineSoftmaxState mergeSoftmaxStates(const OnlineSoftmaxState& a,
                                                          const OnlineSoftmaxState& b) {
    const float m = fmaxf(a.m, b.m);
    const float la = a.l == 0.f ? 0.f : a.l * expf(a.m - m);
    const float lb = b.l == 0.f ? 0.f : b.l * expf(b.m - m);
    return { m, la + lb };
}

#if defined(__NVCC__) || CLANG_HOST_DEVICE

template <typename T, int BLOCK_SIZE = 256>
struct SoftmaxDPP {
private:
    using SelfType = SoftmaxDPP<T, BLOCK_SIZE>;
public:
    FK_STATIC_STRUCT(SoftmaxDPP, SelfType)

    struct Params {
        RawPtr<ND::_2D, T> input;
        RawPtr<ND::_2D, T> output;
    };

    FK_COOP_DEVICE_FUSE exec(const Params& p) {
        __shared__ OnlineSoftmaxState states[BLOCK_SIZE];

        const int row = blockIdx.x;
        const int tid = threadIdx.x;
        const int width = static_cast<int>(p.input.dims.width);

        const T* in = PtrAccessor<ND::_2D>::cr_point(Point{0, row, 0}, p.input);
        T* out = PtrAccessor<ND::_2D>::point(Point{0, row, 0}, p.output);

        OnlineSoftmaxState st{ -FLT_MAX, 0.f };
        for (int x = tid; x < width; x += BLOCK_SIZE) {
            const float v = attnToF32(in[x]);
            const float m = fmaxf(st.m, v);
            st.l = st.l * expf(st.m - m) + expf(v - m);
            st.m = m;
        }
        states[tid] = st;
        __syncthreads();

        for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                states[tid] = mergeSoftmaxStates(states[tid], states[tid + stride]);
            }
            __syncthreads();
        }
        const float m = states[0].m;
        const float invL = 1.f / states[0].l;

        for (int x = tid; x < width; x += BLOCK_SIZE) {
            out[x] = attnFromF32<T>(expf(attnToF32(in[x]) - m) * invL);
        }
    }
};

template <typename T, int BLOCK_SIZE>
__global__ void launchSoftmaxDPP_Kernel(const __grid_constant__ typename SoftmaxDPP<T, BLOCK_SIZE>::Params params) {
    SoftmaxDPP<T, BLOCK_SIZE>::exec(params);
}

template <typename T, int BLOCK_SIZE = 256>
inline void executeSoftmax(const Ptr2D<T>& input, const Ptr2D<T>& output,
                           Stream_<ParArch::GPU_NVIDIA>& stream) {
    using DPP = SoftmaxDPP<T, BLOCK_SIZE>;
    const typename DPP::Params params{ input.ptr(), output.ptr() };
    const dim3 grid(input.dims().height, 1, 1);
    const dim3 block(BLOCK_SIZE, 1, 1);
    launchSoftmaxDPP_Kernel<T, BLOCK_SIZE><<<grid, block, 0, stream.getCUDAStream()>>>(params);
    gpuErrchk(cudaGetLastError());
}

#endif // defined(__NVCC__) || CLANG_HOST_DEVICE

} // namespace fk

#endif // FK_ATTENTION_SOFTMAX_H
