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
 *
 * PROLOGUE FUSION: the DPP takes the input as a Read or ReadBack IOp
 * (the prologue) and reads each data element through that IOp, so any
 * fused preprocessing chain (read.then(Mul...).then(Cast...)) runs
 * in-register at load time, on BOTH passes, with zero extra traffic.
 * Element addressing: thread.x = column, thread.y = row, thread.z = 0.
 */

#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/execution_model/stream.h>
#include <fused_kernel/core/execution_model/operation_model/operation_model.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/core/utils/utils.h>

#include <cfloat>
#include <cmath>
#if defined(__NVCC__) || CLANG_HOST_DEVICE
#include <cuda_fp16.h>
#endif

namespace fk {

#if !defined(__CUDA_ARCH__) && !defined(__NVCC__) && !CLANG_HOST_DEVICE
// CPU-only TU (e.g. g++ in CI): provide expf/fmaxf in fk:: scope.
// NOTE: do NOT `using std::expf` — libstdc++ only guarantees ::expf from
// <cmath>'s C heritage; std::expf is not declared on g++-13 (CI failure).
// gcc: constexpr via __builtin_expf (keeps FK_HOST_DEVICE_CNST callers
// legal). clang: __builtin_expf is not constexpr -> plain inline (clang
// accepts non-constexpr calls inside never-constant-evaluated constexpr
// functions without -Winvalid-constexpr only when the call is reachable,
// so the polynomial fallback keeps it constexpr-clean everywhere).
constexpr float fmaxf(const float a, const float b) {
    return a > b ? a : (b >= a ? b : (a == a ? a : b));  // NaN-safe enough
}
#if defined(__clang__)
inline float expf(const float x) { return __builtin_expf(x); }
#else
constexpr float expf(const float x) { return __builtin_expf(x); }
#endif
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

/* InIOp is an INSTANTIABLE Read or ReadBack IOp (possibly a fusion
 * read.then(compute...)): the prologue. Every element enters the
 * algorithm through InIOp::Operation::exec(thread, iop). */
template <typename InIOp, typename OT, int BLOCK_SIZE = 256>
struct SoftmaxDPP {
private:
    using SelfType = SoftmaxDPP<InIOp, OT, BLOCK_SIZE>;
public:
    FK_STATIC_STRUCT(SoftmaxDPP, SelfType)

    static_assert(isAnyReadType<InIOp>, "Softmax prologue must be a Read or ReadBack IOp");

    struct Params {
        InIOp input;             // prologue Read/ReadBack IOp
        RawPtr<ND::_2D, OT> output;
        int width;               // row length
    };

    FK_DEVICE_FUSE float readElem(const InIOp& iop, const int x, const int y) {
        return attnToF32(InIOp::Operation::exec(Point{ x, y, 0 }, iop));
    }

    FK_COOP_DEVICE_FUSE exec(const Params& p) {
        __shared__ OnlineSoftmaxState states[BLOCK_SIZE];

        const int row = blockIdx.x;
        const int tid = threadIdx.x;
        const int width = p.width;

        OT* out = PtrAccessor<ND::_2D>::point(Point{0, row, 0}, p.output);

        OnlineSoftmaxState st{ -FLT_MAX, 0.f };
        for (int x = tid; x < width; x += BLOCK_SIZE) {
            // PROLOGUE: element read through the IOp (pass 1)
            const float v = readElem(p.input, x, row);
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
            // PROLOGUE: element read through the IOp (pass 2)
            out[x] = attnFromF32<OT>(expf(readElem(p.input, x, row) - m) * invL);
        }
    }
};

template <typename InIOp, typename OT, int BLOCK_SIZE>
__global__ void launchSoftmaxDPP_Kernel(const __grid_constant__ typename SoftmaxDPP<InIOp, OT, BLOCK_SIZE>::Params params) {
    SoftmaxDPP<InIOp, OT, BLOCK_SIZE>::exec(params);
}

/* IOp-first API: input is the prologue Read/ReadBack IOp. */
template <int BLOCK_SIZE = 256, typename InIOp, typename OT>
inline void executeSoftmax(const InIOp& input, const Ptr2D<OT>& output,
                           const int rows, const int width,
                           Stream_<ParArch::GPU_NVIDIA>& stream) {
    using DPP = SoftmaxDPP<InIOp, OT, BLOCK_SIZE>;
    const typename DPP::Params params{ input, output.ptr(), width };
    const dim3 grid(rows, 1, 1);
    const dim3 block(BLOCK_SIZE, 1, 1);
    launchSoftmaxDPP_Kernel<InIOp, OT, BLOCK_SIZE><<<grid, block, 0, stream.getCUDAStream()>>>(params);
    gpuErrchk(cudaGetLastError());
}

/* Pointer convenience API (back-compat): canonical PerThreadRead prologue. */
template <typename T, int BLOCK_SIZE = 256>
inline void executeSoftmax(const Ptr2D<T>& input, const Ptr2D<T>& output,
                           Stream_<ParArch::GPU_NVIDIA>& stream) {
    const auto inIOp = PerThreadRead<ND::_2D, T>::build(input.ptr());
    executeSoftmax<BLOCK_SIZE>(inIOp, output,
                               static_cast<int>(input.dims().height),
                               static_cast<int>(input.dims().width), stream);
}

#endif // defined(__NVCC__) || CLANG_HOST_DEVICE

} // namespace fk

#endif // FK_ATTENTION_SOFTMAX_H
