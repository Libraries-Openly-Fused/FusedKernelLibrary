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
#include <fused_kernel/core/constexpr_libs/constexpr_cmath.h>

#include <cfloat>
#include <cmath>
#if defined(__NVCC__)
#include <cuda_fp16.h>
#endif

namespace fk {

// Cooperative-DPP exec bodies use __shared__ + barriers: they cannot be
// constexpr (FK_DEVICE_FUSE). Plain static device inline qualifier:
#define FK_COOP_DEVICE_FUSE static __device__ __forceinline__ void

    namespace low_precision {
    template <typename T>
    FK_DEVICE_CNST float attnToF32(const T &v) {
        if constexpr (std::is_same_v<T, __half>) {
            return __half2float(v);
        } else {
            return static_cast<float>(v);
        }
    }

    template <typename T>
    FK_DEVICE_CNST T attnFromF32(const float &v) {
        if constexpr (std::is_same_v<T, __half>) {
            return __float2half(v);
        } else {
            return static_cast<T>(v);
        }
    }
    } // namespace low_precision

template <typename IT>
struct CastLowFPToF32 {
  private:
    using SelfType = CastLowFPToF32<IT>;
    using Parent = UnaryOperation<IT, float, SelfType>;

  public:
    FK_STATIC_STRUCT(CastLowFPToF32, SelfType)
    DECLARE_UNARY_PARENT
    FK_DEVICE_FUSE float exec(const InputType &v) { return low_precision::attnToF32(v); }
};

template <typename OT> struct CastF32ToLowFP {
  private:
    using SelfType = CastF32ToLowFP<OT>;
    using Parent = UnaryOperation<float, OT, SelfType>;

  public:
    FK_STATIC_STRUCT(CastF32ToLowFP, SelfType)
    DECLARE_UNARY_PARENT
    FK_DEVICE_FUSE OT exec(const InputType &v) { return low_precision::attnFromF32<OT>(v); }
};

template <int BLOCK_SIZE_ = 256>
struct SoftmaxDPPDetails {
    static constexpr int BLOCK_SIZE = BLOCK_SIZE_;
};

/* InIOp is an INSTANTIABLE Read or ReadBack IOp (possibly a fusion
 * read.then(compute...)): the prologue. Every element enters the
 * algorithm through InIOp::Operation::exec(thread, iop). */
struct SoftmaxDPP {
private:
    using SelfType = SoftmaxDPP;
    struct OnlineSoftmaxState {
        float m; // running max
        float l; // running sum of exp(x - m)
    };

    FK_DEVICE_FUSE OnlineSoftmaxState mergeSoftmaxStates(const OnlineSoftmaxState& a, const OnlineSoftmaxState& b) {
        const float m = cxp::max::f(a.m, b.m);
        const float la = a.l == 0.f ? 0.f : a.l * cxp::expf::f(a.m - m);
        const float lb = b.l == 0.f ? 0.f : b.l * cxp::expf::f(b.m - m);
        return {m, la + lb};
    }

  public:
    FK_STATIC_STRUCT(SoftmaxDPP, SelfType)

    template <typename SoftmaxDetails, typename InIOp, typename OutIOp>
    FK_COOP_DEVICE_FUSE exec(const SoftmaxDetails& p, const InIOp& input, const OutIOp& output) {
        __shared__ OnlineSoftmaxState states[SoftmaxDetails::BLOCK_SIZE];

        const int row = blockIdx.x;
        const int tid = threadIdx.x;
        const int width = InIOp::Operation::num_elems_x(Point{0,0,0}, input);

        OnlineSoftmaxState st{ -FLT_MAX, 0.f };
        for (int x = tid; x < width; x += SoftmaxDetails::BLOCK_SIZE) {
            // PROLOGUE: element read through the IOp (pass 1)
            const float v = InIOp::Operation::exec(Point{ x, row, 0 }, input);
            const float m = cxp::max::f(st.m, v);
            st.l = st.l * cxp::expf::f(st.m - m) + cxp::expf::f(v - m);
            st.m = m;
        }
        states[tid] = st;
        __syncthreads();

        for (int stride = SoftmaxDetails::BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                states[tid] = mergeSoftmaxStates(states[tid], states[tid + stride]);
            }
            __syncthreads();
        }
        const float m = states[0].m;
        const float invL = 1.f / states[0].l;

        for (int x = tid; x < width; x += SoftmaxDetails::BLOCK_SIZE) {
            const float result = cxp::expf::f(InIOp::Operation::exec(Point{x, row, 0}, input) - m) * invL;
            OutIOp::Operation::exec(Point{x, row, 0}, result, output);
        }
    }
};

template <typename DPP, typename DPPDetails, typename InIOp, typename ComputeIOp, typename OutIOp>
__global__ void launchDPP_Kernel(const __grid_constant__ DPPDetails details,
                                        const __grid_constant__ InIOp input,
                                        const __grid_constant__ ComputeIOp compute,
                                        const __grid_constant__ OutIOp output) {
    if constexpr (std::is_same_v<ComputeIOp, NullType>) {
        DPP::exec(details, input, output);
    } else {
        DPP::exec(details, input, compute, output);
    }
}

/* IOp-first API: input is the prologue Read/ReadBack IOp. */
template <int BLOCK_SIZE, typename InIOp, typename OutIOp>
inline void executeSoftmax(const InIOp& input, const OutIOp& output,
                           Stream_<ParArch::GPU_NVIDIA>& stream) {
    const SoftmaxDPPDetails<BLOCK_SIZE> details{};
    const dim3 grid(InIOp::Operation::num_elems_y(Point{0,0,0}, input), 1, 1);
    const dim3 block(BLOCK_SIZE, 1, 1);
    launchDPP_Kernel<SoftmaxDPP><<<grid, block, 0, stream.getCUDAStream()>>>(details, input, NullType{}, output);
    gpuErrchk(cudaGetLastError());
}

} // namespace fk

#endif // FK_ATTENTION_SOFTMAX_H
