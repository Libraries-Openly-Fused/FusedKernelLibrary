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
#include <fused_kernel/fused_kernel.h>

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

/* The row softmax: one block of DPPDetails::BLOCK_SIZE threads per row of the input. InIOp is an INSTANTIABLE Read
 * or ReadBack IOp (possibly a fusion read.then(compute...)): the prologue. Every element enters the algorithm through
 * InIOp::Operation::exec(thread, iop), and every result leaves through the OutIOp (a Write IOp, possibly
 * compute.then(write)). GPU only, as the other attention DPPs. */
template <ParArch PA, typename DPPDetails = SoftmaxDPPDetails<256>>
struct SoftmaxDPP;

#if defined(__NVCC__)
template <typename DPPDetails>
struct SoftmaxDPP<ParArch::GPU_NVIDIA, DPPDetails> {
private:
    using SelfType = SoftmaxDPP<ParArch::GPU_NVIDIA, DPPDetails>;
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
    static constexpr ParArch PAR_ARCH = ParArch::GPU_NVIDIA;

    template <typename InIOp, typename OutIOp>
    FK_COOP_DEVICE_FUSE exec(const DPPDetails& p, const InIOp& input, const OutIOp& output) {
        __shared__ OnlineSoftmaxState states[DPPDetails::BLOCK_SIZE];

        const int row = blockIdx.x;
        const int tid = threadIdx.x;
        const int width = InIOp::Operation::num_elems_x(Point{0,0,0}, input);

        OnlineSoftmaxState st{ -FLT_MAX, 0.f };
        for (int x = tid; x < width; x += DPPDetails::BLOCK_SIZE) {
            // PROLOGUE: element read through the IOp (pass 1)
            const float v = InIOp::Operation::exec(Point{ x, row, 0 }, input);
            const float m = cxp::max::f(st.m, v);
            st.l = st.l * cxp::expf::f(st.m - m) + cxp::expf::f(v - m);
            st.m = m;
        }
        states[tid] = st;
        __syncthreads();

        for (int stride = DPPDetails::BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                states[tid] = mergeSoftmaxStates(states[tid], states[tid + stride]);
            }
            __syncthreads();
        }
        const float m = states[0].m;
        const float invL = 1.f / states[0].l;

        for (int x = tid; x < width; x += DPPDetails::BLOCK_SIZE) {
            const float result = cxp::expf::f(InIOp::Operation::exec(Point{x, row, 0}, input) - m) * invL;
            OutIOp::Operation::exec(Point{x, row, 0}, result, output);
        }
    }
};

// The executor of SoftmaxDPP:
//   executeOperations<SoftmaxDPP<ParArch::GPU_NVIDIA, SoftmaxDPPDetails<BLOCK_SIZE>>>(stream, details, inIOp, outIOp)
// One block of BLOCK_SIZE threads per row: the rows are InIOp's num_elems_y, the row length its num_elems_x. It
// launches launchDPP_Kernel<SoftmaxDPP<...>, DPPDetails, InIOp, OutIOp> (Oscar's decision, 2026-11-26).
template <typename DPPDetails>
struct Executor<SoftmaxDPP<ParArch::GPU_NVIDIA, DPPDetails>> {
  private:
    using DPP = SoftmaxDPP<ParArch::GPU_NVIDIA, DPPDetails>;
    using SelfType = Executor<DPP>;

  public:
    FK_STATIC_STRUCT(Executor, SelfType)
    FK_HOST_FUSE ParArch parArch() { return ParArch::GPU_NVIDIA; }
    template <typename InIOp, typename OutIOp>
    FK_HOST_STATIC void executeOperations(Stream_<ParArch::GPU_NVIDIA>& stream, const DPPDetails& details,
                                          const InIOp& input, const OutIOp& output) {
        const dim3 grid(InIOp::Operation::num_elems_y(Point{0, 0, 0}, input), 1, 1);
        const dim3 block(DPPDetails::BLOCK_SIZE, 1, 1);
        launchDPP_Kernel<DPP><<<grid, block, 0, stream.getCUDAStream()>>>(details, input, output);
        gpuErrchk(cudaGetLastError());
    }
};
#endif // defined(__NVCC__)

} // namespace fk

#endif // FK_ATTENTION_SOFTMAX_H
