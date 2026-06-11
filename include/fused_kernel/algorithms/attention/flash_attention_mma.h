/* Copyright 2026 the Fused Kernel Library authors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_ATTENTION_FLASH_ATTENTION_MMA_H
#define FK_ATTENTION_FLASH_ATTENTION_MMA_H

/* FlashAttentionMmaDPP: FA-2 forward on mma.sync.m16n8k16 tensor cores
 * (sm_80+; tuned on sm_120 per gau-nernst's fa-5090 analysis: on consumer
 * Blackwell there is no TMEM/tcgen05, so the path to speed-of-light is
 * exactly this — ldmatrix + mma.sync + cp.async-style staging).
 *
 * THE FKL DIFFERENCE vs handmade FA kernels (incl. learn-cuda 07_attention
 * and the flash-attention repo): Q, K and V remain Read/ReadBack IOps.
 * The global->shared staging reads every element THROUGH the prologue IOp
 * (dequantization, scaling, casts... fuse in-register at load time), and
 * the epilogue IOp chain runs on the output registers before the single
 * global write. Handmade kernels hardcode raw bf16 pointers; here the
 * compressed int8 KV cache or any preprocessing rides the SAME kernel.
 *
 * Mapping (faithful port of learn-cuda v1, generalized):
 *   - bf16 tiles in smem (built via prologue, fp32 -> bf16 round)
 *   - ldmatrix.x4 / .x2 to registers, mma.sync m16n8k16 bf16->fp32
 *   - FA-2 warp split: each warp owns WARP_Q = BLOCK_Q/NUM_WARPS query
 *     rows; K/V tiles replicated across warps; online softmax on the
 *     fp32 S registers with butterfly reductions in each 4-thread group.
 *   - fp32 accumulation for O; epilogue + single write at the end.
 *   - Causal masking and ragged seq_q/seq_k handled with guards
 *     (staging pads with zeros; S positions beyond seq_k or above the
 *     diagonal are set to -inf before the rowmax).
 *
 * Requirements: HEAD_DIM % 16 == 0 (ldmatrix granularity).
 * Accuracy: bf16 inputs to the dot products (like every tensor-core FA);
 * expect ~1e-2 abs error on unit-range inputs vs an fp64 oracle — same
 * ballpark as flash-attention's own bf16 path. The SIMT DPP remains the
 * fp32-exact option. */

#include <fused_kernel/algorithms/attention/flash_attention.h>

#if (defined(__NVCC__) || CLANG_HOST_DEVICE)

#include <cuda_bf16.h>

namespace fk {

namespace attention_mma_detail {

__device__ __forceinline__ void ldmatrix_x4(uint32_t regs[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2(uint32_t regs[2], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                 : "=r"(regs[0]), "=r"(regs[1])
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t regs[2], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
                 : "=r"(regs[0]), "=r"(regs[1])
                 : "r"(addr));
}

__device__ __forceinline__ void mma_m16n8k16(const uint32_t A[4], const uint32_t B[2],
                                             float D[4]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                 : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
                 : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                   "r"(B[0]), "r"(B[1]),
                   "f"(D[0]), "f"(D[1]), "f"(D[2]), "f"(D[3]));
}

} // namespace attention_mma_detail

template <typename OT, int HEAD_DIM,
          typename QIOp, typename KIOp, typename VIOp,
          typename EpilogueIOp = AttentionIdentityEpilogue,
          int BLOCK_Q = 64, int BLOCK_KV = 64, int NUM_WARPS = 4>
struct FlashAttentionMmaDPP {
private:
    using SelfType = FlashAttentionMmaDPP<OT, HEAD_DIM, QIOp, KIOp, VIOp,
                                          EpilogueIOp, BLOCK_Q, BLOCK_KV, NUM_WARPS>;
public:
    FK_STATIC_STRUCT(FlashAttentionMmaDPP, SelfType)

    static_assert(HEAD_DIM % 16 == 0, "HEAD_DIM must be a multiple of 16 (ldmatrix)");
    static_assert(isAnyReadType<QIOp>, "Q prologue must be a Read or ReadBack IOp");
    static_assert(isAnyReadType<KIOp>, "K prologue must be a Read or ReadBack IOp");
    static_assert(isAnyReadType<VIOp>, "V prologue must be a Read or ReadBack IOp");

    static constexpr int THREADS = NUM_WARPS * 32;
    static constexpr int WARP_Q = BLOCK_Q / NUM_WARPS;        // rows per warp
    static_assert(WARP_Q % 16 == 0, "BLOCK_Q/NUM_WARPS must be a multiple of 16");
    static constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;
    static constexpr int SMEM_ELEMS =
        (BLOCK_Q > 2 * BLOCK_KV ? BLOCK_Q : 2 * BLOCK_KV) * HEAD_DIM;

    struct Params {
        QIOp q; KIOp k; VIOp v;
        OT* o;
        int seq_q, seq_k;
        float scale;
        bool causal;
        EpilogueIOp epilogue;
    };

    template <typename IOp>
    FK_DEVICE_FUSE float readElem(const IOp& iop, const int x, const int y, const int z) {
        // single entry point for ALL data: the prologue IOp's exec.
        return attnToF32(IOp::Operation::exec(Point{ x, y, z }, iop));
    }

    /* Stage a (rows x HEAD_DIM) tile into smem THROUGH a prologue IOp.
       Pads with zeros beyond seqLen. This is where FKL's fusion happens:
       int8 dequant / scaling / any .then chain runs here, in-register. */
    template <typename IOp>
    FK_DEVICE_FUSE void stageTile(__nv_bfloat16* smem, const IOp& iop,
                                  const int rowBase, const int rows,
                                  const int seqLen, const int bh) {
        for (int idx = threadIdx.x; idx < rows * HEAD_DIM; idx += THREADS) {
            const int r = idx / HEAD_DIM;
            const int c = idx % HEAD_DIM;
            const int row = rowBase + r;
            const float val = (row < seqLen) ? readElem(iop, c, row, bh) : 0.f;
            smem[r * HEAD_DIM + c] = __float2bfloat16(val);
        }
    }

    static __device__ void exec(const Params& p) {
        using namespace attention_mma_detail;
        extern __shared__ __nv_bfloat16 smem[];
        __nv_bfloat16* qSmem = smem;                 // BLOCK_Q rows (phase 1)
        __nv_bfloat16* kSmem = smem;                 // BLOCK_KV rows (reused)
        __nv_bfloat16* vSmem = smem + BLOCK_KV * HEAD_DIM;

        const int tid = threadIdx.x;
        const int warpId = tid / 32;
        const int laneId = tid % 32;
        const int bh = blockIdx.y;
        const int qBlockBase = blockIdx.x * BLOCK_Q;

        // ---- phase 1: Q -> smem (through prologue) -> registers ----------
        stageTile(qSmem, p.q, qBlockBase, BLOCK_Q, p.seq_q, bh);
        __syncthreads();

        uint32_t qReg[WARP_Q / MMA_M][HEAD_DIM / MMA_K][4];
        const uint32_t qBaseAddr = static_cast<uint32_t>(__cvta_generic_to_shared(qSmem));
        #pragma unroll
        for (int mq = 0; mq < WARP_Q / MMA_M; ++mq) {
            #pragma unroll
            for (int md = 0; md < HEAD_DIM / MMA_K; ++md) {
                const int row = warpId * WARP_Q + mq * MMA_M + (laneId % 16);
                const int col = md * MMA_K + (laneId / 16) * 8;
                ldmatrix_x4(qReg[mq][md],
                            qBaseAddr + (row * HEAD_DIM + col) * sizeof(__nv_bfloat16));
            }
        }
        __syncthreads();  // done reading qSmem before K overwrites it

        float oAcc[WARP_Q / MMA_M][HEAD_DIM / MMA_N][4] = {};
        float rowMax[WARP_Q / MMA_M][2];
        float rowSum[WARP_Q / MMA_M][2] = {};
        #pragma unroll
        for (int mq = 0; mq < WARP_Q / MMA_M; ++mq) {
            rowMax[mq][0] = -FLT_MAX;
            rowMax[mq][1] = -FLT_MAX;
        }

        const uint32_t kBaseAddr = static_cast<uint32_t>(__cvta_generic_to_shared(kSmem));
        const uint32_t vBaseAddr = static_cast<uint32_t>(__cvta_generic_to_shared(vSmem));

        // causal: rows in this block end at qBlockBase + BLOCK_Q - 1
        const int kvEnd = p.causal ? ::min(p.seq_k, qBlockBase + BLOCK_Q) : p.seq_k;

        for (int offKV = 0; offKV < kvEnd; offKV += BLOCK_KV) {
            // ---- K and V tiles, staged through their prologue IOps -------
            stageTile(kSmem, p.k, offKV, BLOCK_KV, p.seq_k, bh);
            stageTile(vSmem, p.v, offKV, BLOCK_KV, p.seq_k, bh);
            __syncthreads();

            uint32_t kReg[BLOCK_KV / MMA_N][HEAD_DIM / MMA_K][2];
            #pragma unroll
            for (int mkv = 0; mkv < BLOCK_KV / MMA_N; ++mkv) {
                #pragma unroll
                for (int md = 0; md < HEAD_DIM / MMA_K; ++md) {
                    const int row = mkv * MMA_N + (laneId % 8);
                    const int col = md * MMA_K + (laneId / 8) * 8;
                    ldmatrix_x2(kReg[mkv][md],
                                kBaseAddr + (row * HEAD_DIM + col) * sizeof(__nv_bfloat16));
                }
            }

            // ---- S = scale * Q K^T (fp32 accum) ---------------------------
            float sReg[WARP_Q / MMA_M][BLOCK_KV / MMA_N][4] = {};
            #pragma unroll
            for (int mq = 0; mq < WARP_Q / MMA_M; ++mq)
                #pragma unroll
                for (int mkv = 0; mkv < BLOCK_KV / MMA_N; ++mkv)
                    #pragma unroll
                    for (int md = 0; md < HEAD_DIM / MMA_K; ++md)
                        mma_m16n8k16(qReg[mq][md], kReg[mkv][md], sReg[mq][mkv]);

            uint32_t pReg[WARP_Q / MMA_M][BLOCK_KV / MMA_K][4];

            #pragma unroll
            for (int mq = 0; mq < WARP_Q / MMA_M; ++mq) {
                // global row indices of the two row-halves this thread holds
                const int rowA = qBlockBase + warpId * WARP_Q + mq * MMA_M + laneId / 4;
                const int rowB = rowA + 8;

                // scale + bounds/causal mask BEFORE rowmax
                #pragma unroll
                for (int mkv = 0; mkv < BLOCK_KV / MMA_N; ++mkv) {
                    float* r = sReg[mq][mkv];
                    const int colBase = offKV + mkv * MMA_N + (laneId % 4) * 2;
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const int col = colBase + (e & 1);
                        const int row = (e < 2) ? rowA : rowB;
                        const bool dead = (col >= p.seq_k) || (p.causal && col > row);
                        r[e] = dead ? -FLT_MAX : r[e] * p.scale;
                    }
                }

                // online softmax on fp32 registers (butterfly over 4 lanes)
                float mNew[2] = { -FLT_MAX, -FLT_MAX };
                #pragma unroll
                for (int mkv = 0; mkv < BLOCK_KV / MMA_N; ++mkv) {
                    const float* r = sReg[mq][mkv];
                    mNew[0] = fmaxf(mNew[0], fmaxf(r[0], r[1]));
                    mNew[1] = fmaxf(mNew[1], fmaxf(r[2], r[3]));
                }
                #pragma unroll
                for (int half = 0; half < 2; ++half) {
                    mNew[half] = fmaxf(mNew[half], __shfl_xor_sync(0xFFFFFFFFu, mNew[half], 1));
                    mNew[half] = fmaxf(mNew[half], __shfl_xor_sync(0xFFFFFFFFu, mNew[half], 2));
                    mNew[half] = fmaxf(mNew[half], rowMax[mq][half]);
                }

                float corr[2];
                corr[0] = __expf(rowMax[mq][0] - mNew[0]);
                corr[1] = __expf(rowMax[mq][1] - mNew[1]);
                #pragma unroll
                for (int md = 0; md < HEAD_DIM / MMA_N; ++md) {
                    oAcc[mq][md][0] *= corr[0];
                    oAcc[mq][md][1] *= corr[0];
                    oAcc[mq][md][2] *= corr[1];
                    oAcc[mq][md][3] *= corr[1];
                }
                rowMax[mq][0] = mNew[0];
                rowMax[mq][1] = mNew[1];

                float lNew[2] = {};
                #pragma unroll
                for (int mkv = 0; mkv < BLOCK_KV / MMA_N; ++mkv) {
                    float* r = sReg[mq][mkv];
                    r[0] = (r[0] == -FLT_MAX) ? 0.f : __expf(r[0] - mNew[0]);
                    r[1] = (r[1] == -FLT_MAX) ? 0.f : __expf(r[1] - mNew[0]);
                    r[2] = (r[2] == -FLT_MAX) ? 0.f : __expf(r[2] - mNew[1]);
                    r[3] = (r[3] == -FLT_MAX) ? 0.f : __expf(r[3] - mNew[1]);
                    lNew[0] += r[0] + r[1];
                    lNew[1] += r[2] + r[3];

                    // repack P fp32 -> bf16x2 registers for the P @ V mma
                    __nv_bfloat162* pp =
                        reinterpret_cast<__nv_bfloat162*>(pReg[mq][mkv / 2]);
                    pp[(mkv % 2) * 2]     = __float22bfloat162_rn({ r[0], r[1] });
                    pp[(mkv % 2) * 2 + 1] = __float22bfloat162_rn({ r[2], r[3] });
                }
                #pragma unroll
                for (int half = 0; half < 2; ++half) {
                    lNew[half] += __shfl_xor_sync(0xFFFFFFFFu, lNew[half], 1);
                    lNew[half] += __shfl_xor_sync(0xFFFFFFFFu, lNew[half], 2);
                    rowSum[mq][half] = rowSum[mq][half] * corr[half] + lNew[half];
                }
            }

            // ---- V tile -> registers (transposed), O += P V ---------------
            uint32_t vReg[BLOCK_KV / MMA_K][HEAD_DIM / MMA_N][2];
            #pragma unroll
            for (int mkv = 0; mkv < BLOCK_KV / MMA_K; ++mkv) {
                #pragma unroll
                for (int md = 0; md < HEAD_DIM / MMA_N; ++md) {
                    const int row = mkv * MMA_K + (laneId % 16);
                    const int col = md * MMA_N + (laneId / 16) * 8;
                    ldmatrix_x2_trans(vReg[mkv][md],
                                      vBaseAddr + (row * HEAD_DIM + col) * sizeof(__nv_bfloat16));
                }
            }
            #pragma unroll
            for (int mq = 0; mq < WARP_Q / MMA_M; ++mq)
                #pragma unroll
                for (int md = 0; md < HEAD_DIM / MMA_N; ++md)
                    #pragma unroll
                    for (int mkv = 0; mkv < BLOCK_KV / MMA_K; ++mkv)
                        mma_m16n8k16(pReg[mq][mkv], vReg[mkv][md], oAcc[mq][md]);

            __syncthreads();  // protect smem before next tile staging
        }

        // ---- normalize, EPILOGUE FUSION, single global write --------------
        #pragma unroll
        for (int mq = 0; mq < WARP_Q / MMA_M; ++mq) {
            const int rowA = qBlockBase + warpId * WARP_Q + mq * MMA_M + laneId / 4;
            const int rowB = rowA + 8;
            const float invA = rowSum[mq][0] > 0.f ? 1.f / rowSum[mq][0] : 0.f;
            const float invB = rowSum[mq][1] > 0.f ? 1.f / rowSum[mq][1] : 0.f;
            #pragma unroll
            for (int md = 0; md < HEAD_DIM / MMA_N; ++md) {
                const int col = md * MMA_N + (laneId % 4) * 2;
                const float* r = oAcc[mq][md];
                if (rowA < p.seq_q) {
                    const long base = ((long)bh * p.seq_q + rowA) * HEAD_DIM + col;
                    p.o[base]     = attnFromF32<OT>((r[0] * invA) | p.epilogue);
                    p.o[base + 1] = attnFromF32<OT>((r[1] * invA) | p.epilogue);
                }
                if (rowB < p.seq_q) {
                    const long base = ((long)bh * p.seq_q + rowB) * HEAD_DIM + col;
                    p.o[base]     = attnFromF32<OT>((r[2] * invB) | p.epilogue);
                    p.o[base + 1] = attnFromF32<OT>((r[3] * invB) | p.epilogue);
                }
            }
        }
    }
};

template <typename OT, int HEAD_DIM, typename QIOp, typename KIOp, typename VIOp,
          typename EpilogueIOp, int BLOCK_Q, int BLOCK_KV, int NUM_WARPS>
__global__ void launchFlashAttentionMmaDPP_Kernel(
        const __grid_constant__ typename FlashAttentionMmaDPP<
            OT, HEAD_DIM, QIOp, KIOp, VIOp, EpilogueIOp,
            BLOCK_Q, BLOCK_KV, NUM_WARPS>::Params params) {
    FlashAttentionMmaDPP<OT, HEAD_DIM, QIOp, KIOp, VIOp, EpilogueIOp,
                         BLOCK_Q, BLOCK_KV, NUM_WARPS>::exec(params);
}

/* IOp-first API, mirroring executeFlashAttention. Tensor-core path. */
template <int HEAD_DIM, int BLOCK_Q = 64, int BLOCK_KV = 64, int NUM_WARPS = 4,
          typename OT = float, typename QIOp, typename KIOp, typename VIOp,
          typename EpilogueIOp = AttentionIdentityEpilogue>
inline void executeFlashAttentionMma(
        const QIOp& q, const KIOp& k, const VIOp& v, OT* o,
        const int batchHeads, const int seqQ, const int seqK,
        const bool causal, Stream_<ParArch::GPU_NVIDIA>& stream,
        const float scaleOverride = -1.f, const EpilogueIOp& epilogue = {}) {
    using DPP = FlashAttentionMmaDPP<OT, HEAD_DIM, QIOp, KIOp, VIOp,
                                     EpilogueIOp, BLOCK_Q, BLOCK_KV, NUM_WARPS>;
    const float scale = scaleOverride > 0.f ? scaleOverride
                                            : rsqrtf(static_cast<float>(HEAD_DIM));
    const typename DPP::Params params{ q, k, v, o, seqQ, seqK, scale, causal, epilogue };
    const dim3 grid((seqQ + BLOCK_Q - 1) / BLOCK_Q, batchHeads, 1);
    const dim3 block(DPP::THREADS, 1, 1);
    const int smemBytes = DPP::SMEM_ELEMS * sizeof(__nv_bfloat16);
    auto* kernel = launchFlashAttentionMmaDPP_Kernel<OT, HEAD_DIM, QIOp, KIOp, VIOp,
                                                     EpilogueIOp, BLOCK_Q, BLOCK_KV, NUM_WARPS>;
    if (smemBytes > 48000) {
        gpuErrchk(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       smemBytes));
    }
    kernel<<<grid, block, smemBytes, stream.getCUDAStream()>>>(params);
    gpuErrchk(cudaGetLastError());
}

} // namespace fk

#endif // defined(__NVCC__) || CLANG_HOST_DEVICE

#endif // FK_ATTENTION_FLASH_ATTENTION_MMA_H
