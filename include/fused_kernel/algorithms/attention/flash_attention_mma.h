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
 * (sm_80+; tuned on sm_120 per gau-nernst's fa-5090 ladder: on consumer
 * Blackwell there is no TMEM/tcgen05 — see also Dao-AILab/flash-attention
 * PR #2634 bringing FA to sm12x the same way).
 *
 * Optimization ladder applied (fa-5090 v1 -> v4, adapted to FKL):
 *   v1  mma.sync m16n8k16 bf16->fp32, FA-2 warp split, online softmax
 *       on fp32 S registers with butterfly reductions.
 *   v2  XOR-swizzled smem so ldmatrix and the 16-byte staging stores are
 *       bank-conflict free.
 *   v3  software pipelining: K/V double-buffered in smem; the NEXT tile's
 *       global reads are issued (through the prologue IOps, into
 *       registers) BEFORE the current tile's math, so load latency
 *       overlaps the mma work. This is the FKL-compatible counterpart of
 *       cp.async 2-stage pipelining: cp.async copies raw bytes and cannot
 *       run an IOp chain per element; prefetch-to-register can.
 *   v4  ldmatrix.x4 for K and V fragment loads.
 *
 * THE FKL DIFFERENCE vs handmade FA kernels: Q, K and V remain
 * Read/ReadBack IOps. Staging reads every element THROUGH the prologue
 * (int8 dequant, scaling, casts, any .then chain fuse in-register at load
 * time), and the epilogue IOp chain runs on the output registers before
 * the single global write. The compressed int8 KV cache or any
 * preprocessing rides the SAME kernel — flash-attention cannot do that.
 *
 * Layout: (batch*heads, seq, head_dim); HEAD_DIM % 32 == 0; causal,
 * ragged and cross seq lens handled with guards (staging pads with
 * zeros; dead S positions get -inf before the rowmax).
 *
 * Accuracy: bf16 inputs to the dot products (like every tensor-core FA);
 * ~1e-2 abs error on unit-range inputs vs an fp64 oracle — same class as
 * flash-attention's own bf16 path. The SIMT DPP remains the fp32-exact
 * option. */

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

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t regs[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
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

// 8 bf16 packed for one 16-byte swizzled smem store.
struct alignas(16) Bf16x8 { __nv_bfloat162 h[4]; };

} // namespace attention_mma_detail

template <typename OT, int HEAD_DIM,
          typename QIOp, typename KIOp, typename VIOp,
          typename EpilogueIOp = AttentionIdentityEpilogue,
          int BLOCK_Q = (HEAD_DIM <= 64 ? 128 : 64), int BLOCK_KV = 32, int NUM_WARPS = 4>
struct FlashAttentionMmaDPP {
private:
    using SelfType = FlashAttentionMmaDPP<OT, HEAD_DIM, QIOp, KIOp, VIOp,
                                          EpilogueIOp, BLOCK_Q, BLOCK_KV, NUM_WARPS>;
public:
    FK_STATIC_STRUCT(FlashAttentionMmaDPP, SelfType)

    static_assert(HEAD_DIM % 32 == 0, "HEAD_DIM must be a multiple of 32");
    static_assert(isAnyReadType<QIOp>, "Q prologue must be a Read or ReadBack IOp");
    static_assert(isAnyReadType<KIOp>, "K prologue must be a Read or ReadBack IOp");
    static_assert(isAnyReadType<VIOp>, "V prologue must be a Read or ReadBack IOp");

    static constexpr int THREADS = NUM_WARPS * 32;
    static constexpr int WARP_Q = BLOCK_Q / NUM_WARPS;        // rows per warp
    static_assert(WARP_Q % 16 == 0, "BLOCK_Q/NUM_WARPS must be a multiple of 16");
    static constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;

    static constexpr int STRIDE_B = HEAD_DIM * (int)sizeof(__nv_bfloat16);
    static constexpr int Q_GROUPS = BLOCK_Q * HEAD_DIM / (THREADS * 8);
    static constexpr int KV_GROUPS = BLOCK_KV * HEAD_DIM / (THREADS * 8);
    static_assert(Q_GROUPS >= 1 && BLOCK_Q * HEAD_DIM % (THREADS * 8) == 0,
                  "BLOCK_Q*HEAD_DIM must be a multiple of THREADS*8");
    static_assert(KV_GROUPS >= 1 && BLOCK_KV * HEAD_DIM % (THREADS * 8) == 0,
                  "BLOCK_KV*HEAD_DIM must be a multiple of THREADS*8");

    // Q buffer overlaps the K/V double buffers (Q is consumed into
    // registers before the first K/V tile is staged).
    static constexpr int KV_BUF_B = BLOCK_KV * STRIDE_B;       // one buffer, bytes
    static constexpr int SMEM_BYTES =
        (BLOCK_Q * STRIDE_B > 4 * KV_BUF_B ? BLOCK_Q * STRIDE_B : 4 * KV_BUF_B);

    struct Params {
        QIOp q; KIOp k; VIOp v;
        OT* o;
        int seq_q, seq_k;
        float scale;
        bool causal;
        EpilogueIOp epilogue;
    };

    /* XOR swizzle on a byte offset relative to a tile buffer (buffers are
       KV_BUF_B-aligned, i.e. multiples of 8*STRIDE_B). Same scheme as
       fa-5090 / CUTLASS: kills bank conflicts for both the 16B staging
       stores and the ldmatrix loads. */
    FK_DEVICE_FUSE uint32_t swz(const uint32_t byteOff) {
        const uint32_t row = (byteOff / STRIDE_B) % 8;
        constexpr uint32_t div = (64 / STRIDE_B > 1 ? 64 / STRIDE_B : 1);
        return byteOff ^ ((row / div) << 4);
    }

    template <typename IOp>
    FK_DEVICE_FUSE float readElem(const IOp& iop, const int x, const int y, const int z) {
        // single entry point for ALL data: the prologue IOp's exec.
        return attnToF32(IOp::Operation::exec(Point{ x, y, z }, iop));
    }

    /* v3 pipelining, FKL style: issue the global reads for a tile through
       the prologue IOp into registers. Called BEFORE the current tile's
       math so the loads overlap the mma work (the data is only consumed
       by storeTile afterwards). */
    template <int GROUPS, typename IOp>
    FK_DEVICE_FUSE void prefetchTile(const IOp& iop, const int rowBase,
                                     const int seqLen, const int bh,
                                     float (&regs)[GROUPS][8]) {
        #pragma unroll
        for (int g = 0; g < GROUPS; ++g) {
            const int idx = (g * THREADS + (int)threadIdx.x) * 8;
            const int r = idx / HEAD_DIM;
            const int c = idx % HEAD_DIM;
            const int row = rowBase + r;
            if (row < seqLen) {
                #pragma unroll
                for (int e = 0; e < 8; ++e)
                    regs[g][e] = readElem(iop, c + e, row, bh);
            } else {
                #pragma unroll
                for (int e = 0; e < 8; ++e) regs[g][e] = 0.f;
            }
        }
    }

    // fp32 regs -> bf16x8 -> ONE 16-byte store per group, swizzled.
    template <int GROUPS>
    FK_DEVICE_FUSE void storeTile(char* smemBytes, const uint32_t bufOff,
                                  const float (&regs)[GROUPS][8]) {
        using attention_mma_detail::Bf16x8;
        #pragma unroll
        for (int g = 0; g < GROUPS; ++g) {
            const uint32_t idx = (g * THREADS + threadIdx.x) * 8;
            Bf16x8 packed;
            #pragma unroll
            for (int h = 0; h < 4; ++h)
                packed.h[h] = __float22bfloat162_rn({ regs[g][2 * h],
                                                      regs[g][2 * h + 1] });
            *reinterpret_cast<Bf16x8*>(
                smemBytes + bufOff + swz(idx * (uint32_t)sizeof(__nv_bfloat16))) = packed;
        }
    }

    static __device__ void exec(const Params& p) {
        using namespace attention_mma_detail;
        extern __shared__ char smemRaw[];
        char* smemBytes = smemRaw;
        const uint32_t smemBase =
            static_cast<uint32_t>(__cvta_generic_to_shared(smemRaw));

        const int tid = threadIdx.x;
        const int warpId = tid / 32;
        const int laneId = tid % 32;
        const int bh = blockIdx.y;
        const int qBlockBase = blockIdx.x * BLOCK_Q;

        // buffer byte offsets (Q overlaps K buffers; all KV_BUF_B-aligned)
        const auto kBuf = [](int i) -> uint32_t { return i * KV_BUF_B; };
        const auto vBuf = [](int i) -> uint32_t { return (2 + i) * KV_BUF_B; };

        // pre-swizzled per-thread ldmatrix base offsets (v2)
        // A-tile (Q): rows lane%16, col halves lane/16
        const uint32_t qThread = swz(((warpId * WARP_Q + laneId % 16) * HEAD_DIM
                                      + (laneId / 16) * 8) * sizeof(__nv_bfloat16));
        // B-tile (K, x4): rows lane%8, 4 col-chunks lane/8
        const uint32_t kThread = swz(((laneId % 8) * HEAD_DIM
                                      + (laneId / 8) * 8) * sizeof(__nv_bfloat16));
        // B-tile trans (V, x4): rows lane%16, col halves lane/16
        const uint32_t vThread = swz(((laneId % 16) * HEAD_DIM
                                      + (laneId / 16) * 8) * sizeof(__nv_bfloat16));

        // ---- phase 1: Q -> smem (through prologue, swizzled) -> registers
        {
            float qStage[Q_GROUPS][8];
            prefetchTile<Q_GROUPS>(p.q, qBlockBase, p.seq_q, bh, qStage);
            storeTile<Q_GROUPS>(smemBytes, 0, qStage);
        }
        __syncthreads();

        uint32_t qReg[WARP_Q / MMA_M][HEAD_DIM / MMA_K][4];
        #pragma unroll
        for (int mq = 0; mq < WARP_Q / MMA_M; ++mq) {
            #pragma unroll
            for (int md = 0; md < HEAD_DIM / MMA_K; ++md) {
                uint32_t addr = smemBase + qThread;
                addr += mq * MMA_M * STRIDE_B;                       // row step
                addr ^= md * MMA_K * sizeof(__nv_bfloat16);          // col step
                ldmatrix_x4(qReg[mq][md], addr);
            }
        }
        __syncthreads();  // done reading Q smem before K tile 0 overwrites it

        float oAcc[WARP_Q / MMA_M][HEAD_DIM / MMA_N][4] = {};
        float rowMax[WARP_Q / MMA_M][2];
        float rowSum[WARP_Q / MMA_M][2] = {};
        #pragma unroll
        for (int mq = 0; mq < WARP_Q / MMA_M; ++mq) {
            rowMax[mq][0] = -FLT_MAX;
            rowMax[mq][1] = -FLT_MAX;
        }

        const int kvEnd = p.causal ? ::min(p.seq_k, qBlockBase + BLOCK_Q) : p.seq_k;
        const int numIter = (kvEnd + BLOCK_KV - 1) / BLOCK_KV;

        float kPre[KV_GROUPS][8], vPre[KV_GROUPS][8];

        // stage tile 0
        prefetchTile<KV_GROUPS>(p.k, 0, p.seq_k, bh, kPre);
        prefetchTile<KV_GROUPS>(p.v, 0, p.seq_k, bh, vPre);
        storeTile<KV_GROUPS>(smemBytes, kBuf(0), kPre);
        storeTile<KV_GROUPS>(smemBytes, vBuf(0), vPre);
        __syncthreads();

        for (int kv = 0; kv < numIter; ++kv) {
            const int offKV = kv * BLOCK_KV;
            const int cur = kv % 2;
            const int nxt = (kv + 1) % 2;

            // ---- v3: issue NEXT tile's prologue reads now; they overlap
            // the mma math below and land in smem just before the sync.
            const bool havePrefetch = (kv + 1) < numIter;
            if (havePrefetch) {
                prefetchTile<KV_GROUPS>(p.k, offKV + BLOCK_KV, p.seq_k, bh, kPre);
                prefetchTile<KV_GROUPS>(p.v, offKV + BLOCK_KV, p.seq_k, bh, vPre);
            }

            // ---- K fragments (v4: ldmatrix.x4 covers two MMA_K chunks) ---
            uint32_t kReg[BLOCK_KV / MMA_N][HEAD_DIM / MMA_K][2];
            #pragma unroll
            for (int mkv = 0; mkv < BLOCK_KV / MMA_N; ++mkv) {
                #pragma unroll
                for (int md = 0; md < HEAD_DIM / MMA_K; md += 2) {
                    uint32_t addr = smemBase + kBuf(cur) + kThread;
                    addr += mkv * MMA_N * STRIDE_B;
                    addr ^= md * MMA_K * sizeof(__nv_bfloat16);
                    ldmatrix_x4(&kReg[mkv][md][0], addr);
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

            // ---- V fragments (x4 trans), O += P V -------------------------
            uint32_t vReg[BLOCK_KV / MMA_K][HEAD_DIM / MMA_N][2];
            #pragma unroll
            for (int mkv = 0; mkv < BLOCK_KV / MMA_K; ++mkv) {
                #pragma unroll
                for (int md = 0; md < HEAD_DIM / MMA_N; md += 2) {
                    uint32_t addr = smemBase + vBuf(cur) + vThread;
                    addr += mkv * MMA_K * STRIDE_B;
                    addr ^= md * MMA_N * sizeof(__nv_bfloat16);
                    ldmatrix_x4_trans(&vReg[mkv][md][0], addr);
                }
            }
            #pragma unroll
            for (int mq = 0; mq < WARP_Q / MMA_M; ++mq)
                #pragma unroll
                for (int md = 0; md < HEAD_DIM / MMA_N; ++md)
                    #pragma unroll
                    for (int mkv = 0; mkv < BLOCK_KV / MMA_K; ++mkv)
                        mma_m16n8k16(pReg[mq][mkv], vReg[mkv][md], oAcc[mq][md]);

            // ---- land the prefetched tile in the other buffer -------------
            if (havePrefetch) {
                storeTile<KV_GROUPS>(smemBytes, kBuf(nxt), kPre);
                storeTile<KV_GROUPS>(smemBytes, vBuf(nxt), vPre);
            }
            __syncthreads();
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
template <int HEAD_DIM, int BLOCK_Q = (HEAD_DIM <= 64 ? 128 : 64), int BLOCK_KV = 32, int NUM_WARPS = 4,
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
    const int smemBytes = DPP::SMEM_BYTES;
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
