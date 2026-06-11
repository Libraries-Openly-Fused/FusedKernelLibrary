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

#ifndef FK_ATTENTION_FLASH_ATTENTION_H
#define FK_ATTENTION_FLASH_ATTENTION_H

/* FlashAttentionDPP: FlashAttention-2 forward as ONE cooperative DPP
 * (one kernel), with two FKL-only superpowers that handmade FA kernels
 * (flash-attention repo and friends) cannot offer:
 *
 *  1. EPILOGUE FUSION: any FKL compute IOp chain (Mul, Add, Cast,
 *     Saturate, ... composed with .then()) is applied IN-REGISTER to the
 *     attention output before the single global write. out = (o/l) | chain.
 *     With handmade kernels that is always a second kernel + a DRAM
 *     round-trip; here it is zero extra cost and type-checked by FKL.
 *
 *  2. COMPRESSED KV CACHE (KVLayout::INT8_PER_TOKEN): K and V live in
 *     global memory as int8 with one fp32 scale per token:
 *       k_int8[t][d] = round(k[t][d]/kScale[t]), kScale[t] = max|k[t]|/127
 *     -> 4x smaller cache vs fp32 (2x vs fp16) + 2 floats/token.
 *     Dequantization is a fused prologue: it happens in-register inside
 *     the q.k dot product and the p*v accumulation. The compressed cache
 *     is NEVER inflated in global memory.
 *
 * Algorithm (Dao, FlashAttention-2): never materialize S = QK^T. Each
 * query row keeps a running (m, l, o); every KV position updates them with
 * the online-softmax rescaling trick. One pass over KV, O(seq) memory,
 * exact (fp32 accumulation).
 *
 * Mapping (SM 12x friendly — no TMEM/tcgen05, per the fa-5090 analysis,
 * correctness-first SIMT baseline before mma.sync tiling):
 *   grid.y = batch*heads; grid.x = ceil(seq_q / WARPS_PER_BLOCK)
 *   1 warp per query row; q and o live in registers spread across lanes
 *   (HEAD_DIM/32 each); dot products warp-reduce via __shfl_xor_sync;
 *   K/V tiles staged cooperatively in shared memory.
 */

#include <fused_kernel/algorithms/attention/softmax.h>

namespace fk {

enum class KVLayout { DENSE, INT8_PER_TOKEN };

#if defined(__NVCC__) || CLANG_HOST_DEVICE

// Identity epilogue: applied when no IOp chain is given.
struct AttentionIdentityEpilogue {
    FK_HOST_DEVICE_CNST friend float operator|(const float v,
                                               const AttentionIdentityEpilogue&) {
        return v;
    }
};

template <typename T, int HEAD_DIM, KVLayout KVL = KVLayout::DENSE,
          typename EpilogueIOp = AttentionIdentityEpilogue,
          int BLOCK_N = (sizeof(T) == 2 ? 64 : 32), int WARPS_PER_BLOCK = 4>
struct FlashAttentionDPP {
private:
    using SelfType = FlashAttentionDPP<T, HEAD_DIM, KVL, EpilogueIOp, BLOCK_N, WARPS_PER_BLOCK>;
public:
    FK_STATIC_STRUCT(FlashAttentionDPP, SelfType)

    static_assert(HEAD_DIM % 32 == 0, "HEAD_DIM must be a multiple of 32");
    static constexpr int ELEMS_PER_LANE = HEAD_DIM / 32;
    static constexpr int THREADS = WARPS_PER_BLOCK * 32;
    static constexpr bool QUANT = (KVL == KVLayout::INT8_PER_TOKEN);

    // dense: KVT = T. int8: KVT = int8_t + per-token scales.
    using KVT = std::conditional_t<QUANT, int8_t, T>;

    struct Params {
        const T* q;            // (batch*heads, seq_q, HEAD_DIM)
        const KVT* k;          // (batch*heads, seq_k, HEAD_DIM) [int8 if QUANT]
        const KVT* v;
        const float* kScale;   // (batch*heads, seq_k) per-token scales (QUANT)
        const float* vScale;
        T* o;                  // (batch*heads, seq_q, HEAD_DIM)
        int seq_q;
        int seq_k;
        float scale;           // logit scale, usually rsqrt(HEAD_DIM)
        bool causal;
        EpilogueIOp epilogue;  // fused FKL IOp chain applied before the write
    };

    FK_COOP_DEVICE_FUSE exec(const Params& p) {
        __shared__ KVT kTile[BLOCK_N][HEAD_DIM];
        __shared__ KVT vTile[BLOCK_N][HEAD_DIM];
        __shared__ float kScaleTile[BLOCK_N];
        __shared__ float vScaleTile[BLOCK_N];

        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        const int qIdx = blockIdx.x * WARPS_PER_BLOCK + warp;
        const long bh = blockIdx.y;
        const bool active = qIdx < p.seq_q;

        const T* qBase = p.q + bh * (long)p.seq_q * HEAD_DIM;
        const KVT* kBase = p.k + bh * (long)p.seq_k * HEAD_DIM;
        const KVT* vBase = p.v + bh * (long)p.seq_k * HEAD_DIM;
        T* oBase = p.o + bh * (long)p.seq_q * HEAD_DIM;
        const float* kS = QUANT ? p.kScale + bh * (long)p.seq_k : nullptr;
        const float* vS = QUANT ? p.vScale + bh * (long)p.seq_k : nullptr;

        float qReg[ELEMS_PER_LANE];
        float oAcc[ELEMS_PER_LANE];
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_LANE; ++e) {
            qReg[e] = active ? attnToF32(qBase[(long)qIdx * HEAD_DIM + lane + 32 * e]) : 0.f;
            oAcc[e] = 0.f;
        }
        float m = -FLT_MAX;
        float l = 0.f;

        const int blockMaxQ = blockIdx.x * WARPS_PER_BLOCK + WARPS_PER_BLOCK - 1;
        const int kvEnd = p.causal ? ::min(p.seq_k, blockMaxQ + 1) : p.seq_k;

        for (int tile = 0; tile < kvEnd; tile += BLOCK_N) {
            const int tileLen = ::min(BLOCK_N, kvEnd - tile);

            __syncthreads();
            for (int idx = threadIdx.x; idx < tileLen * HEAD_DIM; idx += THREADS) {
                const int r = idx / HEAD_DIM;
                const int c = idx % HEAD_DIM;
                kTile[r][c] = kBase[(long)(tile + r) * HEAD_DIM + c];
                vTile[r][c] = vBase[(long)(tile + r) * HEAD_DIM + c];
            }
            if constexpr (QUANT) {
                for (int r = threadIdx.x; r < tileLen; r += THREADS) {
                    kScaleTile[r] = kS[tile + r];
                    vScaleTile[r] = vS[tile + r];
                }
            }
            __syncthreads();

            if (!active) { continue; }

            for (int j = 0; j < tileLen; ++j) {
                const int kvIdx = tile + j;
                if (p.causal && kvIdx > qIdx) { break; }

                // s = q . k_j (dequantize in-register if compressed)
                float partial = 0.f;
                #pragma unroll
                for (int e = 0; e < ELEMS_PER_LANE; ++e) {
                    float kv;
                    if constexpr (QUANT) {
                        kv = static_cast<float>(kTile[j][lane + 32 * e]) * kScaleTile[j];
                    } else {
                        kv = attnToF32(kTile[j][lane + 32 * e]);
                    }
                    partial += qReg[e] * kv;
                }
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    partial += __shfl_xor_sync(0xffffffffu, partial, offset);
                }
                const float s = partial * p.scale;

                const float mNew = fmaxf(m, s);
                const float corr = expf(m - mNew);
                const float pj = expf(s - mNew);
                l = l * corr + pj;
                #pragma unroll
                for (int e = 0; e < ELEMS_PER_LANE; ++e) {
                    float vv;
                    if constexpr (QUANT) {
                        vv = static_cast<float>(vTile[j][lane + 32 * e]) * vScaleTile[j];
                    } else {
                        vv = attnToF32(vTile[j][lane + 32 * e]);
                    }
                    oAcc[e] = oAcc[e] * corr + pj * vv;
                }
                m = mNew;
            }
        }

        if (active) {
            const float invL = l > 0.f ? 1.f / l : 0.f;
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_LANE; ++e) {
                // EPILOGUE FUSION: the FKL IOp chain runs in-register on the
                // normalized output, then ONE global write. This is the
                // "fuse more than handmade FA" hook: any Mul/Add/Cast/
                // Saturate/... composition rides inside the same kernel.
                const float r = (oAcc[e] * invL) | p.epilogue;
                oBase[(long)qIdx * HEAD_DIM + lane + 32 * e] = attnFromF32<T>(r);
            }
        }
    }
};

template <typename T, int HEAD_DIM, KVLayout KVL, typename EpilogueIOp,
          int BLOCK_N, int WARPS_PER_BLOCK>
__global__ void launchFlashAttentionDPP_Kernel(
        const __grid_constant__ typename FlashAttentionDPP<T, HEAD_DIM, KVL, EpilogueIOp, BLOCK_N, WARPS_PER_BLOCK>::Params params) {
    FlashAttentionDPP<T, HEAD_DIM, KVL, EpilogueIOp, BLOCK_N, WARPS_PER_BLOCK>::exec(params);
}

template <typename T, int HEAD_DIM, KVLayout KVL = KVLayout::DENSE,
          int BLOCK_N = (sizeof(T) == 2 ? 64 : 32), int WARPS_PER_BLOCK = 4,
          typename EpilogueIOp = AttentionIdentityEpilogue>
inline void executeFlashAttention(
        const T* q,
        const typename FlashAttentionDPP<T, HEAD_DIM, KVL, EpilogueIOp, BLOCK_N, WARPS_PER_BLOCK>::KVT* k,
        const typename FlashAttentionDPP<T, HEAD_DIM, KVL, EpilogueIOp, BLOCK_N, WARPS_PER_BLOCK>::KVT* v,
        T* o, const int batchHeads, const int seqQ, const int seqK,
        const bool causal, Stream_<ParArch::GPU_NVIDIA>& stream,
        const float* kScale = nullptr, const float* vScale = nullptr,
        const float scaleOverride = -1.f,
        const EpilogueIOp& epilogue = {}) {
    using DPP = FlashAttentionDPP<T, HEAD_DIM, KVL, EpilogueIOp, BLOCK_N, WARPS_PER_BLOCK>;
    const float scale = scaleOverride > 0.f ? scaleOverride
                                            : rsqrtf(static_cast<float>(HEAD_DIM));
    const typename DPP::Params params{ q, k, v, kScale, vScale, o,
                                       seqQ, seqK, scale, causal, epilogue };
    const dim3 grid((seqQ + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, batchHeads, 1);
    const dim3 block(DPP::THREADS, 1, 1);
    launchFlashAttentionDPP_Kernel<T, HEAD_DIM, KVL, EpilogueIOp, BLOCK_N, WARPS_PER_BLOCK>
        <<<grid, block, 0, stream.getCUDAStream()>>>(params);
    gpuErrchk(cudaGetLastError());
}

#endif // defined(__NVCC__) || CLANG_HOST_DEVICE

// ---- host-side KV cache compression helper (reference packing) -----------
// Per-token symmetric int8: scale[t] = max|row|/127; q(x) = round(x/scale).
// Usable from tests and from frameworks that own the cache; the kernel
// dequantizes in-register (the cache is never inflated in global memory).
template <typename T>
inline void quantizeKVCacheHost(const T* dense, int8_t* q8, float* scales,
                                const int tokens, const int headDim) {
    for (int t = 0; t < tokens; ++t) {
        float mx = 0.f;
        for (int d = 0; d < headDim; ++d) {
            mx = std::max(mx, std::abs(attnToF32(dense[(long)t * headDim + d])));
        }
        const float sc = mx > 0.f ? mx / 127.f : 1.f;
        scales[t] = sc;
        for (int d = 0; d < headDim; ++d) {
            const float x = attnToF32(dense[(long)t * headDim + d]) / sc;
            q8[(long)t * headDim + d] = static_cast<int8_t>(std::nearbyint(x));
        }
    }
}

} // namespace fk

#endif // FK_ATTENTION_FLASH_ATTENTION_H
