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

#ifndef FK_ATTENTION_FLASH_ATTENTION_WMMA_H
#define FK_ATTENTION_FLASH_ATTENTION_WMMA_H

/* FlashAttentionWmmaDPP: FA-2 forward on RDNA4 (gfx12) WMMA, the HIP
 * counterpart of FlashAttentionMmaDPP (PTX mma.sync, CUDA only).
 *
 * WMMA 16x16x16 (wave32, gfx12): D = A * B + C, every lane l holding
 *   A (M x K): row l%16, k = 8*(l/16) .. +7       (8 bf16, or 8 fp8 bytes)
 *   B (K x N): column l%16, k = 8*(l/16) .. +7
 *   C, D:      column l%16, rows 8*(l/16) .. +7    (8 fp32)
 * (validated against a CPU matmul on the RX 9060; gfx11 lays A/B out
 * differently, so this is gfx12 only).
 *
 * TRANSPOSED FORMULATION: each wave owns 16 queries and computes
 *   S^T = K Q^T     (A = K rows from smem, B = Q^T held in registers)
 *   O^T += V^T P^T  (A = V^T from a transposed smem tile, B = P^T)
 * because the accumulator layout of S^T (lane = query, 8 keys) IS the B
 * layout of P^T: P feeds the second WMMA straight from registers, with no
 * smem round trip. It also makes the online softmax lane-local: a lane owns
 * one query, so row max and row sum are 8 values plus one xor-16 exchange.
 *
 * PROLOGUES/EPILOGUES as in FlashAttentionDPP: K and V tiles are staged in
 * shared memory through the K/V Read IOps (dequant, fused chains...), Q
 * through the Q IOp, once per block, and the epilogue IOp chain runs
 * in-register before the single write of O. Q, K, V and P enter the WMMA
 * as bf16 (fp32 accumulation and softmax), so fp32 inputs lose precision
 * against FlashAttentionDPP: the tests bound it.
 *
 * FP8 QK^T (FlashAttentionWmmaDPP with an Fp8TokenDequantRead K, see
 * FP8_QK): the K tile is staged as the raw e4m3 bytes of the cache, Q is
 * quantized per query in-kernel (scale = max|q| / 448) and S^T runs on the
 * fp8 WMMA, its accumulators rescaled by kScale[key] * qScale[query]. V and
 * P stay bf16 (the CUDA FP8_QK design of flash_attention_mma.h).
 *
 * Element addressing of the IOps as in FlashAttentionDPP: thread.x = head
 * element, thread.y = token, thread.z = batch*head plane. */

#include <fused_kernel/algorithms/attention/flash_attention.h>

#if defined(__HIPCC__)

#include <cmath>
#include <cstring>
#include <string>

namespace fk {

namespace attention_wmma_detail {
typedef short Bf16x8 __attribute__((ext_vector_type(8)));
typedef float F32x8 __attribute__((ext_vector_type(8)));
typedef int Fp8x8 __attribute__((ext_vector_type(2)));

constexpr int TILE = 16; // M, N and K of one WMMA

__device__ __forceinline__ short bf16Bits(const float x) {
    return __builtin_bit_cast(short, static_cast<__bf16>(x));
}

// The builtins only exist for gfx12 targets: other targets get a trap, and
// executeFlashAttentionWmma refuses to launch there.
__device__ __forceinline__ F32x8 wmmaBf16(const Bf16x8 a, const Bf16x8 b, const F32x8 c) {
#if defined(__GFX12__)
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
#else
    (void)a; (void)b;
    __builtin_trap();
    return c;
#endif
}

__device__ __forceinline__ F32x8 wmmaFp8(const Fp8x8 a, const Fp8x8 b, const F32x8 c) {
#if defined(__GFX12__)
    return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
#else
    (void)a; (void)b;
    __builtin_trap();
    return c;
#endif
}
} // namespace attention_wmma_detail

// True when the current device runs gfx12 WMMA (RDNA4: gfx1200, gfx1201).
inline bool attnDeviceHasWmmaGfx12() {
    int device = 0;
    gpuErrchk(hipGetDevice(&device));
    hipDeviceProp_t prop{};
    gpuErrchk(hipGetDeviceProperties(&prop, device));
    return std::strncmp(prop.gcnArchName, "gfx12", 5) == 0;
}

template <typename OT, int HEAD_DIM, typename QIOp, typename KIOp, typename VIOp,
          typename EpilogueIOp = AttentionIdentityEpilogue, int BLOCK_KV = 32, int NUM_WAVES = 8>
struct FlashAttentionWmmaDPP {
private:
    using SelfType = FlashAttentionWmmaDPP<OT, HEAD_DIM, QIOp, KIOp, VIOp, EpilogueIOp, BLOCK_KV, NUM_WAVES>;
    using Bf16x8 = attention_wmma_detail::Bf16x8;
    using F32x8 = attention_wmma_detail::F32x8;
    using Fp8x8 = attention_wmma_detail::Fp8x8;
    static constexpr int TILE = attention_wmma_detail::TILE;

public:
    FK_STATIC_STRUCT(FlashAttentionWmmaDPP, SelfType)

    static_assert(HEAD_DIM % TILE == 0, "HEAD_DIM must be a multiple of 16");
    static_assert(BLOCK_KV % TILE == 0, "BLOCK_KV must be a multiple of 16");
    static_assert(isAnyReadType<QIOp>, "Q prologue must be a Read or ReadBack IOp");
    static_assert(isAnyReadType<KIOp>, "K prologue must be a Read or ReadBack IOp");
    static_assert(isAnyReadType<VIOp>, "V prologue must be a Read or ReadBack IOp");

    static constexpr int BLOCK_Q = TILE * NUM_WAVES; // 16 queries per wave
    static constexpr int THREADS = NUM_WAVES * ATTN_WAVE_SIZE;
    static constexpr int D_TILES = HEAD_DIM / TILE;  // K steps of S^T, row tiles of O^T
    static constexpr int KV_TILES = BLOCK_KV / TILE; // key tiles of S^T, K steps of O^T
    // FP8_QK: S^T on the fp8 WMMA, straight from the e4m3 bytes of the K cache
    static constexpr bool FP8_QK = isFp8KVRead<KIOp>;
    // smem rows padded by 16 bytes: the 16 lanes of a fragment load read 16
    // rows at the same column, which would otherwise hit the same LDS banks
    static constexpr int K_STRIDE = FP8_QK ? HEAD_DIM + 16 : HEAD_DIM + 8; // bf16 elements, or fp8 bytes
    static constexpr int VT_STRIDE = BLOCK_KV + 8;                          // bf16 elements

    struct Params {
        QIOp q;               // prologue Read/ReadBack IOp for Q
        KIOp k;               // prologue Read/ReadBack IOp for K
        VIOp v;               // prologue Read/ReadBack IOp for V
        OT* o;                // (batch*heads, seq_q, HEAD_DIM) output
        int seq_q;
        int seq_k;
        float scale;          // logit scale, usually rsqrt(HEAD_DIM)
        bool causal;
        EpilogueIOp epilogue; // fused IOp chain on the output (pre-write)
    };

    // a K/V tile is BLOCK_KV * HEAD_DIM / 8 chunks of 8 head elements
    static constexpr int TILE_CHUNKS = BLOCK_KV * HEAD_DIM / 8;
    static constexpr int CHUNKS_PER_THREAD = (TILE_CHUNKS + THREADS - 1) / THREADS;
    // Loading tile n+1 into registers while tile n computes: measured on the
    // RX 9060, +1..7% with bf16 QK^T and -12..20% with FP8_QK, whose extra
    // registers (Q quantization, scales) it pushes over the occupancy edge.
    static constexpr bool PREFETCH = !FP8_QK;
    // 8 head elements of K as staged in smem: raw e4m3 bytes, or bf16
    using KChunk = std::conditional_t<FP8_QK, uint2, Bf16x8>;

    template <typename IOp>
    FK_DEVICE_FUSE float readElem(const IOp& iop, const int x, const int y, const int z) {
        return low_precision::attnToF32(IOp::Operation::exec(Point{ x, y, z }, iop));
    }

    /* Head elements c0 .. c0+7 of a token, as bf16 bits: one 16-byte load for
       the raw bf16 read, one 8-byte load + dequant for the int8 and fp8
       caches, and an element-wise read through the IOp for any other
       prologue. c0 is a multiple of 8. */
    template <typename IOp>
    static __device__ __forceinline__ Bf16x8 load8(const IOp& iop, const int c0, const int token, const int bh) {
        using attention_wmma_detail::bf16Bits;
        Bf16x8 out;
        if constexpr (isRawBf16Read<IOp>) {
            out = *reinterpret_cast<const Bf16x8*>(PtrAccessor<ND::_3D>::cr_point(Point{ c0, token, bh }, iop.params));
        } else if constexpr (isInt8KVRead<IOp> || isFp8KVRead<IOp>) {
            const auto& prm = iop.params;
            const uint2 packed =
                *reinterpret_cast<const uint2*>(PtrAccessor<ND::_3D>::cr_point(Point{ c0, token, bh }, prm.data));
            const float sc = prm.scales[(long)bh * prm.data.dims.height + token];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int8_t b = static_cast<int8_t>(((i < 4 ? packed.x : packed.y) >> (8 * (i % 4))) & 0xFF);
                if constexpr (isInt8KVRead<IOp>) {
                    out[i] = bf16Bits(static_cast<float>(b) * sc);
                } else {
#ifdef FK_HAS_FP8
                    out[i] = bf16Bits(attnFp8ToF32(b) * sc);
#endif
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                out[i] = bf16Bits(readElem(iop, c0 + i, token, bh));
            }
        }
        return out;
    }

    // This thread's chunks of the K/V tile starting at key kv0 (zeros past seq_k).
    static __device__ __forceinline__ void loadTile(const Params& p, const int kv0, const int bh,
                                                    KChunk (&kChunks)[CHUNKS_PER_THREAD],
                                                    Bf16x8 (&vChunks)[CHUNKS_PER_THREAD]) {
#pragma unroll
        for (int c = 0; c < CHUNKS_PER_THREAD; ++c) {
            const int idx = threadIdx.x + c * THREADS;
            const int r = idx / (HEAD_DIM / 8);
            const int c0 = 8 * (idx % (HEAD_DIM / 8));
            const bool in = idx < TILE_CHUNKS && kv0 + r < p.seq_k;
            if constexpr (FP8_QK) {
                kChunks[c] = in ? *reinterpret_cast<const uint2*>(
                                      PtrAccessor<ND::_3D>::cr_point(Point{ c0, kv0 + r, bh }, p.k.params.data))
                                : uint2{ 0u, 0u };
            } else {
                kChunks[c] = in ? load8(p.k, c0, kv0 + r, bh) : Bf16x8{};
            }
            vChunks[c] = in ? load8(p.v, c0, kv0 + r, bh) : Bf16x8{};
        }
    }

    static __device__ void exec(const Params& p) {
        using attention_wmma_detail::bf16Bits;
        // K tile: BLOCK_KV keys x HEAD_DIM, row-major (bf16, or raw e4m3 bytes)
        __shared__ __attribute__((aligned(16))) std::conditional_t<FP8_QK, int8_t, short> kTile[BLOCK_KV * K_STRIDE];
        __shared__ float kScaleTile[FP8_QK ? BLOCK_KV : 1];
        // V tile TRANSPOSED: HEAD_DIM x BLOCK_KV keys, so V^T fragments are contiguous
        __shared__ __attribute__((aligned(16))) short vtTile[HEAD_DIM * VT_STRIDE];

        attnRequireWave32();
        const int lane = threadIdx.x % ATTN_WAVE_SIZE;
        const int wave = threadIdx.x / ATTN_WAVE_SIZE;
        const int col = lane % TILE;  // this lane's query: column N of S^T and O^T
        const int half = lane / TILE; // which 8 of the 16 K (or M) indices the lane holds
        const int bh = blockIdx.y;
        const int q = blockIdx.x * BLOCK_Q + wave * TILE + col;
        const bool qValid = q < p.seq_q;

        // Q^T as B operands (lane: Q[q][16t + 8half + i]), bf16, or e4m3
        // with one scale per query for FP8_QK
        Bf16x8 qFrag[FP8_QK ? 1 : D_TILES];
        Fp8x8 qFrag8[FP8_QK ? D_TILES : 1];
        float qScale = 1.f;
        if constexpr (FP8_QK) {
            float qv[HEAD_DIM / 2]; // this lane's half of the row
            float qMax = 0.f;
#pragma unroll
            for (int t = 0; t < D_TILES; ++t) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float x = qValid ? readElem(p.q, TILE * t + 8 * half + i, q, bh) : 0.f;
                    qv[8 * t + i] = x;
                    qMax = fmaxf(qMax, fabsf(x));
                }
            }
            qMax = fmaxf(qMax, attnShflXor(qMax, TILE)); // the other half of the row
            qScale = qMax > 0.f ? qMax / 448.f : 1.f;
            const float invQScale = 1.f / qScale;
#pragma unroll
            for (int t = 0; t < D_TILES; ++t) {
                unsigned char bytes[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    bytes[i] = attn_fp8_e4m3(qv[8 * t + i] * invQScale).__x;
                }
                __builtin_memcpy(&qFrag8[t], bytes, sizeof(bytes));
            }
        } else {
#pragma unroll
            for (int t = 0; t < D_TILES; ++t) {
                qFrag[t] = qValid ? load8(p.q, TILE * t + 8 * half, q, bh) : Bf16x8{};
            }
        }

        F32x8 oAcc[D_TILES]; // O^T: lane holds O[q][16t + 8half + i]
#pragma unroll
        for (int t = 0; t < D_TILES; ++t) {
            oAcc[t] = F32x8{};
        }
        float m = -INFINITY;
        float l = 0.f;
        // log2-domain softmax: exp(x) = exp2(x * log2(e)), folded into the scale
        const float scaleLog2 = p.scale * 1.4426950408889634f * qScale;
        const int blockLastQ = ::min(p.seq_q, static_cast<int>(blockIdx.x + 1) * BLOCK_Q) - 1;
        const int kvEnd = p.causal ? ::min(p.seq_k, blockLastQ + 1) : p.seq_k;

        // K/V PROLOGUES: each tile is read through the IOps into registers,
        // 8 head elements per thread and chunk (consecutive threads read
        // consecutive 8s), ONE TILE AHEAD: the loads of tile n+1 are in flight
        // while tile n runs on the WMMA units, then go to smem.
        KChunk kNext[CHUNKS_PER_THREAD];
        Bf16x8 vNext[CHUNKS_PER_THREAD];
        if constexpr (PREFETCH) {
            loadTile(p, 0, bh, kNext, vNext);
        }
        for (int kv0 = 0; kv0 < kvEnd; kv0 += BLOCK_KV) {
            if constexpr (!PREFETCH) {
                loadTile(p, kv0, bh, kNext, vNext);
            }
            __syncthreads(); // the previous tile is no longer read
#pragma unroll
            for (int c = 0; c < CHUNKS_PER_THREAD; ++c) {
                const int idx = threadIdx.x + c * THREADS;
                if (idx >= TILE_CHUNKS) {
                    break;
                }
                const int r = idx / (HEAD_DIM / 8);
                const int c0 = 8 * (idx % (HEAD_DIM / 8));
                *reinterpret_cast<KChunk*>(&kTile[r * K_STRIDE + c0]) = kNext[c];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    vtTile[(c0 + i) * VT_STRIDE + r] = vNext[c][i];
                }
            }
            if constexpr (FP8_QK) {
                for (int r = threadIdx.x; r < BLOCK_KV; r += THREADS) {
                    kScaleTile[r] = kv0 + r < p.seq_k ? p.k.params.scales[(long)bh * p.seq_k + kv0 + r] : 0.f;
                }
            }
            __syncthreads();
            if (PREFETCH && kv0 + BLOCK_KV < kvEnd) {
                loadTile(p, kv0 + BLOCK_KV, bh, kNext, vNext);
            }

            // S^T = K Q^T: s[j][i] is the score of key kv0 + 16j + 8half + i
            F32x8 s[KV_TILES];
#pragma unroll
            for (int j = 0; j < KV_TILES; ++j) {
                s[j] = F32x8{};
#pragma unroll
                for (int t = 0; t < D_TILES; ++t) {
                    const int kOffset = (TILE * j + col) * K_STRIDE + TILE * t + 8 * half;
                    if constexpr (FP8_QK) {
                        const Fp8x8 kFrag = *reinterpret_cast<const Fp8x8*>(&kTile[kOffset]);
                        s[j] = attention_wmma_detail::wmmaFp8(kFrag, qFrag8[t], s[j]);
                    } else {
                        const Bf16x8 kFrag = *reinterpret_cast<const Bf16x8*>(&kTile[kOffset]);
                        s[j] = attention_wmma_detail::wmmaBf16(kFrag, qFrag[t], s[j]);
                    }
                }
            }

            // scale, mask (ragged tail, causal), and the max of this tile
            float tileMax = -INFINITY;
#pragma unroll
            for (int j = 0; j < KV_TILES; ++j) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int keyInTile = TILE * j + 8 * half + i;
                    const int key = kv0 + keyInTile;
                    float x = s[j][i] * scaleLog2;
                    if constexpr (FP8_QK) {
                        x *= kScaleTile[keyInTile];
                    }
                    if (key >= p.seq_k || (p.causal && key > q)) {
                        x = -INFINITY;
                    }
                    s[j][i] = x;
                    tileMax = fmaxf(tileMax, x);
                }
            }
            tileMax = fmaxf(tileMax, attnShflXor(tileMax, TILE)); // the other 8 keys of every 16

            // online softmax; a query whose keys are all masked so far keeps
            // m = -inf and adds nothing
            const float mNew = fmaxf(m, tileMax);
            const bool anyKey = mNew != -INFINITY;
            const float corr = anyKey ? exp2f(m - mNew) : 1.f;
            float rowSum = 0.f;
            Bf16x8 pFrag[KV_TILES]; // P^T as B operands: same layout as s
#pragma unroll
            for (int j = 0; j < KV_TILES; ++j) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float pj = anyKey ? exp2f(s[j][i] - mNew) : 0.f;
                    rowSum += pj;
                    pFrag[j][i] = bf16Bits(pj);
                }
            }
            rowSum += attnShflXor(rowSum, TILE);
            l = l * corr + rowSum;
            m = mNew;

            // O^T = O^T * corr + V^T P^T
#pragma unroll
            for (int t = 0; t < D_TILES; ++t) {
                oAcc[t] *= corr;
#pragma unroll
                for (int j = 0; j < KV_TILES; ++j) {
                    const Bf16x8 vFrag =
                        *reinterpret_cast<const Bf16x8*>(&vtTile[(TILE * t + col) * VT_STRIDE + TILE * j + 8 * half]);
                    oAcc[t] = attention_wmma_detail::wmmaBf16(vFrag, pFrag[j], oAcc[t]);
                }
            }
        }

        if (qValid) {
            const float invL = l > 0.f ? 1.f / l : 0.f;
            OT* const dst = p.o + ((long)bh * p.seq_q + q) * HEAD_DIM;
#pragma unroll
            for (int t = 0; t < D_TILES; ++t) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    // EPILOGUE FUSION: the IOp chain in-register, then ONE write
                    const float r = (oAcc[t][i] * invL) | p.epilogue;
                    dst[TILE * t + 8 * half + i] = low_precision::attnFromF32<OT>(r);
                }
            }
        }
    }
};

template <typename OT, int HEAD_DIM, typename QIOp, typename KIOp, typename VIOp, typename EpilogueIOp, int BLOCK_KV,
          int NUM_WAVES>
__global__ void __launch_bounds__(NUM_WAVES * ATTN_WAVE_SIZE) launchFlashAttentionWmmaDPP_Kernel(
    const __grid_constant__
    typename FlashAttentionWmmaDPP<OT, HEAD_DIM, QIOp, KIOp, VIOp, EpilogueIOp, BLOCK_KV, NUM_WAVES>::Params params) {
    FlashAttentionWmmaDPP<OT, HEAD_DIM, QIOp, KIOp, VIOp, EpilogueIOp, BLOCK_KV, NUM_WAVES>::exec(params);
}

/* IOp-first API, as executeFlashAttention. Throws std::runtime_error on a
 * device without gfx12 WMMA (see executeFlashAttentionAuto for a fallback).
 * An Fp8TokenDequantRead K selects the FP8 QK^T path. The default tile
 * (BLOCK_KV 32, 8 waves = 128 queries per block) was the fastest of
 * BLOCK_KV {32, 64} x {4, 8} waves on every shape measured on the RX 9060
 * (d64/d128, s1024/s4096, dense/causal). */
template <int HEAD_DIM, int BLOCK_KV = 32, int NUM_WAVES = 8, typename OT = float, typename QIOp, typename KIOp,
          typename VIOp, typename EpilogueIOp = AttentionIdentityEpilogue>
inline void executeFlashAttentionWmma(const QIOp& q, const KIOp& k, const VIOp& v, OT* o, const int batchHeads,
                                      const int seqQ, const int seqK, const bool causal,
                                      Stream_<ParArch::GPU_AMD>& stream, const float scaleOverride = -1.f,
                                      const EpilogueIOp& epilogue = {}) {
    using DPP = FlashAttentionWmmaDPP<OT, HEAD_DIM, QIOp, KIOp, VIOp, EpilogueIOp, BLOCK_KV, NUM_WAVES>;
    static const bool hasWmma = attnDeviceHasWmmaGfx12();
    if (!hasWmma) {
        throw std::runtime_error("executeFlashAttentionWmma needs a gfx12 (RDNA4) GPU");
    }
    const float scale = scaleOverride > 0.f ? scaleOverride : 1.f / std::sqrt(static_cast<float>(HEAD_DIM));
    const typename DPP::Params params{ q, k, v, o, seqQ, seqK, scale, causal, epilogue };
    const dim3 grid((seqQ + DPP::BLOCK_Q - 1) / DPP::BLOCK_Q, batchHeads, 1);
    const dim3 block(DPP::THREADS, 1, 1);
    launchFlashAttentionWmmaDPP_Kernel<OT, HEAD_DIM, QIOp, KIOp, VIOp, EpilogueIOp, BLOCK_KV, NUM_WAVES>
        <<<grid, block, 0, stream.getHIPStream()>>>(params);
    gpuErrchk(hipGetLastError());
}

/* Dispatcher: the WMMA kernel on gfx12, FlashAttentionDPP (scalar, fp32)
 * otherwise. Note the two differ in precision: the WMMA kernel rounds Q, K,
 * V and P to bf16 (or Q and K to fp8 with an fp8 K cache). */
template <int HEAD_DIM, typename OT = float, typename QIOp, typename KIOp, typename VIOp,
          typename EpilogueIOp = AttentionIdentityEpilogue>
inline void executeFlashAttentionAuto(const QIOp& q, const KIOp& k, const VIOp& v, OT* o, const int batchHeads,
                                      const int seqQ, const int seqK, const bool causal,
                                      Stream_<ParArch::GPU_AMD>& stream, const float scaleOverride = -1.f,
                                      const EpilogueIOp& epilogue = {}) {
    static const bool hasWmma = attnDeviceHasWmmaGfx12();
    if (hasWmma) {
        executeFlashAttentionWmma<HEAD_DIM>(q, k, v, o, batchHeads, seqQ, seqK, causal, stream, scaleOverride,
                                            epilogue);
    } else {
        executeFlashAttention<HEAD_DIM>(q, k, v, o, batchHeads, seqQ, seqK, causal, stream, scaleOverride,
                                        epilogue);
    }
}

} // namespace fk

#endif // defined(__HIPCC__)

#endif // FK_ATTENTION_FLASH_ATTENTION_WMMA_H
