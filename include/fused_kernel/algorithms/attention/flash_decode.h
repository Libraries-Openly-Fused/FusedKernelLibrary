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

#ifndef FK_ATTENTION_FLASH_DECODE_H
#define FK_ATTENTION_FLASH_DECODE_H

/* FlashDecodeDPP: split-KV attention decode (seq_q == 1), FlashDecoding
 * style. Decode is BANDWIDTH-bound (one query token vs a long KV cache:
 * arithmetic intensity ~1 FLOP/byte), so the kernel optimizes for
 * occupancy + bytes, not tensor cores:
 *
 *  - grid = (NUM_SPLITS, batch*heads): the KV sequence is split across
 *    blocks so ALL SMs work even at batch*heads << #SM (the reason a
 *    single-pass kernel loses 10x at decode).
 *  - each warp owns a private online-softmax state (m, l, o) and walks
 *    tokens strided; warps combine in smem; splits combine in a tiny
 *    second kernel (numerically exact online-softmax merge).
 *  - K/V are read THROUGH prologue IOps: the fp8/int8 compressed cache
 *    dequantizes in-register while streaming, so the global traffic is
 *    HALF of bf16 — decode latency scales with bytes, so fp8 cache
 *    ~= 2x faster decode at long seq (and 2x more tokens in VRAM).
 *
 * This pairs with FA4-sm12x's fp8 decode direction (Dao-AILab PR #2634)
 * but keeps FKL's universal prologue: ANY Read/ReadBack IOp works. */

#include <fused_kernel/algorithms/attention/flash_attention.h>
#include <fused_kernel/algorithms/attention/flash_attention_mma.h>  // IOp traits + bf16 read

#if (defined(__NVCC__) || CLANG_HOST_DEVICE)

namespace fk {

template <typename OT, int HEAD_DIM, typename QIOp, typename KIOp, typename VIOp,
          int NUM_WARPS = 8>
struct FlashDecodeDPP {
private:
    using SelfType = FlashDecodeDPP<OT, HEAD_DIM, QIOp, KIOp, VIOp, NUM_WARPS>;
public:
    FK_STATIC_STRUCT(FlashDecodeDPP, SelfType)

    static_assert(HEAD_DIM % 32 == 0, "HEAD_DIM must be a multiple of 32");
    static constexpr int THREADS = NUM_WARPS * 32;
    static constexpr int ELEMS = HEAD_DIM / 32;   // per-lane o/q elements

    struct Params {
        QIOp q; KIOp k; VIOp v;
        float* partial;      // workspace [bh, splits, (HEAD_DIM+2)] + counters
        unsigned int* counters;  // [bh] arrival counters (self-resetting)
        OT* o;               // final [bh, 1, HEAD_DIM]
        int seq_k;
        int splits;
        float scale;
    };

    template <typename IOp>
    FK_DEVICE_FUSE float readElem(const IOp& iop, const int x, const int y, const int z) {
        return attnToF32(IOp::Operation::exec(Point{ x, y, z }, iop));
    }

    /* Vectorized row loader: lane owns elements [ELEMS*lane, ELEMS*lane+ELEMS).
       Raw bf16 rows load as bfloat162 pairs (4B), fp8/int8 rows as packed
       bytes (ELEMS bytes in one load) — decode is bandwidth-bound, wide
       loads are the whole game. Generic IOps fall back to per-element. */
    template <typename IOp>
    FK_DEVICE_FUSE void loadRow(const IOp& iop, const int lane, const int t,
                                const int bh, const int seqLen,
                                float (&out)[ELEMS]) {
        if constexpr (isRawBf16Read<IOp>) {
            const auto& prm = iop.params;
            const __nv_bfloat16* base = prm.data
                + ((long)bh * seqLen + t) * HEAD_DIM + ELEMS * lane;
            #pragma unroll
            for (int e = 0; e < ELEMS; e += 2) {
                const __nv_bfloat162 pair =
                    *reinterpret_cast<const __nv_bfloat162*>(base + e);
                out[e] = __bfloat162float(pair.x);
                out[e + 1] = __bfloat162float(pair.y);
            }
        } else if constexpr (isFp8KVRead<IOp> || isInt8KVRead<IOp>) {
            const auto& prm = iop.params;
            const float sc = prm.scales[(long)bh * seqLen + t];
            const int8_t* base = reinterpret_cast<const int8_t*>(prm.data.data)
                + ((long)bh * seqLen + t) * HEAD_DIM + ELEMS * lane;
            // one packed load: 2 bytes (d64) or 4 bytes (d128)
            uint32_t packed = 0;
            if constexpr (ELEMS == 2) packed = *reinterpret_cast<const uint16_t*>(base);
            else packed = *reinterpret_cast<const uint32_t*>(base);
            #pragma unroll
            for (int e = 0; e < ELEMS; ++e) {
                const int8_t b = static_cast<int8_t>((packed >> (8 * e)) & 0xFF);
                if constexpr (isFp8KVRead<IOp>) {
#ifdef FK_HAS_FP8
                    __nv_fp8_e4m3 f8;
                    f8.__x = static_cast<__nv_fp8_storage_t>(b);
                    out[e] = static_cast<float>(f8) * sc;
#endif
                } else {
                    out[e] = static_cast<float>(b) * sc;
                }
            }
        } else {
            #pragma unroll
            for (int e = 0; e < ELEMS; ++e)
                out[e] = readElem(iop, ELEMS * lane + e, t, bh);
        }
    }

    static __device__ void exec(const Params& p) {
        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        const int split = blockIdx.x;
        const int bh = blockIdx.y;

        const int chunk = (p.seq_k + p.splits - 1) / p.splits;
        const int kvBegin = split * chunk;
        const int kvEnd = ::min(p.seq_k, kvBegin + chunk);
        const bool emptySplit = kvBegin >= kvEnd;
        if (emptySplit) {
            // empty split: emit a neutral partial, then STILL participate in
            // the arrival counter below (otherwise the fused merge never fires)
            if (threadIdx.x == 0 && p.splits > 1) {
                float* dst = p.partial + ((long)bh * p.splits + split) * (HEAD_DIM + 2);
                dst[HEAD_DIM] = -FLT_MAX; dst[HEAD_DIM + 1] = 0.f;
                for (int dd = 0; dd < HEAD_DIM; ++dd) dst[dd] = 0.f;
            }
            if (p.splits == 1) return;
        }

        if (!emptySplit) {
        // q -> registers; lane owns elements [ELEMS*lane, ELEMS*lane+ELEMS)
        float qReg[ELEMS];
        #pragma unroll
        for (int e = 0; e < ELEMS; ++e)
            qReg[e] = readElem(p.q, ELEMS * lane + e, 0, bh);

        // each warp walks tokens strided by NUM_WARPS
        float m = -FLT_MAX, l = 0.f;
        float oAcc[ELEMS];
        #pragma unroll
        for (int e = 0; e < ELEMS; ++e) oAcc[e] = 0.f;

        for (int t = kvBegin + warp; t < kvEnd; t += NUM_WARPS) {
            float kReg[ELEMS], vReg[ELEMS];
            loadRow(p.k, lane, t, bh, p.seq_k, kReg);
            loadRow(p.v, lane, t, bh, p.seq_k, vReg);
            float partialDot = 0.f;
            #pragma unroll
            for (int e = 0; e < ELEMS; ++e) partialDot += qReg[e] * kReg[e];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                partialDot += __shfl_xor_sync(0xFFFFFFFFu, partialDot, off);
            const float s = partialDot * p.scale;

            const float mNew = fmaxf(m, s);
            const float corr = __expf(m - mNew);
            const float pj = __expf(s - mNew);
            l = l * corr + pj;
            #pragma unroll
            for (int e = 0; e < ELEMS; ++e)
                oAcc[e] = oAcc[e] * corr + pj * vReg[e];
            m = mNew;
        }

        // combine NUM_WARPS partials in smem (online-softmax merge)
        __shared__ float sM[NUM_WARPS], sL[NUM_WARPS];
        __shared__ float sO[NUM_WARPS][HEAD_DIM];
        #pragma unroll
        for (int e = 0; e < ELEMS; ++e) sO[warp][ELEMS * lane + e] = oAcc[e];
        if (lane == 0) { sM[warp] = m; sL[warp] = l; }
        __syncthreads();

        if (warp == 0) {
            float gm = -FLT_MAX;
            #pragma unroll
            for (int w = 0; w < NUM_WARPS; ++w) gm = fmaxf(gm, sM[w]);
            float gl = 0.f;
            float gO[ELEMS];
            #pragma unroll
            for (int e = 0; e < ELEMS; ++e) gO[e] = 0.f;
            #pragma unroll
            for (int w = 0; w < NUM_WARPS; ++w) {
                const float c = (sM[w] == -FLT_MAX) ? 0.f : __expf(sM[w] - gm);
                gl += sL[w] * c;
                #pragma unroll
                for (int e = 0; e < ELEMS; ++e)
                    gO[e] += sO[w][ELEMS * lane + e] * c;
            }

            if (p.splits == 1) {
                const float inv = gl > 0.f ? 1.f / gl : 0.f;
                #pragma unroll
                for (int e = 0; e < ELEMS; ++e)
                    p.o[(long)bh * HEAD_DIM + ELEMS * lane + e] =
                        attnFromF32<OT>(gO[e] * inv);
            } else {
                float* dst = p.partial + ((long)bh * p.splits + split) * (HEAD_DIM + 2);
                #pragma unroll
                for (int e = 0; e < ELEMS; ++e) dst[ELEMS * lane + e] = gO[e];
                if (lane == 0) { dst[HEAD_DIM] = gm; dst[HEAD_DIM + 1] = gl; }
            }
        }
        } // !emptySplit
        if (p.splits == 1) return;

        // ---- FUSED split merge: last CTA to arrive merges (no 2nd kernel).
        // GPU-side fusion of the tiny epilogue kernel that profiling showed
        // wasting a full launch (~1.5us+gap) per decode step.
        __shared__ bool amLast;
        __threadfence();                       // partials visible device-wide
        __syncthreads();
        if (threadIdx.x == 0) {
            const unsigned int arrived =
                atomicAdd(&p.counters[bh], 1u);
            amLast = (arrived == (unsigned int)p.splits - 1u);
            if (amLast) p.counters[bh] = 0u;   // self-reset for next call
        }
        __syncthreads();
        if (!amLast) return;

        // merge all splits for this bh row: thread d owns output element d.
        const float* base = p.partial + (long)bh * p.splits * (HEAD_DIM + 2);
        for (int d = threadIdx.x; d < HEAD_DIM; d += THREADS) {
            float gm2 = -FLT_MAX;
            for (int s = 0; s < p.splits; ++s)
                gm2 = fmaxf(gm2, base[s * (HEAD_DIM + 2) + HEAD_DIM]);
            float gl2 = 0.f, acc = 0.f;
            for (int s = 0; s < p.splits; ++s) {
                const float sm = base[s * (HEAD_DIM + 2) + HEAD_DIM];
                const float sl = base[s * (HEAD_DIM + 2) + HEAD_DIM + 1];
                const float c = (sm == -FLT_MAX) ? 0.f : __expf(sm - gm2);
                gl2 += sl * c;
                acc += base[s * (HEAD_DIM + 2) + d] * c;
            }
            const float inv = gl2 > 0.f ? 1.f / gl2 : 0.f;
            p.o[(long)bh * HEAD_DIM + d] = attnFromF32<OT>(acc * inv);
        }
    }
};

template <typename OT, int HEAD_DIM, typename QIOp, typename KIOp, typename VIOp,
          int NUM_WARPS>
__global__ void launchFlashDecodeDPP_Kernel(
        const __grid_constant__ typename FlashDecodeDPP<
            OT, HEAD_DIM, QIOp, KIOp, VIOp, NUM_WARPS>::Params params) {
    FlashDecodeDPP<OT, HEAD_DIM, QIOp, KIOp, VIOp, NUM_WARPS>::exec(params);
}

// split-merge: one block per bh row; exact online-softmax combine.
template <typename OT, int HEAD_DIM>
__global__ void flashDecodeMerge_Kernel(const float* partial, OT* o,
                                        const int splits) {
    const int bh = blockIdx.x;
    const int lane = threadIdx.x;   // HEAD_DIM threads
    const float* base = partial + (long)bh * splits * (HEAD_DIM + 2);

    float gm = -FLT_MAX;
    for (int s = 0; s < splits; ++s) gm = fmaxf(gm, base[s * (HEAD_DIM + 2) + HEAD_DIM]);
    float gl = 0.f, acc = 0.f;
    for (int s = 0; s < splits; ++s) {
        const float sm = base[s * (HEAD_DIM + 2) + HEAD_DIM];
        const float sl = base[s * (HEAD_DIM + 2) + HEAD_DIM + 1];
        const float c = (sm == -FLT_MAX) ? 0.f : __expf(sm - gm);
        gl += sl * c;
        acc += base[s * (HEAD_DIM + 2) + lane] * c;
    }
    const float inv = gl > 0.f ? 1.f / gl : 0.f;
    o[(long)bh * HEAD_DIM + lane] = attnFromF32<OT>(acc * inv);
}

/* Decode workspace: caller-owned, reusable across steps.
   Size: batchHeads * splits * (HEAD_DIM+2) floats. */
inline int flashDecodeSplits(const int batchHeads, const int seqK) {
    // Swept on RTX PRO 6000 (188 SMs, benchmarks/sweep_decode_splits.py):
    // best split counts cluster around bh*splits ~ 2048 blocks — the old
    // target of 384 left 2-5x perf on the table for bf16 KV (which moves
    // 2x the bytes of fp8 and needs the extra memory parallelism).
    // Cap at 128 splits (FA splitkv-style) and keep chunks >= 256 tokens.
    const int targetBlocks = 2048;
    int splits = (targetBlocks + batchHeads - 1) / batchHeads;
    const int maxSplits = (seqK + 255) / 256;
    splits = ::min(splits, maxSplits);
    splits = ::min(splits, 128);
    return ::max(1, splits);
}

// workspace floats needed for a decode call (partials + arrival counters).
// IMPORTANT: zero the workspace ONCE after allocation (counters start at 0
// and self-reset after every call, so one memset at alloc time is enough).
inline size_t flashDecodeWorkspaceFloats(const int batchHeads, const int splits,
                                         const int headDim) {
    return (size_t)batchHeads * splits * (headDim + 2) + batchHeads;
}

template <int HEAD_DIM, int NUM_WARPS = 8, typename OT = float,
          typename QIOp, typename KIOp, typename VIOp>
inline void executeFlashDecode(
        const QIOp& q, const KIOp& k, const VIOp& v, OT* o,
        float* workspace /* flashDecodeWorkspaceFloats(); zeroed at alloc;
                            nullptr ok if splits==1 */,
        const int batchHeads, const int seqK,
        Stream_<ParArch::GPU_NVIDIA>& stream,
        const float scaleOverride = -1.f, const int splitsOverride = 0) {
    using DPP = FlashDecodeDPP<OT, HEAD_DIM, QIOp, KIOp, VIOp, NUM_WARPS>;
    const float scale = scaleOverride > 0.f ? scaleOverride
                                            : rsqrtf(static_cast<float>(HEAD_DIM));
    const int splits = splitsOverride > 0 ? splitsOverride
                                          : flashDecodeSplits(batchHeads, seqK);
    // counters live after the partials (reinterpreted as uint32)
    unsigned int* counters = (splits > 1)
        ? reinterpret_cast<unsigned int*>(
              workspace + (size_t)batchHeads * splits * (HEAD_DIM + 2))
        : nullptr;
    const typename DPP::Params params{ q, k, v, workspace, counters, o,
                                       seqK, splits, scale };
    const dim3 grid(splits, batchHeads, 1);
    // single kernel: the split merge is FUSED (last-CTA-arrives pattern) —
    // profiling showed the separate 1.4us merge kernel + launch gap was pure
    // overhead at decode latencies.
    launchFlashDecodeDPP_Kernel<OT, HEAD_DIM, QIOp, KIOp, VIOp, NUM_WARPS>
        <<<grid, DPP::THREADS, 0, stream.getCUDAStream()>>>(params);
    gpuErrchk(cudaGetLastError());
}

} // namespace fk

#endif // defined(__NVCC__) || CLANG_HOST_DEVICE

#endif // FK_ATTENTION_FLASH_DECODE_H
