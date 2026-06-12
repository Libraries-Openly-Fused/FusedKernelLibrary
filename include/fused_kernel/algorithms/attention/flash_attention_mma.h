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
 * for consumer Blackwell (sm_120: no WGMMA/tcgen05/TMEM — same constraint
 * set as Dao-AILab/flash-attention PR #2634 and CUTLASS PR #3030).
 *
 * AUTOMATIC SCHEDULE SELECTION (the FKL "smart fusion" play):
 * the kernel inspects the prologue IOp types AT COMPILE TIME:
 *
 *  - RAW prologue (identity read of bf16, from makeAttentionRead on a
 *    __nv_bfloat16 pointer): K/V tiles stream with REAL cp.async
 *    (16B cg copies, commit/wait groups, K double-buffered + V
 *    single-buffered — the fa-5090 v5 schedule). Loads are truly
 *    asynchronous: bytes fly while tensor cores work.
 *
 *  - FUSED prologue (dequant int8, Mul/Add chains, any .then):
 *    cp.async copies raw bytes and cannot run an IOp chain per element,
 *    so tiles are prefetched THROUGH the IOp into registers one
 *    iteration ahead (2-stage), then packed to swizzled smem.
 *
 * Either way the user writes the same compose-style code; FKL picks the
 * fastest legal schedule. Epilogue IOps always run in-register on the
 * output before the single global write.
 *
 * Ladder: v1 mma.sync + FA-2 warp split + online softmax on fp32 regs;
 * v2 XOR-swizzled smem (bank-conflict-free stores + ldmatrix);
 * v3/v5 pipelining (cp.async groups or register prefetch);
 * v4 ldmatrix.x4 for K and V. */

#include <fused_kernel/algorithms/attention/flash_attention.h>

#if (defined(__NVCC__) || CLANG_HOST_DEVICE)

#include <cuda_bf16.h>
#include <vector>

namespace fk {

// ---- bf16 identity Read IOp (the "raw" prologue) ---------------------------
struct Bf16AttentionRead {
private:
    using Parent = ReadOperation<__nv_bfloat16, RawPtr<ND::_3D, __nv_bfloat16>,
                                 float, TF::DISABLED, Bf16AttentionRead>;
    using SelfType = Bf16AttentionRead;
public:
    FK_STATIC_STRUCT(Bf16AttentionRead, SelfType)
    DECLARE_READ_PARENT

    FK_HOST_DEVICE_FUSE float exec(const Point thread, const ParamsType& params) {
#if defined(__CUDA_ARCH__)
        return __bfloat162float(*PtrAccessor<ND::_3D>::cr_point(thread, params));
#else
        return 0.f;  // host never executes this
#endif
    }
    FK_HOST_DEVICE_FUSE uint num_elems_x(const Point thread, const OperationDataType& opData) {
        return opData.params.dims.width;
    }
    FK_HOST_DEVICE_FUSE uint num_elems_y(const Point thread, const OperationDataType& opData) {
        return opData.params.dims.height;
    }
    FK_HOST_DEVICE_FUSE uint num_elems_z(const Point thread, const OperationDataType& opData) {
        return opData.params.dims.planes;
    }
    FK_HOST_DEVICE_FUSE uint pitch(const Point thread, const OperationDataType& opData) {
        return opData.params.dims.pitch;
    }
    FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
        return { num_elems_x(Point{0,0,0}, opData),
                 num_elems_y(Point{0,0,0}, opData),
                 num_elems_z(Point{0,0,0}, opData) };
    }
};

// bf16 overload of the canonical prologue builder.
inline auto makeAttentionRead(const __nv_bfloat16* data, const int batchHeads,
                              const int seq, const int headDim) {
    const RawPtr<ND::_3D, __nv_bfloat16> ptr{
        const_cast<__nv_bfloat16*>(data),
        PtrDims<ND::_3D>(static_cast<uint>(headDim), static_cast<uint>(seq),
                         static_cast<uint>(batchHeads), 1,
                         static_cast<uint>(headDim * sizeof(__nv_bfloat16))) };
    return Bf16AttentionRead::build(ptr);
}

template <typename IOp>
constexpr bool isRawBf16Read = std::is_same_v<typename IOp::Operation, Bf16AttentionRead>;

#ifdef FK_HAS_FP8
template <typename IOp>
constexpr bool isFp8KVRead = std::is_same_v<typename IOp::Operation, Fp8TokenDequantRead>;
#else
template <typename IOp>
constexpr bool isFp8KVRead = false;
#endif
template <typename IOp>
constexpr bool isInt8KVRead = std::is_same_v<typename IOp::Operation, Int8TokenDequantRead>;

// ---- FLEX ATTENTION: score mods (FlexAttention-style score_mod) ------------
// A score mod is a tiny device functor applied to each attention score
// AFTER scaling and bounds/causal masking, BEFORE the online softmax:
//     s' = mod(s, q_idx, kv_idx)
// Return -FLT_MAX to mask a position (mask_mod semantics). Composable with
// everything else (prologues, epilogues, compressed KV, block sparsity).
struct NoScoreMod {
    __device__ __forceinline__ float operator()(const float s, const int,
                                                const int) const { return s; }
};
struct ALiBiScoreMod {              // s - slope * (q - k)
    float slope;
    __device__ __forceinline__ float operator()(const float s, const int q,
                                                const int k) const {
        return s - slope * static_cast<float>(q - k);
    }
};
struct SoftCapScoreMod {            // Gemma-2 style logit soft capping
    float cap;
    __device__ __forceinline__ float operator()(const float s, const int,
                                                const int) const {
        return cap * tanhf(s / cap);
    }
};
struct SlidingWindowMask {          // mask_mod: keep only last `window` keys
    int window;
    __device__ __forceinline__ float operator()(const float s, const int q,
                                                const int k) const {
        return (q - k) >= window ? -FLT_MAX : s;
    }
};

/* BLOCK-SPARSE ATTENTION: a (bh, nQBlocks, nKVBlocks) uint8 mask at
 * (maskBQ x maskBKV) granularity. Inactive KV tiles are SKIPPED ENTIRELY —
 * no global reads, no mma, no softmax (both bandwidth and compute scale
 * with the sparsity). nullptr = dense. Requirements (checked at launch):
 * maskBQ % BLOCK_Q == 0 and maskBKV % BLOCK_KV == 0. */
struct BlockSparsity {
    const unsigned char* mask = nullptr;   // nullptr = dense
    int nQBlocks = 0, nKVBlocks = 0;
    int maskBQ = 128, maskBKV = 128;
};

/* Host helper: build a sliding-window block mask at (blockQ x blockKV)
 * granularity. Tile (qb, kb) is active iff ANY (q, k) pair inside it
 * satisfies causal q >= k AND q - k < window. Finer blocks (e.g. 64 to
 * match the raw-path kernel tiles) skip more tiles -> faster. The exact
 * per-element window edge is enforced by SlidingWindowMask (score mod);
 * compose both: mask for the skip, mod for the edge. */
inline std::vector<unsigned char>
makeSlidingWindowBlockMask(const int batchHeads, const int seqQ, const int seqK,
                           const int window, const int blockQ, const int blockKV) {
    const int nQB = (seqQ + blockQ - 1) / blockQ;
    const int nKB = (seqK + blockKV - 1) / blockKV;
    std::vector<unsigned char> m((size_t)batchHeads * nQB * nKB, 0);
    for (int qb = 0; qb < nQB; ++qb) {
        const int qLo = qb * blockQ;
        const int qHi = ::min(seqQ - 1, qLo + blockQ - 1);
        for (int kb = 0; kb < nKB; ++kb) {
            const int kLo = kb * blockKV;
            // active iff intervals [kLo, kHi] and [qHi-window+1, qHi] overlap
            // under causality (k <= qHi) — widest query row decides.
            const bool active = (kLo <= qHi) && (kLo + blockKV - 1 >= qLo - window + 1);
            if (active)
                for (int b = 0; b < batchHeads; ++b)
                    m[((size_t)b * nQB + qb) * nKB + kb] = 1;
        }
    }
    return m;
}

/* FP8 tensor-core QK^T (kind::f8f6f4 m16n8k32, e4m3xe4m3->f32): Q is
 * quantized per-row IN-KERNEL, K^T runs directly on the raw fp8 KV-cache
 * bytes (the K dequant pass disappears), and qScale[row]*kScale[col] is
 * folded into the scores post-mma. Opt-in: the instruction needs the
 * arch-FEATURE set — compile the TU with
 *   -gencode arch=compute_120a,code=sm_120a -DFK_ENABLE_FP8_QK=1
 * (plain sm_120 ptxas REJECTS kind::f8f6f4 — verified). Measured raw mma
 * on RTX PRO 6000 (spikes/spike_fp8_mma.cu): 1006 TFLOPS vs 551 bf16 =
 * 1.83x; fragment mapping validated vs fp64 oracle
 * (spikes/spike_fp8_qk_mapping.cu, maxErr 1e-6). */
#ifndef FK_ENABLE_FP8_QK
#define FK_ENABLE_FP8_QK 0
#endif

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

// 16-byte async copy; srcSize=0 zero-fills (tail/ragged guard).
__device__ __forceinline__ void cp_async_16(uint32_t dst, const void* src,
                                            const int srcSize) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 :: "r"(dst), "l"(src), "r"(srcSize));
}
// 8-byte async copy (ca path; cg only supports 16B).
__device__ __forceinline__ void cp_async_8(uint32_t dst, const void* src,
                                           const int srcSize) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;"
                 :: "r"(dst), "l"(src), "r"(srcSize));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;" :: "n"(N));
}

struct alignas(16) Bf16x8 { __nv_bfloat162 h[4]; };

// two-phase-lookup-safe fast exp (visible at template definition time).
__device__ __forceinline__ float fastExp(const float x) {
#if defined(__CUDA_ARCH__)
    return __expf(x);
#else
    return 0.f;
#endif
}

// pack two fp32 into one bf16x2 register (uint32_t) — used by the in-place
// S/P union pack in softmaxTile.
__device__ __forceinline__ uint32_t packBf162(const float lo, const float hi) {
    const __nv_bfloat162 h = __float22bfloat162_rn({ lo, hi });
    uint32_t r;
    memcpy(&r, &h, sizeof(r));
    return r;
}

// fp8 e4m3 mma: kind::f8f6f4 m16n8k32, A 4 regs (16 e4m3), B 2 regs (8 e4m3).
// Always declared so `if constexpr` branches type-check; the instruction is
// only emitted when FK_ENABLE_FP8_QK (needs sm_120a/121a feature set).
__device__ __forceinline__ void mma_fp8_m16n8k32(const uint32_t A[4],
                                                 const uint32_t B[2],
                                                 float D[4]) {
#if FK_ENABLE_FP8_QK
    asm volatile(
        "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]));
#else
    (void)A; (void)B; (void)D;
#endif
}

} // namespace attention_mma_detail

template <typename OT, int HEAD_DIM,
          typename QIOp, typename KIOp, typename VIOp,
          typename EpilogueIOp = AttentionIdentityEpilogue,
          int BLOCK_Q = (HEAD_DIM <= 64 ? 128 : 64), int BLOCK_KV = 32,
          int NUM_WARPS = 4, typename ScoreModOp = NoScoreMod>
struct FlashAttentionMmaDPP {
private:
    using SelfType = FlashAttentionMmaDPP<OT, HEAD_DIM, QIOp, KIOp, VIOp,
                                          EpilogueIOp, BLOCK_Q, BLOCK_KV, NUM_WARPS,
                                          ScoreModOp>;
public:
    static constexpr bool HAS_SCORE_MOD = !std::is_same_v<ScoreModOp, NoScoreMod>;
    // ALiBi SPECIALIZATION: the bias -slope*(q-k) = -slope*q + slope*k is
    // linear, and the per-row term -slope*q is CONSTANT within a row, so it
    // cancels in softmax. Applying only the column term +slope*k makes the
    // bias row-independent, introduces no -FLT_MAX sentinels, and keeps the
    // NO_MASK exp fast path for interior tiles (the generic scoreMod path
    // disabled it, costing 0.81-0.91x vs FA at s4096-8192 in the gauntlet).
    // Both branches (interior + masked) use the column-only form so every
    // score of a row shifts by the same constant -> softmax identical.
    static constexpr bool IS_ALIBI = std::is_same_v<ScoreModOp, ALiBiScoreMod>;
private:
public:
    FK_STATIC_STRUCT(FlashAttentionMmaDPP, SelfType)

    static_assert(HEAD_DIM % 32 == 0, "HEAD_DIM must be a multiple of 32");
    static_assert(isAnyReadType<QIOp>, "Q prologue must be a Read or ReadBack IOp");
    static_assert(isAnyReadType<KIOp>, "K prologue must be a Read or ReadBack IOp");
    static_assert(isAnyReadType<VIOp>, "V prologue must be a Read or ReadBack IOp");

    static constexpr int THREADS = NUM_WARPS * 32;
    static constexpr int WARP_Q = BLOCK_Q / NUM_WARPS;
    static_assert(WARP_Q % 16 == 0, "BLOCK_Q/NUM_WARPS must be a multiple of 16");
    static constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;

    // RAW = identity bf16 reads -> real cp.async streaming (fa-5090 v5).
    static constexpr bool RAW_KV = isRawBf16Read<KIOp> && isRawBf16Read<VIOp>;
    static constexpr bool RAW_Q = isRawBf16Read<QIOp>;
    // QUANT_KV = fp8/int8 per-token KV -> cp.async the RAW QUANTIZED bytes
    // (HALF the global traffic of bf16!) + smem->smem dequant pass.
    static constexpr bool QUANT_KV =
        (isFp8KVRead<KIOp> && isFp8KVRead<VIOp>) ||
        (isInt8KVRead<KIOp> && isInt8KVRead<VIOp>);
    // FP8_QK: QK^T runs on the fp8 tensor-core path (kind::f8f6f4 m16n8k32)
    // directly on the raw e4m3 K bytes; Q is quantized per-row in-kernel.
    // The K dequant pass disappears. PV stays bf16 (P is computed online).
    static constexpr bool FP8_QK =
        FK_ENABLE_FP8_QK != 0 && isFp8KVRead<KIOp> && isFp8KVRead<VIOp>;

    static constexpr int STRIDE_B = HEAD_DIM * (int)sizeof(__nv_bfloat16);
    static constexpr int Q_GROUPS = BLOCK_Q * HEAD_DIM / (THREADS * 8);
    static constexpr int KV_GROUPS = BLOCK_KV * HEAD_DIM / (THREADS * 8);
    static_assert(Q_GROUPS >= 1 && BLOCK_Q * HEAD_DIM % (THREADS * 8) == 0,
                  "BLOCK_Q*HEAD_DIM must be a multiple of THREADS*8");
    static_assert(KV_GROUPS >= 1 && BLOCK_KV * HEAD_DIM % (THREADS * 8) == 0,
                  "BLOCK_KV*HEAD_DIM must be a multiple of THREADS*8");

    static constexpr int KV_BUF_B = BLOCK_KV * STRIDE_B;
    // RAW schedule uses K double + V single (3 bf16 buffers) — keeping smem
    // at 3 bufs is REQUIRED for 4 blocks/SM (ncu: smem limited occupancy to
    // 3 blocks at 4 bufs). QUANT/register-prefetch use 4 (+byte staging).
    static constexpr int KV_BUFS = RAW_KV ? 3 : 4;
    static constexpr int Q_STAGE_B = QUANT_KV ? 4 * (KV_BUF_B / 2) : 0;
    // FP8_QK: the Q phase additionally holds the e4m3 Q tile + per-row
    // scales NEXT TO the staged bf16 Q (both alive during quantizeQTile).
    static constexpr int Q_PHASE_B =
        BLOCK_Q * STRIDE_B + (FP8_QK ? BLOCK_Q * HEAD_DIM + BLOCK_Q * 4 : 0);
    static constexpr int SMEM_BYTES =
        (Q_PHASE_B > KV_BUFS * KV_BUF_B + Q_STAGE_B
             ? Q_PHASE_B
             : KV_BUFS * KV_BUF_B + Q_STAGE_B);

    struct Params {
        QIOp q; KIOp k; VIOp v;
        OT* o;
        int seq_q, seq_k;
        float scale;
        bool causal;
        EpilogueIOp epilogue;
        ScoreModOp scoreMod;     // flex-attention score_mod / mask_mod
        BlockSparsity sparse;    // block-sparse tile skipping (nullptr = dense)
    };

    FK_DEVICE_FUSE uint32_t swz(const uint32_t byteOff) {
        const uint32_t row = (byteOff / STRIDE_B) % 8;
        constexpr uint32_t div = (64 / STRIDE_B > 1 ? 64 / STRIDE_B : 1);
        return byteOff ^ ((row / div) << 4);
    }

    template <typename IOp>
    FK_DEVICE_FUSE float readElem(const IOp& iop, const int x, const int y, const int z) {
        return attnToF32(IOp::Operation::exec(Point{ x, y, z }, iop));
    }

    // ---- generic staging (prologue runs per element, packs 16B stores) -----
    template <int GROUPS, typename IOp>
    FK_DEVICE_FUSE void prefetchTile(const IOp& iop, const int rowBase,
                                     const int seqLen, const int bh,
                                     float (&regs)[GROUPS][8]) {
        // VECTORIZED FAST PATH for the quantized KV prologues: ONE 8-byte
        // load per thread-group (vs 8 scattered byte loads) + the per-token
        // scale hoisted out of the element loop. Generic IOps keep the
        // element-wise path (they may fuse arbitrary compute chains).
        constexpr bool QUANT8 = isFp8KVRead<IOp> || isInt8KVRead<IOp>;
        #pragma unroll
        for (int g = 0; g < GROUPS; ++g) {
            const int idx = (g * THREADS + (int)threadIdx.x) * 8;
            const int r = idx / HEAD_DIM;
            const int c = idx % HEAD_DIM;
            const int row = rowBase + r;
            if (row < seqLen) {
                if constexpr (QUANT8) {
                    const auto& prm = iop.params;
                    const int seq = static_cast<int>(prm.data.dims.height);
                    const float sc = prm.scales[(long)bh * seq + row];
                    const int8_t* base = reinterpret_cast<const int8_t*>(prm.data.data)
                                       + ((long)bh * seq + row) * HEAD_DIM + c;
                    // one coalesced 8-byte load
                    const uint64_t packed = *reinterpret_cast<const uint64_t*>(base);
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        const int8_t b = static_cast<int8_t>((packed >> (8 * e)) & 0xFF);
                        if constexpr (isFp8KVRead<IOp>) {
#ifdef FK_HAS_FP8
                            __nv_fp8_e4m3 f8;
                            f8.__x = static_cast<__nv_fp8_storage_t>(b);
                            regs[g][e] = static_cast<float>(f8) * sc;
#endif
                        } else {
                            regs[g][e] = static_cast<float>(b) * sc;
                        }
                    }
                } else {
                    #pragma unroll
                    for (int e = 0; e < 8; ++e)
                        regs[g][e] = readElem(iop, c + e, row, bh);
                }
            } else {
                #pragma unroll
                for (int e = 0; e < 8; ++e) regs[g][e] = 0.f;
            }
        }
    }

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

    // ---- raw staging: real cp.async, zero-fill on ragged tails -------------
    template <int GROUPS>
    FK_DEVICE_FUSE void cpasyncTile(const uint32_t dstBase /* smem addr + buf */,
                                    const __nv_bfloat16* srcPlane, const int rowBase,
                                    const int seqLen) {
        using namespace attention_mma_detail;
        #pragma unroll
        for (int g = 0; g < GROUPS; ++g) {
            const int idx = (g * THREADS + (int)threadIdx.x) * 8;
            const int r = idx / HEAD_DIM;
            const int c = idx % HEAD_DIM;
            const int row = rowBase + r;
            const bool ok = row < seqLen;
            const int safeRow = ok ? row : (seqLen > 0 ? seqLen - 1 : 0);
            cp_async_16(dstBase + swz(idx * (uint32_t)sizeof(__nv_bfloat16)),
                        srcPlane + (long)safeRow * HEAD_DIM + c, ok ? 16 : 0);
        }
    }

    // ---- quantized staging: cp.async the RAW BYTES (half the traffic) ------
    // SWZ8: XOR-swizzle each 8B chunk by ((row&7)*8) — used by the FP8 QK^T
    // path so direct 4B fragment reads don't bank-conflict (d128 rows are
    // exactly 32 banks wide -> every token would hit bank 0 unswizzled).
    template <int GROUPS, bool SWZ8 = false>
    FK_DEVICE_FUSE void cpasyncQuantTile(const uint32_t dstBase,
                                         const int8_t* srcPlane, const int rowBase,
                                         const int seqLen) {
        using namespace attention_mma_detail;
        #pragma unroll
        for (int g = 0; g < GROUPS; ++g) {
            const int idx = (g * THREADS + (int)threadIdx.x) * 8;
            const int r = idx / HEAD_DIM;
            const int c = idx % HEAD_DIM;
            const int row = rowBase + r;
            const bool ok = row < seqLen;
            const int safeRow = ok ? row : (seqLen > 0 ? seqLen - 1 : 0);
            // unswizzled byte staging: 8B per thread-group
            uint32_t off = (uint32_t)idx;
            if constexpr (SWZ8) off ^= (uint32_t)((r & 7) * 8);
            cp_async_8(dstBase + off,
                       srcPlane + (long)safeRow * HEAD_DIM + c, ok ? 8 : 0);
        }
    }

    /* smem bytes -> swizzled bf16 smem, dequantizing in-register. scales
       read from global (coalesced; BLOCK_KV floats per tile). */
    template <int GROUPS, bool FP8>
    FK_DEVICE_FUSE void dequantTile(char* smemBytes, const uint32_t stageOff,
                                    const uint32_t bf16Off, const float* scales,
                                    const int rowBase, const int seqLen) {
        using attention_mma_detail::Bf16x8;
        #pragma unroll
        for (int g = 0; g < GROUPS; ++g) {
            const int idx = (g * THREADS + (int)threadIdx.x) * 8;
            const int r = idx / HEAD_DIM;
            const int row = rowBase + r;
            const float sc = (row < seqLen) ? scales[row] : 0.f;
            const uint64_t packed =
                *reinterpret_cast<const uint64_t*>(smemBytes + stageOff + idx);
            Bf16x8 out;
            #pragma unroll
            for (int h = 0; h < 4; ++h) {
                float lo, hi;
                const int8_t b0 = static_cast<int8_t>((packed >> (16 * h)) & 0xFF);
                const int8_t b1 = static_cast<int8_t>((packed >> (16 * h + 8)) & 0xFF);
                if constexpr (FP8) {
#ifdef FK_HAS_FP8
                    __nv_fp8_e4m3 f0, f1;
                    f0.__x = static_cast<__nv_fp8_storage_t>(b0);
                    f1.__x = static_cast<__nv_fp8_storage_t>(b1);
                    lo = static_cast<float>(f0) * sc;
                    hi = static_cast<float>(f1) * sc;
#else
                    lo = hi = 0.f;
#endif
                } else {
                    lo = static_cast<float>(b0) * sc;
                    hi = static_cast<float>(b1) * sc;
                }
                out.h[h] = __float22bfloat162_rn({ lo, hi });
            }
            *reinterpret_cast<Bf16x8*>(
                smemBytes + bf16Off + swz(idx * (uint32_t)sizeof(__nv_bfloat16))) = out;
        }
    }

    // ================= FP8 tensor-core QK^T (FK_ENABLE_FP8_QK) ==============
    /* Fused S/P register tile. The fp32 scores (s) and the packed bf16x2
     * probabilities (p) used to live in SEPARATE arrays (32 + 16 regs at
     * BKV=64), but their live ranges only overlap inside softmaxTile, and
     * the pack is strictly in-place-safe: iteration mkv writes p bytes
     * [8*mkv, 8*mkv+8) while s reads touch [16*mkv, 16*mkv+16) — the write
     * head never catches the read head (only mkv==0 overlaps, and there the
     * statement order consumes s[0..1] before overwriting them). Overlaying
     * p on the first half of s saves BLOCK_KV/4 regs/thread (16 at BKV=64). */
    union SPTile {
        float    s[BLOCK_KV / MMA_N][4];   // QK^T scores / exp(s - m), fp32
        uint32_t p[BLOCK_KV / MMA_K][4];   // packed bf16x2 P (first half of s)
    };

    // Q8 stage layout (Q-phase only; dead once qReg8/qs are in registers):
    //   [BLOCK_Q*STRIDE_B, +BLOCK_Q*HEAD_DIM)        row-major e4m3 Q bytes
    //   [.. +BLOCK_Q*4)                              per-row scales (fp32)
    static constexpr uint32_t Q8_OFF = BLOCK_Q * STRIDE_B;
    static constexpr uint32_t QSC_OFF = Q8_OFF + BLOCK_Q * HEAD_DIM;

#ifdef FK_HAS_FP8
    /* Per-row symmetric e4m3 quantization of the staged Q tile (each warp
       quantizes its own WARP_Q rows; warp-synchronous, no block barrier). */
    FK_DEVICE_FUSE void quantizeQTile(char* smemBytes, const int warpId,
                                      const int laneId) {
        constexpr int EPL = HEAD_DIM / 32;     // elements per lane per row
        float* qScalesArr = reinterpret_cast<float*>(smemBytes + QSC_OFF);
        for (int r = 0; r < WARP_Q; ++r) {
            const int row = warpId * WARP_Q + r;   // block-local row
            float vals[EPL];
            float am = 0.f;
            #pragma unroll
            for (int e = 0; e < EPL; ++e) {
                const int c = laneId * EPL + e;
                const __nv_bfloat16 b = *reinterpret_cast<const __nv_bfloat16*>(
                    smemBytes + swz((uint32_t)(row * HEAD_DIM + c) * 2u));
                vals[e] = __bfloat162float(b);
                am = fmaxf(am, fabsf(vals[e]));
            }
            #pragma unroll
            for (int off = 16; off; off >>= 1)
                am = fmaxf(am, __shfl_xor_sync(0xFFFFFFFFu, am, off));
            const float sc = am > 0.f ? am / 448.f : 1.f;
            const float inv = am > 0.f ? 448.f / am : 0.f;
            if (laneId == 0) qScalesArr[row] = sc;
            #pragma unroll
            for (int e = 0; e < EPL; ++e) {
                const __nv_fp8_e4m3 f8{ vals[e] * inv };
                smemBytes[Q8_OFF + (uint32_t)(row * HEAD_DIM + laneId * EPL + e)] =
                    static_cast<char>(f8.__x);
            }
        }
        __syncwarp();
    }

    /* A-fragments for kind::f8f6f4 m16n8k32 from the row-major Q8 stage:
       a0 = row g     dims [4t, 4t+4)   a2 = row g     dims [4t+16, ..)
       a1 = row g+8   dims [4t, 4t+4)   a3 = row g+8   dims [4t+16, ..)
       (t = lane%4, g = lane/4; validated vs fp64 oracle in
       spikes/spike_fp8_qk_mapping.cu). Also pulls the 2 per-row scales. */
    FK_DEVICE_FUSE void loadQF8Frags(const char* smemBytes,
                                     uint32_t (&qReg8)[WARP_Q / MMA_M][HEAD_DIM / 32][4],
                                     float (&qs)[WARP_Q / MMA_M][2],
                                     const int warpId, const int laneId) {
        const float* qScalesArr = reinterpret_cast<const float*>(smemBytes + QSC_OFF);
        const int g = laneId >> 2, tig = laneId & 3;
        #pragma unroll
        for (int mq = 0; mq < WARP_Q / MMA_M; ++mq) {
            const int rowA = warpId * WARP_Q + mq * MMA_M + g;
            const int rowB = rowA + 8;
            qs[mq][0] = qScalesArr[rowA];
            qs[mq][1] = qScalesArr[rowB];
            #pragma unroll
            for (int md = 0; md < HEAD_DIM / 32; ++md) {
                const uint32_t a = Q8_OFF + (uint32_t)(rowA * HEAD_DIM + md * 32 + 4 * tig);
                const uint32_t b = Q8_OFF + (uint32_t)(rowB * HEAD_DIM + md * 32 + 4 * tig);
                qReg8[mq][md][0] = *reinterpret_cast<const uint32_t*>(smemBytes + a);
                qReg8[mq][md][1] = *reinterpret_cast<const uint32_t*>(smemBytes + b);
                qReg8[mq][md][2] = *reinterpret_cast<const uint32_t*>(smemBytes + a + 16);
                qReg8[mq][md][3] = *reinterpret_cast<const uint32_t*>(smemBytes + b + 16);
            }
        }
    }

    /* QK^T on raw e4m3: stream B-fragments (2x u32 = 8 tokens' k-chunks)
       straight from the SWZ8-swizzled byte stage — no kReg staging, no K
       dequant pass. B mapping: b0 = token (mkv*8+g), dims [4t, 4t+4);
       b1 = +16 dims. After the md accumulation completes for a token
       column block, folds qScale[row]*kScale[col] into the raw scores
       (D cols of lane (g,t) are 2t and 2t+1 — validated in
       spikes/spike_fp8_qk_mapping.cu). */
    FK_DEVICE_FUSE void sMmaFp8(const uint32_t (&qReg8)[WARP_Q / MMA_M][HEAD_DIM / 32][4],
                                const float (&qs)[WARP_Q / MMA_M][2],
                                const char* smemBytes, const uint32_t kStageOff,
                                const float* kScalesPlane, const int offKV,
                                const int seqK,
                                SPTile (&sp)[WARP_Q / MMA_M], const int laneId) {
        using namespace attention_mma_detail;
        const int g = laneId >> 2, tig = laneId & 3;
        #pragma unroll
        for (int mkv = 0; mkv < BLOCK_KV / MMA_N; ++mkv) {
            const int tok = mkv * MMA_N + g;
            const uint32_t rowOff = (uint32_t)(tok * HEAD_DIM);
            const uint32_t sw = (uint32_t)((tok & 7) * 8);
            #pragma unroll
            for (int md = 0; md < HEAD_DIM / 32; ++md) {
                uint32_t b[2];
                const uint32_t base = rowOff + (uint32_t)(md * 32 + 4 * tig);
                b[0] = *reinterpret_cast<const uint32_t*>(
                    smemBytes + kStageOff + (base ^ sw));
                b[1] = *reinterpret_cast<const uint32_t*>(
                    smemBytes + kStageOff + ((base + 16) ^ sw));
                #pragma unroll
                for (int mq = 0; mq < WARP_Q / MMA_M; ++mq)
                    mma_fp8_m16n8k32(qReg8[mq][md], b, sp[mq].s[mkv]);
            }
            // scale fold: cols of this lane within the 8-wide N block.
            // L1-cached global reads (whole block touches BLOCK_KV floats);
            // ragged guard avoids OOB scale reads past seq_k.
            const int c0 = offKV + mkv * MMA_N + 2 * tig;
            const float kc0 = (c0 < seqK) ? kScalesPlane[c0] : 0.f;
            const float kc1 = (c0 + 1 < seqK) ? kScalesPlane[c0 + 1] : 0.f;
            #pragma unroll
            for (int mq = 0; mq < WARP_Q / MMA_M; ++mq) {
                float* r = sp[mq].s[mkv];
                r[0] *= qs[mq][0] * kc0;
                r[1] *= qs[mq][0] * kc1;
                r[2] *= qs[mq][1] * kc0;
                r[3] *= qs[mq][1] * kc1;
            }
        }
    }
#endif // FK_HAS_FP8

    // ---- shared fragment loaders / mma helpers ------------------------------
    FK_DEVICE_FUSE void loadKFrags(const uint32_t base, const uint32_t kThread,
                                   uint32_t (&kReg)[BLOCK_KV / MMA_N][HEAD_DIM / MMA_K][2]) {
        using namespace attention_mma_detail;
        #pragma unroll
        for (int mkv = 0; mkv < BLOCK_KV / MMA_N; ++mkv) {
            #pragma unroll
            for (int md = 0; md < HEAD_DIM / MMA_K; md += 2) {
                uint32_t addr = base + kThread;
                addr += mkv * MMA_N * STRIDE_B;
                addr ^= md * MMA_K * sizeof(__nv_bfloat16);
                ldmatrix_x4(&kReg[mkv][md][0], addr);
            }
        }
    }

    FK_DEVICE_FUSE void loadVFrags(const uint32_t base, const uint32_t vThread,
                                   uint32_t (&vReg)[BLOCK_KV / MMA_K][HEAD_DIM / MMA_N][2]) {
        using namespace attention_mma_detail;
        #pragma unroll
        for (int mkv = 0; mkv < BLOCK_KV / MMA_K; ++mkv) {
            #pragma unroll
            for (int md = 0; md < HEAD_DIM / MMA_N; md += 2) {
                uint32_t addr = base + vThread;
                addr += mkv * MMA_K * STRIDE_B;
                addr ^= md * MMA_N * sizeof(__nv_bfloat16);
                ldmatrix_x4_trans(&vReg[mkv][md][0], addr);
            }
        }
    }

    /* Fused S/P register tile — defined above the FP8 helpers (sMmaFp8 takes
     * SPTile&). See the union doc there. */

    /* mask + online softmax + P pack for one KV tile (shared by both
       schedules). noMask: compile-out the bounds/causal branch for interior
       tiles (the common case at long seq). */
    template <bool NO_MASK>
    FK_DEVICE_FUSE void softmaxTile(SPTile (&sp)[WARP_Q / MMA_M],
                                    float (&oAcc)[WARP_Q / MMA_M][HEAD_DIM / MMA_N][4],
                                    float (&rowMax)[WARP_Q / MMA_M][2],
                                    float (&rowSum)[WARP_Q / MMA_M][2],
                                    const Params& p, const int offKV,
                                    const int qBlockBase, const int warpId,
                                    const int laneId) {
        #pragma unroll
        for (int mq = 0; mq < WARP_Q / MMA_M; ++mq) {
            const int rowA = qBlockBase + warpId * WARP_Q + mq * MMA_M + laneId / 4;
            const int rowB = rowA + 8;

            #pragma unroll
            for (int mkv = 0; mkv < BLOCK_KV / MMA_N; ++mkv) {
                float* r = sp[mq].s[mkv];
                const int colBase = offKV + mkv * MMA_N + (laneId % 4) * 2;
                if constexpr (NO_MASK && (!HAS_SCORE_MOD || IS_ALIBI)) {
                    if constexpr (IS_ALIBI) {
                        // column-only ALiBi (row term cancels in softmax):
                        // keeps the fast path — no sentinels, no row math.
                        const float b0 = p.scoreMod.slope * (float)colBase;
                        const float b1 = p.scoreMod.slope * (float)(colBase + 1);
                        r[0] = r[0] * p.scale + b0;
                        r[1] = r[1] * p.scale + b1;
                        r[2] = r[2] * p.scale + b0;
                        r[3] = r[3] * p.scale + b1;
                    } else {
                        #pragma unroll
                        for (int e = 0; e < 4; ++e) r[e] *= p.scale;
                    }
                } else {
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const int col = colBase + (e & 1);
                        const int row = (e < 2) ? rowA : rowB;
                        bool dead;
                        if constexpr (NO_MASK) { dead = false; }
                        else { dead = (col >= p.seq_k) || (p.causal && col > row); }
                        float s = r[e] * p.scale;
                        if constexpr (IS_ALIBI) {
                            // same column-only form as the fast path (exact:
                            // both branches shift each row by -slope*row).
                            s += p.scoreMod.slope * (float)col;
                        } else if constexpr (HAS_SCORE_MOD) {
                            // flex score_mod / mask_mod: AFTER scaling
                            if (!dead) s = p.scoreMod(s, row, col);
                        }
                        r[e] = dead ? -FLT_MAX : s;
                    }
                }
            }

            float mNew[2] = { -FLT_MAX, -FLT_MAX };
            #pragma unroll
            for (int mkv = 0; mkv < BLOCK_KV / MMA_N; ++mkv) {
                const float* r = sp[mq].s[mkv];
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
            corr[0] = attention_mma_detail::fastExp(rowMax[mq][0] - mNew[0]);
            corr[1] = attention_mma_detail::fastExp(rowMax[mq][1] - mNew[1]);
            // FA4-style rescale skip: at long seq most KV tiles do NOT move
            // the running max (rowMax == mNew -> corr == 1), so the
            // HEAD_DIM/MMA_N*4 FMULs over oAcc are identity work. The
            // condition is uniform within each lane quad (rows are shared
            // across laneId%4 after the shfl reduction), so divergence cost
            // is one predicated branch.
            if (rowMax[mq][0] != mNew[0] || rowMax[mq][1] != mNew[1]) {
                #pragma unroll
                for (int md = 0; md < HEAD_DIM / MMA_N; ++md) {
                    oAcc[mq][md][0] *= corr[0];
                    oAcc[mq][md][1] *= corr[0];
                    oAcc[mq][md][2] *= corr[1];
                    oAcc[mq][md][3] *= corr[1];
                }
            }
            rowMax[mq][0] = mNew[0];
            rowMax[mq][1] = mNew[1];

            float lNew[2] = {};
            #pragma unroll
            for (int mkv = 0; mkv < BLOCK_KV / MMA_N; ++mkv) {
                // Read scores BEFORE packing: iteration mkv packs into union
                // bytes [8*mkv, 8*mkv+8) while reading s bytes
                // [16*mkv, 16*mkv+16) — the pack write head never reaches
                // unread scores (s[mkv'] for mkv' > mkv starts at byte
                // 16*mkv+16 > 8*mkv+8). exp values live only in `e` locals;
                // s is dead after this loop, so nothing is written back.
                const float* r = sp[mq].s[mkv];
                float e[4];
                if constexpr (NO_MASK && (!HAS_SCORE_MOD || IS_ALIBI)) {
                    // ALiBi fast path qualifies too: column-only bias adds
                    // no -FLT_MAX sentinels on interior tiles.
                    e[0] = attention_mma_detail::fastExp(r[0] - mNew[0]);
                    e[1] = attention_mma_detail::fastExp(r[1] - mNew[0]);
                    e[2] = attention_mma_detail::fastExp(r[2] - mNew[1]);
                    e[3] = attention_mma_detail::fastExp(r[3] - mNew[1]);
                } else {
                    e[0] = (r[0] == -FLT_MAX) ? 0.f : attention_mma_detail::fastExp(r[0] - mNew[0]);
                    e[1] = (r[1] == -FLT_MAX) ? 0.f : attention_mma_detail::fastExp(r[1] - mNew[0]);
                    e[2] = (r[2] == -FLT_MAX) ? 0.f : attention_mma_detail::fastExp(r[2] - mNew[1]);
                    e[3] = (r[3] == -FLT_MAX) ? 0.f : attention_mma_detail::fastExp(r[3] - mNew[1]);
                }
                lNew[0] += e[0] + e[1];
                lNew[1] += e[2] + e[3];

                // Store through the union member (NOT a reinterpret_cast'ed
                // pointer): reads (.s) and writes (.p) share the same union
                // lvalue base, so the compiler cannot reorder the pack store
                // ahead of the score loads via type-based alias analysis.
                sp[mq].p[mkv / 2][(mkv % 2) * 2]     = attention_mma_detail::packBf162(e[0], e[1]);
                sp[mq].p[mkv / 2][(mkv % 2) * 2 + 1] = attention_mma_detail::packBf162(e[2], e[3]);
            }
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
                lNew[half] += __shfl_xor_sync(0xFFFFFFFFu, lNew[half], 1);
                lNew[half] += __shfl_xor_sync(0xFFFFFFFFu, lNew[half], 2);
                rowSum[mq][half] = rowSum[mq][half] * corr[half] + lNew[half];
            }
        }
    }

    FK_DEVICE_FUSE void sMma(const uint32_t (&qReg)[WARP_Q / MMA_M][HEAD_DIM / MMA_K][4],
                             const uint32_t (&kReg)[BLOCK_KV / MMA_N][HEAD_DIM / MMA_K][2],
                             SPTile (&sp)[WARP_Q / MMA_M]) {
        using namespace attention_mma_detail;
        #pragma unroll
        for (int mq = 0; mq < WARP_Q / MMA_M; ++mq)
            #pragma unroll
            for (int mkv = 0; mkv < BLOCK_KV / MMA_N; ++mkv)
                #pragma unroll
                for (int md = 0; md < HEAD_DIM / MMA_K; ++md)
                    mma_m16n8k16(qReg[mq][md], kReg[mkv][md], sp[mq].s[mkv]);
    }

    /* REGISTER-PRESSURE-FUSED QK^T: stream K fragments from smem with
     * ldmatrix.x4 immediately before their mma instead of staging the whole
     * tile (kReg was 64 regs/thread live across the loop -> now 4). ncu
     * showed 154 regs/thread capped occupancy at 3 blocks/SM; this fusion
     * (+V below, +3 smem bufs) unlocks the 4th block. */
    FK_DEVICE_FUSE void sMmaStream(const uint32_t (&qReg)[WARP_Q / MMA_M][HEAD_DIM / MMA_K][4],
                                   const uint32_t base, const uint32_t kThread,
                                   SPTile (&sp)[WARP_Q / MMA_M]) {
        using namespace attention_mma_detail;
        #pragma unroll
        for (int mkv = 0; mkv < BLOCK_KV / MMA_N; ++mkv) {
            #pragma unroll
            for (int md = 0; md < HEAD_DIM / MMA_K; md += 2) {
                // identical x4 pattern to loadKFrags: one mkv row, fragments
                // for md (regs 0-1) and md+1 (regs 2-3).
                uint32_t frag[4];
                uint32_t addr = base + kThread;
                addr += mkv * MMA_N * STRIDE_B;
                addr ^= md * MMA_K * sizeof(__nv_bfloat16);
                ldmatrix_x4(frag, addr);
                #pragma unroll
                for (int mq = 0; mq < WARP_Q / MMA_M; ++mq) {
                    mma_m16n8k16(qReg[mq][md],     frag + 0, sp[mq].s[mkv]);
                    mma_m16n8k16(qReg[mq][md + 1], frag + 2, sp[mq].s[mkv]);
                }
            }
        }
    }

    FK_DEVICE_FUSE void pvMma(const SPTile (&sp)[WARP_Q / MMA_M],
                              const uint32_t (&vReg)[BLOCK_KV / MMA_K][HEAD_DIM / MMA_N][2],
                              float (&oAcc)[WARP_Q / MMA_M][HEAD_DIM / MMA_N][4]) {
        using namespace attention_mma_detail;
        #pragma unroll
        for (int mq = 0; mq < WARP_Q / MMA_M; ++mq)
            #pragma unroll
            for (int md = 0; md < HEAD_DIM / MMA_N; ++md)
                #pragma unroll
                for (int mkv = 0; mkv < BLOCK_KV / MMA_K; ++mkv)
                    mma_m16n8k16(sp[mq].p[mkv], vReg[mkv][md], oAcc[mq][md]);
    }

    /* Same fusion for P·V: stream V fragments (trans ldmatrix.x4) right
     * before use. vReg was another 32-64 live regs. */
    FK_DEVICE_FUSE void pvMmaStream(const SPTile (&sp)[WARP_Q / MMA_M],
                                    const uint32_t base, const uint32_t vThread,
                                    float (&oAcc)[WARP_Q / MMA_M][HEAD_DIM / MMA_N][4]) {
        using namespace attention_mma_detail;
        #pragma unroll
        for (int mkv = 0; mkv < BLOCK_KV / MMA_K; ++mkv) {
            #pragma unroll
            for (int md = 0; md < HEAD_DIM / MMA_N; md += 2) {
                uint32_t frag[4];
                uint32_t addr = base + vThread;
                addr += mkv * MMA_K * STRIDE_B;
                addr ^= md * MMA_N * sizeof(__nv_bfloat16);
                ldmatrix_x4_trans(frag, addr);
                #pragma unroll
                for (int mq = 0; mq < WARP_Q / MMA_M; ++mq) {
                    mma_m16n8k16(sp[mq].p[mkv], frag + 0, oAcc[mq][md]);
                    mma_m16n8k16(sp[mq].p[mkv], frag + 2, oAcc[mq][md + 1]);
                }
            }
        }
    }

    FK_DEVICE_FUSE void writeOut(const float (&oAcc)[WARP_Q / MMA_M][HEAD_DIM / MMA_N][4],
                                 const float (&rowSum)[WARP_Q / MMA_M][2],
                                 const Params& p, const int qBlockBase,
                                 const int warpId, const int laneId, const int bh) {
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
                    storePair(p.o + base, (r[0] * invA) | p.epilogue,
                                          (r[1] * invA) | p.epilogue);
                }
                if (rowB < p.seq_q) {
                    const long base = ((long)bh * p.seq_q + rowB) * HEAD_DIM + col;
                    storePair(p.o + base, (r[2] * invB) | p.epilogue,
                                          (r[3] * invB) | p.epilogue);
                }
            }
        }
    }

    FK_DEVICE_FUSE void storePair(OT* dst, const float a, const float b) {
        if constexpr (std::is_same_v<OT, __nv_bfloat16>) {
            *reinterpret_cast<__nv_bfloat162*>(dst) = __float22bfloat162_rn({ a, b });
        } else {
            dst[0] = attnFromF32<OT>(a);
            dst[1] = attnFromF32<OT>(b);
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

        const uint32_t qThread = swz(((warpId * WARP_Q + laneId % 16) * HEAD_DIM
                                      + (laneId / 16) * 8) * sizeof(__nv_bfloat16));
        const uint32_t kThread = swz(((laneId % 8) * HEAD_DIM
                                      + (laneId / 8) * 8) * sizeof(__nv_bfloat16));
        const uint32_t vThread = swz(((laneId % 16) * HEAD_DIM
                                      + (laneId / 16) * 8) * sizeof(__nv_bfloat16));

        // ---- Q -> smem -> registers (one-time) -----------------------------
        if constexpr (RAW_Q) {
            const __nv_bfloat16* qPlane = p.q.params.data
                                          + (long)bh * p.seq_q * HEAD_DIM;
            cpasyncTile<Q_GROUPS>(smemBase, qPlane, qBlockBase, p.seq_q);
            cp_async_commit();
            cp_async_wait<0>();
        } else {
            float qStage[Q_GROUPS][8];
            prefetchTile<Q_GROUPS>(p.q, qBlockBase, p.seq_q, bh, qStage);
            storeTile<Q_GROUPS>(smemBytes, 0, qStage);
        }
        __syncthreads();

        uint32_t qReg[WARP_Q / MMA_M][HEAD_DIM / MMA_K][4];
        // FP8_QK state (dead/eliminated when the path is off — all uses sit
        // inside `if constexpr (FP8_QK)`).
        uint32_t qReg8[WARP_Q / MMA_M][HEAD_DIM / 32][4];
        float qs[WARP_Q / MMA_M][2];
        if constexpr (FP8_QK) {
#ifdef FK_HAS_FP8
            // quantize the staged bf16 Q per-row to e4m3 + pull A-fragments.
            quantizeQTile(smemBytes, warpId, laneId);
            loadQF8Frags(smemBytes, qReg8, qs, warpId, laneId);
#endif
        } else {
            #pragma unroll
            for (int mq = 0; mq < WARP_Q / MMA_M; ++mq) {
                #pragma unroll
                for (int md = 0; md < HEAD_DIM / MMA_K; ++md) {
                    uint32_t addr = smemBase + qThread;
                    addr += mq * MMA_M * STRIDE_B;
                    addr ^= md * MMA_K * sizeof(__nv_bfloat16);
                    ldmatrix_x4(qReg[mq][md], addr);
                }
            }
        }
        __syncthreads();

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
        // tiles fully inside [0, seq_k) AND fully below the causal diagonal
        // of every row in this block need no masking at all.
        const int firstRowOfBlock = qBlockBase;  // min row over all warps
        const auto tileNeedsMask = [&](const int offKV) {
            if (offKV + BLOCK_KV > p.seq_k) return true;               // ragged
            if (p.causal && offKV + BLOCK_KV - 1 > firstRowOfBlock) return true;
            return false;
        };
        // BLOCK SPARSITY: inactive KV tiles skip loads AND math entirely.
        // commit/wait groups still run (possibly empty) to keep counting sane.
        const auto tileActive = [&](const int offKV) -> bool {
            if (offKV >= kvEnd) return false;
            if (p.sparse.mask == nullptr) return true;
            const int qb = qBlockBase / p.sparse.maskBQ;
            const int kb = offKV / p.sparse.maskBKV;
            return p.sparse.mask[((long)bh * p.sparse.nQBlocks + qb)
                                 * p.sparse.nKVBlocks + kb] != 0;
        };
        // ITERATION-RANGE TRIM: with a mask, scan this q-block's row once and
        // iterate only [itBegin, itEnd) — skipped iterations otherwise still
        // pay syncthreads + empty commit overhead (measured: a w=512 sliding
        // window at s4096 wastes ~85% of iterations without this).
        int itBegin = 0, itEnd = numIter;
        if (p.sparse.mask != nullptr) {
            const int qb = qBlockBase / p.sparse.maskBQ;
            const unsigned char* row = p.sparse.mask
                + ((long)bh * p.sparse.nQBlocks + qb) * p.sparse.nKVBlocks;
            const int tilesPerMask = p.sparse.maskBKV / BLOCK_KV;
            int first = -1, last = -1;
            for (int kb = 0; kb < p.sparse.nKVBlocks; ++kb) {
                if (row[kb]) { if (first < 0) first = kb; last = kb; }
            }
            if (first < 0) {
                itBegin = itEnd = 0;     // fully masked row -> zero output
            } else {
                itBegin = ::min(numIter, first * tilesPerMask);
                itEnd = ::min(numIter, (last + 1) * tilesPerMask);
            }
        }

        if constexpr (RAW_KV) {
            // ============= cp.async schedule (fa-5090 v5 staggering) =========
            // K double-buffered, V single-buffered; K[kv+1] is issued right
            // after the QK^T mma so it streams during softmax + PV.
            const __nv_bfloat16* kPlane = p.k.params.data + (long)bh * p.seq_k * HEAD_DIM;
            const __nv_bfloat16* vPlane = p.v.params.data + (long)bh * p.seq_k * HEAD_DIM;
            const auto kBuf = [&](int i) { return (uint32_t)(i % 2) * KV_BUF_B; };
            const uint32_t vBufOff = 2 * KV_BUF_B;  // single V buffer

            // prefetch first in-range tile
            if (itBegin < itEnd && tileActive(itBegin * BLOCK_KV)) {
                cpasyncTile<KV_GROUPS>(smemBase + kBuf(itBegin), kPlane,
                                       itBegin * BLOCK_KV, p.seq_k);
            }
            cp_async_commit();

            for (int kv = itBegin; kv < itEnd; ++kv) {
                const int offKV = kv * BLOCK_KV;
                const bool act = tileActive(offKV);

                // V uses a single buffer: previous PV must be done.
                __syncthreads();
                if (act) {
                    cpasyncTile<KV_GROUPS>(smemBase + vBufOff, vPlane, offKV, p.seq_k);
                }
                cp_async_commit();

                // wait K[kv] (1 group outstanding: V[kv])
                cp_async_wait<1>();
                __syncthreads();

                SPTile sp[WARP_Q / MMA_M] = {};
                if (act) {
                    uint32_t kReg[BLOCK_KV / MMA_N][HEAD_DIM / MMA_K][2];
                    loadKFrags(smemBase + kBuf(kv), kThread, kReg);
                    sMma(qReg, kReg, sp);
                }

                // prefetch K[kv+1] (overlaps softmax + PV, v5 staggering)
                if (kv + 1 < itEnd && tileActive(offKV + BLOCK_KV)) {
                    cpasyncTile<KV_GROUPS>(smemBase + kBuf(kv + 1), kPlane,
                                           offKV + BLOCK_KV, p.seq_k);
                }
                cp_async_commit();

                if (act) {
                    if (tileNeedsMask(offKV)) {
                        softmaxTile<false>(sp, oAcc, rowMax, rowSum, p, offKV,
                                           qBlockBase, warpId, laneId);
                    } else {
                        softmaxTile<true>(sp, oAcc, rowMax, rowSum, p, offKV,
                                          qBlockBase, warpId, laneId);
                    }
                }

                // wait V[kv] (1 group outstanding: K[kv+1])
                cp_async_wait<1>();
                __syncthreads();

                if (act) {
                    // stream V frags ldmatrix->mma (no vReg staging)
                    pvMmaStream(sp, smemBase + vBufOff, vThread, oAcc);
                }
            }
        } else if constexpr (QUANT_KV) {
            // ====== quantized cp.async schedule: stream RAW BYTES (2x less
            // global traffic than bf16), dequantize smem->smem in-register.
            // Same v5 staggering as the raw path.
            constexpr bool IS_FP8 = isFp8KVRead<KIOp>;
            const int seqK = static_cast<int>(p.k.params.data.dims.height);
            const int8_t* kPlane = reinterpret_cast<const int8_t*>(p.k.params.data.data)
                                   + (long)bh * seqK * HEAD_DIM;
            const int8_t* vPlane = reinterpret_cast<const int8_t*>(p.v.params.data.data)
                                   + (long)bh * seqK * HEAD_DIM;
            const float* kScales = p.k.params.scales + (long)bh * seqK;
            const float* vScales = p.v.params.scales + (long)bh * seqK;

            const auto kBuf = [&](int i) { return (uint32_t)(i % 2) * KV_BUF_B; };
            const uint32_t vBufOff = 2 * KV_BUF_B;
            // byte staging area after the 4 bf16 buffers
            const uint32_t stageBase = 4 * KV_BUF_B;
            const auto kStage = [&](int i) {
                return stageBase + (uint32_t)(i % 2) * (KV_BUF_B / 2); };
            const uint32_t vStage = stageBase + 2 * (KV_BUF_B / 2);

            // prefetch first in-range tile bytes (FP8_QK: SWZ8-swizzled K so
            // sMmaFp8's direct 4B fragment reads don't bank-conflict)
            if (itBegin < itEnd && tileActive(itBegin * BLOCK_KV)) {
                cpasyncQuantTile<KV_GROUPS, FP8_QK>(smemBase + kStage(itBegin), kPlane,
                                                    itBegin * BLOCK_KV, p.seq_k);
            }
            cp_async_commit();

            for (int kv = itBegin; kv < itEnd; ++kv) {
                const int offKV = kv * BLOCK_KV;
                const bool act = tileActive(offKV);

                __syncthreads();
                if (act) {
                    cpasyncQuantTile<KV_GROUPS>(smemBase + vStage, vPlane, offKV, p.seq_k);
                }
                cp_async_commit();

                cp_async_wait<1>();   // K bytes ready
                __syncthreads();
                if constexpr (!FP8_QK) {
                    // bf16 fallback: dequant K smem->smem, then ldmatrix
                    if (act) {
                        dequantTile<KV_GROUPS, IS_FP8>(smemBytes, kStage(kv), kBuf(kv),
                                                       kScales, offKV, p.seq_k);
                    }
                    __syncthreads();
                }

                SPTile sp[WARP_Q / MMA_M] = {};
                if (act) {
                    if constexpr (FP8_QK) {
#ifdef FK_HAS_FP8
                        // QK^T straight on the raw e4m3 bytes (no K dequant)
                        sMmaFp8(qReg8, qs, smemBytes, kStage(kv), kScales,
                                offKV, p.seq_k, sp, laneId);
#endif
                    } else {
                        uint32_t kReg[BLOCK_KV / MMA_N][HEAD_DIM / MMA_K][2];
                        loadKFrags(smemBase + kBuf(kv), kThread, kReg);
                        sMma(qReg, kReg, sp);
                    }
                }

                if (kv + 1 < itEnd && tileActive(offKV + BLOCK_KV)) {
                    cpasyncQuantTile<KV_GROUPS, FP8_QK>(smemBase + kStage(kv + 1), kPlane,
                                                        offKV + BLOCK_KV, p.seq_k);
                }
                cp_async_commit();

                if (act) {
                    if (tileNeedsMask(offKV)) {
                        softmaxTile<false>(sp, oAcc, rowMax, rowSum, p, offKV,
                                           qBlockBase, warpId, laneId);
                    } else {
                        softmaxTile<true>(sp, oAcc, rowMax, rowSum, p, offKV,
                                          qBlockBase, warpId, laneId);
                    }
                }

                cp_async_wait<1>();   // V bytes ready
                __syncthreads();
                if (act) {
                    dequantTile<KV_GROUPS, IS_FP8>(smemBytes, vStage, vBufOff,
                                                   vScales, offKV, p.seq_k);
                }
                __syncthreads();

                if (act) {
                    uint32_t vReg[BLOCK_KV / MMA_K][HEAD_DIM / MMA_N][2];
                    loadVFrags(smemBase + vBufOff, vThread, vReg);
                    pvMma(sp, vReg, oAcc);
                }
            }
        } else {
            // ============== register-prefetch schedule (fused prologues) =====
            const auto kBuf = [](int i) -> uint32_t { return (i % 2) * KV_BUF_B; };
            const auto vBuf = [](int i) -> uint32_t { return (2 + (i % 2)) * KV_BUF_B; };

            float kPre[KV_GROUPS][8], vPre[KV_GROUPS][8];
            if (itBegin < itEnd) {
                prefetchTile<KV_GROUPS>(p.k, itBegin * BLOCK_KV, p.seq_k, bh, kPre);
                prefetchTile<KV_GROUPS>(p.v, itBegin * BLOCK_KV, p.seq_k, bh, vPre);
                storeTile<KV_GROUPS>(smemBytes, kBuf(itBegin), kPre);
                storeTile<KV_GROUPS>(smemBytes, vBuf(itBegin), vPre);
            }
            __syncthreads();

            for (int kv = itBegin; kv < itEnd; ++kv) {
                const int offKV = kv * BLOCK_KV;
                const bool act = tileActive(offKV);
                const bool nextAct = (kv + 1) < itEnd && tileActive(offKV + BLOCK_KV);
                if (nextAct) {
                    prefetchTile<KV_GROUPS>(p.k, offKV + BLOCK_KV, p.seq_k, bh, kPre);
                    prefetchTile<KV_GROUPS>(p.v, offKV + BLOCK_KV, p.seq_k, bh, vPre);
                }

                if (act) {
                    uint32_t kReg[BLOCK_KV / MMA_N][HEAD_DIM / MMA_K][2];
                    loadKFrags(smemBase + kBuf(kv), kThread, kReg);

                    SPTile sp[WARP_Q / MMA_M] = {};
                    sMma(qReg, kReg, sp);

                    if (tileNeedsMask(offKV)) {
                        softmaxTile<false>(sp, oAcc, rowMax, rowSum, p, offKV,
                                           qBlockBase, warpId, laneId);
                    } else {
                        softmaxTile<true>(sp, oAcc, rowMax, rowSum, p, offKV,
                                          qBlockBase, warpId, laneId);
                    }

                    uint32_t vReg[BLOCK_KV / MMA_K][HEAD_DIM / MMA_N][2];
                    loadVFrags(smemBase + vBuf(kv), vThread, vReg);
                    pvMma(sp, vReg, oAcc);
                }

                if (nextAct) {
                    storeTile<KV_GROUPS>(smemBytes, kBuf(kv + 1), kPre);
                    storeTile<KV_GROUPS>(smemBytes, vBuf(kv + 1), vPre);
                }
                __syncthreads();
            }
        }

        writeOut(oAcc, rowSum, p, qBlockBase, warpId, laneId, bh);
    }
};

template <typename OT, int HEAD_DIM, typename QIOp, typename KIOp, typename VIOp,
          typename EpilogueIOp, int BLOCK_Q, int BLOCK_KV, int NUM_WARPS,
          typename ScoreModOp>
__global__ void launchFlashAttentionMmaDPP_Kernel(
        const __grid_constant__ typename FlashAttentionMmaDPP<
            OT, HEAD_DIM, QIOp, KIOp, VIOp, EpilogueIOp,
            BLOCK_Q, BLOCK_KV, NUM_WARPS, ScoreModOp>::Params params) {
    FlashAttentionMmaDPP<OT, HEAD_DIM, QIOp, KIOp, VIOp, EpilogueIOp,
                         BLOCK_Q, BLOCK_KV, NUM_WARPS, ScoreModOp>::exec(params);
}

/* IOp-first API. The DPP auto-selects cp.async streaming (raw bf16
 * prologues) or register prefetch (fused prologues) at compile time, and
 * auto-tunes tile sizes per schedule (pass BLOCK_Q/BLOCK_KV > 0 to
 * override): raw path prefers BQ64/BKV64 (deep cp.async overlap; with the
 * S/P union d128 fits at 174 regs and BKV64 beats BKV32 by ~13%),
 * fused-prologue path prefers BQ128/BKV32 (register headroom). */
template <int HEAD_DIM, int BLOCK_Q = 0, int BLOCK_KV = 0, int NUM_WARPS = 4,
          typename OT = float, typename QIOp, typename KIOp, typename VIOp,
          typename EpilogueIOp = AttentionIdentityEpilogue,
          typename ScoreModOp = NoScoreMod>
inline void executeFlashAttentionMma(
        const QIOp& q, const KIOp& k, const VIOp& v, OT* o,
        const int batchHeads, const int seqQ, const int seqK,
        const bool causal, Stream_<ParArch::GPU_NVIDIA>& stream,
        const float scaleOverride = -1.f, const EpilogueIOp& epilogue = {},
        const ScoreModOp& scoreMod = {}, const BlockSparsity& sparse = {}) {
    constexpr bool RAW = isRawBf16Read<KIOp> && isRawBf16Read<VIOp>;
    constexpr int BQ = BLOCK_Q > 0 ? BLOCK_Q
                                   : (RAW ? 64 : (HEAD_DIM <= 64 ? 128 : 64));
    constexpr int BKV = BLOCK_KV > 0 ? BLOCK_KV : (RAW ? 64 : 32);
    using DPP = FlashAttentionMmaDPP<OT, HEAD_DIM, QIOp, KIOp, VIOp,
                                     EpilogueIOp, BQ, BKV, NUM_WARPS, ScoreModOp>;
    const float scale = scaleOverride > 0.f ? scaleOverride
                                            : rsqrtf(static_cast<float>(HEAD_DIM));
    if (sparse.mask != nullptr) {
        // block-sparse mask granularity must contain whole kernel tiles
        if (sparse.maskBQ % BQ != 0 || sparse.maskBKV % BKV != 0) {
            throw std::invalid_argument(
                "BlockSparsity: maskBQ/maskBKV must be multiples of the kernel "
                "tiles (BQ=" + std::to_string(BQ) + ", BKV=" + std::to_string(BKV) + ")");
        }
    }
    const typename DPP::Params params{ q, k, v, o, seqQ, seqK, scale, causal,
                                       epilogue, scoreMod, sparse };
    const dim3 grid((seqQ + BQ - 1) / BQ, batchHeads, 1);
    const dim3 block(DPP::THREADS, 1, 1);
    const int smemBytes = DPP::SMEM_BYTES;
    auto* kernel = launchFlashAttentionMmaDPP_Kernel<OT, HEAD_DIM, QIOp, KIOp, VIOp,
                                                     EpilogueIOp, BQ, BKV, NUM_WARPS,
                                                     ScoreModOp>;
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
