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
 * The kernel inspects the prologue IOp types AT COMPILE TIME for tile
 * selection and feature gating, but every global-memory Q/K/V load now
 * goes THROUGH the provided Read IOp. Raw bf16 reads, dequantizing reads,
 * and fused read.then(...) chains all stage tiles through the same IOp
 * entry point before the MMA pipeline consumes them. Epilogue IOps always
 * run in-register on the output before the single global write.
 *
 * Ladder: v1 mma.sync + FA-2 warp split + online softmax on fp32 regs;
 * v2 XOR-swizzled smem (bank-conflict-free stores + ldmatrix);
 * v3/v5 pipelining (cp.async groups or register prefetch);
 * v4 ldmatrix.x4 for K and V. */

#include <fused_kernel/algorithms/attention/flash_attention.h>

#if defined(__NVCC__)

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

inline auto makeAttentionPartialRead(const float* data, const int batchHeads,
                                     const int seq, const int splits,
                                     const int elemsPerSplit) {
    const RawPtr<ND::_3D, float> ptr{
        const_cast<float*>(data),
        PtrDims<ND::_3D>(static_cast<uint>(elemsPerSplit), static_cast<uint>(seq),
                         static_cast<uint>(batchHeads * splits), 1,
                         static_cast<uint>(elemsPerSplit * sizeof(float))) };
    return PerThreadRead<ND::_3D, float>::build(ptr);
}

inline auto makeAttentionPartialWrite(float* data, const int batchHeads,
                                      const int seq, const int splits,
                                      const int elemsPerSplit) {
    const RawPtr<ND::_3D, float> ptr{
        data,
        PtrDims<ND::_3D>(static_cast<uint>(elemsPerSplit), static_cast<uint>(seq),
                         static_cast<uint>(batchHeads * splits), 1,
                         static_cast<uint>(elemsPerSplit * sizeof(float))) };
    return PerThreadWrite<ND::_3D, float>::build(ptr);
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

struct alignas(16) Bf16x8 { __nv_bfloat162 h[4]; };

// two-phase-lookup-safe fast exp (visible at template definition time).
__device__ __forceinline__ float fastExp(const float x) {
#if defined(__CUDA_ARCH__)
    return __expf(x);
#else
    return 0.f;
#endif
}

/* raw ex2.approx (2^x). __expf(x) lowers to FMUL(x, log2e) + MUFU.EX2 —
 * folding log2e into the softmax scale ONCE removes that per-element FMUL
 * (FA does the same). Inline asm so it does not depend on --use_fast_math. */
__device__ __forceinline__ float fastExp2(const float x) {
#if defined(__CUDA_ARCH__)
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
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

template <int HEAD_DIM, typename QIOp, typename KIOp, typename VIOp,
          typename OutIOp,
          int BLOCK_Q = (HEAD_DIM <= 64 ? 128 : 64), int BLOCK_KV = 32,
          int NUM_WARPS = 4, typename ScoreModOp = NoScoreMod>
struct FlashAttentionMmaDPP {
private:
    using SelfType = FlashAttentionMmaDPP<HEAD_DIM, QIOp, KIOp, VIOp,
                                          OutIOp, BLOCK_Q, BLOCK_KV, NUM_WARPS,
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
    static_assert(isAnyWriteType<OutIOp>, "Output must be a Write IOp");

    static constexpr int THREADS = NUM_WARPS * 32;
    static constexpr int WARP_Q = BLOCK_Q / NUM_WARPS;
    static_assert(WARP_Q % 16 == 0, "BLOCK_Q/NUM_WARPS must be a multiple of 16");
    static constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;

    // Prologue classifiers used for tile heuristics / feature gating.
    static constexpr bool RAW_KV = isRawBf16Read<KIOp> && isRawBf16Read<VIOp>;
    static constexpr bool RAW_Q = isRawBf16Read<QIOp>;
    static constexpr bool QUANT_KV =
        (isFp8KVRead<KIOp> && isFp8KVRead<VIOp>) ||
        (isInt8KVRead<KIOp> && isInt8KVRead<VIOp>);
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
    static constexpr int KV_BUFS = 4;
    static constexpr int Q_STAGE_B = 0;
    // DEEP_Q_SMEM: BUILT AND MEASURED OFF. Streaming Q from resident smem
    // with the mkv-outer loop does eliminate qReg, but replays each Q
    // fragment for every K fragment (0.255 ms vs BQ64's 0.216 on d128
    // bh32 s2048 dense). Kept as scaffolding; superseded by DUAL_Q_SMEM.
    static constexpr bool DEEP_Q_SMEM = false;
    // DUAL_Q_SMEM — the bucket-C schedule derived from FA's SASS
    // (HMMA/LDSM 3.20 vs our 1.78; benchmarks/sass/ in fkl_attention).
    // Q resident in smem like DEEP_Q_SMEM, but the QK^T loop is TRANSPOSED:
    // md-pair OUTER, mkv INNER. Per md-pair the mq Q fragment pairs are
    // loaded ONCE and reused across ALL BLOCK_KV columns, while each K
    // fragment feeds mq row-blocks (dual reuse of BOTH operands). ldmatrix
    // count per d128/BQ128/BKV64 tile: Q 16 + K 32 + V 32 = 80 x4 for 256
    // HMMA -> ratio 3.2, exactly FA's. qReg (64 hot regs at mq=2) never
    // exists; live operands inside the loop are 16 Q regs + 4 K regs, so
    // any ptxas spill lands on softmax bookkeeping (cold, inter-phase) —
    // the spill discipline FA's own 60-97 LDL/STL demonstrates is viable.
    // Gated to the deep-tile d128 regime. BUILT AND MEASURED (2026-06-12,
    // spikes/spike_dual_q.cu + abab_dual_q.cu, ABAB interleaved, real data):
    // it reproduces FA's SASS signature EXACTLY — HMMA/LDSM 3.20 (FA 3.20,
    // plain 1.78), spills confined to cold inter-phase state (LDL/STL
    // 40/24, ZERO in the QK^T loop — vs the old BQ128 deep's hot-operand
    // spills) — and accuracy passes vs the fp64 oracle. It still LOSES to
    // plain BQ128/BKV64/NW8 by 3-7% dense / 9-20% causal: at 81920B smem
    // it is hard-capped at 1 CTA/SM (8.3% occupancy), and the recovered
    // issue slots don't cover the lost warp-level parallelism; ncu shows
    // dual IPC 0.71 ~= FA's 0.69, i.e. the ILP discipline WORKS, but FA
    // pairs it with ~5.6 total-instr/HMMA vs our ~10 (softmax bookkeeping
    // dominates our non-mma mix). Lesson: the instruction-ratio gap was
    // necessary but NOT sufficient; the remaining lever is shrinking
    // softmax/mask arithmetic per score, not operand staging. OFF by
    // default; -DFK_FA_DUAL_Q=1 re-enables for experiments.
#ifndef FK_FA_DUAL_Q
#define FK_FA_DUAL_Q 0
#endif
    static constexpr bool DUAL_Q_SMEM =
        FK_FA_DUAL_Q != 0 && RAW_KV && !FP8_QK && HEAD_DIM == 128 &&
        BLOCK_Q >= 128 && WARP_Q >= 32;
    static constexpr int Q_RES_B =
        (DEEP_Q_SMEM || DUAL_Q_SMEM) ? BLOCK_Q * STRIDE_B : 0;
    // FP8_QK: the Q phase additionally holds the e4m3 Q tile + per-row
    // scales NEXT TO the staged bf16 Q (both alive during quantizeQTile).
    static constexpr int Q_PHASE_B = BLOCK_Q * STRIDE_B;
    static constexpr int SMEM_BYTES =
        (Q_PHASE_B > Q_RES_B + KV_BUFS * KV_BUF_B + Q_STAGE_B
             ? Q_PHASE_B
             : Q_RES_B + KV_BUFS * KV_BUF_B + Q_STAGE_B);

    struct Params {
        using PartialWriteIOp =
            decltype(makeAttentionPartialWrite(static_cast<float*>(nullptr), 0, 0, 0, HEAD_DIM + 2));
        QIOp q; KIOp k; VIOp v;
        OutIOp out;
        int seq_q, seq_k;
        float scale;
        bool causal;
        ScoreModOp scoreMod;     // flex-attention score_mod / mask_mod
        BlockSparsity sparse;    // block-sparse tile skipping (nullptr = dense)
        // SPLIT-KV FORWARD (FA-style): when splits > 1, blockIdx.z selects a
        // KV chunk and the kernel writes UNNORMALIZED partials (oAcc, m, l)
        // to `partial` instead of O; a combine kernel reduces them. Sized
        // [bh, seq_q, splits, HEAD_DIM+2]. Saturates the GPU on small grids
        // (bh*numQ << 1 wave) — ncu: FA runs 5.45 waves vs our 0.68 on
        // d128 bh8 s2048, hiding everything; this is that lever.
        PartialWriteIOp partial =
            makeAttentionPartialWrite(nullptr, 0, 0, 0, HEAD_DIM + 2);
        int splits = 1;
    };
    // log2-domain softmax is active for these functor types (see softmaxTile);
    // the split combine must interpret stored row-maxes in the same base.
    static constexpr bool LOG2_DOMAIN = !HAS_SCORE_MOD || IS_ALIBI;

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

    // ================= FP8 tensor-core QK^T (FK_ENABLE_FP8_QK) ==============
    /* Fused S/P register tile. The fp32 scores (s) and the packed bf16x2
     * probabilities (p) used to live in SEPARATE arrays (32 + 16 regs at
     * BKV=64), but their live ranges only overlap inside softmaxTile, and
     * the pack is strictly in-place-safe: iteration mkv writes p bytes
     * [8*mkv, 8*mkv+8) while s reads touch [16*mkv, 16*mkv+16) — the write
     * head never catches the read head (only mkv==0 overlaps, and there the
     * statement order consumes s[0..1] before overwriting them). Overlaying
     * p on the first half of s saves BLOCK_KV/4 regs/thread (16 at BKV=64). */
    template <int SPAN>
    union SPTileT {
        float    s[SPAN / MMA_N][4];   // QK^T scores / exp(s - m), fp32
        uint32_t p[SPAN / MMA_K][4];   // packed bf16x2 P (first half of s)
    };
    using SPTile = SPTileT<BLOCK_KV>;
    // SPLIT-S: measured ineffective everywhere it was tried (d64: allocator
    // already overlaps; d128 deep: replaced by the Q-resident DEEP schedule
    // below, which removes the pressure at the source). Machinery kept.
    static constexpr bool SPLIT_S = false;
    static constexpr int KV_SPAN = SPLIT_S ? BLOCK_KV / 2 : BLOCK_KV;
    using SPSpan = SPTileT<KV_SPAN>;

    // DEEP SCHEDULE (d128 dense bucket — the last FA-wins bucket): FA's
    // d128 kernel spends 255 regs/thread to run BQ128 with mq=2 K-fragment
    // reuse, executing 1.69x FEWER instructions than our BQ64 (ncu: 61.5M
    // vs 104M same shape). Our BQ128/NW4 spills (qReg 64 + oAcc 64 + sp 64
    // > 255). FIX AT THE SOURCE: keep Q RESIDENT in smem for the whole
    // kernel (32KB + 48KB KV bufs = 80KB/block -> 1 CTA/SM, the same
    // low-occupancy/high-ILP design point FA runs at) and STREAM Q
    // fragments per md-pair inside sMmaStream — qReg's 64 regs vanish,
    // the deep tile fits, and each K fragment still serves mq=2 mmas.
    // (DEEP_Q_SMEM and Q_RES_B are declared above with the smem sizing.)

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
    template <bool NO_MASK, int SPAN = BLOCK_KV>
    FK_DEVICE_FUSE void softmaxTile(SPTileT<SPAN> (&sp)[WARP_Q / MMA_M],
                                    float (&oAcc)[WARP_Q / MMA_M][HEAD_DIM / MMA_N][4],
                                    float (&rowMax)[WARP_Q / MMA_M][2],
                                    float (&rowSum)[WARP_Q / MMA_M][2],
                                    const Params& p, const int offKV,
                                    const int qBlockBase, const int warpId,
                                    const int laneId) {
        // LOG2 DOMAIN: __expf lowers to FMUL(x, log2e) + MUFU.EX2. Folding
        // log2e into the scale ONCE turns every per-element exp into a bare
        // ex2.approx (removes BLOCK_KV/2 FMULs per row per tile) — same trick
        // FA uses. Generic score_mods see natural-domain scores, so they keep
        // the e-domain path; ALiBi folds log2e into the slope (exact).
        constexpr bool LOG2D = !HAS_SCORE_MOD || IS_ALIBI;
        constexpr float LOG2E = 1.4426950408889634f;
        const float scl = LOG2D ? p.scale * LOG2E : p.scale;
        // RAW-DOMAIN SOFTMAX (arithmetic-density round, SASS-driven): with no
        // score mod the scale pass is pure overhead — max commutes with a
        // positive scale (max(r)*scl == max(r*scl), bitwise: fp mul is
        // monotone), so we max RAW scores, scale the row max ONCE per half,
        // and fold scale+subtract into a single FFMA inside the exp:
        //   e = ex2(fma(r, scl, -mScaled)).
        // This deletes 4 FMULs + 4 FADDs per mkv per mq (the whole scale
        // loop) from the hot path; FA's SASS shows the same shape (FFMA 0.27
        // vs our old 0.03 per HMMA, FMUL 0.53 vs our 1.02). Masked tiles
        // keep sentinel semantics: dead elements hold raw -FLT_MAX and the
        // exp keeps the explicit ==-FLT_MAX -> 0 guard (an all-dead row with
        // rowMax still -FLT_MAX — empty split-KV chunk, causal row above its
        // first in-range tile — would otherwise see fma(-FLT_MAX, scl,
        // +FLT_MAX*scl) == 0 -> exp == 1 and poison lNew).
        constexpr bool RAW_SMAX = !HAS_SCORE_MOD;
        #pragma unroll
        for (int mq = 0; mq < WARP_Q / MMA_M; ++mq) {
            const int rowA = qBlockBase + warpId * WARP_Q + mq * MMA_M + laneId / 4;
            const int rowB = rowA + 8;
            // FUSED MASK LIMIT (arithmetic-density round 3): the two dead
            // conditions (col >= seq_k, causal && col > row) collapse into
            // ONE per-row column limit hoisted out of the mkv loop:
            //   lim = causal ? min(seq_k, row+1) : seq_k;  dead = col >= lim
            // 1 ISETP per element instead of 2 ISETP + logic ops. Computed
            // only on masked tiles (NO_MASK path never reads it).
            const int limA = p.causal ? ::min(p.seq_k, rowA + 1) : p.seq_k;
            const int limB = p.causal ? ::min(p.seq_k, rowB + 1) : p.seq_k;

            #pragma unroll
            for (int mkv = 0; mkv < SPAN / MMA_N; ++mkv) {
                float* r = sp[mq].s[mkv];
                const int colBase = offKV + mkv * MMA_N + (laneId % 4) * 2;
                if constexpr (RAW_SMAX && NO_MASK) {
                    // raw-domain fast path: nothing to do here — scores stay
                    // raw; scale folds into the exp FFMA below.
                } else if constexpr (RAW_SMAX) {
                    // raw-domain masked: sentinels only, no scale.
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const int col = colBase + (e & 1);
                        if (col >= ((e < 2) ? limA : limB)) r[e] = -FLT_MAX;
                    }
                } else if constexpr (NO_MASK && IS_ALIBI) {
                    // column-only ALiBi (row term cancels in softmax):
                    // keeps the fast path — no sentinels, no row math.
                    const float slope2 = p.scoreMod.slope * LOG2E;
                    const float b0 = slope2 * (float)colBase;
                    const float b1 = slope2 * (float)(colBase + 1);
                    r[0] = r[0] * scl + b0;
                    r[1] = r[1] * scl + b1;
                    r[2] = r[2] * scl + b0;
                    r[3] = r[3] * scl + b1;
                } else {
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const int col = colBase + (e & 1);
                        const int row = (e < 2) ? rowA : rowB;
                        bool dead;
                        if constexpr (NO_MASK) { dead = false; }
                        else { dead = col >= ((e < 2) ? limA : limB); }
                        float s = r[e] * scl;
                        if constexpr (IS_ALIBI) {
                            // same column-only form as the fast path (exact:
                            // both branches shift each row by -slope*row).
                            s += p.scoreMod.slope * LOG2E * (float)col;
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
            for (int mkv = 0; mkv < SPAN / MMA_N; ++mkv) {
                const float* r = sp[mq].s[mkv];
                mNew[0] = fmaxf(mNew[0], fmaxf(r[0], r[1]));
                mNew[1] = fmaxf(mNew[1], fmaxf(r[2], r[3]));
            }
            // mCand = running max merged with this tile's row max (scaled once
            // here under RAW_SMAX so rowMax stays in the scaled log2 domain;
            // writePartial / split combine unchanged).
            float mCand[2];
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
                mNew[half] = fmaxf(mNew[half], __shfl_xor_sync(0xFFFFFFFFu, mNew[half], 1));
                mNew[half] = fmaxf(mNew[half], __shfl_xor_sync(0xFFFFFFFFu, mNew[half], 2));
                const float mTile = RAW_SMAX ? mNew[half] * scl : mNew[half];
                mCand[half] = fmaxf(mTile, rowMax[mq][half]);
            }

            // FA4 §3.1.4 CONDITIONAL RESCALE (threshold τ). Standard online
            // softmax bumps the running max every tile and rescales oAcc by
            // exp(mOld - mCand). FA4's insight: if the max grows by ≤ τ, KEEP
            // the old max — exp(score - mOld) then stays bounded by 2^τ (=256
            // at τ=8), no overflow, and the result is exact after the final
            // /rowSum normalization. This skips the rescale FMULs on far more
            // tiles than the exact-equality test it replaces, and anchors exp
            // arguments to a stable max across long KV runs. τ lives in the
            // active exp domain so the 256 bound holds in both: log2 → 8.0,
            // natural → ln(256). The compare is uniform within each lane quad
            // (rows shared across laneId%4 after the shfl), so the branch is
            // predicated, not divergent. mOld == mCand == -FLT_MAX (fully dead
            // row) yields Δ==0 ≤ τ -> keep -FLT_MAX, corr==EXP(0)==1, skip —
            // identical to the previous sentinel behavior.
            constexpr float TAU = LOG2D ? 8.0f : 5.5451774f;  // log2(256) / ln(256)
            float mUsed[2];
            mUsed[0] = (mCand[0] - rowMax[mq][0] > TAU) ? mCand[0] : rowMax[mq][0];
            mUsed[1] = (mCand[1] - rowMax[mq][1] > TAU) ? mCand[1] : rowMax[mq][1];

            // domain-aware exp: scores are in log2 domain when LOG2D
            const auto EXP = [](const float x) {
                return LOG2D ? attention_mma_detail::fastExp2(x)
                             : attention_mma_detail::fastExp(x);
            };
            float corr[2];
            corr[0] = EXP(rowMax[mq][0] - mUsed[0]);
            corr[1] = EXP(rowMax[mq][1] - mUsed[1]);
            if (rowMax[mq][0] != mUsed[0] || rowMax[mq][1] != mUsed[1]) {
                #pragma unroll
                for (int md = 0; md < HEAD_DIM / MMA_N; ++md) {
                    oAcc[mq][md][0] *= corr[0];
                    oAcc[mq][md][1] *= corr[0];
                    oAcc[mq][md][2] *= corr[1];
                    oAcc[mq][md][3] *= corr[1];
                }
            }
            rowMax[mq][0] = mUsed[0];
            rowMax[mq][1] = mUsed[1];
            // exp below references mNew as the subtracted max; make it the
            // chosen (scaled) running max so e = ex2(fma(r, scl, -mUsed)).
            mNew[0] = mUsed[0];
            mNew[1] = mUsed[1];

            float lNew[2] = {};
            #pragma unroll
            for (int mkv = 0; mkv < SPAN / MMA_N; ++mkv) {
                // Read scores BEFORE packing: iteration mkv packs into union
                // bytes [8*mkv, 8*mkv+8) while reading s bytes
                // [16*mkv, 16*mkv+16) — the pack write head never reaches
                // unread scores (s[mkv'] for mkv' > mkv starts at byte
                // 16*mkv+16 > 8*mkv+8). exp values live only in `e` locals;
                // s is dead after this loop, so nothing is written back.
                const float* r = sp[mq].s[mkv];
                float e[4];
                if constexpr (RAW_SMAX && NO_MASK) {
                    // raw scores: scale+subtract fused in one FFMA per
                    // element (ex2(fma(r, scl, -m))). This is the whole
                    // arithmetic-density win: 1 FFMA replaces FMUL+FADD.
                    // NOTE (round 10, REJECTED): packed bf16x2 exp via
                    // h2exp2 was built and measured — sm_120 has NO packed
                    // MUFU.EX2: h2exp2 lowers to 2 scalar MUFU + extra
                    // F2FP/SHF/LOP3 (SASS total/HMMA 14.6 -> 15.5, MUFU
                    // 40 -> 45), measured -2..-22% across shapes. The
                    // scalar fp32 pipeline below IS the optimal form here.
                    e[0] = EXP(fmaf(r[0], scl, -mNew[0]));
                    e[1] = EXP(fmaf(r[1], scl, -mNew[0]));
                    e[2] = EXP(fmaf(r[2], scl, -mNew[1]));
                    e[3] = EXP(fmaf(r[3], scl, -mNew[1]));
                } else if constexpr (RAW_SMAX) {
                    e[0] = (r[0] == -FLT_MAX) ? 0.f : EXP(fmaf(r[0], scl, -mNew[0]));
                    e[1] = (r[1] == -FLT_MAX) ? 0.f : EXP(fmaf(r[1], scl, -mNew[0]));
                    e[2] = (r[2] == -FLT_MAX) ? 0.f : EXP(fmaf(r[2], scl, -mNew[1]));
                    e[3] = (r[3] == -FLT_MAX) ? 0.f : EXP(fmaf(r[3], scl, -mNew[1]));
                } else if constexpr (NO_MASK && (!HAS_SCORE_MOD || IS_ALIBI)) {
                    // ALiBi fast path qualifies too: column-only bias adds
                    // no -FLT_MAX sentinels on interior tiles.
                    e[0] = EXP(r[0] - mNew[0]);
                    e[1] = EXP(r[1] - mNew[0]);
                    e[2] = EXP(r[2] - mNew[1]);
                    e[3] = EXP(r[3] - mNew[1]);
                } else {
                    e[0] = (r[0] == -FLT_MAX) ? 0.f : EXP(r[0] - mNew[0]);
                    e[1] = (r[1] == -FLT_MAX) ? 0.f : EXP(r[1] - mNew[0]);
                    e[2] = (r[2] == -FLT_MAX) ? 0.f : EXP(r[2] - mNew[1]);
                    e[3] = (r[3] == -FLT_MAX) ? 0.f : EXP(r[3] - mNew[1]);
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
     * (+V below, +3 smem bufs) unlocks the 4th block.
     * SPAN/tokOff: split-S processes the smem tile in KV_SPAN-token spans. */
    template <int SPAN = BLOCK_KV>
    FK_DEVICE_FUSE void sMmaStream(const uint32_t (&qReg)[WARP_Q / MMA_M][HEAD_DIM / MMA_K][4],
                                   const uint32_t base, const uint32_t kThread,
                                   SPTileT<SPAN> (&sp)[WARP_Q / MMA_M],
                                   const int tokOff = 0) {
        using namespace attention_mma_detail;
        #pragma unroll
        for (int mkv = 0; mkv < SPAN / MMA_N; ++mkv) {
            #pragma unroll
            for (int md = 0; md < HEAD_DIM / MMA_K; md += 2) {
                // identical x4 pattern to loadKFrags: one mkv row, fragments
                // for md (regs 0-1) and md+1 (regs 2-3).
                uint32_t frag[4];
                uint32_t addr = base + kThread;
                addr += (tokOff + mkv * MMA_N) * STRIDE_B;
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

    /* HOISTED-ADDRESS variant (arithmetic-density round): kmd[md] =
       (smemBase + kThread) ^ (md*MMA_K*2) precomputed ONCE before the kv
       loop. Valid because the XOR bits (< STRIDE_B) never overlap the
       additive offsets (bufOff / token offsets are STRIDE_B multiples, no
       carries into the XOR field). Every address is then reg + constant,
       folded into the LDSM immediate — the per-fragment LOP3+IADD chains
       disappear from the hot loop (SASS: LOP3 was 2.1/HMMA vs FA 0.086). */
    template <int SPAN = BLOCK_KV>
    FK_DEVICE_FUSE void sMmaStreamH(const uint32_t (&qReg)[WARP_Q / MMA_M][HEAD_DIM / MMA_K][4],
                                    const uint32_t bufOff,
                                    const uint32_t (&kmd)[HEAD_DIM / MMA_K],
                                    SPTileT<SPAN> (&sp)[WARP_Q / MMA_M],
                                    const int tokOff = 0) {
        using namespace attention_mma_detail;
        #pragma unroll
        for (int mkv = 0; mkv < SPAN / MMA_N; ++mkv) {
            #pragma unroll
            for (int md = 0; md < HEAD_DIM / MMA_K; md += 2) {
                uint32_t frag[4];
                ldmatrix_x4(frag, kmd[md] + bufOff
                                  + (tokOff + mkv * MMA_N) * STRIDE_B);
                #pragma unroll
                for (int mq = 0; mq < WARP_Q / MMA_M; ++mq) {
                    mma_m16n8k16(qReg[mq][md],     frag + 0, sp[mq].s[mkv]);
                    mma_m16n8k16(qReg[mq][md + 1], frag + 2, sp[mq].s[mkv]);
                }
            }
        }
    }

    /* DEEP-Q QK^T: BOTH operands streamed from smem (Q resident at smem[0],
       K in the cp.async buffer). Loop order keeps the K fragment in regs
       across all WARP_Q/MMA_M row-blocks (mq=2 at BQ128/NW4 — each ldmatrix
       feeds 2 mmas, FA's instruction-efficiency trick) while Q fragments
       stream per (mq, md-pair) — qReg's 64 regs/thread never exist. */
    FK_DEVICE_FUSE void sMmaStreamDeepQ(const uint32_t qBase, const uint32_t qThread,
                                        const uint32_t kBase, const uint32_t kThread,
                                        SPTile (&sp)[WARP_Q / MMA_M]) {
        using namespace attention_mma_detail;
        #pragma unroll
        for (int mkv = 0; mkv < BLOCK_KV / MMA_N; ++mkv) {
            #pragma unroll
            for (int md = 0; md < HEAD_DIM / MMA_K; md += 2) {
                uint32_t kFrag[4];
                uint32_t kAddr = kBase + kThread;
                kAddr += mkv * MMA_N * STRIDE_B;
                kAddr ^= md * MMA_K * sizeof(__nv_bfloat16);
                ldmatrix_x4(kFrag, kAddr);
                #pragma unroll
                for (int mq = 0; mq < WARP_Q / MMA_M; ++mq) {
                    uint32_t qFrag[4], qFrag2[4];
                    uint32_t qAddr = qBase + qThread;
                    qAddr += mq * MMA_M * STRIDE_B;
                    ldmatrix_x4(qFrag,  qAddr ^ (md       * MMA_K * (int)sizeof(__nv_bfloat16)));
                    ldmatrix_x4(qFrag2, qAddr ^ ((md + 1) * MMA_K * (int)sizeof(__nv_bfloat16)));
                    mma_m16n8k16(qFrag,  kFrag + 0, sp[mq].s[mkv]);
                    mma_m16n8k16(qFrag2, kFrag + 2, sp[mq].s[mkv]);
                }
            }
        }
    }

    /* DUAL-REUSE QK^T (bucket-C, SASS-derived — see DUAL_Q_SMEM above):
       Q resident in smem, loop TRANSPOSED vs sMmaStreamDeepQ: md-pair OUTER,
       mkv INNER. Each Q fragment pair is loaded once per md-pair and serves
       all SPAN/MMA_N columns; each K fragment serves mq row-blocks.
       ADDRESSING: the swizzle XOR (^ md*32, bits 5-7) commutes with the
       additive tile offsets that follow it (mq*MMA_M*STRIDE_B, kBuf
       multiples of KV_BUF_B, token offsets multiples of STRIDE_B — all
       multiples of 256, no carries into bits 5-7). Callers precompute
       qmd[md] = (qBase + qThread) ^ (md*32) and kmd[md] = (smemBase +
       kThread) ^ (md*32) ONCE outside the kv loop; every address below is
       then reg + compile-time constant, which ptxas folds into the LDSM
       offset field — zero LOP3/IADD in the hot loop (the SASS histogram
       showed LOP3 at 1.06/HMMA vs FA's 0.086 before this). */
    template <int SPAN = BLOCK_KV>
    FK_DEVICE_FUSE void sMmaStreamDualQ(const uint32_t kBufOff,
                                        const uint32_t (&qmd)[HEAD_DIM / MMA_K],
                                        const uint32_t (&kmd)[HEAD_DIM / MMA_K],
                                        SPTileT<SPAN> (&sp)[WARP_Q / MMA_M],
                                        const int tokOff = 0) {
        using namespace attention_mma_detail;
        // K-fragment register double-buffer: at 1 CTA/SM (this schedule's
        // occupancy point, same as FA's) there is no second warp to hide
        // LDSM->HMMA latency, so the NEXT fragment's ldmatrix must issue
        // BEFORE the current mmas (CUTLASS mainloop discipline).
        #pragma unroll
        for (int md = 0; md < HEAD_DIM / MMA_K; md += 2) {
            uint32_t qFrag[WARP_Q / MMA_M][2][4];
            #pragma unroll
            for (int mq = 0; mq < WARP_Q / MMA_M; ++mq) {
                const uint32_t rowOff = mq * MMA_M * STRIDE_B;
                ldmatrix_x4(qFrag[mq][0], qmd[md]     + rowOff);
                ldmatrix_x4(qFrag[mq][1], qmd[md + 1] + rowOff);
            }
            uint32_t kFrag[2][4];
            ldmatrix_x4(kFrag[0], kmd[md] + kBufOff + tokOff * STRIDE_B);
            #pragma unroll
            for (int mkv = 0; mkv < SPAN / MMA_N; ++mkv) {
                if (mkv + 1 < SPAN / MMA_N) {
                    ldmatrix_x4(kFrag[(mkv + 1) & 1],
                                kmd[md] + kBufOff
                                + (tokOff + (mkv + 1) * MMA_N) * STRIDE_B);
                }
                const uint32_t (&kf)[4] = kFrag[mkv & 1];
                #pragma unroll
                for (int mq = 0; mq < WARP_Q / MMA_M; ++mq) {
                    mma_m16n8k16(qFrag[mq][0], kf + 0, sp[mq].s[mkv]);
                    mma_m16n8k16(qFrag[mq][1], kf + 2, sp[mq].s[mkv]);
                }
            }
        }
    }

    /* P·V with the same fragment double-buffer discipline (dual schedule
       runs at 1 CTA/SM: V fragment ldmatrix must lead its mmas). */
    template <int SPAN = BLOCK_KV>
    FK_DEVICE_FUSE void pvMmaStreamDual(const SPTileT<SPAN> (&sp)[WARP_Q / MMA_M],
                                        const uint32_t base, const uint32_t vThread,
                                        float (&oAcc)[WARP_Q / MMA_M][HEAD_DIM / MMA_N][4],
                                        const int tokOff = 0) {
        using namespace attention_mma_detail;
        constexpr int MD_IT = HEAD_DIM / MMA_N / 2;
        #pragma unroll
        for (int mkv = 0; mkv < SPAN / MMA_K; ++mkv) {
            const uint32_t rowAddr = base + vThread + (tokOff + mkv * MMA_K) * STRIDE_B;
            uint32_t frag[2][4];
            ldmatrix_x4_trans(frag[0], rowAddr ^ 0u);
            #pragma unroll
            for (int mdp = 0; mdp < MD_IT; ++mdp) {
                if (mdp + 1 < MD_IT) {
                    ldmatrix_x4_trans(frag[(mdp + 1) & 1],
                                      rowAddr ^ ((mdp + 1) * 2 * MMA_N
                                                 * (uint32_t)sizeof(__nv_bfloat16)));
                }
                const uint32_t (&vf)[4] = frag[mdp & 1];
                #pragma unroll
                for (int mq = 0; mq < WARP_Q / MMA_M; ++mq) {
                    mma_m16n8k16(sp[mq].p[mkv], vf + 0, oAcc[mq][mdp * 2]);
                    mma_m16n8k16(sp[mq].p[mkv], vf + 2, oAcc[mq][mdp * 2 + 1]);
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
    template <int SPAN = BLOCK_KV>
    FK_DEVICE_FUSE void pvMmaStream(const SPTileT<SPAN> (&sp)[WARP_Q / MMA_M],
                                    const uint32_t base, const uint32_t vThread,
                                    float (&oAcc)[WARP_Q / MMA_M][HEAD_DIM / MMA_N][4],
                                    const int tokOff = 0) {
        using namespace attention_mma_detail;
        #pragma unroll
        for (int mkv = 0; mkv < SPAN / MMA_K; ++mkv) {
            #pragma unroll
            for (int md = 0; md < HEAD_DIM / MMA_N; md += 2) {
                uint32_t frag[4];
                uint32_t addr = base + vThread;
                addr += (tokOff + mkv * MMA_K) * STRIDE_B;
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

    /* Hoisted-address P·V (same trick as sMmaStreamH): vmd[md] =
       (smemBase + vBufOff + vThread) ^ (md*MMA_N*2), precomputed once. */
    template <int SPAN = BLOCK_KV>
    FK_DEVICE_FUSE void pvMmaStreamH(const SPTileT<SPAN> (&sp)[WARP_Q / MMA_M],
                                     const uint32_t (&vmd)[HEAD_DIM / MMA_N],
                                     float (&oAcc)[WARP_Q / MMA_M][HEAD_DIM / MMA_N][4],
                                     const int tokOff = 0) {
        using namespace attention_mma_detail;
        #pragma unroll
        for (int mkv = 0; mkv < SPAN / MMA_K; ++mkv) {
            #pragma unroll
            for (int md = 0; md < HEAD_DIM / MMA_N; md += 2) {
                uint32_t frag[4];
                ldmatrix_x4_trans(frag, vmd[md]
                                        + (tokOff + mkv * MMA_K) * STRIDE_B);
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
                    OutIOp::Operation::exec(Point{ col, rowA, bh }, r[0] * invA, p.out);
                    OutIOp::Operation::exec(Point{ col + 1, rowA, bh }, r[1] * invA, p.out);
                }
                if (rowB < p.seq_q) {
                    OutIOp::Operation::exec(Point{ col, rowB, bh }, r[2] * invB, p.out);
                    OutIOp::Operation::exec(Point{ col + 1, rowB, bh }, r[3] * invB, p.out);
                }
            }
        }
    }

    /* split-KV partial writeout: UNNORMALIZED oAcc plus per-row (m, l).
       Layout [bh, seq_q, splits, HEAD_DIM+2] — fp32. Lane (laneId%4) owns
       cols {2t, 2t+1} of each MMA_N block, rows rowA/rowB as in writeOut. */
    FK_DEVICE_FUSE void writePartial(const float (&oAcc)[WARP_Q / MMA_M][HEAD_DIM / MMA_N][4],
                                     const float (&rowMax)[WARP_Q / MMA_M][2],
                                     const float (&rowSum)[WARP_Q / MMA_M][2],
                                     const Params& p, const int qBlockBase,
                                     const int warpId, const int laneId, const int bh) {
        const int split = blockIdx.z;
        const int splitPlane = bh * p.splits + split;
        #pragma unroll
        for (int mq = 0; mq < WARP_Q / MMA_M; ++mq) {
            const int rowA = qBlockBase + warpId * WARP_Q + mq * MMA_M + laneId / 4;
            const int rowB = rowA + 8;
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int row = half == 0 ? rowA : rowB;
                if (row >= p.seq_q) continue;
                #pragma unroll
                for (int md = 0; md < HEAD_DIM / MMA_N; ++md) {
                    const int col = md * MMA_N + (laneId % 4) * 2;
                    Params::PartialWriteIOp::Operation::exec(
                        Point{ col, row, splitPlane }, oAcc[mq][md][half * 2], p.partial);
                    Params::PartialWriteIOp::Operation::exec(
                        Point{ col + 1, row, splitPlane }, oAcc[mq][md][half * 2 + 1], p.partial);
                }
                if (laneId % 4 == 0) {
                    Params::PartialWriteIOp::Operation::exec(
                        Point{ HEAD_DIM, row, splitPlane }, rowMax[mq][half], p.partial);
                    Params::PartialWriteIOp::Operation::exec(
                        Point{ HEAD_DIM + 1, row, splitPlane }, rowSum[mq][half], p.partial);
                }
            }
        }
    }

    template <bool SPLIT_KV = false>
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
        // CAUSAL LOAD-BALANCE: q-block i does O(i) KV tiles of work, so the
        // natural launch order puts the heaviest blocks LAST and leaves a
        // long single-block tail (ncu: achieved occupancy 10.9% vs 33%
        // theoretical on s2048 causal). GPUs schedule blocks roughly in
        // launch order -> reversing the raster for causal starts the heavy
        // blocks first and the light ones fill the tail. Dense work is
        // uniform; keep natural order (better L2 locality on K/V).
        // (A small-grid guard was tried and REVERTED: disabling the reversal
        // below 1024 blocks made bh8 s2048 WORSE, 0.82->0.76 vs FA — even at
        // ~1.4 waves the tail dominates. Reversal is unconditional.)
        const int qBlockIdx = p.causal ? (int)(gridDim.x - 1 - blockIdx.x)
                                       : (int)blockIdx.x;
        const int qBlockBase = qBlockIdx * BLOCK_Q;

        const uint32_t qThread = swz(((warpId * WARP_Q + laneId % 16) * HEAD_DIM
                                      + (laneId / 16) * 8) * sizeof(__nv_bfloat16));
        const uint32_t kThread = swz(((laneId % 8) * HEAD_DIM
                                      + (laneId / 8) * 8) * sizeof(__nv_bfloat16));
        const uint32_t vThread = swz(((laneId % 16) * HEAD_DIM
                                      + (laneId / 16) * 8) * sizeof(__nv_bfloat16));

        // ---- Q -> smem -> registers (one-time, always through the Read IOp) -
        float qStage[Q_GROUPS][8];
        prefetchTile<Q_GROUPS>(p.q, qBlockBase, p.seq_q, bh, qStage);
        storeTile<Q_GROUPS>(smemBytes, 0, qStage);
        __syncthreads();

        uint32_t qReg[(DEEP_Q_SMEM || DUAL_Q_SMEM) ? 1 : WARP_Q / MMA_M]
                     [(DEEP_Q_SMEM || DUAL_Q_SMEM) ? 1 : HEAD_DIM / MMA_K][4];
        if constexpr (!DEEP_Q_SMEM && !DUAL_Q_SMEM) {
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
        // DEEP_Q_SMEM: Q stays resident at smem[0..Q_RES_B); fragments are
        // streamed inside the QK^T mma loop. No qReg, no extracting sync.
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
        // SPLIT-KV: blockIdx.z takes an equal chunk of THIS q-block's
        // iteration range (adapts to causal: each q-block splits its own
        // numIter). Compile-time gated: the runtime-branch version kept
        // rowMax live through the epilogue in EVERY kernel and cost ~20%
        // on splits==1 shapes (d64 bh32 s2048 dense 0.88->0.70 measured).
        if constexpr (SPLIT_KV) {
            const int span = itEnd - itBegin;
            const int chunk = (span + p.splits - 1) / p.splits;
            const int s0 = itBegin + (int)blockIdx.z * chunk;
            itEnd = ::min(itEnd, s0 + chunk);
            itBegin = ::min(s0, itEnd);
        }

        // ============== register-prefetch schedule (all DRAM via IOps) ======
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

        if constexpr (SPLIT_KV) {
            writePartial(oAcc, rowMax, rowSum, p, qBlockBase, warpId, laneId, bh);
        } else {
            writeOut(oAcc, rowSum, p, qBlockBase, warpId, laneId, bh);
        }
    }
};

/* split-KV combine: one block per (bh, row-chunk); exact online-softmax
   merge of `splits` unnormalized partials. expBase: 2^x when the producer
   kernel ran log2-domain softmax, e^x otherwise. The fused epilogue runs
   here (the producer wrote UNNORMALIZED partials); a write-type epilogue
   performs the global writes itself. */
template <int HEAD_DIM, bool LOG2D, typename PartialReadIOp, typename OutIOp>
__global__ void flashFwdSplitCombineKernel(const PartialReadIOp partial,
                                           const OutIOp out,
                                           const int seqQ, const int splits) {
    const int bh = blockIdx.y;
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= seqQ) return;
    // global max over splits (uniform across the row's threads)
    float gm = -FLT_MAX;
    for (int s = 0; s < splits; ++s)
        gm = fmaxf(gm, PartialReadIOp::Operation::exec(
            Point{ HEAD_DIM, row, bh * splits + s }, partial));
    // each thread owns HEAD_DIM/32 columns
    float acc[HEAD_DIM / 32] = {};
    float l = 0.f;
    for (int s = 0; s < splits; ++s) {
        const int splitPlane = bh * splits + s;
        const float m = PartialReadIOp::Operation::exec(
            Point{ HEAD_DIM, row, splitPlane }, partial);
        if (m == -FLT_MAX) continue;            // empty chunk
        const float w = LOG2D ? exp2f(m - gm) : __expf(m - gm);
        l += PartialReadIOp::Operation::exec(
            Point{ HEAD_DIM + 1, row, splitPlane }, partial) * w;
        #pragma unroll
        for (int c = 0; c < HEAD_DIM / 32; ++c)
            acc[c] += PartialReadIOp::Operation::exec(
                Point{ (int)threadIdx.x + c * 32, row, splitPlane }, partial) * w;
    }
    const float inv = l > 0.f ? 1.f / l : 0.f;
    #pragma unroll
    for (int c = 0; c < HEAD_DIM / 32; ++c) {
        OutIOp::Operation::exec(Point{ (int)threadIdx.x + c * 32, row, bh },
                                acc[c] * inv, out);
    }
}

template <int HEAD_DIM, typename QIOp, typename KIOp, typename VIOp,
          typename OutIOp, int BLOCK_Q, int BLOCK_KV, int NUM_WARPS,
          typename ScoreModOp, bool SPLIT_KV = false>
__global__ void launchFlashAttentionMmaDPP_Kernel(
        const __grid_constant__ typename FlashAttentionMmaDPP<
            HEAD_DIM, QIOp, KIOp, VIOp, OutIOp,
            BLOCK_Q, BLOCK_KV, NUM_WARPS, ScoreModOp>::Params params) {
    FlashAttentionMmaDPP<HEAD_DIM, QIOp, KIOp, VIOp, OutIOp,
                         BLOCK_Q, BLOCK_KV, NUM_WARPS,
                         ScoreModOp>::template exec<SPLIT_KV>(params);
}

/* IOp-first API. All global-memory reads/writes go through the provided
 * Read/Write IOps. The launch still auto-tunes tile sizes from the
 * prologue types (pass BLOCK_Q/BLOCK_KV > 0 to override). */
template <int HEAD_DIM, int BLOCK_Q = 0, int BLOCK_KV = 0, int NUM_WARPS = 4,
          typename QIOp, typename KIOp, typename VIOp, typename WriteIOp,
          typename ScoreModOp = NoScoreMod,
          std::enable_if_t<isAnyWriteType<WriteIOp>, int> = 0>
inline void executeFlashAttentionMma(
        const QIOp& q, const KIOp& k, const VIOp& v, const WriteIOp& out,
        const int batchHeads, const int seqQ, const int seqK,
        const bool causal, Stream_<ParArch::GPU_NVIDIA>& stream,
        const float scaleOverride = -1.f,
        const ScoreModOp& scoreMod = {}, const BlockSparsity& sparse = {}) {
    constexpr bool RAW = isRawBf16Read<KIOp> && isRawBf16Read<VIOp>;
    // RAW auto-tile is REGIME-DEPENDENT. Swept exhaustively with REAL data
    // (benchmarks/sweep_tiles.cu, bh in {8,32,64} x s in {2048,4096,8192}
    // x {causal,dense}, RTX PRO 6000):
    //  d64 dense:  BQ128 (WARP_Q=32 -> mq=2, two independent mma chains;
    //              kernel is ILP-limited) once the grid stays saturated:
    //              bh*seqQ >= 64K. Below that BQ64's extra blocks win.
    //  d64 causal: effective work is halved -> BQ128 only at seqQ>=8192,
    //              or seqQ>=4096 when bh>=64 (reversed raster keeps balance).
    //  d128 long:  BQ128/BKV64/NW8 at seqQ>=8192 && bh>=32 (+3..8%); bh8
    //              stays BQ64 (BQ128 loses 9% there).
    // Explicit BLOCK_Q/BLOCK_KV template args bypass the heuristic.
    if constexpr (BLOCK_Q == 0 && BLOCK_KV == 0 && RAW) {
      // block-sparse masks fix their own tile granularity — only reroute
      // when the mask granularity divides the candidate tile (sliding-
      // window masks at s8192 DO want the BQ128 pick: 1.11x vs 1.01x).
      const auto maskFits = [&](const int bq, const int bkv) {
          return sparse.mask == nullptr ||
                 (sparse.maskBQ % bq == 0 && sparse.maskBKV % bkv == 0);
      };
        if constexpr (HEAD_DIM <= 64) {
            const bool wantBQ128 =
                causal ? (seqQ >= 8192 || (seqQ >= 4096 && batchHeads >= 64))
                       : ((long)batchHeads * seqQ >= 65536);
            if (wantBQ128 && maskFits(128, 64)) {
                executeFlashAttentionMma<HEAD_DIM, 128, 64, NUM_WARPS>(
                    q, k, v, out, batchHeads, seqQ, seqK, causal, stream,
                    scaleOverride, scoreMod, sparse);
                return;
            }
            // ROUND-8 RESWEEP (spikes/resweep_r8.cu + guard_r8.cu, post
            // density rounds): mid-range CAUSAL prefers a WIDER KV tile —
            // BQ64/BKV128 amortizes the per-tile softmax bookkeeping over
            // 2x the columns (bh8 s4096 +33%, bh8 s1024 +29%, bh32 s4096
            // +7%; only bh32 s512 regresses -3.5%). Dense keeps BQ64/BKV64
            // (BKV128 measured slower on every dense shape).
            if (causal && seqQ >= 1024 && maskFits(64, 128)) {
                executeFlashAttentionMma<HEAD_DIM, 64, 128, NUM_WARPS>(
                    q, k, v, out, batchHeads, seqQ, seqK, causal, stream,
                    scaleOverride, scoreMod, sparse);
                return;
            }
        } else if constexpr (HEAD_DIM == 128) {
            // ROUND-8 RESWEEP: the old NW8 rule (bh*seqQ>=256K -> BQ128/
            // BKV64/NW8) went STALE after the density rounds — BQ64/BKV64
            // now beats it on every swept shape below s16384 (bh64 s4096:
            // 341 vs 321 TF; bh32 s8192 causal: 326 vs 300). The deep-wide
            // tile BQ128/BKV128/NW8 (96KB smem, 1 CTA/SM) wins only at
            // s>=16384 where per-CTA work is huge (bh32 dense 337 vs 311;
            // bh32 causal 322 vs 300); s8192 is parity (+-1%). Rule:
            // seqQ >= 16384 -> BQ128/BKV128/NW8; else plain BQ64/BKV64.
            if (seqQ >= 16384 && maskFits(128, 128)) {
                executeFlashAttentionMma<HEAD_DIM, 128, 128, 8>(
                    q, k, v, out, batchHeads, seqQ, seqK, causal, stream,
                    scaleOverride, scoreMod, sparse);
                return;
            }
        }
    }
    constexpr int BQ = BLOCK_Q > 0 ? BLOCK_Q
                                   : (RAW ? 64 : (HEAD_DIM <= 64 ? 128 : 64));
    constexpr int BKV = BLOCK_KV > 0 ? BLOCK_KV : (RAW ? 64 : 32);
    using DPP = FlashAttentionMmaDPP<HEAD_DIM, QIOp, KIOp, VIOp,
                                     WriteIOp, BQ, BKV, NUM_WARPS, ScoreModOp>;
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
    typename DPP::Params params{ q, k, v, out, seqQ, seqK, scale, causal,
                                 scoreMod, sparse };
    const int numQ = (seqQ + BQ - 1) / BQ;
    const dim3 block(DPP::THREADS, 1, 1);
    const int smemBytes = DPP::SMEM_BYTES;
    auto* kernel = launchFlashAttentionMmaDPP_Kernel<HEAD_DIM, QIOp, KIOp, VIOp,
                                                     WriteIOp, BQ, BKV, NUM_WARPS,
                                                     ScoreModOp>;
    if (smemBytes > 48000) {
        gpuErrchk(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       smemBytes));
    }
    // SPLIT-KV FORWARD for under-saturated grids (ncu, d128 bh8 s2048: FA
    // launches 5.45 waves via splits while we ran 0.68 — SMs idle). When
    // numQ*bh covers < ~70% of one resident wave, split the KV range across
    // blockIdx.z, write unnormalized partials, and reduce with an exact
    // online-softmax combine kernel. Disabled for block-sparse (the sparse
    // iteration trim already reshapes the range per q-block).
    int splits = 1;
    float* partialData = nullptr;
    if (sparse.mask == nullptr) {
        static const int wave = [&] {
            int bpm = 0, dev = 0, sms = 0;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpm, kernel,
                                                          DPP::THREADS, smemBytes);
            cudaGetDevice(&dev);
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
            return (bpm > 0 ? bpm : 1) * sms;
        }();
        const long gridBlocks = (long)numQ * batchHeads;
        // average KV iterations per q-block (causal averages ~half the row)
        const int avgIter = ::max(1, (int)(((long)(causal ? seqK / 2 : seqK)) / BKV));
        // THRESHOLD (measured per regime): causal benefits up to ~0.7 wave
        // (the triangular tail leaves SMs idle even at 0.68 wave: d128 bh8
        // s2048 causal 0.76->0.94 with splits); dense only below ~0.45 wave
        // (at 0.68 wave the fp32 partial traffic costs more than the idle
        // SMs: d64 bh32 s2048 dense 0.88->0.71 when split at 0.68 wave).
        const long num = gridBlocks * 10;
        const bool under = causal ? (num < (long)wave * 7)
                                  : (num < (long)wave * 9 / 2);
        if (under && avgIter >= 8) {
            const int bySat  = (int)((wave + gridBlocks - 1) / gridBlocks);
            const int byWork = avgIter / 4;   // keep >=4 KV tiles per chunk
            splits = ::min(::min(bySat, byWork), 16);
            splits = ::max(splits, 1);
        }
        if (splits > 1) {
            const size_t bytes = (size_t)batchHeads * seqQ * splits
                                 * (HEAD_DIM + 2) * sizeof(float);
            gpuErrchk(cudaMallocAsync(&partialData, bytes, stream.getCUDAStream()));
            params.partial = makeAttentionPartialWrite(
                partialData, batchHeads, seqQ, splits, HEAD_DIM + 2);
            params.splits = splits;
        }
    }
    const dim3 grid(numQ, batchHeads, splits);
    if (splits > 1) {
        auto* splitKernel = launchFlashAttentionMmaDPP_Kernel<
            HEAD_DIM, QIOp, KIOp, VIOp, WriteIOp, BQ, BKV, NUM_WARPS,
            ScoreModOp, /*SPLIT_KV=*/true>;
        if (smemBytes > 48000) {
            gpuErrchk(cudaFuncSetAttribute(splitKernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes));
        }
        splitKernel<<<grid, block, smemBytes, stream.getCUDAStream()>>>(params);
        gpuErrchk(cudaGetLastError());
        // combine: 32 threads per row (HEAD_DIM/32 cols each), 4 rows/block
        const dim3 cblock(32, 4, 1);
        const dim3 cgrid((seqQ + 3) / 4, batchHeads, 1);
        const auto partialRead = makeAttentionPartialRead(
            partialData, batchHeads, seqQ, splits, HEAD_DIM + 2);
        flashFwdSplitCombineKernel<HEAD_DIM, DPP::LOG2_DOMAIN, decltype(partialRead), WriteIOp>
            <<<cgrid, cblock, 0, stream.getCUDAStream()>>>(partialRead, out, seqQ, splits);
        gpuErrchk(cudaGetLastError());
        gpuErrchk(cudaFreeAsync(partialData, stream.getCUDAStream()));
    } else {
        kernel<<<grid, block, smemBytes, stream.getCUDAStream()>>>(params);
        gpuErrchk(cudaGetLastError());
    }
}

} // namespace fk

#endif // defined(__NVCC__)

#endif // FK_ATTENTION_FLASH_ATTENTION_MMA_H
