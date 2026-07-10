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
__device__ __forceinline__ void ldmatrix_x4(uint regs[4], uint addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint regs[4], uint addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
                 : "r"(addr));
}

__device__ __forceinline__ void mma_m16n8k16(const uint A[4], const uint B[2],
                                             float D[4]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                 : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
                 : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                   "r"(B[0]), "r"(B[1]),
                   "f"(D[0]), "f"(D[1]), "f"(D[2]), "f"(D[3]));
}

// 16-byte async copy; srcSize=0 zero-fills (tail/ragged guard).
__device__ __forceinline__ void cp_async_16(uint dst, const void* src,
                                            const int srcSize) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 :: "r"(dst), "l"(src), "r"(srcSize));
}
// 8-byte async copy (ca path; cg only supports 16B).
__device__ __forceinline__ void cp_async_8(uint dst, const void* src,
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

// pack two fp32 into one bf16x2 register (uint) — used by the in-place
// S/P union pack in softmaxTile.
__device__ __forceinline__ uint packBf162(const float lo, const float hi) {
    const __nv_bfloat162 h = __float22bfloat162_rn({ lo, hi });
    uint r;
    memcpy(&r, &h, sizeof(r));
    return r;
}

// fp8 e4m3 mma: kind::f8f6f4 m16n8k32, A 4 regs (16 e4m3), B 2 regs (8 e4m3).
// Always declared so `if constexpr` branches type-check; the instruction is
// only emitted when FK_ENABLE_FP8_QK (needs sm_120a/121a feature set).
__device__ __forceinline__ void mma_fp8_m16n8k32(const uint A[4],
                                                 const uint B[2],
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
    static constexpr int Q_PHASE_B =
        BLOCK_Q * STRIDE_B + (FP8_QK ? BLOCK_Q * HEAD_DIM + BLOCK_Q * 4 : 0);
    static constexpr int SMEM_BYTES =
        (Q_PHASE_B > Q_RES_B + KV_BUFS * KV_BUF_B + Q_STAGE_B
             ? Q_PHASE_B
             : Q_RES_B + KV_BUFS * KV_BUF_B + Q_STAGE_B);

    struct Params {
        QIOp q; KIOp k; VIOp v;
        OT* o;
        int seq_q, seq_k;
        float scale;
        bool causal;
        EpilogueIOp epilogue;
        ScoreModOp scoreMod;     // flex-attention score_mod / mask_mod
        BlockSparsity sparse;    // block-sparse tile skipping (nullptr = dense)
        // SPLIT-KV FORWARD (FA-style): when splits > 1, blockIdx.z selects a
        // KV chunk and the kernel writes UNNORMALIZED partials (oAcc, m, l)
        // to `partial` instead of O; a combine kernel reduces them. Sized
        // [bh, seq_q, splits, HEAD_DIM+2]. Saturates the GPU on small grids
        // (bh*numQ << 1 wave) — ncu: FA runs 5.45 waves vs our 0.68 on
        // d128 bh8 s2048, hiding everything; this is that lever.
        float* partial = nullptr;
        int splits = 1;
    };
    // log2-domain softmax is active for these functor types (see softmaxTile);
    // the split combine must interpret stored row-maxes in the same base.
    static constexpr bool LOG2_DOMAIN = !HAS_SCORE_MOD || IS_ALIBI;

    FK_DEVICE_FUSE uint swz(const uint byteOff) {
        const uint row = (byteOff / STRIDE_B) % 8;
        constexpr uint div = (64 / STRIDE_B > 1 ? 64 / STRIDE_B : 1);
        return byteOff ^ ((row / div) << 4);
    }

    template <typename IOp>
    FK_DEVICE_FUSE float readElem(const IOp& iop, const int x, const int y, const int z) {
        return low_precission::attnToF32(IOp::Operation::exec(Point{ x, y, z }, iop));
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
                    const ulonglong packed = *reinterpret_cast<const ulonglong*>(base);
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
    FK_DEVICE_FUSE void storeTile(char* smemBytes, const uint bufOff,
                                  const float (&regs)[GROUPS][8]) {
        using attention_mma_detail::Bf16x8;
        #pragma unroll
        for (int g = 0; g < GROUPS; ++g) {
            const uint idx = (g * THREADS + threadIdx.x) * 8;
            Bf16x8 packed;
            #pragma unroll
            for (int h = 0; h < 4; ++h)
                packed.h[h] = __float22bfloat162_rn({ regs[g][2 * h],
                                                      regs[g][2 * h + 1] });
            *reinterpret_cast<Bf16x8*>(
                smemBytes + bufOff + swz(idx * (uint)sizeof(__nv_bfloat16))) = packed;
        }
    }

    // ---- raw staging: real cp.async, zero-fill on ragged tails -------------
    template <int GROUPS>
    FK_DEVICE_FUSE void cpasyncTile(const uint dstBase /* smem addr + buf */,
                                    const __nv_bfloat16* srcPlane, const int rowBase,
                                    const int seqLen) {
        using namespace attention_mma_detail;
        constexpr int ROWS = GROUPS * THREADS * 8 / HEAD_DIM;
#ifndef FK_CPASYNC_FAST
#define FK_CPASYNC_FAST 1
#endif
        if (FK_CPASYNC_FAST && rowBase + ROWS <= seqLen) {
            // FULL-TILE FAST PATH (the common case on interior tiles): no
            // per-group bounds predicate, no safeRow select — and the src
            // address becomes base + compile-time stride (idx = g*THREADS*8
            // + tid*8), which ptxas folds into the LDGSTS immediate. The
            // ragged path below costs ISETP+SEL+IMAD per group per tile in
            // the hot loop (SASS: ISETP was 1.3/HMMA vs FA's 0.19).
            const __nv_bfloat16* base = srcPlane + (long)rowBase * HEAD_DIM;
            #pragma unroll
            for (int g = 0; g < GROUPS; ++g) {
                const int idx = (g * THREADS + (int)threadIdx.x) * 8;
                cp_async_16(dstBase + swz(idx * (uint)sizeof(__nv_bfloat16)),
                            base + idx, 16);
            }
        } else {
            #pragma unroll
            for (int g = 0; g < GROUPS; ++g) {
                const int idx = (g * THREADS + (int)threadIdx.x) * 8;
                const int r = idx / HEAD_DIM;
                const int c = idx % HEAD_DIM;
                const int row = rowBase + r;
                const bool ok = row < seqLen;
                const int safeRow = ok ? row : (seqLen > 0 ? seqLen - 1 : 0);
                cp_async_16(dstBase + swz(idx * (uint)sizeof(__nv_bfloat16)),
                            srcPlane + (long)safeRow * HEAD_DIM + c, ok ? 16 : 0);
            }
        }
    }

    // ---- quantized staging: cp.async the RAW BYTES (half the traffic) ------
    // SWZ8: XOR-swizzle each 8B chunk by ((row&7)*8) — used by the FP8 QK^T
    // path so direct 4B fragment reads don't bank-conflict (d128 rows are
    // exactly 32 banks wide -> every token would hit bank 0 unswizzled).
    template <int GROUPS, bool SWZ8 = false>
    FK_DEVICE_FUSE void cpasyncQuantTile(const uint dstBase,
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
            uint off = (uint)idx;
            if constexpr (SWZ8) off ^= (uint)((r & 7) * 8);
            cp_async_8(dstBase + off,
                       srcPlane + (long)safeRow * HEAD_DIM + c, ok ? 8 : 0);
        }
    }

    /* smem bytes -> swizzled bf16 smem, dequantizing in-register. scales
       read from global (coalesced; BLOCK_KV floats per tile). */
    template <int GROUPS, bool FP8>
    FK_DEVICE_FUSE void dequantTile(char* smemBytes, const uint stageOff,
                                    const uint bf16Off, const float* scales,
                                    const int rowBase, const int seqLen) {
        using attention_mma_detail::Bf16x8;
        #pragma unroll
        for (int g = 0; g < GROUPS; ++g) {
            const int idx = (g * THREADS + (int)threadIdx.x) * 8;
            const int r = idx / HEAD_DIM;
            const int row = rowBase + r;
            const float sc = (row < seqLen) ? scales[row] : 0.f;
            const ulonglong packed =
                *reinterpret_cast<const ulonglong*>(smemBytes + stageOff + idx);
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
                smemBytes + bf16Off + swz(idx * (uint)sizeof(__nv_bfloat16))) = out;
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
        uint p[SPAN / MMA_K][4];   // packed bf16x2 P (first half of s)
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
    static constexpr uint Q8_OFF = BLOCK_Q * STRIDE_B;
    static constexpr uint QSC_OFF = Q8_OFF + BLOCK_Q * HEAD_DIM;

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
                    smemBytes + swz((uint)(row * HEAD_DIM + c) * 2u));
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
                smemBytes[Q8_OFF + (uint)(row * HEAD_DIM + laneId * EPL + e)] =
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
                                     uint (&qReg8)[WARP_Q / MMA_M][HEAD_DIM / 32][4],
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
                const uint a = Q8_OFF + (uint)(rowA * HEAD_DIM + md * 32 + 4 * tig);
                const uint b = Q8_OFF + (uint)(rowB * HEAD_DIM + md * 32 + 4 * tig);
                qReg8[mq][md][0] = *reinterpret_cast<const uint*>(smemBytes + a);
                qReg8[mq][md][1] = *reinterpret_cast<const uint*>(smemBytes + b);
                qReg8[mq][md][2] = *reinterpret_cast<const uint*>(smemBytes + a + 16);
                qReg8[mq][md][3] = *reinterpret_cast<const uint*>(smemBytes + b + 16);
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
    FK_DEVICE_FUSE void sMmaFp8(const uint (&qReg8)[WARP_Q / MMA_M][HEAD_DIM / 32][4],
                                const float (&qs)[WARP_Q / MMA_M][2],
                                const char* smemBytes, const uint kStageOff,
                                const float* kScalesPlane, const int offKV,
                                const int seqK,
                                SPTile (&sp)[WARP_Q / MMA_M], const int laneId) {
        using namespace attention_mma_detail;
        const int g = laneId >> 2, tig = laneId & 3;
        #pragma unroll
        for (int mkv = 0; mkv < BLOCK_KV / MMA_N; ++mkv) {
            const int tok = mkv * MMA_N + g;
            const uint rowOff = (uint)(tok * HEAD_DIM);
            const uint sw = (uint)((tok & 7) * 8);
            #pragma unroll
            for (int md = 0; md < HEAD_DIM / 32; ++md) {
                uint b[2];
                const uint base = rowOff + (uint)(md * 32 + 4 * tig);
                b[0] = *reinterpret_cast<const uint*>(
                    smemBytes + kStageOff + (base ^ sw));
                b[1] = *reinterpret_cast<const uint*>(
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
    FK_DEVICE_FUSE void loadKFrags(const uint base, const uint kThread,
                                   uint (&kReg)[BLOCK_KV / MMA_N][HEAD_DIM / MMA_K][2]) {
        using namespace attention_mma_detail;
        #pragma unroll
        for (int mkv = 0; mkv < BLOCK_KV / MMA_N; ++mkv) {
            #pragma unroll
            for (int md = 0; md < HEAD_DIM / MMA_K; md += 2) {
                uint addr = base + kThread;
                addr += mkv * MMA_N * STRIDE_B;
                addr ^= md * MMA_K * sizeof(__nv_bfloat16);
                ldmatrix_x4(&kReg[mkv][md][0], addr);
            }
        }
    }

    FK_DEVICE_FUSE void loadVFrags(const uint base, const uint vThread,
                                   uint (&vReg)[BLOCK_KV / MMA_K][HEAD_DIM / MMA_N][2]) {
        using namespace attention_mma_detail;
        #pragma unroll
        for (int mkv = 0; mkv < BLOCK_KV / MMA_K; ++mkv) {
            #pragma unroll
            for (int md = 0; md < HEAD_DIM / MMA_N; md += 2) {
                uint addr = base + vThread;
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

    FK_DEVICE_FUSE void sMma(const uint (&qReg)[WARP_Q / MMA_M][HEAD_DIM / MMA_K][4],
                             const uint (&kReg)[BLOCK_KV / MMA_N][HEAD_DIM / MMA_K][2],
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
    FK_DEVICE_FUSE void sMmaStream(const uint (&qReg)[WARP_Q / MMA_M][HEAD_DIM / MMA_K][4],
                                   const uint base, const uint kThread,
                                   SPTileT<SPAN> (&sp)[WARP_Q / MMA_M],
                                   const int tokOff = 0) {
        using namespace attention_mma_detail;
        #pragma unroll
        for (int mkv = 0; mkv < SPAN / MMA_N; ++mkv) {
            #pragma unroll
            for (int md = 0; md < HEAD_DIM / MMA_K; md += 2) {
                // identical x4 pattern to loadKFrags: one mkv row, fragments
                // for md (regs 0-1) and md+1 (regs 2-3).
                uint frag[4];
                uint addr = base + kThread;
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
    FK_DEVICE_FUSE void sMmaStreamH(const uint (&qReg)[WARP_Q / MMA_M][HEAD_DIM / MMA_K][4],
                                    const uint bufOff,
                                    const uint (&kmd)[HEAD_DIM / MMA_K],
                                    SPTileT<SPAN> (&sp)[WARP_Q / MMA_M],
                                    const int tokOff = 0) {
        using namespace attention_mma_detail;
        #pragma unroll
        for (int mkv = 0; mkv < SPAN / MMA_N; ++mkv) {
            #pragma unroll
            for (int md = 0; md < HEAD_DIM / MMA_K; md += 2) {
                uint frag[4];
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
    FK_DEVICE_FUSE void sMmaStreamDeepQ(const uint qBase, const uint qThread,
                                        const uint kBase, const uint kThread,
                                        SPTile (&sp)[WARP_Q / MMA_M]) {
        using namespace attention_mma_detail;
        #pragma unroll
        for (int mkv = 0; mkv < BLOCK_KV / MMA_N; ++mkv) {
            #pragma unroll
            for (int md = 0; md < HEAD_DIM / MMA_K; md += 2) {
                uint kFrag[4];
                uint kAddr = kBase + kThread;
                kAddr += mkv * MMA_N * STRIDE_B;
                kAddr ^= md * MMA_K * sizeof(__nv_bfloat16);
                ldmatrix_x4(kFrag, kAddr);
                #pragma unroll
                for (int mq = 0; mq < WARP_Q / MMA_M; ++mq) {
                    uint qFrag[4], qFrag2[4];
                    uint qAddr = qBase + qThread;
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
    FK_DEVICE_FUSE void sMmaStreamDualQ(const uint kBufOff,
                                        const uint (&qmd)[HEAD_DIM / MMA_K],
                                        const uint (&kmd)[HEAD_DIM / MMA_K],
                                        SPTileT<SPAN> (&sp)[WARP_Q / MMA_M],
                                        const int tokOff = 0) {
        using namespace attention_mma_detail;
        // K-fragment register double-buffer: at 1 CTA/SM (this schedule's
        // occupancy point, same as FA's) there is no second warp to hide
        // LDSM->HMMA latency, so the NEXT fragment's ldmatrix must issue
        // BEFORE the current mmas (CUTLASS mainloop discipline).
        #pragma unroll
        for (int md = 0; md < HEAD_DIM / MMA_K; md += 2) {
            uint qFrag[WARP_Q / MMA_M][2][4];
            #pragma unroll
            for (int mq = 0; mq < WARP_Q / MMA_M; ++mq) {
                const uint rowOff = mq * MMA_M * STRIDE_B;
                ldmatrix_x4(qFrag[mq][0], qmd[md]     + rowOff);
                ldmatrix_x4(qFrag[mq][1], qmd[md + 1] + rowOff);
            }
            uint kFrag[2][4];
            ldmatrix_x4(kFrag[0], kmd[md] + kBufOff + tokOff * STRIDE_B);
            #pragma unroll
            for (int mkv = 0; mkv < SPAN / MMA_N; ++mkv) {
                if (mkv + 1 < SPAN / MMA_N) {
                    ldmatrix_x4(kFrag[(mkv + 1) & 1],
                                kmd[md] + kBufOff
                                + (tokOff + (mkv + 1) * MMA_N) * STRIDE_B);
                }
                const uint (&kf)[4] = kFrag[mkv & 1];
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
                                        const uint base, const uint vThread,
                                        float (&oAcc)[WARP_Q / MMA_M][HEAD_DIM / MMA_N][4],
                                        const int tokOff = 0) {
        using namespace attention_mma_detail;
        constexpr int MD_IT = HEAD_DIM / MMA_N / 2;
        #pragma unroll
        for (int mkv = 0; mkv < SPAN / MMA_K; ++mkv) {
            const uint rowAddr = base + vThread + (tokOff + mkv * MMA_K) * STRIDE_B;
            uint frag[2][4];
            ldmatrix_x4_trans(frag[0], rowAddr ^ 0u);
            #pragma unroll
            for (int mdp = 0; mdp < MD_IT; ++mdp) {
                if (mdp + 1 < MD_IT) {
                    ldmatrix_x4_trans(frag[(mdp + 1) & 1],
                                      rowAddr ^ ((mdp + 1) * 2 * MMA_N
                                                 * (uint)sizeof(__nv_bfloat16)));
                }
                const uint (&vf)[4] = frag[mdp & 1];
                #pragma unroll
                for (int mq = 0; mq < WARP_Q / MMA_M; ++mq) {
                    mma_m16n8k16(sp[mq].p[mkv], vf + 0, oAcc[mq][mdp * 2]);
                    mma_m16n8k16(sp[mq].p[mkv], vf + 2, oAcc[mq][mdp * 2 + 1]);
                }
            }
        }
    }

    FK_DEVICE_FUSE void pvMma(const SPTile (&sp)[WARP_Q / MMA_M],
                              const uint (&vReg)[BLOCK_KV / MMA_K][HEAD_DIM / MMA_N][2],
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
                                    const uint base, const uint vThread,
                                    float (&oAcc)[WARP_Q / MMA_M][HEAD_DIM / MMA_N][4],
                                    const int tokOff = 0) {
        using namespace attention_mma_detail;
        #pragma unroll
        for (int mkv = 0; mkv < SPAN / MMA_K; ++mkv) {
            #pragma unroll
            for (int md = 0; md < HEAD_DIM / MMA_N; md += 2) {
                uint frag[4];
                uint addr = base + vThread;
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
                                     const uint (&vmd)[HEAD_DIM / MMA_N],
                                     float (&oAcc)[WARP_Q / MMA_M][HEAD_DIM / MMA_N][4],
                                     const int tokOff = 0) {
        using namespace attention_mma_detail;
        #pragma unroll
        for (int mkv = 0; mkv < SPAN / MMA_K; ++mkv) {
            #pragma unroll
            for (int md = 0; md < HEAD_DIM / MMA_N; md += 2) {
                uint frag[4];
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

    /* split-KV partial writeout: UNNORMALIZED oAcc plus per-row (m, l).
       Layout [bh, seq_q, splits, HEAD_DIM+2] — fp32. Lane (laneId%4) owns
       cols {2t, 2t+1} of each MMA_N block, rows rowA/rowB as in writeOut. */
    FK_DEVICE_FUSE void writePartial(const float (&oAcc)[WARP_Q / MMA_M][HEAD_DIM / MMA_N][4],
                                     const float (&rowMax)[WARP_Q / MMA_M][2],
                                     const float (&rowSum)[WARP_Q / MMA_M][2],
                                     const Params& p, const int qBlockBase,
                                     const int warpId, const int laneId, const int bh) {
        const int split = blockIdx.z;
        const long strideRow = (long)p.splits * (HEAD_DIM + 2);
        #pragma unroll
        for (int mq = 0; mq < WARP_Q / MMA_M; ++mq) {
            const int rowA = qBlockBase + warpId * WARP_Q + mq * MMA_M + laneId / 4;
            const int rowB = rowA + 8;
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int row = half == 0 ? rowA : rowB;
                if (row >= p.seq_q) continue;
                float* dst = p.partial
                    + ((long)bh * p.seq_q + row) * strideRow
                    + (long)split * (HEAD_DIM + 2);
                #pragma unroll
                for (int md = 0; md < HEAD_DIM / MMA_N; ++md) {
                    const int col = md * MMA_N + (laneId % 4) * 2;
                    dst[col]     = oAcc[mq][md][half * 2];
                    dst[col + 1] = oAcc[mq][md][half * 2 + 1];
                }
                if (laneId % 4 == 0) {
                    dst[HEAD_DIM]     = rowMax[mq][half];
                    dst[HEAD_DIM + 1] = rowSum[mq][half];
                }
            }
        }
    }

    FK_DEVICE_FUSE void storePair(OT* dst, const float a, const float b) {
        if constexpr (std::is_same_v<OT, __nv_bfloat16>) {
            *reinterpret_cast<__nv_bfloat162*>(dst) = __float22bfloat162_rn({ a, b });
        } else {
            dst[0] = low_precission::attnFromF32<OT>(a);
            dst[1] = low_precission::attnFromF32<OT>(b);
        }
    }

    template <bool SPLIT_KV = false>
    static __device__ void exec(const Params& p) {
        using namespace attention_mma_detail;
        extern __shared__ char smemRaw[];
        char* smemBytes = smemRaw;
        const uint smemBase =
            static_cast<uint>(__cvta_generic_to_shared(smemRaw));

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

        const uint qThread = swz(((warpId * WARP_Q + laneId % 16) * HEAD_DIM
                                      + (laneId / 16) * 8) * sizeof(__nv_bfloat16));
        const uint kThread = swz(((laneId % 8) * HEAD_DIM
                                      + (laneId / 8) * 8) * sizeof(__nv_bfloat16));
        const uint vThread = swz(((laneId % 16) * HEAD_DIM
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

        uint qReg[(DEEP_Q_SMEM || DUAL_Q_SMEM) ? 1 : WARP_Q / MMA_M]
                     [(DEEP_Q_SMEM || DUAL_Q_SMEM) ? 1 : HEAD_DIM / MMA_K][4];
        // FP8_QK state (dead/eliminated when the path is off — all uses sit
        // inside `if constexpr (FP8_QK)`).
        uint qReg8[WARP_Q / MMA_M][HEAD_DIM / 32][4];
        float qs[WARP_Q / MMA_M][2];
        if constexpr (FP8_QK) {
#ifdef FK_HAS_FP8
            // quantize the staged bf16 Q per-row to e4m3 + pull A-fragments.
            quantizeQTile(smemBytes, warpId, laneId);
            loadQF8Frags(smemBytes, qReg8, qs, warpId, laneId);
#endif
        } else if constexpr (!DEEP_Q_SMEM && !DUAL_Q_SMEM) {
            #pragma unroll
            for (int mq = 0; mq < WARP_Q / MMA_M; ++mq) {
                #pragma unroll
                for (int md = 0; md < HEAD_DIM / MMA_K; ++md) {
                    uint addr = smemBase + qThread;
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

        if constexpr (RAW_KV) {
            // ============= cp.async schedule (fa-5090 v5 staggering) =========
            // K double-buffered, V single-buffered; K[kv+1] is issued right
            // after the QK^T mma so it streams during softmax + PV.
            const __nv_bfloat16* kPlane = p.k.params.data + (long)bh * p.seq_k * HEAD_DIM;
            const __nv_bfloat16* vPlane = p.v.params.data + (long)bh * p.seq_k * HEAD_DIM;
            // DEEP_Q_SMEM: KV buffers live after the resident Q tile.
            const auto kBuf = [&](int i) {
                return Q_RES_B + (uint)(i % 2) * KV_BUF_B; };
            const uint vBufOff = Q_RES_B + 2 * KV_BUF_B;  // single V buffer

            // prefetch first in-range tile
            if (itBegin < itEnd && tileActive(itBegin * BLOCK_KV)) {
                cpasyncTile<KV_GROUPS>(smemBase + kBuf(itBegin), kPlane,
                                       itBegin * BLOCK_KV, p.seq_k);
            }
            cp_async_commit();

            // Hoisted per-md swizzled base addresses (see sMmaStreamH /
            // sMmaStreamDualQ addressing notes): computed ONCE before the kv
            // loop; the hot loop then addresses with reg + constant only.
            // GATED to d128 (measured, 4-binary ABAB): d128 +1-6% (causal
            // s2048 257->272 TF); d64 BQ128 (250 regs) has no headroom for
            // the kmd/vmd arrays -> s8192 dense 369->348 TF. d64 keeps the
            // in-loop addressing.
            static constexpr bool HOIST_ADDR = HEAD_DIM == 128;
            uint qmd[DUAL_Q_SMEM ? HEAD_DIM / MMA_K : 1];
            uint kmd[HOIST_ADDR ? HEAD_DIM / MMA_K : 1];
            uint vmd[HOIST_ADDR ? HEAD_DIM / MMA_N : 1];
            if constexpr (HOIST_ADDR) {
                #pragma unroll
                for (int md = 0; md < HEAD_DIM / MMA_K; ++md) {
                    const uint x = md * MMA_K * (uint)sizeof(__nv_bfloat16);
                    kmd[md] = (smemBase + kThread) ^ x;
                    if constexpr (DUAL_Q_SMEM) qmd[md] = (smemBase + qThread) ^ x;
                }
                #pragma unroll
                for (int md = 0; md < HEAD_DIM / MMA_N; ++md) {
                    vmd[md] = (smemBase + vBufOff + vThread)
                              ^ (md * MMA_N * (uint)sizeof(__nv_bfloat16));
                }
            } else if constexpr (DUAL_Q_SMEM) {
                #pragma unroll
                for (int md = 0; md < HEAD_DIM / MMA_K; ++md) {
                    qmd[md] = (smemBase + qThread)
                              ^ (md * MMA_K * (uint)sizeof(__nv_bfloat16));
                }
            }

            // NOTE (round 9bis, rejected): #pragma unroll 2 on this loop was
            // measured at -9..-14% on most shapes (+2..6% on two) — the
            // bigger body blows the I-cache/sched window; reverted.
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

                if constexpr (SPLIT_S) {
                    // ============== split-S: two SEQUENTIAL KV_SPAN passes ==
                    // Only ONE SPSpan (16 fp32 regs at BKV64) is live at a
                    // time: span0 runs QK^T->softmax->PV to completion before
                    // span1 starts. Safe because K[kv] is double-buffered
                    // (K[kv+1] streams into the OTHER buffer) and V[kv] is
                    // single-buffered but only overwritten after the next
                    // iteration's __syncthreads. The K[kv+1] prefetch is
                    // issued between span0's QK^T and softmax (v5 stagger);
                    // V wait happens before span0's PV, so span1 runs with
                    // both tiles resident — zero extra syncs vs the fused
                    // tile.
                    // span 0 ---------------------------------------------
                    SPSpan sp[WARP_Q / MMA_M] = {};
                    if (act) {
                        if constexpr (HOIST_ADDR) {
                            sMmaStreamH<KV_SPAN>(qReg, kBuf(kv), kmd, sp, 0);
                        } else {
                            sMmaStream<KV_SPAN>(qReg, smemBase + kBuf(kv),
                                                kThread, sp, 0);
                        }
                    }

                    // prefetch K[kv+1] (overlaps softmax + PV, v5 staggering)
                    if (kv + 1 < itEnd && tileActive(offKV + BLOCK_KV)) {
                        cpasyncTile<KV_GROUPS>(smemBase + kBuf(kv + 1), kPlane,
                                               offKV + BLOCK_KV, p.seq_k);
                    }
                    cp_async_commit();

                    const bool needsMask = tileNeedsMask(offKV);
                    if (act) {
                        if (needsMask) {
                            softmaxTile<false, KV_SPAN>(sp, oAcc, rowMax, rowSum,
                                                        p, offKV, qBlockBase,
                                                        warpId, laneId);
                        } else {
                            softmaxTile<true, KV_SPAN>(sp, oAcc, rowMax, rowSum,
                                                       p, offKV, qBlockBase,
                                                       warpId, laneId);
                        }
                    }

                    // wait V[kv] (1 group outstanding: K[kv+1])
                    cp_async_wait<1>();
                    __syncthreads();

                    if (act) {
                        if constexpr (HOIST_ADDR) {
                            pvMmaStreamH<KV_SPAN>(sp, vmd, oAcc, 0);
                        } else {
                            pvMmaStream<KV_SPAN>(sp, smemBase + vBufOff,
                                                 vThread, oAcc, 0);
                        }
                        // span 1 ------------------------------------------
                        // reuses the same sp storage; K[kv]/V[kv] still
                        // resident (see buffer-lifetime note above).
                        #pragma unroll
                        for (int mq = 0; mq < WARP_Q / MMA_M; ++mq)
                            #pragma unroll
                            for (int i = 0; i < KV_SPAN / MMA_N; ++i)
                                #pragma unroll
                                for (int e = 0; e < 4; ++e) sp[mq].s[i][e] = 0.f;
                        if constexpr (HOIST_ADDR) {
                            sMmaStreamH<KV_SPAN>(qReg, kBuf(kv), kmd, sp, KV_SPAN);
                        } else {
                            sMmaStream<KV_SPAN>(qReg, smemBase + kBuf(kv),
                                                kThread, sp, KV_SPAN);
                        }
                        if (needsMask) {
                            softmaxTile<false, KV_SPAN>(sp, oAcc, rowMax, rowSum,
                                                        p, offKV + KV_SPAN,
                                                        qBlockBase, warpId, laneId);
                        } else {
                            softmaxTile<true, KV_SPAN>(sp, oAcc, rowMax, rowSum,
                                                       p, offKV + KV_SPAN,
                                                       qBlockBase, warpId, laneId);
                        }
                        pvMmaStream<KV_SPAN>(sp, smemBase + vBufOff, vThread,
                                             oAcc, KV_SPAN);
                    }
                } else {
                    SPTile sp[WARP_Q / MMA_M] = {};
                    if (act) {
                        if constexpr (DUAL_Q_SMEM) {
                            // dual-reuse schedule: Q resident at smem[0],
                            // md-pair outer / mkv inner (see sMmaStreamDualQ)
                            sMmaStreamDualQ(kBuf(kv), qmd, kmd, sp);
                        } else if constexpr (DEEP_Q_SMEM) {
                            // deep schedule: Q AND K streamed from smem
                            sMmaStreamDeepQ(smemBase, qThread,
                                            smemBase + kBuf(kv), kThread, sp);
                        } else {
                            uint kReg[BLOCK_KV / MMA_N][HEAD_DIM / MMA_K][2];
                            loadKFrags(smemBase + kBuf(kv), kThread, kReg);
                            sMma(qReg, kReg, sp);
                        }
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
                        if constexpr (DUAL_Q_SMEM) {
                            pvMmaStreamDual(sp, smemBase + vBufOff, vThread, oAcc);
                        } else if constexpr (HOIST_ADDR) {
                            // stream V frags ldmatrix->mma (no vReg staging)
                            pvMmaStreamH(sp, vmd, oAcc);
                        } else {
                            pvMmaStream(sp, smemBase + vBufOff, vThread, oAcc);
                        }
                    }
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

            const auto kBuf = [&](int i) { return (uint)(i % 2) * KV_BUF_B; };
            const uint vBufOff = 2 * KV_BUF_B;
            // byte staging area after the 4 bf16 buffers
            const uint stageBase = 4 * KV_BUF_B;
            const auto kStage = [&](int i) {
                return stageBase + (uint)(i % 2) * (KV_BUF_B / 2); };
            const uint vStage = stageBase + 2 * (KV_BUF_B / 2);

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
                        uint kReg[BLOCK_KV / MMA_N][HEAD_DIM / MMA_K][2];
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
                    uint vReg[BLOCK_KV / MMA_K][HEAD_DIM / MMA_N][2];
                    loadVFrags(smemBase + vBufOff, vThread, vReg);
                    pvMma(sp, vReg, oAcc);
                }
            }
        } else {
            // ============== register-prefetch schedule (fused prologues) =====
            const auto kBuf = [](int i) -> uint { return (i % 2) * KV_BUF_B; };
            const auto vBuf = [](int i) -> uint { return (2 + (i % 2)) * KV_BUF_B; };

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
                    uint kReg[BLOCK_KV / MMA_N][HEAD_DIM / MMA_K][2];
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

                    uint vReg[BLOCK_KV / MMA_K][HEAD_DIM / MMA_N][2];
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

        if constexpr (SPLIT_KV) {
            writePartial(oAcc, rowMax, rowSum, p, qBlockBase, warpId, laneId, bh);
        } else {
            writeOut(oAcc, rowSum, p, qBlockBase, warpId, laneId, bh);
        }
    }
};

/* split-KV combine: one block per (bh, row-chunk); exact online-softmax
   merge of `splits` unnormalized partials. expBase: 2^x when the producer
   kernel ran log2-domain softmax, e^x otherwise. */
template <typename OT, int HEAD_DIM, bool LOG2D>
__global__ void flashFwdSplitCombineKernel(const float* __restrict__ partial,
                                           OT* __restrict__ o,
                                           const int seqQ, const int splits) {
    const int bh = blockIdx.y;
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= seqQ) return;
    const float* base = partial
        + ((long)bh * seqQ + row) * (long)splits * (HEAD_DIM + 2);
    // global max over splits (uniform across the row's threads)
    float gm = -FLT_MAX;
    for (int s = 0; s < splits; ++s)
        gm = fmaxf(gm, base[s * (HEAD_DIM + 2) + HEAD_DIM]);
    // each thread owns HEAD_DIM/32 columns
    float acc[HEAD_DIM / 32] = {};
    float l = 0.f;
    for (int s = 0; s < splits; ++s) {
        const float* ps = base + s * (HEAD_DIM + 2);
        const float m = ps[HEAD_DIM];
        if (m == -FLT_MAX) continue;            // empty chunk
        const float w = LOG2D ? exp2f(m - gm) : __expf(m - gm);
        l += ps[HEAD_DIM + 1] * w;
        #pragma unroll
        for (int c = 0; c < HEAD_DIM / 32; ++c)
            acc[c] += ps[threadIdx.x + c * 32] * w;
    }
    const float inv = l > 0.f ? 1.f / l : 0.f;
    OT* dst = o + ((long)bh * seqQ + row) * HEAD_DIM;
    #pragma unroll
    for (int c = 0; c < HEAD_DIM / 32; ++c) {
        if constexpr (std::is_same_v<OT, __nv_bfloat16>) {
            dst[threadIdx.x + c * 32] = __float2bfloat16(acc[c] * inv);
        } else {
            dst[threadIdx.x + c * 32] = static_cast<OT>(acc[c] * inv);
        }
    }
}

template <typename OT, int HEAD_DIM, typename QIOp, typename KIOp, typename VIOp,
          typename EpilogueIOp, int BLOCK_Q, int BLOCK_KV, int NUM_WARPS,
          typename ScoreModOp, bool SPLIT_KV = false>
__global__ void launchFlashAttentionMmaDPP_Kernel(
        const __grid_constant__ typename FlashAttentionMmaDPP<
            OT, HEAD_DIM, QIOp, KIOp, VIOp, EpilogueIOp,
            BLOCK_Q, BLOCK_KV, NUM_WARPS, ScoreModOp>::Params params) {
    FlashAttentionMmaDPP<OT, HEAD_DIM, QIOp, KIOp, VIOp, EpilogueIOp,
                         BLOCK_Q, BLOCK_KV, NUM_WARPS,
                         ScoreModOp>::template exec<SPLIT_KV>(params);
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
                    q, k, v, o, batchHeads, seqQ, seqK, causal, stream,
                    scaleOverride, epilogue, scoreMod, sparse);
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
                    q, k, v, o, batchHeads, seqQ, seqK, causal, stream,
                    scaleOverride, epilogue, scoreMod, sparse);
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
                    q, k, v, o, batchHeads, seqQ, seqK, causal, stream,
                    scaleOverride, epilogue, scoreMod, sparse);
                return;
            }
        }
    }
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
    typename DPP::Params params{ q, k, v, o, seqQ, seqK, scale, causal,
                                 epilogue, scoreMod, sparse };
    const int numQ = (seqQ + BQ - 1) / BQ;
    const dim3 block(DPP::THREADS, 1, 1);
    const int smemBytes = DPP::SMEM_BYTES;
    auto* kernel = launchFlashAttentionMmaDPP_Kernel<OT, HEAD_DIM, QIOp, KIOp, VIOp,
                                                     EpilogueIOp, BQ, BKV, NUM_WARPS,
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
    float* partial = nullptr;
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
            gpuErrchk(cudaMallocAsync(&partial, bytes, stream.getCUDAStream()));
            params.partial = partial;
            params.splits = splits;
        }
    }
    const dim3 grid(numQ, batchHeads, splits);
    if (splits > 1) {
        auto* splitKernel = launchFlashAttentionMmaDPP_Kernel<
            OT, HEAD_DIM, QIOp, KIOp, VIOp, EpilogueIOp, BQ, BKV, NUM_WARPS,
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
        flashFwdSplitCombineKernel<OT, HEAD_DIM, DPP::LOG2_DOMAIN>
            <<<cgrid, cblock, 0, stream.getCUDAStream()>>>(partial, o, seqQ, splits);
        gpuErrchk(cudaGetLastError());
        gpuErrchk(cudaFreeAsync(partial, stream.getCUDAStream()));
    } else {
        kernel<<<grid, block, smemBytes, stream.getCUDAStream()>>>(params);
        gpuErrchk(cudaGetLastError());
    }
}

} // namespace fk

#endif // defined(__NVCC__)

#endif // FK_ATTENTION_FLASH_ATTENTION_MMA_H
