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
 * (one kernel), with FKL-only superpowers that handmade FA kernels
 * (flash-attention repo and friends) cannot offer:
 *
 *  1. PROLOGUE FUSION (the fa-5090 insight, generalized): the DPP does
 *     NOT read raw pointers. Q, K and V are each a Read or ReadBack IOp
 *     (PerThreadRead, Int8TokenDequantRead, or any read.then(compute...)
 *     fusion) and the algorithm reads EVERY data element through that
 *     IOp. Whatever preprocessing the prologue encodes (dequantization,
 *     scaling, RoPE-style maps, casts...) happens in-register at load
 *     time - one global read, no preprocessing kernel, no DRAM round-trip.
 *
 *  2. EPILOGUE FUSION: any FKL compute IOp chain (Mul, Add, Cast,
 *     Saturate, ... composed with .then()) is applied IN-REGISTER to the
 *     attention output before the single global write. out = (o/l) | chain.
 *
 *  3. COMPRESSED KV CACHE (Int8TokenDequantRead / KVLayout::INT8_PER_TOKEN):
 *     K and V live in global memory as int8 with one fp32 scale per token:
 *       k_int8[t][d] = round(k[t][d]/kScale[t]), kScale[t] = max|k[t]|/127
 *     -> 4x smaller cache vs fp32 (2x vs fp16) + 2 floats/token.
 *     Dequantization is just the K/V prologue Read IOp: it happens
 *     in-register inside the q.k dot product and the p*v accumulation.
 *     The compressed cache is NEVER inflated in global memory.
 *
 * Algorithm (Dao, FlashAttention-2): never materialize S = QK^T. Each
 * query row keeps a running (m, l, o); every KV position updates them with
 * the online-softmax rescaling trick. One pass over KV, O(seq) memory,
 * exact (fp32 accumulation).
 *
 * Mapping (SM 12x friendly - no TMEM/tcgen05, per the fa-5090 analysis,
 * correctness-first SIMT baseline before mma.sync tiling):
 *   grid.y = batch*heads; grid.x = ceil(seq_q / WARPS_PER_BLOCK)
 *   1 warp per query row; q and o live in registers spread across lanes
 *   (HEAD_DIM/32 each); dot products warp-reduce via __shfl_xor_sync;
 *   K/V tiles staged cooperatively in shared memory THROUGH the prologue
 *   IOps (tiles hold post-prologue fp32, so the prologue runs exactly
 *   once per element).
 *
 * Element addressing convention for the prologue IOps (3D):
 *   thread.x = position inside the head (0..HEAD_DIM-1)
 *   thread.y = token index in the sequence
 *   thread.z = batch*head plane
 */

#include <fused_kernel/algorithms/attention/softmax.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>

namespace fk {

enum class KVLayout { DENSE, INT8_PER_TOKEN };

// Identity IOp: applied when no epilogue chain is given.
struct AttentionIdentityEpilogue {
    FK_HOST_DEVICE_CNST friend float operator|(const float v,
                                               const AttentionIdentityEpilogue&) {
        return v;
    }
};

/* Int8TokenDequantRead: Read IOp for the compressed KV cache.
 * data  : (batch*heads, seq, head_dim) int8, C-contiguous
 * scales: one float per token, laid out (batch*heads * seq)
 * exec(thread) returns float(data[z][y][x]) * scales[z*seq + y].
 * Usable standalone in any FKL pipeline, and as the K/V prologue of
 * FlashAttentionDPP. Compose further preprocessing with .then(...). */
struct Int8TokenDequantReadParams {
    RawPtr<ND::_3D, int8_t> data;
    const float* scales;
};

struct Int8TokenDequantRead {
private:
    using Parent = ReadOperation<int8_t, Int8TokenDequantReadParams, float,
                                 TF::DISABLED, Int8TokenDequantRead>;
    using SelfType = Int8TokenDequantRead;
public:
    FK_STATIC_STRUCT(Int8TokenDequantRead, SelfType)
    DECLARE_READ_PARENT

    FK_HOST_DEVICE_FUSE float exec(const Point thread, const ParamsType& params) {
        const int8_t q = *PtrAccessor<ND::_3D>::cr_point(thread, params.data);
        const int seq = static_cast<int>(params.data.dims.height);
        const float sc = params.scales[(long)thread.z * seq + thread.y];
        return static_cast<float>(q) * sc;
    }

    FK_HOST_DEVICE_FUSE uint num_elems_x(const Point thread, const OperationDataType& opData) {
        return opData.params.data.dims.width;
    }
    FK_HOST_DEVICE_FUSE uint num_elems_y(const Point thread, const OperationDataType& opData) {
        return opData.params.data.dims.height;
    }
    FK_HOST_DEVICE_FUSE uint num_elems_z(const Point thread, const OperationDataType& opData) {
        return opData.params.data.dims.planes;
    }
    FK_HOST_DEVICE_FUSE uint pitch(const Point thread, const OperationDataType& opData) {
        return opData.params.data.dims.pitch;
    }
    FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
        return { num_elems_x(Point{0,0,0}, opData),
                 num_elems_y(Point{0,0,0}, opData),
                 num_elems_z(Point{0,0,0}, opData) };
    }
};

// ---- prologue builders ----------------------------------------------------
// Wrap a raw (batch*heads, seq, head_dim) C-contiguous pointer as the
// canonical attention Read IOp. Fuse extra preprocessing with .then(...).
template <typename T>
inline auto makeAttentionRead(const T* data, const int batchHeads,
                              const int seq, const int headDim) {
    const RawPtr<ND::_3D, T> ptr{
        const_cast<T*>(data),
        PtrDims<ND::_3D>(static_cast<uint>(headDim), static_cast<uint>(seq),
                         static_cast<uint>(batchHeads), 1,
                         static_cast<uint>(headDim * sizeof(T))) };
    return PerThreadRead<ND::_3D, T>::build(ptr);
}

// int8-per-token compressed K or V cache as a dequantizing Read IOp.
inline auto makeInt8KVRead(const int8_t* data, const float* scales,
                           const int batchHeads, const int seq,
                           const int headDim) {
    const RawPtr<ND::_3D, int8_t> ptr{
        const_cast<int8_t*>(data),
        PtrDims<ND::_3D>(static_cast<uint>(headDim), static_cast<uint>(seq),
                         static_cast<uint>(batchHeads), 1,
                         static_cast<uint>(headDim)) };
    return Int8TokenDequantRead::build(Int8TokenDequantReadParams{ ptr, scales });
}

#if defined(__NVCC__)

/* The DPP. QIOp/KIOp/VIOp are INSTANTIABLE Read or ReadBack IOps (possibly
 * fused with compute continuations): the algorithm reads each element of
 * Q, K and V exclusively through IOp::Operation::exec(thread, iop). */
template <typename OT, int HEAD_DIM,
          typename QIOp, typename KIOp, typename VIOp,
          typename EpilogueIOp = AttentionIdentityEpilogue,
          int BLOCK_N = 32, int WARPS_PER_BLOCK = 4>
struct FlashAttentionDPP {
private:
    using SelfType = FlashAttentionDPP<OT, HEAD_DIM, QIOp, KIOp, VIOp,
                                       EpilogueIOp, BLOCK_N, WARPS_PER_BLOCK>;
public:
    FK_STATIC_STRUCT(FlashAttentionDPP, SelfType)

    static_assert(HEAD_DIM % 32 == 0, "HEAD_DIM must be a multiple of 32");
    static_assert(isAnyReadType<QIOp>, "Q prologue must be a Read or ReadBack IOp");
    static_assert(isAnyReadType<KIOp>, "K prologue must be a Read or ReadBack IOp");
    static_assert(isAnyReadType<VIOp>, "V prologue must be a Read or ReadBack IOp");
    static constexpr int ELEMS_PER_LANE = HEAD_DIM / 32;
    static constexpr int THREADS = WARPS_PER_BLOCK * 32;

    struct Params {
        QIOp q;                  // prologue Read/ReadBack IOp for Q
        KIOp k;                  // prologue Read/ReadBack IOp for K
        VIOp v;                  // prologue Read/ReadBack IOp for V
        OT* o;                   // (batch*heads, seq_q, HEAD_DIM) output
        int seq_q;
        int seq_k;
        float scale;             // logit scale, usually rsqrt(HEAD_DIM)
        bool causal;
        EpilogueIOp epilogue;    // fused IOp chain on the output (pre-write)
    };

    template <typename IOp>
    FK_DEVICE_FUSE float readElem(const IOp& iop, const int x, const int y, const int z) {
        // ALL data elements enter the algorithm through this single point:
        // the prologue IOp's exec. Fused continuations run here, in-register.
        return low_precision::attnToF32(IOp::Operation::exec(Point{ x, y, z }, iop));
    }

    FK_COOP_DEVICE_FUSE exec(const Params& p) {
        // Tiles hold POST-prologue fp32: the prologue runs once per element.
        __shared__ float kTile[BLOCK_N][HEAD_DIM];
        __shared__ float vTile[BLOCK_N][HEAD_DIM];

        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        const int qIdx = blockIdx.x * WARPS_PER_BLOCK + warp;
        const int bh = blockIdx.y;
        const bool active = qIdx < p.seq_q;

        float qReg[ELEMS_PER_LANE];
        float oAcc[ELEMS_PER_LANE];
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_LANE; ++e) {
            // Q PROLOGUE: read through the IOp, in-register, at load time.
            qReg[e] = active ? readElem(p.q, lane + 32 * e, qIdx, bh) : 0.f;
            oAcc[e] = 0.f;
        }
        float m = -FLT_MAX;
        float l = 0.f;

        const int blockMaxQ = blockIdx.x * WARPS_PER_BLOCK + WARPS_PER_BLOCK - 1;
        const int kvEnd = p.causal ? ::min(p.seq_k, blockMaxQ + 1) : p.seq_k;

        for (int tile = 0; tile < kvEnd; tile += BLOCK_N) {
            const int tileLen = ::min(BLOCK_N, kvEnd - tile);

            __syncthreads();
            // K/V PROLOGUES: cooperative staging reads every element of the
            // tile through the K and V IOps (dequant & friends fuse here).
            for (int idx = threadIdx.x; idx < tileLen * HEAD_DIM; idx += THREADS) {
                const int r = idx / HEAD_DIM;
                const int c = idx % HEAD_DIM;
                kTile[r][c] = readElem(p.k, c, tile + r, bh);
                vTile[r][c] = readElem(p.v, c, tile + r, bh);
            }
            __syncthreads();

            if (!active) { continue; }

            for (int j = 0; j < tileLen; ++j) {
                const int kvIdx = tile + j;
                if (p.causal && kvIdx > qIdx) { break; }

                // s = q . k_j
                float partial = 0.f;
                #pragma unroll
                for (int e = 0; e < ELEMS_PER_LANE; ++e) {
                    partial += qReg[e] * kTile[j][lane + 32 * e];
                }
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    partial += __shfl_xor_sync(0xffffffffu, partial, offset);
                }
                const float s = partial * p.scale;

                const float mNew = cxp::max::f(m, s);
                const float corr = cxp::expf::f(m - mNew);
                const float pj = cxp::expf::f(s - mNew);
                l = l * corr + pj;
                #pragma unroll
                for (int e = 0; e < ELEMS_PER_LANE; ++e) {
                    oAcc[e] = oAcc[e] * corr + pj * vTile[j][lane + 32 * e];
                }
                m = mNew;
            }
        }

        if (active) {
            const float invL = l > 0.f ? 1.f / l : 0.f;
            const long oRow = ((long)bh * p.seq_q + qIdx) * HEAD_DIM;
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_LANE; ++e) {
                // EPILOGUE FUSION: the FKL IOp chain runs in-register on the
                // normalized output, then ONE global write.
                const float r = (oAcc[e] * invL) | p.epilogue;
                p.o[oRow + lane + 32 * e] = low_precision::attnFromF32<OT>(r);
            }
        }
    }
};

template <typename OT, int HEAD_DIM, typename QIOp, typename KIOp, typename VIOp,
          typename EpilogueIOp, int BLOCK_N, int WARPS_PER_BLOCK>
__global__ void launchFlashAttentionDPP_Kernel(
        const __grid_constant__ typename FlashAttentionDPP<OT, HEAD_DIM, QIOp, KIOp, VIOp, EpilogueIOp, BLOCK_N, WARPS_PER_BLOCK>::Params params) {
    FlashAttentionDPP<OT, HEAD_DIM, QIOp, KIOp, VIOp, EpilogueIOp, BLOCK_N, WARPS_PER_BLOCK>::exec(params);
}

/* IOp-first API: pass Q/K/V as Read/ReadBack IOps (the prologues). */
template <int HEAD_DIM, int BLOCK_N = 32, int WARPS_PER_BLOCK = 4,
          typename OT = float, typename QIOp, typename KIOp, typename VIOp,
          typename EpilogueIOp = AttentionIdentityEpilogue>
inline void executeFlashAttention(
        const QIOp& q, const KIOp& k, const VIOp& v, OT* o,
        const int batchHeads, const int seqQ, const int seqK,
        const bool causal, Stream_<ParArch::GPU_NVIDIA>& stream,
        const float scaleOverride = -1.f, const EpilogueIOp& epilogue = {}) {
    using DPP = FlashAttentionDPP<OT, HEAD_DIM, QIOp, KIOp, VIOp,
                                  EpilogueIOp, BLOCK_N, WARPS_PER_BLOCK>;
    const float scale = scaleOverride > 0.f ? scaleOverride
                                            : rsqrtf(static_cast<float>(HEAD_DIM));
    const typename DPP::Params params{ q, k, v, o, seqQ, seqK, scale, causal, epilogue };
    const dim3 grid((seqQ + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, batchHeads, 1);
    const dim3 block(DPP::THREADS, 1, 1);
    launchFlashAttentionDPP_Kernel<OT, HEAD_DIM, QIOp, KIOp, VIOp, EpilogueIOp, BLOCK_N, WARPS_PER_BLOCK>
        <<<grid, block, 0, stream.getCUDAStream()>>>(params);
    gpuErrchk(cudaGetLastError());
}

/* Pointer convenience API (kept for compatibility): builds the canonical
 * prologue Read IOps and forwards. KVLayout::INT8_PER_TOKEN selects the
 * Int8TokenDequantRead prologue for K and V. */
template <typename T, int HEAD_DIM, KVLayout KVL = KVLayout::DENSE,
          int BLOCK_N = 32, int WARPS_PER_BLOCK = 4,
          typename EpilogueIOp = AttentionIdentityEpilogue>
inline void executeFlashAttention(
        const T* q,
        const std::conditional_t<KVL == KVLayout::INT8_PER_TOKEN, int8_t, T>* k,
        const std::conditional_t<KVL == KVLayout::INT8_PER_TOKEN, int8_t, T>* v,
        T* o, const int batchHeads, const int seqQ, const int seqK,
        const bool causal, Stream_<ParArch::GPU_NVIDIA>& stream,
        const float* kScale = nullptr, const float* vScale = nullptr,
        const float scaleOverride = -1.f,
        const EpilogueIOp& epilogue = {}) {
    const auto qIOp = makeAttentionRead(q, batchHeads, seqQ, HEAD_DIM);
    if constexpr (KVL == KVLayout::INT8_PER_TOKEN) {
        const auto kIOp = makeInt8KVRead(k, kScale, batchHeads, seqK, HEAD_DIM);
        const auto vIOp = makeInt8KVRead(v, vScale, batchHeads, seqK, HEAD_DIM);
        executeFlashAttention<HEAD_DIM, BLOCK_N, WARPS_PER_BLOCK>(
            qIOp, kIOp, vIOp, o, batchHeads, seqQ, seqK, causal, stream,
            scaleOverride, epilogue);
    } else {
        const auto kIOp = makeAttentionRead(k, batchHeads, seqK, HEAD_DIM);
        const auto vIOp = makeAttentionRead(v, batchHeads, seqK, HEAD_DIM);
        executeFlashAttention<HEAD_DIM, BLOCK_N, WARPS_PER_BLOCK>(
            qIOp, kIOp, vIOp, o, batchHeads, seqQ, seqK, causal, stream,
            scaleOverride, epilogue);
    }
}

#endif // defined(__NVCC__)

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
            mx = std::max(mx, std::abs(low_precision::attnToF32(dense[(long)t * headDim + d])));
        }
        const float sc = mx > 0.f ? mx / 127.f : 1.f;
        scales[t] = sc;
        for (int d = 0; d < headDim; ++d) {
            const float x = low_precision::attnToF32(dense[(long)t * headDim + d]) / sc;
            q8[(long)t * headDim + d] = static_cast<int8_t>(std::nearbyint(x));
        }
    }
}

} // namespace fk

// ============================ FP8 KV CACHE ==================================
// fp8 e4m3 per-token-scaled KV cache (the FA4-sm12x / PR #2634 recipe,
// expressed as a prologue Read IOp). Same 2x-vs-bf16 memory saving as int8
// but the e4m3 grid is non-uniform (more precision near 0), which suits
// attention tails. Storage: data fp8 e4m3, one fp32 scale per token
// (scale = max|row| / 448, e4m3 max normal = 448).
#if __has_include(<cuda_fp8.h>)
#include <cuda_fp8.h>
#define FK_HAS_FP8 1

namespace fk {

struct Fp8TokenDequantReadParams {
    RawPtr<ND::_3D, int8_t> data;   // raw e4m3 bytes (int8_t storage)
    const float* scales;            // one fp32 scale per token
};

struct Fp8TokenDequantRead {
private:
    using Parent = ReadOperation<int8_t, Fp8TokenDequantReadParams, float,
                                 TF::DISABLED, Fp8TokenDequantRead>;
    using SelfType = Fp8TokenDequantRead;
public:
    FK_STATIC_STRUCT(Fp8TokenDequantRead, SelfType)
    DECLARE_READ_PARENT

    // NOTE: plain static (not FK_HOST_DEVICE_FUSE): fp8 conversion ops are
    // not constexpr (same trap as __shared__ in cooperative DPP exec).
    static __host__ __device__ __forceinline__
    float exec(const Point thread, const ParamsType& params) {
        const int8_t raw = *PtrAccessor<ND::_3D>::cr_point(thread, params.data);
        const int seq = static_cast<int>(params.data.dims.height);
        const float sc = params.scales[(long)thread.z * seq + thread.y];
#if defined(__CUDA_ARCH__)
        const __nv_fp8_e4m3* f8 = reinterpret_cast<const __nv_fp8_e4m3*>(&raw);
        return static_cast<float>(*f8) * sc;
#else
        __nv_fp8_e4m3 f8;
        f8.__x = static_cast<__nv_fp8_storage_t>(raw);
        return static_cast<float>(f8) * sc;
#endif
    }

    FK_HOST_DEVICE_FUSE uint num_elems_x(const Point thread, const OperationDataType& opData) {
        return opData.params.data.dims.width;
    }
    FK_HOST_DEVICE_FUSE uint num_elems_y(const Point thread, const OperationDataType& opData) {
        return opData.params.data.dims.height;
    }
    FK_HOST_DEVICE_FUSE uint num_elems_z(const Point thread, const OperationDataType& opData) {
        return opData.params.data.dims.planes;
    }
    FK_HOST_DEVICE_FUSE uint pitch(const Point thread, const OperationDataType& opData) {
        return opData.params.data.dims.pitch;
    }
    FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
        return { num_elems_x(Point{0,0,0}, opData),
                 num_elems_y(Point{0,0,0}, opData),
                 num_elems_z(Point{0,0,0}, opData) };
    }
};

// fp8-per-token compressed K or V cache as a dequantizing Read IOp.
inline auto makeFp8KVRead(const void* data, const float* scales,
                          const int batchHeads, const int seq,
                          const int headDim) {
    const RawPtr<ND::_3D, int8_t> ptr{
        const_cast<int8_t*>(static_cast<const int8_t*>(data)),
        PtrDims<ND::_3D>(static_cast<uint>(headDim), static_cast<uint>(seq),
                         static_cast<uint>(batchHeads), 1,
                         static_cast<uint>(headDim)) };
    return Fp8TokenDequantRead::build(Fp8TokenDequantReadParams{ ptr, scales });
}

// host-side reference packing: scale[t] = max|row|/448 (e4m3 max normal).
template <typename T>
inline void quantizeKVCacheFp8Host(const T* dense, void* f8out, float* scales,
                                   const int tokens, const int headDim) {
    int8_t* out = static_cast<int8_t*>(f8out);
    for (int t = 0; t < tokens; ++t) {
        float mx = 0.f;
        for (int d = 0; d < headDim; ++d) {
            mx = std::max(mx, std::abs(low_precision::attnToF32(dense[(long)t * headDim + d])));
        }
        const float sc = mx > 0.f ? mx / 448.f : 1.f;
        scales[t] = sc;
        for (int d = 0; d < headDim; ++d) {
            const float x = low_precision::attnToF32(dense[(long)t * headDim + d]) / sc;
            const __nv_fp8_e4m3 f8(x);
            out[(long)t * headDim + d] = static_cast<int8_t>(f8.__x);
        }
    }
}

} // namespace fk

#endif // __has_include(<cuda_fp8.h>)

#endif // FK_ATTENTION_FLASH_ATTENTION_H
