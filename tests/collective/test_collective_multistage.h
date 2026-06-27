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

#define __ONLY_CU__  // multi-stage cp.async device-only pipeline
#include <tests/main.h>

#include <fused_kernel/algorithms/collective/multistage.h>
#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/data/tuple.h>

#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <cuda_bf16.h>

using namespace fk;

template <typename T>
struct TileReadOp {
    using Operation = TileReadOp<T>;
    struct ParamsType { RawPtr<ND::_2D, T> ptr; };
    ParamsType params;
    __device__ __forceinline__ static T exec(const Point thread, const ParamsType& p) {
        return *PtrAccessor<ND::_2D>::template cr_point<T, T>(thread, p.ptr);
    }
    __host__ static TileReadOp build(const RawPtr<ND::_2D, T>& r) { return TileReadOp{ { r } }; }
};
template <typename T>
struct TileWriteOp {
    using Operation = TileWriteOp<T>;
    struct ParamsType { RawPtr<ND::_2D, T> ptr; };
    ParamsType params;
    __device__ __forceinline__ static void exec(const Point thread, const T value, const ParamsType& p) {
        *PtrAccessor<ND::_2D>::template point<T, T>(thread, p.ptr) = value;
    }
    __host__ static TileWriteOp build(const RawPtr<ND::_2D, T>& r) { return TileWriteOp{ { r } }; }
};

/* smem ring: STAGES buffers, each one MmaBf16 K-tile (16x16 A + 8x16 B). */
template <int STAGES>
struct Ring { __nv_bfloat16 A[STAGES][16*16]; __nv_bfloat16 B[STAGES][8*16]; };

/* StagePolicy: stages a K-tile gmem->smem ring slot through the Read IOps,
 * accumulates a 16x8 atom per compute() call, and writes via the Write IOp. */
struct GemmStage {
    using Atom = MmaBf16_16x8x16;
    float d[4] = {0,0,0,0};
    int lane;
    __device__ GemmStage(int lane_) : lane(lane_) {}

    template <typename ReadIOps, typename RingT>
    __device__ __forceinline__ void stage(const ReadIOps& reads, int buf, int kTile, RingT& ring) const {
        const auto& A = get<0>(reads); const auto& B = get<1>(reads);
        using AOp = std::decay_t<decltype(A)>; using BOp = std::decay_t<decltype(B)>;
        for (int i = lane; i < 16*16; i += 32) {
            const int r = i/16, c = i%16;
            ring.A[buf][i] = AOp::Operation::exec(Point{ kTile*16 + c, r, 0 }, A.params);
        }
        for (int i = lane; i < 8*16; i += 32) {
            const int r = i/16, c = i%16;
            ring.B[buf][i] = BOp::Operation::exec(Point{ kTile*16 + c, r, 0 }, B.params);
        }
    }
    __device__ __forceinline__ static uint32_t p2(const __nv_bfloat16* p) {
        uint32_t r; __nv_bfloat16 t[2] = { p[0], p[1] }; memcpy(&r, t, 4); return r; }
    template <typename RingT>
    __device__ __forceinline__ void compute(int buf, RingT& ring) {
        const int g = lane >> 2, t = lane & 3;
        const __nv_bfloat16* sA = ring.A[buf]; const __nv_bfloat16* sB = ring.B[buf];
        uint32_t a[4], b[2];
        a[0]=p2(&sA[g*16+2*t]); a[1]=p2(&sA[(g+8)*16+2*t]);
        a[2]=p2(&sA[g*16+2*t+8]); a[3]=p2(&sA[(g+8)*16+2*t+8]);
        b[0]=p2(&sB[g*16+2*t]); b[1]=p2(&sB[g*16+2*t+8]);
        MmaWarpDPP<Atom>::exec(a, b, d);
    }
    template <typename WIOp>
    __device__ __forceinline__ void epilogue(const WIOp& D) const {
        const int g = lane >> 2, t = lane & 3;
        WIOp::Operation::exec(Point{ 2*t,   g,   0 }, d[0], D.params);
        WIOp::Operation::exec(Point{ 2*t+1, g,   0 }, d[1], D.params);
        WIOp::Operation::exec(Point{ 2*t,   g+8, 0 }, d[2], D.params);
        WIOp::Operation::exec(Point{ 2*t+1, g+8, 0 }, d[3], D.params);
    }
};

template <int STAGES, typename AIOp, typename BIOp, typename DIOp>
__global__ void multistageGemmKernel(AIOp aIOp, BIOp bIOp, DIOp dIOp, int K) {
    __shared__ Ring<STAGES> ring;
    const int lane = threadIdx.x;
    const auto reads = make_tuple(aIOp, bIOp);
    GemmStage stage(lane);
    MultiStageMainloopDPP<ParArch::GPU_NVIDIA, STAGES, GemmStage>::exec(
        MultiStageDetails{ K / 16 }, reads, dIOp, stage, ring);
}

static int failures = 0;

template <int STAGES>
static void runGemm(int K) {
    std::mt19937 rng(13);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float> Af(16*K), Bf(8*K);
    for (auto& x : Af) x = dist(rng);
    for (auto& x : Bf) x = dist(rng);
    std::vector<__nv_bfloat16> Ah(16*K), Bh(8*K);
    for (int i=0;i<16*K;++i) Ah[i]=__float2bfloat16(Af[i]);
    for (int i=0;i<8*K;++i)  Bh[i]=__float2bfloat16(Bf[i]);

    Ptr2D<__nv_bfloat16> A(K, 16), B(K, 8);
    Ptr2D<float> D(8, 16);
    cudaMemcpy2D(A.ptr().data, A.ptr().dims.pitch, Ah.data(), K*sizeof(__nv_bfloat16), K*sizeof(__nv_bfloat16), 16, cudaMemcpyHostToDevice);
    cudaMemcpy2D(B.ptr().data, B.ptr().dims.pitch, Bh.data(), K*sizeof(__nv_bfloat16), K*sizeof(__nv_bfloat16), 8, cudaMemcpyHostToDevice);

    const auto aIOp = TileReadOp<__nv_bfloat16>::build(A.ptr());
    const auto bIOp = TileReadOp<__nv_bfloat16>::build(B.ptr());
    const auto dIOp = TileWriteOp<float>::build(D.ptr());
    multistageGemmKernel<STAGES><<<1,32>>>(aIOp, bIOp, dIOp, K);
    cudaDeviceSynchronize();

    std::vector<float> Dout(16*8);
    cudaMemcpy2D(Dout.data(), 8*sizeof(float), D.ptr().data, D.ptr().dims.pitch, 8*sizeof(float), 16, cudaMemcpyDeviceToHost);

    double maxErr = 0;
    for (int m=0;m<16;++m) for (int n=0;n<8;++n) {
        double acc=0; for (int k=0;k<K;++k) acc += (double)__bfloat162float(Ah[m*K+k])*(double)__bfloat162float(Bh[n*K+k]);
        maxErr = std::max(maxErr, std::abs((double)Dout[m*8+n]-acc));
    }
    const double tol = 5e-2 * (K/16.0);
    if (maxErr > tol) { printf("FAIL multistage STAGES=%d K=%d maxErr=%.4e\n", STAGES, K, maxErr); ++failures; }
    else printf("MultiStageMainloopDPP STAGES=%d GEMM 16x8x%-4d vs fp64: PASS (maxErr=%.2e)\n", STAGES, K, maxErr);
}

int launch() {
    runGemm<2>(64);    // double buffer
    runGemm<3>(128);   // 3-stage
    runGemm<4>(256);   // 4-stage
    return failures == 0 ? 0 : -1;
}
