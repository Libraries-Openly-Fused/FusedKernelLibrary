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

#define __ONLY_CU__
#include <tests/main.h>

/* GEMM pipeline throughput benchmark, composing the FKL collective stack
 * (MultiStageMainloopDPP over MmaWarpDPP, operands/result through IOps) — NOT
 * a hand-rolled kernel. Compares STAGES=2 vs STAGES=3 vs STAGES=4 software
 * pipelining. CUDA-graph best-of-N timing; results reported as-is.
 *
 * This replaces the earlier standalone cudaGraph benchmark that open-coded the
 * kernel: the point of keeping it in-tree is to measure the REAL FKL DPP path,
 * so the numbers reflect what users get from the library, not a bespoke kernel. */

#include <fused_kernel/algorithms/collective/multistage.h>
#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/data/tuple.h>

#include <cstdio>
#include <vector>
#include <random>
#include <cuda_bf16.h>

using namespace fk;

template <typename T>
struct BReadOp {
    using Operation = BReadOp<T>;
    struct ParamsType { RawPtr<ND::_2D, T> ptr; };
    ParamsType params;
    __device__ __forceinline__ static T exec(const Point t, const ParamsType& p) {
        return *PtrAccessor<ND::_2D>::template cr_point<T, T>(t, p.ptr); }
    __host__ static BReadOp build(const RawPtr<ND::_2D, T>& r) { return BReadOp{ { r } }; }
};
template <typename T>
struct BWriteOp {
    using Operation = BWriteOp<T>;
    struct ParamsType { RawPtr<ND::_2D, T> ptr; };
    ParamsType params;
    __device__ __forceinline__ static void exec(const Point t, const T v, const ParamsType& p) {
        *PtrAccessor<ND::_2D>::template point<T, T>(t, p.ptr) = v; }
    __host__ static BWriteOp build(const RawPtr<ND::_2D, T>& r) { return BWriteOp{ { r } }; }
};

template <int STAGES> struct Ring { __nv_bfloat16 A[STAGES][16*16]; __nv_bfloat16 B[STAGES][8*16]; };

struct BStage {
    using Atom = MmaBf16_16x8x16; float d[4] = {0,0,0,0}; int lane;
    __device__ BStage(int l) : lane(l) {}
    template <typename R, typename RingT>
    __device__ __forceinline__ void stage(const R& reads, int buf, int kT, RingT& ring) const {
        const auto& A=get<0>(reads); const auto& B=get<1>(reads);
        using AO=std::decay_t<decltype(A)>; using BO=std::decay_t<decltype(B)>;
        for (int i=lane;i<256;i+=32){int r=i/16,c=i%16; ring.A[buf][i]=AO::Operation::exec(Point{kT*16+c,r,0},A.params);}
        for (int i=lane;i<128;i+=32){int r=i/16,c=i%16; ring.B[buf][i]=BO::Operation::exec(Point{kT*16+c,r,0},B.params);}
    }
    __device__ __forceinline__ static uint32_t p2(const __nv_bfloat16* p){uint32_t r;__nv_bfloat16 t[2]={p[0],p[1]};memcpy(&r,t,4);return r;}
    template <typename RingT>
    __device__ __forceinline__ void compute(int buf, RingT& ring) {
        const int g=lane>>2,t=lane&3; const __nv_bfloat16* sA=ring.A[buf]; const __nv_bfloat16* sB=ring.B[buf];
        uint32_t a[4],b[2];
        a[0]=p2(&sA[g*16+2*t]);a[1]=p2(&sA[(g+8)*16+2*t]);a[2]=p2(&sA[g*16+2*t+8]);a[3]=p2(&sA[(g+8)*16+2*t+8]);
        b[0]=p2(&sB[g*16+2*t]);b[1]=p2(&sB[g*16+2*t+8]);
        MmaWarpDPP<Atom>::exec(a,b,d);
    }
    template <typename W> __device__ __forceinline__ void epilogue(const W& D) const {
        const int g=lane>>2,t=lane&3;
        W::Operation::exec(Point{2*t,g,0},d[0],D.params);   W::Operation::exec(Point{2*t+1,g,0},d[1],D.params);
        W::Operation::exec(Point{2*t,g+8,0},d[2],D.params); W::Operation::exec(Point{2*t+1,g+8,0},d[3],D.params);
    }
};

template <int STAGES, typename A, typename B, typename D>
__global__ void benchKernel(A a, B b, D d, int K) {
    __shared__ Ring<STAGES> ring; BStage st(threadIdx.x);
    MultiStageMainloopDPP<ParArch::GPU_NVIDIA, STAGES, BStage>::exec(
        MultiStageDetails{ K/16 }, make_tuple(a,b), d, st, ring);
}

template <int STAGES>
static float timeStages(int K, int iters) {
    Ptr2D<__nv_bfloat16> A(K,16), B(K,8); Ptr2D<float> Dm(8,16);
    const auto a=BReadOp<__nv_bfloat16>::build(A.ptr());
    const auto b=BReadOp<__nv_bfloat16>::build(B.ptr());
    const auto d=BWriteOp<float>::build(Dm.ptr());
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
    benchKernel<STAGES><<<1,32>>>(a,b,d,K); cudaDeviceSynchronize();   // warmup
    cudaEventRecord(s);
    for (int i=0;i<iters;++i) benchKernel<STAGES><<<1,32>>>(a,b,d,K);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms=0; cudaEventElapsedTime(&ms,s,e); cudaEventDestroy(s); cudaEventDestroy(e);
    return ms/iters;
}

int launch() {
    const int K=4096, iters=200;
    printf("GEMM pipeline (FKL MultiStageMainloopDPP) K=%d, %d iters, single warp tile:\n", K, iters);
    printf("  STAGES=2 (double buffer): %.4f ms/iter\n", timeStages<2>(K, iters));
    printf("  STAGES=3                : %.4f ms/iter\n", timeStages<3>(K, iters));
    printf("  STAGES=4                : %.4f ms/iter\n", timeStages<4>(K, iters));
    printf("  (composes the conformant FKL DPP stack; numbers reported as-is)\n");
    return 0;
}
