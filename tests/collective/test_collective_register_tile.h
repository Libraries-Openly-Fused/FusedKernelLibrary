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

#define __ONLY_CU__  // register-tiled device-only mainloop
#include <tests/main.h>

#include <fused_kernel/algorithms/collective/register_tile.h>
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

/* RegTile fragment loader: A is (16*WM, K) row-major, B is (8*WN, K) row-major,
 * D is (16*WM, 8*WN) row-major — all reached through IOps. rowAtom/colAtom
 * select the sub-tile within the warp's WM x WN block. Mapping verified in
 * test_collective_mma.h. */
struct RegTileLoader {
    using Atom = MmaBf16_16x8x16;
    template <typename IOp>
    __device__ __forceinline__ uint32_t pk(const IOp&, const typename IOp::ParamsType& pr, int row, int col) const {
        const __nv_bfloat16 x0 = IOp::Operation::exec(Point{ col,     row, 0 }, pr);
        const __nv_bfloat16 x1 = IOp::Operation::exec(Point{ col + 1, row, 0 }, pr);
        uint32_t r; __nv_bfloat16 two[2] = { x0, x1 }; memcpy(&r, two, 4); return r;
    }
    template <typename AIOp>
    __device__ __forceinline__ void loadA(const AIOp& A, int rowAtom, int kBase, int lane, uint32_t (&a)[4]) const {
        const int g = lane >> 2, t = lane & 3, rb = rowAtom * 16;
        a[0]=pk(A,A.params, rb+g,   kBase+2*t);   a[1]=pk(A,A.params, rb+g+8, kBase+2*t);
        a[2]=pk(A,A.params, rb+g,   kBase+2*t+8); a[3]=pk(A,A.params, rb+g+8, kBase+2*t+8);
    }
    template <typename BIOp>
    __device__ __forceinline__ void loadB(const BIOp& B, int colAtom, int kBase, int lane, uint32_t (&b)[2]) const {
        const int g = lane >> 2, t = lane & 3, cb = colAtom * 8;
        b[0]=pk(B,B.params, cb+g, kBase+2*t); b[1]=pk(B,B.params, cb+g, kBase+2*t+8);
    }
    template <typename WIOp>
    __device__ __forceinline__ void storeD(const WIOp& D, int rowAtom, int colAtom, int lane, const float (&d)[4]) const {
        const int g = lane >> 2, t = lane & 3, rb = rowAtom*16, cb = colAtom*8;
        WIOp::Operation::exec(Point{ cb+2*t,   rb+g,   0 }, d[0], D.params);
        WIOp::Operation::exec(Point{ cb+2*t+1, rb+g,   0 }, d[1], D.params);
        WIOp::Operation::exec(Point{ cb+2*t,   rb+g+8, 0 }, d[2], D.params);
        WIOp::Operation::exec(Point{ cb+2*t+1, rb+g+8, 0 }, d[3], D.params);
    }
};

template <int WM, int WN, typename AIOp, typename BIOp, typename DIOp>
__global__ void regTileGemmKernel(AIOp aIOp, BIOp bIOp, DIOp dIOp, int K) {
    const auto reads = make_tuple(aIOp, bIOp);
    RegTileLoader loader;
    WarpTileMmaMainloopDPP<ParArch::GPU_NVIDIA, MmaBf16_16x8x16, WM, WN, RegTileLoader>::exec(
        WarpTileMainloopDetails{ K }, reads, dIOp, loader);
}

static int failures = 0;

template <int WM, int WN>
static void runCase(int K) {
    constexpr int M = 16 * WM, N = 8 * WN;
    std::mt19937 rng(9);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float> Af(M*K), Bf(N*K);
    for (auto& x : Af) x = dist(rng);
    for (auto& x : Bf) x = dist(rng);
    std::vector<__nv_bfloat16> Ah(M*K), Bh(N*K);
    for (int i=0;i<M*K;++i) Ah[i]=__float2bfloat16(Af[i]);
    for (int i=0;i<N*K;++i) Bh[i]=__float2bfloat16(Bf[i]);

    Ptr2D<__nv_bfloat16> A(K, M), B(K, N);
    Ptr2D<float> D(N, M);
    cudaMemcpy2D(A.ptr().data, A.ptr().dims.pitch, Ah.data(), K*sizeof(__nv_bfloat16), K*sizeof(__nv_bfloat16), M, cudaMemcpyHostToDevice);
    cudaMemcpy2D(B.ptr().data, B.ptr().dims.pitch, Bh.data(), K*sizeof(__nv_bfloat16), K*sizeof(__nv_bfloat16), N, cudaMemcpyHostToDevice);

    const auto aIOp = TileReadOp<__nv_bfloat16>::build(A.ptr());
    const auto bIOp = TileReadOp<__nv_bfloat16>::build(B.ptr());
    const auto dIOp = TileWriteOp<float>::build(D.ptr());
    regTileGemmKernel<WM, WN><<<1, 32>>>(aIOp, bIOp, dIOp, K);
    cudaDeviceSynchronize();

    std::vector<float> Dout(M*N);
    cudaMemcpy2D(Dout.data(), N*sizeof(float), D.ptr().data, D.ptr().dims.pitch, N*sizeof(float), M, cudaMemcpyDeviceToHost);

    double maxErr = 0;
    for (int m=0;m<M;++m) for (int n=0;n<N;++n) {
        double acc=0; for (int k=0;k<K;++k) acc += (double)__bfloat162float(Ah[m*K+k])*(double)__bfloat162float(Bh[n*K+k]);
        maxErr = std::max(maxErr, std::abs((double)Dout[m*N+n]-acc));
    }
    const double tol = 5e-2 * (K/16.0);
    if (maxErr > tol) { printf("FAIL regtile %dx%d K=%d maxErr=%.4e\n", WM, WN, K, maxErr); ++failures; }
    else printf("WarpTileMmaMainloopDPP %dx%d (1 warp) GEMM %dx%dx%-4d vs fp64: PASS (maxErr=%.2e)\n", WM, WN, M, N, K, maxErr);
}

int launch() {
    runCase<2, 2>(64);    // 1 warp computes 32x16 via 4 atoms/K-step
    runCase<2, 4>(128);   // 32x32 via 8 atoms/K-step (reuse)
    runCase<4, 2>(256);   // 64x16
    return failures == 0 ? 0 : -1;
}
