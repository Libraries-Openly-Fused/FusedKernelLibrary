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

#define __ONLY_CU__  // composes device-only MMA + mainloop
#include <tests/main.h>

#include <fused_kernel/algorithms/collective/mainloop.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/data/tuple.h>

#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <cuda_bf16.h>

using namespace fk;

/* Minimal Read/Write IOps for the operand/output tiles. FKL's PerThreadRead
 * needs VectorTraits, which isn't defined for __nv_bfloat16, so the test
 * supplies tiny Read/Write Ops over a RawPtr — still the FKL contract (data
 * reached through IOp::Operation::exec, no raw pointer in the DPP), just
 * without thread fusion. */
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

/* FragLoader for MmaBf16_16x8x16 over row-major operand tiles reached through
 * Read IOps. A is (M=16, K) row-major, B is (N=8, K) row-major; output D
 * (16x8) is written through a Write IOp. The per-lane mapping (verified in
 * test_collective_mma.h) lives here in the policy; the mainloop DPP doesn't
 * know it. Operand elements are read via IOp::Operation::exec(Point), so no
 * raw device pointer ever enters the DPP. */
struct Bf16FragLoaderRowMajor {
    using Atom = MmaBf16_16x8x16;

    template <typename AIOp>
    __device__ __forceinline__ uint32_t pack2(const AIOp& iop, int row, int col) const {
        const __nv_bfloat16 x0 = AIOp::Operation::exec(Point{ col,     row, 0 }, iop.params);
        const __nv_bfloat16 x1 = AIOp::Operation::exec(Point{ col + 1, row, 0 }, iop.params);
        uint32_t r; __nv_bfloat16 two[2] = { x0, x1 }; memcpy(&r, two, 4); return r;
    }
    template <typename AIOp>
    __device__ __forceinline__ void loadA(const AIOp& A, int lane, int kBase, uint32_t (&a)[4]) const {
        const int g = lane >> 2, t = lane & 3;
        a[0] = pack2(A, g,     kBase + 2 * t);
        a[1] = pack2(A, g + 8, kBase + 2 * t);
        a[2] = pack2(A, g,     kBase + 2 * t + 8);
        a[3] = pack2(A, g + 8, kBase + 2 * t + 8);
    }
    template <typename BIOp>
    __device__ __forceinline__ void loadB(const BIOp& B, int lane, int kBase, uint32_t (&b)[2]) const {
        const int g = lane >> 2, t = lane & 3;
        b[0] = pack2(B, g, kBase + 2 * t);
        b[1] = pack2(B, g, kBase + 2 * t + 8);
    }
    template <typename WIOp>
    __device__ __forceinline__ void storeD(const WIOp& D, int lane, const float (&d)[4]) const {
        const int g = lane >> 2, t = lane & 3;
        WIOp::Operation::exec(Point{ 2 * t,     g,     0 }, d[0], D.params);
        WIOp::Operation::exec(Point{ 2 * t + 1, g,     0 }, d[1], D.params);
        WIOp::Operation::exec(Point{ 2 * t,     g + 8, 0 }, d[2], D.params);
        WIOp::Operation::exec(Point{ 2 * t + 1, g + 8, 0 }, d[3], D.params);
    }
};

template <typename AIOp, typename BIOp, typename DIOp>
__global__ void gemmTileKernel(AIOp aIOp, BIOp bIOp, DIOp dIOp, int K) {
    const auto reads = make_tuple(aIOp, bIOp);
    Bf16FragLoaderRowMajor loader;
    TileMmaMainloopDPP<ParArch::GPU_NVIDIA, MmaBf16_16x8x16, Bf16FragLoaderRowMajor>::exec(
        TileMmaMainloopDetails{ K }, reads, dIOp, loader);
}

static int failures = 0;

static void runGemm(int K) {
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float> Af(16*K), Bf(8*K);
    for (auto& x : Af) x = dist(rng);
    for (auto& x : Bf) x = dist(rng);
    std::vector<__nv_bfloat16> Ah(16*K), Bh(8*K);
    for (int i = 0; i < 16*K; ++i) Ah[i] = __float2bfloat16(Af[i]);
    for (int i = 0; i < 8*K;  ++i) Bh[i] = __float2bfloat16(Bf[i]);

    Ptr2D<__nv_bfloat16> A(K, 16), B(K, 8);
    Ptr2D<float> D(8, 16);
    cudaMemcpy2D(A.ptr().data, A.ptr().dims.pitch, Ah.data(), K*sizeof(__nv_bfloat16),
                 K*sizeof(__nv_bfloat16), 16, cudaMemcpyHostToDevice);
    cudaMemcpy2D(B.ptr().data, B.ptr().dims.pitch, Bh.data(), K*sizeof(__nv_bfloat16),
                 K*sizeof(__nv_bfloat16), 8, cudaMemcpyHostToDevice);

    const auto aIOp = TileReadOp<__nv_bfloat16>::build(A.ptr());
    const auto bIOp = TileReadOp<__nv_bfloat16>::build(B.ptr());
    const auto dIOp = TileWriteOp<float>::build(D.ptr());
    gemmTileKernel<<<1, 32>>>(aIOp, bIOp, dIOp, K);
    cudaDeviceSynchronize();

    std::vector<float> Dout(16*8);
    cudaMemcpy2D(Dout.data(), 8*sizeof(float), D.ptr().data, D.ptr().dims.pitch,
                 8*sizeof(float), 16, cudaMemcpyDeviceToHost);

    double maxErr = 0;
    for (int m = 0; m < 16; ++m)
        for (int n = 0; n < 8; ++n) {
            double acc = 0;
            for (int k = 0; k < K; ++k)
                acc += (double)__bfloat162float(Ah[m*K+k]) * (double)__bfloat162float(Bh[n*K+k]);
            maxErr = std::max(maxErr, std::abs((double)Dout[m*8+n] - acc));
        }
    const double tol = 5e-2 * (K / 16.0);
    if (maxErr > tol) { printf("FAIL gemm K=%d maxErr=%.4e (tol %.3e)\n", K, maxErr, tol); ++failures; }
    else printf("TileMmaMainloopDPP GEMM 16x8x%-4d vs fp64 oracle: PASS (maxErr=%.2e)\n", K, maxErr);
}

int launch() {
    runGemm(16);    // single atom (loop runs once)
    runGemm(64);    // 4 K-tiles
    runGemm(256);   // 16 K-tiles — real accumulation over the loop
    return failures == 0 ? 0 : -1;
}
