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

#define __ONLY_CU__  // composes device-only mainloop + MMA + epilogue
#include <tests/main.h>

#include <fused_kernel/algorithms/collective/mainloop.h>
#include <fused_kernel/algorithms/collective/epilogue.h>
#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/data/tuple.h>

#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <cuda_bf16.h>

using namespace fk;

/* Minimal Read/Write IOps over a RawPtr (bf16 has no VectorTraits, so we
 * can't use PerThreadRead). Data still flows through IOp::Operation::exec. */
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

/* Epilogue functors composing the FKL `value | epilogue` fusion pattern. */
struct ScaleBiasEpilogue {   // 2x + 1
    FK_HOST_DEVICE_CNST friend float operator|(const float v, const ScaleBiasEpilogue&) { return 2.f * v + 1.f; }
};
struct ReluEpilogue {
    FK_HOST_DEVICE_CNST friend float operator|(const float v, const ReluEpilogue&) { return v > 0.f ? v : 0.f; }
};

struct Bf16FragLoaderRowMajor {
    using Atom = MmaBf16_16x8x16;
    template <typename AIOp>
    __device__ __forceinline__ uint32_t pack2(const AIOp&, int row, int col) const {
        const __nv_bfloat16 x0 = AIOp::Operation::exec(Point{ col,     row, 0 }, *params_);
        const __nv_bfloat16 x1 = AIOp::Operation::exec(Point{ col + 1, row, 0 }, *params_);
        uint32_t r; __nv_bfloat16 two[2] = { x0, x1 }; memcpy(&r, two, 4); return r;
    }
    const typename TileReadOp<__nv_bfloat16>::ParamsType* params_ = nullptr;
    template <typename AIOp>
    __device__ __forceinline__ void loadA(const AIOp& A, int lane, int kBase, uint32_t (&a)[4]) {
        params_ = &A.params; const int g = lane >> 2, t = lane & 3;
        a[0] = pack2(A, g, kBase+2*t); a[1] = pack2(A, g+8, kBase+2*t);
        a[2] = pack2(A, g, kBase+2*t+8); a[3] = pack2(A, g+8, kBase+2*t+8);
    }
    template <typename BIOp>
    __device__ __forceinline__ void loadB(const BIOp& B, int lane, int kBase, uint32_t (&b)[2]) {
        params_ = &B.params; const int g = lane >> 2, t = lane & 3;
        b[0] = pack2(B, g, kBase+2*t); b[1] = pack2(B, g, kBase+2*t+8);
    }
    template <typename WIOp>
    __device__ __forceinline__ void storeD(const WIOp&, int, const float (&)[4]) const {}
};

template <typename AIOp, typename BIOp, typename DIOp, typename Epilogue>
__global__ void gemmEpiKernel(AIOp aIOp, BIOp bIOp, DIOp dIOp, int K, Epilogue epi) {
    const int lane = threadIdx.x;
    float d[4] = {0,0,0,0};
    Bf16FragLoaderRowMajor ld;
    // Mainloop accumulates D in registers...
    {
        const int g = lane >> 2, t = lane & 3;
        for (int kB = 0; kB < K; kB += 16) {
            uint32_t a[4], b[2];
            ld.loadA(aIOp, lane, kB, a);
            ld.loadB(bIOp, lane, kB, b);
            MmaWarpDPP<MmaBf16_16x8x16>::exec(a, b, d);
        }
    }
    // ...then the cooperative fused epilogue writes through the Write IOp.
    EpilogueDPP<ParArch::GPU_NVIDIA, MmaBf16_16x8x16, MmaBf16DStore>::exec(0, 0, lane, d, dIOp, epi);
}

static int failures = 0;

template <int EPI, typename Epilogue>
static void runCase(const char* name, int K, Epilogue epilogue) {
    std::mt19937 rng(7);
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
    gemmEpiKernel<<<1,32>>>(aIOp, bIOp, dIOp, K, epilogue);
    cudaDeviceSynchronize();

    std::vector<float> Dout(16*8);
    cudaMemcpy2D(Dout.data(), 8*sizeof(float), D.ptr().data, D.ptr().dims.pitch, 8*sizeof(float), 16, cudaMemcpyDeviceToHost);

    double maxErr = 0;
    for (int m=0;m<16;++m) for (int n=0;n<8;++n) {
        double acc = 0;
        for (int k=0;k<K;++k) acc += (double)__bfloat162float(Ah[m*K+k]) * (double)__bfloat162float(Bh[n*K+k]);
        double ref = (EPI==0)? acc : (EPI==1)? 2.0*acc+1.0 : (acc>0.0?acc:0.0);
        maxErr = std::max(maxErr, std::abs((double)Dout[m*8+n] - ref));
    }
    const double tol = 5e-2 * (K/16.0) + 1e-3;
    if (maxErr > tol) { printf("FAIL %s K=%d maxErr=%.4e (tol %.3e)\n", name, K, maxErr, tol); ++failures; }
    else printf("EpilogueDPP %-16s K=%-4d vs fp64 oracle: PASS (maxErr=%.2e)\n", name, K, maxErr);
}

int launch() {
    runCase<0>("identity", 64, GemmIdentityEpilogue{});
    runCase<0>("identity", 256, GemmIdentityEpilogue{});
    runCase<1>("scale+bias 2x+1", 64, ScaleBiasEpilogue{});
    runCase<1>("scale+bias 2x+1", 256, ScaleBiasEpilogue{});
    runCase<2>("ReLU", 64, ReluEpilogue{});
    runCase<2>("ReLU", 256, ReluEpilogue{});
    return failures == 0 ? 0 : -1;
}
