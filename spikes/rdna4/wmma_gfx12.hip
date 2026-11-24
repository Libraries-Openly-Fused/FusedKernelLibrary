// Build and run (not part of the CMake build):
//   $HIP_PATH/bin/amdclang++ -x hip --offload-arch=gfx1200 -O3 -std=c++20 wmma_gfx12.hip -o wmma_gfx12 && ./wmma_gfx12
// Spike: gfx12 (RDNA4) wave32 WMMA 16x16x16, bf16 and fp8 (OCP e4m3) -> f32.
// 1) validates the fragment-to-lane mapping against a CPU matmul,
// 2) probes it with one-hot inputs when it doesn't match,
// 3) measures raw throughput (independent accumulator chains per wave).
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <random>
#include <type_traits>
#include <vector>

#define CHECK(x)                                                                                                       \
    do {                                                                                                               \
        hipError_t e_ = (x);                                                                                           \
        if (e_ != hipSuccess) {                                                                                        \
            printf("HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__);                              \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

typedef short short8 __attribute__((ext_vector_type(8)));
typedef float float8 __attribute__((ext_vector_type(8)));
typedef int int2v __attribute__((ext_vector_type(2)));

// Hypothesis (gfx12 wave32, 16x16x16):
//   A (M x K, row-major):  lane l -> row l%16, k = 8*(l/16) + i, i = 0..7
//   B (K x N, row-major):  lane l -> col l%16, k = 8*(l/16) + i
//   D (M x N):             lane l -> col l%16, row = 8*(l/16) + i
__device__ __forceinline__ int aRow(int lane) { return lane % 16; }
__device__ __forceinline__ int kOf(int lane, int i) { return 8 * (lane / 16) + i; }
__device__ __forceinline__ int dRow(int lane, int i) { return 8 * (lane / 16) + i; }

__global__ void mmaBf16(const unsigned short* A, const unsigned short* B, float* D) {
    const int lane = threadIdx.x;
    short8 a, b;
    for (int i = 0; i < 8; ++i) {
        a[i] = (short)A[aRow(lane) * 16 + kOf(lane, i)];
        b[i] = (short)B[kOf(lane, i) * 16 + lane % 16];
    }
    float8 c = {};
    c = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
    for (int i = 0; i < 8; ++i) D[dRow(lane, i) * 16 + lane % 16] = c[i];
}

__global__ void mmaFp8(const unsigned char* A, const unsigned char* B, float* D) {
    const int lane = threadIdx.x;
    unsigned char a8[8], b8[8];
    for (int i = 0; i < 8; ++i) {
        a8[i] = A[aRow(lane) * 16 + kOf(lane, i)];
        b8[i] = B[kOf(lane, i) * 16 + lane % 16];
    }
    int2v a, b;
    memcpy(&a, a8, 8);
    memcpy(&b, b8, 8);
    float8 c = {};
    c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
    for (int i = 0; i < 8; ++i) D[dRow(lane, i) * 16 + lane % 16] = c[i];
}

// Throughput: CHAINS independent accumulators per wave, ITERS wmma each.
template <int CHAINS, int ITERS>
__global__ void tputBf16(float* out, const short seed) {
    short8 a, b;
    for (int i = 0; i < 8; ++i) { a[i] = (short)(seed + threadIdx.x + i); b[i] = (short)(seed - i); }
    float8 c[CHAINS] = {};
    for (int it = 0; it < ITERS; ++it) {
#pragma unroll
        for (int ch = 0; ch < CHAINS; ++ch) c[ch] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c[ch]);
    }
    float s = 0.f;
    for (int ch = 0; ch < CHAINS; ++ch)
        for (int i = 0; i < 8; ++i) s += c[ch][i];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

template <int CHAINS, int ITERS>
__global__ void tputFp8(float* out, const int seed) {
    int2v a = {seed + (int)threadIdx.x, seed * 3}, b = {seed ^ 0x01010101, seed};
    float8 c[CHAINS] = {};
    for (int it = 0; it < ITERS; ++it) {
#pragma unroll
        for (int ch = 0; ch < CHAINS; ++ch) c[ch] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c[ch]);
    }
    float s = 0.f;
    for (int ch = 0; ch < CHAINS; ++ch)
        for (int i = 0; i < 8; ++i) s += c[ch][i];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

static unsigned short f2bf(float f) { __hip_bfloat16 h(f); unsigned short u; memcpy(&u, &h, 2); return u; }
static float bf2f(unsigned short h) { unsigned int u = (unsigned int)h << 16; float f; memcpy(&f, &u, 4); return f; }
static unsigned char f2f8(float f) { return __hip_fp8_e4m3(f).__x; }
static float f82f(unsigned char b) { __hip_fp8_e4m3 v; v.__x = b; return (float)v; }

template <typename T, typename Kernel, typename ToF>
static double runCheck(Kernel kernel, const std::vector<T>& A, const std::vector<T>& B, ToF toF, std::vector<float>& D) {
    T *dA, *dB;
    float* dD;
    CHECK(hipMalloc(&dA, 256 * sizeof(T)));
    CHECK(hipMalloc(&dB, 256 * sizeof(T)));
    CHECK(hipMalloc(&dD, 256 * sizeof(float)));
    CHECK(hipMemcpy(dA, A.data(), 256 * sizeof(T), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dB, B.data(), 256 * sizeof(T), hipMemcpyHostToDevice));
    kernel<<<1, 32>>>(dA, dB, dD);
    CHECK(hipGetLastError());
    D.resize(256);
    CHECK(hipMemcpy(D.data(), dD, 256 * sizeof(float), hipMemcpyDeviceToHost));
    CHECK(hipFree(dA)); CHECK(hipFree(dB)); CHECK(hipFree(dD));
    double maxErr = 0.0;
    for (int m = 0; m < 16; ++m)
        for (int n = 0; n < 16; ++n) {
            double ref = 0.0;
            for (int k = 0; k < 16; ++k) ref += (double)toF(A[m * 16 + k]) * (double)toF(B[k * 16 + n]);
            maxErr = std::max(maxErr, std::abs(ref - (double)D[m * 16 + n]));
        }
    return maxErr;
}

template <typename Kernel>
static double timeKernel(Kernel launch, int reps) {
    hipEvent_t s, e;
    CHECK(hipEventCreate(&s)); CHECK(hipEventCreate(&e));
    launch(); CHECK(hipDeviceSynchronize());   // warm-up
    std::vector<float> ms;
    for (int r = 0; r < reps; ++r) {
        CHECK(hipEventRecord(s));
        launch();
        CHECK(hipEventRecord(e));
        CHECK(hipEventSynchronize(e));
        float t; CHECK(hipEventElapsedTime(&t, s, e)); ms.push_back(t);
    }
    std::sort(ms.begin(), ms.end());
    return ms[ms.size() / 2];
}

int main() {
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<unsigned short> Ab(256), Bb(256);
    for (auto& x : Ab) x = f2bf(dist(rng));
    for (auto& x : Bb) x = f2bf(dist(rng));
    std::vector<float> D;
    const double eb = runCheck(mmaBf16, Ab, Bb, bf2f, D);
    printf("bf16 16x16x16 mapping check: maxErr = %.3g %s\n", eb, eb < 1e-4 ? "OK" : "MISMATCH");

    std::vector<unsigned char> A8(256), B8(256);
    for (auto& x : A8) x = f2f8(dist(rng) * 4.f);
    for (auto& x : B8) x = f2f8(dist(rng) * 4.f);
    const double e8 = runCheck(mmaFp8, A8, B8, f82f, D);
    printf("fp8 e4m3 16x16x16 mapping check: maxErr = %.3g %s\n", e8, e8 < 1e-3 ? "OK" : "MISMATCH");

    if (eb >= 1e-4) {
        // one-hot probe: A = e_{m,k}, B = ones -> D row m is all 1 iff the mapping puts (m,k) where we think
        for (int m = 0; m < 16; m += 5)
            for (int k = 0; k < 16; k += 3) {
                std::vector<unsigned short> a(256, 0), b(256, f2bf(1.f));
                a[m * 16 + k] = f2bf(1.f);
                runCheck(mmaBf16, a, b, bf2f, D);
                printf("A one-hot (m=%d,k=%d): nonzero D rows:", m, k);
                for (int r = 0; r < 16; ++r) if (D[r * 16] != 0.f) printf(" %d", r);
                printf("\n");
            }
    }

    // throughput
    hipDeviceProp_t prop;
    CHECK(hipGetDeviceProperties(&prop, 0));
    const int blocks = prop.multiProcessorCount * 16, threads = 256;   // 8 waves per block
    float* out;
    CHECK(hipMalloc(&out, (size_t)blocks * threads * sizeof(float)));
    constexpr int ITERS = 4096;
    auto sweep = [&](auto chainsTag) {
        constexpr int CHAINS = decltype(chainsTag)::value;
        const double flops = 2.0 * 16 * 16 * 16 * (double)CHAINS * ITERS * blocks * (threads / 32);
        const double tb = timeKernel([&] { tputBf16<CHAINS, ITERS><<<blocks, threads>>>(out, 3); }, 20);
        const double t8 = timeKernel([&] { tputFp8<CHAINS, ITERS><<<blocks, threads>>>(out, 3); }, 20);
        printf("%s, %d CUs reported, %d chains/wave: bf16 %.1f TFLOPS, fp8 %.1f TFLOPS (%.2fx)\n", prop.name,
               prop.multiProcessorCount, CHAINS, flops / (tb * 1e-3) / 1e12, flops / (t8 * 1e-3) / 1e12, tb / t8);
    };
    sweep(std::integral_constant<int, 1>{});
    sweep(std::integral_constant<int, 2>{});
    sweep(std::integral_constant<int, 4>{});
    sweep(std::integral_constant<int, 8>{});
    sweep(std::integral_constant<int, 16>{});
    CHECK(hipFree(out));
    return 0;
}
