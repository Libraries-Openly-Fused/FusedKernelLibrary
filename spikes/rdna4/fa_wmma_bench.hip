// Build and run from the repository root (not part of the CMake build):
//   $HIP_PATH/bin/amdclang++ -x hip --offload-arch=gfx1200 -O3 -std=c++20 -ftemplate-depth=1000 -I. -Iinclude \
//       spikes/rdna4/fa_wmma_bench.hip -o fa_wmma_bench && ./fa_wmma_bench
// Spike benchmark: FlashAttentionWmmaDPP (bf16 and fp8 QK^T) vs FlashAttentionDPP (scalar) on HIP.
// Median of 20 timed runs after one warm-up; TFLOPS = 4*bh*sq*sk*d (halved for causal).
#include <fused_kernel/algorithms/attention/flash_attention_wmma.h>

#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

using namespace fk;

template <typename F>
static float medianMs(F&& launch, Stream& stream) {
    hipEvent_t s, e;
    gpuErrchk(hipEventCreate(&s));
    gpuErrchk(hipEventCreate(&e));
    launch();
    stream.sync();
    std::vector<float> ms;
    for (int r = 0; r < 20; ++r) {
        gpuErrchk(hipEventRecord(s, stream.getHIPStream()));
        launch();
        gpuErrchk(hipEventRecord(e, stream.getHIPStream()));
        gpuErrchk(hipEventSynchronize(e));
        float t;
        gpuErrchk(hipEventElapsedTime(&t, s, e));
        ms.push_back(t);
    }
    std::sort(ms.begin(), ms.end());
    return ms[ms.size() / 2];
}

template <typename T>
static T* deviceCopy(const std::vector<T>& h) {
    T* d;
    gpuErrchk(hipMalloc(&d, h.size() * sizeof(T)));
    gpuErrchk(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice));
    return d;
}

template <int D>
static void shape(const int bh, const int s, const bool causal) {
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    const size_t n = (size_t)bh * s * D;
    std::vector<float> hf(n);
    std::vector<attn_bf16> hb(n);
    for (size_t i = 0; i < n; ++i) { hf[i] = dist(rng); hb[i] = __float2bfloat16(hf[i]); }
    std::vector<int8_t> k8(n);
    std::vector<float> ks((size_t)bh * s);
    quantizeKVCacheFp8Host(hf.data(), k8.data(), ks.data(), bh * s, D);
    attn_bf16* qb = deviceCopy(hb);
    attn_bf16* kb = deviceCopy(hb);
    attn_bf16* vb = deviceCopy(hb);
    int8_t* k8d = deviceCopy(k8);
    int8_t* v8d = deviceCopy(k8);
    float* ksd = deviceCopy(ks);
    float* o;
    gpuErrchk(hipMalloc(&o, n * sizeof(float)));
    Stream stream;
    const auto qI = makeAttentionRead(static_cast<const attn_bf16*>(qb), bh, s, D);
    const auto kI = makeAttentionRead(static_cast<const attn_bf16*>(kb), bh, s, D);
    const auto vI = makeAttentionRead(static_cast<const attn_bf16*>(vb), bh, s, D);
    const auto k8I = makeFp8KVRead(k8d, ksd, bh, s, D);
    const auto v8I = makeFp8KVRead(v8d, ksd, bh, s, D);
    const double flops = 4.0 * bh * s * (double)s * D * (causal ? 0.5 : 1.0);
    const float tScalar = medianMs([&] { executeFlashAttention<D>(qI, kI, vI, o, bh, s, s, causal, stream); }, stream);
    const float t32w4 = medianMs([&] { executeFlashAttentionWmma<D, 32, 4>(qI, kI, vI, o, bh, s, s, causal, stream); }, stream);
    const float t64w4 = medianMs([&] { executeFlashAttentionWmma<D, 64, 4>(qI, kI, vI, o, bh, s, s, causal, stream); }, stream);
    const float t32w8 = medianMs([&] { executeFlashAttentionWmma<D, 32, 8>(qI, kI, vI, o, bh, s, s, causal, stream); }, stream);
    const float t64w8 = medianMs([&] { executeFlashAttentionWmma<D, 64, 8>(qI, kI, vI, o, bh, s, s, causal, stream); }, stream);
    const float tF4 = medianMs([&] { executeFlashAttentionWmma<D, 64, 4>(qI, k8I, v8I, o, bh, s, s, causal, stream); }, stream);
    const float tF8 = medianMs([&] { executeFlashAttentionWmma<D, 64, 8>(qI, k8I, v8I, o, bh, s, s, causal, stream); }, stream);
    const auto tf = [&](float ms) { return flops / (ms * 1e-3) / 1e12; };
    const float best = std::min(std::min(t32w4, t64w4), std::min(t32w8, t64w8));
    printf("d%-3d bh%-3d s%-5d %-6s | scalar %6.2f TF | bf16 BKV32/W4 %5.2f BKV64/W4 %5.2f BKV32/W8 %5.2f BKV64/W8 %5.2f TF"
           " | fp8-QK W4 %5.2f W8 %5.2f TF | best bf16 %7.3f ms = %5.1fx scalar\n",
           D, bh, s, causal ? "causal" : "dense", tf(tScalar), tf(t32w4), tf(t64w4), tf(t32w8), tf(t64w8), tf(tF4),
           tf(tF8), best, tScalar / best);
    for (void* p : { (void*)qb, (void*)kb, (void*)vb, (void*)k8d, (void*)v8d, (void*)ksd, (void*)o }) gpuErrchk(hipFree(p));
}

int main() {
    for (const bool causal : { false, true }) {
        shape<64>(16, 1024, causal);
        shape<64>(16, 4096, causal);
        shape<128>(16, 1024, causal);
        shape<128>(8, 4096, causal);
    }
    return 0;
}
