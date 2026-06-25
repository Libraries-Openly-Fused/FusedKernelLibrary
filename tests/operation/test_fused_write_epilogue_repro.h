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

/* Regression test for the compute->write fused-IOp path (PR #287, review
 * r3473806517 / fix r3473994080).
 *
 * Chaining a COMPUTE op onto a Write IOp — the `epilogue.then(D)` shape DPP
 * epilogues use — must both COMPOSE into a write-type IOp and INSTANTIATE its
 * exec(). This previously failed to compile in the WriteType (and ClosedType)
 * FusedOperation_ specialisations:
 *
 *     fused_operation.h: LastType_t<typename ParamsType::Operations>::Operation
 *     -> error: name followed by "::" must be a class or namespace name
 *
 * Root cause: `LastType_t` takes a parameter PACK, but those two specialisations
 * passed `ParamsType::Operations` (a single TypeList) instead of the `IOps...`
 * pack, so LastType_t collapsed to the TypeList itself (which has no
 * ::Operation). Fixed by using the pack form `LastType_t<IOps...>` (matching the
 * Unary specialisation). This test exercises both:
 *   - identity epilogue:  Cast<float,float>().then(D)
 *   - scale+bias chain:    Mul(2).then(Add(0.5)).then(D)
 * and checks the stored result against a CPU oracle. */

#define __ONLY_CU__
#include <tests/main.h>

#include <fused_kernel/algorithms/basic_ops/basic_ops.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/basic_ops/cast.h>
#include <fused_kernel/core/data/ptr_nd.h>

#include <cstdio>
#include <vector>
#include <cmath>

using namespace fk;

// Invoke the fused compute->write IOp the way a DPP epilogue would: pass the
// whole fused IOp (epilogue.then(D)) and call its Operation::exec(thread, value, iop).
template <typename FusedW>
__global__ void epilogueWriteKernel(FusedW out, int W, float val) {
    const int x = threadIdx.x;
    if (x < W) FusedW::Operation::exec(Point{ x, 0, 0 }, val, out);
}

template <typename Epi>
static bool runCase(const char* name, const Epi& epilogue, float in, float expected) {
    constexpr int W = 8;
    Ptr2D<float> d(W, 1);
    const auto output = epilogue.then(PerThreadWrite<ND::_2D, float>::build(d));
    static_assert(isAnyWriteType<decltype(output)>,
                  "epilogue.then(D) must compose into a write-type IOp");

    Stream_<ParArch::GPU_NVIDIA> stream;
    epilogueWriteKernel<<<1, W, 0, stream.getCUDAStream()>>>(output, W, in);
    gpuErrchk(cudaGetLastError());
    stream.sync();

    std::vector<float> h(W);
    cudaMemcpy2D(h.data(), W * sizeof(float), d.ptr().data, d.ptr().dims.pitch,
                 W * sizeof(float), 1, cudaMemcpyDeviceToHost);
    for (int i = 0; i < W; ++i)
        if (std::abs(h[i] - expected) > 1e-5f) {
            printf("FAIL %s: out[%d]=%.4f expected %.4f\n", name, i, h[i], expected);
            return false;
        }
    printf("fused epilogue.then(D) %-12s: PASS (in=%.1f -> %.4f)\n", name, in, expected);
    return true;
}

int launch() {
    bool ok = true;
    // identity epilogue: Cast<float,float> pass-through, fused with the write.
    ok &= runCase("identity",   Cast<float, float>::build(),                              10.0f, 10.0f);
    // scale+bias IOp chain: (v*2) then (+0.5), fused with the write.
    ok &= runCase("scale+bias", Mul<float>::build(2.f).then(Add<float>::build(0.5f)),     10.0f, 20.5f);
    return ok ? 0 : -1;
}
