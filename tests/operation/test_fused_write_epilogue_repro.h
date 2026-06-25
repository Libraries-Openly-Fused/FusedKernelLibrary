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

/* Reproducer for the compute->write fused-IOp instantiation bug discussed in
 * PR #287 (review r3473806517) and reported at r3473913994.
 *
 * Building a fused write IOp by chaining a COMPUTE op onto a Write IOp — the
 * `epilogue.then(D)` shape @morousg asked DPP epilogues to use — fails to
 * instantiate. Both of these:
 *
 *     auto w = Cast<float,float>::build().then(PerThreadWrite<ND::_2D,float>::build(d));
 *     auto w = fuse(Add<float>::build(5.f), PerThreadWrite<ND::_2D,float>::build(d));
 *
 * trip, in the WriteType FusedOperation_ specialisation
 * (include/.../operation_model/fused_operation.h):
 *
 *     error: name followed by "::" must be a class or namespace name
 *         LastType_t<typename ParamsType::Operations>::Operation::exec(...)
 *
 * `ParamsType` is `OperationTuple<IOps...>`; the line reaches for
 * `ParamsType::Operations` instead of the enclosing FusedOperation_'s own
 * `using Operations = TypeList<IOps...>`. Composition succeeds
 * (isAnyWriteType<decltype(w)> == true); only instantiating its exec() breaks.
 *
 * This test is COMPILE-CLEAN by default so CI stays green: the offending
 * instantiation is guarded behind FKL_REPRO_287. To reproduce the failure,
 * configure/build this single target with -DFKL_REPRO_287, e.g.:
 *
 *     nvcc -std=c++17 -I include -DFKL_REPRO_287 \
 *          tests/operation/test_fused_write_epilogue_repro.h ...
 *
 * When the core is fixed, drop the guard (make the fused-write path the
 * default) and this becomes a positive regression test: epilogue.then(D)
 * applied to a value then stored, checked against a CPU oracle. */

#define __ONLY_CU__
#include <tests/main.h>

#include <fused_kernel/algorithms/basic_ops/basic_ops.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/basic_ops/cast.h>
#include <fused_kernel/core/data/ptr_nd.h>

#include <cstdio>
#include <vector>

using namespace fk;

#if defined(FKL_REPRO_287)
// Invoke the fused compute->write IOp the way a DPP epilogue would. With the
// current core this fails to compile at fused_operation.h:200 when the kernel
// (and thus exec_helper) is instantiated.
template <typename FusedW>
__global__ void reproKernel(FusedW out, int W) {
    const int x = threadIdx.x;
    if (x < W) FusedW::Operation::exec(Point{ x, 0, 0 }, 10.0f, out);
}
#endif

int launch() {
    Ptr2D<float> d(8, 1);

    // Composition itself is fine: the result IS a write-type IOp.
    const auto epilogueThenD =
        Cast<float, float>::build().then(PerThreadWrite<ND::_2D, float>::build(d));
    static_assert(isAnyWriteType<decltype(epilogueThenD)>,
                  "epilogue.then(D) should compose into a write-type IOp");

#if defined(FKL_REPRO_287)
    // --- bug path (only with -DFKL_REPRO_287) -----------------------------
    // Instantiating the fused write's exec() trips fused_operation.h:200.
    Stream_<ParArch::GPU_NVIDIA> stream;
    reproKernel<<<1, 8, 0, stream.getCUDAStream()>>>(epilogueThenD, 8);
    gpuErrchk(cudaGetLastError());
    stream.sync();
    std::vector<float> h(8);
    cudaMemcpy2D(h.data(), 8 * sizeof(float), d.ptr().data, d.ptr().dims.pitch,
                 8 * sizeof(float), 1, cudaMemcpyDeviceToHost);
    // identity epilogue: stored value should equal the input (10.0).
    for (int i = 0; i < 8; ++i)
        if (h[i] != 10.0f) { printf("FAIL repro287: out[%d]=%.1f expected 10\n", i, h[i]); return -1; }
    printf("fused epilogue.then(D) write: PASS (core bug fixed)\n");
    return 0;
#else
    // Default: CI-green. Composition compiled; the exec() instantiation that
    // triggers the bug is gated out. Build with -DFKL_REPRO_287 to reproduce.
    (void)epilogueThenD;
    printf("test_fused_write_epilogue_repro: composition OK; "
           "exec() instantiation gated behind FKL_REPRO_287 (see #287)\n");
    return 0;
#endif
}
