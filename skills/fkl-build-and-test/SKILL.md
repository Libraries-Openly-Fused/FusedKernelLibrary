---
name: fkl-build-and-test
description: Build the FusedKernelLibrary repository and run its unit tests — CMake configuration, the utests/tests structure, TestCaseBuilder, adding tests for new operations, running the full binary suite, and what CI expects. Use when building FKL from source, running or adding tests, or debugging a compile failure in the test tree.
---

# Building and testing FKL

## Build (verified commands)

```bash
git clone --recursive https://github.com/Libraries-Openly-Fused/FusedKernelLibrary.git
cd FusedKernelLibrary            # LTS-C++17 is the active branch; main is frozen
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90 \
         # 120 for Blackwell, 90 Hopper, 86 Ampere...
make -j$(nproc)
```

- Requires CMake >= 3.28, a C++ host compiler, and (optionally) CUDA
  and/or HIP (ROCm, amdclang++: `-DCMAKE_HIP_COMPILER=/opt/rocm/bin/amdclang++`).
- Options: `ENABLE_CUDA` (default ON if found), `ENABLE_HIP` (default ON if
  found), `ENABLE_CPU` (ON), `ENABLE_BENCHMARK` (OFF) for the benchmark targets.
  `ENABLE_CPU_MULTITHREAD` (ON when the `thread-pool` submodule is checked out:
  `git submodule update --init`) builds the multithreaded CPU backend,
  `ParArch::CPU_MULTITHREAD`, and its `utest_cpu_multithread*` (`_cpp` only).
  HIP architectures: the local GPU by default, or `-DHIP_ARCH=gfx1200` on the
  first configure of a build directory (`-DCMAKE_HIP_ARCHITECTURES` later).
- Without CUDA and HIP the CPU backend still builds and tests run on CPU.
- Binaries land in `build/bin/Release/` — one executable per test unit,
  suffixed `_cu` (CUDA) / `_hip` (HIP) / `_cpp` (CPU). Test sources marked
  `__ONLY_CU__` / `__ONLY_CPU__` / `__ONLY_GPU__` (CUDA and HIP) skip the other
  backends.
- Under HIP the global `float3`... are HIP's types: tests use `fk::float3`...

## Running the suite

```bash
cd build/bin/Release
shopt -s nullglob  # a backend that is not built has no binaries
for t in *_cu *_hip *_cpp; do
  case $t in benchmark_*) continue;; esac  # ENABLE_BENCHMARK builds: the benchmarks take far longer
  ./$t >/dev/null 2>&1 && echo "$t OK" || echo "$t FAIL"
done
```
All binaries exit 0 on success. The full suite is the merge gate: a
header change that compiles can still break a distant instantiation, so
ALWAYS run everything (66+ binaries, a few minutes).

## Test tree layout

```
utests/                    # unit tests (TestCaseBuilder pattern)
  algorithm/image_processing/utest_color_conversion.h ...
  core/...
tests/                     # larger example-style tests
benchmarks/                # fusion benchmarks (ENABLE_BENCHMARK=ON)
```
CMake auto-discovers test headers via the SUBDIRLIST macros; a new
`utest_*.h` in an existing directory is picked up without CMake edits.

## Writing a utest (TestCaseBuilder pattern)

```cpp
// utests/algorithm/basic_ops/utest_myop.h
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <tests/operation_test_utils.h>

void testMyOp() {
    std::array<float, 2> inputVals{2.f, 3.f};
    std::array<float, 2> expectedVals{4.f, 6.f};
    TestCaseBuilder<fk::MyOp<float>>::addTest(testCases, inputVals, expectedVals);
}

START_ADDING_TESTS
testMyOp();
STOP_ADDING_TESTS

int launch() { RUN_ALL_TESTS }
```

- TestCaseBuilder instantiates the op, runs exec on every input and
  compares against expected, printing `Running test for fk::...: Success!!`.
- For ops with params, pass them through the builder's params overloads.
- Test EVERY public alias and build() overload: template code only breaks
  on instantiation (the ColorConversion alias bug shipped because no test
  instantiated `COLOR_BGR2GRAY` — see issue #244).

## CI

GitHub workflows build on linux-amd64, linux-arm64 and windows-amd64,
with a matrix over host compilers and CUDA toolkits (on Linux currently
12.9 and 13.4). They build the benchmarks (`-DENABLE_BENCHMARK=ON`) but
leave them out of ctest; no workflow builds HIP. Keep changes
warning-clean on BOTH nvcc and clang: clang is a first-class compiler for
FKL (single-step host+device compiles matter for downstream packaging).

## Debugging compile failures in templates

1. Read nvcc/clang template errors BOTTOM-UP: the last "instantiation of"
   frame names the user-level line; the first error names the real culprit.
2. `qualifiers dropped in binding reference of type 'X&&'` => somebody
   passed explicit template args to a forwarding-reference function
   (see issue #245); let deduction happen.
3. `name followed by "::" must be a class or namespace name` inside
   fused_operation.h => a raw Operation was passed where an IOp was
   expected; wrap with `Unary<...>` / `Binary<...>`.
4. Reduce: extract the failing instantiation into a 20-line main() with
   only `#include <fused_kernel/fused_kernel.h>` + execution_model +
   algorithms headers — it compiles in seconds instead of minutes and
   makes upstream bug reports trivial.
