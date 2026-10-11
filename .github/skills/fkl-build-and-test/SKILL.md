---
name: fkl-build-and-test
description: Build the FusedKernelLibrary tests and run them. Covers cloning with the thread-pool submodule
(ParArch::CPU_MULTITHREAD), CMake configuration for Linux and Windows using out-of-source builds, the utests/tests structure, TestCaseBuilder, adding tests for
new operations, running the full binary suite with ctest, and what CI expects. Use when building FKL tests,
running or adding tests, or debugging a compile failure in the test tree on any OS.
---

# Building and testing FKL

## Build Options & Requirements
- Requires CMake >= 3.28, a C++20 host compiler, and CUDA.
- **Only nvcc is supported as the CUDA compiler**; clang-as-CUDA-compiler is not supported despite the
  `CLANG_HOST_DEVICE` macro.
- Options: `ENABLE_CUDA` (default ON if found), `ENABLE_CPU` (ON), `ENABLE_CPU_MULTITHREAD` (see below),
  `ENABLE_BENCHMARK` (OFF).
- Binaries land in the out-of-source build folder at `../build` — one executable per test unit,
  suffixed `_cu` (CUDA) / `_cpp` (CPU).

## The thread-pool submodule (ParArch::CPU_MULTITHREAD)
- The multithreaded CPU backend, `ParArch::CPU_MULTITHREAD`, uses the `thread-pool` git submodule. ALWAYS clone
  with `--recursive`, or run `git submodule update --init` in an existing clone.
- `ENABLE_CPU_MULTITHREAD` defaults to ON when `thread-pool/include/ThreadPool.h` exists and to OFF otherwise. Forcing
  it ON without the submodule is a `FATAL_ERROR`. When ON, the FKL target defines `FK_CPU_MULTITHREAD_ENABLED` (every
  test target gets it), adds `thread-pool/include` and links `Threads::Threads`.
- Without the submodule the backend is **silently off**: CMake only prints a STATUS line, the CPU_MULTITHREAD code of
  the tests is compiled out, and `utest_cpu_multithread_cpp` passes while printing that it tested nothing. Check the
  configure output for `CPU_MULTITHREAD backend (ENABLE_CPU_MULTITHREAD): ON`.
- CI checks out submodules (`actions/checkout` with `submodules: true`).

## Build (Linux & WSL2)

```bash
mkdir build
git clone --recursive https://github.com/Libraries-Openly-Fused/FusedKernelLibrary.git
cd FusedKernelLibrary
# By default using native CUDA architecture
cmake -G "Ninja" -B ../build -DCMAKE_BUILD_TYPE=Release -S .
cmake --build ../build --config Release
```

- Requires CMake >= 3.28, a C++ host compiler, and CUDA.
- Options: `ENABLE_CUDA` (default ON if found), `ENABLE_CPU` (ON), `ENABLE_CPU_MULTITHREAD` (ON if the
  thread-pool submodule is checked out), `ENABLE_BENCHMARK` (OFF) for the benchmark targets.
- Binaries land in `../build` — one executable per test unit,
  suffixed `_cu` (CUDA) / `_cpp` (CPU).

## Build (Windows)
```powershell
mkdir build
git clone --recursive https://github.com/Libraries-Openly-Fused/FusedKernelLibrary.git
cd FusedKernelLibrary

# IMPORTANT: You MUST activate the VS Developer Shell before running CMake (Enter-VsDevShell)
# By default using native CUDA architecture
cmake -G "Ninja" -B ..\build -DCMAKE_BUILD_TYPE=Release -S .

# WORKAROUND: The Ninja generator may generate an empty nvcc path in rules.ninja. Patch it before building:
(Get-Content ..\build\CMakeFiles\rules.ninja) -replace "\\nvcc\\bin\\nvcc.exe", "$env:CUDACXX" | Set-Content ..\build\CMakeFiles\rules.ninja

cmake --build ..\build --config Release
```

## Running the suite (Cross-Platform)

```bash
cd ../build
ctest --build-config Release --output-junit test_results.xml
```
All tests must pass. The full suite is the merge gate: a header change that compiles can still break a distant instantiation, so ALWAYS run everything using `ctest` (66+ binaries, a few minutes). Tests are automatically registered with CTest.

## Test tree layout

```text
utests/ # unit tests (TestCaseBuilder pattern)
  algorithm/image_processing/utest_color_conversion.h ...
  core/...
tests/ # larger example-style tests
benchmarks/ # fusion benchmarks (ENABLE_BENCHMARK=ON)
```
Tests are not written with a traditional framework. CMake auto-discovers test headers (`*.h`) via `discover_tests()` (excluding files matching `*_common.*`) and generates a `launcher.cpp`/`launcher.cu` stub. A new `utest_*.h` in an existing directory is picked up without CMake edits.

## Writing a utest (Agent Instructions)
- ALWAYS use the TestCaseBuilder pattern when I ask you to write a test.
- DO NOT write standard Google Test/Catch2 macros; instantiate the op and pass it through TestCaseBuilder::addTest.
- Every test header must define a `launch()` function returning `int` (0 = pass, non-zero = fail).

```cpp
// utests/algorithm/basic_ops/utest_myop.h
#include <tests/main.h>

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

- `TestCaseBuilder` instantiates the op, runs exec on every input and compares against expected,
  printing `Running test for fk::...: Success!!`.
- For ops with params, pass them through the builder's params overloads.
- Test EVERY public alias and `build()` overload: template code only breaks on instantiation
  (the ColorConversion alias bug shipped because no test instantiated `COLOR_BGR2GRAY` — see issue #244).

## Testing ParArch::CPU_MULTITHREAD code
- Extend the EXISTING test of the DPP family under `#ifdef FK_CPU_MULTITHREAD_ENABLED`: run every case also through
  the `Stream_<ParArch::CPU_MULTITHREAD>&` entry point, `stream.sync()`, and require the output to match the
  `ParArch::CPU` output, plus the existing oracle checks: exactly (`memcmp` or `==`) for integer outputs and float
  pipelines without multiply-add; within a tight tolerance (e.g. 1e-5 relative) for float pipelines that multiply and
  add, since g++ with FMA (`-march=native`, arm64) may contract `a * b + c` differently in the two versions.
- Use host memory (`MemType::Host`): the `_cu` target compiles the same header, and there `defaultMemType` is
  `DeviceAndPinned`. The CPU_MULTITHREAD code must compile under nvcc too.
- A check that means "GPU" must be `PA == ParArch::GPU_NVIDIA || PA == ParArch::GPU_AMD`, not `PA != ParArch::CPU`.
- The backend itself (stream order, sync, errors, buffer lifetime) is tested by
  `utests/core/execution_model/utest_cpu_multithread.h`, which is `_cpp` only.
- Also check that the code compiles with the backend OFF (`-DENABLE_CPU_MULTITHREAD=OFF`), and run the affected
  `_cpp` tests under ThreadSanitizer. There is no CMake option for sanitizers; compile the generated launcher directly:
  `g++ -O1 -g -fsanitize=thread -std=c++20 -ftemplate-depth=1000 -DFK_CPU_MULTITHREAD_ENABLED -I<repo> -I<repo>/include -I<repo>/thread-pool/include <build>/{tests,utests}/<name>_cpp/launcher.cpp <repo>/tests/main.cpp -pthread`.

## CI Matrix

GitHub workflows build on `linux-amd64`, `linux-arm64` and `windows-amd64` using self-hosted runners.
- Every workflow checks out submodules, so `ParArch::CPU_MULTITHREAD` is built and tested in CI.
- Linux builds against `g++-13` and `clang++-23`.
- Windows builds against MSVC (`cl` versions 14.44 and 14.51) and `clang-cl`.
- Keep changes warning-clean on BOTH compilers: clang is a first-class compiler for FKL (single-step host+device compiles matter for downstream packaging).

## How to debug compile failures
Follow these steps when you get compilation errors when iterating your work:
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
