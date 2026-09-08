---
name: fkl-build-and-test
description: >-
  Build and test FKL on CPU and CUDA using CMake and CTest. Use when adding
  Operation or DPP tests, validating a fused pipeline, or debugging template
  compilation. Covers test discovery, supported toolchains, targeted tests,
  and full-suite validation.
---

# Building and testing FKL

## Build Options & Requirements
- Requires CMake >= 3.28 and a C++20 host compiler. CUDA tests require nvcc;
  without it, CMake configures CPU-only. CPU-only success does not validate CUDA.
- **Only nvcc is supported as the CUDA compiler**; clang is a supported host
  compiler, not a replacement CUDA compiler.
- Options: `ENABLE_CUDA` (ON if found), `ENABLE_CPU` (ON), `BUILD_TEST` (ON),
  `BUILD_UTEST` (ON), `ENABLE_BENCHMARK` (OFF).
- Run commands from the existing repository root; do not clone another copy.
  Ninja binaries land in `build/bin/` — one executable per test header,
  suffixed `_cu` (CUDA) / `_cpp` (CPU).

## Build (Linux & WSL2)

```bash
# By default using native CUDA architecture
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release -S .
cmake --build build --config Release
```

For explicit CPU-only validation, configure a separate directory with
`-DENABLE_CUDA=OFF -DENABLE_CPU=ON`. `CUDA_ARCH` is passed to CMake's
`CUDA_ARCHITECTURES`; select an architecture supported by the installed nvcc
and target GPU rather than assuming architecture filtering exists.

## Build (Windows)
```powershell
# IMPORTANT: You MUST activate the VS Developer Shell before running CMake (Enter-VsDevShell)
# Set CUDACXX to the installed nvcc.exe, as in the Windows workflow.
# By default using native CUDA architecture
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release -S .
cmake --build build --config Release
```

If Ninja generates an incorrect nvcc path, inspect `build/CMakeFiles/rules.ninja`
and follow the matching workaround in `.github/workflows/cmake-windows-amd64.yml`;
do not blindly patch a correctly generated file.

## Targeted validation first

```bash
cmake --build build --target utest_color_conversion_cpp
ctest --test-dir build -R '^utest_color_conversion_cpp$' --output-on-failure
```

Replace the target with the changed test header's basename plus `_cpp` or `_cu`.
For a DPP example use `test_linear_filter_dpp_cpp` / `test_linear_filter_dpp_cu`.
Use `ctest --test-dir build -N` to inspect discovery and `-R '_cpp$'` for CPU tests.
Do not invent a second test harness to work around a missing target.

## Running the suite (Cross-Platform)

```bash
cmake --build build --config Release
ctest --test-dir build --build-config Release --output-on-failure --output-junit test_results.xml
```
After targeted tests pass and implementation changes are complete, build and run
the full enabled suite: header changes can break distant instantiations. Report
which backends ran and any unavailable toolchain/runtime. Documentation-only
changes need link/frontmatter/example review, not a full kernel rebuild.

## Test tree layout

```text
utests/ # unit tests (TestCaseBuilder pattern)
  algorithm/image_processing/utest_color_conversion.h ...
  core/...
tests/ # larger example-style tests
benchmarks/ # fusion benchmarks (ENABLE_BENCHMARK=ON)
```
Tests are not written with a traditional framework. CMake calls `discover_tests()`
on subdirectories of `tests/` and `utests/`, recursively finds `*.h`, excludes
paths containing `_common`, and generates `launcher.cpp`/`launcher.cu` stubs.
Place tests in a subdirectory, not beside top-level harness headers.
Any occurrence of `ONLY_CU` or `ONLY_CPU` suppresses the other backend's target;
the usual spelling is `#define __ONLY_CU__` / `#define __ONLY_CPU__`.

## Writing a utest (Agent Instructions)
- Use existing TestCaseBuilder overloads where they support the Operation category.
  Otherwise follow a nearby pipeline/DPP test with a standalone `launch()`.
- Do not add Google Test/Catch2 or assume every operation has a params overload.
- Every test header must define a `launch()` function returning `int` (0 = pass, non-zero = fail).

```cpp
// Example Unary test header under utests/algorithm/basic_ops/
#include <tests/main.h>

#include <fused_kernel/algorithms/basic_ops/cast.h>
#include <tests/operation_test_utils.h>

void testCast() {
    std::array<float, 2> inputVals{2.f, 3.f};
    std::array<int, 2> expectedVals{2, 3};
    TestCaseBuilder<fk::Cast<float, int>>::addTest(testCases, inputVals, expectedVals);
}

START_ADDING_TESTS
testCast();
STOP_ADDING_TESTS

int launch() { RUN_ALL_TESTS }
```

- `TestCaseBuilder` instantiates the op, runs exec on every input and compares against expected,
  printing `Running test for fk::...: Success!!`.
- For Binary Ops, explicitly build the IOp with params and execute a read →
  compute → write pipeline; the two-array builder above is Unary-only.
- Test EVERY public alias and `build()` overload: template code only breaks on instantiation
  (the ColorConversion alias bug shipped because no test instantiated `COLOR_BGR2GRAY` — see issue #244).

## CI Matrix

GitHub workflows build on `linux-amd64`, `linux-arm64` and `windows-amd64` using self-hosted runners.
- Linux builds against `g++-13` and `clang++-21`.
- Windows builds against MSVC (`cl` versions 14.44 and 14.51) and `clang-cl`.
- Check the current `.github/workflows/cmake-*.yml` files for exact host/CUDA
  versions rather than freezing a CUDA version in a skill. CUDA compilation uses
  nvcc even when clang is the host compiler.

For test requirements specific to new abstractions, see
[implementing operations](../fkl-implementing-operations/SKILL.md) and
[implementing DPPs](../fkl-implementing-data-parallel-patterns/SKILL.md).

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
