---
name: fkl-build-and-test
description: >-
  Build and test FKL using CMake and CTest on CPU and CUDA. Use when adding
  Operation or DPP tests, validating fused pipelines, or debugging template
  compilation. Covers discovery, targeted tests, and the CI toolchain.
---

# Building and testing FKL

## Build options and requirements

- CMake >= 3.28 and a C++20 host compiler are required.
- Use **nvcc** for CUDA. Clang is a supported host compiler, not a supported
  substitute for nvcc's CUDA compilation path.
- CUDA is part of project validation. CMake nevertheless supports a CPU-only
  fallback when nvcc is absent; passing CPU tests alone does not validate CUDA.
- `ENABLE_CPU` defaults to ON; `ENABLE_CUDA` defaults to ON when CUDA is found.
  `BUILD_TEST` and `BUILD_UTEST` default to ON; `ENABLE_BENCHMARK` defaults to OFF.
- `CUDA_ARCH` defaults to `native` and is passed to `CUDA_ARCHITECTURES`.
  For cross-compilation, choose explicit architectures supported by the toolkit.

## Build (Linux & WSL2)

```bash
# From the existing repository root:
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release -S .
cmake --build build --config Release
```

Set `CXX` and `CUDACXX` before configuring a fresh build directory to select
the host compiler and nvcc. Do not diagnose CUDA availability from an ad hoc
compiler invocation before checking the repository's CMake configuration.

With Ninja, binaries are in `build/bin/`; multi-configuration generators use
`build/bin/<config>/`. Test targets end in `_cpp` or `_cu`.

## Build (Windows)
```powershell
# From the repository root, with the VS Developer Shell already activated:
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release -S .
cmake --build build --config Release
```

Activate `Enter-VsDevShell` for both configure and build, and set `CUDACXX` to
nvcc. If Ninja generates an incorrect nvcc path, use the current workaround in
`.github/workflows/cmake-windows-amd64.yml`; inspect `build/CMakeFiles/rules.ninja`
before applying it.

## Targeted validation, then the suite

```bash
cmake --build build --target test_crop_cpp
ctest --test-dir build -C Release -R '^test_crop_cpp$' --output-on-failure
cmake --build build --target test_crop_cu
ctest --test-dir build -C Release -R '^test_crop_cu$' --output-on-failure

cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure --output-junit test_results.xml
```
Choose the test matching the changed area; `test_crop` is an existing example.
The CUDA target exists only when that backend is enabled. Use `ctest --test-dir
build -N` to inspect discovery. Targeted checks catch iteration errors; the full
suite is the merge gate for implementation changes because distant template
instantiations can fail. Documentation-only edits do not need a C++ build.

## Test tree layout

```text
utests/ # unit tests (TestCaseBuilder pattern)
  algorithm/image_processing/utest_color_conversion.h ...
  core/...
tests/ # larger example-style tests
benchmarks/ # fusion benchmarks (ENABLE_BENCHMARK=ON)
```
`tests/CMakeLists.txt` and `utests/CMakeLists.txt` call `discover_tests()` on
their immediate subdirectories. It recursively finds `.h` files, excludes
paths containing `_common`, and generates launcher translation units from
`tests/launcher.in`. Add tests inside a subdirectory, not at the top level.

Any occurrence of `ONLY_CU` suppresses the CPU target; `ONLY_CPU` suppresses the
CUDA target. These are substring checks, including comments. Follow the nearby
test's convention and do not accidentally disable a backend in explanatory text.

## Writing a utest (Agent Instructions)
- Use the existing `TestCaseBuilder` harness for Operation types it supports.
  For other Operations or DPPs, follow an existing direct pipeline/buffer test;
  do not add Google Test or Catch2.
- Every test header must define a `launch()` function returning `int` (0 = pass, non-zero = fail).

```cpp
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
- Check available builder specializations in `tests/operation_test_utils.h`
  rather than assuming an arbitrary params overload exists.
- Test EVERY public alias and `build()` overload: template code only breaks on instantiation
  (the ColorConversion alias bug shipped because no test instantiated `COLOR_BGR2GRAY` — see issue #244).

## CI Matrix

GitHub workflows build on `linux-amd64`, `linux-arm64` and `windows-amd64` using self-hosted runners.
- Linux builds against `g++-13` and `clang++-21`.
- Windows builds against MSVC (`cl` versions 14.44 and 14.51) and `clang-cl`.
- Check current workflow files for toolkit/compiler versions rather than
  hard-coding a matrix into new tests. CUDA compilation remains nvcc-based.

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
4. Isolate the failing instantiation in the nearest test and include its
   operation headers explicitly. The main API header is not an umbrella for
   every algorithm. For DPP calls, verify the whole-IOp invocation form.
