# Copilot Instructions for FusedKernelLibrary (FKL)

## What This Repository Is

The **Fused Kernel Library (FKL)** is a header-only C++20 library (plus a thin CMake-managed interface target) that provides a methodology for GPU (and CPU) kernel fusion. It supports:
- **Vertical Fusion** – chain operations so they execute in a single kernel pass
- **Horizontal Fusion** – run different data planes in the same kernel using `blockIdx.z`
- **Backwards Vertical Fusion** – read only the data required (e.g., crop before resize, only reading necessary pixels)
- **Divergent Horizontal Fusion** – fuse completely different kernels that share GPU resources

The main entry point at call sites is `fk::executeOperations<DPP>(stream, op1, op2, ...)`.

---

## Repository Layout

```
FusedKernelLibrary/
├── .github/workflows/        # CI: Linux x64, Linux ARM64, Windows x64 (self-hosted runners)
├── benchmarks/               # Optional benchmark targets (off by default)
├── cmake/                    # All CMake helpers
│   ├── cmake_init.cmake      # Global settings (OUT_DIR, config types, TEMPLATE_DEPTH)
│   ├── cuda_init.cmake       # Enables CUDA language; defers to cmake/libs/cuda/
│   ├── archflags.cmake       # CPU SIMD flags (AVX2, -march=native, etc.)
│   ├── doxygen.cmake
│   ├── deploy/               # install/deploy helpers
│   ├── generators/           # version_header.cmake, export_header.cmake
│   ├── libs/cuda/            # CUDA-specific helpers
│   │   ├── cuda.cmake        # find_package(CUDAToolkit), add_cuda_to_target()
│   │   ├── archs.cmake       # CUDA_ARCH, remove_pre70gpus(), set_target_cuda_arch_flags()
│   │   ├── debug.cmake       # ENABLE_DEBUG, ENABLE_NVTX, ENABLE_LINE_INFO
│   │   ├── deploy.cmake      # Windows DLL deployment helper
│   │   └── target_generation.cmake  # set_default_cuda_target_properties()
│   └── tests/
│       ├── discover_tests.cmake     # Auto-discovers *.h test headers; generates .cpp and .cu targets
│       └── add_generated_test.cmake # configure_file + add_executable for each test
├── include/fused_kernel/
│   ├── fused_kernel.h        # Main entry: fk::executeOperations<DPP>(stream, ops...)
│   ├── core/
│   │   ├── data/             # Ptr2D, Tensor, Rect, Size, Point, Tuple, array, vector_types
│   │   ├── execution_model/
│   │   │   ├── operation_model/   # Operation types, InstantiableOperations, FusedOperation
│   │   │   ├── data_parallel_patterns.h  # TransformDPP, BatchTransformDPP, etc.
│   │   │   ├── executors.h        # Executor<DPP> – launches CUDA kernels or CPU loops
│   │   │   ├── stream.h           # fk::Stream / fk::Stream_<PAR_ARCH>
│   │   │   ├── parallel_architectures.h  # ParArch enum (CPU, GPU_NVIDIA, …)
│   │   │   └── thread_fusion.h    # ThreadFusionInfo; packs multiple reads per thread
│   │   ├── constexpr_libs/   # constexpr math
│   │   └── utils/            # compiler_macros.h, type_lists.h, parameter_pack_utils.h, etc.
│   └── algorithms/
│       ├── basic_ops/        # arithmetic.h, cast.h, logical.h, memory_operations.h, set.h, …
│       └── image_processing/ # crop.h, resize.h, color_conversion.h, interpolation.h, …
├── lib/                      # CMake interface target FKL::FKL; no compiled sources (header-only)
├── tests/                    # Integration tests (auto-generated from *.h headers)
│   ├── main.cpp / main.h     # Test harness entry point
│   ├── launcher.in           # configure_file template used by discover_tests
│   └── <category>/<test>.h   # Each .h is compiled as both _cpp and _cu target
└── utests/                   # Unit tests (same auto-generation mechanism)
```

---

## Build System

### Requirements
- **CMake ≥ 3.22**
- **C++20** host compiler (GCC, Clang, MSVC ≥ 2019)
- **CUDA ≥ 12.x** (CUDA 11 is **not** supported; CUDA 13.x is also supported)
- **Ninja** generator is strongly preferred (used in CI)

### Key CMake Options

| Option | Default | Meaning |
|--------|---------|---------|
| `ENABLE_CPU` | `ON` | Build CPU backend targets |
| `ENABLE_CUDA` | `ON` (if nvcc found) | Build CUDA backend targets |
| `BUILD_TEST` | `ON` | Build integration tests in `tests/` |
| `BUILD_UTEST` | `ON` | Build unit tests in `utests/` |
| `ENABLE_BENCHMARK` | `OFF` | Build benchmarks in `benchmarks/` |
| `CUDA_ARCH` | `native` (CMake ≥ 3.24) / `all` | CUDA GPU architectures to target |
| `TEMPLATE_DEPTH` | `1000` | `-ftemplate-depth` (avoids deep-recursion compiler errors) |
| `ENABLE_DEBUG` | `OFF` | Enable CUDA `-G` device debug flag |
| `ENABLE_NVTX` | `OFF` | Enable NVTX profiling annotations |
| `ENABLE_LINE_INFO` | `ON` | Pass `-lineinfo` to nvcc (only when `ENABLE_DEBUG=OFF`) |
| `ARCH_FLAGS` | `native`/`AVX2` | Host CPU SIMD flags |

### Typical Local Configure + Build

**Linux (with CUDA):**
```bash
cmake -G Ninja -B build \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.9 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

**Linux (CPU only, no CUDA):**
```bash
cmake -G Ninja -B build \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

**Run tests:**
```bash
cd build
ctest --build-config Release --output-junit test_results.xml
```

### Known CI Quirks / Workarounds
- The Linux CI workflows use **self-hosted runners** (`runs-on: [self-hosted, linux, x64]` etc.) so they cannot run on standard GitHub-hosted runners.
- On Windows with Ninja, CMake sometimes generates an empty NVCC path in `rules.ninja`; the Windows CI workflow has a PowerShell workaround that patches `rules.ninja` after configure.
- The CI matrix uses **comma-separated strings** for the Linux/ARM64 workflows (e.g., `"g++,nvcc,12.9"`) but the Windows workflow uses a `toolset` variable with a similar comma-separated format.
- The ARM64 workflow passes `-DCUDA_ARCH="87;"` explicitly to target Jetson Orin / Grace-Hopper hardware.
- Only GPU compute capabilities ≥ 7.0 (Volta and newer) are supported; `archs.cmake` automatically filters out older architectures.

---

## Core Architecture Concepts

### Operation Types

Every operation is a **struct** with a static `exec(...)` function and a nested `InstanceType` tag. The complete table:

| InstanceType | `exec` signature | Notes |
|---|---|---|
| `ReadType` | `Out exec(Point, Params)` | Reads from memory at thread index |
| `WriteType` | `void exec(Point, In, Params)` | Writes to memory |
| `UnaryType` | `Out exec(In)` | Pure computation, no params |
| `BinaryType` | `Out exec(In, Params)` | Computation with parameters |
| `ReadBackType` | `Out exec(Point, Params, BackIOp)` | Reads back through a chained IOp |
| `TernaryType` | `Out exec(In, Params, BackIOp)` | Computation with params + back IOp |
| `MidWriteType` | `In exec(Point, In, Params)` | Write with pass-through (In == Out) |
| `OpenType` | `Out exec(Point, In, Params)` | FusedOperation only |
| `ClosedType` | `void exec(Point, Params)` | FusedOperation only |

### InstantiableOperations (IOps)

Every `Operation` struct is wrapped in an **InstantiableOperation** that owns the runtime parameters and exposes a `build(...)` static factory:

```cpp
Read<MyOp>        // for ReadType ops
Write<MyOp>       // for WriteType ops
Unary<MyOp>       // for UnaryType ops
Binary<MyOp>      // for BinaryType ops
ReadBack<MyOp>    // for ReadBackType ops
MidWrite<MyOp>    // for MidWriteType ops
```

- `InputFoldType` is always passed **by value** (no `const` or `&`) in `operator|` and fuser overloads.
- Operations in a `FusedOperation<...>` **must** be wrapped in their InstantiableOperation type.

### FusedOperation

`FusedOperation<IOp1, IOp2, ...>` composes multiple IOps at compile time. The resulting type's `InstanceType` is inferred:
- All `UnaryType` → `UnaryType`
- Starts with `ReadType`/`ReadBackType` + ends with `WriteType` → `ClosedType`
- Contains `MidWriteType` (no read/write terminals) → `OpenType`

### Data Parallel Patterns (DPP)

Passed as the first template parameter to `executeOperations<DPP>(...)`:
- `TransformDPP<>` – most common: one thread per output element
- `BatchTransformDPP<BATCH>` – horizontal fusion over a batch dimension

### Parallel Architectures

`ParArch` enum: `CPU`, `GPU_NVIDIA`, `GPU_NVIDIA_JIT`, `GPU_AMD`, `CPU_OMP`, …

Use `CMAKE_CUDA_COMPILER_ID` (values: `"NVIDIA"`, `"Clang"`) rather than `CMAKE_CUDA_COMPILER` to detect the CUDA compiler type in CMake scripts.

### Stream

`fk::Stream` (alias for `fk::Stream_<ParArch::GPU_NVIDIA>`) wraps `cudaStream_t`. On CPU it is a no-op object.

---

## Testing Conventions

### Test Discovery (Auto-generated)

Tests are **not** `*.cpp` files. Each test is a **`*.h` header** placed inside `tests/<category>/` or `utests/<category>/`. The CMake function `discover_tests()` automatically:
1. Finds all `*.h` files in the directory tree.
2. For each header, generates a `launcher.cpp` / `launcher.cu` from `tests/launcher.in` (which `#include`s the header).
3. Creates two CMake targets: `<TestName>_cpp` (CPU) and `<TestName>_cu` (CUDA), unless the header contains the token `ONLY_CU` or `ONLY_CPU`.

### Writing a New Test

1. Create `tests/<category>/<test_name>.h` (or `utests/<category>/`).
2. Include `<tests/main.h>` at the top and implement `int launch() { ... return 0; }`.
3. Link against `FKL::FKL` (done automatically by the CMake machinery).
4. No need to modify any `CMakeLists.txt`; `discover_tests` picks it up automatically.
5. If the test is CUDA-only, add the token `ONLY_CU` anywhere in a comment. If CPU-only, add `ONLY_CPU`.

### Example test structure

```cpp
#include <tests/main.h>
#include <fused_kernel/fused_kernel.h>
// ... other includes

using namespace fk;

int launch() {
    Stream stream;
    // ... test code ...
    return 0; // return non-zero on failure
}
```

---

## Code Style

- **Formatting:** LLVM-based clang-format (see `.clang-format`). Column limit: 120. Indent width: 4 spaces. No tabs.
- **Namespace:** All library code lives in `namespace fk`.
- **Header guards:** `#ifndef FK_<UPPERCASE_NAME>` / `#define FK_<UPPERCASE_NAME>`.
- **Macros:** Key utility macros are in `compiler_macros.h`:
  - `FK_HOST_DEVICE_CNST` – `__host__ __device__ constexpr`
  - `FK_HOST_DEVICE_FUSE` – `__forceinline__ __host__ __device__`
  - `FK_HOST_CNST` – `constexpr` (host only)
  - `FK_HOST_FUSE` – `inline` (host only)
  - `CLANG_HOST_DEVICE` – `1` when Clang is compiling CUDA (both host and device), `0` otherwise
- **Copyright headers:** All source files start with `/* Copyright YYYY-YYYY <Author> ... Apache 2.0 ... */`.
- **Templates:** Heavy use of C++20 template metaprogramming with `std::enable_if_t`, `constexpr bool` predicates, and type lists (`TypeList<...>`).

---

## Adding a New Algorithm / Operation

1. Create `include/fused_kernel/algorithms/<category>/<name>.h`.
2. Define an `Operation` struct with the appropriate `InstanceType` tag and a static `exec(...)` function matching its type signature.
3. If the operation needs runtime parameters, add a `ParamsType` member typedef and match it in `exec`.
4. Wrap the operation in its corresponding `InstantiableOperation` (e.g., `using MyAlgorithm = Binary<MyAlgorithmOp>;`).
5. Expose a `build(...)` static method on the instantiable type.
6. Include the new header in `include/fused_kernel/algorithms/algorithms.h` (or the appropriate sub-aggregator).
7. Write a test header in `tests/algorithm/<category>/` or `utests/algorithm/<category>/`.

---

## Common Pitfalls

- **CUDA 11 is not supported.** Only CUDA 12.x and 13.x. Attempting to build with CUDA 11 will not work with the current `deploy.cmake` and architecture guards.
- **GPU compute capability < 7.0 is filtered out** by `archs.cmake`. Do not target Maxwell or older GPUs.
- **`InputFoldType` must be passed by value** in `operator|` and fuser overloads; do not add `const` or `&`.
- **FusedOperation wrapping:** Operations passed to `FusedOperation<>` must be wrapped in their InstantiableOperation type (`Unary<>`, `Binary<>`, etc.), not raw Operation structs.
- **Relocatable Device Code (RDC) is disabled** (`-rdc=false`) for performance reasons. Do not add `__device__` functions that require RDC.
- **Template depth:** Default is 1000 (`-ftemplate-depth=1000`). If you hit recursive instantiation limits, check `TEMPLATE_DEPTH`.
- **`CMAKE_CUDA_COMPILER_ID`** (not `CMAKE_CUDA_COMPILER`) should be used to detect the CUDA compiler type in CMake scripts. Valid values are `"NVIDIA"` (nvcc) and `"Clang"`.
- The `parallel_architectures.h` header has a `static_assert(false)` guarded by `#ifndef CLANG_HOST_DEVICE`. Always ensure `compiler_macros.h` is included before `parallel_architectures.h`.
