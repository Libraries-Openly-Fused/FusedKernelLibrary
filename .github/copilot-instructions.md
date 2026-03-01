# Copilot Instructions for FusedKernelLibrary

## Overview

FusedKernelLibrary (FKL) is a **C++20, header-only** library implementing a GPU/CPU kernel fusion methodology. It allows the user to compose chains of typed operations (read → compute → write) that are fused into a single kernel launch at runtime. The library supports CUDA (via nvcc or clang-CUDA) and a CPU backend.

**Use the `LTS-C++17` branch as the stable reference.** The `main` branch is where the API may change and C++ standard can be raised.

All public types live under the `fk::` namespace.

---

## Repository Layout

```
FusedKernelLibrary/
├── include/fused_kernel/         # All public headers (header-only library)
│   ├── fused_kernel.h            # Single-include entry point → executeOperations()
│   ├── core/
│   │   ├── execution_model/      # DPPs, Executors, Operation types, IOps, Fuser
│   │   │   ├── operation_model/  # Core type system: types, IOps, parent ops, FusedOp
│   │   │   ├── data_parallel_patterns.h  # TransformDPP, DivergentBatchTransformDPP
│   │   │   ├── executors.h       # Executor<DPP> – launches kernels
│   │   │   └── executor_details/ # CUDA kernel launchers (nvcc/clang-CUDA only)
│   │   ├── data/                 # Ptr2D, Tensor, RawPtr, Size, Point, Array, Tuple
│   │   ├── constexpr_libs/       # constexpr math, saturate, vector helpers
│   │   └── utils/                # compiler_macros.h, type lists, parameter pack utils
│   └── algorithms/
│       ├── basic_ops/            # Add, Sub, Mul, Div, Cast, Set, MemOps, etc.
│       └── image_processing/     # Resize, Crop, ColorConversion, Saturate, etc.
├── lib/                          # CMake install/export targets (FKL::FKL)
├── tests/                        # Integration tests (.h files, auto-discovered)
├── utests/                       # Unit tests (.h files, auto-discovered)
├── benchmarks/                   # Benchmark targets (disabled by default)
├── cmake/
│   ├── cmake_init.cmake          # Output dirs, config types, TEMPLATE_DEPTH
│   ├── cuda_init.cmake           # Enables CUDA language, includes cuda/*.cmake
│   ├── libs/cuda/                # CUDA helpers: archs, debug, deploy, target_generation
│   └── tests/                   # Test discovery and generation macros
└── .github/workflows/            # CI: Linux AMD64, Linux ARM64, Windows AMD64
```

---

## Build System

### Requirements
- **CMake ≥ 3.22** (CI uses 4.x)
- **Ninja** generator is preferred and required in CI
- C++20 host compiler: GCC, Clang, or MSVC (≥ VS2019)
- Optional: CUDA Toolkit **12.x or 13.x** (CUDA 11 is not supported)

### Key CMake Options

| Option | Default | Description |
|---|---|---|
| `ENABLE_CPU` | `ON` | Build the CPU backend |
| `ENABLE_CUDA` | `ON` (if nvcc found) | Build the CUDA backend |
| `BUILD_TEST` | `ON` | Build integration tests |
| `BUILD_UTEST` | `ON` | Build unit tests |
| `ENABLE_BENCHMARK` | `OFF` | Build benchmark targets |
| `ENABLE_DEBUG` | `OFF` | CUDA `-G` debug device code |
| `ENABLE_NVTX` | `OFF` | Link NVTX profiling support |
| `ENABLE_LINE_INFO` | `ON` | Add `-lineinfo` to CUDA builds |
| `CUDA_ARCH` | `"native"` | CUDA architectures to build (native/all/explicit list) |
| `TEMPLATE_DEPTH` | `"1000"` | `-ftemplate-depth` (avoids recursion limits) |

### Typical Local Build

```bash
cmake -G Ninja -B build \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_CUDA_COMPILER=nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.9 \
  -DCMAKE_BUILD_TYPE=Release \
  -S .
cmake --build build
cd build && ctest --build-config Release
```

### CPU-only Build (no CUDA)

```bash
cmake -G Ninja -B build \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_BUILD_TYPE=Release \
  -S .
cmake --build build
cd build && ctest
```

### Known CI Quirks

- **Linux AMD64/ARM64**: compilers are passed as comma-separated strings in the matrix (e.g., `"g++,nvcc,12.9"`), parsed by `IFS=","` in the workflow shell step.
- **Windows AMD64**: matrix uses `toolset` strings like `"cl,14.44,nvcc,12.9"`. The workflow invokes VS Dev Shell via PowerShell and patches `rules.ninja` when using nvcc (workaround for an empty-path bug in rules.ninja generation).
- The `ctest` command in the Linux workflows has a stray `}` in `--build-config Release}` – this is a known typo in the workflow files but does not block execution because ctest ignores unknown tokens after valid option values.
- Self-hosted runners are required for all CI workflows; there are no hosted-runner fallbacks.

---

## Core Concepts

### Operation Types

Every operation in FKL has an `InstanceType` tag that controls the `exec()` signature:

| InstanceType | `exec` signature | Description |
|---|---|---|
| `ReadType` | `OutputType exec(Point, ParamsType)` | Read from memory |
| `WriteType` | `void exec(Point, InputType, ParamsType)` | Write to memory |
| `UnaryType` | `OutputType exec(InputType)` | Transform, no params |
| `BinaryType` | `OutputType exec(InputType, ParamsType)` | Transform with params |
| `ReadBackType` | `OutputType exec(Point, ParamsType, BackIOp)` | Backwards-vertical read |
| `IncompleteReadBackType` | *(no exec)* | Builder for ReadBackType batching |
| `TernaryType` | `OutputType exec(InputType, ParamsType, BackIOp)` | Transform with params + back-IOp |
| `MidWriteType` | `InputType exec(Point, InputType, ParamsType)` | Write, passes input through |
| `OpenType` | `OutputType exec(Point, InputType, ParamsType)` | FusedOperation with mid-write |
| `ClosedType` | `void exec(Point, ParamsType)` | Self-contained read+write |

### Instantiable Operations (IOps)

Every Operation is wrapped in an **InstantiableOperation** (IOp) to hold runtime parameters:

```cpp
Read<Op>          // wraps a ReadType Operation
Write<Op>         // wraps a WriteType Operation
Unary<Op>         // wraps a UnaryType Operation
Binary<Op>        // wraps a BinaryType Operation (holds params)
Ternary<Op>       // wraps a TernaryType Operation (holds params + backIOp)
ReadBack<Op>      // wraps a ReadBackType Operation
IncompleteReadBack<Op>
MidWrite<Op>
Open<Op>
Closed<Op>
```

**Use `Op::build(params)` to construct IOps** – never construct them manually.

### The `operator|` Pipeline

IOps are chained with `operator|`, producing an `InputFoldType<T>` that carries the thread index and the current value:

```cpp
// Inside a kernel:
(InputFoldType(thread, i_data) | ... | iOpInstances).input
```

### Fusing IOps at Host Side

Two IOps can be fused at host side with `operator&` or the `.then()` method:

```cpp
auto fusedIOp = readOp.then(mulOp).then(writeOp);
// or
auto fusedIOp = readOp & mulOp & writeOp;
```

The `Fuser::fuse()` function combines them into a `FusedOperation<...>`.

The free function `fuse(op1, op2, ...)` also works for combining multiple IOps at once.

### FusedOperation

`FusedOperation<IOp1, IOp2, ...>` is a single Operation whose `InstanceType` is inferred from its components (ReadType, WriteType, UnaryType, BinaryType, OpenType, or ClosedType). The `IS_FUSED_OP` flag is set to `true` on all FusedOperation types.

When adding operations inside a FusedOperation, they must be wrapped in the appropriate IOp type (e.g., `Unary<MyOp>`, `Binary<MyOp>`).

### Data Parallel Patterns (DPPs)

DPPs define how a grid of threads executes a sequence of IOps. The main one is:

```cpp
TransformDPP<ParArch::GPU_NVIDIA>   // CUDA kernel
TransformDPP<ParArch::CPU>          // CPU nested loops
```

`DivergentBatchTransformDPP` launches different IOp sequences on different z-planes of the same grid.

### executeOperations

The public entry point:

```cpp
#include <fused_kernel/fused_kernel.h>

fk::executeOperations<TransformDPP<>>(stream,
    PerThreadRead<ND::_2D, uchar3>::build(inputPtr),
    Mul<float3>::build(mulValue),
    PerThreadWrite<ND::_2D, uchar3>::build(outputPtr));
```

`TransformDPP<>` defaults to `TransformDPP<ParArch::GPU_NVIDIA, TF::DISABLED, void>` when CUDA is available, or `TransformDPP<ParArch::CPU, ...>` otherwise.

### Thread Fusion (TF)

When `TF::ENABLED`, a single thread processes multiple adjacent elements to improve memory throughput. Enable with `TransformDPP<ParArch::GPU_NVIDIA, TF::ENABLED>`. Thread fusion requires both the Read and Write operations to support it (`THREAD_FUSION = true` in their `ReadOperation`/`WriteOperation` parent).

---

## Defining a New Operation

### Unary Operation Example

```cpp
namespace fk {
    template <typename I, typename O = I>
    struct MyOp {
    private:
        using SelfType = MyOp<I, O>;
    public:
        FK_STATIC_STRUCT(MyOp, SelfType)
        using Parent = UnaryOperation<I, O, MyOp<I, O>>;
        DECLARE_UNARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType input) {
            return static_cast<O>(input * 2);
        }
    };
} // namespace fk

// Usage:
auto iop = fk::Unary<MyOp<float>>::build();  // or MyOp<float>::build()
```

### Binary Operation Example

```cpp
namespace fk {
    template <typename I, typename P = I, typename O = I>
    struct MyScaleOp {
    private:
        using SelfType = MyScaleOp<I, P, O>;
    public:
        FK_STATIC_STRUCT(MyScaleOp, SelfType)
        using Parent = BinaryOperation<I, P, O, MyScaleOp<I, P, O>>;
        DECLARE_BINARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType input, const ParamsType& params) {
            return input * params;
        }
    };
} // namespace fk

// Usage:
auto iop = fk::MyScaleOp<float>::build(2.5f);
```

### Key Macros

| Macro | Purpose |
|---|---|
| `FK_STATIC_STRUCT(Name, SelfType)` | Deletes copy/move constructors to enforce static-only usage |
| `DECLARE_UNARY_PARENT` | Inherits types and `build()` from `UnaryOperation` parent |
| `DECLARE_BINARY_PARENT` | Inherits types, `build()`, and `exec(opData)` from `BinaryOperation` parent |
| `DECLARE_READ_PARENT` | Inherits types and `build()` from `ReadOperation` parent |
| `DECLARE_WRITE_PARENT_BASIC` | Inherits types and `build()` from `WriteOperation` parent |
| `DECLARE_TERNARY_PARENT` | Inherits types and `build()` from `TernaryOperation` parent |
| `FK_HOST_DEVICE_FUSE` | `__host__ __device__ __forceinline__` (or plain `inline` on CPU) |
| `FK_HOST_FUSE` | `__host__ __forceinline__` (or `inline`) |
| `FK_HOST_CNST` | `constexpr` host function |
| `FK_HOST_DEVICE_CNST` | `constexpr __host__ __device__` |
| `CLANG_HOST_DEVICE` | `1` when clang is compiling CUDA (both host and device code) |

---

## Compiler / CUDA Notes

- CUDA compiler detection: use `CMAKE_CUDA_COMPILER_ID` (values: `"NVIDIA"` for nvcc, `"Clang"` for clang-CUDA). Do **not** use `CMAKE_CUDA_COMPILER`.
- Only CUDA 12.x and 13.x are actively supported. CUDA 11 support was intentionally removed.
- Minimum supported GPU architecture is SM 7.0 (Volta). Pre-7.0 architectures are filtered out automatically in `cmake/libs/cuda/archs.cmake`.
- `CUDA_STANDARD` is set to **20** for all CUDA targets. `CXX_STANDARD` is set to **20** for all test targets.
- The `TEMPLATE_DEPTH` CMake variable defaults to `1000` and is forwarded to `-ftemplate-depth=` on GCC/Clang to avoid template instantiation depth errors in deeply recursive code.
- On Windows with MSVC, `/bigobj` is added automatically; clang-cl with the `llvm` frontend uses GNU-mode clang (`clang++.exe`, not `clang-cl.exe`) for CUDA compilation.

---

## Test Infrastructure

### Test Discovery

Tests are **auto-discovered** by CMake from `.h` files inside `tests/` and `utests/` subdirectories. A generated `launcher.{cpp,cu}` file wraps each test header using `tests/launcher.in`.

### Test Structure

Each test `.h` file must:
1. `#include <tests/main.h>` (declares `int launch();`)
2. Define an `int launch()` function that returns `0` on success, non-zero on failure.

```cpp
#include <tests/main.h>

int launch() {
    // test logic
    return 0; // success
}
```

### CPU-only vs CUDA-only Tests

- Add the string `ONLY_CU` anywhere in the file to build only as `.cu` (CUDA only).
- Add the string `ONLY_CPU` anywhere in the file to build only as `.cpp` (CPU only).
- If neither marker is present, both a `.cpp` (CPU) and a `.cu` (CUDA) target are generated.

### Test Targets

Each test file `my_test.h` produces:
- `my_test_cpp` – CPU executable
- `my_test_cu` – CUDA executable

Both are registered with CTest.

---

## Data Types

- `fk::Point` – 3D thread index `{x, y, z}`
- `fk::ActiveThreads` – `{x, y, z}` unsigned grid dimensions
- `fk::RawPtr<ND, T>` – raw pointer + dims (1D/2D/3D)
- `fk::Ptr<ND, T>` / `fk::Ptr2D<T>` – managed pointer wrappers
- `fk::Tensor<T>` – 3D contiguous device buffer
- `fk::Size` – `{width, height}`
- `fk::Rect` – `{x, y, width, height}` crop region
- `fk::Tuple<...>` – GPU-safe tuple (use `fk::get<N>()`, `fk::apply()`)
- `fk::Array<T, N>` – GPU-safe fixed-size array

CUDA vector types (`float3`, `uchar3`, etc.) are available via `<fused_kernel/core/data/vector_types.h>`. Helper `make_<T>(...)` and `make_set<T>(val)` construct vector types.

---

## Common Patterns

### Batch (Horizontal Fusion)

Use `PerThreadRead<ND::_2D, T>::build(std::array<Ptr2D<T>, BATCH>)` to build a batch read IOp that processes `BATCH` planes with one kernel.

### Backwards Vertical Fusion (Resize/Crop)

Use `Resize<InterpolationType, AspectRatio>::build(backIOp, outputSize)` to create a `ReadBack` IOp. The `backIOp` is fused in and only the pixels required by interpolation are fetched from source memory.

### Color Conversion

`ColorConversion<code, I, O>` is a type alias resolving to either a plain operation or a `FusedOperation` depending on the conversion code. Use it directly as the operation type in `Unary<>` or `Binary<>`.

---

## Style / Formatting

- Code style is enforced by `.clang-format` (LLVM-based, column limit 120, indent 4 spaces, `PointerAlignment: Right`).
- All source files use the Apache 2.0 license header.
- `InputFoldType` is passed **by value** (no `const &`) in `operator|` overloads for instantiable operations.
- Prefer `constexpr` everywhere possible; many tests are `constexpr` functions validated with `static_assert`.

---

## Errors and Workarounds

### Template Depth Exceeded
**Symptom**: Compiler error about exceeding maximum template instantiation depth.
**Fix**: Increase `TEMPLATE_DEPTH` in CMake (e.g., `-DTEMPLATE_DEPTH=1500`). The default is 1000.

### Empty nvcc Path in rules.ninja (Windows)
**Symptom**: Build fails because `rules.ninja` contains an empty nvcc path.
**Workaround**: The Windows CI workflow patches `rules.ninja` after CMake configure:
```powershell
(Get-Content build\CMakeFiles\rules.ninja) -replace "\\nvcc\\bin\\nvcc.exe", "$env:CUDACXX" | Set-Content build\CMakeFiles\rules.ninja
```

### Pre-SM70 GPU Architecture Filtered Out
**Symptom**: CMake prints a warning about skipping deprecated GPU architectures.
**Explanation**: `cmake/libs/cuda/archs.cmake` removes architectures older than SM 7.0 automatically. This is expected.

### MSVC < 2019 CPU Backend Disabled
**Symptom**: `ENABLE_CPU` is forced to `OFF`.
**Explanation**: The CPU backend requires C++20 features not available in VS 2017. Use VS 2019 or later.


