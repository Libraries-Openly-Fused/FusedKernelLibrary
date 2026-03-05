# Copilot Instructions for FusedKernelLibrary (FKL)

## Project Overview

**FusedKernelLibrary (FKL)** is a header-only C++17 library that enables automatic GPU kernel fusion (Vertical, Horizontal, Backwards Vertical, and Divergent Horizontal Fusion) for CUDA and CPU backends. The library lives under `include/fused_kernel/`. All public types are in the `fk` namespace.

- The primary entry point header is `include/fused_kernel/fused_kernel.h`.
- The main user-facing function is `fk::executeOperations<DPPType>(stream, iop1, iop2, ...)`.
- Current version: `0.1.14-LTS` (C++17 API freeze branch).

## Repository Structure

```
FusedKernelLibrary/
├── .github/workflows/       # CI workflows (Linux x86_64, Linux ARM64, Windows x64)
├── cmake/                   # All CMake helper modules
│   ├── cmake_init.cmake     # Global settings, output dirs, config types
│   ├── cuda_init.cmake      # CUDA language enable + arch detection
│   ├── libs/cuda/           # CUDA-specific helpers (archs, deploy, debug, target generation)
│   ├── tests/               # Test discovery and generation (discover_tests.cmake, add_generated_test.cmake)
│   └── generators/          # Code generators (version_header.cmake, export_header.cmake)
├── include/fused_kernel/    # All library headers (header-only)
│   ├── fused_kernel.h       # Top-level include + executeOperations free functions
│   ├── core/
│   │   ├── execution_model/ # Operation types, instantiable ops, DPPs, executors, stream
│   │   ├── data/            # Data types: Ptr2D, Tensor, Size, Rect, Point, Tuple, Array, etc.
│   │   ├── utils/           # Compiler macros, template utils, type lists, vector utils
│   │   └── constexpr_libs/  # Constexpr math (constexpr_cmath.h)
│   └── algorithms/
│       ├── basic_ops/       # Arithmetic, cast, logical, memory ops, set, static loop, vector ops
│       └── image_processing/ # Crop, Resize, ColorConversion, BorderReader, Interpolation, Warp, etc.
├── lib/                     # CMake INTERFACE library target (FKL::FKL) and install config
├── tests/                   # Integration tests (discovered from .h files by CMake)
├── utests/                  # Unit tests (discovered from .h files by CMake)
├── benchmarks/              # Performance benchmarks (off by default)
├── CMakeLists.txt           # Root CMake, version 0.1.14, requires CMake 3.24+
└── .clang-format            # LLVM-based style, 4-space indent, 120 column limit
```

## Build System

### Requirements
- **CMake ≥ 3.24** (CI uses cmake 4.2.1 custom install)
- **C++17** standard required (enforced via `CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO`)
- **CUDA 12.x or 13.x** — CUDA 11 is **not** supported
- **Host compilers**: `g++-13`, `g++-11` (ARM64), `clang++-21`, `cl` (MSVC 14.44), `clang-cl`
- Only **nvcc** is supported as the CUDA compiler (clang-as-CUDA-compiler is not supported)
- **Ninja** generator is used in CI; Visual Studio generator also works on Windows

### CMake Options
| Option | Default | Description |
|---|---|---|
| `ENABLE_CPU` | ON | Enable CPU backend (disabled for MSVC < 2019) |
| `ENABLE_CUDA` | ON (if nvcc found) | Enable CUDA backend |
| `BUILD_TEST` | ON | Build integration tests under `tests/` |
| `BUILD_UTEST` | ON | Build unit tests under `utests/` |
| `ENABLE_BENCHMARK` | OFF | Build benchmarks under `benchmarks/` |
| `CUDA_ARCH` | `native` | Target CUDA architectures (e.g., `native`, `all`, `89`, `86;89`) |

### Build Commands (Linux)
```bash
# Configure
cmake -G "Ninja" -B build -DCMAKE_BUILD_TYPE=Release -S .

# Build
cmake --build build --config Release

# Test
cd build && ctest --build-config Release --output-junit test_results.xml
```

### Build Commands (Windows, in VS Developer Shell with Ninja)
```powershell
# Set compilers via env vars (as CI does)
$env:CUDACXX = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe"
$env:CC = "cl"  # or "clang-cl"
$env:CXX = "cl"

cmake -G "Ninja" -B build -DCMAKE_BUILD_TYPE=Release -S .
cmake --build build --config Release
```

### Known Windows Workaround
On Windows with Ninja, the generated `CMakeFiles/rules.ninja` may have an empty path for nvcc. The CI workaround patches it:
```powershell
(Get-Content build\CMakeFiles\rules.ninja) -replace "\\nvcc\\bin\\nvcc.exe", $env:CUDACXX | Set-Content build\CMakeFiles\rules.ninja
```

## CI Workflows

Workflows trigger on **pull requests** targeting branches matching `LTS-C*`. All runners are **self-hosted**.

| Workflow | Runner | Compilers | CUDA versions |
|---|---|---|---|
| `cmake-linux-amd64.yml` | `linux, x64` | `g++-13`, `clang++-21` | 12.9, 13.0 |
| `cmake-linux-arm64.yml` | `linux, arm64` | `g++-11`, `clang++-21` | 12.9 |
| `cmake-windows-amd64.yml` | `windows, x64` | `cl`, `clang-cl` (LLVM 21.1.0) | 12.9, 13.0 |

Compilers are set via `CC`, `CXX`, `CUDACXX` environment variables in the "Set reusable strings" step — not as CMake `-D` flags.

## Test Infrastructure

### How Tests Are Discovered
CMake auto-discovers tests from `.h` files in `tests/` and `utests/` subdirectories using `discover_tests()` in `cmake/tests/discover_tests.cmake`. For each `.h` file:
- A `.cpp` target is generated (CPU backend) unless the file contains `ONLY_CU`
- A `.cu` target is generated (CUDA backend) unless the file contains `ONLY_CPU`
- Files matching `*_common*` are excluded from auto-discovery

A `configure_file()` step generates a launcher from `tests/launcher.in` that includes the test header and calls `launch()`.

### Test Conventions
- Each test `.h` file must define a function `int launch()` that returns 0 on success
- Tests that are CPU-only contain the string `ONLY_CPU` (as a marker, not necessarily as a macro)
- Tests that are CUDA-only contain the string `ONLY_CU`
- Tests link against `FKL::FKL` (the header-only interface library)

### Adding a New Test
1. Create a `.h` file in an appropriate subdirectory of `tests/` or `utests/`
2. Include the necessary FKL headers
3. Define `int launch() { ... return 0; }`
4. Add `ONLY_CPU` or `ONLY_CU` in a comment if needed to restrict to one backend

## Core Concepts

### Operation Types
Operations are classified by their `InstanceType` member (defined in `operation_types.h`):

| Type | exec signature | Description |
|---|---|---|
| `ReadType` | `OutputType exec(Point, ParamsType)` | Reads from memory |
| `WriteType` | `void exec(Point, InputType, ParamsType)` | Writes to memory |
| `UnaryType` | `OutputType exec(InputType)` | Pure computation, no params |
| `BinaryType` | `OutputType exec(InputType, ParamsType)` | Computation with params |
| `ReadBackType` | `OutputType exec(Point, ParamsType, BackIOp)` | Read with backward-fused op |
| `TernaryType` | `OutputType exec(InputType, ParamsType, BackIOp)` | Compute with params and backward op |
| `MidWriteType` | `InputType exec(Point, InputType, ParamsType)` | Writes and passes input through |

### Instantiable Operations (IOps)
Operations are wrapped in `InstantiableOperation` structs that hold runtime parameters. Aliases:
- `fk::Read<Op>`, `fk::Write<Op>`, `fk::Unary<Op>`, `fk::Binary<Op>`, `fk::Ternary<Op>`, `fk::ReadBack<Op>`, `fk::MidWrite<Op>`, `fk::Open<Op>`, `fk::Closed<Op>`
- Use `fk::Instantiable<Op>` to automatically select the right wrapper based on `Op::InstanceType`

Operations are constructed via a static `build(...)` method that returns the wrapped IOp.

### Data Parallel Patterns (DPPs)
DPPs determine how threads are organized. The main one is `TransformDPP<THREAD_FUSION>` (where `THREAD_FUSION` defaults to `false`). Pass the DPP as the first template argument to `executeOperations`.

### Key Data Types
- `fk::Ptr2D<T>` / `fk::Ptr3D<T>` — 2D/3D pitched GPU pointers
- `fk::Tensor<T>` — contiguous multi-plane GPU array
- `fk::Size` — width/height size
- `fk::Rect` — x, y, width, height rectangle
- `fk::Point` — thread index (x, y, z)
- `fk::Tuple<Ts...>` — GPU-safe tuple (use instead of `std::tuple` in device code)
- `fk::Stream` / `fk::Stream_<ParArch::GPU_NVIDIA>` — CUDA stream wrapper

### Fusion API (`.then()` and `operator&`)
IOps support chaining:
```cpp
auto fusedIOp = readIOp.then(unaryIOp1, unaryIOp2, writeIOp);
// equivalent to
auto fusedIOp = readIOp & unaryIOp1 & unaryIOp2 & writeIOp;
```

### Compiler Macros (`compiler_macros.h`)
- `_MSC_VER_EXISTS` — 1 when compiling with MSVC
- `CLANG_HOST_DEVICE` — 1 when clang compiles CUDA in host+device mode
- `VS2017_COMPILER` / `NO_VS2017_COMPILER` — detect VS2017 compiler
- `FK_HOST_DEVICE_CNST`, `FK_HOST_FUSE`, `FK_DEVICE_FUSE`, etc. — cross-platform `__host__ __device__ __forceinline__ constexpr` equivalents defined in `utils.h`

### NVRTC Support
The library supports NVRTC (runtime compilation) via the `NVRTC_COMPILER` define. When set, `INSTANTIABLE_OPERATION_THEN` and some host-only features are disabled.

## Code Style

- **Formatting**: LLVM-based, 4-space indent, 120-column limit (`.clang-format` in repo root)
- **C++ Standard**: C++17 strictly (no extensions)
- **Copyright header**: Every file begins with an Apache 2.0 license header
- **Include guards**: `#ifndef FK_XXX_H` / `#define FK_XXX_H` (not `#pragma once` in most files)
- **Namespace**: All public API is in namespace `fk`
- **Templates**: Heavy use of SFINAE (`std::enable_if_t`), type traits, and variadic templates
- **No exceptions in device code**: Only host code uses `std::runtime_error`
- **Pointer alignment**: Right (i.e., `T* ptr`, not `T *ptr`)

## CUDA Architecture Notes

- Minimum supported compute capability: **7.0** (sm_70, Volta)
- `CUDA_ARCH=native` (default) auto-detects via `nvidia-smi` for CUDA < 13
- For CUDA 12: curand DLL is `curand64_11`, cufft DLL is `cufft64_11`
- For CUDA 13: curand DLL is `curand64_10`, cufft DLL is `cufft64_12`; DLLs are in `x64/` subdirectory

## Common Errors and Workarounds

1. **Windows/Ninja: empty nvcc path in `rules.ninja`** — Apply the `rules.ninja` patch in CI (`cmake-windows-amd64.yml` step "Configure CMake").
2. **CUDA < 13 + `CUDA_ARCH=all`** — The build system automatically filters out GPU architectures below sm_70.
3. **MSVC < 2019** — CPU backend is automatically disabled (`ENABLE_CPU OFF`).
4. **Template depth** — `TEMPLATE_DEPTH` is set to 1000 via `cmake_init.cmake` for deeply nested fusion expressions.
5. **`/bigobj` on MSVC** — Required due to large generated test binaries; added automatically in `add_generated_test.cmake`.
6. **`/Zc:preprocessor` on MSVC** — Required to avoid traditional preprocessor warnings; added in `add_generated_test.cmake`.

## How to Add a New Operation

1. Create a struct in `include/fused_kernel/algorithms/` (or `core/execution_model/`) with:
   - `using InstanceType = fk::<SomeOperationType>;`
   - `using InputType = ...;` / `using OutputType = ...;` / `using ParamsType = ...;` as required
   - A static `FK_HOST_DEVICE_CNST`/`FK_DEVICE_FUSE` `exec(...)` function matching the InstanceType signature
2. If the operation needs a `build()` factory, wrap it in an `Instantiable<YourOp>` specialization or provide a custom `build()` static method
3. Add a test `.h` in `tests/` or `utests/` with `int launch()` to exercise it
