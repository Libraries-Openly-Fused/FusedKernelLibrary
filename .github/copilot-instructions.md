# Copilot Instructions for FusedKernelLibrary (FKL)

## Project Overview

**FusedKernelLibrary (FKL)** is a header-only C++17 library that enables automatic GPU kernel fusion (Vertical, Horizontal, Backwards Vertical, and Divergent Horizontal Fusion) for CUDA, HIP (AMD ROCm) and CPU backends. The library lives under `include/fused_kernel/`. All public types are in the `fk` namespace. Name the FKL vector types `fk::float3`, `fk::uchar3`...: under HIP the global `float3`, `uchar3`... are HIP's own types, which are not FKL vector types (with nvcc and on the CPU the `fk::` names are the global types).

- The primary entry point header is `include/fused_kernel/fused_kernel.h`.
- The main user-facing function is `fk::executeOperations<DPPType>(stream, iop1, iop2, ...)`.
- Current version: `0.1.14-LTS` (C++17 API freeze branch).

## Repository Structure

```
FusedKernelLibrary/
├── .github/workflows/       # CI workflows (Linux x86_64, Linux ARM64, Linux x86_64 ROCm, Windows x64)
├── cmake/                   # All CMake helper modules
│   ├── cmake_init.cmake     # Global settings, output dirs, config types
│   ├── cuda_init.cmake      # CUDA language enable + arch detection
│   ├── libs/cuda/           # CUDA-specific helpers (archs, deploy, debug, target generation)
│   ├── libs/hip/            # HIP (ROCm) helpers (hip_init, archs, debug, target generation)
│   ├── tests/               # Test discovery and generation (discover_tests.cmake, add_generated_test.cmake)
│   └── generators/          # Code generators (version_header.cmake, export_header.cmake)
├── include/fused_kernel/    # All library headers (header-only)
│   ├── fused_kernel.h       # Top-level include + executeOperations free functions
│   ├── core/
│   │   ├── execution_model/ # Operation types, instantiable ops, DPPs, executors, stream
│   │   ├── data/            # Data types: Ptr2D, Tensor, Size, Rect, Point, Tuple, Array, etc.
│   │   ├── utils/           # Compiler macros, template utils, type lists, vector utils
│   │   ├── constexpr_libs/  # Constexpr math (constexpr_cmath.h)
│   │   └── core.h           # Include everything in core folder
│   └── algorithms/
│       ├── basic_ops/        # Arithmetic, cast, logical, memory ops, set, static loop, vector ops
│       ├── image_processing/ # Crop, Resize, ColorConversion, BorderReader, Interpolation, Warp, etc.
│       └── algorithms.h      # Include everything in Algorithms
├── lib/                     # CMake INTERFACE library target (FKL::FKL) and install config
├── tests/                   # Integration tests (discovered from .h files by CMake)
├── utests/                  # Unit tests (discovered from .h files by CMake)
├── CMakeLists.txt           # Root CMake, version 0.1.14, requires CMake 3.28+
└── .clang-format            # LLVM-based style, 4-space indent, 120 column limit
```

## Build System

### ⚠️ Copilot Constraints
- **Build Directory:** Always output compiled binaries, artifacts, or generated files to a `build` directory located strictly **outside** the current source folder (e.g., `../build`). Never create the build folder within the project repository.
- **Source Directory** Never add any file that should not be part of the repository, in the source folder. Always create folders outside the source folder.
- **Git Ignore:** As a consequence of the previous two rules, under no circumstances should you modify, append to, or suggest changes to the `.gitignore` file.

### Requirements
- **CMake ≥ 3.28** (CI uses cmake 4.3.3 custom install)
- **C++17** standard required (enforced via `CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO`)
- **CUDA 12.x or 13.x** 
- **HIP (ROCm)**, optional: amdclang++ through CMake's HIP language (`-DCMAKE_HIP_COMPILER=/opt/rocm/bin/amdclang++`); HIP sources compile as C++17 with `-Werror=c++20-extensions`
- **Host compilers**: ``g++-13` (ARM64, x86_64 for cuda 12.9),g++-15` (x86_64), `clang++-21`, `cl` (MSVC 14.44, MSVC 14.51), `clang-cl`
- Only **nvcc** is supported as the CUDA compiler 
- **Ninja** generator is used in CI; Visual Studio generator also works on Windows

### CMake Options
| Option | Default | Description |
|---|---|---|
| `ENABLE_CPU` | ON | Enable tests on CPU backend |
| `ENABLE_CPU_MULTITHREAD` | ON (if the `thread-pool` submodule is checked out) | Enable the multithreaded CPU backend (`ParArch::CPU_MULTITHREAD`) |
| `ENABLE_CUDA` | ON (if nvcc found) | Enable tests on CUDA backend |
| `BUILD_TEST` | ON | Build integration tests under `tests/` |
| `BUILD_UTEST` | ON | Build unit tests under `utests/` |
| `CUDA_ARCH` | `native` | Target CUDA architectures (e.g., `native`, `all`, `89`, `86;89`) |
| `ENABLE_HIP` | ON (if a HIP compiler for AMD GPUs is found) | Enable tests on HIP backend |
| `HIP_ARCH` | the local GPU | HIP architectures on the first configure of a build directory (e.g., `gfx1200`); `CMAKE_HIP_ARCHITECTURES` afterwards |

### Build Commands (Linux)
```bash
#setup compilers
   
export PATH=/home/cudeiro/cmake-4.3.3-linux-aarch64/bin/:$PATH #update with linux-aarch64 for arm64, linux-x86_64 for x86_64
export CUDACXX=/usr/local/cuda-12.9/bin/nvcc  # can be 13.4 as well
export CC=g++-13 # e.g. "g++-13" (ARM64), "g++-15" (x86_64), "clang++-21"
export CXX=g++-13 # e.g. "g++-13" (ARM64), "g++-15" (x86_64), "clang++-21"
cmake -G "Ninja" -B build -DCMAKE_BUILD_TYPE=Release -S .

# Build
cmake --build build --config Release

# Test
cd build && ctest --build-config Release --output-junit test_results.xml
```

### Build Commands (Windows, in VS Developer Shell with Ninja)
```powershell
# Set compilers via env vars (as CI does)
# note:CUDA Toolkit v12.9 can also be 13.0 or 13.3 but only >=13.2 supports >=14.50 developer tools (MSVC 2026)
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
| `cmake-linux-amd64.yml` | `linux, x64, cuda` | `g++-15`, `clang++-21` | 13.4 |
| `cmake-linux-arm64.yml` | `linux, arm64, cuda` | `g++-13`, `clang++-21` | 12.9, 13.4 |
| `cmake-linux-ubuntu2404-amd64.yml` | `linux, x64, ubuntu2404` | `g++-13`, `clang++-21` | 12.9 |
| `cmake-linux-rocm-amd64.yml` | `linux, x64, rocm` | `amdclang++` (ROCm 10.1) for the CPU and the HIP targets | none: HIP on an AMD GPU (gfx1200) |
| `cmake-windows-amd64.yml` | `windows, x64, cuda` | `cl`, `clang-cl` (LLVM 21.1.0) | 12.9, 13.3 (toolkits the runner no longer has: these jobs build the CPU backend only) |

`CC` and `CXX` are set as environment variables in the "Set reusable strings" step; they select the compiler of the CPU targets (nvcc uses the system g++ as host compiler). The Linux workflows pass nvcc as `-DCMAKE_CUDA_COMPILER`, so a missing toolkit fails the configure step, and configure with `-DENABLE_HIP=OFF`. The ROCm workflow is the one that builds the HIP backend, and runs its tests on an AMD GPU: it configures with `-DENABLE_HIP=ON -DENABLE_CUDA=OFF -DENABLE_CPU=ON -DENABLE_CPU_MULTITHREAD=ON` and compiles both the CPU and the HIP targets with `amdclang++` and `-Werror=c++20-extensions -pedantic-errors -Wno-gnu-anonymous-struct`.

## Test Infrastructure

### How Tests Are Discovered
CMake auto-discovers tests from `.h` files in `tests/` and `utests/` subdirectories using `discover_tests()` in `cmake/tests/discover_tests.cmake`. For each `.h` file:
- A `.cpp` target is generated (CPU backend) unless the file contains `ONLY_CU` or `ONLY_GPU`
- A `.cu` target is generated (CUDA backend) unless the file contains `ONLY_CPU`
- A `.hip` target is generated (HIP backend, when enabled) unless the file contains `ONLY_CPU` or `ONLY_CU`
- Files matching `*_common*` are excluded from auto-discovery

A `configure_file()` step generates a launcher from `tests/launcher.in` that includes the test header and calls `launch()`.

### Test Conventions
- Each test `.h` file must define a function `int launch()` that returns 0 on success
- Tests that are CPU-only contain the string `ONLY_CPU` (as a marker, not necessarily as a macro)
- Tests that are CUDA-only contain the string `ONLY_CU`
- Tests for the GPU backends only (CUDA and HIP) contain the string `ONLY_GPU`
- Tests use the `fk::` vector type names, so that they compile for every backend
- Tests link against `FKL::FKL` (the header-only interface library)

### Adding a New Test
1. Create a `.h` file in an appropriate subdirectory of `tests/` or `utests/`
2. Include the necessary FKL headers
3. Define `int launch() { ... return 0; }`
4. Add `ONLY_CPU`, `ONLY_CU` or `ONLY_GPU` if needed to restrict the backends (the marker text anywhere in the file counts)

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
| `IncompleteReadBackType` | `` | ReadBackType that has no info on the BackIOp type and has no exec function, but can store params |
| `TernaryType` | `OutputType exec(InputType, ParamsType, BackIOp)` | Compute with params and backward op |
| `MidWriteType` | `InputType exec(Point, InputType, ParamsType)` | Writes and passes input through |
| `OpenType` | `OutputType exec(Point, InputType, ParamsType)` | Gets the input in registers via InputType parameter, and returns result in registers with OutputType. It can have a MidWrite Operation internally |
| `ClosedType` | `void exec(Point, ParamsType)` | Reads from memory and writes the results to memory, for the coordinate passed in Point. It effectively performs a transform on each coordinate. |

### Instantiable Operations (IOps)
Operations are wrapped in `InstantiableOperation` structs that hold runtime parameters. Aliases:
- `fk::Read<Op>`, `fk::Write<Op>`, `fk::Unary<Op>`, `fk::Binary<Op>`, `fk::Ternary<Op>`, `fk::ReadBack<Op>`, `fk::MidWrite<Op>`, `fk::Open<Op>`, `fk::Closed<Op>`
- Use `fk::Instantiable<Op>` to automatically select the right wrapper based on `Op::InstanceType`
 
Instantiable Operations (IOps) are constructed via a static `build(...)` method on each Operation that returns the wrapped IOp.

### Data Parallel Patterns (DPPs)
DPPs determine how threads are organized. The main one is `TransformDPP<THREAD_FUSION>` (where `THREAD_FUSION` defaults to `false`). Pass the DPP as the first template argument to `executeOperations`.

### Key Data Types
- `fk::Ptr1D<T>` / `fk::Ptr2D<T>` / `fk::Ptr3D<T>` — 1D/2D/3D pitched GPU pointers
- `fk::Tensor<T>` — contiguous multi-plane GPU array
- `fk::Size` — width/height size
- `fk::Rect` — x, y, width, height rectangle
- `fk::Point` — thread index (x, y, z)
- `fk::Tuple<Ts...>` — GPU-safe tuple (use instead of `std::tuple` in device code)
- `fk::Stream` — stream of the default backend; `fk::Stream_<ParArch::GPU_NVIDIA>` / `fk::Stream_<ParArch::GPU_AMD>` — CUDA / HIP stream wrappers (copies share the native stream)

### Fusion API (`.then()` and `operator&`)
IOps support chaining:
```cpp
auto fusedIOp = readIOp.then(unaryIOp1).then(unaryIOp2).then(writeIOp);
// equivalent to
auto fusedIOp = readIOp & unaryIOp1 & unaryIOp2 & writeIOp;
```

### Compiler Macros (`compiler_macros.h`)
- `_MSC_VER_EXISTS` — 1 when compiling with MSVC
- `FK_HOST_DEVICE_CNST`, `FK_HOST_FUSE`, `FK_DEVICE_FUSE`, etc.
- CNST means __forceinline__ constexpr with nvcc and HIP, inline constexpr with CPU compilers.
- FUSE means __forceinline__ static constexpr with nvcc and HIP, inline static constexpr with CPU compilers.
- HOST means __host__ with nvcc and HIP, nothing with CPU compilers.
- DEVICE  means __device__ with nvcc and HIP, nothing with CPU compilers.
- `FK_HIP_BACKEND` is defined for HIP compilations by clang (amdclang++); use it, not `__HIPCC__` (which the HIP headers also define under nvcc), to select HIP code.

## Code Style

- **Formatting**: LLVM-based, 4-space indent, 120-column limit (`.clang-format` in repo root)
- **C++ Standard**: C++17 strictly (no extensions)
- **Copyright header**: Every file begins with an Apache 2.0 license header
- **Include guards**: `#ifndef FK_XXX_H` / `#define FK_XXX_H` (not `#pragma once`)
- **Namespace**: All public API is in namespace `fk`; use the `fk::` vector type names
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
3. **Template depth** — `TEMPLATE_DEPTH` is set to 1000 via `cmake_init.cmake` for deeply nested fusion expressions.
4. **`/bigobj` on MSVC** — Required due to large generated test binaries; added automatically in `add_generated_test.cmake`.
5. **`/Zc:preprocessor` on MSVC** — Required to avoid traditional preprocessor warnings; added in `add_generated_test.cmake`.

## How to Add a New Operation

1. Create a struct in `include/fused_kernel/algorithms/` with:
   - `private: using SelfType = StructName<TemplateTypes...>;`
   - `using Parent = /*Parent operation according to the InstanceType of the operation*/;`
   - `public: FK_STATIC_STRUCT(StructName, SelfType)`
   - `DECLARE_/*use macro according to InstanceType*/_PARENT`
   - A `FK_HOST_DEVICE_FUSE` `exec(...)` function matching the InstanceType signature
2. If the operation needs a `build()` factory, wrap it in an `Instantiable<YourOp>` specialization or provide a custom `build()` static method
3. Add a test `.h` in `utests/` with `int launch()` to exercise it
