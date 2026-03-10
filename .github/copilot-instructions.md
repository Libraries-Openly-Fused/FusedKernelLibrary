# Copilot Instructions for FusedKernelLibrary

## Project Overview

**FusedKernelLibrary (FKL)** is a C++20 header-only library that enables automatic GPU kernel fusion without requiring CUDA expertise. It implements four fusion techniques:
- **Vertical Fusion**: Chain operations into a single kernel with no intermediate memory writes.
- **Horizontal Fusion**: Process multiple data planes in parallel using `blockIdx.z`.
- **Backwards Vertical Fusion**: Read-backwards through a pipeline (like OpenCV Filters), computing only required pixels.
- **Divergent Horizontal Fusion**: Execute different kernels simultaneously using different SM components.

The library has CPU and CUDA backends. HIP support is architecturally possible but not yet implemented.

**License**: Apache 2.0  
**Version**: 0.2.0 (main branch — API may break for maintainability)  
**LTS branch**: `LTS-C++17` (frozen API, C++17 minimum, adds features cautiously)

---

## Repository Layout

```
FusedKernelLibrary/
├── .clang-format               # LLVM-based style, 4-space indent, 120-char column limit
├── .github/workflows/          # CI: cmake-linux-amd64.yml, cmake-linux-arm64.yml, cmake-windows-amd64.yml
├── CMakeLists.txt              # Root build (v0.2.0, requires CMake >= 3.24, C++ and optional CUDA)
├── cmake/                      # CMake helpers: arch flags, CUDA init, test discovery, generators
│   ├── archflags.cmake         # CPU SIMD flags (AVX2 default on MSVC x64, native on Unix)
│   ├── cmake_init.cmake        # Global CMake settings
│   ├── cuda_init.cmake         # CUDA language enablement and NVCC path (Ninja/Windows workaround)
│   ├── libs/cuda/archs.cmake   # CUDA arch selection/filtering (requires compute_70+ for CUDA < 13)
│   └── tests/                  # Test discovery and stub generation
│       ├── discover_tests.cmake
│       └── add_generated_test.cmake
├── include/fused_kernel/       # All public headers (header-only library)
│   ├── fused_kernel.h          # Main API entry point (includes executors.h)
│   ├── algorithms/             # Operations: arithmetic, cast, image processing, etc.
│   └── core/                   # Infrastructure: execution model, data types, utils
│       ├── execution_model/    # Executors, DPP patterns, operation model
│       ├── data/               # Ptr, Ptr2D, Tensor, RawPtr types
│       └── utils/              # Macros (utils.h), compiler detection (compiler_macros.h)
├── lib/                        # CMake library target definition and version config
├── tests/                      # Integration tests (header .h files, auto-discovered)
├── utests/                     # Unit tests (header .h files, auto-discovered)
└── benchmarks/                 # Benchmarks (disabled by default, ENABLE_BENCHMARK=ON)
```

---

## Build System

### Requirements
- **CMake** >= 3.24
- **C++ compiler** with C++20 support
- **CUDA** (optional): requires NVCC. **Only nvcc is supported as the CUDA compiler**; clang-as-CUDA-compiler is not supported despite `CLANG_HOST_DEVICE` macro existing.
- **MSVC**: Visual Studio 2019+ (MSVC_VERSION >= 1920) required; older versions disable the CPU backend.

### Configure and Build (typical)
```bash
# Linux (Ninja)
cmake -G "Ninja" -B build -DCMAKE_BUILD_TYPE=Release -S .
cmake --build build --config Release

# Windows (Ninja, inside VS Developer Shell)
cmake -G "Ninja" -B build -DCMAKE_BUILD_TYPE=Release -S .
cmake --build build --config Release
```

### Key CMake Options
| Option | Default | Description |
|--------|---------|-------------|
| `ENABLE_CPU` | `ON` | Build CPU backend (auto-disabled for MSVC < 2019) |
| `ENABLE_CUDA` | `ON` (if NVCC found) | Build CUDA backend |
| `BUILD_TEST` | `ON` | Build integration tests |
| `BUILD_UTEST` | `ON` | Build unit tests |
| `ENABLE_BENCHMARK` | `OFF` | Build benchmark targets |
| `CUDA_ARCH` | `"native"` | CUDA architecture(s); use `"native"`, `"all"`, `"all-major"`, or explicit list |
| `ARCH_FLAGS` | `AVX2`/`native` | CPU SIMD flags (MSVC: AVX/AVX2/AVX512; Unix: native/haswell/…) |

### CUDA Architecture Notes
- **CUDA < 13**: Architectures below `compute_70` (Volta) are filtered out automatically. A GPU with compute < 70 will trigger an error.
- **`native` with CUDA < 13**: `nvidia-smi --query-gpu=compute_cap` is executed at CMake configure time to detect the local GPU.
- **CUDA >= 13**: All architectures allowed.

### Windows-Specific Notes
- CI uses self-hosted runners with LLVM 21.1.0 at `D:/clang+llvm-21.1.0-x86_64-pc-windows-msvc/bin/`.
- The Ninja generator requires a workaround: after `cmake`, `rules.ninja` may contain a wrong NVCC path that must be patched (see `cmake-windows-amd64.yml`).
- The VS Developer Shell (`Enter-VsDevShell`) must be activated for both configure and build steps on Windows.
- `utf8cp.manifest` is embedded into test executables on Windows for UTF-8 codepage support.

---

## Running Tests

```bash
cd build
ctest --build-config Release --output-junit test_results.xml
```

Tests are registered with CTest automatically. Individual targets follow the naming pattern `<TestName>_cpp` (CPU) and `<TestName>_cu` (CUDA).

---

## Test Infrastructure

### How Tests Are Discovered
Tests in `tests/` and `utests/` are **not** written with a traditional test framework. Instead:

1. Each test is a **header file** (`.h`) inside `tests/` or `utests/`.
2. CMake's `discover_tests()` function (in `cmake/tests/discover_tests.cmake`) **recursively finds all `*.h` files** in the test directories, excluding files matching `*_common.*`.
3. For each test header, a `launcher.cpp` or `launcher.cu` stub is generated from `tests/launcher.in` using `configure_file`. The stub includes the test header and calls `launch()`.
4. Two executables are generated per test file (unless restricted):
   - `<TestName>_cpp` — compiled as C++ (CPU backend)
   - `<TestName>_cu` — compiled as CUDA (if NVCC is available)
5. Use `// ONLY_CU` in a test header to suppress the `_cpp` target.
6. Use `// ONLY_CPU` in a test header to suppress the `_cu` target.

### Test File Structure
Every test header must define a `launch()` function returning `int`:
```cpp
#include <tests/main.h>
#include <fused_kernel/fused_kernel.h>
// ... other FKL headers

int launch() {
    // test code here
    return 0;  // 0 = pass, non-zero = fail
}
```

Files ending in `_common.h` (matching `*_common.*`) are shared helpers, not test entry points — they are excluded from test discovery.

---

## Code Conventions

### Namespace
All library code lives in namespace `fk`. Use `using namespace fk;` in test/example files.

### Function/Method Macros
All functions in headers use one of these macros (from `include/fused_kernel/core/utils/utils.h`):

| Macro | Expands to (NVCC mode) | Use case |
|-------|------------------------|----------|
| `FK_HOST_DEVICE_FUSE` | `__host__ __device__ __forceinline__ static constexpr` | Dual-mode static constexpr |
| `FK_HOST_DEVICE_CNST` | `__host__ __device__ __forceinline__ constexpr` | Dual-mode constexpr |
| `FK_DEVICE_FUSE` | `__device__ __forceinline__ static constexpr` | Device-only static constexpr |
| `FK_HOST_FUSE` | `__host__ __forceinline__ static constexpr` | Host-only static constexpr |
| `FK_HOST_STATIC` | `__host__ __forceinline__ static` | Host-only static inline |
| `FK_HOST_DEVICE_STATIC` | `__host__ __device__ __forceinline__ static` | Dual-mode static inline |
| `FK_RESTRICT` | `__restrict__` | Restrict pointer qualifier |

In CPU-only mode (no NVCC, no CLANG_HOST_DEVICE), these macros degrade to standard C++ equivalents.

### Compiler Macros Header
`include/fused_kernel/core/utils/compiler_macros.h` only defines `CLANG_HOST_DEVICE` (1 when clang compiles CUDA code with `__CUDA__`, 0 otherwise). This macro enables Clang host-device compilation but **nvcc remains the only supported CUDA compiler**; clang-as-CUDA-compiler is not a supported configuration.

### Static-Only Structs
The `FK_STATIC_STRUCT(StructName, StructAlias)` macro marks a struct as non-constructible and non-copyable (deletes default/copy/move constructors and assignment operators).

### Type Aliases
The library defines CUDA-compatible type aliases (also available in CPU mode):
```cpp
using uchar    = unsigned char;
using schar    = signed char;
using uint     = unsigned int;
using ushort   = unsigned short;
using ulong    = unsigned long;
using longlong = long long;
using ulonglong = unsigned long long;
```

### CUDA Error Checking
Use the `gpuErrchk(expr)` macro for CUDA API calls. It throws `std::runtime_error` on failure with file and line info.

### Code Formatting
Run `clang-format` using the `.clang-format` file at the repo root:
- Based on LLVM style
- **4-space indentation**, no tabs
- **120-character column limit**
- `PointerAlignment: Right` (e.g., `int* ptr`)
- `AlwaysBreakTemplateDeclarations: MultiLine`

---

## Core API Patterns

### Executing Fused Operations
The primary entry point is `fk::executeOperations<DPPType>(stream, op1, op2, ...)`:
```cpp
#include <fused_kernel/fused_kernel.h>
using namespace fk;

Stream stream;
executeOperations<TransformDPP<>>(stream,
    PerThreadRead<ND::_2D, uchar3>::build(inputPtr),
    Crop<>::build(crops),
    Resize<InterpolationType::INTER_LINEAR>::build(outputSize),
    SaturateCast<float3, uchar3>::build(),
    TensorWrite<uchar3>::build(output.ptr()));
stream.sync();
```

### Operation Building Pattern
Each operation type exposes a `static build(...)` factory method returning an instance containing the operation parameters. The kernel itself is encoded in the type — parameters are runtime values.

### Data Types
- `Ptr2D<T>`: 2D GPU pointer with width/height/pitch
- `Tensor<T>`: Contiguous 3D GPU tensor (width × height × batch)
- `Ptr<ND, T>`: Generic N-dimensional GPU pointer
- `Stream` / `Stream_<PAR_ARCH>`: CUDA stream wrapper

### Data Parallel Patterns (DPP)
`TransformDPP<>` is the standard choice. It drives the kernel grid/block dimensions based on the operations in the pipeline.

---

## CI / Workflow Details

All three workflow files trigger on **pull requests to `main`** (push triggers are commented out). All runners are **self-hosted**.

### Linux (cmake-linux-amd64.yml, cmake-linux-arm64.yml)
- **Compilers**: `g++-13`, `clang++-21`
- **CUDA**: 12.9, 13.2 (via `/usr/local/cuda-<version>/bin/nvcc`)
- **CMake**: Custom installation at `/home/cudeiro/cmake-4.2.1-linux-x86_64/bin/` (added to PATH)
- **Generator**: Ninja
- **Build type**: Release

### Windows (cmake-windows-amd64.yml)
- **Host compilers**: `cl` (MSVC), `clang-cl`
- **MSVC versions**: 14.44, 14.50 (via `-vcvars_ver`)
- **CUDA**: 12.9, 13.2 (NVCC at `%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v<version>\bin\nvcc.exe`)
- **LLVM**: `D:/clang+llvm-21.1.0-x86_64-pc-windows-msvc/bin/` (added to PATH)
- **Generator**: Ninja
- **Workaround**: After CMake configure, `rules.ninja` may contain an empty NVCC path that is patched with PowerShell string replacement.

### Environment Variables Set by CI
| Variable | Value |
|----------|-------|
| `CUDACXX` | Path to nvcc executable |
| `CC` | Host C compiler |
| `CXX` | Host C++ compiler |

---

## Adding a New Operation

New operations follow the same pattern as existing ones in `include/fused_kernel/algorithms/`. Each operation is a struct with:
- A `static build(...)` factory method
- A `static exec(...)` or operator that performs the computation
- Appropriate `FK_HOST_DEVICE_FUSE` / `FK_DEVICE_FUSE` function qualifiers
- Placement in the `fk` namespace

See existing operations like `Mul`, `Add`, `SaturateCast` in `include/fused_kernel/algorithms/basic_ops/` for reference patterns.

---

## Known Issues and Workarounds

1. **Windows Ninja + NVCC path**: After CMake configure on Windows with Ninja, `<build_dir>/CMakeFiles/rules.ninja` may contain an incorrect path to `nvcc.exe`. The CI workflow patches this with PowerShell `Set-Content`. If you hit this locally, check that `CUDACXX` env var is set before invoking CMake and verify the generated `rules.ninja`.

2. **CUDA < 13 + old GPUs**: If your GPU has compute capability < 7.0 (pre-Volta), building will fail. Use a newer GPU or set `CUDA_ARCH` explicitly to a supported arch.

3. **MSVC < 2019**: CPU backend is automatically disabled with a warning. The CUDA backend may still work if NVCC is available.

4. **Clang as CUDA compiler**: While `CLANG_HOST_DEVICE` macro exists and Clang can be used as a host compiler (including `clang++-21` on Linux and `clang-cl` on Windows with nvcc as the CUDA compiler), **using Clang as the CUDA compiler itself (replacing nvcc) is not supported**.
