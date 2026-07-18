# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

FusedKernelLibrary (FKL) is a header-only C++20 library for automatic GPU kernel fusion (Vertical, Horizontal, Backwards Vertical, and Divergent Horizontal Fusion), with CPU and CUDA backends. Only nvcc is supported as the CUDA compiler. `main` is the development branch (v0.2.0, API may break); `LTS-C++17` is the frozen-API branch.

Deep per-topic docs live in `.github/skills/*/SKILL.md` (implementing operations, implementing DPPs, fusion techniques, data structures, using the library, language bindings, build/test). Read the relevant skill before nontrivial work. `.github/copilot-instructions.md` overlaps with them but contains stale claims — see "Stale documentation" below.

## Build and test commands

Requires CMake >= 3.28, a C++20 host compiler, and CUDA >= 13.3 (nvcc). Project policy and CI treat CUDA as required, but CMake degrades gracefully: without nvcc it configures CPU-only and generates only `*_cpp` targets (an nvcc older than 13.3 is rejected with a FATAL_ERROR at configure time and a `#error` in `core/utils/utils.h`; `-DFK_ALLOW_OLDER_CUDA=ON` bypasses both for newer header-only CCCL >= 3.3 on an older toolkit).

```bash
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release -S .   # configure
cmake --build build --config Release                       # build everything (slow: ~50 nvcc TUs)
cd build && ctest --build-config Release                   # run all tests (the merge gate)
```

Single test — each test header generates an executable target whose name equals its ctest name:

```bash
cmake --build build --target utest_softmax_cu   # build just one test
./build/bin/utest_softmax_cu                     # run directly (Ninja puts binaries in build/bin/)
ctest -R '^utest_softmax_cu$'                    # or via ctest, from build/
ctest -R '_cpp$'                                 # CPU-backend tests only (no GPU needed at runtime)
```

Key CMake options: `ENABLE_CPU` (ON), `ENABLE_CUDA` (ON if nvcc found), `BUILD_TEST` (ON), `BUILD_UTEST` (ON), `ENABLE_BENCHMARK` (OFF), `CUDA_ARCH` ("native", passed verbatim to `CUDA_ARCHITECTURES` — no arch filtering exists), `ARCH_FLAGS` (CPU SIMD), `ENABLE_NVTX`, `ENABLE_DEBUG`, `TEMPLATE_DEPTH` (1000).

There is no lint/format gate. Format manually with `clang-format -i` using the repo-root `.clang-format` (LLVM base, 4-space indent, 120-char lines, `PointerAlignment: Right`). The merge gate is the full ctest suite building and passing across CI's compiler matrix (no `-Werror` anywhere, though the skills' PR checklists ask for warning-clean nvcc and clang builds). CI runs only on PRs to `main` (self-hosted runners): Linux amd64/arm64 with g++-13 and clang++-21 + CUDA 13.3; Windows with cl 14.44 + CUDA 13.3, cl 14.51 + CUDA 13.3, and clang-cl 14.51 + CUDA 13.3.

## Test infrastructure (no framework — no GTest/Catch2)

- Tests are header files auto-discovered by CMake (`cmake/tests/discover_tests.cmake`, GLOB_RECURSE with CONFIGURE_DEPENDS) inside immediate **subdirectories** of `tests/` and `utests/`. Top-level files (`tests/main.h`, `tests/operation_test_utils.h`) are never tests. Paths containing `_common` are excluded (shared helpers). Adding a new `.h` test requires no CMake edits.
- Every test header defines `int launch()` returning 0 for pass. A generated launcher TU compiles the header twice: as C++ (`<name>_cpp` target, `ParArch::CPU`) and as CUDA (`<name>_cu`, `ParArch::GPU_NVIDIA`). The same source targets both backends because `defaultParArch` switches on `__NVCC__`.
- Backend suppression is a raw **substring** search: any occurrence of `ONLY_CU` / `ONLY_CPU` anywhere in the file (even in a comment) suppresses the `_cpp` / `_cu` target. Repo convention: `#define __ONLY_CU__`.
- Test harness helpers (`START_ADDING_TESTS`, `STOP_ADDING_TESTS`, `RUN_ALL_TESTS`, `TestCaseBuilder`, the `testCases` map) live in `tests/operation_test_utils.h`; `tests/main.h` only declares `launch()`.
- Every public alias and `build()` overload needs a utest that instantiates it — template code only breaks on instantiation (see issue #244: shipped aliases that never compiled).

## Architecture

### The Operation / IOp / DPP / Executor model

- **Operation**: a stateless static-only struct (`FK_STATIC_STRUCT` deletes all ctors) that is always **single-thread** code. Anything needing thread cooperation (shared memory, `__syncthreads`, shuffles) belongs in a **DPP** (Data Parallel Pattern), never in an Operation.
- **IOp (InstantiableOperation)** = Operation type + runtime params, produced by `Op::build(...)`. Wrappers `Read<Op>`, `ReadBack<Op>`, `Unary<Op>`, `Binary<Op>`, `Ternary<Op>`, `MidWrite<Op>`, `Write<Op>` are defined in `core/execution_model/operation_model/instantiable_operations.h`. **Types define the kernel; `build()` values are runtime parameters** — changing a value never creates a new kernel, changing a type does.
- The 11 OperationTypes and their exact `exec()` signatures are documented authoritatively in the comment table of `core/execution_model/operation_model/operation_types.h` (ReadType, WriteType, UnaryType, BinaryType, ReadBackType, TernaryType, MidWriteType, Incomplete*/Open/Closed). `IncompleteTernaryType` is declared but unused — do not implement against it.
- **Entry point**: `fk::executeOperations<TransformDPP<>>(stream, readIOp, computeIOps..., writeIOp)` via `#include <fused_kernel/fused_kernel.h>`. First IOp must be a complete Read, last a Write; each op's `OutputType` must equal the next op's `InputType`.
- **Fusion machinery**: vertical fusion is an `operator|` fold over IOps in `core/execution_model/data_parallel_patterns.h`; host-side fusion (`operator&`, `.then()`) is `Fuser` and backwards vertical fusion is `BackFuser::fuse_back` in `operation_model/iop_fuser.h`. Note `fuse_back` returns a `Tuple` of IOps (callers unpack with `get<0>`), and fusing onto an IncompleteReadBack continuation (Crop/Resize) completes that op's own IOp — a `FusedOperation` (`fused_operation.h`) is only produced for non-ReadBack continuations. Horizontal fusion: passing `std::array` params to `build()` produces `BatchRead`/`BatchWrite` (`batch_operations.h`), one grid z-plane per batch item.
- **Grid sizing**: comes from the first read IOp's `getActiveThreads()` — the OUTPUT geometry (for a ReadBack stack, the last ReadBack), not the input size. Block size heuristics live in `core/execution_model/executors.h`.
- **Executors/backends**: `ParArch::CPU` and `ParArch::GPU_NVIDIA` are the only implemented backends (`executors.h`, `parallel_architectures.h`). `Stream` = `Stream_<defaultParArch>`; default-constructed owns its cudaStream_t, constructing from an existing `cudaStream_t` is non-owning.
- **Thread fusion** (vectorized loads, `core/execution_model/thread_fusion.h`): opt-in via `TransformDPP<PA, TF::ENABLED>`; requires both first read and last write to have `THREAD_FUSION=true`. Any ReadBack (Crop/Resize) or fused read chain silently disables it.

### Implementing a new Operation

Pattern (see `Mul` in `algorithms/basic_ops/arithmetic.h`, `Crop` in `algorithms/image_processing/crop.h`):
1. Private `using SelfType = MyOp<...>;` then `FK_STATIC_STRUCT(MyOp, SelfType)`.
2. CRTP parent: `using Parent = BinaryOperation<I, P, O, SelfType>;` (parents in `operation_model/parent_operations.h`).
3. A `DECLARE_*_PARENT` macro generates the typedefs and `build()` overloads. **ReadBack ops must use `DECLARE_READBACK_PARENT` from `batch_operations.h`**, not the `_BASIC` variant — the `_BASIC` one silently drops the `std::array` batch builders that enable horizontal fusion.
4. Hand-write `FK_HOST_DEVICE_FUSE OutputType exec(...)` — it must also run on the CPU backend. Compute ops need nothing else; Read/ReadBack ops additionally hand-write geometry (`num_elems_x/y/z`, `getActiveThreads`) and extra `build()` overloads the macros don't generate (e.g. Crop's `build(backIOp, rect)`).
- ReadBack ops (Crop, Resize, Warping, BorderReader) come in incomplete/complete pairs: `Crop<>` (BackIOp=NullType) is what users build with params only; `BackFuser` later calls the two-arg `build(backIOp, iOp)`.
- **Golden rule**: values users change per call (factors, rects, sizes) go in `ParamsType`; anything that changes generated code (dtype, channel count, batch size, interpolation mode) is a template parameter.
- Ops are grouped thematically per file (`arithmetic.h` holds Add/Sub/Mul/Div); the same op name is specialized on a trailing InstanceType tag rather than renamed. Note `SaturateCast` lives in `image_processing/saturate.h`, not `basic_ops/`.
- Umbrella headers intentionally omit files: `basic_ops.h` omits `memory_operations.h`; `algorithms.h` omits `attention/` entirely. Header guards are `#ifndef` style (usually `FK_*`), never `#pragma once`.

### Namespaces, macros, data types

- `fk::` for nearly everything. `cxp::` is strictly reserved for `core/constexpr_libs/` — std functionality that is unavailable on GPU, not constexpr, or both.
- Function macros from `core/utils/utils.h` branch on `__NVCC__` / `NVRTC_COMPILER` / plain C++: `FK_HOST_DEVICE_FUSE` (= `__host__ __device__ __forceinline__ static constexpr`), `FK_DEVICE_FUSE`, `FK_HOST_FUSE`, etc. Naming: `FUSE` = static constexpr, `CNST` = constexpr non-static, `STATIC` = static non-constexpr. `gpuErrchk(expr)` throws `std::runtime_error` (only defined under `__NVCC__`).
- GPU-compatible std replacements: `fk::Tuple`/`fk::apply`/`fk::apply_d` in `core/data/tuple.h`, `fk::Array` in `core/data/array.h`, vector-type traits (`VBase`, `cn<>`, `make_`, `make_set`) and channel-wise operators in `core/utils/vector_utils.h`. CPU-only builds still get `uchar3`/`float4` etc. via `core/data/vector_types.h`.
- `Ptr<ND,T>` (`core/data/ptr_nd.h`) is a ref-counted owner — copies are shallow and share memory; wrapping an external pointer is non-owning (FKL never frees it). Kernels only ever see the POD `RawPtr<ND,T>`; host code passes `container.ptr()` into `build()`. Default `MemType` under NVCC is `DeviceAndPinned` (device buffer + pinned host mirror, moved with `.upload(stream)`/`.download(stream)`); `Host` in CPU-only builds. `ND::_3D` dims are width × height × planes × color_planes (planes = batch). `Tensor<T>` forces contiguous pitch.

### attention/ — GPU-only exception to the rules

`algorithms/attention/` (softmax, flash_attention, flash_attention_mma, flash_decode) contains self-contained cooperative DPPs (whole kernels: online softmax, FlashAttention-2 SIMT and mma.sync tensor-core variants, split-KV FlashDecoding), not composable Operations. The DPP structs are inside `#if defined(__NVCC__)` with **no CPU implementation** (unlike the DPP skill's stated rule), though the directory also holds composable CPU-compiling pieces outside the guard (e.g. `Int8TokenDequantRead`, a normal Read Operation usable in any FKL pipeline, plus host packing helpers). Nothing in `attention/` is included by any umbrella header. Cooperative exec bodies can't be constexpr (shared memory + barriers), so softmax and flash_attention use `FK_COOP_DEVICE_FUSE` while the mma and decode variants hand-write `static __device__ void exec`. Fusion happens via Read-IOp prologues (dequantization runs in-register at load time) and IOp-chain epilogues. `DivergentBatchTransformDPP` (Divergent HF) is likewise GPU-only in practice: no CPU executor specialization exists and its base class hardcodes `ParArch::GPU_NVIDIA`.

### Decoding common compile failures

- Read nvcc/clang template errors **bottom-up**: the last "instantiation of" frame names your line; the first error names the real culprit.
- `qualifiers dropped in binding reference of type 'X&&'` → explicit template args on a forwarding-reference helper (never call `BackFuser::fuse_back<Explicit...>`; let deduction happen).
- `name followed by "::" must be a class or namespace name` inside `fused_operation.h` → a raw Operation was passed where an IOp was expected; wrap it (`Unary<...>`, `Binary<...>`).
- In DPP code, pass the **whole IOp** to `IOp::Operation::exec(thread, ..., iop)` — passing `iop.params` routes to the wrong overload and fails to compile.

## Stale documentation — do not trust these claims

- `CLANG_HOST_DEVICE` does not exist; `core/utils/compiler_macros.h` defines only `_MSC_VER_EXISTS` (docs referencing it are stale).
- `ENABLE_CPU` is never auto-disabled for old MSVC.
- The README example includes `<fused_kernel/core/execution_model/memory_operations.h>` — the real path is `include/fused_kernel/algorithms/basic_ops/memory_operations.h`.
- `attention/` is absent from all skills, README, and copilot-instructions.
