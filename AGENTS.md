# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

FusedKernelLibrary (FKL) is a header-only C++20 library for automatic GPU kernel fusion (Vertical, Horizontal, Backwards Vertical, and Divergent Horizontal Fusion), with CPU (single-thread and multithreaded), CUDA and HIP backends. Only nvcc is supported as the CUDA compiler. `main` is the development branch (v0.2.0, API may break); `LTS-C++17` is the frozen-API branch.

Deep per-topic docs live in `.github/skills/*/SKILL.md` (implementing operations, implementing DPPs, fusion techniques, data structures, using the library, language bindings, build/test). Read the relevant skill before nontrivial work. `.github/copilot-instructions.md` overlaps with them but contains stale claims — see "Stale documentation" below.

## Build and test commands

Requires CMake >= 3.28, a C++20 host compiler, and CUDA (nvcc) or HIP (amdclang). Project policy and CI treat CUDA or HIP as required, but CMake degrades gracefully: without nvcc or amdclang it configures CPU-only and generates only `*_cpp` targets.

Clone with `git clone --recursive` (or run `git submodule update --init` in an existing clone): the multithreaded CPU backend needs the `thread-pool` submodule. Without it the backend is **silently off** (a STATUS line only) and `utest_cpu_multithread_cpp` still passes, printing that it tested nothing; check the configure output for `CPU_MULTITHREAD backend (ENABLE_CPU_MULTITHREAD): ON`. CI checks out submodules and configures with `-DENABLE_CPU_MULTITHREAD=ON`, so a missing submodule fails the configure instead of silently skipping the backend's tests.

```bash
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release -S .   # configure
cmake --build build --config Release                       # build everything
cd build && ctest --build-config Release                   # run all tests (the merge gate)
```

Single test — each test header generates an executable target whose name equals its ctest name:

```bash
cmake --build build --target utest_softmax_cu   # build just one test
./build/bin/utest_softmax_cu                     # run directly (Ninja puts binaries in build/bin/)
ctest -R '^utest_softmax_cu$'                    # or via ctest, from build/
ctest -R '_cpp$'                                 # CPU-backend tests only (no GPU needed at runtime)
```

Key CMake options: `ENABLE_CPU` (ON), `ENABLE_CPU_MULTITHREAD` (ON if the `thread-pool` submodule is checked out, OFF otherwise; forcing it ON without the submodule is a `FATAL_ERROR`; when ON the FKL target defines `FK_CPU_MULTITHREAD_ENABLED`, adds `thread-pool/include` (installed as `include/fused_kernel/thread_pool`) and links `Threads::Threads`), `ENABLE_CUDA` (ON if nvcc found), `ENABLE_HIP` (ON if amdclang is found), `BUILD_TEST` (ON), `BUILD_UTEST` (ON), `ENABLE_BENCHMARK` (OFF), `CUDA_ARCH` ("native", passed verbatim to `CUDA_ARCHITECTURES` — no arch filtering exists), `HIP_ARCH` (`gfx1200;gfx1201`, RX 9060 and RX 9070, which can't load each other's code objects; it seeds `CMAKE_HIP_ARCHITECTURES`, which wins when passed explicitly, and `hip::device` follows it through `GPU_TARGETS`), `ARCH_FLAGS` (CPU SIMD), `ENABLE_NVTX`, `ENABLE_DEBUG`, `TEMPLATE_DEPTH` (1000).

`cmake --install build --prefix <dir>` installs the headers (`include/fused_kernel`, plus `include/fused_kernel/thread_pool` with the backend) and the `FKL` package (`share/cmake/FKL`): consumers use `find_package(FKL)` and link `FKL::FKL`, as with `add_subdirectory`. `FKL::FKL` requires C++20 with the `cxx_std_20` compile feature, which CMake also applies to CUDA and HIP sources that don't set `CUDA_STANDARD`/`HIP_STANDARD` (never add `cuda_std_20`/`hip_std_20`: with `add_subdirectory` from a C++-only project they fail with "No known features for CUDA compiler"); the project using FKL must enable CXX, and both the root `CMakeLists.txt` (as a subproject) and the package configuration stop with a clear message otherwise. Without `-DCMAKE_INSTALL_PREFIX` the install goes to `<build>/install`. `lib/export/` is not the installed package: it is the package that projects with FKL as a git submodule (cvGPUSpeedup, FastNPP) find without building FKL, and configure keeps its version file up to date; don't remove it while they use it. The package configuration template is `cmake/FKLConfig.cmake.in`, which `find_dependency(Threads)` when the backend is enabled; the version file accepts the same minor version before 1.0 (`find_package(FKL 0.2)` finds only 0.2.x) and the same major version from 1.0 on.

There is no lint/format gate. Format manually with `clang-format -i` using the repo-root `.clang-format` (LLVM base, 4-space indent, 120-char lines, `PointerAlignment: Right`). The merge gate is the full ctest suite building and passing across CI's compiler matrix (no `-Werror` anywhere, though the skills' PR checklists ask for warning-clean nvcc and clang builds). CI runs only on PRs to `main` (self-hosted runners): Linux amd64/arm64 with g++-13 + CUDA 13.4; g++13 and  clang++-23 for CPU backend;  Windows with cl 14.44 + CUDA 13.0, cl 14.51 + CUDA 13.4, and clang-cl 14.51 or cl 14.51 for CPU builds.

## Test infrastructure (no framework — no GTest/Catch2)

- Tests are header files auto-discovered by CMake (`cmake/tests/discover_tests.cmake`, GLOB_RECURSE with CONFIGURE_DEPENDS) inside immediate **subdirectories** of `tests/` and `utests/`. Top-level files (`tests/main.h`, `tests/operation_test_utils.h`) are never tests. Paths containing `_common` are excluded (shared helpers). Adding a new `.h` test requires no CMake edits.
- Every test header defines `int launch()` returning 0 for pass. A generated launcher TU compiles the header twice: as C++ (`<name>_cpp` target, `ParArch::CPU`) and as CUDA (`<name>_cu`, `ParArch::GPU_NVIDIA`). The same source targets both backends because `defaultParArch` switches on `__NVCC__`.
- Backend suppression is a raw **substring** search: any occurrence of `ONLY_CU` / `ONLY_CPU` / `ONLY_GPU` anywhere in the file (even in a comment) suppresses targets: `ONLY_CU` keeps only `_cu`, `ONLY_CPU` keeps only `_cpp`, `ONLY_GPU` keeps `_cu` and `_hip` (no `_cpp`). A HIP target (`_hip`) is generated only when neither `ONLY_CU` nor `ONLY_CPU` appears. Repo convention: `#define __ONLY_CU__` / `#define __ONLY_GPU__`.
- Test harness helpers (`START_ADDING_TESTS`, `STOP_ADDING_TESTS`, `RUN_ALL_TESTS`, `TestCaseBuilder`, the `testCases` map) live in `tests/operation_test_utils.h`; `tests/main.h` only declares `launch()`.
- Every public alias and `build()` overload needs a utest that instantiates it — template code only breaks on instantiation (see issue #244: shipped aliases that never compiled).
- Test targets link `FKL::FKL`, so with `ENABLE_CPU_MULTITHREAD=ON` both `_cpp` and `_cu` get `FK_CPU_MULTITHREAD_ENABLED`. CPU_MULTITHREAD coverage goes in each DPP family's existing test under `#ifdef FK_CPU_MULTITHREAD_ENABLED` (it must also compile under nvcc): run every case through the `Stream_<ParArch::CPU_MULTITHREAD>&` entry point with host memory, `sync()`, and require the output to match `ParArch::CPU`, plus the oracle: exactly (`memcmp`/`==`) for integer outputs and float pipelines without multiply-add, within a tight tolerance (e.g. 1e-5 relative) for float pipelines that multiply and add (fused multiply-add contraction). A test that means "GPU" must say `PA == ParArch::GPU_NVIDIA || PA == ParArch::GPU_AMD`, not `PA != ParArch::CPU`. The backend itself is tested by `utests/core/execution_model/utest_cpu_multithread.h` (`_cpp` only).

## Architecture

### The Operation / IOp / DPP / Executor model

- **Operation**: a stateless static-only struct (`FK_STATIC_STRUCT` deletes all ctors) that is always **single-thread** code. Anything needing thread cooperation (shared memory, `__syncthreads`, shuffles) belongs in a **DPP** (Data Parallel Pattern), never in an Operation.
- **IOp (InstantiableOperation)** = Operation type + runtime params, produced by `Op::build(...)`. Wrappers `Read<Op>`, `ReadBack<Op>`, `Unary<Op>`, `Binary<Op>`, `Ternary<Op>`, `MidWrite<Op>`, `Write<Op>` are defined in `core/execution_model/operation_model/instantiable_operations.h`. **Types define the kernel; `build()` values are runtime parameters** — changing a value never creates a new kernel, changing a type does.
- The 11 OperationTypes and their exact `exec()` signatures are documented authoritatively in the comment table of `core/execution_model/operation_model/operation_types.h` (ReadType, WriteType, UnaryType, BinaryType, ReadBackType, TernaryType, MidWriteType, Incomplete*/Open/Closed). `IncompleteTernaryType` is declared but unused — do not implement against it.
- **Entry point**: `fk::executeOperations<TransformDPP<>>(stream, readIOp, computeIOps..., writeIOp)` via `#include <fused_kernel/fused_kernel.h>`. First IOp must be a complete Read, last a Write; each op's `OutputType` must equal the next op's `InputType`.
- **Fusion machinery**: vertical fusion is an `operator|` fold over IOps in `core/execution_model/data_parallel_patterns.h`; host-side fusion (`operator&`, `.then()`) is `Fuser` and backwards vertical fusion is `BackFuser::fuse_back` in `operation_model/iop_fuser.h`. Note `fuse_back` returns a `Tuple` of IOps (callers unpack with `get<0>`), and fusing onto an IncompleteReadBack continuation (Crop/Resize) completes that op's own IOp — a `FusedOperation` (`fused_operation.h`) is only produced for non-ReadBack continuations. Horizontal fusion: passing `std::array` params to `build()` produces `BatchRead`/`BatchWrite` (`batch_operations.h`), one grid z-plane per batch item.
- **Grid sizing**: comes from the first read IOp's `getActiveThreads()` — the OUTPUT geometry (for a ReadBack stack, the last ReadBack), not the input size. Block size heuristics live in `core/execution_model/executors.h`.
- **Executors/backends**: the implemented backends are `ParArch::CPU` (single thread), `ParArch::CPU_MULTITHREAD` (optional, see below), `ParArch::GPU_NVIDIA` (nvcc) and `ParArch::GPU_AMD` (HIP) (`executors.h`, `parallel_architectures.h`). `TransformDPP` and `DivergentBatchTransformDPP` (Divergent HF) have executors for all of them. `Stream` = `Stream_<defaultParArch>` (GPU_NVIDIA under nvcc, GPU_AMD under hipcc, CPU otherwise — never CPU_MULTITHREAD); a default-constructed GPU stream owns its cudaStream_t, constructing from an existing `cudaStream_t` is non-owning.
- **Thread fusion** (vectorized loads, `core/execution_model/thread_fusion.h`): opt-in via `TransformDPP<PA, TF::ENABLED>`; requires both first read and last write to have `THREAD_FUSION=true`. Any ReadBack (Crop/Resize) or fused read chain silently disables it.

### ParArch::CPU_MULTITHREAD (multithreaded CPU backend)

- **Where**: everything is guarded by `#if defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)` and lives outside any `__NVCC__`/`__HIPCC__` block (it also compiles in `_cu` TUs). `core/execution_model/cpu_multithread.h` is only an umbrella (it `#error`s without the define); the pieces are `cpu_thread_pool.h` (`fk::cpuThreadPool()`, one shared pool with a worker per hardware thread; `cpu_multithread::Completion`; `cpu_multithread::enqueueChunks`), `stream.h` (`Stream_<ParArch::CPU_MULTITHREAD>`, `cpu_multithread::keepAlive`), `data_parallel_patterns.h` and `executors.h` (TransformDPP, DivergentBatchTransformDPP); the image-processing DPPs have their CPU_MULTITHREAD specialization and `Stream_<ParArch::CPU_MULTITHREAD>&` entry-function overloads in their own headers. `fused_kernel.h` already reaches the core pieces when the define is set.
- **Layering rule (repo owner, mandatory)**: the stream only orders executions and never touches the pool; a DPP never sees a stream. Like a GPU kernel, the DPP's `exec(const cpu_multithread::Completion& done, details, iOps...)` splits its work into chunks, enqueues them with `cpu_multithread::enqueueChunks`, and calls `done` once when all of it has finished. Only executors/entry functions use the stream: `stream.enqueue([=](const cpu_multithread::Completion& done) { DPP::exec(done, details, iOps...); })`. The CPU_MULTITHREAD DPP runs the CPU DPP's public per-row function (`exec_row`, or one per phase for a multi-phase DPP, like MorphologyDPP's `load_row`/`reduce_row`/`reduce_write_row`), so its results match `ParArch::CPU`: integer results bit for bit; float results can differ in the last bits only, because the compiler may contract `a * b + c` into a fused multiply-add differently where the shared row code is inlined. See the DPP skill.
- **Stream semantics** (like a CUDA stream): `enqueue` returns immediately; executions of one stream run in order, different streams run concurrently on the shared pool. `sync()` waits only for executions enqueued before it and rethrows the first exception their work threw since the previous `sync()`. At most 1024 executions are pending per stream (`MAX_PENDING`; past it `enqueue` waits). Copies share the queue; the last copy's destructor waits for its executions (unsynced errors are lost). `sync()` on a pool worker runs queued chunks instead of blocking.
- **Memory**: as with cudaMemcpyAsync host buffers, everything an execution reads or writes must stay valid until it has finished (`sync()`). The `executeOperations` overloads that take `Ptr`/`Tensor`/`std::array<Ptr2D>` arguments, and `setTo`, keep those buffers alive; IOps only hold a `RawPtr` (also when `build()` got a `Ptr` or an `Image`), so buffers given to `build()`, to the filter entry points or to `CircularTensor::update` are not kept alive. Memory must be `Host` or `HostPinned`: in CUDA/HIP builds `defaultMemType` is `DeviceAndPinned`, so pass `MemType::Host` explicitly; on a CPU_MULTITHREAD stream `Ptr`/`Image` `upload`/`download` (ordered no-ops) and `setTo` throw `std::runtime_error` for device memory.

### Implementing a new Operation

Pattern (see `Mul` in `algorithms/basic_ops/arithmetic.h`, `Crop` in `algorithms/image_processing/crop.h`):
1. Private `using SelfType = MyOp<...>;` then `FK_STATIC_STRUCT(MyOp, SelfType)`.
2. CRTP parent: `using Parent = BinaryOperation<I, P, O, SelfType>;` (parents in `operation_model/parent_operations.h`).
3. A `DECLARE_*_PARENT` macro generates the typedefs and `build()` overloads. **ReadBack ops must use `DECLARE_READBACK_PARENT` from `batch_operations.h`**, not the `_BASIC` variant — the `_BASIC` one silently drops the `std::array` batch builders that enable horizontal fusion.
4. Hand-write `FK_HOST_DEVICE_FUSE OutputType exec(...)` — it must also run on the CPU backend. Compute ops need nothing else; Read/ReadBack ops additionally hand-write geometry (`num_elems_x/y/z`, `getActiveThreads`) and extra `build()` overloads the macros don't generate (e.g. Crop's `build(backIOp, rect)`).
- ReadBack ops (Crop, Resize, Warping, BorderReader) come in incomplete/complete pairs: `Crop<>` (BackIOp=NullType) is what users build with params only; `BackFuser` later calls the two-arg `build(backIOp, iOp)`.
- **Golden rule**: values users change per call (factors, rects, sizes) go in `ParamsType`; anything that changes generated code (dtype, channel count, batch size, interpolation mode) is a template parameter.
- Ops are grouped thematically per file (`arithmetic.h` holds Add/Sub/Mul/Div); the same op name is specialized on a trailing InstanceType tag rather than renamed. Note `SaturateCast` lives in `image_processing/saturate.h`, not `basic_ops/`.
- Umbrella headers intentionally omit files: `basic_ops.h` omits `memory_operations.h`; `algorithms.h` omits `attention/` entirely. `core/execution_model/cpu_multithread.h` is the umbrella of the CPU_MULTITHREAD backend (its pieces are in `cpu_thread_pool.h`, `stream.h`, `data_parallel_patterns.h`, `executors.h`). Header guards are `#ifndef` style (usually `FK_*`), never `#pragma once`.

### Namespaces, macros, data types

- `fk::` for nearly everything. `cxp::` is strictly reserved for `core/constexpr_libs/` — std functionality that is unavailable on GPU, not constexpr, or both.
- Function macros from `core/utils/utils.h` branch on `__NVCC__` / `NVRTC_COMPILER` / plain C++: `FK_HOST_DEVICE_FUSE` (= `__host__ __device__ __forceinline__ static constexpr`), `FK_DEVICE_FUSE`, `FK_HOST_FUSE`, etc. Naming: `FUSE` = static constexpr, `CNST` = constexpr non-static, `STATIC` = static non-constexpr. `gpuErrchk(expr)` throws `std::runtime_error` (only defined under `__NVCC__`).
- GPU-compatible std replacements: `fk::Tuple`/`fk::apply`/`fk::apply_d` in `core/data/tuple.h`, `fk::Array` in `core/data/array.h`, vector-type traits (`VBase`, `cn<>`, `make_`, `make_set`) and channel-wise operators in `core/utils/vector_utils.h`. CPU-only builds still get `uchar3`/`float4` etc. via `core/data/vector_types.h`.
- `Ptr<ND,T>` (`core/data/ptr_nd.h`) is a ref-counted owner — copies are shallow and share memory; wrapping an external pointer is non-owning (FKL never frees it). Kernels only ever see the POD `RawPtr<ND,T>`; host code passes `container.ptr()` into `build()`. Default `MemType` under NVCC is `DeviceAndPinned` (device buffer + pinned host mirror, moved with `.upload(stream)`/`.download(stream)`); `Host` in CPU-only builds. `ND::_3D` dims are width × height × planes × color_planes (planes = batch). `Tensor<T>` forces contiguous pitch.

### attention/ — GPU-only exception to the rules

`algorithms/attention/` (softmax, flash_attention, flash_attention_mma, flash_decode) contains self-contained cooperative DPPs (whole kernels: online softmax, FlashAttention-2 SIMT and mma.sync tensor-core variants, split-KV FlashDecoding), not composable Operations. The DPP structs are inside `#if defined(__NVCC__) || defined(__HIPCC__)` (softmax, flash_attention, flash_decode; their entry points take `Stream_<defaultParArch>&`) or `#if defined(__NVCC__)` (flash_attention_mma: PTX `mma.sync`, CUDA only), and remain **GPU-only: no `ParArch::CPU` nor `ParArch::CPU_MULTITHREAD` implementation**, so their tests use `__ONLY_GPU__`. The HIP DPPs need wave32 (`attention_wave.h`: `ATTN_WAVE_SIZE`, `attnShflXor`, `attnRequireWave32`). FP8 is OCP e4m3 on both backends (`attn_fp8_e4m3` = `__nv_fp8_e4m3` / `__hip_fp8_e4m3`, decoded by `attnFp8ToF32`; `FK_HAS_FP8` when the header exists), bf16 is `attn_bf16`. This is a documented exception, decided by the repo owner, to the DPP skill's rule that every DPP has both; it is not a precedent for other DPPs. The directory also holds composable CPU-compiling pieces outside the guard (e.g. `Int8TokenDequantRead`, a normal Read Operation usable in any FKL pipeline, plus host packing helpers). Nothing in `attention/` is included by any umbrella header. Cooperative exec bodies can't be constexpr (shared memory + barriers), so softmax and flash_attention use `FK_COOP_DEVICE_FUSE` while the mma and decode variants hand-write `static __device__ void exec`. Fusion happens via Read-IOp prologues (dequantization runs in-register at load time) and IOp-chain epilogues. `DivergentBatchTransformDPP` (Divergent HF) is not an exception: it has `ParArch::CPU` and `ParArch::CPU_MULTITHREAD` DPPs and executors.

### Decoding common compile failures

- Read nvcc/clang template errors **bottom-up**: the last "instantiation of" frame names your line; the first error names the real culprit.
- `qualifiers dropped in binding reference of type 'X&&'` → explicit template args on a forwarding-reference helper (never call `BackFuser::fuse_back<Explicit...>`; let deduction happen).
- `name followed by "::" must be a class or namespace name` inside `fused_operation.h` → a raw Operation was passed where an IOp was expected; wrap it (`Unary<...>`, `Binary<...>`).
- In DPP code, pass the **whole IOp** to `IOp::Operation::exec(thread, ..., iop)` — passing `iop.params` routes to the wrong overload and fails to compile.

## Stale documentation — do not trust these claims

- `CLANG_HOST_DEVICE` does not exist; `core/utils/compiler_macros.h` defines only `_MSC_VER_EXISTS` (docs referencing it are stale).
- No CUDA arch filtering / `nvidia-smi` detection / compute_70 minimum exists in cmake — `CUDA_ARCH` is passed verbatim (copilot-instructions claim is stale).
- `ENABLE_CPU` is never auto-disabled for old MSVC.
- The README example includes `<fused_kernel/core/execution_model/memory_operations.h>` — the real path is `include/fused_kernel/algorithms/basic_ops/memory_operations.h`.
- The README, copilot-instructions and skills only mention `attention/` as the GPU-only DPP exception; this file is its only real documentation.
