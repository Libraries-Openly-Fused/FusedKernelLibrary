---
name: fkl-implementing-data-parallel-patterns
description: Implement a new Data Parallel Pattern (DPP) for the Fused Kernel Library. Covers the anatomy of a DPP struct, device-side invocation of IOps (the IOp-form table, reading inputs, fusing epilogues into the write), compile-time vs runtime types, vector types, the ParArch::CPU and ParArch::CPU_MULTITHREAD (thread pool) implementations, the ParArch::GPU_INTEL (SYCL) implementation, and testing. Use when adding an algorithm to FKL that requires thread coordination, when porting a DPP to another backend, or when debugging a DPP's internal device execution.
---

# Implementing FKL Data Parallel Patterns

## Anatomy of a DPP

A DPP is composed of three parts:

- Executor implementation: one `Executor<MyDPP<PA, ...>>` per back-end, as in
  include/fused_kernel/core/execution_model/executors.h and the image-processing DPPs. Users call
  `executeOperations<MyDPP<PA, DPPDetails>>(stream, details, iOps...)`; there are no per-DPP entry functions.
- For GPU implementations, the kernel is always `launchDPP_Kernel<DPP, DPPDetails, IOps...>`
  (include/fused_kernel/core/execution_model/executor_details/executor_kernels.h): it only calls
  `DPP::exec(details, iOps...)`, every parameter `const __grid_constant__`. Do not write a kernel per DPP. For CPU the
  executor calls `DPP::exec` directly; for CPU_MULTITHREAD it enqueues the DPP on the stream (see below); for the SYCL
  backend (GPU_INTEL), where there are no kernel functions, it calls the SYCL `launchDPP_Kernel<DPP>(stream, grid, ...)`
  of the same file, which enqueues the kernel that calls the DPP (see below).
- The DPP implementation.

The rule (repo owner's decision): each `executeOperations` call launches exactly ONE kernel, fixed by the template and
parameter types (only TransformDPP chooses among its thread fusion variants at run time). The details come first
after the stream. The executor checks them, and anything else it can check on the host, before it launches or enqueues
anything: invalid arguments throw `std::invalid_argument` through
`executor_detail::requireArguments(valid, "MyDPP", "what is wrong")`; executors return `void`, never a status.

```cpp
template <typename DPPDetails>
struct Executor<MyDPP<ParArch::CPU, DPPDetails>> {
private:
    using DPP = MyDPP<ParArch::CPU, DPPDetails>;
    using SelfType = Executor<DPP>;
public:
    FK_STATIC_STRUCT(Executor, SelfType)
    FK_HOST_FUSE ParArch parArch() { return ParArch::CPU; }
    template <typename InputIOp, typename OutputIOp>
    FK_HOST_STATIC void executeOperations(Stream_<ParArch::CPU>&, const DPPDetails& details,
                                          const InputIOp& input, const OutputIOp& output) {
        executor_detail::requireArguments(DPPDetails::valid(details), "MyDPP", "invalid details");
        DPP::exec(details, input, output);
    }
};

#if defined(__NVCC__) || defined(__HIPCC__)
// One GPU executor for both GPUs, constrained as the GPU DPP is
template <ParArch PA, typename DPPDetails>
    requires(PA == ParArch::GPU_NVIDIA || PA == ParArch::GPU_AMD)
struct Executor<MyDPP<PA, DPPDetails>> {
private:
    using DPP = MyDPP<PA, DPPDetails>;
    using SelfType = Executor<DPP>;
public:
    FK_STATIC_STRUCT(Executor, SelfType)
    FK_HOST_FUSE ParArch parArch() { return PA; }
    template <typename InputIOp, typename OutputIOp>
    FK_HOST_STATIC void executeOperations(Stream_<PA>& stream, const DPPDetails& details,
                                          const InputIOp& input, const OutputIOp& output) {
        executor_detail::requireArguments(DPPDetails::valid(details), "MyDPP", "invalid details");
        const dim3 block(/* ... */), grid(/* ... */);
#if defined(__HIPCC__)
        launchDPP_Kernel<DPP><<<grid, block, 0, stream.getHIPStream()>>>(details, input, output);
        gpuErrchk(hipGetLastError());
#else
        launchDPP_Kernel<DPP><<<grid, block, 0, stream.getCUDAStream()>>>(details, input, output);
        gpuErrchk(cudaGetLastError());
#endif
    }
};
#endif
```

Every DPP must always have a single thread CPU implementation using the ParArch::CPU, and a ParArch::CPU_MULTITHREAD
implementation that reuses it (see "The ParArch::CPU_MULTITHREAD implementation" below), and then it can have
implementations for other ParArchs, as in include/fused_kernel/core/execution_model/parallel_architectures.h

The only exception are the attention DPPs in include/fused_kernel/algorithms/attention/ (softmax, flash_attention,
flash_attention_mma, flash_decode): the repo owner decided that they remain GPU-only, with no ParArch::CPU nor
ParArch::CPU_MULTITHREAD implementation. Do not take them as a precedent for new DPPs.

Every DPP is a STATELESS struct: static `exec()`. The exec function will get as parameters, DPP details which are external parameters
not directly related to the data being processed, and IOps (Instantiable Operations) which will contain the implementation
of the instructions to be applied over the data being processed, along with dimensions information.

A DPP must not contain the definition of code that modifies the data (except for the case of tensor core code where it's impossible to apply that abstraction).
In order to operate on the data the DPP must use IOps that must have been passed to the DPP as exec function parameters.

Those IOps will be used by each thread to access the device input data, to operate on the data and to write the results back to device memory.
The implementation of the DPP exec function is responsible for deciding which threads will execute which IOps in which order,
as well as applying any thread synchronization or using any shared memory.

Each one of the IOps can contain a single Operation, or a FusedOperation (several consecutive Operations), or a fk::Tuple of IOps,
depending on the structure the DPP requires.

Each DPP defines the number of input ReadOperations and or ReadBackOperations it requires.
Each DPP defines the number of compute Operations (non Read/ReadBack and non Write) it requires.
Each DPP defines the number of output WriteOperations it requires.

Never pass a raw pointer (T*) to device (global) memory as an input or output parameter to exec function.
Device pointers must be passed as part of a Read/ReadBack or Write IOps.

Example TransformDPP which is a special case, because it does not require any specific structure in the IOps passed as parameters.
The IOps in this DPP will be executed one after the other by all the threads that have to participate.

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

template <typename DPPDetails>
struct TransformDPP<ParArch::GPU_NVIDIA, DPPDetails> {
private:
    using Parent = TransformDPPBase<DPPDetails>;
    using SelfType = TransformDPP<ParArch::GPU_NVIDIA, DPPDetails>;
    using Details = DPPDetails;
public:
    FK_STATIC_STRUCT(TransformDPP, SelfType)  // deletes ctors: pure static
    static constexpr ParArch PAR_ARCH = ParArch::GPU_NVIDIA;
    
    template <typename FirstIOp>
    FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const Details& details,
                                                       const FirstIOp& iOp) {
        return Parent::getActiveThreads(details, iOp);
    }

    template <typename... IOps>
    FK_DEVICE_FUSE void exec(const Details& details, const IOps&... iOps) {
        const cg::thread_block g = cg::this_thread_block();

        const int x = (g.dim_threads().x * g.group_index().x) + g.thread_index().x;
        const int y = (g.dim_threads().y * g.group_index().y) + g.thread_index().y;
        const int z = g.group_index().z; // So far we only consider the option of using the z dimension to specify n (x*y) thread planes
        const Point thread{ x, y, z };

        const ActiveThreads activeThreads = getActiveThreads(details, fk::get_arg<0>(iOps...));

        if (x < activeThreads.x && y < activeThreads.y) {
            Parent::execute_thread(thread, activeThreads, iOps...);
        }
    }
};
```

## The IOp invocation contract (The IOp-form table)

Understanding the IOp signatures is critical when debugging compiler errors or when authoring a new DPP. Every IOp's device-side `exec()` signature is fixed by its operation type:

| Operation Type | exec call site |
|---|---|
| ReadType     | `T    IOp::Operation::exec(thread, iop)` |
| WriteType    | `void IOp::Operation::exec(thread, value, iop)` |
| UnaryType    | `O    IOp::Operation::exec(input)` |
| BinaryType   | `O    IOp::Operation::exec(input, iop)` |
| ReadBackType | `O    IOp::Operation::exec(thread, iop)` |
| TernaryType  | `O    IOp::Operation::exec(input, iop)` |
| MidWriteType | `In   IOp::Operation::exec(thread, input, iop)` (writes AND forwards) |
| OpenType     | `O    IOp::Operation::exec(thread, input, iop)` |
| ClosedType   | `void IOp::Operation::exec(thread, iop)` |

Key rules for DPP authors invoking these:
- **Invoke through the IOp:** Call `IOp::Operation::exec(...)`, passing the whole `iop` instance for the data, NOT just `iop.params` or the raw op struct.
- **Read** takes the `Point thread` + `iop` and returns the value.
- **Write** takes `thread`, the `value` to store, and `iop`.
- **Unary/Binary** are pure register compute: no `thread`, just the input (+ `iop` for Binary).

## Invoking Operations Inside the DPP (Device Side)

Inside the DPP's execution logic, you must invoke the IOps passed to the `exec()` function. If the executor pre-fused a compute chain into the write operation (the epilogue), you invoke it as a single `OutputIOp` type:

```cpp
template <typename DPPDetails, typename InputIOp, typename OutputIOp>
struct MyDPP {
    // ... boilerplate ...
    
    FK_HOST_DEVICE_FUSE static void exec(const DPPDetails& details,
                                         const InputIOp& input,
                                         const OutputIOp& output) {
        
        Point thread{ /* computed from thread_index */ };

        // 1. Read the input operand
        auto val = InputIOp::Operation::exec(thread, input);
        
        // 2. ... do any DPP-specific reductions, shared memory operations, etc ...
        auto result = val; // (placeholder)
        
        // 3. Output as the fused write: invoke with the whole output IOp.
        // The epilogue compute chain runs in-register before the single global write.
        OutputIOp::Operation::exec(thread, result, output);
    }
};
```

## Pitfalls

- **Passing `iop.params` instead of `iop`:** If you are writing a DPP and pass `iop.params` to an operation's `exec()`, it will route through the wrong template fold path and fail to compile. Always pass the whole IOp wrapper.

## The ParArch::CPU_MULTITHREAD implementation

The multithreaded CPU backend runs DPP work on `fk::cpuThreadPool()`, one pool shared by all streams, with one
worker per hardware thread. It exists when `FK_CPU_MULTITHREAD_ENABLED` is defined (CMake option
`ENABLE_CPU_MULTITHREAD`, which needs the `thread-pool` submodule). Reference implementation:
`TransformDPP<ParArch::CPU>::exec_row` and `TransformDPP<ParArch::CPU_MULTITHREAD>` in
`core/execution_model/data_parallel_patterns.h`, and `Executor<TransformDPP<ParArch::CPU_MULTITHREAD, TFEN, void>>`
in `core/execution_model/executors.h`.

The layering rule (mandatory, decided by the repo owner), the same as with a GPU:

- **The stream only orders executions.** `Stream_<ParArch::CPU_MULTITHREAD>::enqueue(start)` queues an execution;
  when the previous executions of the stream have finished, it calls `start(const cpu_multithread::Completion& done)`
  on some thread. The stream never touches the pool.
- **The DPP never sees a stream.** Like a GPU kernel, when its execution starts the DPP splits its work into chunks,
  enqueues them with `cpu_multithread::enqueueChunks(numChunks, chunkFunction, onComplete)`, returns, and calls the
  `Completion` exactly once when all its work has finished (passing `done` as `onComplete` does that, forwarding the
  first exception a chunk threw).
- **Only executors use the stream**: they validate synchronously (throwing `std::invalid_argument`, before anything
  is enqueued), then `stream.enqueue(...)` a lambda that calls `DPP::exec(done, ...)`.
- **The CPU version exposes `exec_row`**: the body of its outer row loop, as a public static function used by both
  its own `exec()` and the CPU_MULTITHREAD `exec()` (a multi-phase DPP exposes one per phase instead, like
  MorphologyDPP's `load_row`/`reduce_row`/`reduce_write_row`). Both run the same per-row code, so their results match:
  integer results bit for bit; float results can differ in the last bits only, because the compiler may contract
  `a * b + c` into a fused multiply-add differently where the shared code is inlined.

```cpp
template <typename DPPDetails>
struct MyDPP<ParArch::CPU, DPPDetails> {
    // ... SelfType, FK_STATIC_STRUCT, PAR_ARCH = ParArch::CPU, validity helpers ...
    template <typename InputIOp, typename OutputIOp>
    FK_HOST_FUSE void exec_row(const int y, const DPPDetails& details,
                               const InputIOp& input, const OutputIOp& output) {
        for (int x = 0; x < details.width; ++x) { /* the per-thread code of point (x, y) */ }
    }
    template <typename InputIOp, typename OutputIOp>
    FK_HOST_FUSE void exec(const DPPDetails& details, const InputIOp& input, const OutputIOp& output) {
        for (int y = 0; y < details.height; ++y) {
            exec_row(y, details, input, output);
        }
    }
};

#if defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)  // outside any __NVCC__/__HIPCC__ block
template <typename DPPDetails>
struct MyDPP<ParArch::CPU_MULTITHREAD, DPPDetails> {
private:
    using SelfType = MyDPP<ParArch::CPU_MULTITHREAD, DPPDetails>;
    using CPUDPP = MyDPP<ParArch::CPU, DPPDetails>;
public:
    FK_STATIC_STRUCT(MyDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::CPU_MULTITHREAD;
    // + forwarders to CPUDPP's validity helpers (accepts/valid/...), if it has them

    // Enqueues one chunk per row and returns; done is called when all of them have finished
    template <typename InputIOp, typename OutputIOp>
    FK_HOST_STATIC void exec(const cpu_multithread::Completion& done, const DPPDetails& details,
                             const InputIOp& input, const OutputIOp& output) {
        cpu_multithread::enqueueChunks(static_cast<std::size_t>(details.height),
            [details, input, output](const std::size_t row) {  // BY VALUE: exec returns before the rows run
                CPUDPP::exec_row(static_cast<int>(row), details, input, output);
            }, done);
    }
};

// Its Executor, next to the CPU and GPU ones (the same arguments)
template <typename DPPDetails>
struct Executor<MyDPP<ParArch::CPU_MULTITHREAD, DPPDetails>> {
private:
    using DPP = MyDPP<ParArch::CPU_MULTITHREAD, DPPDetails>;
    using SelfType = Executor<DPP>;
public:
    FK_STATIC_STRUCT(Executor, SelfType)
    FK_HOST_FUSE ParArch parArch() { return ParArch::CPU_MULTITHREAD; }
    template <typename InputIOp, typename OutputIOp>
    FK_HOST_STATIC void executeOperations(Stream_<ParArch::CPU_MULTITHREAD>& stream, const DPPDetails& details,
                                          const InputIOp& input, const OutputIOp& output) {
        executor_detail::requireArguments(DPPDetails::valid(details), "MyDPP", "invalid details");  // as the others
        stream.enqueue([=](const cpu_multithread::Completion& done) {
            DPP::exec(done, details, input, output);
        });
    }
};
#endif
```

Multi-phase DPPs (a phase must finish before the next one starts) chain `enqueueChunks` in `onComplete`, which runs
on the worker that finished the last chunk and must not throw. Scratch memory shared by the phases goes in a
`std::shared_ptr` captured by value. See `MorphologyDPP<ParArch::CPU_MULTITHREAD>` in
`algorithms/image_processing/morphology.h`:

```cpp
cpu_multithread::enqueueChunks(rows, firstPhaseRow,
    [=](const std::exception_ptr error) noexcept {
        if (error) { done(error); return; }
        try {
            cpu_multithread::enqueueChunks(rows, lastPhaseRow, done);  // the last phase ends the execution
        } catch (...) {
            done(std::current_exception());
        }
    });
```

Rules:
- Capture everything BY VALUE (IOps, details, shared scratch); never references to the caller's locals. Both
  callables of `enqueueChunks` and the lambda given to `stream.enqueue` are stored by value.
- Call `done` exactly once (right away if there is nothing to do). If `exec` throws, it must not have enqueued
  anything nor call `done`: the stream reports the error.
- Chunks run concurrently and in any order, as GPU threads do. Document the preconditions in a comment (for instance,
  the output must not alias the input) as the GPU versions already require.
- Capture an `fk::Tuple` parameter of one element with an init-capture (`t = t`): a plain copy of a `const Tuple&`
  is a const member, and moving it (the pool and the stream move the callables) selects Tuple's forwarding
  constructor, which does not compile.
- Executors that take owning objects (`Ptr`, `Tensor`) may call `cpu_multithread::keepAlive(objects...)` before
  enqueueing, so the next execution enqueued from this thread keeps copies of them until it has finished; IOps only
  hold a `RawPtr`, so otherwise the buffers must stay valid until `sync()`.

## The ParArch::GPU_INTEL implementation (SYCL)

The SYCL backend exists in translation units compiled as SYCL (`FK_SYCL_ENABLED`, which `compiler_macros.h` defines
when the compiler predefines `SYCL_LANGUAGE_VERSION`: DPC++ with `-fsycl`). Reference implementation:
`TransformDPP<ParArch::GPU_INTEL>` in `core/execution_model/data_parallel_patterns.h`, its executor in
`core/execution_model/executors.h`, and, for a DPP with shared memory, `LinearFilterDPP<ParArch::GPU_INTEL>` in
`algorithms/image_processing/linear_filter.h`. As for CUDA, the SYCL helpers are where they are used: the launch in
`executor_details/executor_kernels.h` (`launchDPP_Kernel`, `launchDPP_KernelWithLocalMemory`), the grid in `executors.h`
(`syclGrid`, from two `CtxDim3`), and the thread position in `data_parallel_patterns.h` (`ThreadPosition`, `globalThread`).

A SYCL DPP is the CUDA one with the same per thread code, written next to it. What changes:

| CUDA | SYCL |
|---|---|
| `launchDPP_Kernel<DPP>`, the `__global__` kernel that calls `DPP::exec(details, iOps...)` | no kernel function: `launchDPP_Kernel<DPP>(stream, grid, details, iOps...)` enqueues a kernel that calls `DPP::exec(item, details, iOps...)` |
| `launchDPP_Kernel<DPP><<<grid, block, 0, stream>>>(...)` | `syclGrid(CtxDim3{gx, gy, gz}, CtxDim3{bx, by, bz})`, a `sycl::nd_range<3>` (x is dimension 2 of the range) |
| `threadIdx`, `blockIdx`, `blockDim` | `const ThreadPosition thread(item);` then `thread.threadIdx.x`, `thread.blockIdx.y`, `thread.blockDim.x`... |
| `blockDim.x * blockIdx.x + threadIdx.x` | `item.get_global_id(2)` (`globalThread(item)` gives the `Point` of a grid with one thread in z per block) |
| `__shared__ T halo[N];` inside `exec` | `launchDPP_KernelWithLocalMemory<DPP, T>(stream, grid, DPP::LOCAL_ELEMENTS, params...)`, and `exec(item, T* halo, params...)` |
| `__syncthreads()` | `sycl::group_barrier(item.get_group())` |
| `gpuErrchk(cudaGetLastError())` | nothing: errors are `sycl::exception`s, thrown when enqueueing or by `stream.sync()` |

```cpp
#if defined(FK_SYCL_ENABLED)   // next to the CUDA/HIP version, never inside its #if
template <typename DPPDetails>
struct MyDPP<ParArch::GPU_INTEL, DPPDetails> {
private:
    using SelfType = MyDPP<ParArch::GPU_INTEL, DPPDetails>;
    using T = typename DPPDetails::ValueType;
public:
    FK_STATIC_STRUCT(MyDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::GPU_INTEL;
    // Elements of each array of T in work-group local memory (the shared memory of the block)
    static constexpr std::array<std::size_t, 1> LOCAL_ELEMENTS{ DPPDetails::TILE_WIDTH * DPPDetails::TILE_HEIGHT };

    template <typename InputIOp, typename OutputIOp>
    FK_DEVICE_STATIC void exec(const sycl::nd_item<3>& item, T* const shared, const DPPDetails& details,
                               const InputIOp& input, const OutputIOp& output) {
        const ThreadPosition thread(item);
        const int tid = thread.threadIdx.y * thread.blockDim.x + thread.threadIdx.x;
        const Point p{ thread.blockIdx.x * thread.blockDim.x + thread.threadIdx.x,
                       thread.blockIdx.y * thread.blockDim.y + thread.threadIdx.y, thread.blockIdx.z };
        shared[tid] = InputIOp::Operation::exec(p, input);     // (bounds checks omitted)
        sycl::group_barrier(item.get_group());
        OutputIOp::Operation::exec(p, shared[tid], output);
    }
};

// Its Executor, next to the CPU and GPU ones (the same arguments and the same checks)
template <typename DPPDetails>
struct Executor<MyDPP<ParArch::GPU_INTEL, DPPDetails>> {
private:
    using DPP = MyDPP<ParArch::GPU_INTEL, DPPDetails>;
    using SelfType = Executor<DPP>;
public:
    FK_STATIC_STRUCT(Executor, SelfType)
    FK_HOST_FUSE ParArch parArch() { return ParArch::GPU_INTEL; }
    template <typename InputIOp, typename OutputIOp>
    FK_HOST_STATIC void executeOperations(Stream_<ParArch::GPU_INTEL>& stream, const DPPDetails& details,
                                          const InputIOp& input, const OutputIOp& output) {
        executor_detail::requireArguments(DPPDetails::valid(details), "MyDPP", "invalid details");  // as the others
        const CtxDim3 block{ DPPDetails::TILE_WIDTH, DPPDetails::TILE_HEIGHT, 1 };
        const CtxDim3 grid{ static_cast<uint>((details.width + DPPDetails::TILE_WIDTH - 1) / DPPDetails::TILE_WIDTH),
                            static_cast<uint>((details.height + DPPDetails::TILE_HEIGHT - 1) / DPPDetails::TILE_HEIGHT), 1 };
        // The one kernel of the call, the SYCL launchDPP_Kernel of a DPP with shared memory
        launchDPP_KernelWithLocalMemory<DPP, typename DPPDetails::ValueType>(
            stream, syclGrid(grid, block), DPP::LOCAL_ELEMENTS, details, input, output);
    }
};
#endif
```

A DPP without shared memory uses `launchDPP_Kernel<DPP>(stream, grid, details, iOps...)`, and its `exec` gets the
`nd_item` and then the parameters. As on the other back-ends, the executor checks its arguments before it launches
anything, and each `executeOperations` call launches one kernel.

Rules:
- `exec` is `FK_DEVICE_STATIC` (it reads the `nd_item`, so it can not be `constexpr`), and its first parameter is
  `const sycl::nd_item<3>&`. Helpers that take a thread position computed by the caller can stay `FK_DEVICE_FUSE`
  (`TransformDPP::exec_thread`).
- SYCL compiles `exec` for the host and for the device from the same code: do not copy the
  `#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)` guards of the CUDA version.
- The code a kernel calls can not throw, recurse (template recursion is fine), call through function pointers or
  virtual functions, nor use static variables: these are compile errors that name the kernel. A kernel that uses
  `double` needs a device with `sycl::aspect::fp64`.
- Everything given to `launchDPP_Kernel` (details and IOps) is copied to the device: it must be trivially copyable,
  which FKL's IOps and `fk::Tuple` are. `launchDPP_Kernel` passes them as the arguments of the kernel while they are at most
  `FK_SYCL_MAX_KERNEL_ARGUMENTS_BYTES` (1024, Intel GPUs reject more than 2048) and through device memory otherwise;
  the DPP does not see the difference.
- Local memory is requested when the kernel is enqueued, not declared in `exec`: up to 2 arrays of one type, with
  their number of elements in a `static constexpr std::array<std::size_t, N>` of the DPP (a variable template if it
  depends on the IOps, as `MorphologyDPP::LOCAL_ELEMENTS<Reducers>`).
- Every thread of a block must reach `sycl::group_barrier` the same number of times, as with `__syncthreads()`.
- The grid must be made of whole blocks (`syclGrid` does that): threads past the end of the data return, as in CUDA.

## Runtime values vs compile-time types (the golden rule)

Anything users may change per call (factors, rects, matrices, sizes) goes
in ParamsType. Anything that changes the generated code (dtype, channel
count, batch size, interpolation mode) is a template parameter. Getting
this wrong either recompiles on every value change or silently bakes
stale values into kernels.

## Vector types

Use the helpers instead of hand-rolled per-channel code:
- `VBase<T>` scalar base; `cn<T>` channel count; `VectorType_t<T, N>`.
- `make_<float3>(x, y, z)` construction; binary operators are already
  overloaded channel-wise for CUDA vector types (vector_utils.h).
- Write exec() once with `if constexpr (cn<I> == ...)` branches only when
  semantics differ per arity (see Equal, TensorSplit).

## Forwarding references in helpers

Never call `fuse_back<ExplicitArgs...>(...)`: explicit template arguments
turn `IOps&&...` into rvalue refs that cannot bind to const lvalues stored
in tuples (issue #245). Let deduction happen, e.g. through a lambda:

```cpp
apply([](const auto&... iOps) { return BackFuser::fuse_back(iOps...); }, tup);
```

## Testing a new DPP

1. Add a utest header under `utests/<area>/`.
2. Register it in the test list before `STOP_ADDING_TESTS`.
3. Under `#ifdef FK_CPU_MULTITHREAD_ENABLED`, run every case also through
   `executeOperations<MyDPP<ParArch::CPU_MULTITHREAD, ...>>` on a `Stream_<ParArch::CPU_MULTITHREAD>&` (host memory,
   then `stream.sync()`) and require the output to match the `ParArch::CPU` output, plus the
   oracle checks: exactly (`memcmp` or `==`) for integer outputs and float pipelines without multiply-add, within a
   tight tolerance (e.g. 1e-5 relative) when floats are multiplied and added (fused multiply-add contraction). The same header is compiled by nvcc for the `_cu` target, so
   that code must compile there too. A test that means "GPU" must check
   `PA == ParArch::GPU_NVIDIA || PA == ParArch::GPU_AMD || PA == ParArch::GPU_INTEL`, not `PA != ParArch::CPU`, and
   the preprocessor guard of its GPU code is `#if defined(__NVCC__) || defined(__HIPCC__) || defined(FK_SYCL_ENABLED)`:
   written with `Stream`, `defaultParArch` and the default `MemType`, the same GPU code tests the three backends.
4. Build and run: see the fkl-build-and-test skill (also with the backend OFF, and under ThreadSanitizer).

## Checklist before opening a PR

- [ ] `FK_STATIC_STRUCT`
- [ ] CPU exec() is `FK_HOST_FUSE`; GPU exec() is `FK_DEVICE_FUSE`; CPU_MULTITHREAD exec(done, ...) is `FK_HOST_STATIC`
- [ ] ParArch::CPU implementation with a public per-row function (`exec_row`, or one per phase), AND a ParArch::CPU_MULTITHREAD implementation that
      enqueues its chunks with `cpu_multithread::enqueueChunks` and calls the `Completion` (no stream in the DPP);
      only the attention DPPs are exempt
- [ ] CPU_MULTITHREAD code under `#if defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)`, outside any
      `__NVCC__`/`__HIPCC__` block; builds with the backend OFF
- [ ] one `Executor<MyDPP<PA, ...>>` per back-end (CPU, CPU_MULTITHREAD, and a GPU one constrained with
      `requires(PA == ParArch::GPU_NVIDIA || PA == ParArch::GPU_AMD)`), each running one kernel,
      `launchDPP_Kernel<DPP>`; invalid arguments throw `std::invalid_argument` (`executor_detail::requireArguments`)
      before anything runs; no per-DPP entry function, no per-DPP kernel
- [ ] the utest checks that invalid arguments throw on every back-end
- [ ] for a ParArch::GPU_INTEL (SYCL) version: under `#if defined(FK_SYCL_ENABLED)`, outside the `__NVCC__`/`__HIPCC__`
      block; `exec(const sycl::nd_item<3>&, ...)` is `FK_DEVICE_STATIC`; its `Executor<MyDPP<ParArch::GPU_INTEL, ...>>`
      makes the same checks as the others and launches with `launchDPP_Kernel` or `launchDPP_KernelWithLocalMemory`;
      its `_sycl` test passes
- [ ] preconditions (aliasing, buffer lifetime until `sync()`) documented in a comment
- [ ] utest instantiating every public alias/build path, with the CPU_MULTITHREAD path compared to ParArch::CPU
- [ ] compiles warning-clean on nvcc AND clang (CUDA 12.x and 13.x)