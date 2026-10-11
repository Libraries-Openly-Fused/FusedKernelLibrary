---
name: fkl-implementing-data-parallel-patterns
description: Implement a new Data Parallel Pattern (DPP) for the Fused Kernel Library. Covers the anatomy of a DPP struct, device-side invocation of IOps (the IOp-form table, reading inputs, fusing epilogues into the write), compile-time vs runtime types, vector types, the ParArch::CPU and ParArch::CPU_MULTITHREAD (thread pool) implementations, and testing. Use when adding an algorithm to FKL that requires thread coordination, or when debugging a DPP's internal device execution.
---

# Implementing FKL Data Parallel Patterns

## Anatomy of a DPP

A DPP is composed of three parts:

- Executor implementation: as in include/fused_kernel/core/execution_model/executors.h
- For GPU implementations, a kernel function that gets the DPP parameters and internally calls the DPP (for CPU, nothing;
  for CPU_MULTITHREAD, the executor/entry function enqueues the DPP on the stream, see below).
- The DPP implementation.

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
- **Only executors and entry functions use the stream**: they validate synchronously, then
  `stream.enqueue(...)` a lambda that calls `DPP::exec(done, ...)`.
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

// Entry function overload, next to the GPU/CPU ones (same name and argument order)
template <typename DPPDetails, typename InputIOp, typename OutputIOp>
inline bool executeMyDPP(const DPPDetails& details, const InputIOp& input, const OutputIOp& output,
                         Stream_<ParArch::CPU_MULTITHREAD>& stream) {
    if (!DPPDetails::valid(details)) return false;  // same validity behaviour as the other overloads
    stream.enqueue([=](const cpu_multithread::Completion& done) {
        MyDPP<ParArch::CPU_MULTITHREAD, DPPDetails>::exec(done, details, input, output);
    });
    return true;
}
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
- Entry points that take owning objects (`Ptr`, `Tensor`) may call `cpu_multithread::keepAlive(objects...)` before
  enqueueing, so the next execution enqueued from this thread keeps copies of them until it has finished; IOps only
  hold a `RawPtr`, so otherwise the buffers must stay valid until `sync()`.

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
3. Under `#ifdef FK_CPU_MULTITHREAD_ENABLED`, run every case also through the `Stream_<ParArch::CPU_MULTITHREAD>&`
   entry point (host memory, then `stream.sync()`) and require the output to match the `ParArch::CPU` output, plus the
   oracle checks: exactly (`memcmp` or `==`) for integer outputs and float pipelines without multiply-add, within a
   tight tolerance (e.g. 1e-5 relative) when floats are multiplied and added (fused multiply-add contraction). The same header is compiled by nvcc for the `_cu` target, so
   that code must compile there too. A test that means "GPU" must check
   `PA == ParArch::GPU_NVIDIA || PA == ParArch::GPU_AMD`, not `PA != ParArch::CPU`.
4. Build and run: see the fkl-build-and-test skill (also with the backend OFF, and under ThreadSanitizer).

## Checklist before opening a PR

- [ ] `FK_STATIC_STRUCT`
- [ ] CPU exec() is `FK_HOST_FUSE`; GPU exec() is `FK_DEVICE_FUSE`; CPU_MULTITHREAD exec(done, ...) is `FK_HOST_STATIC`
- [ ] ParArch::CPU implementation with a public per-row function (`exec_row`, or one per phase), AND a ParArch::CPU_MULTITHREAD implementation that
      enqueues its chunks with `cpu_multithread::enqueueChunks` and calls the `Completion` (no stream in the DPP);
      only the attention DPPs are exempt
- [ ] CPU_MULTITHREAD code under `#if defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)`, outside any
      `__NVCC__`/`__HIPCC__` block; builds with the backend OFF
- [ ] entry function overload taking `Stream_<ParArch::CPU_MULTITHREAD>&`, not ambiguous with the GPU `requires` ones
- [ ] preconditions (aliasing, buffer lifetime until `sync()`) documented in a comment
- [ ] utest instantiating every public alias/build path, with the CPU_MULTITHREAD path compared to ParArch::CPU
- [ ] compiles warning-clean on nvcc AND clang (CUDA 12.x and 13.x)