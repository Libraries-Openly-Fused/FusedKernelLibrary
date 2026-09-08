---
name: fkl-implementing-data-parallel-patterns
description: Implement a new Data Parallel Pattern (DPP) for the Fused Kernel Library. Covers the anatomy of a DPP struct, device-side invocation of IOps (the IOp-form table, reading inputs, fusing epilogues into the write), compile-time vs runtime types, vector types, and testing. Use when adding an algorithm to FKL that requires thread coordination, or when debugging a DPP's internal device execution.
---

# Implementing FKL Data Parallel Patterns

## Anatomy of a DPP

A DPP is composed of three parts:

- Executor implementation: as in include/fused_kernel/core/execution_model/executors.h
- For GPU implementations, a kernel function that gets the DPP parameters and internally calls the DPP (for CPU, nothing).
- The DPP implementation.

Every DPP must always have a single thread CPU implementation using the ParArch::CPU and then it can have implementations
for other ParArchs, as in include/fused_kernel/core/execution_model/parallel_architectures.h

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
3. Build and run: see the fkl-build-and-test skill.

## Checklist before opening a PR

- [ ] `FK_STATIC_STRUCT`
- [ ] CPU exec() is `FK_HOST_FUSE`; GPU exec() is `FK_DEVICE_FUSE`
- [ ] utest instantiating every public alias/build path
- [ ] compiles warning-clean on nvcc AND clang (CUDA 12.x and 13.x)