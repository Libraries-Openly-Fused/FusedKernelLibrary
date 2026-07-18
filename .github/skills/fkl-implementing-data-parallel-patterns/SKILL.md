---
name: fkl-implementing-data-parallel-patterns
description: Implement a new Data Parallel Pattern (DPP) for the Fused Kernel Library. Covers the InstantiableDPP protocol (DPP::build() -> InstantiableDPP, the DPPIOSpec IO contract, the generic fk::execute path), the anatomy of a DPP struct, device-side invocation of IOps (the IOp-form table, reading inputs, fusing epilogues into the write), compile-time vs runtime types, vector types, and testing. Use when adding an algorithm to FKL that requires thread coordination, or when debugging a DPP's internal device execution.
---

# Implementing FKL Data Parallel Patterns

## DPPs are instantiable values (the InstantiableDPP model)

DPPs follow the same design language as Operations: **types define the kernel;
`build()` values are runtime parameters**.

- An Operation is a stateless static struct; `Op::build(params)` produces an
  **IOp** (Operation type + runtime `OperationData`).
- A DPP is a stateless static struct; `DPP::build(iOps...)` produces an
  **InstantiableDPP** (DPP type + runtime `Details` + the IOps it will execute).

An InstantiableDPP is executed with the generic entry point:

```cpp
#include <fused_kernel/fused_kernel.h>

const auto iDPP = TransformDPP<>::build(readIOp, mulIOp, writeIOp);
fk::execute(stream, iDPP);
```

`fk::execute` (executors.h) is ONE generic executor path for every conforming
DPP: on CPU it calls `DPP::exec(details, iOps...)` directly (the DPP implements
the whole sequential loop); on GPU it launches the single generic
`launchInstantiableDPP_Kernel` trampoline with the grid/block returned by the
DPP's `getLaunchConfig()`. **A new conforming DPP needs NO hand-written
`Executor<DPP>` specialization and NO dedicated `__global__` kernel.**

`executeOperations<TransformDPP<...>>(stream, iOps...)` keeps working unchanged
(legacy Executor path).

## The InstantiableDPP protocol (what a conforming DPP must implement)

A DPP front type (per `ParArch` backend) provides:

| Member | Kind | Purpose |
|---|---|---|
| `static constexpr ParArch PAR_ARCH` | constant | the backend this specialization implements |
| `static constexpr DPPIOSpec IO_SPEC` | constant | the compile-time IO contract (see below) |
| `build_details(iOps...)` | `FK_HOST_FUSE` | builds the runtime Details (non-data parameters); may return an empty struct |
| `build(iOps...)` | `FK_HOST_FUSE` | usually one line: `return buildInstantiableDPP<SelfType>(iOps...);` |
| `exec(details, iOps...)` | `FK_DEVICE_FUSE` / coop / `FK_HOST_FUSE` | the pattern itself (GPU device code / CPU sequential loop) |
| `getLaunchConfig(details, iOps...)` | `FK_HOST_FUSE`, GPU only | returns a `DPPLaunchConfig{ grid, block }`; `defaultDPPLaunchConfig(activeThreads)` gives the TransformDPP heuristic |

`buildInstantiableDPP<DPP>(iOps...)` (core/execution_model/instantiable_dpp.h)
does three things: it brings the IOps to canonical form (fusing ReadBack stacks
via Backwards Vertical Fusion), it static_asserts the IO contract, and it calls
`DPP::build_details` to produce the `InstantiableDPP` value.

## The IO contract (DPPIOSpec): what goes in and what comes out

Every DPP declares its IO API as a compile-time contract. All data enters and
leaves a DPP through IOps — inputs through complete Read/ReadBack IOps, outputs
through Write IOps, never through raw pointers.

```cpp
struct DPPIOSpec {
    size_t inputIOps;         // exact number of leading Read/ReadBack IOps consumed
    size_t outputIOps;        // exact number of trailing Write IOps produced into
    bool acceptsComputeIOps;  // whether a compute/MidWrite IOp chain is accepted in between
    bool argsAreIOpSequences; // whether the DPP consumes whole IOpSequences (one per
                              // divergent branch); the counts then apply to EACH sequence
};
```

Current contracts:

| DPP | IO_SPEC | Meaning |
|---|---|---|
| `TransformDPP` | `{1, 1, true, false}` | 1 Read/ReadBack in, optional compute chain, 1 Write out |
| `DivergentBatchTransformDPP` (GPU) | `{1, 1, true, true}` | N IOpSequences, each: 1 Read + compute chain + 1 Write |
| `RowReduceDPP` | `{1, 1, false, false}` | 1 Read/ReadBack in (2D only), 1 Write out, NO compute chain (fuse preprocessing into the read: `readIOp.then(...)`); ReduceOp must be associative + commutative (see below) |

Conformance is enforced with granular `static_assert` messages ("DPP IO
contract violation: ...") when `build()`ing and when `fk::execute()`ing. The
soft (non-asserting) check `dppIOContractSatisfied<DPP, IOps...>()` is usable
in tests and SFINAE. The repo has no compile-fail test infra, so utests
document the failing `build()` calls as comments next to soft-check
static_asserts (see utest_instantiable_dpp.h, utest_row_reduce_dpp.h).

## Anatomy of a DPP

A DPP is composed of:

- The DPP implementation (one specialization per ParArch backend).
- Its `DPPIOSpec IO_SPEC` and the other InstantiableDPP protocol members.
- Nothing else: the generic executor and generic kernel do the launching.

Every DPP must always have a single thread CPU implementation using the ParArch::CPU and then it can have implementations
for other ParArchs, as in include/fused_kernel/core/execution_model/parallel_architectures.h
(exception so far: DivergentBatchTransformDPP is GPU-only in practice — its CPU exec is not implemented).

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

Never pass a raw pointer (T*) to device (global) memory as an input or output parameter to exec function.
Device pointers must be passed as part of a Read/ReadBack or Write IOps.

Cooperative exec bodies (`__shared__` + barriers) cannot be constexpr: use
`FK_COOP_DEVICE_FUSE` (= `static __device__ __forceinline__ void`) instead of
`FK_DEVICE_FUSE` for the GPU exec in that case.

## Reference implementation of a new DPP: RowReduceDPP

`include/fused_kernel/algorithms/reductions/row_reduce.h` is the reference for
authoring a DPP through the pure InstantiableDPP API: CPU + GPU backends,
`IO_SPEC{1, 1, false, false}`, empty Details, `build()` =
`buildInstantiableDPP<SelfType>(iOps...)`, a `getLaunchConfig` that maps one
thread block per row, and a cooperative shared-memory tree reduction on GPU vs
a plain sequential loop on CPU. It is not in any umbrella header; include it
directly.

Semantic requirements the type system cannot check (documented, not
static_asserted):

- The reduction ORDER is backend-dependent (strict sequential left fold on CPU;
  strided per-lane accumulation + shared-memory tree on GPU), so the ReduceOp
  must be **associative and commutative** (Add, Max, Min...). A non-commutative
  BinaryType Operation (e.g. Sub) compiles cleanly but produces
  backend-dependent results. For floating-point types both backends are
  correct, but reassociation means their rounding may differ, so exact bitwise
  equality between backends is not guaranteed.
- **2D inputs only**: both backends fix the Point z coordinate to 0, so
  `build()` throws `std::invalid_argument` if the input IOp's ActiveThreads.z
  is not 1 (a batched/3D read would otherwise silently reduce only plane
  z == 0).

Usage:

```cpp
#include <fused_kernel/algorithms/reductions/row_reduce.h>

const auto iDPP = RowReduceDPP<Add<int>>::build(
    PerThreadRead<ND::_2D, int>::build(input),   // or a fused prologue: .then(Mul<int>::build(3))
    PerThreadWrite<ND::_1D, int>::build(output)); // one element per row, written at Point{row, 0, 0}
fk::execute(stream, iDPP);
```

Example TransformDPP GPU exec, which is a special case because it does not
require any specific structure in the IOps passed as parameters. The IOps in
this DPP will be executed one after the other by all the threads that have to
participate:

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

Inside the DPP's execution logic, you must invoke the IOps passed to the `exec()` function. If the compute chain was fused into the input or output IOp (prologue/epilogue), you invoke it as a single IOp type:

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
- **Forgetting `IO_SPEC`:** `DPP::build()`/`fk::execute()` static_assert that the DPP declares `static constexpr DPPIOSpec IO_SPEC{...}`.
- **GPU DPP without `getLaunchConfig`:** `fk::execute(gpuStream, iDPP)` static_asserts its presence.

## Runtime values vs compile-time types (the golden rule)

Anything users may change per call (factors, rects, matrices, sizes) goes
in ParamsType. Anything that changes the generated code (dtype, channel
count, batch size, interpolation mode) is a template parameter. Getting
this wrong either recompiles on every value change or silently bakes
stale values into kernels. The same rule applies to DPPs: runtime values go in
the Details struct (built by `build_details`), code-shaping choices (the reduce
Operation, BLOCK_SIZE, thread fusion enablement) are template parameters.

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

1. Add a utest header under `utests/<area>/`. Exercise `DPP::build()` +
   `fk::execute()` on BOTH backends (the harness compiles every test as
   `_cpp` and `_cu`; GPU-only DPPs use `#define __ONLY_CU__`).
2. Add compile-time contract checks with
   `static_assert(dppIOContractSatisfied<DPP, IOps...>() == expected)` for both
   conforming and non-conforming packs, plus commented compile-fail
   documentation of the `build()` static_assert messages.
3. Build and run: see the fkl-build-and-test skill.

## Checklist before opening a PR

- [ ] `FK_STATIC_STRUCT`
- [ ] `IO_SPEC`, `build_details()`, `build()` (= `buildInstantiableDPP<SelfType>`), `getLaunchConfig()` on GPU
- [ ] CPU exec() is `FK_HOST_FUSE`; GPU exec() is `FK_DEVICE_FUSE` (or `FK_COOP_DEVICE_FUSE` if it needs shared memory/barriers)
- [ ] No `Executor` specialization and no dedicated `__global__` kernel (the generic path handles both)
- [ ] utest instantiating every public alias/build path, on both backends, plus contract checks
- [ ] compiles warning-clean on nvcc AND clang (CUDA 12.x and 13.x)
