---
name: fkl-implementing-data-parallel-patterns
description: Implement an FKL Data Parallel Pattern for traversal, thread mapping, or cooperation, invoking supplied IOps. Use after architecture classification or when debugging DPP execution, fused inputs/outputs, and Executor integration. Not for ordinary host pipeline composition or per-thread Operations.
---

# Implementing FKL Data Parallel Patterns

Read [architecture overview](../fkl-architecture-overview/SKILL.md) first.
If existing `TransformDPP` can schedule the desired IOps, compose them rather
than implementing another DPP.

## Design the contract before the kernel

1. Define traversal, output ownership, and any required communication.
2. Specify read, compute, and write IOp roles and the value types between them.
   Pass IOp instances to `exec`; their types can be template parameters. A
   grouped role can use `fk::Tuple`. Do not hide these roles in scheduling details.
3. Put global memory access in Read/ReadBack and Write IOps, rather than raw
   global pointers in the DPP interface. Keep reusable arithmetic in compute
   IOps. The DPP owns thread mapping, staging, and synchronization.
4. Implement a CPU reference specialization and the intended accelerator
   specialization. Keep shared declarations and CPU code outside CUDA guards.
5. Supply the host launch path and tests for both plain and fused IOps.

Use `FK_STATIC_STRUCT` for the static-only DPP type. Choose qualifiers that fit
each backend; code using shared memory and barriers cannot simply inherit a
`constexpr` signature from an independent-thread example.

## Read the current implementation, not a copied skeleton

The following paths are under `include/fused_kernel/core/execution_model/`:

| Source | What to inspect |
|---|---|
| `data_parallel_patterns.h` | `TransformDPP`, its base/details, CPU loops, and GPU coordinate mapping |
| `executors.h` | `Executor` specializations, `BaseExecutor`, back-fusion, and kernel launch |
| `data_parallel_patterns.h` | `DivergentBatchTransformDPP` for selecting distinct IOp sequences |
| `parallel_architectures.h` | Backend identifiers and the default backend |

`TransformDPP` has architecture, thread-fusion, and details template arguments;
check their current order in the header. Its execution shape is
`exec(details, iOps...)`, not a universal signature for all possible patterns.

The primary `Executor<DPP>` does not implement arbitrary DPP dispatch. To use
`executeOperations<MyDPP>`, provide a compatible specialization and
`PAR_ARCH`. Reusing `BaseExecutor` also requires its helper contract. On CUDA,
the launch wrapper calls the device `exec`; on CPU, the executor calls the CPU
implementation without launching a GPU kernel.

## The IOp invocation contract (The IOp-form table)

These are execution-side call forms, not the implementation signatures in
[implementing operations](../fkl-implementing-operations/SKILL.md).
Generated parent overloads unpack the whole IOp's operation data as needed.

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

Incomplete operations have no executable form; complete them through
back-fusion before invocation.

## Preserve fusion at the DPP boundary

- Accept a complete Read IOp, including a fused read or ReadBack stack, wherever
  the role permits one. Invoke the supplied IOp rather than extracting a pointer
  and bypassing its prologue.
- A compute chain composed with a write, such as `compute.then(write)`, can be
  passed as one output IOp. Invoke
  `OutputIOp::Operation::exec(thread, result, output)` so its compute and write
  both execute. A read-led chain is not a value-consuming epilogue.
- For a sequence, follow the real execution fold in `TransformDPPBase`.
  `value | writeIOp` alone does not store: Transform separately invokes the
  terminal write with its output coordinate.
- Test coordinates and output geometry after back-fusion. Cropping/resizing
  changes the logical output domain; the original input extent is not enough.

## Pitfalls

- **Passing `iop.params` instead of `iop`:** This may work for a simple operation
  but bypasses the generic contract and can fail for fused or ReadBack IOps.
- **Early return before a barrier:** Partial tiles must not cause participating
  threads to skip a required synchronization.
- **Hard-coded transforms:** Replacing an input/output IOp with direct memory
  access silently discards any fused work attached to it.

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

1. Add a test header in a subdirectory of `utests/` or `tests/`, defining
   `int launch()`. Use the test harness where its builders support the pattern;
   custom DPP tests can check buffers and return nonzero directly.
2. Compare against an independent reference, including dimensions smaller than
   a tile, partial tiles, boundary coordinates, and multiple planes.
3. Exercise plain IOps and nontrivial fused input/output chains, with runtime
   parameter changes. Construction alone does not instantiate execution.
4. Follow [build and test](../fkl-build-and-test/SKILL.md). Existing references:
   `utests/core/execution_model/utest_executors.h` and
   `tests/data_parallel_patterns/test_divergent_batch.h`.

## Checklist before opening a PR

- [ ] `FK_STATIC_STRUCT`
- [ ] Backend-appropriate qualifiers and a working executor/launch contract
- [ ] Read/compute/write semantics remain replaceable through IOps
- [ ] Synchronization and output ownership are correct on partial tiles
- [ ] Standalone and fused execution pass on CPU and CUDA
- [ ] nvcc and supported host-compiler configurations are covered