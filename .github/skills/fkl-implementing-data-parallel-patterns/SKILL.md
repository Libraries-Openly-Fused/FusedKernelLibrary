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

Trace and implement the entire launch path:

1. Expose the host entry point and validate the supported IOp roles.
2. Complete ReadBack IOps before deriving geometry. If using `BaseExecutor`,
   provide `executeOperations_helper` and its access/friend contract;
   `DECLARE_EXECUTOR_PARENT_IMPL` exposes the existing overload family.
3. Build runtime details from the completed IOps. Define input, output, and
   scratch extents separately when the pattern changes the number of outputs.
4. Dispatch the CPU implementation or launch the CUDA implementation with
   matching block/grid/shared-memory requirements and the caller's stream.
   `executor_details/executor_kernels.h` contains pattern-specific wrappers,
   not a universal launcher for arbitrary DPPs.
5. Follow the existing executor's CUDA error-checking convention, and test via
   the public entry point so back-fusion and launch setup are instantiated too.

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
| MidWriteType | `void IOp::Operation::exec(thread, input, iop)` (underlying Write; preserve input separately) |
| OpenType     | `O    IOp::Operation::exec(thread, input, iop)` |
| ClosedType   | `void IOp::Operation::exec(thread, iop)` |

Key rules for DPP authors invoking these:
- **Invoke through the IOp:** Call `IOp::Operation::exec(...)`, passing the whole `iop` instance for the data, NOT just `iop.params` or the raw op struct.
- **Read** takes the `Point thread` + `iop` and returns the value.
- **Write** takes `thread`, the `value` to store, and `iop`.
- **Unary/Binary** are pure register compute: no `thread`, just the input (+ `iop` for Binary).

Incomplete operations have no executable form; complete them through
back-fusion before invocation.

For MidWrite, forwarding is performed by the `InputFoldType` execution fold in
`operation_model/instantiable_operations.h`, not by the underlying Write's
return value. A manual caller must retain the input after the write.
For a compute role, `fk::compute(input, iop)` in
`operation_model/operation_types.h` selects the Unary/Binary/Ternary call form;
it does not schedule threads or invoke writes.

## Worked decomposition: cooperative reduction

This is a design exercise from the paper's §IV-C/Figure 14, not a shipped
reduction API to call:

- **Read IOp:** obtains each input value and any fused input transformation.
- **Combine IOp:** combines register values. For addition,
  `Add<float, float, float, UnaryType>` takes a tuple of two floats; the DPP
  supplies that tuple rather than hard-coding addition in its reduction tree.
- **DPP:** distributes inputs, stages partial results, synchronizes participants,
  connects combine invocations, and chooses which thread owns each result.
- **Output IOp:** performs any final transform and writes at the output coordinate.

Specify identity values, accumulation types, ordering/associativity assumptions,
partial-tile behavior, and scratch lifetime. A sum-to-max substitution also
changes the identity; not every type-compatible IOp is semantically valid.
The CPU reference must implement the same contract without GPU synchronization.

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

## Place parameters in the owning layer

| Kind of value | Where it belongs |
|---|---|
| Per-call data semantics: factors, rects, memory descriptors | Operation `ParamsType`, carried by the supplied IOp |
| Per-call scheduling: active domain, plane count, tail handling | DPP details/launch arguments, as in `TransformDPPDetails` |
| Code-shaping choice: dtype, backend, compile-time tile or batch size | Template parameters |

Do not put reusable transforms in DPP details or hide scheduling state inside
an unrelated Operation. Reuse the vector helpers described in
[implementing operations](../fkl-implementing-operations/SKILL.md) for per-thread
values instead of duplicating channel-wise computation in the DPP.

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
4. Substitute supported IOps to prove that their semantics are not hard-coded;
   check results, not just successful compilation. For thread fusion, test
   enabled/disabled execution and scalar tails separately.
5. Follow [build and test](../fkl-build-and-test/SKILL.md). Existing references:
   - `utests/core/execution_model/utest_executors.h`: back-fusion types and
     equivalence of automatic versus explicit composition, not an independent oracle.
   - `tests/data_parallel_patterns/test_divergent_batch.h`: executed numerical
     checks for selected sequences.
   - `tests/operation/test_fused_write_epilogue_repro.h`: CUDA fused-output
     execution regression; add corresponding CPU coverage for a new general DPP.

## Checklist before opening a PR

- [ ] `FK_STATIC_STRUCT`
- [ ] Backend-appropriate qualifiers and a working executor/launch contract
- [ ] Read/compute/write semantics remain replaceable through IOps
- [ ] Synchronization and output ownership are correct on partial tiles
- [ ] Standalone and fused execution pass on CPU and CUDA
- [ ] nvcc and supported host-compiler configurations are covered