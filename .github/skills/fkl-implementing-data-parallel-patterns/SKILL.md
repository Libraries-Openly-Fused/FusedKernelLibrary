---
name: fkl-implementing-data-parallel-patterns
description: Implement FKL Data Parallel Patterns that own traversal, thread mapping, tiling, or cooperation and invoke supplied IOps. Use after classifying the work with fkl-architecture-overview, or to debug DPP execution, fused prologues/epilogues, and backend launch contracts. Not for per-thread Operations or ordinary host pipeline composition.
---

# Implementing FKL Data Parallel Patterns

## Start with the contract

First use [architecture overview](../fkl-architecture-overview/SKILL.md).
Cooperation requires a DPP, but a DPP can also schedule independent work.
Reuse `TransformDPP` for ordinary elementwise pipelines; do not invent a DPP
for each combination of Operations.

Before implementing, write down:

1. The traversal and output ownership; cooperating lanes/threads and barriers.
2. The Read/ReadBack, compute/reducer/selector, and Write IOp roles and types.
3. Runtime scheduling `Details`, compile-time tile policy, and shape constraints.
4. Host launcher/Executor, supported backends, and any workspace lifetime.
5. A plain and a fused-IOp test with an independent numerical reference.

## Anatomy and canonical implementations

A DPP is a stateless type with static `exec()`. It organizes traversal, shared
storage, synchronization, and calls to supplied IOps. Keep reusable data
transformations in Operations, not hard-coded in the DPP. Global input/output
pointers belong in Read/Write IOps, so callers can substitute a fused prologue
or epilogue without editing the pattern.

Pass IOp instances as `exec()` arguments, with deduced template types. Each role
may be a single IOp, a fused IOp, or an explicitly supported `fk::Tuple` of IOps.
A tuple is a container, not itself an IOp. Keep runtime scheduling geometry in
`Details` and operation parameters in their IOps.

Read these repository-root-relative sources instead of copying an abbreviated
specialization whose template arguments may drift:

| Need | Source and what to inspect |
|---|---|
| Independent transform | `include/fused_kernel/core/execution_model/data_parallel_patterns.h`: `TransformDPP`, `build_details`, `exec_thread`, CPU traversal, thread-fusion tails |
| Executor integration | `include/fused_kernel/core/execution_model/executors.h`: `BaseExecutor` backward fusion, `Executor<TransformDPP<...>>`, kernel launch |
| Cooperative pattern with replaceable arithmetic | `include/fused_kernel/algorithms/image_processing/linear_filter.h`: `LinearFilterDPP`, tuple of image/coefficient Read IOps, multiply/accumulate IOps, fused Write, `executeLinearFilter` |
| Shared staging versus single-thread selection | `include/fused_kernel/algorithms/image_processing/median_filter.h`: `MedianFilterDPP` versus `MedianWindowSelect`; shared staging in `neighborhood.h` |
| Full fusion tests | `tests/image_processing/test_linear_filter_dpp.h`, `tests/operation/test_fused_write_epilogue_repro.h` |

`TransformDPP` requires a complete read first and write last. It is not a generic
launcher for every DPP: neighborhood patterns use dedicated `execute*` helpers.
A new pattern needs either its own launcher and GPU kernel forwarding to `exec`,
or a matching Executor specialization. CPU execution needs traversal, not a
`__global__` wrapper.

### Backends and existing exceptions

Prefer CPU and GPU implementations sharing the same IOp contract, with `Details`
and the CPU specialization outside `__NVCC__` guards. Do not force cooperative
bodies to be constexpr: shared memory and barriers use `FK_DEVICE_STATIC`
(CPU counterpart `FK_HOST_STATIC`) in neighborhood DPPs. `FK_*_FUSE` already
includes `static constexpr`; do not add a second `static`.

Current `algorithms/attention/` DPPs are GPU-only and explicitly included, not
exported by the umbrella header. They have specialized launchers, internal
reduction/tensor-core math, and in some cases raw output/workspace pointers plus
compute-only epilogues. These are existing limitations, not the default template
for a new composable DPP. Inspect the specific header and test; do not claim a CPU
oracle is a CPU backend or assume its epilogue is a fused Write.

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
| MidWriteType | Underlying Write `exec(thread, input, iop)` stores; the `InputFoldType \| iop` wrapper forwards the unchanged input |
| OpenType     | `O    IOp::Operation::exec(thread, input, iop)` |
| ClosedType   | `void IOp::Operation::exec(thread, iop)` |

Incomplete ReadBack/Ternary forms cannot be executed. Complete ReadBacks before
device invocation. The author-side `exec(..., ParamsType, BackIOp)` signatures
are a different layer; see [implementing operations](../fkl-implementing-operations/SKILL.md)
and `operation_model/parent_operations.h`.

Key rules for DPP authors invoking these:
- **Invoke through the IOp:** Call `IOp::Operation::exec(...)`, passing the whole `iop` instance for the data, NOT just `iop.params` or the raw op struct.
- **Read** takes the `Point thread` + `iop` and returns the value.
- **Write** takes `thread`, the `value` to store, and `iop`.
- **Unary/Binary** are pure register compute: no `thread`, just the input (+ `iop` for Binary).

## Invoking Operations Inside the DPP (Device Side)

For a DPP with Read and Write roles, build the fused IOps on the host (fragment;
`src` and `dst` are compatible float buffers):

```cpp
const auto input = PerThreadRead<ND::_2D, float>::build(src).then(Mul<float>::build(2.f));
const auto output = Add<float>::build(0.5f).then(PerThreadWrite<ND::_2D, float>::build(dst));
```

Inside `exec`, for an input/output pair of deduced IOp types:
`auto value = InputIOp::Operation::exec(point, input);` and
`OutputIOp::Operation::exec(point, result, output);`.
The DPP supplies `point` and computes `result` through its algorithm/compute IOps.
The fused output starts with compute, not Read; use `Cast<T,T>::build()` for
an identity epilogue. A complete read-to-write sequence is Closed, not Write.

Use the real `value | computeIOp` or `InputFoldType | iop` machinery for compute
chains; do not manually unpack a fused IOp's params. For example, LinearFilter
passes `make_tuple(value, coefficient) | multiply`: two mathematical operands
can be the **one input** of a Unary IOp.

## Pitfalls

- **Passing `iop.params` instead of `iop`:** A concrete Op may accept params,
  but generic DPP code must accept fused and ReadBack IOps too. Pass the whole
  wrapper so the parent/fusion machinery selects the right overload.
- **Returning before a barrier:** Out-of-range lanes may still be needed for
  tile loading or synchronization. Guard output accesses without skipping a
  barrier other participating lanes will reach.
- **Manual global epilogue stores:** These bypass a supplied Write IOp. Invoke
  the fused Write once per valid output at the final ownership point.

## Runtime values vs compile-time types (the golden rule)

Runtime data behavior belongs in IOp params; runtime scheduling dimensions belong
in DPP `Details`. Dtypes, tile capacities, and deliberate policy specializations
belong in templates. `std::array` HF has compile-time capacity; an active batch
count or a specialized DPP's batch dimension can still be runtime.

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

1. Add a header under `utests/<area>/` or `tests/<area>/` defining `int launch()`.
   Use the closest DPP test, not a Unary-only TestCaseBuilder overload.
2. Test plain and fused reads/writes and at least one substituted compute IOp.
   Instantiate execution, not only `build()`.
3. Cover partial tiles, small/irregular dimensions, borders, invalid Details,
   supported dtypes and numerical tolerances against an independent reference.
4. Build/run supported backends via [build and test](../fkl-build-and-test/SKILL.md).
   Use `utests/algorithm/attention/utest_softmax.h` for GPU-only testing patterns.

## Checklist before opening a PR

- [ ] `FK_STATIC_STRUCT`
- [ ] Scheduling and shared storage separated from reusable IOp data semantics
- [ ] Launcher, Details validation, supported backends, and workspace contract explicit
- [ ] Correct constexpr/non-constexpr function macros and uniform barriers
- [ ] Plain and fused IOp execution tested, including every public build/launch path
- [ ] Targeted tests pass; full relevant suite run before finalizing implementation changes