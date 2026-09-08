---
name: fkl-implementing-data-parallel-patterns
description: Implement or debug FKL Data Parallel Patterns (DPPs) that organize traversal, thread cooperation, shared tiles, reductions, or tensor-core work. Covers IOp invocation, launch integration, fusion boundaries, and composition tests. Use after architecture classification; independent per-thread math or sampling belongs in the Operation skill.
---

# Implementing FKL Data Parallel Patterns

## Start here: contract before kernel

Apply the [architecture decision procedure](../fkl-architecture-overview/SKILL.md).
A DPP owns thread organization, not merely a large amount of arithmetic.
Independent per-thread sampling belongs in a ReadBack Operation.

Before coding, write down:
1. Inputs and outputs: number of Read/ReadBack and Write IOps, their value types,
   coordinates, shapes, and any supported fused prologue/epilogue.
2. Replaceable compute IOps and their input/output types. A reduction combiner
   must accept two runtime values; default `Add<T>` is Binary (input + stored
   parameter), whereas `Add<T,T,T,UnaryType>` consumes a tuple of two values.
3. Thread/output ownership, traversal, synchronization, tail handling and scratch
   storage. Keep these in the DPP; keep reusable per-thread work in Operations.
4. Runtime details, supported backends, host validation, and launch entry point.

## Implementation recipe and source templates

Use a stateless struct (`FK_STATIC_STRUCT`, static `exec`) and deduce IOp types
from instance arguments. Put scheduling scalars in a details object and IOps in
separate arguments; a `fk::Tuple` may group multiple reads. User data pointers
should be encapsulated in IOps, not used to bypass their fusion boundaries.

Follow [LinearFilterDPP](../../../include/fused_kernel/algorithms/image_processing/linear_filter.h)
for a cooperative design: details + image/coefficient Read IOps + multiply and
accumulate IOps + output Write IOp. Its CPU traversal and GPU shared-tile
implementation use the same contract. Reuse the neighborhood staging machinery
it uses rather than duplicating halo and boundary logic.

Complete the launch path as well as `exec()`:
- For the general executor model, inspect
  [data_parallel_patterns.h](../../../include/fused_kernel/core/execution_model/data_parallel_patterns.h)
  and [executors.h](../../../include/fused_kernel/core/execution_model/executors.h).
  The current Transform template is `TransformDPP<PA, TFEN, DPPDetails>`, not
  `TransformDPP<PA, DPPDetails>`.
- A GPU path needs a kernel wrapper and host dispatch/grid configuration.
  A CPU path needs traversal, not a CUDA launch.
- Do not assume `executeOperations<MyDPP>` works just because `MyDPP::exec`
  exists: it requires a matching Executor implementation. LinearFilter and
  attention headers also demonstrate dedicated launch functions.

For new portable patterns, implement CPU semantics outside `__NVCC__` guards
and guard CUDA-only implementations. Existing attention DPPs are CUDA-only
and include internal arithmetic and raw output/workspace pointers; treat those
as current limitations, not the general composable template to copy.

`FK_*_FUSE` already includes `static constexpr`. Do not add another `static`,
or use constexpr qualifiers for bodies with shared memory/barriers. LinearFilter
uses `FK_HOST_STATIC` / `FK_DEVICE_STATIC`; attention defines cooperative
non-constexpr qualifiers locally.

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
| MidWriteType | underlying Write `exec(thread, input, iop)` returns `void`; the IOp fold forwards the input |
| OpenType     | `O    IOp::Operation::exec(thread, input, iop)` |
| ClosedType   | `void IOp::Operation::exec(thread, iop)` |

Key rules for DPP authors invoking these:
- **Invoke through the IOp:** Pass the whole `iop` to support fused and backed
  operations. Concrete Ops also have implementation overloads taking params;
  those are not a generic replacement for the IOp interface.
- **Read** takes the `Point thread` + `iop` and returns the value.
- **Write** takes `thread`, the `value` to store, and `iop`.
- **Unary/Binary** are pure register compute: no `thread`, just the input (+ `iop` for Binary).
- `fk::compute(value, iop)` dispatches Unary/Binary/Ternary correctly; `value | iop`
  is the existing compute fold interface.
- Incomplete ReadBack IOps have no `exec`; complete/fuse them on the host first.
  `IncompleteTernaryType` is declared but unused.
- The [Operation skill](../fkl-implementing-operations/SKILL.md) owns the
  implementation-signature table. Do not confuse those params-form signatures
  with this IOp call-site table.

## Invoking Operations Inside the DPP (Device Side)

For DPPs accepting a fused Write IOp, build the epilogue **with the write** on the
host. This is a fragment using already-built `read` and `write` IOps:

```cpp
const auto input = read.then(Mul<float>::build(2.f)).then(Add<float>::build(1.f));
const auto output = Mul<float>::build(0.25f).then(write);
```

Inside `exec`, use `InputIOp::Operation::exec(point, input)` to load and
`OutputIOp::Operation::exec(point, result, output)` to apply the epilogue and write.
Do not run a fused Write as a compute-only function or extract its pointer.
The output chain starts with compute, not a Read; `.then()` **can** start from a
Read when constructing an input prologue, as above.

See the [LinearFilter composition test](../../../tests/image_processing/test_linear_filter_dpp.h).
Attention has different contracts: inspect its actual launcher rather than
assuming every epilogue is a fused Write.

## Runtime values vs compile-time types (the golden rule)

Operation values go in IOp params; DPP dimensions, anchors and scheduling values
go in runtime details. Dtypes, tile capacities and policies are template choices.
Distinguish fixed `std::array` batch capacity from runtime active batch counts;
not every DPP has a compile-time batch size.

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

1. Add a header with `int launch()` under `utests/<area>/` or `tests/<area>/`.
   Use an independent CPU oracle; do not compare two paths sharing the same bug.
2. Test base IOps, a non-identity read prologue, a multi-op write epilogue, and
   replacement compute IOps. LinearFilter's arithmetic-mutation cases catch
   hard-coded arithmetic that identity fusion tests miss.
3. Test odd/non-tile-aligned sizes, boundary/halo handling, parameter changes,
   multiple batches, and invalid/empty sizes according to the public contract.
4. Check that tail threads do not return before required block barriers, shuffle
   masks match participating lanes, scratch is initialized, and each output has
   one intended writer.
5. Build/run available CPU and CUDA targets using
   [build and test](../fkl-build-and-test/SKILL.md); explicitly report missing
   backend/runtime coverage. GPU-only tests use the repository's `ONLY_CU` marker.

## Checklist before opening a PR

- [ ] `FK_STATIC_STRUCT`
- [ ] Correct qualifiers for CPU/GPU bodies, including non-constexpr cooperative code
- [ ] IOp contracts and host launch path implemented; backend restrictions explicit
- [ ] Numerical, composition, replacement-IOp, boundary and tail tests
- [ ] utest instantiating every public alias/build path
- [ ] Validated with supported host compilers and nvcc; no new warnings