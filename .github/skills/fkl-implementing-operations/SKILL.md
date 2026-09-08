---
name: fkl-implementing-operations
description: Implement single-thread FKL Operations for computation, sampling, or memory access, with Parent aliases, build overloads, and tests. Use after architecture classification or when a FusedOperation alias fails to instantiate. Traversal and thread cooperation belong in a DPP.
---

# Implementing FKL operations

## Anatomy of an Operation

First use [architecture overview](../fkl-architecture-overview/SKILL.md) to
confirm the missing behavior belongs in an Operation. Each invocation must work
independently in one thread; barriers, shared staging, and thread scheduling
belong in a DPP.

Every Operation is a stateless struct in `fk::`: static `exec()`, type aliases
from a CRTP parent, and `build()` factories producing IOps. Follow `Mul` in
`include/fused_kernel/algorithms/basic_ops/arithmetic.h`. This illustrative
definition belongs inside `namespace fk` with that header's dependencies:

```cpp
template <typename I, typename P = I, typename O = I>
struct MyOp {
private:
    using SelfType = MyOp<I, P, O>;
public:
    FK_STATIC_STRUCT(MyOp, SelfType)               // deletes ctors: pure static
    using Parent = BinaryOperation<I, P, O, SelfType>;
    DECLARE_BINARY_PARENT                          // pulls in aliases + build()
    FK_HOST_DEVICE_FUSE OutputType exec(const InputType input,
                                        const ParamsType& params) {
        return input * params;                     // your math here
    }
};
```

## Choosing the Operation type

Operation types describe the implementation's `exec()` signature, not the
number of mathematical operands. `BinaryType` means input plus runtime params;
a `UnaryType` may consume a tuple of two values (as `Add` does).
The elements that can change across Operation types are:
- OutputType: whether the exec function returns a value or not, and which type it is. The value resides on registers.
- ElementIdx: whether the exec function gets the thread idx as input or not. It is used to compute DRAM or Shared Memory addresses to read from or write into.
- InputType: whether the exec function gets an input value or not. This value resides on registers.
- ParamsType: whether the exec function gets any additional data that is not computed inside the kernel and that is needed for the execution of the operation.
- BackIOp: whether the exec function gets an additional IOp as input, that is executed as part of the operation implementation.

The table below describes Operation implementation forms. Generic DPP callers
use the whole-IOp overloads documented in
[implementing DPPs](../fkl-implementing-data-parallel-patterns/SKILL.md).
The source of truth is
`include/fused_kernel/core/execution_model/operation_model/operation_types.h`.

| Operation Type | OutputType | ElementIdx | InputType | ParamsType | BackIOp | exec function |
|---|---|---|---|---|---|---|
| ReadType | X | X | | X | | OutputType exec(Point, ParamsType) |
| WriteType | | X | X | X | | void exec(Point, InputType, ParamsType) |
| UnaryType | X | | X | | | OutputType exec(InputType) |
| BinaryType | X | | X | X | | OutputType exec(InputType, ParamsType) |
| ReadBackType | X | X | | X | X | OutputType exec(Point, ParamsType, BackIOp) |
| IncompleteReadBackType | | | | | | no exec function present |
| TernaryType | X | | X | X | X | OutputType exec(InputType, ParamsType, BackIOp) |
| IncompleteTernaryType | | | | | | no exec function present |
| MidWriteType \* | X | X | X | X | | InputType exec(Point, InputType, ParamsType) |
| OpenType \*\* | X | X | X | X | | OutputType exec(Point, InputType, ParamsType) |
| ClosedType \*\* | | X | | X | | void exec(Point, ParamsType) |

\* Applicable only to Instantiable Operations. In and Out must be the same type and value. Operation must be of WriteType.

\*\* OpenType and ClosedType are only applicable to FusedOperations. FusedOperations can also be ReadType or WriteType.

`IncompleteTernaryType` is declared but has no current implementation pattern
to copy. Do not design a new public API around the enum alone.

## Choosing the parent

Each OperationType has its associated parent type. You can find them in the file include/fused_kernel/core/execution_model/operation_model/parent_operations.h

Notes:
- Unary ops carry no runtime params.
- Implement geometry required by the chosen read/write parent and memory
  interface. The executor derives the logical domain from the completed read's
  `getActiveThreads()`; inspect `memory_operations.h` for dimensions and pitch.
- ReadBack ops define output geometry: a Resize returns its target Size from num_elems_x/y regardless of the source size.
- Use `DECLARE_READBACK_PARENT` from `batch_operations.h` for public ReadBack
  operations needing batch builders; the `_BASIC` form alone omits those builders.

## The IncompleteReadBack pattern (BVF ops)

`Crop<>::build(rect)` is incomplete: it has parameters but no source to sample.
Back-fusion calls `build(backIOp, selfIOp)` to complete it. Follow the incomplete
and complete `Crop` specializations in
`include/fused_kernel/algorithms/image_processing/crop.h`, including:

- The parameter-only public builder.
- Completion from a back-IOp and an incomplete IOp.
- Any explicit source-plus-parameters builder.
- Batch builders and output geometry.

Do not reuse aggregate initializers from an unrelated operation: the parent
determines its `OperationData` layout. `tests/algorithm/test_crop.h` exercises
explicit completion, `.then()`, batches, and output dimensions.

## Runtime values vs compile-time types (the golden rule)

Anything users may change per call (factors, rects, matrices, sizes) goes in ParamsType. Anything that changes the generated code (dtype, channel count, batch size, interpolation mode) is a template parameter. Getting this wrong either recompiles on every value change or silently bakes stale values into kernels.

## Vector types

Use the helpers instead of hand-rolled per-channel code:
- `VBase<T>` scalar base; `cn<T>` channel count; `VectorType_t<T, N>`.
- `make_<float3>(x, y, z)` construction; binary operators are already overloaded channel-wise for CUDA vector types (vector_utils.h).
- Write exec() once with `if constexpr (cn<I> == ...)` branches only when semantics differ per arity (see Equal, TensorSplit).

## FusedOperation aliases — wrap IOps, not raw Operations

When an alias composes several ops into one (e.g. BGR2GRAY = reorder + gray), `FusedOperation<...>` expects IOps (types with `::Operation`):

```cpp
// WRONG (ill-formed on instantiation: raw Ops have no ::Operation):
using type = FusedOperation<VectorReorder<I,2,1,0>, RGB2Gray<I,O>>;
// RIGHT:
using type = FusedOperation<Unary<VectorReorder<I,2,1,0>>, Unary<RGB2Gray<I,O>>>;
```
This exact bug shipped in four ColorConversion aliases (issues #244, fixed in LTS-C++17) — the alias compiles fine until someone instantiates it, so ALWAYS add a utest that calls `::build()` on every alias you define.

## Forwarding references in helpers

Never call `fuse_back<ExplicitArgs...>(...)`: explicit template arguments turn `IOps&&...` into rvalue refs that cannot bind to const lvalues stored in tuples (issue #245). Let deduction happen, e.g. through a lambda:

```cpp
apply([](const auto&... iOps) { return BackFuser::fuse_back(iOps...); }, tup);
```

## Testing a new op

1. Add a header in `utests/<area>/` with `int launch()`.
2. Use a supported `TestCaseBuilder` specialization from
   `tests/operation_test_utils.h`; do not assume it supports every Operation
   type. Otherwise follow an existing direct pipeline test.
3. Check scalar/vector behavior and relevant boundaries against expected values.
   Include a fused pipeline that actually executes the new Operation.
4. Instantiate every public alias and `build()` overload, including batch and
   ReadBack completion paths. Construction-only tests miss execution errors.
5. Follow [build and test](../fkl-build-and-test/SKILL.md).

## Checklist before opening a PR

- [ ] `FK_STATIC_STRUCT` + Parent alias + `DECLARE_*_PARENT`
- [ ] exec() is `FK_HOST_DEVICE_FUSE` (runs on CPU backend too)
- [ ] Geometry and pitch required by the selected read/write interface
- [ ] both build() overloads for IncompleteReadBack ops
- [ ] values in params, types in templates
- [ ] utest instantiating every public alias/build path
- [ ] CPU and nvcc tests pass with the supported host compilers