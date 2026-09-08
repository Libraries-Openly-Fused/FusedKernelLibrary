---
name: fkl-implementing-operations
description: Implement single-thread FKL Operations for arithmetic, color conversion, coordinate sampling, or memory access, using Parent aliases, DECLARE_*_PARENT macros, build() and tests. Use for new per-thread behavior or failing FusedOperation aliases; thread scheduling, barriers, shared tiles, and collective algorithms belong in the DPP skill.
---

# Implementing FKL operations

## Scope and workflow

First apply [architecture classification](../fkl-architecture-overview/SKILL.md).
An Operation is strictly single-thread work, even if a DPP invokes it on many
threads. It may sample several pixels; it must not launch kernels, assign work
to threads, or use barriers/shuffles. For those, use
[implementing DPPs](../fkl-implementing-data-parallel-patterns/SKILL.md).
For host-only composition, use [using operations](../fkl-using-operations/SKILL.md).

1. Search existing algorithms for reusable Ops and the closest parent/build pattern.
2. Specify input, output, runtime params and (if needed) backing IOp types.
3. Implement in namespace `fk`, in the appropriate algorithm header, with its
   existing include guard and function qualifiers.
4. Instantiate every public build/alias path and test it in a fused pipeline.

## Anatomy of an Operation

Every operation is a STATELESS struct: static `exec()`, type aliases from a Parent, and `build()` factories producing InstantiableOperations (IOps).
The following is a Binary Op sketch, to place in namespace `fk` with the
operation-model headers included (compare `Mul` in
[arithmetic.h](../../../include/fused_kernel/algorithms/basic_ops/arithmetic.h)):

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

This table describes **implementation overloads**, not generic DPP call sites.
Parent macros adapt an IOp/OperationData argument to these params-form overloads.
For invocation through `IOp::Operation::exec`, use the
[DPP call-site table](../fkl-implementing-data-parallel-patterns/SKILL.md#the-iop-invocation-contract-the-iop-form-table).
The source contract is
[operation_types.h](../../../include/fused_kernel/core/execution_model/operation_model/operation_types.h).

Operation types are linked to the exec() function definition in the Operation. 
The elements that can change across Operation types are:
- OutputType: whether the exec function returns a value or not, and which type it is. The value resides on registers.
- ElementIdx: whether the exec function gets the thread idx as input or not. It is used to compute DRAM or Shared Memory addresses to read from or write into.
- InputType: whether the exec function gets an input value or not. This value resides on registers.
- ParamsType: whether the exec function gets any additional data that is not computed inside the kernel and that is needed for the execution of the operation.
- BackIOp: whether the exec function gets an additional IOp as input, that is executed as part of the operation implementation.

An example of the exec function with all the types would be: `OutputType exec(Point, InputType, ParamsType, BackIOp)`

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

\* MidWrite is an IOp wrapper around a Write Operation, not a new standalone Op.
The table describes its write-and-forward effect in the fold. The underlying
`Operation::exec` still returns `void`; the wrapper forwards the unchanged input.

\*\* OpenType and ClosedType are only applicable to FusedOperations. FusedOperations can also be ReadType or WriteType.

`IncompleteTernaryType` is declared but unused; do not use it as an implementation
template. Unary/Binary describe function inputs, not the number of mathematical
operands: a Unary Op may consume `Tuple<A,B>`, while a Binary Op consumes an input
and stored params.

## Choosing the parent

Each OperationType has its associated parent type. You can find them in the file include/fused_kernel/core/execution_model/operation_model/parent_operations.h

Notes:
- Unary ops carry no stored runtime params; their input value is still runtime data.
- Implement geometry methods required by the consuming DPP and parent, following
  the closest existing memory Op. Transform derives its grid from the read side's
  `getActiveThreads()`; not every specialized coefficient Read or Write needs all
  geometry/pitch methods.
- ReadBack ops define output geometry: a Resize returns its target Size from num_elems_x/y regardless of the source size.
- Use the complete `DECLARE_READBACK_PARENT` / `DECLARE_INCOMPLETEREADBACK_PARENT`
  macros from [batch_operations.h](../../../include/fused_kernel/core/execution_model/operation_model/batch_operations.h);
  the `_BASIC` variants omit batch builders needed for Horizontal Fusion.

## The IncompleteReadBack pattern (BVF ops)

User-facing geometric ops (Crop, Resize, Warping) are declared with `BackIOp = NullType`: the user builds them WITHOUT knowing the read (`Crop<>::build(rect)`). The BackFuser later calls `build(backIOp, selfIOp)` to complete them with the actual read. Implement BOTH build() overloads:

Sketch only; adapt the template arguments and aggregate layout to your parent:

```cpp
FK_HOST_FUSE auto build(const ParamsType& params) {     // user-facing
    return InstantiableType{{params, {}}};
}
template <typename BIOp>
FK_HOST_FUSE auto build(const BIOp& backIOp, const InstantiableType& iOp) {
    return ReadBack<MyGeo<WT, BIOp>>{ {iOp.params, backIOp} };  // fused
}
```

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

1. Add a utest header under `utests/<area>/` using TestCaseBuilder:
   first check for a matching specialization in
   [operation_test_utils.h](../../../tests/operation_test_utils.h). Its scalar
   `addTest(testCases, inputs, expected)` overload is for **Unary** Ops, not the
   Binary `MyOp` above. For that Op, test `MyOp<float>::build(2.f)` in an explicit
   read → MyOp → write pipeline and compare `{2.f, 3.f}` against `{4.f, 6.f}`.
2. Register builder cases before `STOP_ADDING_TESTS` and define `int launch()`;
   for unsupported builder categories use direct pipeline checks as in
   [utest_executors.h](../../../utests/core/execution_model/utest_executors.h).
3. Build and run: see [build and test](../fkl-build-and-test/SKILL.md).
4. If the op is an alias or has multiple build() overloads, instantiate EVERY public path in the test — template bugs hide until instantiation.
5. Cover changed runtime params, scalar/vector types where supported, and
   non-identity predecessor/successor Ops. For ReadBack, test output dimensions,
   border behavior, stacked sampling, and scalar **and batch** build paths.

## Checklist before opening a PR

- [ ] `FK_STATIC_STRUCT` + Parent alias + `DECLARE_*_PARENT`
- [ ] exec() is `FK_HOST_DEVICE_FUSE` (runs on CPU backend too)
- [ ] Geometry/pitch methods required by the parent and consuming DPP
- [ ] both build() overloads for IncompleteReadBack ops
- [ ] values in params, types in templates
- [ ] utest instantiating every public alias/build path
- [ ] Validated with supported host compilers and nvcc; no new warnings