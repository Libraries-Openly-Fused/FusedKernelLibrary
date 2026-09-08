---
name: fkl-implementing-operations
description: Implement single-thread FKL Operations for arithmetic, color conversion, geometric sampling, or memory access, with Parent aliases, build overloads, and tests. Use after fkl-architecture-overview classifies the work as an Operation, or when a FusedOperation alias fails to compile. Thread orchestration and cooperation belong in the DPP skill.
---

# Implementing FKL operations

## Before coding

Use [architecture overview](../fkl-architecture-overview/SKILL.md) to classify
the missing behavior. An Operation's `exec()` is single-thread code: no kernel
launch, thread scheduling, shared-memory cooperation, barriers, or warp
collectives. Loops, tuple inputs, and multiple samples are allowed; they do not
by themselves require a DPP. Reuse existing Ops before adding one.

For host-side `build()`/composition rather than authoring, use
[using operations](../fkl-using-operations/SKILL.md).

## Anatomy of an Operation

Every operation is a STATELESS struct: static `exec()`, type aliases from a Parent, and `build()` factories producing InstantiableOperations (IOps).

```cpp
namespace fk {
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
} // namespace fk
```

This is an authoring fragment, not a complete header. Include the parent and
arithmetic headers it uses, follow neighboring header guards, and keep the Op in
`fk::`. `FK_HOST_DEVICE_FUSE` already supplies `static constexpr`.

## Choosing the Operation type

Operation types are linked to the exec() function definition in the Operation. 
**Unary/Binary/Ternary describe the calling contract, not mathematical arity.**
For example, a Unary Op can consume a tuple containing two values; a Binary Op
combines one input value with runtime params.
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
| MidWriteType \* | X | X | X | X | | underlying Write exec returns void; IOp fold forwards InputType |
| OpenType \*\* | X | X | X | X | | OutputType exec(Point, InputType, ParamsType) |
| ClosedType \*\* | | X | | X | | void exec(Point, ParamsType) |

\* Applicable only to Instantiable Operations. In and Out must be the same type and value. Operation must be of WriteType.

\*\* OpenType and ClosedType are only applicable to FusedOperations. FusedOperations can also be ReadType or WriteType.

These are **author-side signatures**, not the generic DPP invocation table.
Parent adapters let DPPs pass the whole IOp instead of extracting params/back-IOp;
see [implementing DPPs](../fkl-implementing-data-parallel-patterns/SKILL.md).
The live table is in
`include/fused_kernel/core/execution_model/operation_model/operation_types.h`.
`IncompleteTernaryType` is declared but has no implemented authoring path;
do not use it as a template for new code.

## Choosing the parent

Each OperationType has its associated parent type. You can find them in the file include/fused_kernel/core/execution_model/operation_model/parent_operations.h

Notes:
- Unary ops carry no separate runtime params; their input value is still runtime data.
- Implement geometry/addressing hooks required by the chosen parent and consumers
  (inspect the nearest Read/Write example); not every custom Read needs pitch or
  grid hooks. Transform derives grid dimensions from the first complete read's
  `getActiveThreads()`.
- ReadBack ops define output geometry: a Resize returns its target Size from num_elems_x/y regardless of the source size.
- Use `DECLARE_READBACK_PARENT` from `batch_operations.h`, not the `_BASIC`
  variant, to preserve array builders for horizontal fusion. Incomplete pairs
  use `DECLARE_INCOMPLETEREADBACK_PARENT`.

## The IncompleteReadBack pattern (BVF ops)

User-facing geometric ops (Crop, Resize, Warping) are declared with `BackIOp = NullType`: the user builds them WITHOUT knowing the read (`Crop<>::build(rect)`). The BackFuser later calls `build(backIOp, selfIOp)` to complete them with the actual read. Implement BOTH build() overloads:

Schematic only (`MyGeo` and `WT` stand for your operation's types); use
`include/fused_kernel/algorithms/image_processing/crop.h` and `resize.h` for
complete/incomplete specializations, parent macros, and batch builders:

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

Per-call factors, rects, matrices, and sizes belong in ParamsType. Dtypes,
channel counts, and deliberate interpolation policies are template parameters.
Array-based HF has compile-time capacity; an active batch count can be runtime.
Do not introduce a specialization for each ordinary runtime value.

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

1. Add a header under `utests/<area>/` with `int launch()`. Follow a test for the
   same Operation category. `TestCaseBuilder` does not cover every category;
   its two-array overload is for Unary Ops, not the Binary `MyOp` above.
2. For `MyOp<float>`, build `MyOp<float>::build(2.f)` between a float Read and
   Write and compare inputs `{2.f, 3.f}` to `{4.f, 6.f}`. Test a second factor
   with the same IOp type to exercise runtime params.
3. Test every public alias and `build()` path **through execution**, not just
   construction. Include scalar/vector types, relevant borders, and batched
   ReadBack construction when supported.
4. Add a fused pipeline case to prove input/output compatibility; compare
   against an independent reference.
5. Build and run via [build and test](../fkl-build-and-test/SKILL.md).

## Checklist before opening a PR

- [ ] `FK_STATIC_STRUCT` + Parent alias + `DECLARE_*_PARENT`
- [ ] exec() is `FK_HOST_DEVICE_FUSE` (runs on CPU backend too)
- [ ] Geometry/addressing hooks required by this Read/Write/ReadBack contract
- [ ] both build() overloads for IncompleteReadBack ops
- [ ] values in params, types in templates
- [ ] Tests execute every public alias/build path, standalone and fused
- [ ] Supported CPU/CUDA tests pass using the repository's current toolchain configuration