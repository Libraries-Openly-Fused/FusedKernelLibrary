---
name: fkl-implementing-operations
description: Implement a new Operation for the Fused Kernel Library — Unary, Binary, Ternary, Read, Write, ReadBack or IncompleteReadBack structs, with Parent aliases, DECLARE_*_PARENT macros, build() overloads and unit tests. Use when adding an algorithm to FKL (new arithmetic, color conversion, geometric transform, memory pattern) or when a FusedOperation/alias fails to compile.
---

# Implementing FKL operations

## Anatomy of an Operation

Every operation is a STATELESS struct: static `exec()`, type aliases from
a Parent, and `build()` factories producing InstantiableOperations (IOps).

```cpp
template <typename I, typename P = I, typename O = I>
struct MyOp {
private:
    using SelfType = MyOp<I, P, O>;
public:
    FK_STATIC_STRUCT(MyOp, SelfType)              // deletes ctors: pure static
    using Parent = BinaryOperation<I, P, O, SelfType>;
    DECLARE_BINARY_PARENT                          // pulls in aliases + build()
    FK_HOST_DEVICE_FUSE OutputType exec(const InputType input,
                                        const ParamsType& params) {
        return input * params;                     // your math here
    }
};
```

## Choosing the parent (decision table)

| parent | input | extra | examples |
|---|---|---|---|
| `UnaryOperation<I, O, Self>` | value | — | Cast, VectorReorder, RGB2Gray |
| `BinaryOperation<I, P, O, Self>` | value | params | Mul, Add, Saturate |
| `TernaryOperation<I, P, B, O, Self>` | value | params + backIOp | Interpolate |
| `ReadOperation<RDT, P, O, TF, Self>` | thread coords | params | PerThreadRead, ReadSet |
| `WriteOperation<I, P, WDT, TF, Self>` | value+coords | params | TensorWrite, TensorSplit |
| `ReadBackOperation<...>` | thread coords | params + backIOp | Crop/Resize (complete) |
| `IncompleteReadBackOperation<...>` | — | params, backIOp later | Crop<NullType> pre-fusion |

Notes:
- Unary ops carry NO runtime params: everything is in the types. They are
  the cheapest to fuse and the easiest to test.
- Read/Write ops must also provide `num_elems_x/y/z(thread, opData)` (and
  `pitch` for memory ops); the executor derives grid dimensions from the
  read side's `getActiveThreads()`.
- ReadBack ops define output geometry: a Resize returns its target Size
  from num_elems_x/y regardless of the source size.

## The IncompleteReadBack pattern (BVF ops)

User-facing geometric ops (Crop, Resize, Warping) are declared with
`BackIOp = NullType`: the user builds them WITHOUT knowing the read
(`Crop<>::build(rect)`). The BackFuser later calls
`build(backIOp, selfIOp)` to complete them with the actual read. Implement
BOTH build() overloads:

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

## FusedOperation aliases — wrap IOps, not raw Operations

When an alias composes several ops into one (e.g. BGR2GRAY = reorder +
gray), `FusedOperation<...>` expects IOps (types with `::Operation`):

```cpp
// WRONG (ill-formed on instantiation: raw Ops have no ::Operation):
using type = FusedOperation<VectorReorder<I,2,1,0>, RGB2Gray<I,O>>;
// RIGHT:
using type = FusedOperation<Unary<VectorReorder<I,2,1,0>>, Unary<RGB2Gray<I,O>>>;
```
This exact bug shipped in four ColorConversion aliases (issues #244, fixed
in LTS-C++17) — the alias compiles fine until someone instantiates it, so
ALWAYS add a utest that calls `::build()` on every alias you define.

## Forwarding references in helpers

Never call `fuse_back<ExplicitArgs...>(...)`: explicit template arguments
turn `IOps&&...` into rvalue refs that cannot bind to const lvalues stored
in tuples (issue #245). Let deduction happen, e.g. through a lambda:

```cpp
apply([](const auto&... iOps) { return BackFuser::fuse_back(iOps...); }, tup);
```

## Testing a new op

1. Add a utest header under `utests/<area>/` using TestCaseBuilder:
```cpp
void testMyOp() {
    std::array<float, 2> in{2.f, 3.f};
    std::array<float, 2> expected{4.f, 6.f};
    TestCaseBuilder<MyOp<float>>::addTest(testCases, in, expected);
}
```
2. Register it in the test list before `STOP_ADDING_TESTS`.
3. Build and run: see the fkl-build-and-test skill.
4. If the op is an alias or has multiple build() overloads, instantiate
   EVERY public path in the test — template bugs hide until instantiation.

## Checklist before opening a PR

- [ ] FK_STATIC_STRUCT + Parent alias + DECLARE_*_PARENT
- [ ] exec() is FK_HOST_DEVICE_FUSE (runs on CPU backend too)
- [ ] num_elems_* / pitch for Read/Write/ReadBack ops
- [ ] both build() overloads for IncompleteReadBack ops
- [ ] values in params, types in templates
- [ ] utest instantiating every public alias/build path
- [ ] compiles warning-clean on nvcc AND clang (CUDA 12.x and 13.x)
