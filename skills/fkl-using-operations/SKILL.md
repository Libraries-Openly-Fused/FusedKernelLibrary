---
name: fkl-using-operations
description: Invoke and compose FKL Operations and Instantiable Operations (IOps) from the CONSUMER side — how to call an op's exec() correctly for each operation type, how to fuse a compute chain with a write IOp (epilogue.then(D)), and how to invoke that fused write from inside a DPP. Use when writing a DPP or kernel that reads inputs, applies a compute chain, and writes a result through IOps, or when an IOp invocation / fused write fails to compile. Complements fkl-implementing-operations (which is about AUTHORING ops) with the call-site contract.
---

# Using FKL Operations and IOps (consumer side)

`fkl-implementing-operations` covers how to AUTHOR an Operation struct. This
skill covers the other half: given an Instantiable Operation (IOp), how do you
**invoke** it correctly, and how do you **compose** a compute chain with a
write so a DPP can emit its result through a single fused IOp.

## The invocation contract is the IOp-form table

Every IOp's `exec()` signature is fixed by its operation type. This is the
SAME table as in `fkl-implementing-operations` — read it here as the call-site
contract (what arguments YOU must pass), not as an authoring spec:

| Operation Type | exec call site |
|---|---|
| ReadType     | `T   IOp::Operation::exec(thread, iop.params)` |
| WriteType    | `void IOp::Operation::exec(thread, value, iop.params)` |
| UnaryType    | `O   IOp::Operation::exec(input)` |
| BinaryType   | `O   IOp::Operation::exec(input, iop.params)` |
| ReadBackType | `O   IOp::Operation::exec(thread, iop.params, backIOp)` |
| TernaryType  | `O   IOp::Operation::exec(input, iop.params, backIOp)` |
| MidWriteType | `In  IOp::Operation::exec(thread, input, iop.params)` (writes AND forwards) |
| OpenType     | `O   IOp::Operation::exec(thread, input, iop.params)` (FusedOperation) |
| ClosedType   | `void IOp::Operation::exec(thread, iop.params)` (FusedOperation) |

Key consumer rules:
- You invoke through `IOp::Operation::exec(...)` — the IOp type's `::Operation`
  static, passing `iop.params` for the data, NOT the raw op struct.
- **Read** takes the `Point thread` + `params` and returns the value.
- **Write** takes `thread`, the `value` to store, and `params`.
- **Unary/Binary** are pure register compute: no `thread`, just the input
  (+ `params` for Binary). These are the epilogue building blocks.

## Composing a compute→write epilogue: `epilogue.then(D)`

A DPP should not hardcode a destination write. The caller fuses the epilogue
IOp chain with the destination write IOp and passes ONE fused IOp:

```cpp
// destination write (D is the output matrix/image, a real fk write IOp)
const auto dWrite = PerThreadWrite<ND::_2D, float>::build(D);

// epilogue as an IOp chain, fused with the write:
const auto output = Cast<float,float>::build().then(dWrite);              // identity
const auto output = Mul<float>::build(2.f).then(Add<float>::build(0.5f))  // scale+bias
                                          .then(dWrite);
```

- The first link of the chain must be a **compute** IOp (Unary/Binary) — NOT a
  Read (a Read in front makes it a full Read→…→Write pipeline, a different
  fused form). For a pass-through epilogue use `Cast<T,T>`.
- The result is a write-type fused IOp: `isAnyWriteType<decltype(output)>`
  is `true`.

## Invoking the fused write inside a DPP

Pass the fused IOp to the DPP and invoke it with the WHOLE IOp (the epilogue
is applied inside it, then the destination write):

```cpp
template <typename Inputs, typename Output>
static __device__ void exec(const Details& d, const Inputs& inputs,
                            const Output& output) {
    const auto& A = get<0>(inputs);          // inputs is fk::Tuple<AIOp, BIOp>
    const auto& B = get<1>(inputs);
    ...
    // result is the in-register accumulator; the epilogue + store happen in the
    // fused write IOp. Pass the whole `output`, NOT output.params:
    Output::Operation::exec(thread, result, output);
}
```

- **Inputs as a tuple:** read operands arrive as `const fk::Tuple<AIOp,BIOp>&`
  (built caller-side with `make_tuple(aIOp, bIOp)`), unpacked with
  `get<0>/get<1>`. Use `std::decay_t<decltype(A)>::Operation` to name the op.
- **Output as the fused write:** invoke `Output::Operation::exec(thread, value,
  output)` — the whole fused IOp, NOT `.params`. This is what lets the epilogue
  chain run in-register before the single global write.

## Pitfalls

- Passing `output.params` instead of `output` to a fused write's exec() routes
  through the wrong fold path. Pass the whole IOp.
- A `Read` as the first link of `.then(...)` produces a Read→Write CLOSED
  fused op, not the compute→write epilogue you want — start the chain with a
  compute op (`Cast<T,T>` for identity).
- The operand tuple must be `fk::Tuple` (via `make_tuple`), not two separate
  exec parameters — matches the DPP definition in
  `fkl-implementing-data-parallel-patterns`.

## See also

- `fkl-implementing-operations` — authoring the op structs and the IOp-form table.
- `fkl-implementing-data-parallel-patterns` — the DPP `exec(details, inputs,
  output)` contract these invocations live inside.
- `fkl-fusion-techniques` — how `executeOperations` wires read→compute→write.
