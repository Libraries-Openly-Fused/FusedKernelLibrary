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
| ReadType     | `T   IOp::Operation::exec(thread, iop)` |
| WriteType    | `void IOp::Operation::exec(thread, value, iop)` |
| UnaryType    | `O   IOp::Operation::exec(input)` |
| BinaryType   | `O   IOp::Operation::exec(input, iop)` |
| ReadBackType | `O   IOp::Operation::exec(thread, iop)` |
| TernaryType  | `O   IOp::Operation::exec(input, iop)` |
| MidWriteType | `In  IOp::Operation::exec(thread, input, iop)` (writes AND forwards) |
| OpenType     | `O   IOp::Operation::exec(thread, input, iop)` |
| ClosedType   | `void IOp::Operation::exec(thread, iop)` |

Key consumer rules:
- The DPP invokes through `IOp::Operation::exec(...)` — the IOp type's `::Operation`
  static, passing `iop` for the data, NOT the raw op struct.
- **Read** takes the `Point thread` + `params` and returns the value.
- **Write** takes `thread`, the `value` to store, and `params`.
- **Unary/Binary** are pure register compute: no `thread`, just the input
  (+ `params` for Binary). These are the epilogue building blocks.

## Invoking the fused write inside a DPP

Pass the fused IOp to the DPP and invoke it with the WHOLE IOp (the epilogue
is applied inside it, then the destination write):

```cpp
template <typename Inputs, typename Output>
FK_HOST_DEVICE_FUSE void exec(const Details& d, const Inputs& inputs,
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
  `get<0>/get<1>`.
- **Output as the fused write:** invoke `Output::Operation::exec(thread, value,
  output)` — the whole fused IOp, NOT `.params`. This is what lets the epilogue
  chain run in-register before the single global write.

## Pitfalls

- Passing `output.params` instead of `output` to a fused write's exec() routes
  through the wrong fold path. Pass the whole IOp.
- A `Read` as the first link of `.then(...)` produces a Read→Write CLOSED
  fused op, not the compute→write epilogue you want — if there is no epiloge wimply
  use the Write IOp.
- The operand tuple must be `fk::Tuple` (via `make_tuple`), not two separate
  exec parameters — matches the DPP definition in
  `fkl-implementing-data-parallel-patterns`.

## See also

- `fkl-implementing-operations` — authoring the op structs and the IOp-form table.
- `fkl-implementing-data-parallel-patterns` — the DPP `exec(details, inputs, compute,
  output)` contract these invocations live inside.
- `fkl-fusion-techniques` — understanding fusion techniques for discussion.
