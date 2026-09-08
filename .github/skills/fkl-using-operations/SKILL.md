---
name: fkl-using-operations
description: Build and compose existing FKL IOps on the host using build, then, fuse, and executeOperations. Use for host API overloads, read/compute/write type mismatches, or fused prologues and epilogues. For full application recipes use fkl-using-the-library; for new Operation or DPP implementations use the authoring skills.
---

# Using FKL Operations and IOps (consumer side)

[Implementing operations](../fkl-implementing-operations/SKILL.md) covers how to AUTHOR an Operation struct. This
skill covers the other half: how to **create** Instantiable Operations (IOps) on the host, **compose** them into a pipeline, and pass them to an Executor. 

For device-side IOp invocation, see
[implementing DPPs](../fkl-implementing-data-parallel-patterns/SKILL.md).
An Op is a stateless type; `build()` returns the IOp carrying runtime values.
Neither one schedules threads: the chosen DPP does.

## Creating the operations and executing them in order (Host Side)

Before launching TransformDPP, build the IOps. Its executor handles backward
fusion and dispatch. Fragment below assumes initialized, compatible float
buffers `in_ptr`, `out_ptr`, and a matching `stream`:

```cpp
// 1. Build the input Read IOp
auto input = fk::PerThreadRead<fk::ND::_2D, float>::build(in_ptr);

// 2. Build the compute IOp
auto compute_iop = fk::Add<float>::build(5.0f);

// 3. Build the destination write IOp
auto write_iop = fk::PerThreadWrite<fk::ND::_2D, float>::build(out_ptr);

// 4. Pass them sequentially to the Executor, providing the DPP as a template parameter
fk::executeOperations<fk::TransformDPP<>>(stream, input, compute_iop, write_iop);
```

Include `<fused_kernel/fused_kernel.h>` and the headers for the Operations used.
First IOp must be a complete Read/ReadBack, last a Write; adjacent value types
must match. Incomplete `Crop<>`/`Resize<>` become complete when the executor's
BackFuser attaches the preceding read.

## Explicit composition without launching

```cpp
const auto compute = fk::Mul<float>::build(2.f).then(fk::Add<float>::build(1.f));
const auto prologue = input.then(compute);       // Read IOp
const auto epilogue = compute.then(write_iop);   // Write IOp
```

`.then()`, `a & b`, and `fk::fuse(a, b, ...)` compose host-side IOps; they do not
launch kernels. `value | computeIOp` is execution-side value flow, not the host
composition operator. `buildOperationSequence(...)` packages a complete
sequence for DHF; it does not launch it.

Use these forms when a specialized DPP accepts fused read/write roles. Check its
actual launcher and role contracts: not every DPP uses the generic
`executeOperations` API or accepts a Write epilogue. Do not pass raw Operation
types where IOp instances are required.

Changing a `build()` value preserves the IOp type. Changing dtype, the chain's
Operation types, or `std::array` batch capacity changes the kernel specialization.
Keep referenced buffers alive until asynchronous GPU work finishes.

## See also

- [Using the library](../fkl-using-the-library/SKILL.md) — complete pipeline recipes and memory/stream setup.
- [Fusion techniques](../fkl-fusion-techniques/SKILL.md) — batches, ReadBack stacks, and DHF selection.
- [Build and test](../fkl-build-and-test/SKILL.md) — verify both IOp construction and execution.
- Source: `include/fused_kernel/core/execution_model/operation_model/iop_fuser.h`,
  `instantiable_operations.h`, and `tests/operation/test_fused_write_epilogue_repro.h`
  (paths are repository-root-relative; the second header is beside the first).