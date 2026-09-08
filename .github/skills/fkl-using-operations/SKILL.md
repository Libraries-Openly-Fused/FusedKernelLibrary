---
name: fkl-using-operations
description: Invoke and compose FKL Operations and Instantiable Operations (IOps) from the CONSUMER side (Host API). Covers how to create inputs, compute chains, and outputs via ::build(), and how to pass them sequentially to the Executor. Use when writing host code to launch FKL pipelines.
---

# Using FKL Operations and IOps (consumer side)

Use this skill for host composition, not Operation or DPP implementation.
An Operation is a static type; `Op::build(...)` returns the IOp instance passed
to the executor. Changing runtime values does not change the pipeline's type.

## Creating the operations and executing them in order (Host Side)

This helper expects initialized input, matching output dimensions, and memory
accessible to the default backend. Include the operation headers explicitly.

```cpp
#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>

void addFive(fk::Stream& stream, fk::RawPtr<fk::ND::_2D, float> in,
              fk::RawPtr<fk::ND::_2D, float> out) {
    const auto read = fk::PerThreadRead<fk::ND::_2D, float>::build(in);
    const auto add = fk::Add<float>::build(5.f);
    const auto write = fk::PerThreadWrite<fk::ND::_2D, float>::build(out);
    fk::executeOperations<fk::TransformDPP<>>(stream, read, add, write);
}
```

## Composition rules

- A Transform pipeline starts with a complete read and ends with a write.
  Adjacent value types must match; insert `Cast` or `SaturateCast` deliberately.
- `read.then(Crop<>::build(rect))` completes the ReadBack operation with its
  source. Passing read and crop separately to the executor also performs
  backwards fusion. Include the crop header for this expression.
- `.then()` composes IOps without launching. For example,
  `Add<float>::build(5.f).then(write)` is a fused output accepting a float.
  Do not confuse this host composition with the DPP's per-thread `operator|`
  execution fold.
- A supported `std::array` builder creates a batch IOp. Match output planes to
  that batch; an arbitrary container does not automatically enable HF.
- Container overloads can infer reads/writes; inspect
  `include/fused_kernel/fused_kernel.h` and `core/execution_model/executors.h`
  under `include/fused_kernel/` before choosing an overload.
- An arbitrary DPP needs its own `Executor` contract. Do not substitute an
  undefined `MyDPP` into an otherwise valid Transform example.

Reference: `utests/core/execution_model/utest_executors.h` exercises automatic
back-fusion and explicit composition; `tests/algorithm/test_crop.h` checks
completed and batched ReadBack types and output geometry.

## See also

- [Using the library](../fkl-using-the-library/SKILL.md) — application recipes and lifetimes.
- [Implementing operations](../fkl-implementing-operations/SKILL.md) — static Operation definitions.
- [Implementing DPPs](../fkl-implementing-data-parallel-patterns/SKILL.md) — execution-side IOp calls.
- [Fusion techniques](../fkl-fusion-techniques/SKILL.md) — selecting VF, BVF, HF, or DHF.