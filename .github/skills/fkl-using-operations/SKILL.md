---
name: fkl-using-operations
description: Invoke and compose FKL Operations and Instantiable Operations (IOps) from the CONSUMER side (Host API). Covers how to create inputs, compute chains, and outputs via ::build(), and how to pass them sequentially to the Executor. Use when writing host code to launch FKL pipelines.
---

# Using FKL Operations and IOps (consumer side)

`fkl-implementing-operations` covers how to AUTHOR an Operation struct. This
skill covers the other half: how to **create** Instantiable Operations (IOps) on the host, **compose** them into a pipeline, and pass them to an Executor. 

*(Note: For the device-side implementation of how a DPP actually invokes these operations internally, see the `fkl-implementing-data-parallel-patterns` skill).*

## Creating the operations and executing them in order (Host Side)

Before launching the DPP, the host code must build the IOps. The executor handles fusing the sequence automatically:

```cpp
// 1. Build the input Read IOp
auto input = fk::PerThreadRead<fk::ND::_2D, float>::build(in_ptr);

// 2. Build the compute IOp (the epilogue)
auto compute_iop = fk::Add<float>::build(5.0f);

// 3. Build the destination write IOp
auto write_iop = fk::PerThreadWrite<fk::ND::_2D, float>::build(out_ptr);

// 4. Pass them sequentially to the Executor, providing the DPP as a template parameter
fk::executeOperations<MyDPP>(stream, input, compute_iop, write_iop);
```

## See also

- `fkl-implementing-operations` — authoring the op structs.
- `fkl-implementing-data-parallel-patterns` — the device-side DPP `exec(...)` contract where these IOps are actually invoked.
- `fkl-fusion-techniques` — understanding fusion techniques for discussion.