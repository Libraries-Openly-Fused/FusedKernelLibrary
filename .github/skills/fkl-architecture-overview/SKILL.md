---
name: fkl-architecture-overview
description: "Classify FKL work as an existing pipeline, a single-thread Operation, a thread-organizing Data Parallel Pattern (DPP), or a constexpr_lib helper. Use before adding algorithms, porting kernels, or responding to 'should be a DPP' or 'not fusible' reviews; then follow the relevant implementation or consumer skill."
---

# FKL Architecture Overview

Use this decision guide before writing code. Classify the implementation, not
the algorithm's name. Prefer composing existing components over adding new ones.

## Choose the next skill

| Task | Read next |
|---|---|
| Write an application / port an image pipeline | [Using the library](../fkl-using-the-library/SKILL.md) |
| Build or compose IOps on the host | [Using operations](../fkl-using-operations/SKILL.md) |
| Add per-thread math, memory access, or sampling | [Implementing operations](../fkl-implementing-operations/SKILL.md) |
| Add thread scheduling, cooperation, or a launch pattern | [Implementing DPPs](../fkl-implementing-data-parallel-patterns/SKILL.md) |
| Choose VF / BVF / HF / DHF | [Fusion techniques](../fkl-fusion-techniques/SKILL.md) |
| Allocate or wrap pointers; choose memory layouts | [Data structures](../fkl-data-structures/SKILL.md) |
| Generate pipelines from another language | [Language bindings](../fkl-language-bindings/SKILL.md) |
| Configure, compile, or add/run tests | [Build and test](../fkl-build-and-test/SKILL.md) |

Load only relevant follow-up skills. Exact APIs come from current headers and
tests; the paper explains the design, not current C++ signatures.

## The four layers

| Layer | Responsibility | Example |
|---|---|---|
| Operation (Op) | Stateless, static per-thread read, compute, or write | `Mul<float>` |
| Instantiable Operation (IOp) | Op type plus runtime parameters, created by `build()` | `Mul<float>::build(2.f)` |
| Data Parallel Pattern (DPP) | Thread mapping, iteration, cooperation, and routing through IOps | `TransformDPP<>` |
| Executor | Host fusion/setup and dispatch for a DPP/backend | `executeOperations<TransformDPP<>>(stream, read, compute, write)` |

Types select generated code; IOp **instances** supply runtime values. DPP functions
are templated on IOp types, normally deduced from instance arguments. "Pass IOps,
not raw Operations" does **not** mean "never use IOp template parameters."

## Decision procedure

1. **Can existing IOps and a DPP express the task?** Compose them. Crop + resize +
   normalization does not need a new DPP or a handwritten CUDA kernel.
2. **Is the missing piece work one thread executes independently?** Add an
   Operation. It may loop, read multiple coordinates, or return a vector/tuple.
   It must not coordinate threads, choose the launch grid, or launch a kernel.
3. **Is the missing piece thread organization or cooperation?** Add/extend a DPP.
   Cross-thread reduction, shared-tile staging, barriers, shuffles and
   warp-collective `mma.sync` belong here, not in an Operation.
4. **Is it a GPU/constexpr replacement for standard functionality rather than
   a pipeline component?** Put that helper in `core/constexpr_libs/`, namespace
   `cxp`. Ordinary FKL algorithms and infrastructure belong in `fk`.

> An Operation executes within one thread. Running the same Operation on a
> million threads does not turn it into a DPP. Conversely, a DPP need not use
> barriers: `TransformDPP` organizes independent threads.

### Classify by implementation, not by name

| Requested work | Classification and reason |
|---|---|
| Scale, clamp, color conversion | Compute Op; one thread transforms its input |
| Crop, bilinear resize, neighborhood sampling | ReadBack Op when each thread independently samples its upstream read; multiple pixels are not cooperating threads |
| Sum channels in one `float4` | Compute Op; all channels already belong to one thread |
| Sum a row using warp shuffles | DPP for cooperation + IOps for reads, combination, and output |
| Convolution using a shared tile | DPP for tile ownership/synchronization + Ops for reusable per-thread work |
| Matrix multiplication using `mma.sync` | DPP; a tensor-core instruction is warp-collective |
| Batch of crops with the same processing | Existing `TransformDPP` + batch ReadBack IOps (HF) |
| Different pipelines on different z planes | Existing `DivergentBatchTransformDPP` (DHF), not an Op with hidden scheduling |

## DPP + Operations, not DPP instead of Operations

For a cooperative algorithm, identify input IOps (prologues), reusable per-thread
compute, and output writes (epilogues) separately from scheduling. LinearFilter
is a useful model: two reads, replaceable multiply/accumulate IOps, and a write,
with CPU traversal and CUDA shared-tile implementations.
Merely renaming a monolithic pointer-based kernel `MyDPP` does not make it
composable.

For new DPPs, keep runtime scheduling details separate from IOps, use IOp
read/write boundaries for user data, and use reusable compute IOps where
separable. Hardware-coupled tensor-core math stays inside the DPP.

Distinguish that design target from current coverage: the DPPs in
`algorithms/attention/` are CUDA-only, use dedicated static launch APIs, and
some retain raw output/workspace pointers and internal arithmetic. They are
not drop-in `Executor<MyDPP>` implementations or templates for every new DPP.
Plan CPU semantics and explicitly document/test any backend restriction;
a CPU reference oracle is not a CPU DPP implementation.

## Review-fix mapping

| Review comment | Fix |
|---|---|
| "not fusible" | Check for bypassed IOp boundaries; test a non-identity prologue and epilogue |
| "should be a DPP" | Move scheduling/cooperation out of the Op; retain reusable per-thread work as Ops |
| "use standard Ops" | Pass built compute IOps instead of hard-coding replaceable arithmetic |
| "faking IOps" | Use real `build()` instances and fusion, not unused wrappers around pointer-based code |
| "epilogue done by a single thread" | Check output ownership and parallelism in the DPP; do not add barriers inside an Op |

## Verification

Before implementing, state:
- Classification and reason (what, if anything, crosses thread boundaries).
- Existing components reused and the IOp read/compute/write contract.
- Supported backends and actual host launch API.
- Tests for numerical correctness **and** composition, including changed runtime params.

## Source map

- [Operation signatures](../../../include/fused_kernel/core/execution_model/operation_model/operation_types.h)
  and [IOp wrappers](../../../include/fused_kernel/core/execution_model/operation_model/instantiable_operations.h).
- [Transform / divergent DPPs](../../../include/fused_kernel/core/execution_model/data_parallel_patterns.h)
  and [executors](../../../include/fused_kernel/core/execution_model/executors.h).
- [Crop](../../../include/fused_kernel/algorithms/image_processing/crop.h):
  independent ReadBack sampling, not thread cooperation.
- [LinearFilter DPP](../../../include/fused_kernel/algorithms/image_processing/linear_filter.h)
  and [composition tests](../../../tests/image_processing/test_linear_filter_dpp.h).
- [Paper, sections IV-A–IV-D](https://arxiv.org/html/2508.07071v1#S4):
  per-thread compute/memory Ops, IOps, DPP thread organization, and host interface.
- [GitHub skill format](https://docs.github.com/en/copilot/how-tos/copilot-on-github/customize-copilot/customize-cloud-agent/add-skills):
  named directories containing `SKILL.md`, YAML `name`/`description`, task-specific
  instructions and linked resources. Keep detailed contracts in their owning skill.
