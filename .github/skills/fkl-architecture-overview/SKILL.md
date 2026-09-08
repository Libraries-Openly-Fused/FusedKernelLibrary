---
name: fkl-architecture-overview
description: "Use this skill to classify any FKL work as an Operation (Op), Data Parallel Pattern (DPP), or constexpr_lib. Trigger this before writing a new algorithm, when adapting an existing kernel to FKL standards, or when addressing PR review comments like 'should be a DPP' or 'not FKL'."
---

# FKL Architecture Overview

Classify the work before choosing an API or copying an implementation. An
algorithm name (filter, reduction, attention) does not determine its abstraction:
separate **per-thread data semantics** from **thread orchestration**.

## Load only the next relevant skill

| Task | Next skill |
|---|---|
| Write an application pipeline from existing pieces | [Using the library](../fkl-using-the-library/SKILL.md) |
| Build or compose IOps on the host | [Using operations](../fkl-using-operations/SKILL.md) |
| Add per-thread computation or addressing | [Implementing operations](../fkl-implementing-operations/SKILL.md) |
| Add traversal, tiling, or cooperative execution | [Implementing DPPs](../fkl-implementing-data-parallel-patterns/SKILL.md) |
| Choose VF, BVF, HF, or DHF | [Fusion techniques](../fkl-fusion-techniques/SKILL.md) |
| Allocate/wrap memory or diagnose layout | [Data structures](../fkl-data-structures/SKILL.md) |
| Expose pipelines to another language | [Language bindings](../fkl-language-bindings/SKILL.md) |
| Add tests or validate a change | [Build and test](../fkl-build-and-test/SKILL.md) |

Links are relative to this skill. Source paths below are repository-root-relative;
resolve them against the checkout, not the skill directory. Current headers and
tests define the API; the paper explains the design, not today's exact signatures.

## When to Use

- Before writing a new algorithm, to decide Op vs DPP vs cxp::.
- When a review says "this should be a DPP, not an Operation or standard Kernel",
  "this is not FKL / I can't fuse this", or "use standard Ops".
- When adapting an existing kernel to FKL standards, and having to refactor how 
a kernel reads, computes, and writes, using IOps.

## The boundary: who computes versus who schedules

> An Operation's invocation is single-thread code. A DPP decides which threads
> invoke which IOps, in what order, and how they exchange intermediate data.

| Layer | Owns | Does not own |
|---|---|---|
| **Operation (Op)** | Stateless type, static `exec()`, per-thread compute/read/write semantics | Kernel launch, thread scheduling, barriers or warp collectives |
| **IOp** | An Operation type plus runtime params/back-IOp, produced by `Op::build(...)` | A kernel or a new scheduling policy |
| **DPP** | Traversal, thread-to-data mapping, staging, synchronization, invocation of IOps | Hard-coded application transforms that can be supplied as IOps |
| **Executor / launcher** | Host-side validation, launch geometry, backend dispatch | Per-element application semantics |
| **constexpr_lib (`cxp::`)** | Standard-library-like utilities unavailable on GPU or as constexpr | Arbitrary FKL algorithms merely because they are constexpr |

A DPP need not use cooperation: `TransformDPP` schedules independent work.
Conversely, loops, multiple reads, multiple channels, or multiple mathematical
operands do not by themselves make an Operation a DPP. An Operation can be
invoked by many threads; each invocation must remain independent of their execution.

## Operation types

The Operation implementation signatures live in
[implementing operations](../fkl-implementing-operations/SKILL.md); the generic
IOp call-site signatures live in
[implementing DPPs](../fkl-implementing-data-parallel-patterns/SKILL.md). Do not
interchange these two contracts. The types:
`ReadType, WriteType, UnaryType, BinaryType, ReadBackType,
IncompleteReadBackType, TernaryType, IncompleteTernaryType, MidWriteType,
OpenType, ClosedType`.

## Decision procedure

1. Search existing operations, DPPs, and tests first. If they express the task,
   compose their IOps; do not create a custom kernel or DPP just to fuse a chain.
2. Is the missing part a standard-library-like utility? Consider `cxp::`.
   Otherwise keep FKL behavior in `fk::`.
3. Can it compute a value/address from its inputs in one thread, without
   coordinating other threads? That part is an **Operation**.
4. Does it choose traversal, distribute work, stage a shared tile, synchronize,
   or combine values across lanes? That part is a **DPP**. Extract reusable
   per-thread math and global reads/writes into Operations.
5. Define the DPP's IOp roles before coding: reads, compute/reducer/selector,
   output. Pass IOp **instances** as execution arguments (their types can be
   deduced template parameters). A role may be an `fk::Tuple` of IOps, as in
   `LinearFilterDPP`; keep them out of scheduling-only `Details`.
6. Choose the actual launcher and supported backends. Do not assume a new DPP
   works with `executeOperations<MyDPP>` without an Executor specialization.

## Classification examples and counterexamples

All source paths in this table start at `include/fused_kernel/`.

| Work | Classification and source |
|---|---|
| Scale, cast, reorder channels | Operation; `algorithms/basic_ops/arithmetic.h`, `cast.h`, `vector_ops.h` |
| Crop or interpolate by sampling a back-IOp | ReadBack Operation, even with multiple samples; `algorithms/image_processing/crop.h`, `interpolation.h` |
| Schedule the same read → compute → write chain for every pixel | Reuse `TransformDPP`; `core/execution_model/data_parallel_patterns.h` |
| Sort/select a median from one thread's local window | Operation (`MedianWindowSelect`); `algorithms/image_processing/median_filter.h` |
| Cooperatively load a halo and distribute windows to that selector | DPP (`MedianFilterDPP`), in the same file |
| Tiled convolution with image/coefficient reads and replaceable multiply/accumulate | DPP + IOps; `algorithms/image_processing/linear_filter.h` |
| Warp reduction, `__shfl_sync`, `__syncthreads`, collective `mma.sync` | DPP, never an Operation; see `algorithms/attention/` for specialized examples |

Renaming a monolithic CUDA kernel to `SomethingDPP` is not sufficient: callers
must be able to substitute/fuse the supported read, compute, and write roles.
The neighborhood DPPs are the preferred examples of this separation. Attention
currently has GPU-only, specialized launch and epilogue contracts; consult the
DPP skill before using it as a template for new code.

## Pitfalls

- `mma.sync` is warp-collective → it must be used inside a DPP, never an `Op`.
- `__nv_bfloat16` has no `VectorTraits`, so `PerThreadRead`/`PerThreadWrite`
  will not instantiate for it — supply a small Read/Write Op over a `RawPtr` in
  tests that use bf16.

## Review-fix mapping

| Review comment | Fix |
|---|---|
| "not FKL" / "breaks the FKL philosophy" | classify the missing behavior, reuse a DPP where possible, and expose data semantics as IOps |
| "should be a DPP, not an Operation/kernel" | move scheduling/cooperation into a DPP and extract reusable read/compute/write Operations |
| "this is cudaGraphs, no FKL Ops/DPPs" | rebuild on the FKL execution model; benchmark the DPP path |
| "faking the IOps, not using operator\|" | compose with the real `\|` / `.then()` fusion |
| "epilogue done by a single thread" | inspect output ownership; the DPP distributes outputs when appropriate, each epilogue IOp remains single-thread |
| "Max/Min/Sum should be standard Ops as args" | pass `Add`/`Max`/`Min` as IOps in exec function parameters |
| "duplicates code" | reuse an existing Op or a prior primitive |

## Verification

Before implementing, state: **“Reuse/add ___ because ___; reads ___, computes
___, writes ___; launched via ___ on ___ backend(s).”** Identify the closest
in-tree test. Validate both the standalone behavior and a fused composition;
building an IOp alone does not instantiate its execution body.

Background: [The Fused Kernel Library: A C++ API to Develop Highly-Efficient GPU
Libraries](https://arxiv.org/abs/2508.07071), especially the Operations/IOps and
Data Parallel Patterns discussion. The paper's ReduceDPP is a design example,
not a promise of a current public API or arbitrary DPP-to-DPP fusion.

Skill maintenance: follow [GitHub's SKILL.md structure](https://docs.github.com/en/copilot/how-tos/copilot-on-github/customize-copilot/customize-cloud-agent/add-skills)
with YAML `name` and a task-triggering `description`. Keep this classification
guide short; link to the owning skill and executable source examples rather than
duplicating implementation templates across skills.
