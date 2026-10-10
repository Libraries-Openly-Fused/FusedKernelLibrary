---
name: fkl-architecture-overview
description: "Use this skill to classify any FKL work as an Operation (Op), Data Parallel Pattern (DPP), or constexpr_lib. Trigger this before writing a new algorithm, when adapting an existing kernel to FKL standards, or when addressing PR review comments like 'should be a DPP' or 'not FKL'."
---

# FKL Architecture Overview

Start with existing IOps and an existing DPP. Add an abstraction only for the
part of the task that cannot already be expressed by composition.

## Choose the next skill

| Task | Skill |
|---|---|
| Write an application pipeline | [Using the library](../fkl-using-the-library/SKILL.md) |
| Build and combine IOps on the host | [Using operations](../fkl-using-operations/SKILL.md) |
| Implement per-thread data behavior | [Implementing operations](../fkl-implementing-operations/SKILL.md) |
| Implement traversal or cooperation | [Implementing DPPs](../fkl-implementing-data-parallel-patterns/SKILL.md) |
| Select a fusion technique | [Fusion techniques](../fkl-fusion-techniques/SKILL.md) |
| Allocate or wrap memory | [Data structures](../fkl-data-structures/SKILL.md) |
| Design a language binding | [Language bindings](../fkl-language-bindings/SKILL.md) |
| Add or run tests | [Build and test](../fkl-build-and-test/SKILL.md) |

Skill links are relative to this directory. Source paths in these skills are
relative to the repository root; resolve them against the checkout.

## When to Use

- Before writing a new algorithm, to decide Op vs DPP vs cxp::.
- When a review says "this should be a DPP, not an Operation or standard Kernel",
  "this is not FKL / I can't fuse this", or "use standard Ops".
- When adapting an existing kernel and separating how it reads, computes, and
  writes from how it schedules threads.

## Separate data behavior from execution

> An Operation invocation is single-thread code. A DPP determines which threads
> invoke the supplied IOps, in which order, and with what coordination.

| Abstraction | Responsibility | Starting point under `include/fused_kernel/` |
|---|---|---|
| Operation | Stateless type with static per-thread computation, addressing, or memory access | `algorithms/basic_ops/arithmetic.h`, `algorithms/image_processing/crop.h` |
| IOp | Operation type plus runtime parameters and, where applicable, a back-IOp | `core/execution_model/operation_model/instantiable_operations.h` |
| DPP | Traversal, thread mapping, synchronization, and routing data through IOps | `core/execution_model/data_parallel_patterns.h` |
| Executor | Host-side composition and backend launch | `core/execution_model/executors.h` |
| constexpr_lib | Standard-library-like functionality unavailable on GPU or as constexpr, in `cxp::` | `core/constexpr_libs/` |

`TransformDPP` is a DPP although its threads do not cooperate. Conversely,
interpolation can sample several values through a back-IOp while remaining an
Operation. Loops, multiple operands, and multiple reads are not sufficient
reasons to create a DPP. FKL-specific algorithms belong in `fk::`, not `cxp::`.

## Classify the responsibility, not the algorithm's name

| Requested behavior | Classification | Why |
|---|---|---|
| Crop → resize → normalize → planar output | Compose existing IOps with `TransformDPP` | No new data semantics or traversal are needed |
| A new scalar/vector conversion | Operation | Each invocation transforms its input independently |
| Sample four source pixels and interpolate one value | Operation (`InterpolateComplete` in `algorithms/image_processing/interpolation.h`) | One thread samples a back-IOp; multiple reads are not cooperation |
| Return a constant when a coordinate is outside the source | Operation (`BorderReader` in `algorithms/image_processing/border_reader.h`) | The branch defines this invocation's read result, not the schedule |
| Traverse independent output coordinates | DPP (`TransformDPP`) | Mapping work to threads is a DPP responsibility even without barriers |
| Cooperatively combine values across a block | DPP plus supplied read/combine/write IOps | The DPP organizes participants and communication; IOps define data behavior |
| Choose different operation sequences for different planes | Reuse `DivergentBatchTransformDPP` | Sequence selection already has a DPP/executor implementation |

Paths in the table are under `include/fused_kernel/`. The cooperative example
is a design case, not a claim that a reduction DPP is a shipped API.

Ask two questions before introducing a type:

1. **Does this define one invocation's data result or memory access?** Keep it
   in an Operation. Use the supplied logical coordinate; do not derive it from
   CUDA built-ins inside the Operation.
2. **Does this decide which invocations run, their ownership, or how they
   communicate?** That part belongs in a DPP. Split mixed implementations
   rather than moving an entire algorithm into either abstraction.

A local loop or data-dependent branch is not by itself thread orchestration.
Conversely, running a whole array algorithm in one thread is not a reason to
hide traversal in an Operation.

## Operation types

The implementation signature table lives in
[implementing operations](../fkl-implementing-operations/SKILL.md).
The corresponding IOp call-site table belongs to
[implementing DPPs](../fkl-implementing-data-parallel-patterns/SKILL.md).
Do not confuse the two forms. The declared types are:
`ReadType, WriteType, UnaryType, BinaryType, ReadBackType,
IncompleteReadBackType, TernaryType, IncompleteTernaryType, MidWriteType,
OpenType, ClosedType`.

## Procedure

1. Locate the nearest existing pipeline and test. Reuse `TransformDPP` for a
   read → compute → write chain rather than introducing a custom kernel.
2. If the missing behavior computes or samples independently in one thread,
   implement an Operation. Start from `Add`, `Cast`, `Crop`, or `InterpolateComplete`.
3. If it changes traversal, work distribution, shared staging, or communication
   across threads, implement a DPP. Keep reusable data transformations in IOps.
4. Specify the DPP's read, compute, and write roles. Pass IOp instances as
   execution arguments; their types may be deduced template parameters. A role
   may contain a tuple of IOps. Keep scheduling details separate from these roles.
5. Identify the executor, backend specializations, and launch geometry. Naming
   a struct `DPP` does not make `executeOperations` dispatch to it automatically.
6. Check standalone behavior and a fused composition. Test CPU and CUDA paths
   for new general-purpose patterns.

## Check that the abstraction is composable

- Can a caller change the read, compute, or write IOp without editing the DPP's
  scheduling code, within its documented type and semantic constraints?
- Does a supplied fused read actually execute its attached computation, rather
  than having the DPP extract a raw pointer and bypass it?
- Does a supplied compute-plus-write output actually execute both parts?
- Are traversal and synchronization absent from Operations, and reusable
  arithmetic/addressing absent from the DPP's scheduling logic?
- Is there a real executor path and an executed test, not just a struct named
  `DPP` or a construction-only example?

Use these checks when a review says "not FKL" or "should be a DPP". Renaming a
kernel or accepting IOps that are never invoked does not fix the boundary.

## Verification

Before coding, state: **“Reuse/add ___ because ___; reads ___, computes ___,
writes ___; launched through ___ on ___.”** Use current headers and tests for
API details, not a copied template or a name from the paper.

Background: [The Fused Kernel Library: A C++ API to Develop Highly-Efficient GPU
Libraries](https://arxiv.org/abs/2508.07071). The paper describes the architecture,
not a guarantee that every illustrated DPP is a shipped API.

Read §IV-A/B for compute/memory Operations and §IV-C, especially Figures 13–14,
for DPPs that connect supplied IOps. Figure 14 separates a reduction's thread
organization from its combining Operations; it is a conceptual example, not
a template to copy into the current API. The consumer/author skill split
corresponds to the paper's Library Users/Methodology Users distinction.
