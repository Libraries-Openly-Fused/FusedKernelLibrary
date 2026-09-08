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
   implement an Operation. Start from `Add`, `Cast`, `Crop`, or `Interpolation`.
3. If it changes traversal, work distribution, shared staging, or communication
   across threads, implement a DPP. Keep reusable data transformations in IOps.
4. Specify the DPP's read, compute, and write roles. Pass IOp instances as
   execution arguments; their types may be deduced template parameters. A role
   may contain a tuple of IOps. Keep scheduling details separate from these roles.
5. Identify the executor, backend specializations, and launch geometry. Naming
   a struct `DPP` does not make `executeOperations` dispatch to it automatically.
6. Check standalone behavior and a fused composition. Test CPU and CUDA paths
   for new general-purpose patterns.

## Review-fix mapping

| Review comment | Fix |
|---|---|
| "not FKL" / "breaks the FKL philosophy" | classify the missing behavior and expose data semantics through composable IOps |
| "should be a DPP, not an Operation/kernel" | separate traversal/cooperation from reusable read, compute, and write Operations |
| "this is cudaGraphs, no FKL Ops/DPPs" | rebuild on the FKL execution model; benchmark the DPP path |
| "faking the IOps, not using operator\|" | compose with the real `\|` / `.then()` fusion |
| "epilogue done by a single thread" | check output ownership and distribute independent outputs in the DPP; each IOp invocation stays single-thread |
| "Max/Min/Sum should be standard Ops as args" | pass `Add`/`Max`/`Min` as IOps in exec function parameters |
| "duplicates code" | reuse an existing Op or a prior primitive |

## Verification

Before coding, state: **“Reuse/add ___ because ___; reads ___, computes ___,
writes ___; launched through ___ on ___.”** Use current headers and tests for
API details, not a copied template or a name from the paper.

Background: [The Fused Kernel Library: A C++ API to Develop Highly-Efficient GPU
Libraries](https://arxiv.org/abs/2508.07071). The paper describes the architecture,
not a guarantee that every illustrated DPP is a shipped API.
