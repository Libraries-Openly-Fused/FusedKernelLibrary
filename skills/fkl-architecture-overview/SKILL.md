---
name: fkl-architecture-overview
description: "Classify FKL work as Op, DPP, policy, or fused IOp."
version: 0.1.0
metadata:
  hermes:
    tags: [GPU, CUDA, FusedKernelLibrary, Architecture]
---

# FKL Architecture Overview

A fast classification guide for the Fused Kernel Library: given a piece of work,
decide whether it is an **Operation**, a **Data Parallel Pattern (DPP)**, a
**layout/scheduler policy**, or a **fused IOp**, and route its data correctly.
It does NOT replace `fkl-implementing-operations`,
`fkl-implementing-data-parallel-patterns`, or `fkl-fusion-techniques` (those are
authoritative and go deeper) — it is the 60-second "what am I building?" filter
to apply before reaching for them, plus the most common review-fix mappings.

## When to Use

- Before writing a new algorithm, to decide Op vs DPP vs policy.
- When a review says "this should be a DPP, not an Operation/kernel",
  "this is not FKL / I can't fuse this", or "use standard Ops as args".
- When deciding how a kernel's read, compute, and write should flow through IOps.

## The single classifying rule

> An Operation is strictly ALWAYS single-thread code. If more than one thread
> must participate in the code, that belongs in a DPP.

- **Op** — one thread's work; stateless struct, static `exec()`. Single-thread.
- **DPP** — anything needing >1 thread to cooperate: shared memory,
  `__syncthreads`, `__shfl`, warp-collective `mma.sync`, cp.async tile staging.
  A DPP composes standard Ops and routes data through IOps.
- **IOp** (Instantiable Operation) — an Op plus its params, produced by
  `::build()`, invoked as `IOp::Operation::exec(...)`, never the raw Op struct.
- **Layout / scheduler / rasterizer policy** — pure compile-time index math
  (`offset(row,col)`, warp→tile, CTA→tile), host+device, no Ops or pointers.
  This is a policy, NOT a DPP.
- **Tile** is not an abstraction: a `__shared__` buffer addressed by a layout
  policy. There is no `Tile` struct and no `TileRead`/`TileWrite` Op.

## Operation types

The full IOp-form table (with exact `exec()` signatures) lives in
`fkl-implementing-operations`. The types:
`ReadType, WriteType, UnaryType, BinaryType, ReadBackType,
IncompleteReadBackType, TernaryType, IncompleteTernaryType, MidWriteType,
OpenType, ClosedType`.

## Procedure

1. Does correctness require more than one thread to cooperate? Yes → **DPP**;
   no → **Op**.
2. Is it pure index remap (block/warp → output tile)? → **policy**, not a DPP.
3. A DPP takes its read as a **Read IOp** (prologue), its write as a **Write
   IOp** (epilogue), and intermediate compute as IOps. The reduction/combine
   operator (`Add`, `Max`, `Min`, `Mul`) is a STANDARD Op passed as a template
   argument — not a bespoke combine functor.
4. DPP shape: `exec(details, readIOp, computeIOp, writeIOp)` — IOps are separate
   exec parameters, not packed into a struct; a `*Details` struct carries only
   external scalars (dims, identity, mask). Provide both `ParArch::GPU_NVIDIA`
   and `ParArch::CPU` specialisations; keep `*Details` and the CPU spec outside
   any `#if defined(__NVCC__)` guard.
5. For an output written through a fused epilogue, fuse the epilogue chain with
   the destination write (`epilogue.then(D)`) and invoke the whole fused IOp:
   `Output::Operation::exec(thread, value, output)`. The first link of the
   `.then()` chain must be a compute op (`Cast<T,T>` for identity), not a Read.
6. Verify against a CPU/fp64 oracle; see `fkl-build-and-test`.

## Pitfalls

- `mma.sync` is warp-collective → it is a DPP, never an `MmaOp`.
- A non-broadcast block reduction passes warp/block tests but fails the grid
  test; stage the finished result and return it on lane0/warp0.
- `__nv_bfloat16` has no `VectorTraits`, so `PerThreadRead`/`PerThreadWrite`
  will not instantiate for it — supply a small Read/Write Op over a `RawPtr` in
  tests that use bf16.
- A standalone cudaGraph benchmark that open-codes a kernel is not FKL; a
  benchmark should compose the real Op/DPP stack.

## Review-fix mapping

| Review comment | Fix |
|---|---|
| "not FKL" / "breaks the FKL philosophy" | make it a DPP composing IOps + standard Ops |
| "should be a DPP, not an Operation/kernel" | wrap the kernel as a DPP taking Read/Write IOps |
| "this is cudaGraphs, no FKL Ops/DPPs" | rebuild on the FKL execution model; benchmark the DPP path |
| "faking the IOps, not using operator\|" | compose with the real `\|` / `.then()` fusion |
| "epilogue done by a single thread" | parallelise the epilogue with shared memory / a warp-cooperative DPP |
| "Max/Min/Sum should be standard Ops as args" | pass `Add`/`Max`/`Min` as template params |
| "duplicates code" | reuse an existing Op or a prior primitive |

## Verification

For any proposed change you should be able to state, in one line: (a) Op vs DPP
vs policy and why (the single-thread rule), and (b) which IOps carry its read,
compute, and write. Cross-check the type names against the live table in
`fkl-implementing-operations/SKILL.md`.

Background: "The Fused Kernel Library: A C++ API to Develop Highly-Efficient GPU
Libraries" (arXiv:2508.07071).
