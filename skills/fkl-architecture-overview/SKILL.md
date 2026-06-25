---
name: fkl-architecture-overview
description: "Classify FKL work as Op, DPP, policy, or fused IOp."
version: 0.1.0
metadata:
  hermes:
    tags: [GPU, CUDA, FusedKernelLibrary, Architecture]
---

# FKL Architecture Overview

A fast classification guide for the Fused Kernel Library: given an algorithm,
decide whether it is an **Operation** or a **Data Parallel Pattern (DPP)**, or if
it should be split into a DPP and several Operations and route its data correctly.
It does NOT replace `fkl-implementing-operations` or
`fkl-implementing-data-parallel-patterns` (those are
authoritative and go deeper) — it is the 60-second "what am I building?" filter
to apply before reaching for them, plus the most common review-fix mappings.

## When to Use

- Before writing a new algorithm, to decide Op vs DPP.
- When a review says "this should be a DPP, not an Operation or standard Kernel",
  "this is not FKL / I can't fuse this", or "use standard Ops".
- When deciding how a kernel's read, compute, and write should flow through IOps.

## The single classifying rule

> An Operation is strictly ALWAYS single-thread code. If more than one thread
> must participate in the code, that code belongs in a DPP.

- **Op** — one thread's work; stateless struct, static `exec()`. Single-thread.
- **DPP** — anything needing >1 thread to cooperate: shared memory,
  `__syncthreads`, `__shfl`, warp-collective `mma.sync`, cp.async tile staging.
  A DPP routes data through IOps and never modifies the data directly, it always
  does it using IOps. 
- **IOp** (Instantiable Operation) — an Op plus its params, produced by
  `::build()`, invoked as `IOp::Operation::exec(...)`, never the raw Op struct.

## Operation types

The full IOp-form table (with exact `exec()` signatures) lives in
`fkl-implementing-operations`. The types:
`ReadType, WriteType, UnaryType, BinaryType, ReadBackType,
IncompleteReadBackType, TernaryType, IncompleteTernaryType, MidWriteType,
OpenType, ClosedType`.

## Procedure

1. Does correctness require more than one thread to cooperate? Yes → **DPP**;
   no → **Op**.
2. A DPP takes its read as a **Read IOp** (prologue), its write as a **Write
   IOp** (epilogue), and intermediate compute as IOps. Ops will always be
   passed as IOp instances to the DPPs. Never as template parameters.
3. DPP shape: `exec(details, readIOp, computeIOp, writeIOp)` — IOps are separate
   exec parameters, not packed into a struct; a `*Details` struct carries only
   external scalars (dims, identity, mask). Provide both `ParArch::GPU_NVIDIA`
   and `ParArch::CPU` specialisations; keep `*Details` and the CPU spec outside
   any `#if defined(__NVCC__)` guard.
4. For an output written through a fused epilogue, fuse the epilogue chain with
   the destination write (`epilogue.then(D)`) and invoke the whole fused IOp:
   `Output::Operation::exec(thread, value, output)`. The first link of the
   `.then()` chain must be a compute op (`Cast<T,T>` for identity), not a Read.

## Pitfalls

- `mma.sync` is warp-collective → it must be used inside a DPP, never an `Op`.
- `__nv_bfloat16` has no `VectorTraits`, so `PerThreadRead`/`PerThreadWrite`
  will not instantiate for it — supply a small Read/Write Op over a `RawPtr` in
  tests that use bf16.

## Review-fix mapping

| Review comment | Fix |
|---|---|
| "not FKL" / "breaks the FKL philosophy" | make it a DPP composing IOps + standard Ops |
| "should be a DPP, not an Operation/kernel" | wrap the kernel as a DPP taking Read/Write IOps |
| "this is cudaGraphs, no FKL Ops/DPPs" | rebuild on the FKL execution model; benchmark the DPP path |
| "faking the IOps, not using operator\|" | compose with the real `\|` / `.then()` fusion |
| "epilogue done by a single thread" | parallelise the epilogue with shared memory / a warp-cooperative DPP |
| "Max/Min/Sum should be standard Ops as args" | pass `Add`/`Max`/`Min` as IOps in exec function parameters |
| "duplicates code" | reuse an existing Op or a prior primitive |

## Verification

For any proposed change you should be able to state, in one line: (a) Op vs DPP
and why (the single-thread rule), and (b) which IOps carry its read,
compute, and write. Cross-check the type names against the live table in
`fkl-implementing-operations/SKILL.md`.

Background: "The Fused Kernel Library: A C++ API to Develop Highly-Efficient GPU
Libraries" (arXiv:2508.07071).
