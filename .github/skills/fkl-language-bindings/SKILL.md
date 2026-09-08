---
name: fkl-language-bindings
description: Design language bindings for FKL using generated host-side IOp composition and nvcc compilation. Use for graph-to-C++ generation, runtime parameter ABI, memory ownership, and cache design. This is design guidance, not a shipped stable FFI.
---

# Building language bindings for FKL

First read [using operations](../fkl-using-operations/SKILL.md) and
[data structures](../fkl-data-structures/SKILL.md). A binding should emit the same
host-side IOp composition as a C++ consumer. The design below is a proposal for
a binding, not an ABI or JIT service provided by this repository.

## Choose a compilation strategy

1. A fixed binding can expose a finite set of pre-instantiated pipelines. It
   cannot instantiate arbitrary new C++ type sequences at runtime on its own.
2. For open-ended composition, generate host-side `build()` and executor calls,
   then compile them with nvcc and a supported host toolchain.
3. NVRTC alone is not a replacement for that host compilation path. The
   `BackFuser`, builders, and launch setup require host-side C++ machinery.
4. Cache generated modules to avoid recompiling a type signature on every call.

## The AST/Graph Architecture

Represent supported Operations and DPPs as validated graph nodes. Map known
descriptors to C++ types; do not accept arbitrary untrusted source or compiler
flags as operation names.

The generated `.cu` translation unit should:

1. Include the required operation headers and `fused_kernel.h`.
2. Validate external metadata and construct correctly typed runtime params and
   `RawPtr` descriptors in C++.
3. Build read, ReadBack, compute, and write IOps in the graph's semantic order.
4. Invoke the selected DPP through its supported executor, using a non-owning
   wrapper for the caller's stream when supplied.
5. Expose a binding-owned entry point with explicit error reporting. Catch C++
   exceptions before returning across a C ABI boundary.

Use the concrete `TransformDPP<>` example in the consumer skill as the initial
generated pipeline; do not emit placeholder template arguments.

Key decisions:

1. **Types compile, values are parameters.** Key the cache on the graph/type
   signature, generated source, FKL revision, compiler/toolkit and flags, target
   architectures, platform, and binding ABI version. Parameter-only changes
   should reuse the module; topology or type changes may require recompilation.
2. **Generate what a C++ user would write.** The binding emits the exact `::build(params)` calls and static `executeOperations` sequences a human would write.
3. **Use the supported toolchain.** Compile CUDA with nvcc, including its host
   compilation step. Follow [build and test](../fkl-build-and-test/SKILL.md);
   do not introduce clang-as-CUDA shims as the binding's default path.

## The ABI (Pointer Array Architecture)

A pointer array is one possible binding ABI, not an alignment or aliasing
guarantee. Casting `void*` does not create an object or fix its layout.

- Prefer C-compatible descriptors with explicit field widths, constructing FKL
  objects in C++ rather than reproducing their ABI in another language.
- If accepting pointers to typed objects, require matching size, alignment,
  layout, lifetime, and host addressability. Validate counts and null pointers
  before reading entries; keep pointed-to device allocations alive until completion.
- Validate dimensions, pitches, integer ranges, device identity, and output
  capacity before launch. A valid metadata pointer does not prove a device
  buffer is large enough.
- Define null-stream semantics explicitly. Wrapping a null `cudaStream_t`
  selects CUDA's default stream; default-constructing `fk::Stream` creates an
  owned stream. Those are different policies.
- Specify who owns synchronization and how asynchronous errors are reported.

## Zero-copy interop

- Retain the external allocation's owner until all asynchronous work completes.
  Require a supported contiguous layout or represent its strides faithfully;
  a row pitch cannot encode arbitrary strided views.
- Allocate outputs in the consuming framework when practical. If exporting
  DLPack, use a managed ownership/deleter contract and prevent double frees.
- Keep producer and consumer stream ordering explicit; zero-copy does not
  imply synchronization.

## Mapping the fusion techniques

| technique | binding surface |
|---|---|
| VF | ordered list of op descriptors between read and write |
| BVF | geometry descriptors right after the read; emit ReadBack builds |
| HF | LISTS of parameters: list of rects -> `std::array<Rect,N>`; N is part of the type |
| DHF | list of chains + plane->sequence map; generate a SequenceSelector struct |
| CircularTensor | stateful handle (create/update/snapshot/destroy in the same library) |

## Measure rather than promise

Measure cold compilation, cache hits, FFI overhead, and kernel execution
separately on the target hardware. Neither compile time nor fusion speedup is
a fixed property of this interface.

## Testing a binding

- Validate against CPU references computed in the host language, not against FKL itself.
- Test nvcc with the supported host toolchains, including cache invalidation.
- Test cross-platform generation (`.dll` on Windows, `.so` on Linux).
- Test that parameter value changes hit the cache (same library path) and that chain topology changes miss it.
- Test malformed metadata, unsupported layouts, stream ordering, and ownership
  across asynchronous calls. Reject invalid inputs before kernel launch.