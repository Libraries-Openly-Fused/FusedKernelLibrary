---
name: fkl-language-bindings
description: Design a language binding for FKL using generated host-side IOp composition and nvcc compilation. Use for graph-to-C++ generation, runtime parameter ABI, zero-copy ownership, and cache design. Read the consumer skills first; this is binding design guidance, not a shipped stable FFI or support for clang-as-CUDA.
---

# Building language bindings for FKL

This is a design recipe, not a public FKL FFI contract. Dynamic front-ends can
compose an AST of operations and generate a fused C++ translation unit.
First express the same pipeline in C++ using
[using operations](../fkl-using-operations/SKILL.md) and
[data structures](../fkl-data-structures/SKILL.md). Keep Operation nodes
(per-thread semantics) distinct from the selected DPP/launcher.

## Why the obvious approaches fail

1. **Fixed pybind11/SWIG API**: suitable for a finite set of pre-instantiated
   pipelines, but does not expose arbitrary new C++ type compositions by itself.
2. **NVRTC alone**: does not compile the normal host `build()`/BackFuser/Executor
   path. Use nvcc for generated host-plus-device translation units; an alternative
   device-only architecture would need separate design and validation.
3. **Precompiled kernels**: useful for known pipelines, but cannot enumerate all
   possible user-defined chains.

## The AST/Graph Architecture

Instead of binding C++ functions, the host language represents Operations and DPPs as graph nodes. Each node stores the literal C++ type string (e.g., `"fk::Add<float>"`) and the host language allocates its parameter struct.

The host language (Python, Rust, etc.) is responsible for **writing the C++ source strings** for steps 1, 2, 3, and 4 below, concatenating them to generate the final `.cu` file:

```cpp
#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/crop.h>

// user composes lazily (Python graph nodes)
//      |
//      v   first call with concrete dtype/shape
// generate ONE .cu translation unit:

extern "C" void fkl_entry(const void** params_ptrs, void* stream)
{ 
    // THE HOST LANGUAGE GENERATES THE C++ SOURCE STRINGS FOR STEPS 1 THROUGH 4:

    // 1. Caller must supply valid, aligned objects of exactly these C++ ABI types.
    auto in_ptr  = *reinterpret_cast<const fk::RawPtr<fk::ND::_2D, float>*>(params_ptrs[0]);
    auto p1      = *reinterpret_cast<const float*>(params_ptrs[1]);
    auto p2      = *reinterpret_cast<const fk::Rect*>(params_ptrs[2]);
    auto out_ptr = *reinterpret_cast<const fk::RawPtr<fk::ND::_2D, float>*>(params_ptrs[3]);
    
    // 2. Instantiate Read IOp via ::build()
    auto in_op  = fk::PerThreadRead<fk::ND::_2D, float>::build(in_ptr);
    
    // 3. Instantiate Compute and ReadBack IOps via ::build()
    auto op1    = fk::Add<float>::build(p1);
    auto op2    = fk::Crop<>::build(p2);
    
    // 4. Instantiate Write IOp and Execute the static DPP
    auto out_op = fk::PerThreadWrite<fk::ND::_2D, float>::build(out_ptr);
    fk::Stream s(reinterpret_cast<cudaStream_t>(stream)); // non-owning wrapper
    fk::executeOperations<fk::TransformDPP<>>(s, in_op, op1, op2, out_op);
}

//      |
//      v
// single-step host+device compile to shared library (.so on Linux, .dll on Windows)
//      |
//      v
// disk-cache keyed by TYPE SIGNATURE; load + one FFI call per launch
```

Key decisions:

1. **Types compile, values don't.** Key the cache by generated source/type
   signature, specialized capacities, target GPU architecture, compiler/options,
   ABI, and FKL revision. Runtime values are passed on every call. Invalidate the
   cache when code or toolchain changes, not when an ordinary parameter changes.
2. **Generate what a C++ user would write.** The binding emits the exact `::build(params)` calls and static `executeOperations` sequences a human would write.
3. **Use the supported toolchain.** Compile CUDA with nvcc and a supported host
   compiler. Stock FKL does not support clang as the CUDA compiler; do not copy
   downstream shims as if they were part of this repository's supported build.

## The ABI (Pointer Array Architecture)

The example is the launch core, not a production-safe exported ABI. Add argument
validation and an error-status boundary before exposing it to another language;
C++ exceptions from CUDA error checks must not cross the C ABI.

For a pointer-array ABI:
- `const void** params_ptrs`: **All** runtime values, inputs, and outputs passed as an array of pointers. 
  - Each host object must match the compiled C++ type's size, alignment, field
    offsets, and lifetime. Prefer typed C++ construction/shims where native FFI
    layouts cannot establish those guarantees.
  - It then passes an array of pointers to these structs (`const void**`).
  - A `reinterpret_cast` does not create an object, fix alignment, or prove ABI
    compatibility. Validate pointer count/nullability before dereferencing.
- `void* stream`: external CUDA stream, non-owning. In the example, null wraps
  CUDA's default stream; if choosing an owned fallback, define that policy
  explicitly. The caller owns asynchronous ordering and buffer lifetime.

## Zero-copy interop

- Inputs: accept anything exposing a device pointer + shape + dtype (`__cuda_array_interface__` in Python). Allocate a `RawPtr` struct on the host containing its pointer and strides, and pass its address in the `params_ptrs` array. Require C-contiguous or honor strides via the pitch argument.
- Outputs: allocate with the CUDA driver API and expose BOTH the array interface and DLPack so frameworks adopt the memory without copying. On DLPack export, transfer ownership to the consumer's deleter and disarm your own destructor (double-free guard).

## Mapping the fusion techniques

| technique | binding surface |
|---|---|
| VF | ordered list of op descriptors between read and write |
| BVF | geometry descriptors right after the read; emit ReadBack builds |
| HF | LISTS of parameters: list of rects -> `std::array<Rect,N>`; N is part of the type |
| DHF | list of chains + plane->sequence map; generate a SequenceSelector struct |
| CircularTensor | stateful handle (create/update/snapshot/destroy in the same library) |

## Compilation and measurement

Generate from allowlisted Operation/type descriptors, not arbitrary untrusted C++
or shell fragments. Pass compiler arguments without shell interpolation. Measure
cold compilation, warm cache lookup, launch overhead, and GPU execution separately;
do not treat hardware-specific timings as a binding performance guarantee.

## Testing a binding

- Validate against CPU references computed in the host language, not against FKL itself.
- Test nvcc with supported host compilers; invalidate the cache between toolchains.
- Test cross-platform generation (`.dll` on Windows, `.so` on Linux).
- Test that parameter value changes hit the cache (same library path) and that chain topology changes miss it.
- Test ABI layout checks, invalid arguments, exception translation, external
  stream ordering, and buffer ownership through asynchronous completion.