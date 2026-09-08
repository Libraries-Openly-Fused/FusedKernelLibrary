---
name: fkl-data-structures
description: FKL data structures — Ptr2D, Tensor, TensorT, RawPtr, PtrDims, MemType, constructors and memory layouts (packed, planar CHW, transposed T3D). Use when allocating or wrapping GPU memory for FKL pipelines, when interfacing external pointers (torch/cupy buffers), or when a Tensor/TensorT constructor or pitch issue appears.
---

# FKL data structures

Use this skill for allocation, layout, and lifetime questions. For constructing
pipelines, continue with [using operations](../fkl-using-operations/SKILL.md).
The implementations are `include/fused_kernel/core/data/ptr_nd.h`, `rawptr.h`,
and `include/fused_kernel/algorithms/basic_ops/memory_operations.h`.

## The hierarchy

- `RawPtr<ND, T>` — POD: data pointer + `PtrDims<ND>`. What kernels see.
- `Ptr<ND, T>` — ref-counted owner/wrapper around a RawPtr.
- Convenience classes: `Ptr1D`, `Ptr2D`, `Ptr3D`, `Tensor`, `TensorT`.
- `.ptr()` returns the RawPtr used by read/write builders. Some builders also
  accept the container directly; check the overload. Copies of Ptr objects are
  shallow and share the underlying allocation.

## Dimensionalities (ND)

| ND | layout | use |
|---|---|---|
| `_1D` | w | flat arrays |
| `_2D` | w x h (pitched) | images |
| `_3D` | w x h x planes x color_planes | batched images / planar CHW |
| `T3D` | color_planes outermost, then batch planes | channel-major batched data |

## Constructors that matter (and their traps)

```cpp
// allocating
Ptr2D<uchar3> img(width, height);                       // backend-dependent default
Tensor<float> t(width, height, planes, color_planes);   // 3D batch

// wrapping EXTERNAL memory (zero-copy interop):
Ptr2D<float> wrap(devPtr, width, height, pitchBytes, MemType::Device);
Tensor<float> wrapT(devPtr, width, height, planes, color_planes, MemType::Device);
```

Check these details before wrapping an external buffer:
1. `Tensor` has NO PtrDims-taking constructor — pass the dimension list.
2. `Tensor`'s semantics for batch+channels: `planes` = batch (thread.z), `color_planes` = channels. `TensorSplit` writes channel c of plane z at offset `z * plane_pitch * color_planes + c * plane_pitch`.
3. `TensorT(data, ...)` and the four-dimension `PtrDims<ND::T3D>` constructor
   leave pitches at zero. For external T3D data, set `pitch`, `plane_pitch`, and
   `color_planes_pitch` in its dimensions explicitly, construct a
   `RawPtr<ND::T3D, T>`, and pass it to the appropriate builder.
4. Pitch is in BYTES. For tightly-packed external buffers, pitch = width * sizeof(T).

## MemType

`Device`, `Host`, `HostPinned`, and `DeviceAndPinned` are the memory kinds.
The default under nvcc is `DeviceAndPinned` (device buffer plus pinned host
mirror); in a CPU compilation it is `Host`.

For mirrored storage, initialize the host side and call `.upload(stream)`
before GPU reads. Call `.download(stream)` and then synchronize before consuming
GPU results on the host. Allocation alone does not initialize input values.
For an explicitly selected CPU backend under nvcc, allocate host memory
explicitly rather than relying on the CUDA compilation default.

Wrapping external pointers is non-owning. Keep the original owner alive through
all asynchronous work, and explicitly pass `MemType::Device` for device-only
wrappers; the default mirrored type requires a separate pinned pointer.

## Layout cheat-sheet for DNN interop

| want | use | output shape |
|---|---|---|
| packed HWC batch | `TensorWrite<T>` | (batch, H, W, C-packed-in-T) |
| planar CHW per image | `TensorSplit<T>` | (batch, C, H, W) |
| planar, C outermost | `TensorTSplit<T>` into TensorT | (C, batch, H, W) |
| read planar back as packed | `TensorPack<T>` / `TensorTPack<T>` | — |

## Vector pixel types

Channels are encoded in the TYPE: `uchar3`, `float4`, etc.
- `VBase<T>` = scalar base, `cn<T>` = channels, `VectorType_t<base, n>`.
- Per-channel arithmetic operators are predefined (vector_utils.h).
- 16-bit types (`ushort3`, `short2`) work like the 8/32-bit ones.

## External-framework interop (what bindings do)

A torch/cupy CUDA tensor is wrapped without copying:
```cpp
Ptr2D<float> in((float*)cuda_ptr, w, h, w * sizeof(float), MemType::Device);
Stream s(reinterpret_cast<cudaStream_t>(framework_stream));  // non-owning
```
These are CUDA-only wrapping expressions. Validate dtype, dimensions, byte pitch,
device identity, and allocation capacity. A single row pitch cannot represent
every framework view: reject unsupported strides rather than silently treating
them as contiguous. Keep the allocation and external stream valid until work
completes; FKL does not transfer their ownership.