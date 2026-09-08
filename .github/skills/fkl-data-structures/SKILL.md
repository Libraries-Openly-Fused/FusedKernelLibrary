---
name: fkl-data-structures
description: FKL data structures — Ptr2D, Tensor, TensorT, RawPtr, PtrDims, MemType, constructors and memory layouts (packed, planar CHW, transposed T3D). Use when allocating or wrapping GPU memory for FKL pipelines, when interfacing external pointers (torch/cupy buffers), or when a Tensor/TensorT constructor or pitch issue appears.
---

# FKL data structures

## The hierarchy

- `RawPtr<ND, T>` — POD: data pointer + `PtrDims<ND>`. What kernels see.
- `Ptr<ND, T>` — ref-counted owner/wrapper around a RawPtr.
- Convenience classes: `Ptr1D`, `Ptr2D`, `Ptr3D`, `Tensor`, `TensorT`.
- `.ptr()` returns the RawPtr; `Op::build(container)` extracts what it needs. Copies of Ptr objects are SHALLOW (shared refcount).

## Dimensionalities (ND)

| ND | layout | use |
|---|---|---|
| `_1D` | w | flat arrays |
| `_2D` | w x h (pitched) | images |
| `_3D` | w x h x planes x color_planes | batched images / planar CHW |
| `T3D` | transposed: color_planes outermost | NCHW-like DNN ingest |

## Constructors that matter (and their traps)

```cpp
// allocating
Ptr2D<uchar3> img(width, height);                       // device by default
Tensor<float> t(width, height, planes, color_planes);   // 3D batch

// wrapping EXTERNAL memory (zero-copy interop):
Ptr2D<float> wrap(devPtr, width, height, pitchBytes, MemType::Device);
Tensor<float> wrapT(devPtr, width, height, planes, color_planes, MemType::Device);
```

TRAPS (verified the hard way):
1. `Tensor` has NO PtrDims-taking constructor — pass the dimension list.
2. `Tensor`'s semantics for batch+channels: `planes` = batch (thread.z), `color_planes` = channels. `TensorSplit` writes channel c of plane z at offset `z * plane_pitch * color_planes + c * plane_pitch`.
3. `TensorT(data, ...)` and the 4-arg `PtrDims<T3D>` constructor leave pitches at ZERO (they are filled on allocation). When wrapping an external pointer for T3D, build the PtrDims and set pitch, plane_pitch, color_planes_pitch manually, then construct the `RawPtr<T3D>` and pass it to `TensorTSplit<T>::build(rawPtr)`.
4. Pitch is in BYTES. For tightly-packed external buffers, pitch = width * sizeof(T).

## MemType

`Device`, `Host`, `HostPinned`, `DeviceAndPinned` (mirrored pair with `.upload(stream)` / `.download(stream)`). GPU pipelines require Device or DeviceAndPinned memory — CircularTensor enforces this at runtime.

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
Contiguity is the caller's responsibility (require C-contiguous or read strides into the pitch argument).