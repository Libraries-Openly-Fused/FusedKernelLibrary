---
name: fkl-using-the-library
description: Write user code with the Fused Kernel Library (FKL) — compose fused GPU/CPU pipelines with executeOperations, manage Ptr2D/Tensor data, streams, and runtime parameters. Use when writing an application or library that calls FKL, when porting OpenCV-style image pipelines to fused kernels, or when you need the canonical pipeline patterns (DNN preprocessing, multi-ROI crop, color conversion).
---

# Using the Fused Kernel Library

## Mental model (60 seconds)

FKL fuses a SEQUENCE of operations into ONE kernel at C++ compile time.
You write the operations in execution order; intermediates live in
registers, never in DRAM:

```cpp
#include <fused_kernel/fused_kernel.h>
using namespace fk;

Stream stream;
Ptr2D<uchar3> input(width, height);          // allocates device memory
Ptr2D<float3> output(width, height);

executeOperations<TransformDPP<>>(stream,
    PerThreadRead<ND::_2D, uchar3>::build(input),
    Cast<uchar3, float3>::build(),           // explicit dtype changes
    Mul<float3>::build({2.f, 2.f, 2.f}),
    Add<float3>::build({10.f, 20.f, 30.f}),
    PerThreadWrite<ND::_2D, float3>::build(output));
stream.sync();
```

Rules:
1. First IOp must be a complete Read (`PerThreadRead`, `TensorRead`, `ReadSet`, ...). Last must be a Write (`PerThreadWrite`, `TensorWrite`, `TensorSplit`, ...).
2. Each op's OutputType must match the next op's InputType. Type errors are compile errors with the offending pair in the message.
3. TYPES define the kernel. VALUES (`build()` arguments) are runtime parameters: changing a factor or a crop rect does NOT create a new kernel.

## The two layers

- `Operation` structs (e.g. `Mul<float3>`): static `exec()` + type aliases. Pure compute, no state.
- `InstantiableOperation` (IOp) = Operation + its runtime params, created with `Op::build(args...)`. What you pass to `executeOperations`.

## Common pipeline patterns (all verified)

### DNN preprocessing in one kernel (crop -> resize -> normalize -> planar)

```cpp
executeOperations<TransformDPP<>>(stream,
    PerThreadRead<ND::_2D, uchar3>::build(frame),
    Crop<>::build(Rect(40, 30, 240, 180)),                         // ReadBack: fused into read
    Resize<InterpolationType::INTER_LINEAR>::build(Size(32, 32)),  // ReadBack, stacks on Crop
    Sub<float3>::build({123.675f, 116.28f, 103.53f}),
    Div<float3>::build({58.395f, 57.12f, 57.375f}),
    TensorSplit<float3>::build(chwTensor));                        // packed -> planar CHW
```

### Many ROIs from one image (Horizontal Fusion: pass an array)

```cpp
const std::array<Rect, 5> rois{ ... };             // array => batch => HF
Tensor<float3> out(64, 64, 5);                     // 5 planes
executeOperations<TransformDPP<>>(stream,
    PerThreadRead<ND::_2D, uchar3>::build(image),
    Crop<>::build(rois),                           // one kernel, 5 planes
    Resize<InterpolationType::INTER_LINEAR>::build(Size(64, 64)),
    Mul<float3>::build({1/255.f, 1/255.f, 1/255.f}),
    TensorWrite<float3>::build(out));
```

### Batch of separate same-size images

```cpp
const std::array<Ptr2D<float>, 4> inputs{ imgA, imgB, imgC, imgD };
executeOperations<TransformDPP<>>(stream,
    PerThreadRead<ND::_2D, float>::build(inputs),  // BatchRead under the hood
    Mul<float>::build(0.5f),
    TensorWrite<float>::build(out4planes));
```

### Color conversion + channel ops

```cpp
ColorConversion<ColorConversionCodes::COLOR_RGB2GRAY, uchar3, uchar>::build()
VectorReorder<uchar3, 2, 1, 0>::build()      // compile-time channel shuffle
VectorReorderRT<uchar3>::build({2, 1, 0})    // runtime shuffle (params)
Discard<uchar4, uchar3>::build()             // drop alpha
SaturateCast<float3, uchar3>::build()        // clamp + convert
```

## Streams

- `Stream stream;` creates an owning CUDA stream; `stream.sync()` waits.
- Wrap an external stream zero-cost: `Stream s(existingCudaStream);` (FKL will NOT destroy it). Use this to interop with torch/cupy streams.
- All `executeOperations` overloads are async on the given stream.

## Out-of-bounds reads

Wrap the read with a `BorderReader` policy BEFORE ops that may sample outside (Crop past the edge, warps). The backIOp is the complete read:

```cpp
BorderReader<BorderType::REPLICATE>::build(
    PerThreadRead<ND::_2D, float>::build(input))
// CONSTANT takes the fill value too:
BorderReader<BorderType::CONSTANT>::build(readIOp, 0.f)
```
Policies: CONSTANT, REPLICATE, REFLECT, WRAP, REFLECT_101.

## Temporal video windows

`CircularTensor<T, COLOR_PLANES, BATCH, CircularTensorOrder, ColorPlanes>` keeps the last BATCH frames on the GPU; `update(stream, readIOp, ops..., writeIOp)` preprocesses + inserts + rotates in one Divergent-HF kernel.

## CPU backend

The same pipelines run on CPU: `executeOperations<TransformDPP<ParArch::CPU>>` with `Stream_<ParArch::CPU>`. Useful for tests without a GPU.

## Pitfalls (each cost real debugging time)

1. Mismatched adjacent types: read the static_assert chain bottom-up; the first frame names the two ops that disagree.
2. `Ptr2D`/`Tensor` copies are SHALLOW (ref-counted). Use `.ptr()` to get the RawPtr for kernels; don't free the underlying memory manually.
3. Resize with aspect-ratio preservation needs a background value.
4. An array-built ReadBack (e.g. `Crop<>::build(std::array<Rect,N>)`) produces a BatchRead; it should be fused with the previous Read or Readback in the `executeOperations` function before passing the IOps to the kernel/DPP.
5. CUDA 13.x deprecation warnings about long4/double4 are noise; only grep compiler output for "error".