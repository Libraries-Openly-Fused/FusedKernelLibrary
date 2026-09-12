---
name: fkl-using-the-library
description: Write user code with the Fused Kernel Library (FKL) — compose fused GPU/CPU pipelines with executeOperations, manage Ptr2D/Tensor data, streams, and runtime parameters. Use when writing an application or library that calls FKL, when porting OpenCV-style image pipelines to fused kernels, or when you need the canonical pipeline patterns (DNN preprocessing, multi-ROI crop, color conversion).
---

# Using the Fused Kernel Library

## Mental model (60 seconds)

For an ordinary read → compute → write pipeline, use `TransformDPP`.
The types determine the fused computation; `build()` supplies runtime values.
Fusion avoids materialized intermediate arrays, but does not guarantee that
the compiler never spills registers.

This helper expects an initialized input and an output of matching dimensions
in memory accessible to the default backend:

```cpp
#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/cast.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
using namespace fk;

void transformImage(Stream& stream, const Ptr2D<uchar3>& input,
                    const Ptr2D<float3>& output) {
    executeOperations<TransformDPP<>>(stream,
        PerThreadRead<ND::_2D, uchar3>::build(input),
        Cast<uchar3, float3>::build(),
        Mul<float3>::build({2.f, 2.f, 2.f}),
        Add<float3>::build({10.f, 20.f, 30.f}),
        PerThreadWrite<ND::_2D, float3>::build(output));
}
```

Rules:
1. First IOp must be a complete Read (`PerThreadRead`, `TensorRead`, `ReadSet`, ...). Last must be a Write (`PerThreadWrite`, `TensorWrite`, `TensorSplit`, ...).
2. Each op's OutputType must match the next op's InputType. Type errors are compile errors with the offending pair in the message.
3. TYPES define the kernel. VALUES (`build()` arguments) are runtime parameters: changing a factor or a crop rect does NOT create a new kernel.

## Know which layer to change

- `Operation` structs (e.g. `Mul<float3>`): static per-thread behavior and type
  aliases, including compute, reads, or writes.
- `InstantiableOperation` (IOp) = Operation + its runtime params, created with `Op::build(args...)`. What you pass to `executeOperations`.
- The DPP schedules IOp execution; the executor performs host-side setup and
  dispatch. If composition is insufficient, use
  [architecture overview](../fkl-architecture-overview/SKILL.md) before adding
  an Operation or DPP.

## Plan a pipeline before writing code

1. Record input/output dtype, dimensions, batch count, packed/planar layout,
   byte pitch, backend, and owner. Confirm allocation and stream lifetimes with
   [data structures](../fkl-data-structures/SKILL.md).
2. List the required transformations in semantic order. For each stage, record
   its input/output value type and any change to the logical output dimensions.
   Do not reorder a cast, normalization, or sampling stage just to fit a recipe.
3. Find existing builders and their tests; use
   [using operations](../fkl-using-operations/SKILL.md) for host composition.
   Default to `TransformDPP`, then choose batching or DHF only if needed.
4. Allocate the final output for the completed read's domain. Keep intermediates
   as IOp composition rather than allocating a buffer after each stage.
5. Initialize input, upload when needed, execute, download when needed,
   synchronize, and compare against independently expected values.

For the DNN recipe below, the value/geometry trace is:

| Stage | Output value | Logical output |
|---|---|---|
| Read | `uchar3` | Frame width × height |
| Crop | `uchar3` | 240 × 180 |
| Linear resize | `float3` | 32 × 32 |
| Sub / Div | `float3` | 32 × 32 |
| TensorSplit write | No returned value | One image, three 32 × 32 scalar planes |

If a stage has no existing implementation, stop and classify only that missing
responsibility with [architecture overview](../fkl-architecture-overview/SKILL.md).

## Common pipeline patterns

These are fragments using initialized input containers and an existing stream.
Include `crop.h` and `resize.h` from
`fused_kernel/algorithms/image_processing/` for the geometric recipes.

### DNN preprocessing in one kernel (crop -> resize -> normalize -> planar)

```cpp
Tensor<float> chwTensor(32, 32, 1, 3);
executeOperations<TransformDPP<>>(stream,
    PerThreadRead<ND::_2D, uchar3>::build(frame),
    Crop<>::build(Rect(40, 30, 240, 180)),                         // ReadBack: fused into read
    Resize<InterpolationType::INTER_LINEAR>::build(Size(32, 32)),  // ReadBack, stacks on Crop
    Sub<float3>::build({123.675f, 116.28f, 103.53f}),
    Div<float3>::build({58.395f, 57.12f, 57.375f}),
    TensorSplit<float3>::build(chwTensor));                        // packed -> planar CHW
```

Linear resize produces float3 here. `TensorSplit<float3>` writes three scalar
planes per image, hence `Tensor<float>` with `color_planes = 3`. The crop must
fit the input frame unless an appropriate border policy is supplied.

### Many ROIs from one image (Horizontal Fusion: pass an array)

```cpp
const std::array<Rect, 2> rois{Rect(0, 0, 32, 32), Rect(32, 0, 32, 32)};
Tensor<float3> out(64, 64, 2);
executeOperations<TransformDPP<>>(stream,
    PerThreadRead<ND::_2D, uchar3>::build(image),
    Crop<>::build(rois),                           // two batch planes
    Resize<InterpolationType::INTER_LINEAR>::build(Size(64, 64)),
    Mul<float3>::build({1/255.f, 1/255.f, 1/255.f}),
    TensorWrite<float3>::build(out));
```

Both ROIs must be valid for `image`. Each output plane is packed float3, unlike
the scalar planar output in the preceding recipe.

### Batch of separate same-size images

```cpp
const std::array<Ptr2D<float>, 4> inputs{ imgA, imgB, imgC, imgD };
Tensor<float> out4planes(imgA.dims().width, imgA.dims().height, 4);
executeOperations<TransformDPP<>>(stream,
    PerThreadRead<ND::_2D, float>::build(inputs),  // BatchRead under the hood
    Mul<float>::build(0.5f),
    TensorWrite<float>::build(out4planes));
```

### Color conversion + channel ops

```cpp
ColorConversion<ColorConversionCodes::COLOR_RGB2GRAY, uchar3, uchar>::build();
VectorReorder<uchar3, 2, 1, 0>::build();      // compile-time channel shuffle
VectorReorderRT<uchar3>::build({2, 1, 0});    // runtime shuffle (params)
Discard<uchar4, uchar3>::build();            // drop alpha
SaturateCast<float3, uchar3>::build();        // clamp + convert
```

These are separate builder expressions, not a type-compatible pipeline.
Include `color_conversion.h` and `saturate.h` from the image-processing
directory, and `vector_ops.h` from `basic_ops/`.

## Streams

- Under nvcc, `Stream stream;` creates an owning CUDA stream.
  `stream.sync()` waits for completion. Wrapping an external CUDA stream is
  non-owning; FKL does not destroy it.
- GPU launches are asynchronous. Keep buffers and external owners alive until
  completion, and synchronize before host access to downloaded results.
- CPU Transform executes synchronously; its stream's `sync()` is a no-op.
- Allocations default to mirrored `DeviceAndPinned` under nvcc and `Host` in
  CPU compilation. Initialize data and upload/download as needed; allocation
  alone does not populate device input. See
  [data structures](../fkl-data-structures/SKILL.md).

## Out-of-bounds reads

Wrap the read with a `BorderReader` policy BEFORE ops that may sample outside (Crop past the edge, warps). The backIOp is the complete read:

```cpp
BorderReader<BorderType::REPLICATE>::build(
    PerThreadRead<ND::_2D, float>::build(input));
// CONSTANT takes the fill value too:
BorderReader<BorderType::CONSTANT>::build(readIOp, 0.f);
```
Policies: CONSTANT, REPLICATE, REFLECT, WRAP, REFLECT_101.
Include `fused_kernel/algorithms/image_processing/border_reader.h`.

## Temporal video windows

For rolling video batches, inspect `CircularTensor` in
`include/fused_kernel/core/data/circular_tensor.h` and
`tests/operation/test_cricular_batch.h`. Its `update()` builds insertion and
rotation sequences internally; follow that API rather than assuming it takes
an arbitrary destination write. For lower-level DHF composition, see
[fusion techniques](../fkl-fusion-techniques/SKILL.md).

## CPU backend

Use `executeOperations<TransformDPP<ParArch::CPU>>` with
`Stream_<ParArch::CPU>` and host-accessible data. When selecting CPU explicitly
inside an nvcc compilation, request `MemType::Host` explicitly as well.

## Validation

1. Mismatched adjacent types: read the static_assert chain bottom-up; the first frame names the two ops that disagree.
2. Check output dimensions after the last ReadBack, dtype, planar versus packed
   layout, and batch plane count.
3. Test the complete pipeline against expected results, including borders and
   changed runtime parameters. Follow [build and test](../fkl-build-and-test/SKILL.md).