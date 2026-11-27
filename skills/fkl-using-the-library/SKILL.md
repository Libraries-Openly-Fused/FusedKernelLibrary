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
1. First IOp must be a complete Read (`PerThreadRead`, `TensorRead`,
   `ReadSet`, ...). Last must be a Write (`PerThreadWrite`, `TensorWrite`,
   `TensorSplit`, ...).
2. Each op's OutputType must match the next op's InputType. Type errors
   are compile errors with the offending pair in the message.
3. TYPES define the kernel. VALUES (`build()` arguments) are runtime
   parameters: changing a factor or a crop rect does NOT create a new
   kernel.

## The two layers

- `Operation` structs (e.g. `Mul<float3>`): static `exec()` + type
  aliases. Pure compute, no state.
- `InstantiableOperation` (IOp) = Operation + its runtime params, created
  with `Op::build(args...)`. What you pass to `executeOperations`.

## Common pipeline patterns (all verified)

### DNN preprocessing in one kernel (crop -> resize -> normalize -> planar)

```cpp
executeOperations<TransformDPP<>>(stream,
    PerThreadRead<ND::_2D, uchar3>::build(frame),
    Crop<>::build(Rect(40, 30, 240, 180)),         // ReadBack: fused into read
    Resize<InterpolationType::INTER_LINEAR>::build(Size(32, 32)),     // ReadBack, stacks on Crop
    Sub<float3>::build({123.675f, 116.28f, 103.53f}),
    Div<float3>::build({58.395f, 57.12f, 57.375f}),
    TensorSplit<float3>::build(chwTensor));        // packed -> planar CHW
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

### Reductions of the same data in one kernel (ReduceDPP)

```cpp
#include <fused_kernel/core/execution_model/reduce_dpp.h>   // not included by fused_kernel.h
#include <fused_kernel/core/data/ptr_utils.h>               // setTo
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>   // Add, Div
#include <fused_kernel/algorithms/basic_ops/cast.h>         // Cast
#include <fused_kernel/algorithms/basic_ops/logical.h>      // Max, Min

// (fk:: on the vector types: under HIP, uchar3 and float3 are also names of hip_runtime.h)
Ptr1D<fk::float3> results(3);                              // one element per reduction
const float pixels = static_cast<float>(image.dims().width * image.dims().height);
const auto read = PerThreadRead<ND::_2D, fk::uchar3>::build(image);   // any complete read
const auto reductions = make_tuple(                        // Tuple<FirstIOp, AccumulationIOp, FinalIOp>...
    make_tuple(Cast<fk::uchar3, fk::float3>::build(),
               Add<fk::float3, fk::float3, fk::float3, UnaryType>::build(),
               Div<fk::float3>::build(make_set<fk::float3>(pixels))),         // mean
    make_tuple(Cast<fk::uchar3, fk::uchar3>::build(),
               Max<fk::uchar3, fk::uchar3, fk::uchar3, UnaryType>::build(),
               Cast<fk::uchar3, fk::float3>::build()),                         // maximum
    make_tuple(Cast<fk::uchar3, fk::uchar3>::build(),
               Min<fk::uchar3, fk::uchar3, fk::uchar3, UnaryType>::build(),
               Cast<fk::uchar3, fk::float3>::build()));                        // minimum
const auto write = PerThreadWrite<ND::_1D, fk::float3>::build(results);   // reduction i at Point{i, 0, z}

const size_t bytes = ReduceDPP<>::workspaceBytes(read, reductions);   // once; 0: no workspace needed
Ptr1D<uchar> workspace(static_cast<uint>(bytes));
if (bytes > 0) setTo(uchar{0}, workspace, stream);                    // it must start at zero
executeOperations<ReduceDPP<>>(stream, ReduceDPP<>::build_details(workspace), read, reductions, write);
```

- The image is read ONCE for the N reductions. FirstIOp runs on every
  element and gives exactly the accumulator type (the OutputType of the
  AccumulationIOp: put the `Cast` in FirstIOp), AccumulationIOp is a
  compute IOp on `Tuple<A, A>` (associative and commutative, no neutral
  element needed), FinalIOp runs once on the accumulated value.
  `Cast<T, T>` does nothing (T a scalar or a vector).
- Accumulators: scalars, vectors, or any small trivially copyable type
  that is default constructible, copy constructible and copy assignable
  (a `Tuple<float, uint>`, a struct), with user IOps that operate on
  it. Keep them small: on a GPU, accumulators with arrays of KBs
  (histograms) reserve GBs of device memory for thread stacks.
- Every plane of the read (batch, 3D) is reduced on its own, at its own
  size: B images and N reductions write B planes of N values
  (`TensorWrite` on `Tensor<T>(N, 1, B)`).
- A write too small for the reductions, without rows, or with other
  planes than the read, and an empty read, throw `std::invalid_argument`.
- The workspace is device memory that the caller owns: allocate and
  zero it ONCE and reuse it. One per stream is enough (executions that
  share it must not run at the same time). A GPU read with planes of
  more than one block (4096 elements by default) throws
  `std::invalid_argument` without it, saying the bytes it needs (on
  NVIDIA GPUs with compute capability 9.0 or newer, in a build for
  such an architecture, planes of up to 32768 elements still run
  without it, in one thread block cluster).
  `workspaceBytes` is 0 for smaller reads and on the CPU backends, and
  then `executeOperations<ReduceDPP<>>(stream, read, reductions, write)`
  is enough.
- An execution is one asynchronous kernel launch that allocates
  nothing: it can be captured in a CUDA graph (keep the workspace alive
  as long as the graph). A GPU read has 65535 planes at most.
- Many planes: the blocks are shared among the planes of the read, as
  if all of them had the size of the largest one, so a batch is one
  launch sized for all of it. Reduce planes of very different sizes in
  one execution per size: among many small planes a large one gets few
  blocks (an image of 2048 x 1024 among 127 of 10 x 10 took 5.8 times
  longer than as two executions on an RTX 4090, 2.4 on an RTX 5050).
  For expensive reads on a known GPU, tune `ReduceDPPPolicy::BLOCKS`
  (second template parameter; about 2 blocks per multiprocessor
  measured within 2% of the best).
- The output of a plane may be the memory that the same plane reads. It
  must not overlap what another plane reads.
- Sums of floats depend on the order of the additions, which is not the
  same on every backend, nor, on a GPU, for an image alone and for the
  same image in a batch: accumulate in an integer type when the result
  has to be identical everywhere, and in `double` (FirstIOp
  `Cast<float, double>`) when it only has to be more accurate.
- Same call on every backend: `ReduceDPP<ParArch::CPU>` with
  `Stream_<ParArch::CPU>`, `ReduceDPP<ParArch::CPU_MULTITHREAD>`...

## Streams

- `Stream stream;` creates an owning CUDA stream; `stream.sync()` waits.
- Wrap an external stream zero-cost: `Stream s(existingCudaStream);`
  (FKL will NOT destroy it). Use this to interop with torch/cupy streams.
- All `executeOperations` overloads are async on the given stream.

## Out-of-bounds reads

Wrap the read with a `BorderReader` policy BEFORE ops that may sample
outside (Crop past the edge, warps). The backIOp is the complete read:

```cpp
BorderReader<BorderType::REPLICATE>::build(
    PerThreadRead<ND::_2D, float>::build(input))
// CONSTANT takes the fill value too:
BorderReader<BorderType::CONSTANT>::build(readIOp, 0.f)
```
Policies: CONSTANT, REPLICATE, REFLECT, WRAP, REFLECT_101.

## Temporal video windows

`CircularTensor<T, COLOR_PLANES, BATCH, CircularTensorOrder, ColorPlanes>`
keeps the last BATCH frames on the GPU; `update(stream, readIOp, ops...,
writeIOp)` preprocesses + inserts + rotates in one Divergent-HF kernel.

## CPU backend

The same pipelines run on CPU: `executeOperations<TransformDPP<ParArch::CPU>>`
with `Stream_<ParArch::CPU>`. Useful for tests without a GPU.

## Pitfalls (each cost real debugging time)

1. Mismatched adjacent types: read the static_assert chain bottom-up; the
   first frame names the two ops that disagree.
2. `Ptr2D`/`Tensor` copies are SHALLOW (ref-counted). Use `.ptr()` to get
   the RawPtr for kernels; don't free the underlying memory manually.
3. Resize with aspect-ratio preservation needs a background value.
4. An array-built ReadBack (e.g. `Crop<>::build(std::array<Rect,N>)`)
   produces a BatchRead; fuse it with the read explicitly via
   `fuse(readIOp, cropIOp)` if you hit `no operator|` errors at
   data_parallel_patterns.h.
5. CUDA 13.x deprecation warnings about long4/double4 are noise; only
   grep compiler output for "error".
