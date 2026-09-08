---
name: fkl-fusion-techniques
description: Choose and combine FKL's four fusion techniques — Vertical Fusion (VF), Backwards Vertical Fusion (BVF), Horizontal Fusion (HF) and Divergent Horizontal Fusion (DHF) — as described in the paper (arXiv:2508.07071). Use when deciding how to structure a pipeline for maximum fusion, when batching ROIs or images, or when different data planes need different processing in one kernel.
---

# FKL fusion techniques

Choose the technique from the pipeline's data flow. These techniques can combine
within a supported DPP/Executor contract; this does not imply arbitrary
DPP-to-DPP fusion. Use [using operations](../fkl-using-operations/SKILL.md) for
host syntax and [architecture overview](../fkl-architecture-overview/SKILL.md)
before implementing new behavior.

The snippets below are composition fragments with initialized inputs,
compatible outputs, and an existing stream. Include the relevant operation
headers explicitly.

## 1. Vertical Fusion (VF)

Sequential point operations execute in one kernel without materialized
intermediate arrays. Register pressure and compiler spills still matter.

```cpp
executeOperations<TransformDPP<>>(stream,
    read, Mul<float>::build(2.f), Add<float>::build(1.f),
    Sub<float>::build(0.5f), write);
```

- Compatible compute ops can be chained between read and write.
- For VERY long chains (hundreds+ of identical steps) use `StaticLoop<Op, N>`: N fused repetitions, one parameter slot, avoids exploding the kernel parameter space.
- Benchmark the actual workload: fewer launches and intermediate writes do not
  guarantee a fixed speedup.

## 2. Backwards Vertical Fusion (BVF)

ReadBack operations such as Crop, Resize, and BorderReader sample another IOp.
An incomplete builder carries parameters until `BackFuser` attaches its source.
This can eliminate intermediate images, but repeated samples may repeat upstream
computation.

```cpp
executeOperations<TransformDPP<>>(stream,
    PerThreadRead<ND::_2D, uchar3>::build(img),
    Crop<>::build(Rect(40, 30, 240, 180)),                        // samples the read
    Resize<InterpolationType::INTER_LINEAR>::build(Size(32, 32)), // samples the crop
    PerThreadWrite<ND::_2D, float3>::build(resizedOutput));
```

- ReadBacks STACK: each one's backIOp is the previous stage. Crop->Resize means "resize the cropped region", with each output thread computing its source coordinates through the whole stack — no intermediate image.
- Output geometry comes from the LAST ReadBack (`num_elems_x/y/z`).
- Threads are launched for the OUTPUT size, not the input size.
- Here `resizedOutput` is a 32 × 32 float3 image: linear interpolation produces
  floating channels. Add an explicit cast if a different output type is wanted.

## 3. Horizontal Fusion (HF)

Process a batch through the same operation types: logical z selects the batch
item (`blockIdx.z` on the Transform GPU path). Supported `std::array` builders
create batch IOps; not every array argument implies HF.

Two flavours:

```cpp
// (a) batch of ROIs from ONE image
const std::array<Rect, 2> rois{Rect(0, 0, 32, 32), Rect(32, 0, 32, 32)};
const auto crops = Crop<>::build(rois);    // completed when fused with a read

// (b) batch of SEPARATE same-size images
const std::array<Ptr2D<float>, 2> imgs{imageA, imageB};
const auto reads = PerThreadRead<ND::_2D, float>::build(imgs);
```

- Batch size is a TEMPLATE parameter (`std::array`, not `std::vector`): each distinct N is a distinct kernel, compiled once.
- All planes run the SAME op sequence (for different sequences see DHF).
- Use a compatible batched output, such as a `Tensor<T>` with N planes.
- The `activeBatch + defaultValue` overloads of `executeOperations` let a compiled batch size N process fewer than N real items.

## 4. Divergent Horizontal Fusion (DHF)

Different planes execute different fused sequences. `SequenceSelector::at(z)`
returns a **zero-based** sequence index:

```cpp
struct MySelector {
    FK_HOST_DEVICE_FUSE uint at(const uint& z) { return z == 0 ? 0u : 1u; }
};

const auto seq1 = buildOperationSequence(readA, Mul<float>::build(4.f), writeT);
const auto seq2 = buildOperationSequence(readB, Add<float>::build(50.f), writeT);
Executor<DivergentBatchTransformDPP<defaultParArch, MySelector>>::
    executeOperations(stream, seq1, seq2);
```

- Each sequence must be a complete read → compute → write chain. Outputs must
  be valid and non-conflicting for the selected planes; a shared tensor is one
  supported arrangement, not a requirement.
- The executor's grid.z is the SUM of the sequences' z extents — each sequence should cover exactly its own planes. Do NOT give every sequence a full-batch read or you will launch (and write) extra planes.
- In-tree user: `CircularTensor::update` (seq1 = preprocess+insert the new frame, seq2 = rotate-copy the other planes) — the temporal-video pattern from the paper.
- The selected sequence receives the original global z, not a local sequence
  index. Its addressing must handle that coordinate. Keep selector results in
  `[0, number_of_sequences)`; there is no automatic local-plane remapping.
- Core `DivergentBatchTransformDPP` and its executor support CPU and NVIDIA
  backends. The CPU path traverses selected planes without a GPU launch.
- Inspect `include/fused_kernel/core/execution_model/data_parallel_patterns.h`
  and `executors.h` in the same directory. Examples:
  `tests/data_parallel_patterns/test_divergent_batch.h` and
  `tests/examples/test_divergent_hf_executor.h` (CUDA-only).

## Combining all four

Start with a correct read → crop → resize → normalize → write pipeline, then
batch it through supported builders. Use DHF only when planes need different
sequences and their global-z addressing is explicit. Test combined geometry,
boundaries, layouts, and numerical tolerances; fusion can affect floating-point
evaluation and does not fix an invalid pipeline.

## Choosing

| situation | technique | how to express |
|---|---|---|
| chain of point ops | VF | just list them in order |
| crop/resize/warp before compute | BVF | put ReadBacks right after the read |
| N ROIs / N images, same processing | HF | pass std::array of rects/ptrs |
| N planes, different processing | DHF | sequences + SequenceSelector |
| rolling window of last N frames | CircularTensor | `update()` per frame |