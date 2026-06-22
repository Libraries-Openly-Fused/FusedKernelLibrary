---
name: fkl-fusion-techniques
description: Choose and combine FKL's four fusion techniques — Vertical Fusion (VF), Backwards Vertical Fusion (BVF), Horizontal Fusion (HF) and Divergent Horizontal Fusion (DHF) — as described in the paper (arXiv:2508.07071). Use when deciding how to structure a pipeline for maximum fusion, when batching ROIs or images, or when different data planes need different processing in one kernel.
---

# FKL fusion techniques

Four techniques, all composable in the SAME kernel. The executor and
BackFuser apply them automatically from the way you express the pipeline.

## 1. Vertical Fusion (VF)

Sequential point operations collapse into one kernel; intermediates stay
in registers.

```cpp
executeOperations<TransformDPP<>>(stream,
    read, Mul<float>::build(2.f), Add<float>::build(1.f),
    Sub<float>::build(0.5f), write);
```

- Any number of compute ops between read and write.
- For VERY long chains (hundreds+ of identical steps) use
  `StaticLoop<Op, N>`: N fused repetitions, one parameter slot, avoids
  exploding the kernel parameter space.
- Measured effect (RTX PRO 6000, 1080p float, 6-op chain): fused ~19 us
  vs ~95 us as 6 separate kernels => ~5x. The win grows with chain length
  because every unfused boundary is a DRAM round-trip.

## 2. Backwards Vertical Fusion (BVF)

ReadBack operations (Crop, Resize, Warping, BorderReader, Deinterlace)
have no standalone input: they SAMPLE another read. The `BackFuser` folds
them backwards into the read at compile time — like OpenCV's filter
pipelines but with a generic, type-safe API.

```cpp
executeOperations<TransformDPP<>>(stream,
    PerThreadRead<ND::_2D, uchar3>::build(img),
    Crop<>::build(Rect(40, 30, 240, 180)),       // samples the read
    Resize<InterpolationType::INTER_LINEAR>::build(Size(32, 32)),   // samples the crop
    write);
```

- ReadBacks STACK: each one's backIOp is the previous stage. Crop->Resize
  means "resize the cropped region", with each output thread computing
  its source coordinates through the whole stack — no intermediate image.
- Output geometry comes from the LAST ReadBack (`num_elems_x/y/z`).
- Threads are launched for the OUTPUT size, not the input size.

## 3. Horizontal Fusion (HF)

Process a BATCH in one kernel: thread-plane z = batch index. Expressed by
passing ARRAYS of parameters instead of single parameters (this is the
rule of thumb: lists/arrays => HF).

Two flavours:

```cpp
// (a) batch of ROIs from ONE image
const std::array<Rect, 5> rois{...};
Crop<>::build(rois)                       // => BatchRead, 5 planes

// (b) batch of SEPARATE same-size images
const std::array<Ptr2D<float>, 4> imgs{...};
PerThreadRead<ND::_2D, float>::build(imgs)  // => BatchRead, 4 planes
```

- Batch size is a TEMPLATE parameter (std::array, not std::vector): each
  distinct N is a distinct kernel, compiled once.
- All planes run the SAME op sequence (for different sequences see DHF).
- Output is a `Tensor<T>` with N planes.
- The `activeBatch + defaultValue` overloads of `executeOperations` let a
  compiled batch size N process fewer than N real items.

## 4. Divergent Horizontal Fusion (DHF)

Different planes execute DIFFERENT fused sequences in one kernel,
selected per-plane by a SequenceSelector (z -> 1-based sequence index):

```cpp
struct MySelector {
    FK_HOST_DEVICE_FUSE uint at(const uint& z) { return z == 0 ? 1u : 2u; }
};

const auto seq1 = buildOperationSequence(readA, Mul<float>::build(4.f), writeT);
const auto seq2 = buildOperationSequence(readB, Add<float>::build(50.f), writeT);
Executor<DivergentBatchTransformDPP<ParArch::GPU_NVIDIA, MySelector>>::
    executeOperations(stream, seq1, seq2);
```

- Each sequence must be a complete read->...->write chain; all sequences
  share the output tensor (each plane writes its own slice).
- The executor's grid.z is the SUM of the sequences' z extents — each
  sequence should cover exactly its own planes. Do NOT give every
  sequence a full-batch read or you will launch (and write) extra planes.
- In-tree user: `CircularTensor::update` (seq1 = preprocess+insert the
  new frame, seq2 = rotate-copy the other planes) — the temporal-video
  pattern from the paper.
- Selector convention: `at(z)` returns 1-based sequence number
  (see `SequenceSelectorType` in circular_tensor.h).

## Combining all four

One kernel can be: batch crops (HF) of one image, each resized (BVF),
normalized (VF), where plane 0 additionally runs a different chain (DHF).
Fusion never changes results — only memory traffic. Compose the pipeline
that is correct, and express batches as arrays; the library does the rest.

## Choosing

| situation | technique | how to express |
|---|---|---|
| chain of point ops | VF | just list them in order |
| crop/resize/warp before compute | BVF | put ReadBacks right after the read |
| N ROIs / N images, same processing | HF | pass std::array of rects/ptrs |
| N planes, different processing | DHF | sequences + SequenceSelector |
| rolling window of last N frames | CircularTensor | `update()` per frame |
