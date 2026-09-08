---
name: fkl-fusion-techniques
description: Choose and combine FKL's four fusion techniques — Vertical Fusion (VF), Backwards Vertical Fusion (BVF), Horizontal Fusion (HF) and Divergent Horizontal Fusion (DHF) — as described in the paper (arXiv:2508.07071). Use when deciding how to structure a pipeline for maximum fusion, when batching ROIs or images, or when different data planes need different processing in one kernel.
---

# FKL fusion techniques

Use [using operations](../fkl-using-operations/SKILL.md) for the host API and
[architecture overview](../fkl-architecture-overview/SKILL.md) before adding new
primitives. Fusion is composition of IOps under a compatible DPP, not permission
to merge arbitrary cooperative kernels.

The four techniques can be combined in a compatible kernel. Transform's executor
and BackFuser infer VF/BVF/HF from the IOps; DHF requires explicit sequences and
a selector. Fragments below assume matching input/output types and allocations.

## 1. Vertical Fusion (VF)

Sequential point operations collapse into one kernel; intermediates stay in registers.

```cpp
executeOperations<TransformDPP<>>(stream,
    read, Mul<float>::build(2.f), Add<float>::build(1.f),
    Sub<float>::build(0.5f), write);
```

- Any number of compute ops between read and write.
- For VERY long chains (hundreds+ of identical steps) use `StaticLoop<Op, N>`: N fused repetitions, one parameter slot, avoids exploding the kernel parameter space.
- Measure instead of promising a fixed speedup: saved launches/temporary memory
  compete with recomputation, register pressure, and occupancy. Existing examples
  live in `benchmarks/fusion/` (repository-root-relative).

## 2. Backwards Vertical Fusion (BVF)

ReadBack operations (Crop, Resize, Warping, BorderReader, Deinterlace) have no standalone input: they SAMPLE another read. The `BackFuser` folds them backwards into the read at compile time — like OpenCV's filter pipelines but with a generic, type-safe API.

```cpp
executeOperations<TransformDPP<>>(stream,
    PerThreadRead<ND::_2D, uchar3>::build(img),
    Crop<>::build(Rect(40, 30, 240, 180)),                        // samples the read
    Resize<InterpolationType::INTER_LINEAR>::build(Size(32, 32)), // samples the crop
    write);
```

- ReadBacks STACK: each one's backIOp is the previous stage. Crop->Resize means "resize the cropped region", with each output thread computing its source coordinates through the whole stack — no intermediate image.
- Output geometry comes from the LAST ReadBack (`num_elems_x/y/z`).
- Threads are launched for the OUTPUT size, not the input size.
- Compute before a later ReadBack can also be folded into the sampled read.
  Its work may be repeated per sample; fusion is not a cached intermediate image.

## 3. Horizontal Fusion (HF)

Process a BATCH in one kernel: thread-plane z = batch index. Use supported
`std::array` batch builders (for example Crop or PerThreadRead); an arbitrary
array-valued compute parameter does not automatically imply HF.

Two flavours:

```cpp
// (a) batch of ROIs from ONE image
const std::array<Rect, 5> rois{...};
Crop<>::build(rois)                        // => BatchRead, 5 planes

// (b) batch of SEPARATE same-size images
const std::array<Ptr2D<float>, 4> imgs{...};
PerThreadRead<ND::_2D, float>::build(imgs) // => BatchRead, 4 planes
```

- Batch size is a TEMPLATE parameter (`std::array`, not `std::vector`): each distinct N is a distinct kernel, compiled once.
- All planes run the SAME op sequence (for different sequences see DHF).
- Output can be a `Tensor<T>` with N planes or a compatible batched Write;
  output capacity and plane addressing must match the reads.
- The `activeBatch + defaultValue` overloads of `executeOperations` let a compiled batch size N process fewer than N real items.

## 4. Divergent Horizontal Fusion (DHF)

Different planes execute DIFFERENT fused sequences in one GPU kernel, selected
per-plane by a SequenceSelector (z -> **zero-based** sequence index):

```cpp
struct MySelector {
    FK_HOST_DEVICE_FUSE uint at(const uint& z) { return z == 0 ? 0u : 1u; }
};

const auto seq1 = buildOperationSequence(readA, Mul<float>::build(4.f), writeT);
const auto seq2 = buildOperationSequence(readB, Add<float>::build(50.f), writeT);
Executor<DivergentBatchTransformDPP<ParArch::GPU_NVIDIA, MySelector>>::
    executeOperations(stream, seq1, seq2);
```

- Each sequence must be a complete read->...->write chain. This example assumes
  two single-plane reads and `writeT` addressing a two-plane output tensor.
  Sequences need not share an output, but their writes must honor the global z
  coordinate and avoid overlap/out-of-bounds stores.
- The executor's grid.z is the SUM of the sequences' z extents — each sequence should cover exactly its own planes. Do NOT give every sequence a full-batch read or you will launch (and write) extra planes.
- In-tree user: `CircularTensor::update` (seq1 = preprocess+insert the new frame, seq2 = rotate-copy the other planes) — the temporal-video pattern from the paper.
- Selector convention: `at(z)` returns `0 .. numberOfSequences-1`. Read the
  current dispatch in `include/fused_kernel/core/execution_model/data_parallel_patterns.h`
  and `tests/examples/test_divergent_hf_executor.h`, not an older copied selector.

## Combining all four

One kernel can combine batch crops (HF), resize (BVF), normalization (VF), and
different per-plane sequences (DHF). Check sequence extents, selector coverage,
output ownership, and numeric tolerances; do not assume bitwise equality or
automatic fusion between arbitrary DPPs. Thread fusion (`TF::ENABLED`, vectorized
loads/stores) is a separate optimization, not another name for HF.

## Choosing

| situation | technique | how to express |
|---|---|---|
| chain of point ops | VF | just list them in order |
| crop/resize/warp before compute | BVF | put ReadBacks right after the read |
| N ROIs / N images, same processing | HF | pass std::array of rects/ptrs |
| N planes, different processing | DHF | sequences + SequenceSelector |
| rolling window of last N frames | CircularTensor | `update()` per frame |