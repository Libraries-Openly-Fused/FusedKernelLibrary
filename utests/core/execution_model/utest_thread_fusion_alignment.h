/* Copyright 2026 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

// Thread fusion (TF::ENABLED) reads and writes rows as vectors of several elements, which need aligned addresses. When
// the buffers of the first read or the last write are not aligned for them (rows that are not padded, crops, external
// pointers), the executors run the version without thread fusion (issue #372). Before, the CPU backend crashed (clang)
// and the GPU reported "misaligned address".

#include <tests/main.h>

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/cast.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>

#include <array>
#include <iostream>

namespace fk {

// ---------------------------------------------------------------------------------------------------------------------
// The decision, on host buffers with chosen base offsets and pitches (no execution, the same on every backend)
// ---------------------------------------------------------------------------------------------------------------------
alignas(64) static unsigned char alignmentBuffer[1 << 16];

template <typename T>
RawPtr<ND::_2D, T> rawPtr2D(const uint baseOffset, const uint width, const uint pitch) {
    return { reinterpret_cast<T*>(alignmentBuffer + baseOffset), { width, 4, pitch } };
}

template <typename I, typename O>
bool fusedAligned(const RawPtr<ND::_2D, I>& input, const RawPtr<ND::_2D, O>& output) {
    const auto details = TransformDPP<defaultParArch, TF::ENABLED>::build_details(
        PerThreadRead<ND::_2D, I>::build(input), Cast<I, O>::build(), PerThreadWrite<ND::_2D, O>::build(output));
    return details.threadAligned;
}

bool testAlignmentDecision() {
    bool correct = true;
    const auto check = [&](const bool value, const bool expected, const char* what) {
        if (value != expected) {
            std::cout << "Thread fusion alignment: wrong decision for " << what << std::endl;
            correct = false;
        }
    };
    // uchar -> float: reads Uchar4 (alignment 4), writes Float4 (alignment 16)
    check(fusedAligned(rawPtr2D<uchar>(0, 1000, 1024), rawPtr2D<float>(0, 1000, 4096)), true, "aligned buffers");
    check(fusedAligned(rawPtr2D<uchar>(1, 1000, 1024), rawPtr2D<float>(0, 1000, 4096)), false, "input base + 1");
    check(fusedAligned(rawPtr2D<uchar>(0, 1000, 1001), rawPtr2D<float>(0, 1000, 4096)), false, "input pitch 1001");
    check(fusedAligned(rawPtr2D<uchar>(0, 1000, 1004), rawPtr2D<float>(0, 1000, 4096)), true, "input pitch 1004");
    check(fusedAligned(rawPtr2D<uchar>(0, 1000, 1024), rawPtr2D<float>(8, 1000, 4096)), false, "output base + 8");
    check(fusedAligned(rawPtr2D<uchar>(0, 1000, 1024), rawPtr2D<float>(16, 1000, 4096)), true, "output base + 16");
    check(fusedAligned(rawPtr2D<uchar>(0, 1000, 1024), rawPtr2D<float>(0, 1000, 4008)), false, "output pitch 4008");
    // uchar -> double: writes Double4, 32 bytes with alignment 16: the alignment matters, not the size
    check(fusedAligned(rawPtr2D<uchar>(0, 1000, 1024), rawPtr2D<double>(16, 1000, 8016)), true, "Double4 at 16 bytes");
    check(fusedAligned(rawPtr2D<uchar>(0, 1000, 1024), rawPtr2D<double>(8, 1000, 8016)), false, "Double4 at 8 bytes");
    // float -> float: Float2, alignment 8
    check(fusedAligned(rawPtr2D<float>(0, 1000, 4004), rawPtr2D<float>(0, 1000, 4096)), false, "float pitch 4004");
    check(fusedAligned(rawPtr2D<float>(0, 1000, 4008), rawPtr2D<float>(0, 1000, 4096)), true, "float pitch 4008");
    // uchar3 -> float3: one element per thread, nothing to align
    check(fusedAligned(rawPtr2D<uchar3>(1, 1000, 3001), rawPtr2D<float3>(4, 1000, 12004)), true, "uchar3 -> float3");

    // 3D: the plane stride (plane_pitch * color_planes) has to be aligned too
    const RawPtr<ND::_3D, uchar> input3D{ reinterpret_cast<uchar*>(alignmentBuffer), { 1000, 4, 2, 1, 1024 } };
    RawPtr<ND::_3D, float> output3D{ reinterpret_cast<float*>(alignmentBuffer), { 1000, 4, 2, 1, 4096 } };
    const auto details3D = [&]() {
        return TransformDPP<defaultParArch, TF::ENABLED>::build_details(TensorRead<uchar>::build(input3D),
            Cast<uchar, float>::build(), TensorWrite<float>::build(output3D));
    };
    check(details3D().threadAligned, true, "aligned 3D buffers");
    output3D.dims.plane_pitch = 4096 * 4 + 8;
    check(details3D().threadAligned, false, "3D plane pitch + 8");

    // The 1D and T3D strides
    using Uchar4V = VectorType_t<uchar, 4>;
    using Float4V = VectorType_t<float, 4>;
    check(isAlignedFor<Uchar4V>(RawPtr<ND::_1D, uchar>{ reinterpret_cast<uchar*>(alignmentBuffer), { 1000, 1000 } }),
          true, "1D aligned");
    check(isAlignedFor<Uchar4V>(RawPtr<ND::_1D, uchar>{ reinterpret_cast<uchar*>(alignmentBuffer + 2), { 1000, 1000 } }),
          false, "1D base + 2");
    RawPtr<ND::T3D, float> t3D{ reinterpret_cast<float*>(alignmentBuffer), PtrDims<ND::T3D>(1000, 4, 2, 3) };
    t3D.dims.pitch = 4096;
    t3D.dims.plane_pitch = 4096 * 4;
    t3D.dims.color_planes_pitch = 4096 * 4 * 2;
    check(isAlignedFor<Float4V>(t3D), true, "T3D aligned");
    t3D.dims.color_planes_pitch += 4;
    check(isAlignedFor<Float4V>(t3D), false, "T3D color planes pitch + 4");

    // Composite operations check every plane they can access
    using Read2D = PerThreadRead<ND::_2D, uchar>;
    using Write2D = PerThreadWrite<ND::_2D, float>;
    const OperationData<Read2D> alignedRead{ rawPtr2D<uchar>(0, 1000, 1024) };
    const OperationData<Read2D> misalignedRead{ rawPtr2D<uchar>(1, 1000, 1024) };
    const auto alignedWrite = rawPtr2D<float>(0, 1000, 4096);
    const auto misalignedWrite = rawPtr2D<float>(8, 1000, 4096);
    constexpr uint EPT = 4; // uchar -> float
    using BatchWrite2 = BatchWrite<2, Write2D>;
    check(isOperationThreadFusionAligned<BatchWrite2, EPT>({ { alignedWrite, alignedWrite } }), true, "BatchWrite aligned");
    check(isOperationThreadFusionAligned<BatchWrite2, EPT>({ { alignedWrite, misalignedWrite } }), false,
          "BatchWrite with a misaligned plane");
    using ConditionalRead = BatchRead<PlanePolicy::CONDITIONAL_WITH_DEFAULT, 2, Read2D>;
    typename ConditionalRead::OperationDataType conditional{};
    conditional.params.opData[0] = alignedRead;
    conditional.params.opData[1] = misalignedRead;
    check(isOperationThreadFusionAligned<ConditionalRead, EPT>(conditional), false,
          "conditional BatchRead with a misaligned plane");
    using CircularRead = CircularBatchRead<CircularDirection::Ascendent, Read2D, 2>;
    typename CircularRead::OperationDataType circularRead{};
    circularRead.params.opData[0] = alignedRead;
    circularRead.params.opData[1] = alignedRead;
    check(isOperationThreadFusionAligned<CircularRead, EPT>(circularRead), true, "CircularBatchRead aligned");
    circularRead.params.opData[1] = misalignedRead;
    check(isOperationThreadFusionAligned<CircularRead, EPT>(circularRead), false,
          "CircularBatchRead with a misaligned plane");
    using CircularWrite = CircularBatchWrite<CircularDirection::Descendent, Write2D, 2>;
    typename CircularWrite::OperationDataType circularWrite{};
    circularWrite.params.opData[0] = OperationData<Write2D>{ misalignedWrite };
    circularWrite.params.opData[1] = OperationData<Write2D>{ alignedWrite };
    check(isOperationThreadFusionAligned<CircularWrite, EPT>(circularWrite), false,
          "CircularBatchWrite with a misaligned plane");
    using CircularTRead = CircularTensorRead<CircularDirection::Ascendent, TensorRead<uchar>, 2>;
    typename CircularTRead::OperationDataType circularTRead{};
    circularTRead.params.opData = OperationData<TensorRead<uchar>>{ input3D };
    check(isOperationThreadFusionAligned<CircularTRead, EPT>(circularTRead), true, "CircularTensorRead aligned");
    using CircularTWrite = CircularTensorWrite<CircularDirection::Ascendent, TensorWrite<float>, 2>;
    typename CircularTWrite::OperationDataType circularTWrite{};
    circularTWrite.params.opData = OperationData<TensorWrite<float>>{ output3D };
    check(isOperationThreadFusionAligned<CircularTWrite, EPT>(circularTWrite), false,
          "CircularTensorWrite with a misaligned plane stride");
    return correct;
}

// ---------------------------------------------------------------------------------------------------------------------
// Executions with misaligned buffers give the right result on every backend
// ---------------------------------------------------------------------------------------------------------------------
constexpr int H = 37;

uchar inputValue(const int x, const int y, const int z = 0) {
    return static_cast<uchar>((x * 7 + y * 13 + z * 29) & 0xFF);
}

Ptr2D<uchar> makeInput(Stream& stream, const int width) {
    Ptr2D<uchar> input(static_cast<uint>(width), H);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < width; ++x) {
            input.at(x, y) = inputValue(x, y);
        }
    }
    input.upload(stream);
    return input;
}

template <typename Input, typename Output>
void runFused(Stream& stream, const Input& input, const Output& output) {
    executeOperations<TransformDPP<defaultParArch, TF::ENABLED>>(stream, PerThreadRead<ND::_2D, uchar>::build(input.ptr()),
        Cast<uchar, float>::build(), PerThreadWrite<ND::_2D, float>::build(output.ptr()));
}

// Rows that are not padded (on the CPU, a host Ptr2D of 1001 uchar has a pitch of 1001 bytes)
bool testUnpaddedRows(Stream& stream) {
    constexpr int W = 1001;
    const Ptr2D<uchar> input = makeInput(stream, W);
    Ptr2D<float> output(W, H);
    runFused(stream, input, output);
    output.download(stream);
    stream.sync();
    bool correct = true;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            correct &= output.at(x, y) == static_cast<float>(inputValue(x, y));
        }
    }
    return correct;
}

// Input cropped at x = 1 (Uchar4 reads at base + 1) and output cropped at x = 2 (Float4 writes at base + 8)
bool testCrops(Stream& stream) {
    constexpr int W = 1000;
    Ptr2D<uchar> inputParent = makeInput(stream, W + 4);
    const Ptr2D<uchar> input = inputParent.crop(Point{ 1, 0, 0 }, PtrDims<ND::_2D>{ W, H, inputParent.dims().pitch });
    Ptr2D<float> outputParent(W + 4, H);
    const Ptr2D<float> output = outputParent.crop(Point{ 2, 0, 0 }, PtrDims<ND::_2D>{ W, H, outputParent.dims().pitch });
    runFused(stream, input, output);
    outputParent.download(stream);
    stream.sync();
    bool correct = true;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            correct &= outputParent.at(x + 2, y) == static_cast<float>(inputValue(x + 1, y));
        }
    }
    return correct;
}

// Tensors are contiguous: with an odd width their rows are not aligned on any backend
bool testTensor(Stream& stream) {
    constexpr int W = 1001, PLANES = 3;
    Tensor<uchar> input(W, H, PLANES);
    for (int z = 0; z < PLANES; ++z) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                input.at(x, y, z) = inputValue(x, y, z);
            }
        }
    }
    input.upload(stream);
    Tensor<float> output(W, H, PLANES);
    executeOperations<TransformDPP<defaultParArch, TF::ENABLED>>(stream, TensorRead<uchar>::build(input.ptr()),
        Cast<uchar, float>::build(), TensorWrite<float>::build(output.ptr()));
    output.download(stream);
    stream.sync();
    bool correct = true;
    for (int z = 0; z < PLANES; ++z) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                correct &= output.at(x, y, z) == static_cast<float>(inputValue(x, y, z));
            }
        }
    }
    return correct;
}

// A batch where only the second plane is misaligned (a crop at x = 1)
bool testBatchWithOneMisalignedPlane(Stream& stream) {
    constexpr int W = 1000;
    const Ptr2D<uchar> plane0 = makeInput(stream, W);
    Ptr2D<uchar> parent1 = makeInput(stream, W + 4);
    const Ptr2D<uchar> plane1 = parent1.crop(Point{ 1, 0, 0 }, PtrDims<ND::_2D>{ W, H, parent1.dims().pitch });
    const std::array<Ptr2D<uchar>, 2> inputs{ plane0, plane1 };
    Tensor<float> output(W, H, 2);
    executeOperations<TransformDPP<defaultParArch, TF::ENABLED>>(inputs, output, stream, Cast<uchar, float>::build());
    output.download(stream);
    stream.sync();
    bool correct = true;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            correct &= output.at(x, y, 0) == static_cast<float>(inputValue(x, y));
            correct &= output.at(x, y, 1) == static_cast<float>(inputValue(x + 1, y));
        }
    }
    return correct;
}

int launch_impl() {
    Stream stream;
    const bool decision = testAlignmentDecision();
    const bool unpadded = testUnpaddedRows(stream);
    const bool crops = testCrops(stream);
    const bool tensor = testTensor(stream);
    const bool batch = testBatchWithOneMisalignedPlane(stream);
    std::cout << "Thread fusion alignment: decision " << decision << ", unpadded rows " << unpadded << ", crops "
              << crops << ", tensor " << tensor << ", batch with a misaligned plane " << batch << std::endl;
    return decision && unpadded && crops && tensor && batch ? 0 : -1;
}

} // namespace fk

int launch() {
    return fk::launch_impl();
}
