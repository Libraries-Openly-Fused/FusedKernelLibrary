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

// Execution tests for thread fusion (TransformDPP<PA, TF::ENABLED>): every pipeline is executed
// twice, with TF::DISABLED and TF::ENABLED, and both outputs must match. Covers the
// contiguous_data Operations (PerThreadRead/Write, TensorRead/Write), the forwarded_access
// wrappers (BatchRead/BatchWrite, CircularBatchRead), non divisible widths (remainder path),
// heterogeneous per-plane batch widths and type changing pipelines.

#include <tests/main.h>

#include <iostream>

#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/data/ptr_utils.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/basic_ops/logical.h>
#include <fused_kernel/algorithms/image_processing/saturate.h>
#include <fused_kernel/fused_kernel.h>

template <typename T, fk::ND D>
void fillWithPattern(fk::Ptr<D, T>& ptr, fk::Stream& stream) {
    const auto dims = ptr.dims();
    const int planes = [&]() {
        if constexpr (D == fk::ND::_3D) { return static_cast<int>(dims.planes); } else { return 1; }
    }();
    const int height = [&]() {
        if constexpr (D == fk::ND::_1D) { return 1; } else { return static_cast<int>(dims.height); }
    }();
    for (int z = 0; z < planes; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < static_cast<int>(dims.width); ++x) {
                using Base = fk::VBase<T>;
                const Base value = static_cast<Base>((x + (y * 3) + (z * 7)) % 100);
                ptr.at(fk::Point{ x, y, z }) = fk::make_set<T>(value);
            }
        }
    }
    ptr.upload(stream);
}

template <typename T, fk::ND D>
bool outputsMatch(const fk::Ptr<D, T>& expected, const fk::Ptr<D, T>& result) {
    const auto dims = expected.dims();
    const int planes = [&]() {
        if constexpr (D == fk::ND::_3D) { return static_cast<int>(dims.planes); } else { return 1; }
    }();
    const int height = [&]() {
        if constexpr (D == fk::ND::_1D) { return 1; } else { return static_cast<int>(dims.height); }
    }();
    for (int z = 0; z < planes; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < static_cast<int>(dims.width); ++x) {
                const fk::Point p{ x, y, z };
                if (!fk::Equal<T>::exec(fk::make_tuple(expected.at(p), result.at(p)))) {
                    std::cout << "Mismatch at (" << x << ", " << y << ", " << z << ")" << std::endl;
                    return false;
                }
            }
        }
    }
    return true;
}

template <typename T>
bool testSameTypeIO(const uint width, const uint height) {
    fk::Stream stream;
    fk::Ptr2D<T> input(width, height);
    fillWithPattern(input, stream);
    fk::Ptr2D<T> outputNormal(width, height);
    fk::Ptr2D<T> outputTF(width, height);

    fk::executeOperations<fk::TransformDPP<fk::defaultParArch, fk::TF::DISABLED>>(stream,
        fk::PerThreadRead<fk::ND::_2D, T>::build(input),
        fk::PerThreadWrite<fk::ND::_2D, T>::build(outputNormal));
    fk::executeOperations<fk::TransformDPP<fk::defaultParArch, fk::TF::ENABLED>>(stream,
        fk::PerThreadRead<fk::ND::_2D, T>::build(input),
        fk::PerThreadWrite<fk::ND::_2D, T>::build(outputTF));

    outputNormal.download(stream);
    outputTF.download(stream);
    stream.sync();

    return outputsMatch(outputNormal, outputTF);
}

template <typename I, typename O>
bool testDifferentTypeIO(const uint width, const uint height) {
    fk::Stream stream;
    fk::Ptr2D<I> input(width, height);
    fillWithPattern(input, stream);
    fk::Ptr2D<O> outputNormal(width, height);
    fk::Ptr2D<O> outputTF(width, height);

    fk::executeOperations<fk::TransformDPP<fk::defaultParArch, fk::TF::DISABLED>>(stream,
        fk::PerThreadRead<fk::ND::_2D, I>::build(input),
        fk::SaturateCast<I, O>::build(),
        fk::PerThreadWrite<fk::ND::_2D, O>::build(outputNormal));
    fk::executeOperations<fk::TransformDPP<fk::defaultParArch, fk::TF::ENABLED>>(stream,
        fk::PerThreadRead<fk::ND::_2D, I>::build(input),
        fk::SaturateCast<I, O>::build(),
        fk::PerThreadWrite<fk::ND::_2D, O>::build(outputTF));

    outputNormal.download(stream);
    outputTF.download(stream);
    stream.sync();

    return outputsMatch(outputNormal, outputTF);
}

template <typename T>
bool testTensorIO(const uint width, const uint height, const uint planes) {
    fk::Stream stream;
    fk::Tensor<T> input(width, height, planes);
    fillWithPattern(input, stream);
    fk::Tensor<T> outputNormal(width, height, planes);
    fk::Tensor<T> outputTF(width, height, planes);

    fk::executeOperations<fk::TransformDPP<fk::defaultParArch, fk::TF::DISABLED>>(stream,
        fk::TensorRead<T>::build(input.ptr()),
        fk::TensorWrite<T>::build(outputNormal.ptr()));
    fk::executeOperations<fk::TransformDPP<fk::defaultParArch, fk::TF::ENABLED>>(stream,
        fk::TensorRead<T>::build(input.ptr()),
        fk::TensorWrite<T>::build(outputTF.ptr()));

    outputNormal.download(stream);
    outputTF.download(stream);
    stream.sync();

    return outputsMatch(outputNormal, outputTF);
}

template <typename T, size_t BATCH>
bool testBatchIO(const uint width, const uint height) {
    fk::Stream stream;
    std::array<fk::Ptr2D<T>, BATCH> inputs;
    std::array<fk::Ptr2D<T>, BATCH> outputsNormal;
    std::array<fk::Ptr2D<T>, BATCH> outputsTF;
    for (size_t i = 0; i < BATCH; ++i) {
        inputs[i] = fk::Ptr2D<T>(width, height);
        fillWithPattern(inputs[i], stream);
        outputsNormal[i] = fk::Ptr2D<T>(width, height);
        outputsTF[i] = fk::Ptr2D<T>(width, height);
    }

    fk::executeOperations<fk::TransformDPP<fk::defaultParArch, fk::TF::DISABLED>>(stream,
        fk::PerThreadRead<fk::ND::_2D, T>::build(inputs),
        fk::PerThreadWrite<fk::ND::_2D, T>::build(outputsNormal));
    fk::executeOperations<fk::TransformDPP<fk::defaultParArch, fk::TF::ENABLED>>(stream,
        fk::PerThreadRead<fk::ND::_2D, T>::build(inputs),
        fk::PerThreadWrite<fk::ND::_2D, T>::build(outputsTF));

    bool passed = true;
    for (size_t i = 0; i < BATCH; ++i) {
        outputsNormal[i].download(stream);
        outputsTF[i].download(stream);
    }
    stream.sync();
    for (size_t i = 0; i < BATCH; ++i) {
        passed &= outputsMatch(outputsNormal[i], outputsTF[i]);
    }
    return passed;
}

// Heterogeneous per-plane widths on the write side: plane 0 is divisible by elems_per_thread but
// the narrower planes are not, so isThreadDivisible must consider every plane (not just plane 0)
// and select the remainder-path kernel variant. All planes share the same pitch, sized for the
// widest plane, so the accesses that the batch execution model performs beyond a narrower plane's
// width (activeThreads covers the widest plane) stay inside each allocation on both backends.
template <typename T, size_t BATCH>
bool testHeterogeneousBatchWidths(const uint maxWidth, const uint height) {
    fk::Stream stream;
    const uint pitch = static_cast<uint>(maxWidth * sizeof(T));
    std::array<fk::Ptr2D<T>, BATCH> inputs;
    std::array<fk::Ptr2D<T>, BATCH> outputsNormal;
    std::array<fk::Ptr2D<T>, BATCH> outputsTF;
    for (size_t i = 0; i < BATCH; ++i) {
        inputs[i] = fk::Ptr2D<T>(maxWidth, height, pitch);
        fillWithPattern(inputs[i], stream);
        const uint planeWidth = maxWidth - static_cast<uint>(i);
        outputsNormal[i] = fk::Ptr2D<T>(planeWidth, height, pitch);
        outputsTF[i] = fk::Ptr2D<T>(planeWidth, height, pitch);
    }

    fk::executeOperations<fk::TransformDPP<fk::defaultParArch, fk::TF::DISABLED>>(stream,
        fk::PerThreadRead<fk::ND::_2D, T>::build(inputs),
        fk::PerThreadWrite<fk::ND::_2D, T>::build(outputsNormal));
    fk::executeOperations<fk::TransformDPP<fk::defaultParArch, fk::TF::ENABLED>>(stream,
        fk::PerThreadRead<fk::ND::_2D, T>::build(inputs),
        fk::PerThreadWrite<fk::ND::_2D, T>::build(outputsTF));

    bool passed = true;
    for (size_t i = 0; i < BATCH; ++i) {
        outputsNormal[i].download(stream);
        outputsTF[i].download(stream);
    }
    stream.sync();
    for (size_t i = 0; i < BATCH; ++i) {
        passed &= outputsMatch(outputsNormal[i], outputsTF[i]);
    }
    return passed;
}

template <typename T, int BATCH>
bool testCircularBatchIO(const uint width, const uint height) {
    using ReadOp = fk::PerThreadRead<fk::ND::_2D, T>;
    using CircularRead = fk::CircularBatchRead<fk::CircularDirection::Ascendent, ReadOp, BATCH>;

    fk::Stream stream;
    std::array<fk::Ptr2D<T>, BATCH> inputs;
    typename CircularRead::ParamsType circularParams{};
    circularParams.first = 1;
    for (int i = 0; i < BATCH; ++i) {
        inputs[i] = fk::Ptr2D<T>(width, height);
        fillWithPattern(inputs[i], stream);
        circularParams.opData[i] = { inputs[i].ptr() };
    }
    fk::Tensor<T> outputNormal(width, height, BATCH);
    fk::Tensor<T> outputTF(width, height, BATCH);

    fk::executeOperations<fk::TransformDPP<fk::defaultParArch, fk::TF::DISABLED>>(stream,
        CircularRead::build(circularParams),
        fk::TensorWrite<T>::build(outputNormal.ptr()));
    fk::executeOperations<fk::TransformDPP<fk::defaultParArch, fk::TF::ENABLED>>(stream,
        CircularRead::build(circularParams),
        fk::TensorWrite<T>::build(outputTF.ptr()));

    outputNormal.download(stream);
    outputTF.download(stream);
    stream.sync();

    return outputsMatch(outputNormal, outputTF);
}

int launch() {
    bool passed = true;

    // contiguous_data Operations, divisible widths (vectorized path only)
    passed &= testSameTypeIO<uchar>(512, 64);   // elems_per_thread = 4
    passed &= testSameTypeIO<uchar2>(512, 64);  // elems_per_thread = 2
    passed &= testSameTypeIO<uchar3>(512, 64);  // elems_per_thread = 1
    passed &= testSameTypeIO<float>(512, 64);   // elems_per_thread = 2
    passed &= testSameTypeIO<float4>(512, 64);  // elems_per_thread = 1

    // non divisible widths (vectorized path + remainder path)
    passed &= testSameTypeIO<uchar>(1023, 64);
    passed &= testSameTypeIO<float>(1021, 64);

    // type changing pipeline
    passed &= testDifferentTypeIO<uchar, float>(512, 64);
    passed &= testDifferentTypeIO<uchar, float>(1023, 64);

    // Tensor (3D) contiguous_data Operations. Tensor pitch is not padded (pitch == width *
    // sizeof(T)), so thread fusion on Tensors requires widths that keep every row aligned to
    // the vectorized load/store: only divisible widths are exercised here.
    passed &= testTensorIO<float>(256, 32, 3);
    passed &= testTensorIO<uchar>(256, 32, 3);

    // forwarded_access wrappers: BatchRead/BatchWrite (horizontal fusion). Ptr2D rows are
    // pitch-padded by the allocator, so non divisible widths are fine here.
    passed &= testBatchIO<uchar, 3>(512, 32);
    passed &= testBatchIO<float, 3>(511, 32);

    // heterogeneous per-plane widths: plane 0 divisible, narrower planes not — the thread
    // divisibility decision must consider every z plane
    passed &= testHeterogeneousBatchWidths<uchar, 4>(512, 32);
    passed &= testHeterogeneousBatchWidths<float, 4>(512, 32);

    // forwarded_access wrappers: CircularBatchRead (writing to a Tensor: divisible widths only)
    passed &= testCircularBatchIO<uchar, 3>(512, 32);
    passed &= testCircularBatchIO<float, 3>(512, 32);

    if (passed) {
        std::cout << "test_thread_fusion_exec Passed!!!" << std::endl;
        return 0;
    } else {
        std::cout << "test_thread_fusion_exec Failed!!!" << std::endl;
        return -1;
    }
}
