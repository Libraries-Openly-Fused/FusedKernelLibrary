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

#include <tests/main.h>

#include <fused_kernel/core/core.h>
#include <fused_kernel/algorithms/image_processing/saturate.h>
#include <fused_kernel/fused_kernel.h>

#include <array>
#include <cstring>
#include <iostream>
#include <vector>

// Thread fusion reads and writes several elements per thread with bigger vector types.
// Widths that are not a multiple of the elements per thread, and pointers or pitches that
// are not aligned to the vector types, must give the same results as the non fused path.

constexpr std::array<uint, 9> testWidths{ 1, 3, 4, 5, 63, 65, 1023, 1025, 4097 };
constexpr uint testHeight = 7;

template <typename T>
inline bool sameValue(const T& a, const T& b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

template <typename T>
inline T testValue(const uint x, const uint y) {
    return fk::make_set<T>(static_cast<fk::VBase<T>>((x + 3 * y) % 100));
}

template <typename I, typename O, enum fk::ParArch PA, typename ReadIOp, typename WriteIOp>
inline void executeTFAndNoTF(fk::Stream_<PA>& stream, const ReadIOp& read,
                             const WriteIOp& writeTF, const WriteIOp& writeNoTF) {
    if constexpr (std::is_same_v<I, O>) {
        fk::executeOperations<fk::TransformDPP<PA, fk::TF::ENABLED>>(stream, read, writeTF);
        fk::executeOperations<fk::TransformDPP<PA, fk::TF::DISABLED>>(stream, read, writeNoTF);
    } else {
        fk::executeOperations<fk::TransformDPP<PA, fk::TF::ENABLED>>(stream, read, fk::SaturateCast<I, O>::build(), writeTF);
        fk::executeOperations<fk::TransformDPP<PA, fk::TF::DISABLED>>(stream, read, fk::SaturateCast<I, O>::build(), writeNoTF);
    }
}

// Buffers allocated by FKL with the default backend and memory type
template <typename I, typename O>
bool testFKLAllocatedWidths() {
    fk::Stream stream;
    bool passed = true;
    for (const uint width : testWidths) {
        fk::Ptr2D<I> input(width, testHeight);
        fk::Ptr2D<O> outputTF(width, testHeight);
        fk::Ptr2D<O> outputNoTF(width, testHeight);
        for (uint y = 0; y < testHeight; ++y) {
            for (uint x = 0; x < width; ++x) {
                input.at(x, y) = testValue<I>(x, y);
            }
        }
        input.upload(stream);

        executeTFAndNoTF<I, O>(stream, fk::PerThreadRead<fk::ND::_2D, I>::build(input),
                               fk::PerThreadWrite<fk::ND::_2D, O>::build(outputTF),
                               fk::PerThreadWrite<fk::ND::_2D, O>::build(outputNoTF));

        outputTF.download(stream);
        outputNoTF.download(stream);
        stream.sync();

        for (uint y = 0; y < testHeight; ++y) {
            for (uint x = 0; x < width; ++x) {
                if (!sameValue(outputTF.at(x, y), outputNoTF.at(x, y))) {
                    std::cout << "testFKLAllocatedWidths mismatch at width " << width << " (" << x << ", " << y << ")" << std::endl;
                    passed = false;
                }
            }
        }
    }
    return passed;
}

inline fk::Point testPoint(const uint x, const uint y, const uint z = 0) {
    return { static_cast<int>(x), static_cast<int>(y), static_cast<int>(z) };
}

// Tensors allocated by FKL with the default backend and memory type. Tensors are contiguous,
// so their pitch is not padded for thread fusion.
template <typename I, typename O>
bool testTensorWidths() {
    fk::Stream stream;
    constexpr uint planes = 2;
    bool passed = true;
    for (const uint width : testWidths) {
        fk::Tensor<I> input(width, testHeight, planes);
        fk::Tensor<O> outputTF(width, testHeight, planes);
        fk::Tensor<O> outputNoTF(width, testHeight, planes);
        for (uint z = 0; z < planes; ++z) {
            for (uint y = 0; y < testHeight; ++y) {
                for (uint x = 0; x < width; ++x) {
                    input.at(testPoint(x, y, z)) = testValue<I>(x, y + z);
                }
            }
        }
        input.upload(stream);

        executeTFAndNoTF<I, O>(stream, fk::TensorRead<I>::build(input.ptr()),
                               fk::TensorWrite<O>::build(outputTF.ptr()),
                               fk::TensorWrite<O>::build(outputNoTF.ptr()));

        outputTF.download(stream);
        outputNoTF.download(stream);
        stream.sync();

        for (uint z = 0; z < planes; ++z) {
            for (uint y = 0; y < testHeight; ++y) {
                for (uint x = 0; x < width; ++x) {
                    if (!sameValue(outputTF.at(testPoint(x, y, z)), outputNoTF.at(testPoint(x, y, z)))) {
                        std::cout << "testTensorWidths mismatch at width " << width << " (" << x << ", " << y << ", " << z << ")" << std::endl;
                        passed = false;
                    }
                }
            }
        }
    }
    return passed;
}

// Views starting at x = 1 of images allocated by FKL with the default backend, like crops or
// ROIs. The pitch is aligned, but the base pointer is not aligned to the thread fusion vector types.
template <typename I, typename O>
bool testCroppedViews() {
    fk::Stream stream;
    bool passed = true;
    for (const uint width : testWidths) {
        fk::Ptr2D<I> input(width + 1, testHeight);
        fk::Ptr2D<O> outputTF(width + 1, testHeight);
        fk::Ptr2D<O> outputNoTF(width + 1, testHeight);
        for (uint y = 0; y < testHeight; ++y) {
            for (uint x = 0; x < width + 1; ++x) {
                input.at(x, y) = testValue<I>(x, y);
                outputTF.at(x, y) = fk::make_set<O>(0);
                outputNoTF.at(x, y) = fk::make_set<O>(0);
            }
        }
        input.upload(stream);
        outputTF.upload(stream);
        outputNoTF.upload(stream);

        const fk::Point origin{ 1, 0, 0 };
        const auto inputView = input.crop2D(origin, fk::PtrDims<fk::ND::_2D>(width, testHeight, input.dims().pitch));
        const auto outputTFView = outputTF.crop2D(origin, fk::PtrDims<fk::ND::_2D>(width, testHeight, outputTF.dims().pitch));
        const auto outputNoTFView = outputNoTF.crop2D(origin, fk::PtrDims<fk::ND::_2D>(width, testHeight, outputNoTF.dims().pitch));

        executeTFAndNoTF<I, O>(stream, fk::PerThreadRead<fk::ND::_2D, I>::build(inputView),
                               fk::PerThreadWrite<fk::ND::_2D, O>::build(outputTFView),
                               fk::PerThreadWrite<fk::ND::_2D, O>::build(outputNoTFView));

        outputTF.download(stream);
        outputNoTF.download(stream);
        stream.sync();

        for (uint y = 0; y < testHeight; ++y) {
            for (uint x = 0; x < width + 1; ++x) {
                if (!sameValue(outputTF.at(x, y), outputNoTF.at(x, y))) {
                    std::cout << "testCroppedViews mismatch at width " << width << " (" << x << ", " << y << ")" << std::endl;
                    passed = false;
                }
            }
        }
    }
    return passed;
}

// Batch reads, plain and circular, of crops at x = 1 of images allocated by FKL with the default
// backend, written to tensors with aligned pitches: only the batch read params are misaligned.
template <typename I, typename O>
bool testBatchReadCroppedViews() {
    fk::Stream stream;
    constexpr size_t BATCH = 2;
    constexpr std::array<uint, 2> alignedWidths{ 64, 1024 };
    bool passed = true;
    for (const uint width : alignedWidths) {
        const auto makeInput = [&](const uint plane) {
            fk::Ptr2D<I> input(width + 1, testHeight);
            for (uint y = 0; y < testHeight; ++y) {
                for (uint x = 0; x < width + 1; ++x) {
                    input.at(x, y) = testValue<I>(x, y + plane);
                }
            }
            input.upload(stream);
            return input;
        };
        std::array<fk::Ptr2D<I>, BATCH> inputs{ makeInput(0), makeInput(1) };
        const fk::Point origin{ 1, 0, 0 };
        std::array<fk::Ptr2D<I>, BATCH> views{
            inputs[0].crop2D(origin, fk::PtrDims<fk::ND::_2D>(width, testHeight, inputs[0].dims().pitch)),
            inputs[1].crop2D(origin, fk::PtrDims<fk::ND::_2D>(width, testHeight, inputs[1].dims().pitch)) };

        fk::Read<fk::CircularBatchRead<fk::CircularDirection::Ascendent, fk::PerThreadRead<fk::ND::_2D, I>, BATCH>> circularRead;
        circularRead.params.first = 1;
        for (size_t i = 0; i < BATCH; ++i) {
            circularRead.params.opData[i].params = views[i].ptr();
        }

        fk::Tensor<O> batchTF(width, testHeight, BATCH);
        fk::Tensor<O> batchNoTF(width, testHeight, BATCH);
        fk::Tensor<O> circularTF(width, testHeight, BATCH);
        fk::Tensor<O> circularNoTF(width, testHeight, BATCH);

        executeTFAndNoTF<I, O>(stream, fk::PerThreadRead<fk::ND::_2D, I>::build(views),
                               fk::TensorWrite<O>::build(batchTF.ptr()),
                               fk::TensorWrite<O>::build(batchNoTF.ptr()));
        executeTFAndNoTF<I, O>(stream, circularRead,
                               fk::TensorWrite<O>::build(circularTF.ptr()),
                               fk::TensorWrite<O>::build(circularNoTF.ptr()));

        batchTF.download(stream);
        batchNoTF.download(stream);
        circularTF.download(stream);
        circularNoTF.download(stream);
        stream.sync();

        for (uint z = 0; z < BATCH; ++z) {
            for (uint y = 0; y < testHeight; ++y) {
                for (uint x = 0; x < width; ++x) {
                    const fk::Point p = testPoint(x, y, z);
                    if (!sameValue(batchTF.at(p), batchNoTF.at(p)) || !sameValue(circularTF.at(p), circularNoTF.at(p))) {
                        std::cout << "testBatchReadCroppedViews mismatch at width " << width << " (" << x << ", " << y << ", " << z << ")" << std::endl;
                        passed = false;
                    }
                }
            }
        }
    }
    return passed;
}

// Host views starting one element into a buffer, with an unpadded pitch, like RawPtrs wrapping
// user memory. Neither the pointer nor the pitch is aligned to the thread fusion vector types.
template <typename I, typename O>
bool testCPUMisalignedViews() {
    fk::Stream_<fk::ParArch::CPU> stream;
    bool passed = true;
    for (const uint width : testWidths) {
        const uint storageWidth = width + 1;
        std::vector<I> inputStorage(storageWidth * testHeight);
        std::vector<O> outputTFStorage(storageWidth * testHeight, fk::make_set<O>(0));
        std::vector<O> outputNoTFStorage(storageWidth * testHeight, fk::make_set<O>(0));
        for (uint y = 0; y < testHeight; ++y) {
            for (uint x = 0; x < width; ++x) {
                inputStorage[y * storageWidth + x + 1] = testValue<I>(x, y);
            }
        }
        const fk::RawPtr<fk::ND::_2D, I> input{ inputStorage.data() + 1, { width, testHeight, static_cast<uint>(storageWidth * sizeof(I)) } };
        const fk::RawPtr<fk::ND::_2D, O> outputTF{ outputTFStorage.data() + 1, { width, testHeight, static_cast<uint>(storageWidth * sizeof(O)) } };
        const fk::RawPtr<fk::ND::_2D, O> outputNoTF{ outputNoTFStorage.data() + 1, { width, testHeight, static_cast<uint>(storageWidth * sizeof(O)) } };

        executeTFAndNoTF<I, O>(stream, fk::PerThreadRead<fk::ND::_2D, I>::build(input),
                               fk::PerThreadWrite<fk::ND::_2D, O>::build(outputTF),
                               fk::PerThreadWrite<fk::ND::_2D, O>::build(outputNoTF));

        for (size_t i = 0; i < outputTFStorage.size(); ++i) {
            if (!sameValue(outputTFStorage[i], outputNoTFStorage[i])) {
                std::cout << "testCPUMisalignedViews mismatch at width " << width << " element " << i << std::endl;
                passed = false;
            }
        }
    }
    return passed;
}

// Host tensors are contiguous, so their pitch is not padded for thread fusion
template <typename I, typename O>
bool testCPUTensorWidths() {
    fk::Stream_<fk::ParArch::CPU> stream;
    constexpr uint planes = 2;
    bool passed = true;
    for (const uint width : testWidths) {
        fk::Tensor<I> input(width, testHeight, planes, 1, fk::MemType::Host);
        fk::Tensor<O> outputTF(width, testHeight, planes, 1, fk::MemType::Host);
        fk::Tensor<O> outputNoTF(width, testHeight, planes, 1, fk::MemType::Host);
        for (uint z = 0; z < planes; ++z) {
            for (uint y = 0; y < testHeight; ++y) {
                for (uint x = 0; x < width; ++x) {
                    *fk::PtrAccessor<fk::ND::_3D>::point(fk::Point{ static_cast<int>(x), static_cast<int>(y), static_cast<int>(z) }, input.ptr()) =
                        testValue<I>(x, y + z);
                }
            }
        }

        executeTFAndNoTF<I, O>(stream, fk::TensorRead<I>::build(input.ptr()),
                               fk::TensorWrite<O>::build(outputTF.ptr()),
                               fk::TensorWrite<O>::build(outputNoTF.ptr()));

        for (uint z = 0; z < planes; ++z) {
            for (uint y = 0; y < testHeight; ++y) {
                for (uint x = 0; x < width; ++x) {
                    const fk::Point p{ static_cast<int>(x), static_cast<int>(y), static_cast<int>(z) };
                    if (!sameValue(*fk::PtrAccessor<fk::ND::_3D>::cr_point(p, outputTF.ptr()), *fk::PtrAccessor<fk::ND::_3D>::cr_point(p, outputNoTF.ptr()))) {
                        std::cout << "testCPUTensorWidths mismatch at width " << width << " (" << x << ", " << y << ", " << z << ")" << std::endl;
                        passed = false;
                    }
                }
            }
        }
    }
    return passed;
}

template <typename I, typename O>
bool testAll() {
    // Default backend: CPU in the _cpp test, GPU in the _cu test
    const bool fklAllocated = testFKLAllocatedWidths<I, O>();
    const bool tensors = testTensorWidths<I, O>();
    const bool croppedViews = testCroppedViews<I, O>();
    const bool batchReads = testBatchReadCroppedViews<I, O>();
    // CPU backend in both tests
    const bool cpuMisalignedViews = testCPUMisalignedViews<I, O>();
    const bool cpuTensors = testCPUTensorWidths<I, O>();
    return fklAllocated && tensors && croppedViews && batchReads && cpuMisalignedViews && cpuTensors;
}

int launch() {
    bool passed = true;
    passed &= testAll<uchar, float>();
    passed &= testAll<uchar, uchar>();
    passed &= testAll<float, float>();
    passed &= testAll<uchar, double>();
    passed &= testAll<uchar2, float2>();
    passed &= testAll<ushort, float>();
    passed &= testAll<float, double>();
    return passed ? 0 : -1;
}
