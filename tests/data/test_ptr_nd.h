/* Copyright 2024 Oscar Amoros Huguet

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

#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/data/ptr_utils.h>
#include <fused_kernel/core/utils/cuda_vector_utils.h>
#include <fused_kernel/algorithms/basic_ops/cuda_vector.h>
#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/core/execution_model/stream.h>

#include <iostream>

using namespace fk;

using PtrToTest = Ptr2D<uchar3>;
constexpr int WIDTH = 64;
constexpr int HEIGHT = 64;

PtrToTest test_return_by_value() {
    return PtrToTest(WIDTH, HEIGHT);
}

const PtrToTest& test_return_by_const_reference(const PtrToTest& somePtr) {
    return somePtr;
}

PtrToTest& test_return_by_reference(PtrToTest& somePtr) {
    return somePtr;
}

void test_uploadTo(Stream& stream) {
#if defined(__NVCC__) || CLANG_HOST
    // Device pointers
    Ptr1D<uchar3> test1D(1333, 0, MemType::Device);
    Ptr2D<uchar3> test2D(1333, 444, 0, MemType::Device);
    Ptr3D<uchar3> test3D(1333, 444, 22, 1, 0, MemType::Device);
    Tensor<uchar3> testTensor(1333, 444, 22, 1, MemType::Device);

    // Host Pinned Pointers
    Ptr1D<uchar3> test1D_h(1333, 0, MemType::HostPinned);
    Ptr2D<uchar3> test2D_h(1333, 444, 0, MemType::HostPinned);
    Ptr3D<uchar3> test3D_h(1333, 444, 22, 1, 0, MemType::HostPinned);
    Tensor<uchar3> testTensor_h(1333, 444, 22, 1, MemType::HostPinned);

    // Must work
    test1D_h.uploadTo(test1D, stream);
    test2D_h.uploadTo(test2D, stream);
    test3D_h.uploadTo(test3D, stream);
    testTensor_h.uploadTo(testTensor, stream);

    stream.sync();

    // Must not work
    try {
        test1D.uploadTo(test1D_h, stream);
    }
    catch (const std::exception& e) {
        std::cout << "Expected exception: " << e.what() << std::endl;
    }

    // Compile time error
    // test2D_h.uploadTo(test3D);
#endif
}

void test_downloadTo(Stream& stream) {
#if defined(__NVCC__) || CLANG_HOST
    // Device pointers
    Ptr1D<uchar3> test1D(1333, 0, MemType::Device);
    Ptr2D<uchar3> test2D(1333, 444, 0, MemType::Device);
    Ptr3D<uchar3> test3D(1333, 444, 22, 1, 0, MemType::Device);
    Tensor<uchar3> testTensor(1333, 444, 22, 1, MemType::Device);

    // Host Pinned Pointers
    Ptr1D<uchar3> test1D_h(1333, 0, MemType::HostPinned);
    Ptr2D<uchar3> test2D_h(1333, 444, 0, MemType::HostPinned);
    Ptr3D<uchar3> test3D_h(1333, 444, 22, 1, 0, MemType::HostPinned);
    Tensor<uchar3> testTensor_h(1333, 444, 22, 1, MemType::HostPinned);

    // Must work
    test1D.downloadTo(test1D_h, stream);
    test2D.downloadTo(test2D_h, stream);
    test3D.downloadTo(test3D_h, stream);
    testTensor.downloadTo(testTensor_h, stream);

    stream.sync();

    // Must not work
    try {
        test1D_h.downloadTo(test1D, stream);
    } catch (const std::exception& e) {
        std::cout << "Expected exception: " << e.what() << std::endl;
    }
#endif
}

void test_upload(Stream& stream) {
    // Device pointers
    Ptr1D<uchar3> test1D(1333);
    Ptr2D<uchar3> test2D(1333, 444);
    Ptr3D<uchar3> test3D(1333, 444, 22);
    Tensor<uchar3> testTensor(1333, 444, 22);

    // Must work
    test1D.upload(stream);
    test2D.upload(stream);
    test3D.upload(stream);
    testTensor.upload(stream);

    stream.sync();
}

void test_download(Stream& stream) {
    // Device pointers
    Ptr1D<uchar3> test1D(1333);
    Ptr2D<uchar3> test2D(1333, 444);
    Ptr3D<uchar3> test3D(1333, 444, 22);
    Tensor<uchar3> testTensor(1333, 444, 22);

    // Must work
    test1D.download(stream);
    test2D.download(stream);
    test3D.download(stream);
    testTensor.download(stream);

    stream.sync();
}

int launch() {

    Stream stream;

    PtrToTest test0(WIDTH, HEIGHT);
    setTo(make_<uchar3>(1, 2, 3), test0, stream);
    stream.sync();
    bool h_correct{ true };
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            const Bool3 boolVect = *PtrAccessor<ND::_2D>::cr_point(Point(x, y), test0.ptrPinned()) == make_<uchar3>(1, 2, 3);
            h_correct &= VectorAnd<Bool3>::exec(boolVect);
        }
    }

    PtrToTest test1(WIDTH, HEIGHT);

    auto test2 = PtrToTest(WIDTH, HEIGHT);

    PtrToTest test3;
    test3 = PtrToTest(WIDTH, HEIGHT);

    auto test4 = test_return_by_value();
    PtrToTest somePtr(WIDTH, HEIGHT);
    const PtrToTest& test5 = test_return_by_const_reference(somePtr);
    PtrToTest& test6 = test_return_by_reference(somePtr);

    bool result = test1.getRefCount() == 1;
    result &= test2.getRefCount() == 1;
    result &= test3.getRefCount() == 1;
    result &= test4.getRefCount() == 1;
    result &= test5.getRefCount() == 1;
    result &= test6.getRefCount() == 1;

    PtrToTest test7(WIDTH, HEIGHT);
    setTo(make_<uchar3>(3,6,10), test7, stream);
    test7.download(stream);
    stream.sync();

    bool h_correct2{ true };
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            const Bool3 boolVect = test7.at(Point(x, y)) == make_<uchar3>(3, 6, 10);
            h_correct2 &= VectorAnd<Bool3>::exec(boolVect);
        }
    }

    test_uploadTo(stream);
    test_downloadTo(stream);
    test_upload(stream);
    test_download(stream);

    return result && h_correct && h_correct2 ? 0 : -1;
}
