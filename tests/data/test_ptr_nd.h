/* Copyright 2024-2026 Oscar Amoros Huguet

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
#include <fused_kernel/core/utils/vector_utils.h>
#include <fused_kernel/algorithms/basic_ops/vector_ops.h>
#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/core/execution_model/stream.h>

#include <iostream>

using namespace fk;

using PtrToTest = Ptr2D<fk::uchar3>;
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
#if defined(__NVCC__) || defined(FK_HIP_BACKEND)
    // Device pointers
    Ptr1D<fk::uchar3> test1D(1333, 0, MemType::Device);
    Ptr2D<fk::uchar3> test2D(1333, 444, 0, MemType::Device);
    Ptr3D<fk::uchar3> test3D(1333, 444, 22, 1, 0, MemType::Device);
    Tensor<fk::uchar3> testTensor(1333, 444, 22, 1, MemType::Device);

    // Host Pinned Pointers
    Ptr1D<fk::uchar3> test1D_h(1333, 0, MemType::HostPinned);
    Ptr2D<fk::uchar3> test2D_h(1333, 444, 0, MemType::HostPinned);
    Ptr3D<fk::uchar3> test3D_h(1333, 444, 22, 1, 0, MemType::HostPinned);
    Tensor<fk::uchar3> testTensor_h(1333, 444, 22, 1, MemType::HostPinned);

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
#if defined(__NVCC__) || defined(FK_HIP_BACKEND)
    // Device pointers
    Ptr1D<fk::uchar3> test1D(1333, 0, MemType::Device);
    Ptr2D<fk::uchar3> test2D(1333, 444, 0, MemType::Device);
    Ptr3D<fk::uchar3> test3D(1333, 444, 22, 1, 0, MemType::Device);
    Tensor<fk::uchar3> testTensor(1333, 444, 22, 1, MemType::Device);

    // Host Pinned Pointers
    Ptr1D<fk::uchar3> test1D_h(1333, 0, MemType::HostPinned);
    Ptr2D<fk::uchar3> test2D_h(1333, 444, 0, MemType::HostPinned);
    Ptr3D<fk::uchar3> test3D_h(1333, 444, 22, 1, 0, MemType::HostPinned);
    Tensor<fk::uchar3> testTensor_h(1333, 444, 22, 1, MemType::HostPinned);

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
    Ptr1D<fk::uchar3> test1D(1333);
    Ptr2D<fk::uchar3> test2D(1333, 444);
    Ptr3D<fk::uchar3> test3D(1333, 444, 22);
    Tensor<fk::uchar3> testTensor(1333, 444, 22);

    // Must work
    test1D.upload(stream);
    test2D.upload(stream);
    test3D.upload(stream);
    testTensor.upload(stream);

    stream.sync();
}

void test_download(Stream& stream) {
    // Device pointers
    Ptr1D<fk::uchar3> test1D(1333);
    Ptr2D<fk::uchar3> test2D(1333, 444);
    Ptr3D<fk::uchar3> test3D(1333, 444, 22);
    Tensor<fk::uchar3> testTensor(1333, 444, 22);

    // Must work
    test1D.download(stream);
    test2D.download(stream);
    test3D.download(stream);
    testTensor.download(stream);

    stream.sync();
}

// Crops and planes share their parent's allocation: each adds exactly one reference, and whichever is destroyed last,
// the parent or not, frees it
bool test_shared_allocation_ref_counts() {
    bool correct = true;
    PtrToTest parent(WIDTH, HEIGHT);
    const PtrDims<ND::_2D> cropDims{ WIDTH / 2, HEIGHT / 2, parent.dims().pitch };
    {
        const PtrToTest cropped = parent.crop(Point{ 1, 2, 0 }, cropDims);
        correct &= parent.getRefCount() == 2 && cropped.getRefCount() == 2;
        {
            const PtrToTest secondCrop = parent.crop2D(Point{ 3, 4, 0 }, PtrDims<ND::_2D>{ 4, 4, cropDims.pitch });
            correct &= parent.getRefCount() == 3;
        }
        correct &= parent.getRefCount() == 2;
    }
    correct &= parent.getRefCount() == 1;

    // A crop that outlives its parent holds the last reference
    PtrToTest survivor;
    {
        PtrToTest shortLived(WIDTH, HEIGHT);
        survivor = shortLived.crop(Point{ 0, 0, 0 }, cropDims);
    }
    correct &= survivor.getRefCount() == 1;

    Ptr3D<fk::uchar3> volume(WIDTH, HEIGHT, 3);
    Tensor<fk::uchar3> tensor(WIDTH, HEIGHT, 3);
    {
        const Ptr2D<fk::uchar3> plane = volume.getPlane(1);
        const Ptr2D<fk::uchar3> tensorPlane = tensor.getPlane(2);
        correct &= volume.getRefCount() == 2 && tensor.getRefCount() == 2;
    }
    correct &= volume.getRefCount() == 1 && tensor.getRefCount() == 1;

    // Allocating again in a Ptr that shares an allocation releases only its own reference
    try {
        PtrToTest sharer = parent;
        sharer.Alloc(Size(WIDTH / 4, HEIGHT / 4));
        correct &= parent.getRefCount() == 1 && sharer.getRefCount() == 1;
        Tensor<fk::uchar3> tensorSharer = tensor;
        tensorSharer.allocTensor(WIDTH / 4, HEIGHT / 4, 2);
        correct &= tensor.getRefCount() == 1 && tensorSharer.getRefCount() == 1;
    } catch (const std::exception& e) {
        std::cout << "Allocating in a Ptr that shares an allocation threw: " << e.what() << std::endl;
        return false;
    }
    correct &= parent.getRefCount() == 1 && tensor.getRefCount() == 1;

    std::cout << "Shared allocations: " << (correct ? "reference counts OK" : "WRONG reference counts") << std::endl;
    return correct;
}

// setTo on memory the device can not write: the default stream may be a GPU stream, but the values
// must be written from the CPU
template <MemType MT>
bool test_setTo_host(Stream& stream) {
    PtrToTest hostPtr(WIDTH, HEIGHT, 0, MT);
    setTo(make_<fk::uchar3>(7, 8, 9), hostPtr, stream);
    stream.sync();
    bool correct{ true };
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            const bool allTrue = hostPtr.at(Point{x, y, 0}) == fk::uchar3{7, 8, 9};
            correct &= allTrue;
        }
    }
    return correct;
}

#if defined(__NVCC__) || defined(FK_HIP_BACKEND)
// Keeps the GPU busy for some milliseconds, so that the work enqueued after it runs after the host code that follows
__global__ void spinKernel(const long long int cycles) {
    const long long int start = clock64();
    while (clock64() - start < cycles) {}
}

void spin(Stream& stream) {
#if defined(__NVCC__)
    spinKernel<<<1, 1, 0, stream.getCUDAStream()>>>(50000000);
#else
    spinKernel<<<1, 1, 0, stream.getHIPStream()>>>(50000000);
#endif
}

bool allEqual(const PtrToTest& hostPtr, const fk::uchar3& value) {
    bool correct{ true };
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            const bool allTrue = hostPtr.at(Point{x, y, 0}) == value;
            correct &= allTrue;
        }
    }
    return correct;
}

// setTo with a GPU stream is ordered with the work enqueued on the stream before it, here copies that wait behind a
// busy GPU: a download into the memory that setTo fills does not overwrite the values, and an upload from it copies
// the values from before setTo
bool test_setTo_stream_order(Stream& stream) {
    const fk::uchar3 before{ 1, 2, 3 };
    const fk::uchar3 after{ 7, 8, 9 };
    PtrToTest device(WIDTH, HEIGHT, 0, MemType::Device);
    PtrToTest check(WIDTH, HEIGHT, 0, MemType::HostPinned);
    setTo(before, device, stream);

    PtrToTest pinned(WIDTH, HEIGHT, 0, MemType::HostPinned);
    spin(stream);
    device.downloadTo(pinned, stream);
    setTo(after, pinned, stream);
    stream.sync();
    const bool pinnedDownload = allEqual(pinned, after);

    PtrToTest host(WIDTH, HEIGHT, 0, MemType::Host);
    spin(stream);
    device.downloadTo(host, stream);
    setTo(after, host, stream);
    stream.sync();
    const bool hostDownload = allEqual(host, after);

    setTo(before, pinned, stream);
    spin(stream);
    pinned.uploadTo(device, stream);
    setTo(after, pinned, stream);
    device.downloadTo(check, stream);
    stream.sync();
    const bool pinnedUpload = allEqual(check, before) && allEqual(pinned, after);

    // Both copies of DeviceAndPinned memory
    PtrToTest both(WIDTH, HEIGHT, 0, MemType::DeviceAndPinned);
    setTo(before, both, stream);
    spin(stream);
    both.download(stream);
    setTo(after, both, stream);
    both.downloadTo(check, stream);
    stream.sync();
    const bool deviceAndPinnedDownload = allEqual(both, after) && allEqual(check, after);

    std::cout << "setTo in stream order: HostPinned download " << pinnedDownload << ", Host download " << hostDownload
              << ", HostPinned upload " << pinnedUpload << ", DeviceAndPinned download " << deviceAndPinnedDownload
              << std::endl;
    return pinnedDownload && hostDownload && pinnedUpload && deviceAndPinnedDownload;
}
#endif

int launch() {

    Stream stream;

    PtrToTest test0(WIDTH, HEIGHT);
    setTo(make_<fk::uchar3>(1, 2, 3), test0, stream);
    stream.sync();
    bool h_correct{ true };
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            const Bool3 boolVect = *PtrAccessor<ND::_2D>::cr_point(Point{x, y, 0}, test0.ptrPinned()) == make_<fk::uchar3>(1, 2, 3);
            const bool allTrue = boolVect;
            h_correct &= allTrue;
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
    setTo(make_<fk::uchar3>(3,6,10), test7, stream);
    test7.download(stream);
    stream.sync();

    bool h_correct2{ true };
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            const bool allTrue = test7.at(Point{x, y, 0}) == fk::uchar3{3, 6, 10};
            h_correct2 &= allTrue;
        }
    }

    bool h_correct3 = test_setTo_host<MemType::Host>(stream);
#if defined(__NVCC__) || defined(FK_HIP_BACKEND)
    h_correct3 &= test_setTo_host<MemType::HostPinned>(stream);
    h_correct3 &= test_setTo_stream_order(stream);
#endif

    test_uploadTo(stream);
    test_downloadTo(stream);
    test_upload(stream);
    test_download(stream);
    const bool shared = test_shared_allocation_ref_counts();

    return result && h_correct && h_correct2 && h_correct3 && shared ? 0 : -1;
}
