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
#include <stdexcept>
#include <vector>

namespace fk {

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
#if defined(__NVCC__)
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
#if defined(__NVCC__)
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

    Ptr3D<uchar3> volume(WIDTH, HEIGHT, 3);
    Tensor<uchar3> tensor(WIDTH, HEIGHT, 3);
    {
        const Ptr2D<uchar3> plane = volume.getPlane(1);
        const Ptr2D<uchar3> tensorPlane = tensor.getPlane(2);
        correct &= volume.getRefCount() == 2 && tensor.getRefCount() == 2;
    }
    correct &= volume.getRefCount() == 1 && tensor.getRefCount() == 1;

    // Allocating again in a Ptr that shares an allocation releases only its own reference
    try {
        PtrToTest sharer = parent;
        sharer.Alloc(Size(WIDTH / 4, HEIGHT / 4));
        correct &= parent.getRefCount() == 1 && sharer.getRefCount() == 1;
        Tensor<uchar3> tensorSharer = tensor;
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

// ---------------------------------------------------------------------------------------------------------------
// Ptr::crop: the caller gives the position and the extent, the strides always come from the parent
// ---------------------------------------------------------------------------------------------------------------

// A different value in every position (x, y, z), so that a view that walks its parent with other strides reads
// other pixels and is detected. Not a constant.
inline fk::uchar3 cropPixel(const int x, const int y, const int z = 0) {
    return fk::uchar3{ static_cast<uchar>((x * 7 + y * 13 + z * 29) % 251),
                       static_cast<uchar>((x * 5 + y * 3 + z * 11 + 1) % 251),
                       static_cast<uchar>((x * 3 + y * 17 + z * 7 + 2) % 251) };
}

inline bool cropSame(const fk::uchar3& a, const fk::uchar3& b) {
    const bool same = a == b;
    return same;
}

inline bool cropReport(const char* const name, const bool ok) {
    std::cout << "Ptr::crop " << name << ": " << (ok ? "OK" : "WRONG") << std::endl;
    return ok;
}

// The callable must throw std::runtime_error, and nothing else
template <typename F>
bool cropThrows(F&& f) {
    try {
        f();
    } catch (const std::runtime_error&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

inline bool cropSameDims(const RawPtr<ND::_2D, fk::uchar3>& a, const RawPtr<ND::_2D, fk::uchar3>& b) {
    return a.data == b.data && a.dims.width == b.dims.width && a.dims.height == b.dims.height &&
           a.dims.pitch == b.dims.pitch;
}

inline bool cropSameView(const PtrToTest& a, const PtrToTest& b) {
    return cropSameDims(a.ptr(), b.ptr()) && cropSameDims(a.ptrPinned(), b.ptrPinned()) &&
           a.getMemType() == b.getMemType() && a.getDeviceID() == b.getDeviceID();
}

void cropFillHost(PtrToTest& ptr) {
    for (int y = 0; y < static_cast<int>(ptr.dims().height); y++) {
        for (int x = 0; x < static_cast<int>(ptr.dims().width); x++) {
            ptr.at(Point{ x, y, 0 }) = cropPixel(x, y);
        }
    }
}

// The same view from a Rect and from a Point and a PtrDims, with and without the pitch, from a const parent: the
// same pointers, the same dims, the strides of the parent in the device buffer and in the pinned buffer
template <MemType MT>
bool test_crop_rect_and_point() {
    PtrToTest parent(WIDTH, HEIGHT, 0, MT);
    const PtrToTest& constParent = parent;
    const Rect rect(5, 3, 20, 10);
    const PtrToTest byRect = constParent.crop(rect);
    const PtrToTest byRect2D = constParent.crop2D(rect);
    const PtrToTest byPoint =
        constParent.crop2D(Point{ 5, 3, 0 }, PtrDims<ND::_2D>(20, 10, constParent.dims().pitch));
    const PtrToTest byPointNoPitch = constParent.crop2D(Point{ 5, 3, 0 }, PtrDims<ND::_2D>(20, 10));
    const PtrToTest byBase = constParent.crop(Point{ 5, 3, 0 }, PtrDims<ND::_2D>(20, 10, 0));

    bool correct = cropSameView(byRect, byRect2D) && cropSameView(byRect, byPoint) &&
                   cropSameView(byRect, byPointNoPitch) && cropSameView(byRect, byBase);
    // Field by field, against the parent
    const RawPtr<ND::_2D, fk::uchar3> device = byRect.ptr();
    const RawPtr<ND::_2D, fk::uchar3> pinned = byRect.ptrPinned();
    correct &= device.data == PtrAccessor<ND::_2D>::point(Point{ 5, 3, 0 }, parent.ptr());
    correct &= pinned.data == PtrAccessor<ND::_2D>::point(Point{ 5, 3, 0 }, parent.ptrPinned());
    correct &= device.dims.width == 20 && device.dims.height == 10 && device.dims.pitch == parent.ptr().dims.pitch;
    correct &= pinned.dims.width == 20 && pinned.dims.height == 10 &&
               pinned.dims.pitch == parent.ptrPinned().dims.pitch;
    correct &= byRect.getMemType() == MT && byRect.getDeviceID() == parent.getDeviceID();
    // The parent and the 5 views share the allocation
    correct &= parent.getRefCount() == 6;
    return correct;
}

// Non constant data, the pitch of the crop not given: the crop reads the pixels of its position in the parent
template <MemType MT>
bool test_crop_values_without_pitch() {
    PtrToTest parent(WIDTH, HEIGHT, 0, MT);
    cropFillHost(parent);
    const PtrToTest crop = parent.crop(Point{ 9, 5, 0 }, PtrDims<ND::_2D>(20, 30));
    bool correct = crop.dims().pitch == parent.dims().pitch && crop.ptrPinned().dims.pitch == parent.ptrPinned().dims.pitch;
    for (int y = 0; y < 30; y++) {
        for (int x = 0; x < 20; x++) {
            correct &= cropSame(crop.at(Point{ x, y, 0 }), cropPixel(x + 9, y + 5));
            // What a kernel does with the view: walks it with the strides of the view
            correct &= cropSame(*PtrAccessor<ND::_2D>::cr_point(Point{ x, y, 0 }, crop.ptrPinned()),
                                cropPixel(x + 9, y + 5));
        }
    }
    return correct;
}

// A crop of a crop is bounded by the extent of the crop, not by the one of the first parent, and its position
// adds to the one of the crop
template <MemType MT>
bool test_crop_of_a_crop() {
    PtrToTest parent(WIDTH, HEIGHT, 0, MT);
    cropFillHost(parent);
    const PtrToTest first = parent.crop(Rect(8, 4, 32, 40));
    const PtrToTest second = first.crop(Rect(4, 6, 10, 12));
    bool correct = cropSameDims(second.ptrPinned(), parent.crop(Rect(12, 10, 10, 12)).ptrPinned());
    correct &= second.dims().pitch == parent.dims().pitch;
    for (int y = 0; y < 12; y++) {
        for (int x = 0; x < 10; x++) {
            correct &= cropSame(second.at(Point{ x, y, 0 }), cropPixel(x + 12, y + 10));
        }
    }
    // Inside the first parent, but outside the first crop (x: 30 + 8 > 32, y: 35 + 8 > 40)
    correct &= cropThrows([&] { first.crop(Rect(30, 0, 8, 8)); });
    correct &= cropThrows([&] { first.crop(Rect(0, 35, 8, 8)); });
    // The last pixel of the first crop is fine
    correct &= !cropThrows([&] { first.crop(Rect(31, 39, 1, 1)); });
    correct &= parent.getRefCount() == 3 && first.getRefCount() == 3;
    return correct;
}

bool test_crop_1D() {
    Ptr1D<fk::uchar3> line(100, 0, MemType::Host);
    for (int i = 0; i < 100; i++) {
        line.at(i) = cropPixel(i, 0);
    }
    const Ptr1D<fk::uchar3> view = line.crop1D(10, 20);
    // In 1D the pitch is the size in bytes: the view gets the one of its own width
    bool correct = view.dims().width == 20 && view.dims().pitch == sizeof(fk::uchar3) * 20 &&
                   view.sizeInBytes() == sizeof(fk::uchar3) * 20 && view.getNumElements() == 20;
    for (int i = 0; i < 20; i++) {
        correct &= cropSame(view.at(i), cropPixel(i + 10, 0));
    }
    const Ptr1D<fk::uchar3> byPoint = line.crop1D(Point{ 10, 0, 0 }, PtrDims<ND::_1D>(20));
    correct &= byPoint.ptr().data == view.ptr().data && byPoint.dims().pitch == view.dims().pitch;
    // The size of the view, or the one of the parent, are accepted. Another pitch is not
    correct &= !cropThrows([&] { line.crop1D(Point{ 10, 0, 0 }, PtrDims<ND::_1D>(20, sizeof(fk::uchar3) * 20)); });
    correct &= !cropThrows([&] { line.crop1D(Point{ 10, 0, 0 }, PtrDims<ND::_1D>(20, line.dims().pitch)); });
    correct &= cropThrows([&] { line.crop1D(Point{ 10, 0, 0 }, PtrDims<ND::_1D>(20, 7)); });
    // The bounds
    correct &= cropThrows([&] { line.crop1D(90, 11); });
    correct &= !cropThrows([&] { line.crop1D(90, 10); });
    correct &= cropThrows([&] { line.crop1D(-1, 5); });
    correct &= cropThrows([&] { line.crop1D(0, 0); });
    correct &= cropThrows([&] { line.crop1D(Point{ 0, 1, 0 }, PtrDims<ND::_1D>(5)); });
    correct &= cropThrows([&] { line.crop1D(Point{ 0, 0, 1 }, PtrDims<ND::_1D>(5)); });
    // The crop of a crop
    correct &= cropThrows([&] { view.crop1D(15, 6); });
    correct &= !cropThrows([&] { view.crop1D(15, 5); });
    return correct;
}

bool test_crop_3D() {
    constexpr uint W = 16, H = 12, P = 5;
    // Ptr3D: the stride between planes is the one of the parent (pitch * height of the parent), not the one a
    // PtrDims<_3D> derives from the height of the view
    Ptr3D<fk::uchar3> volume(W, H, P, 1, 0, MemType::Host);
    for (int z = 0; z < static_cast<int>(P); z++) {
        for (int y = 0; y < static_cast<int>(H); y++) {
            for (int x = 0; x < static_cast<int>(W); x++) {
                volume.at(x, y, z) = cropPixel(x, y, z);
            }
        }
    }
    const Ptr3D<fk::uchar3> view = volume.crop3D(Point{ 2, 3, 1 }, PtrDims<ND::_3D>(6, 5, 3));
    bool correct = view.dims().width == 6 && view.dims().height == 5 && view.dims().planes == 3 &&
                   view.dims().color_planes == 1 && view.dims().pitch == volume.dims().pitch &&
                   view.dims().plane_pitch == volume.dims().plane_pitch &&
                   view.dims().plane_pitch != view.dims().pitch * view.dims().height;
    for (int z = 0; z < 3; z++) {
        for (int y = 0; y < 5; y++) {
            for (int x = 0; x < 6; x++) {
                correct &= cropSame(view.at(x, y, z), cropPixel(x + 2, y + 3, z + 1));
            }
        }
    }
    // The pitch of the parent is accepted, another is not
    correct &= !cropThrows([&] { volume.crop3D(Point{ 0, 0, 0 }, PtrDims<ND::_3D>(6, 5, 3, 1, volume.dims().pitch)); });
    correct &= cropThrows([&] { volume.crop3D(Point{ 0, 0, 0 }, PtrDims<ND::_3D>(6, 5, 3, 1, volume.dims().pitch + 3)); });
    correct &= cropThrows([&] { volume.crop3D(Point{ 0, 0, 3 }, PtrDims<ND::_3D>(6, 5, 3)); });
    correct &= !cropThrows([&] { volume.crop3D(Point{ 0, 0, 2 }, PtrDims<ND::_3D>(6, 5, 3)); });
    correct &= cropThrows([&] { volume.crop3D(Point{ 0, 0, -1 }, PtrDims<ND::_3D>(6, 5, 1)); });
    correct &= cropThrows([&] { volume.crop3D(Point{ 0, 0, 0 }, PtrDims<ND::_3D>(6, 5, 0)); });

    // Tensor with 2 color planes: the stride between planes is plane_pitch * color_planes
    Tensor<fk::uchar3> tensor(W, H, P, 2, MemType::Host);
    for (int z = 0; z < static_cast<int>(P); z++) {
        for (int y = 0; y < static_cast<int>(H); y++) {
            for (int x = 0; x < static_cast<int>(W); x++) {
                tensor.at(x, y, z) = cropPixel(x, y, z);
            }
        }
    }
    const Ptr<ND::_3D, fk::uchar3> tensorView = tensor.crop(Point{ 1, 2, 1 }, PtrDims<ND::_3D>(5, 4, 2, 2));
    correct &= tensorView.dims().pitch == tensor.dims().pitch && tensorView.dims().plane_pitch == tensor.dims().plane_pitch &&
               tensorView.dims().color_planes == 2;
    for (int z = 0; z < 2; z++) {
        for (int y = 0; y < 4; y++) {
            for (int x = 0; x < 5; x++) {
                correct &= cropSame(tensorView.at(x, y, z), cropPixel(x + 1, y + 2, z + 1));
            }
        }
    }
    // Several planes need the color planes of the parent. One plane can have fewer
    correct &= cropThrows([&] { tensor.crop(Point{ 0, 0, 0 }, PtrDims<ND::_3D>(5, 4, 2, 1)); });
    correct &= !cropThrows([&] { tensor.crop(Point{ 0, 0, 1 }, PtrDims<ND::_3D>(5, 4, 1, 1)); });
    correct &= cropThrows([&] { tensor.crop(Point{ 0, 0, 1 }, PtrDims<ND::_3D>(5, 4, 1, 3)); });
    correct &= cropThrows([&] { tensor.crop(Point{ 0, 0, 1 }, PtrDims<ND::_3D>(5, 4, 1, 0)); });
    return correct;
}

// The value of the color plane c of the pixel (x, y) of the plane z in the transposed 3D tests
inline fk::uchar3 cropPixelT3D(const int x, const int y, const int z, const int c) {
    return cropPixel(x, y, z + 10 * c);
}

bool test_crop_T3D() {
    constexpr uint W = 16, H = 12, P = 5, C = 3;
    PtrT3D<fk::uchar3> volume(W, H, P, C, MemType::Host);
    for (int c = 0; c < static_cast<int>(C); c++) {
        for (int z = 0; z < static_cast<int>(P); z++) {
            for (int y = 0; y < static_cast<int>(H); y++) {
                for (int x = 0; x < static_cast<int>(W); x++) {
                    *PtrAccessor<ND::T3D>::point(Point{ x, y, z }, volume.ptrPinned(), c) = cropPixelT3D(x, y, z, c);
                }
            }
        }
    }
    // Two of the three color planes
    const PtrT3D<fk::uchar3> view = volume.crop3D(Point{ 2, 1, 1 }, PtrDims<ND::T3D>(5, 4, 3, 2));
    const PtrDims<ND::T3D> parentDims = volume.ptrPinned().dims;
    const PtrDims<ND::T3D> viewDims = view.ptrPinned().dims;
    bool correct = viewDims.width == 5 && viewDims.height == 4 && viewDims.planes == 3 && viewDims.color_planes == 2 &&
                   viewDims.pitch == parentDims.pitch && viewDims.plane_pitch == parentDims.plane_pitch &&
                   viewDims.color_planes_pitch == parentDims.color_planes_pitch && parentDims.pitch != 0 &&
                   parentDims.plane_pitch != 0 && parentDims.color_planes_pitch != 0;
    for (int c = 0; c < 2; c++) {
        for (int z = 0; z < 3; z++) {
            for (int y = 0; y < 4; y++) {
                for (int x = 0; x < 5; x++) {
                    correct &= cropSame(*PtrAccessor<ND::T3D>::cr_point(Point{ x, y, z }, view.ptrPinned(), c),
                                        cropPixelT3D(x + 2, y + 1, z + 1, c));
                }
            }
        }
    }
    // The strides of the caller can only be the ones of the parent
    PtrDims<ND::T3D> given(5, 4, 3, 2);
    given.pitch = parentDims.pitch;
    given.plane_pitch = parentDims.plane_pitch;
    given.color_planes_pitch = parentDims.color_planes_pitch;
    correct &= !cropThrows([&] { volume.crop3D(Point{ 0, 0, 0 }, given); });
    given.color_planes_pitch += 1;
    correct &= cropThrows([&] { volume.crop3D(Point{ 0, 0, 0 }, given); });
    given.color_planes_pitch = 0;
    given.plane_pitch += 1;
    correct &= cropThrows([&] { volume.crop3D(Point{ 0, 0, 0 }, given); });
    given.plane_pitch = 0;
    given.pitch += 1;
    correct &= cropThrows([&] { volume.crop3D(Point{ 0, 0, 0 }, given); });
    // The bounds and the color planes
    correct &= cropThrows([&] { volume.crop3D(Point{ 0, 0, 0 }, PtrDims<ND::T3D>(5, 4, 3, 4)); });
    correct &= cropThrows([&] { volume.crop3D(Point{ 0, 0, 0 }, PtrDims<ND::T3D>(5, 4, 3, 0)); });
    correct &= cropThrows([&] { volume.crop3D(Point{ 0, 0, 3 }, PtrDims<ND::T3D>(5, 4, 3, 1)); });
    correct &= !cropThrows([&] { volume.crop3D(Point{ 11, 8, 2 }, PtrDims<ND::T3D>(5, 4, 3, 3)); });
    return correct;
}

// Every case that throws, in every build. A throw leaves the reference count of the parent as it was.
bool test_crop_throws() {
    PtrToTest parent(WIDTH, HEIGHT, 0, MemType::Host);
    const PtrDims<ND::_2D> ok(10, 10);
    bool correct = !cropThrows([&] { parent.crop(Point{ 0, 0, 0 }, ok); });
    // Outside the parent
    correct &= cropThrows([&] { parent.crop(Point{ WIDTH - 9, 0, 0 }, ok); });
    correct &= cropThrows([&] { parent.crop(Point{ 0, HEIGHT - 9, 0 }, ok); });
    correct &= !cropThrows([&] { parent.crop(Point{ WIDTH - 10, HEIGHT - 10, 0 }, ok); });
    correct &= cropThrows([&] { parent.crop(Point{ 0, 0, 0 }, PtrDims<ND::_2D>(WIDTH + 1, 10)); });
    correct &= cropThrows([&] { parent.crop(Point{ 0, 0, 0 }, PtrDims<ND::_2D>(10, HEIGHT + 1)); });
    // A negative position, an empty view, a coordinate the pointer does not have
    correct &= cropThrows([&] { parent.crop(Point{ -1, 0, 0 }, ok); });
    correct &= cropThrows([&] { parent.crop(Point{ 0, -1, 0 }, ok); });
    correct &= cropThrows([&] { parent.crop(Point{ 0, 0, 0 }, PtrDims<ND::_2D>(0, 10)); });
    correct &= cropThrows([&] { parent.crop(Point{ 0, 0, 0 }, PtrDims<ND::_2D>(10, 0)); });
    correct &= cropThrows([&] { parent.crop(Point{ 0, 0, 1 }, ok); });
    // A pitch that is not the one of the parent. 0 and the one of the parent are fine
    correct &= cropThrows([&] { parent.crop(Point{ 0, 0, 0 }, PtrDims<ND::_2D>(10, 10, parent.dims().pitch + 64)); });
    correct &= cropThrows([&] { parent.crop(Point{ 0, 0, 0 }, PtrDims<ND::_2D>(10, 10, 1)); });
    correct &= !cropThrows([&] { parent.crop(Point{ 0, 0, 0 }, PtrDims<ND::_2D>(10, 10, parent.dims().pitch)); });
    correct &= !cropThrows([&] { parent.crop(Point{ 0, 0, 0 }, PtrDims<ND::_2D>(10, 10, 0)); });
    // The same with a Rect
    correct &= cropThrows([&] { parent.crop(Rect(WIDTH - 9, 0, 10, 10)); });
    correct &= cropThrows([&] { parent.crop(Rect(0, 0, 0, 10)); });
    correct &= cropThrows([&] { parent.crop(Rect(0, 0, 10, 0)); });
    correct &= cropThrows([&] { parent.crop(Rect(0, 0, -5, 10)); });
    correct &= cropThrows([&] { parent.crop(Rect(0, 0, 10, -5)); });
    correct &= cropThrows([&] { parent.crop2D(Rect(0, HEIGHT, 10, 1)); });
    correct &= cropThrows([&] { parent.crop(Rect(0xFFFFFFFFu, 0, 1, 1)); });
    correct &= !cropThrows([&] { parent.crop2D(Rect(0, 0, WIDTH, HEIGHT)); });
    // A parent without pitch (a pointer built from a pointer and a size, with pitch 0) has no stride to give
    std::vector<fk::uchar3> storage(16 * 16);
    const Ptr2D<fk::uchar3> noPitch(storage.data(), 16, 16, 0, MemType::Host);
    correct &= cropThrows([&] { noPitch.crop(Point{ 0, 0, 0 }, PtrDims<ND::_2D>(4, 4)); });
    const Ptr2D<fk::uchar3> withPitch(storage.data(), 16, 16, 16 * sizeof(fk::uchar3), MemType::Host);
    correct &= !cropThrows([&] { withPitch.crop(Point{ 4, 4, 0 }, PtrDims<ND::_2D>(4, 4)); });
    const PtrT3D<fk::uchar3> noStrides(storage.data(), PtrDims<ND::T3D>(4, 4, 2, 1), MemType::Host);
    correct &= cropThrows([&] { noStrides.crop3D(Point{ 0, 0, 0 }, PtrDims<ND::T3D>(2, 2, 1, 1)); });
    correct &= parent.getRefCount() == 1;
    return correct;
}

#if defined(__NVCC__) || defined(__HIPCC__)
// DeviceAndPinned: the device buffer and the pinned buffer have different pitches (the device one comes from the
// pitched allocation, the pinned one from the host alignment). The crop walks each buffer with its own pitch, so
// downloading and uploading a crop moves exactly the pixels of the crop.
bool test_crop_device_and_pinned(Stream& stream) {
    PtrToTest parent(WIDTH, HEIGHT, 0, MemType::DeviceAndPinned);
    const uint devicePitch = parent.ptr().dims.pitch;
    const uint hostPitch = parent.ptrPinned().dims.pitch;
    std::cout << "Ptr::crop DeviceAndPinned: device pitch " << devicePitch << ", pinned pitch " << hostPitch << std::endl;
    const Rect rect(5, 3, 20, 10);
    PtrToTest view = parent.crop2D(rect);
    bool correct = view.ptr().dims.pitch == devicePitch && view.ptrPinned().dims.pitch == hostPitch &&
                   view.ptrPinned().data == PtrAccessor<ND::_2D>::point(Point{ 5, 3, 0 }, parent.ptrPinned());

    const fk::uchar3 sentinel{ 250, 251, 252 };
    auto inside = [&](const int x, const int y) {
        return x >= 5 && x < 25 && y >= 3 && y < 13;
    };
    // The device buffer holds the pattern, the pinned buffer the sentinel: downloading the crop must replace the
    // pixels of the crop in the pinned buffer, and no other
    cropFillHost(parent);
    parent.upload(stream);
    stream.sync();
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            parent.at(Point{ x, y, 0 }) = sentinel;
        }
    }
    view.download(stream);
    stream.sync();
    bool downloaded = true;
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            downloaded &= cropSame(parent.at(Point{ x, y, 0 }), inside(x, y) ? cropPixel(x, y) : sentinel);
        }
    }
    // The pinned buffer holds a new pattern in the crop: uploading the crop must replace the pixels of the crop in
    // the device buffer, and no other
    const fk::uchar3 marker{ 1, 2, 3 };
    for (int y = 0; y < 10; y++) {
        for (int x = 0; x < 20; x++) {
            view.at(Point{ x, y, 0 }) = marker;
        }
    }
    view.upload(stream);
    stream.sync();
    parent.download(stream);
    stream.sync();
    bool uploaded = true;
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            uploaded &= cropSame(parent.at(Point{ x, y, 0 }), inside(x, y) ? marker : cropPixel(x, y));
        }
    }
    std::cout << "Ptr::crop DeviceAndPinned: pitch of the views " << (correct ? "OK" : "WRONG") << ", download of a crop "
              << (downloaded ? "OK" : "WRONG") << ", upload of a crop " << (uploaded ? "OK" : "WRONG") << std::endl;
    return correct && downloaded && uploaded;
}
#endif

bool test_crop(Stream& stream) {
    bool correct = cropReport("Rect and Point (Host)", test_crop_rect_and_point<MemType::Host>());
    correct &= cropReport("values without pitch (Host)", test_crop_values_without_pitch<MemType::Host>());
    correct &= cropReport("crop of a crop (Host)", test_crop_of_a_crop<MemType::Host>());
    correct &= cropReport("1D", test_crop_1D());
    correct &= cropReport("3D", test_crop_3D());
    correct &= cropReport("transposed 3D", test_crop_T3D());
    correct &= cropReport("throws", test_crop_throws());
#if defined(__NVCC__) || defined(__HIPCC__)
    correct &= cropReport("Rect and Point (Device)", test_crop_rect_and_point<MemType::Device>());
    correct &= cropReport("Rect and Point (HostPinned)", test_crop_rect_and_point<MemType::HostPinned>());
    correct &= cropReport("Rect and Point (DeviceAndPinned)", test_crop_rect_and_point<MemType::DeviceAndPinned>());
    correct &= cropReport("values without pitch (HostPinned)", test_crop_values_without_pitch<MemType::HostPinned>());
    correct &= cropReport("values without pitch (DeviceAndPinned)", test_crop_values_without_pitch<MemType::DeviceAndPinned>());
    correct &= cropReport("crop of a crop (DeviceAndPinned)", test_crop_of_a_crop<MemType::DeviceAndPinned>());
    correct &= cropReport("DeviceAndPinned transfers", test_crop_device_and_pinned(stream));
#endif
    return correct;
}

int launch_impl() {
    using namespace fk;
    Stream stream;

    PtrToTest test0(WIDTH, HEIGHT);
    setTo(make_<uchar3>(1, 2, 3), test0, stream);
    stream.sync();
    bool h_correct{ true };
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            const bool boolVect = *PtrAccessor<ND::_2D>::cr_point(Point{x, y, 0}, test0.ptrPinned()) == make_<uchar3>(1, 2, 3);
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
    setTo(make_<uchar3>(3,6,10), test7, stream);
    test7.download(stream);
    stream.sync();

    bool h_correct2{ true };
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            const bool allTrue = test7.at(Point{x, y, 0}) == uchar3{3, 6, 10};
            h_correct2 &= allTrue;
        }
    }

    test_uploadTo(stream);
    test_downloadTo(stream);
    test_upload(stream);
    test_download(stream);
    const bool shared = test_shared_allocation_ref_counts();
    const bool crops = test_crop(stream);

    return result && h_correct && h_correct2 && shared && crops ? 0 : -1;
}

} // namespace fk

int launch() {
    return fk::launch_impl();
}
