/* Copyright 2023-2026 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_PTR_ND_H
#define FK_PTR_ND_H

#include <fused_kernel/core/utils/utils.h>
#include <fused_kernel/core/data/rawptr.h>
#include <fused_kernel/core/data/rect.h>
#include <fused_kernel/core/data/size.h>
#if !defined(NVRTC_COMPILER)
#include <fused_kernel/core/execution_model/stream.h>
#endif
#include <fused_kernel/core/utils/vector_utils.h>

#include <atomic>
#include <cstdlib>
#include <stdexcept>
#include <string>
#if defined(_MSC_VER)
#include <malloc.h>
#endif

namespace fk {
	enum class MemType { Device, Host, HostPinned, DeviceAndPinned };

    // Host allocations are aligned, and host 2D rows are padded, to this number of bytes,
    // so that every row can be read and written with the thread fusion vector types,
    // like cudaMallocPitch does for device memory. 64 covers the biggest thread fusion
    // vector types (32 bytes) and matches the cache line size.
    constexpr size_t hostAlignment = 64;

    inline constexpr size_t alignUpToHostAlignment(const size_t sizeInBytes) {
        return ((sizeInBytes + hostAlignment - 1) / hostAlignment) * hostAlignment;
    }

    inline void* hostAlignedMalloc(const size_t sizeInBytes) {
#if defined(_MSC_VER)
        return _aligned_malloc(alignUpToHostAlignment(sizeInBytes), hostAlignment);
#else
        // std::aligned_alloc requires the size to be a multiple of the alignment
        return std::aligned_alloc(hostAlignment, alignUpToHostAlignment(sizeInBytes));
#endif
    }

    inline void hostAlignedFree(void* ptr) {
#if defined(_MSC_VER)
        _aligned_free(ptr);
#else
        std::free(ptr);
#endif
    }
#if defined(__NVCC__) || defined(FK_HIP_BACKEND)
    constexpr MemType defaultMemType = MemType::DeviceAndPinned;
#else
    constexpr MemType defaultMemType = MemType::Host;
#endif

#if defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)
    namespace cpu_multithread {
        // ParArch::CPU_MULTITHREAD executions read and write ptr() from the CPU, so it must be host memory (Host or
        // HostPinned). With Device and DeviceAndPinned memory (defaultMemType in CUDA and HIP builds) ptr() is a device
        // pointer: the CPU_MULTITHREAD helpers that take a Ptr (upload, download, setTo) throw std::runtime_error.
        inline void requireHostMemory(const MemType memType, const char* const what) {
            if (memType != MemType::Host && memType != MemType::HostPinned) {
                throw std::runtime_error(std::string(what) +
                                         ": ParArch::CPU_MULTITHREAD needs Host or HostPinned memory (ptr() of Device "
                                         "and DeviceAndPinned memory is a device pointer)");
            }
        }
    } // namespace cpu_multithread
#endif // defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)

    template <enum ND D, typename T>
    struct PtrImpl;

    template <typename T>
    struct PtrImpl<ND::_1D, T> {
        FK_HOST_FUSE size_t sizeInBytes(const PtrDims<ND::_1D>& dims) {
            return dims.pitch;
        }
        FK_HOST_FUSE uint getNumElements(const PtrDims<ND::_1D>& dims) {
            return dims.width;
        }
        FK_HOST_FUSE void d_malloc(RawPtr<ND::_1D, T>& ptr_a) {
            if (ptr_a.dims.pitch == 0) {
                ptr_a.dims.pitch = sizeof(T) * ptr_a.dims.width;
            }
#if defined(FK_HIP_BACKEND)
            gpuErrchk(hipMalloc(&ptr_a.data, ptr_a.dims.pitch));
#else
            gpuErrchk(cudaMalloc(&ptr_a.data, ptr_a.dims.pitch));
#endif
        }
        FK_HOST_FUSE void h_malloc_init(PtrDims<ND::_1D>& dims) {
            if (dims.pitch == 0) {
                dims.pitch = sizeof(T) * dims.width;
            }
        }
    };

    template <typename T>
    struct PtrImpl<ND::_2D, T> {
        FK_HOST_FUSE size_t sizeInBytes(const PtrDims<ND::_2D>& dims) {
            return dims.pitch * dims.height;
        }
        FK_HOST_FUSE uint getNumElements(const PtrDims<ND::_2D>& dims) {
            return dims.width * dims.height;
        }
        FK_HOST_FUSE void d_malloc(RawPtr<ND::_2D, T>& ptr_a) {
            if (ptr_a.dims.pitch == 0) {
                size_t pitch = 0;
#if defined(FK_HIP_BACKEND)
                gpuErrchk(hipMallocPitch(&ptr_a.data, &pitch, sizeof(T) * ptr_a.dims.width, ptr_a.dims.height));
#else
                gpuErrchk(cudaMallocPitch(&ptr_a.data, &pitch, sizeof(T) * ptr_a.dims.width, ptr_a.dims.height));
#endif
                ptr_a.dims.pitch = static_cast<int>(pitch);
            } else {
#if defined(FK_HIP_BACKEND)
                gpuErrchk(hipMalloc(&ptr_a.data, PtrImpl<ND::_2D, T>::sizeInBytes(ptr_a.dims)));
#else
                gpuErrchk(cudaMalloc(&ptr_a.data, PtrImpl<ND::_2D, T>::sizeInBytes(ptr_a.dims)));
#endif
            }
        }
        FK_HOST_FUSE void h_malloc_init(PtrDims<ND::_2D>& dims) {
            if (dims.pitch == 0) {
                dims.pitch = static_cast<uint>(alignUpToHostAlignment(sizeof(T) * dims.width));
            }
        }
    };

    template <typename T>
    struct PtrImpl<ND::_3D, T> {
        FK_HOST_FUSE size_t sizeInBytes(const PtrDims<ND::_3D>& dims) {
            return dims.pitch * dims.height * dims.planes * dims.color_planes;
        }
        FK_HOST_FUSE uint getNumElements(const PtrDims<ND::_3D>& dims) {
            return dims.width * dims.height * dims.planes * dims.color_planes;
        }
        FK_HOST_FUSE void d_malloc(RawPtr<ND::_3D, T>& ptr_a) {
            if (ptr_a.dims.pitch == 0) {
                ptr_a.dims.pitch = sizeof(T) * ptr_a.dims.width;
            }
#if defined(FK_HIP_BACKEND)
            gpuErrchk(hipMalloc(&ptr_a.data, PtrImpl<ND::_3D, T>::sizeInBytes(ptr_a.dims)));
#else
            gpuErrchk(cudaMalloc(&ptr_a.data, PtrImpl<ND::_3D, T>::sizeInBytes(ptr_a.dims)));
#endif
            ptr_a.dims.plane_pitch = ptr_a.dims.pitch * ptr_a.dims.height;
        }
        FK_HOST_FUSE void h_malloc_init(PtrDims<ND::_3D>& dims) {
            if (dims.pitch == 0) {
                dims.pitch = sizeof(T) * dims.width;
            }
            dims.plane_pitch = dims.pitch * dims.height;
        }
    };

    template <typename T>
    struct PtrImpl<ND::T3D, T> {
        FK_HOST_FUSE size_t sizeInBytes(const PtrDims<ND::T3D>& dims) {
            return dims.color_planes_pitch * dims.color_planes;
        }
        FK_HOST_FUSE uint getNumElements(const PtrDims<ND::T3D>& dims) {
            return dims.width * dims.height * dims.planes * dims.color_planes;
        }
        FK_HOST_FUSE void d_malloc(RawPtr<ND::T3D, T>& ptr_a) {
            PtrImpl<ND::T3D, T>::h_malloc_init(ptr_a.dims);
#if defined(FK_HIP_BACKEND)
            gpuErrchk(hipMalloc(&ptr_a.data, PtrImpl<ND::T3D, T>::sizeInBytes(ptr_a.dims)));
#else
            gpuErrchk(cudaMalloc(&ptr_a.data, PtrImpl<ND::T3D, T>::sizeInBytes(ptr_a.dims)));
#endif
        }
        FK_HOST_FUSE void h_malloc_init(PtrDims<ND::T3D>& dims) {
            if (dims.pitch == 0) {
                dims.pitch = sizeof(T) * dims.width;
            }
            dims.plane_pitch = dims.pitch * dims.height;
            dims.color_planes_pitch = dims.plane_pitch * dims.planes;
        }
    };

    // The rules of Ptr::crop. The caller chooses the position and the extent of the view, never its strides: a view
    // can only walk the memory of its parent with the strides of the parent, so check() validates the request and
    // viewDims() builds the dims of the view with the strides of the buffer it is a view of. A DeviceAndPinned Ptr
    // has two buffers, with their own strides (the device pitch comes from cudaMallocPitch/hipMallocPitch, the
    // pinned pitch from the host alignment), and each view takes the strides of its own buffer.
    namespace ptr_crop {
        [[noreturn]] inline void fail(const std::string& what) {
            throw std::runtime_error("Ptr::crop: " + what);
        }

        // One axis of the view must be non negative, not empty, and inside the parent
        inline void checkAxis(const char* const axis, const int position, const uint extent, const uint parentExtent) {
            if (position < 0) {
                fail(std::string("the position in ") + axis + " is negative (" + std::to_string(position) + ")");
            }
            if (extent == 0) {
                fail(std::string("the extent in ") + axis + " is zero");
            }
            if (static_cast<size_t>(position) + extent > parentExtent) {
                fail(std::string("the view is outside the parent in ") + axis + ": position " +
                     std::to_string(position) + " + extent " + std::to_string(extent) + " > " +
                     std::to_string(parentExtent));
            }
        }

        // A coordinate the pointer does not have (y of a 1D pointer, z of a 2D pointer) can only be 0
        inline void checkNoAxis(const char* const axis, const int position) {
            if (position != 0) {
                fail(std::string("the pointer has no ") + axis + " dimension, but the position in it is " +
                     std::to_string(position));
            }
        }

        // A stride given by the caller can only repeat the one of the parent (of its device buffer, the one dims()
        // returns, if it has two). Zero means "not given".
        inline void checkStride(const char* const stride, const uint given, const uint parent) {
            if (given != 0 && given != parent) {
                fail(std::string("the ") + stride + " given (" + std::to_string(given) + ") is not the " + stride +
                     " of the parent (" + std::to_string(parent) + "). A crop always uses the strides of its parent");
            }
        }

        inline uint checkedStride(const char* const stride, const uint parent) {
            if (parent == 0) {
                fail(std::string("the parent has no ") + stride + " (it is 0)");
            }
            return parent;
        }
    } // namespace ptr_crop

    template <enum ND D, typename T>
    struct PtrCrop;

    template <typename T>
    struct PtrCrop<ND::_1D, T> {
        // In a 1D pointer, pitch is the size of the allocation in bytes: the view gets the one of its own width
        static inline void check(const PtrDims<ND::_1D>& parent, const Point p, const PtrDims<ND::_1D>& requested) {
            ptr_crop::checkAxis("x", p.x, requested.width, parent.width);
            ptr_crop::checkNoAxis("y", p.y);
            ptr_crop::checkNoAxis("z", p.z);
            if (requested.pitch != 0 && requested.pitch != sizeof(T) * requested.width) {
                ptr_crop::checkStride("pitch", requested.pitch, parent.pitch);
            }
        }
        static inline PtrDims<ND::_1D> viewDims(const PtrDims<ND::_1D>&, const PtrDims<ND::_1D>& requested) {
            return PtrDims<ND::_1D>(requested.width, static_cast<uint>(sizeof(T)) * requested.width);
        }
    };

    template <typename T>
    struct PtrCrop<ND::_2D, T> {
        static inline void check(const PtrDims<ND::_2D>& parent, const Point p, const PtrDims<ND::_2D>& requested) {
            ptr_crop::checkAxis("x", p.x, requested.width, parent.width);
            ptr_crop::checkAxis("y", p.y, requested.height, parent.height);
            ptr_crop::checkNoAxis("z", p.z);
            ptr_crop::checkedStride("pitch", parent.pitch);
            ptr_crop::checkStride("pitch", requested.pitch, parent.pitch);
        }
        static inline PtrDims<ND::_2D> viewDims(const PtrDims<ND::_2D>& buffer, const PtrDims<ND::_2D>& requested) {
            return PtrDims<ND::_2D>(requested.width, requested.height, ptr_crop::checkedStride("pitch", buffer.pitch));
        }
    };

    template <typename T>
    struct PtrCrop<ND::_3D, T> {
        // The stride between planes is plane_pitch * color_planes, so a view with several planes has the color
        // planes of its parent. A view of one plane can have fewer.
        static inline void check(const PtrDims<ND::_3D>& parent, const Point p, const PtrDims<ND::_3D>& requested) {
            ptr_crop::checkAxis("x", p.x, requested.width, parent.width);
            ptr_crop::checkAxis("y", p.y, requested.height, parent.height);
            ptr_crop::checkAxis("z", p.z, requested.planes, parent.planes);
            if (requested.color_planes == 0 || requested.color_planes > parent.color_planes) {
                ptr_crop::fail("the color planes of the view (" + std::to_string(requested.color_planes) +
                               ") are zero or more than the ones of the parent (" +
                               std::to_string(parent.color_planes) + ")");
            }
            if (requested.planes > 1 && requested.color_planes != parent.color_planes) {
                ptr_crop::fail("a view of several planes needs the color planes of the parent (" +
                               std::to_string(parent.color_planes) + "), not " +
                               std::to_string(requested.color_planes));
            }
            ptr_crop::checkedStride("pitch", parent.pitch);
            ptr_crop::checkedStride("plane_pitch", parent.plane_pitch);
            // The PtrDims<ND::_3D> constructor derives plane_pitch from the height of the view, which is not the
            // plane_pitch of the parent, so only pitch can be compared
            ptr_crop::checkStride("pitch", requested.pitch, parent.pitch);
        }
        static inline PtrDims<ND::_3D> viewDims(const PtrDims<ND::_3D>& buffer, const PtrDims<ND::_3D>& requested) {
            PtrDims<ND::_3D> dims(requested.width, requested.height, requested.planes, requested.color_planes,
                                  ptr_crop::checkedStride("pitch", buffer.pitch));
            dims.plane_pitch = ptr_crop::checkedStride("plane_pitch", buffer.plane_pitch);
            return dims;
        }
    };

    template <typename T>
    struct PtrCrop<ND::T3D, T> {
        // The color planes are addressed from the first one with color_planes_pitch, so a view can have fewer
        static inline void check(const PtrDims<ND::T3D>& parent, const Point p, const PtrDims<ND::T3D>& requested) {
            ptr_crop::checkAxis("x", p.x, requested.width, parent.width);
            ptr_crop::checkAxis("y", p.y, requested.height, parent.height);
            ptr_crop::checkAxis("z", p.z, requested.planes, parent.planes);
            if (requested.color_planes == 0 || requested.color_planes > parent.color_planes) {
                ptr_crop::fail("the color planes of the view (" + std::to_string(requested.color_planes) +
                               ") are zero or more than the ones of the parent (" +
                               std::to_string(parent.color_planes) + ")");
            }
            ptr_crop::checkedStride("pitch", parent.pitch);
            ptr_crop::checkedStride("plane_pitch", parent.plane_pitch);
            ptr_crop::checkedStride("color_planes_pitch", parent.color_planes_pitch);
            ptr_crop::checkStride("pitch", requested.pitch, parent.pitch);
            ptr_crop::checkStride("plane_pitch", requested.plane_pitch, parent.plane_pitch);
            ptr_crop::checkStride("color_planes_pitch", requested.color_planes_pitch, parent.color_planes_pitch);
        }
        static inline PtrDims<ND::T3D> viewDims(const PtrDims<ND::T3D>& buffer, const PtrDims<ND::T3D>& requested) {
            PtrDims<ND::T3D> dims(requested.width, requested.height, requested.planes, requested.color_planes);
            dims.pitch = ptr_crop::checkedStride("pitch", buffer.pitch);
            dims.plane_pitch = ptr_crop::checkedStride("plane_pitch", buffer.plane_pitch);
            dims.color_planes_pitch = ptr_crop::checkedStride("color_planes_pitch", buffer.color_planes_pitch);
            return dims;
        }
    };

    struct RefPtr {
        void* ptr{nullptr};
        void* pinnedPtr{nullptr};
        std::atomic<int> cnt{1};
    };

    template <enum ND D, typename T>
    class Ptr {
    public:
        using Type = T;
        using At = PtrAccessor<D>;
        static constexpr ND nd = D;
    protected:
        RefPtr* ref{ nullptr };
        RawPtr<D, T> ptr_a{};
        RawPtr<D, T> ptr_pinned{};
        MemType type{defaultMemType};
        int deviceID{0};

        inline constexpr Ptr(const RawPtr<D, T>& ptr_a_, RefPtr* ref_, const MemType& type_, const int& devID) :
            ref(ref_), ptr_a(ptr_a_), ptr_pinned(ptr_a_), type(type_), deviceID(devID) {
            if (ref) {
                ref->cnt.fetch_add(1);
            }
        }

        inline constexpr void allocDevice() {
            #if defined(__NVCC__) || defined(FK_HIP_BACKEND)
            int currentDevice = 0;
#if defined(FK_HIP_BACKEND)
            gpuErrchk(hipGetDevice(&currentDevice));
#else
            gpuErrchk(cudaGetDevice(&currentDevice));
#endif
#if defined(FK_HIP_BACKEND)
            gpuErrchk(hipSetDevice(deviceID));
#else
            gpuErrchk(cudaSetDevice(deviceID));
#endif
            PtrImpl<D, T>::d_malloc(ptr_a);
            if (currentDevice != deviceID) {
#if defined(FK_HIP_BACKEND)
                gpuErrchk(hipSetDevice(currentDevice));
#else
                gpuErrchk(cudaSetDevice(currentDevice));
#endif
            }
            #else
            throw std::runtime_error("Device allocation not supported in non-CUDA compilation.");
            #endif
        }

        inline constexpr void allocHost() {
            PtrImpl<D, T>::h_malloc_init(ptr_a.dims);
            ptr_a.data = (T*)hostAlignedMalloc(PtrImpl<D, T>::sizeInBytes(ptr_a.dims));
        }

        inline constexpr void allocHostPinned() {
            #if defined(__NVCC__) || defined(FK_HIP_BACKEND)
            int currentDevice = 0;
#if defined(FK_HIP_BACKEND)
            gpuErrchk(hipGetDevice(&currentDevice));
#else
            gpuErrchk(cudaGetDevice(&currentDevice));
#endif
#if defined(FK_HIP_BACKEND)
            gpuErrchk(hipSetDevice(deviceID));
#else
            gpuErrchk(cudaSetDevice(deviceID));
#endif
            PtrImpl<D, T>::h_malloc_init(ptr_a.dims);
#if defined(FK_HIP_BACKEND)
            gpuErrchk(hipHostMalloc(&ptr_a.data, PtrImpl<D, T>::sizeInBytes(ptr_a.dims)));
#else
            gpuErrchk(cudaMallocHost(&ptr_a.data, PtrImpl<D, T>::sizeInBytes(ptr_a.dims)));
#endif
            if (currentDevice != deviceID) {
#if defined(FK_HIP_BACKEND)
                gpuErrchk(hipSetDevice(currentDevice));
#else
                gpuErrchk(cudaSetDevice(currentDevice));
#endif
            }
            #else
            throw std::runtime_error("Host pinned allocation not supported in non-CUDA compilation.");
            #endif
        }

        inline constexpr void allocDeviceAndPinned() {
            #if defined(__NVCC__) || defined(FK_HIP_BACKEND)
            int currentDevice = 0;
#if defined(FK_HIP_BACKEND)
            gpuErrchk(hipGetDevice(&currentDevice));
#else
            gpuErrchk(cudaGetDevice(&currentDevice));
#endif
#if defined(FK_HIP_BACKEND)
            gpuErrchk(hipSetDevice(deviceID));
#else
            gpuErrchk(cudaSetDevice(deviceID));
#endif
            PtrImpl<D, T>::d_malloc(ptr_a);
            PtrImpl<D, T>::h_malloc_init(ptr_pinned.dims);
#if defined(FK_HIP_BACKEND)
            gpuErrchk(hipHostMalloc(&ptr_pinned.data, PtrImpl<D, T>::sizeInBytes(ptr_pinned.dims)));
#else
            gpuErrchk(cudaMallocHost(&ptr_pinned.data, PtrImpl<D, T>::sizeInBytes(ptr_pinned.dims)));
#endif
            if (currentDevice != deviceID) {
#if defined(FK_HIP_BACKEND)
                gpuErrchk(hipSetDevice(currentDevice));
#else
                gpuErrchk(cudaSetDevice(currentDevice));
#endif
            }
            #else
            throw std::runtime_error("Host pinned and Device allocations not supported in non-CUDA compilation.");
            #endif
        }

        inline constexpr void freePtr() {
            if (ref && ref->cnt.load() < 1) {
                throw std::runtime_error("Reference count is less than 1, cannot free memory.");
            }
            if (ref && ref->cnt.fetch_sub(1) == 1) {
                switch (type) {
                case MemType::Device:
                    {
                        #if defined(__NVCC__) || defined(FK_HIP_BACKEND)
#if defined(FK_HIP_BACKEND)
                        gpuErrchk(hipFree(ref->ptr));
#else
                        gpuErrchk(cudaFree(ref->ptr));
#endif
                        #else
                        throw std::runtime_error("Device memory deallocation not supported in non-CUDA compilation.");
                        #endif
                        break;
                    }
                case MemType::Host:
                    { 
                        hostAlignedFree(ref->ptr);
                        break;
                    }
                case MemType::HostPinned:
                    {
                        #if defined(__NVCC__) || defined(FK_HIP_BACKEND)
#if defined(FK_HIP_BACKEND)
                        gpuErrchk(hipHostFree(ref->ptr));
#else
                        gpuErrchk(cudaFreeHost(ref->ptr));
#endif
                        #else
                        throw std::runtime_error("Host pinned memory deallocation not supported in non-CUDA compilation.");
                        #endif
                        break;
                    }
                case MemType::DeviceAndPinned:
                {
#if defined(__NVCC__) || defined(FK_HIP_BACKEND)
#if defined(FK_HIP_BACKEND)
                    gpuErrchk(hipFree(ref->ptr));
#else
                    gpuErrchk(cudaFree(ref->ptr));
#endif
#if defined(FK_HIP_BACKEND)
                    gpuErrchk(hipHostFree(ref->pinnedPtr));
#else
                    gpuErrchk(cudaFreeHost(ref->pinnedPtr));
#endif
#else
                    throw std::runtime_error("Device and Host pinned memory deallocation not supported in non-CUDA compilation.");
#endif
                    break;
                }
                default:
                    break;
                }

                delete ref;
            }
            // This Ptr no longer holds a reference, also when others still share the allocation
            ref = nullptr;
        }

#if defined(__NVCC__)
        inline void copy(const RawPtr<D, T>& thisPtr, RawPtr<D, T>& other, const cudaMemcpyKind& kind,
                         cudaStream_t stream = 0) const {
            if ((other.dims.pitch == other.dims.width * sizeof(T)) && (thisPtr.dims.pitch == thisPtr.dims.width * sizeof(T))) {
                if (sizeInBytes() != PtrImpl<D, T>::sizeInBytes(other.dims)) {
                    throw std::runtime_error("Size mismatch in upload.");
                }
                const size_t totalBytes = sizeInBytes();
                gpuErrchk(cudaMemcpyAsync(other.data, thisPtr.data, totalBytes, kind, stream));
            } else {
                if constexpr (D > ND::_2D || D == ND::_1D) {
                    throw std::runtime_error("Padding only supported in 2D pointers");
                } else {
                    gpuErrchk(cudaMemcpy2DAsync(other.data, other.dims.pitch, thisPtr.data, thisPtr.dims.pitch,
                        thisPtr.dims.width * sizeof(T), thisPtr.dims.height, kind, stream));
                }
            }
        }
#elif defined(FK_HIP_BACKEND)
        inline void copy(const RawPtr<D, T>& thisPtr, RawPtr<D, T>& other, const hipMemcpyKind& kind,
                         hipStream_t stream = 0) const {
            if ((other.dims.pitch == other.dims.width * sizeof(T)) && (thisPtr.dims.pitch == thisPtr.dims.width * sizeof(T))) {
                if (sizeInBytes() != PtrImpl<D, T>::sizeInBytes(other.dims)) {
                    throw std::runtime_error("Size mismatch in upload.");
                }
                const size_t totalBytes = sizeInBytes();
                gpuErrchk(hipMemcpyAsync(other.data, thisPtr.data, totalBytes, kind, stream));
            } else {
                if constexpr (D > ND::_2D || D == ND::_1D) {
                    throw std::runtime_error("Padding only supported in 2D pointers");
                } else {
                    gpuErrchk(hipMemcpy2DAsync(other.data, other.dims.pitch, thisPtr.data, thisPtr.dims.pitch,
                        thisPtr.dims.width * sizeof(T), thisPtr.dims.height, kind, stream));
                }
            }
        }
#endif
    public:

        inline constexpr Ptr() {}

        // Copy constructor
        inline Ptr(const Ptr<D, T>& other) {
            ptr_a = other.ptr_a;
            ptr_pinned = other.ptr_pinned;
            type = other.type;
            deviceID = other.deviceID;
            ref = other.ref;
            if (ref) {
                ref->cnt.fetch_add(1);  // Increment reference count
            }
        }

        // Move constructor
        inline constexpr Ptr(Ptr<D, T>&& other) noexcept {
            ptr_a = other.ptr_a;
            ptr_pinned = other.ptr_pinned;
            type = other.type;
            deviceID = other.deviceID;
            ref = other.ref;
            other.ref = nullptr; // Prevent double free
        }

        inline constexpr Ptr(const PtrDims<D>& dims,
                             const MemType& type_ = defaultMemType, const int& deviceID_ = 0) {
            allocPtr(dims, type_, deviceID_);
        }

        inline constexpr Ptr(T* data_, const PtrDims<D>& dims, const MemType& type_, const int& deviceID_) {
            if (type_ == MemType::DeviceAndPinned) {
                throw std::runtime_error("DeviceAndPinned type requires an additional argument for the pinned pointer.");
            }
            ptr_a.data = data_;
            ptr_a.dims = dims;
            ptr_pinned = ptr_a;
            type = type_;
            deviceID = deviceID_;
        }

        inline constexpr Ptr(const RawPtr<D, T>& data_, const MemType& type_, const int& deviceID_) {
            if (type_ == MemType::DeviceAndPinned) {
                throw std::runtime_error("DeviceAndPinned type requires an additional argument for the pinned pointer.");
            }
            ptr_a = data_;
            ptr_pinned = ptr_a;
            type = type_;
            deviceID = deviceID_;
        }

        inline constexpr Ptr(T* data_, T* pinned_data, const PtrDims<D>& dims, const MemType& type_, const int& deviceID_) {
            ptr_a.data = data_;
            ptr_a.dims = dims;
            ptr_pinned.data = pinned_data;
            ptr_pinned.dims = dims;
            type = type_;
            deviceID = deviceID_;
        }

        inline constexpr Ptr(RefPtr* ref_, const RawPtr<D, T>& ptr_a_, const RawPtr<D, T>& ptr_pinned_, const MemType& type_, const int& devID) :
            ref(ref_), ptr_a(ptr_a_), ptr_pinned(ptr_pinned_), type(type_), deviceID(devID) {
            if (ref) {
                ref->cnt.fetch_add(1);  // Increment reference count
            }
        }
        
        template <fk::ND DN = D, std::enable_if_t<DN == ND::_1D, int> = 0>
        inline constexpr Ptr(const uint& num_elems, const uint& size_in_bytes = 0,
                             const MemType& type_ = defaultMemType, const int& deviceID_ = 0) {
            allocPtr(PtrDims<ND::_1D>(num_elems, size_in_bytes), type_, deviceID_);
        }

        template <fk::ND DN = D, std::enable_if_t<DN == ND::_2D, int> = 0>
        inline constexpr Ptr(const uint& width_, const uint& height_, const uint& pitch_ = 0,
                             const MemType& type_ = defaultMemType, const int& deviceID_ = 0) {
            allocPtr(PtrDims<ND::_2D>(width_, height_, pitch_), type_, deviceID_);
        }

        template <fk::ND DN = D, std::enable_if_t<DN == ND::_3D, int> = 0>
        inline constexpr Ptr(const uint& width_, const uint& height_, const uint& planes_,
                             const uint& color_planes_ = 1, const uint& pitch_ = 0,
                             const MemType& type_ = defaultMemType, const int& deviceID_ = 0) {
            allocPtr(PtrDims<ND::_3D>(width_, height_, planes_, color_planes_, pitch_), type_, deviceID_);
        }

        inline constexpr void allocPtr(const PtrDims<D>& dims_, const MemType& type_ = defaultMemType, const int& deviceID_ = 0) {
            if (ref) {
                throw std::runtime_error("Reference pointer already exists. Use a different constructor.");
            }
            ptr_a.dims = dims_;
            ptr_pinned.dims = dims_;
            type = type_;
            deviceID = deviceID_;
            ref = new RefPtr();
            if (ref == nullptr) {
                throw std::runtime_error("Failed to allocate memory for reference counter.");
            }

            switch (type) {
            case MemType::Device:
                {
                    allocDevice();
                    ptr_pinned = ptr_a;
                }
                break;
            case MemType::Host:
                {
                    allocHost();
                    ptr_pinned = ptr_a;
                }
                break;
            case MemType::HostPinned:
                {
                    allocHostPinned();
                    ptr_pinned = ptr_a;
                }
                break;
            case MemType::DeviceAndPinned:
                {
                    allocDeviceAndPinned();
                }
                break;
            default:
                break;
            }

            ref->ptr = ptr_a.data;
            ref->pinnedPtr = ptr_pinned.data;
        }

        inline ~Ptr() {
            freePtr();
        }

        inline constexpr RawPtr<D, T> ptr() const { return ptr_a; }
        inline constexpr RawPtr<D, T> ptrPinned() const { return ptr_pinned; }

        inline constexpr operator RawPtr<D, T>() const { return ptr_a; }

        // A view of this pointer, starting at p and with the extent of newDims. The crop shares the allocation: the
        // constructor adds its reference.
        // The caller only chooses the position and the extent. The strides of the view (pitch, plane_pitch and
        // color_planes_pitch) are always the ones of this pointer, taken from each of its buffers: the device buffer
        // and the pinned buffer of a DeviceAndPinned pointer have different pitches, and the crop of each walks its
        // own buffer. A pitch in newDims is not needed: 0 means "the one of the parent". A pitch that is not 0 and
        // is not the one of the parent throws, because a view can not have it. The pitch of the parent is the one of
        // its device buffer, the one dims() returns (the pinned pitch of a DeviceAndPinned pointer is not compared:
        // the caller has no way to know it). In a 1D pointer the pitch is the size in bytes, and the view gets
        // sizeof(T) * width.
        // It throws std::runtime_error (in every build) if the view is empty, has a negative position, or is not
        // inside this pointer. The crop of a crop is checked against the extent of the crop.
        inline Ptr<D, T> crop(const Point p, const PtrDims<D>& newDims) const {
            PtrCrop<D, T>::check(ptr_a.dims, p, newDims);
            const RawPtr<D, T> newRawPtr = { At::point(p, ptr_a), PtrCrop<D, T>::viewDims(ptr_a.dims, newDims) };
            if (type == MemType::DeviceAndPinned) {
                const RawPtr<D, T> newPinnedRawPtr = { At::point(p, ptr_pinned),
                                                       PtrCrop<D, T>::viewDims(ptr_pinned.dims, newDims) };
                return { ref, newRawPtr, newPinnedRawPtr, type, deviceID };
            } else {
                return { ref, newRawPtr, newRawPtr, type, deviceID };
            }
        }

        // The view of a rectangle of a 2D pointer: the position and the size, nothing else
        template <ND DN = D, std::enable_if_t<DN == ND::_2D, int> = 0>
        inline Ptr<ND::_2D, T> crop(const Rect& rect) const {
            if (rect.width <= 0 || rect.height <= 0) {
                ptr_crop::fail("the rectangle is empty or has a negative size (" + std::to_string(rect.width) + " x " +
                               std::to_string(rect.height) + ")");
            }
            return crop(Point{ static_cast<int>(rect.x), static_cast<int>(rect.y), 0 },
                        PtrDims<ND::_2D>(static_cast<uint>(rect.width), static_cast<uint>(rect.height)));
        }

        inline constexpr PtrDims<D> dims() const {
            return ptr_a.dims;
        }

        inline constexpr MemType getMemType() const {
            return type;
        }

        inline constexpr int getDeviceID() const {
            return deviceID;
        }

        inline constexpr size_t sizeInBytes() const {
            return PtrImpl<D, T>::sizeInBytes(ptr_a.dims);
        }

        inline constexpr uint getNumElements() const {
            return PtrImpl<D, T>::getNumElements(ptr_a.dims);
        }

        inline constexpr int getRefCount() const {
            return ref ? ref->cnt.load() : 0;
        }

        // Copy assignment operator
        Ptr<D, T>& operator=(const Ptr<D, T>& other) {
            if (this != &other) {  // Self-assignment check
                freePtr();         // Clean up current resources first

                ptr_a = other.ptr_a;
                ptr_pinned = other.ptr_pinned;
                type = other.type;
                deviceID = other.deviceID;
                ref = other.ref;
                if (ref) {
                    ref->cnt.fetch_add(1);
                }
            }
            return *this;
        }
        // Move assignment operator
        Ptr<D, T>& operator=(Ptr<D, T>&& other) noexcept {
            if (this != &other) {
                freePtr();  // Clean up current resources

                ptr_a = other.ptr_a;
                ptr_pinned = other.ptr_pinned;
                type = other.type;
                deviceID = other.deviceID;
                ref = other.ref;

                other.ref = nullptr;  // Transfer ownership
            }
            return *this;
        }

#if defined(__NVCC__)
        inline void uploadTo(Ptr<D, T>& other, cudaStream_t stream = 0) {
            constexpr cudaMemcpyKind kind = cudaMemcpyHostToDevice;
            constexpr MemType otherExpectedMemType1 = MemType::Device;
            constexpr MemType otherExpectedMemType2 = MemType::DeviceAndPinned;
            constexpr MemType thisExpectedMemType1 = MemType::Host;
            constexpr MemType thisExpectedMemType2 = MemType::HostPinned;
            if (type == thisExpectedMemType1 || type == thisExpectedMemType2) {
                if (other.getMemType() == otherExpectedMemType1 || other.getMemType() == otherExpectedMemType2) {
                    auto dstRawPtr = other.ptr();
                    copy(ptr_a, dstRawPtr, kind, stream);
                } else {
                    throw std::runtime_error("Upload can only copy to Device pointers");
                }
            } else {
                throw std::runtime_error("Upload can only copy from Host or HostPinned pointers.");
            }
        }

        inline void downloadTo(Ptr<D, T>& other, cudaStream_t stream = 0) {
            constexpr cudaMemcpyKind kind = cudaMemcpyDeviceToHost;
            constexpr MemType otherExpectedMemType1 = MemType::Host;
            constexpr MemType otherExpectedMemType2 = MemType::HostPinned;
            constexpr MemType thisExpectedMemType1 = MemType::Device;
            constexpr MemType thisExpectedMemType2 = MemType::DeviceAndPinned;
            if (type == thisExpectedMemType1 || type == thisExpectedMemType2) {
                if (other.getMemType() == otherExpectedMemType1 || other.getMemType() == otherExpectedMemType2) {
                    auto dstRawPtr = other.ptr();
                    copy(ptr_a, dstRawPtr, kind, stream);
                } else {
                    throw std::runtime_error("Download can only copy to Host or HostPinned pointers.");
                }
            } else {
                throw std::runtime_error("Download can only copy from Device pointers.");
            }
        }

        inline void upload(Stream_<ParArch::GPU_NVIDIA>& stream) {
            if (type == MemType::DeviceAndPinned) {
                constexpr cudaMemcpyKind kind = cudaMemcpyHostToDevice;
                copy(ptr_pinned, ptr_a, kind, stream);
            }
        }
        inline void download(Stream_<ParArch::GPU_NVIDIA>& stream) {
            if (type == MemType::DeviceAndPinned) {
                constexpr cudaMemcpyKind kind = cudaMemcpyDeviceToHost;
                copy(ptr_a, ptr_pinned, kind, stream);
            }
        }
#elif defined(FK_HIP_BACKEND)
        inline void uploadTo(Ptr<D, T>& other, hipStream_t stream = 0) {
            constexpr hipMemcpyKind kind = hipMemcpyHostToDevice;
            constexpr MemType otherExpectedMemType1 = MemType::Device;
            constexpr MemType otherExpectedMemType2 = MemType::DeviceAndPinned;
            constexpr MemType thisExpectedMemType1 = MemType::Host;
            constexpr MemType thisExpectedMemType2 = MemType::HostPinned;
            if (type == thisExpectedMemType1 || type == thisExpectedMemType2) {
                if (other.getMemType() == otherExpectedMemType1 || other.getMemType() == otherExpectedMemType2) {
                    auto dstRawPtr = other.ptr();
                    copy(ptr_a, dstRawPtr, kind, stream);
                } else {
                    throw std::runtime_error("Upload can only copy to Device pointers");
                }
            } else {
                throw std::runtime_error("Upload can only copy from Host or HostPinned pointers.");
            }
        }

        inline void downloadTo(Ptr<D, T>& other, hipStream_t stream = 0) {
            constexpr hipMemcpyKind kind = hipMemcpyDeviceToHost;
            constexpr MemType otherExpectedMemType1 = MemType::Host;
            constexpr MemType otherExpectedMemType2 = MemType::HostPinned;
            constexpr MemType thisExpectedMemType1 = MemType::Device;
            constexpr MemType thisExpectedMemType2 = MemType::DeviceAndPinned;
            if (type == thisExpectedMemType1 || type == thisExpectedMemType2) {
                if (other.getMemType() == otherExpectedMemType1 || other.getMemType() == otherExpectedMemType2) {
                    auto dstRawPtr = other.ptr();
                    copy(ptr_a, dstRawPtr, kind, stream);
                } else {
                    throw std::runtime_error("Download can only copy to Host or HostPinned pointers.");
                }
            } else {
                throw std::runtime_error("Download can only copy from Device pointers.");
            }
        }

        inline void upload(Stream_<ParArch::GPU_AMD>& stream) {
            if (type == MemType::DeviceAndPinned) {
                constexpr hipMemcpyKind kind = hipMemcpyHostToDevice;
                copy(ptr_pinned, ptr_a, kind, stream);
            }
        }
        inline void download(Stream_<ParArch::GPU_AMD>& stream) {
            if (type == MemType::DeviceAndPinned) {
                constexpr hipMemcpyKind kind = hipMemcpyDeviceToHost;
                copy(ptr_a, ptr_pinned, kind, stream);
            }
        }
#else
        inline void upload(Stream& stream) {}
        inline void download(Stream& stream) {}
#endif // defined(__NVCC__) || defined(FK_HIP_BACKEND)

#if defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)
        // ParArch::CPU_MULTITHREAD executions use ptr() directly from the CPU, so there is nothing to copy and these
        // are no-ops. They are ordered with the executions of the stream as the GPU ones are: the executions enqueued
        // after upload() see what the host wrote before it, and after download() and sync() the host sees what the
        // executions enqueued before download() wrote. Unlike DeviceAndPinned memory there is a single buffer: the host
        // must not access it while executions that use it are pending, and after sync() it also sees what executions
        // enqueued after download() wrote. The memory must be Host or HostPinned: with Device or DeviceAndPinned memory
        // (defaultMemType in CUDA and HIP builds) ptr() is a device pointer, and they throw std::runtime_error.
        inline void upload(Stream_<ParArch::CPU_MULTITHREAD>&) {
            cpu_multithread::requireHostMemory(type, "Ptr::upload");
        }
        inline void download(Stream_<ParArch::CPU_MULTITHREAD>&) {
            cpu_multithread::requireHostMemory(type, "Ptr::download");
        }
#endif // defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)

        inline T at(const Point p) const {
            if (type != MemType::Device) {
                return *At::cr_point(p, ptr_pinned);
            } else {
                throw std::runtime_error("Cannot access data in Device memory from host code");
                return make_set<T>(0);
            }
        }

        inline T at(const int x) const {
            return at(Point{x, 0, 0});
        }

        template <ND Dims = D>
        inline std::enable_if_t<(Dims == ND::_2D), T>
        at(const int x, const int y) const {
            return at(Point{x, y, 0});
        }

        template <ND Dims = D>
        inline std::enable_if_t<(Dims == ND::_3D), T>
        at(const int x, const int y, const int z) const {
            return at(Point{x, y, z});
        }

        inline T& at(const Point p) {
            if (type != MemType::Device) {
                return *At::point(p, ptr_pinned);
            } else {
                throw std::runtime_error("Cannot access data in Device memory from host code");
                //return make_set<T>(0);
            }
        }

        inline T& at(const int x) {
            T& val = at(Point{x, 0, 0});
            return val;
        }

        template <ND Dims = D>
        inline std::enable_if_t<(Dims == ND::_2D), T&>
            at(const int x, const int y) {
            T& val = at(Point{x, y, 0});
            return val;
        }

        template <ND Dims = D>
        inline std::enable_if_t<(Dims == ND::_3D), T&>
            at(const int x, const int y, const int z) {
            T& val = at(Point{x, y, z});
            return val;
        }

        template <enum ND DIM = D>
        inline std::enable_if_t<(DIM > ND::_2D), Ptr<ND::_2D, T>> getPlane(const uint& plane) {
            if (plane >= this->ptr_pinned.dims.planes) {
                throw std::runtime_error("Plane index out of bounds");
            }
            const PtrDims<ND::_2D> dims_a{ ptr_a.dims.width, ptr_a.dims.height, ptr_a.dims.pitch };
            const PtrDims<ND::_2D> dims_pinned{ ptr_pinned.dims.width, ptr_pinned.dims.height, ptr_pinned.dims.pitch };
            // The plane shares the allocation: the constructor adds its reference
            const Point p{ 0, 0, static_cast<int>(plane) };
            return { ref, RawPtr<ND::_2D, T>{At::point(p, ptr_a), dims_a}, RawPtr<ND::_2D, T>{At::point(p, ptr_pinned), dims_pinned}, type, deviceID };
        }
    };

    template <ND D, typename T>
    FK_HOST_CNST auto PtrND(const RawPtr<D, T>& ptr) {
        return Ptr<D, T>(ptr);
    }

    template <typename T>
    class Ptr1D : public Ptr<ND::_1D, T> {
    public:
        inline constexpr Ptr1D<T>() {}
        inline constexpr Ptr1D<T>(const uint& num_elems, const uint& size_in_bytes = 0, const MemType& type_ = defaultMemType, const int& deviceID_ = 0) :
            Ptr<ND::_1D, T>(PtrDims<ND::_1D>(num_elems, size_in_bytes), type_, deviceID_) {}

        inline constexpr Ptr1D<T>(const Ptr<ND::_1D, T>& other) : Ptr<ND::_1D, T>(other) {}

        inline constexpr Ptr1D<T>(T* data_, const PtrDims<ND::_1D>& dims_, const MemType& type_ = defaultMemType, const int& deviceID_ = 0) :
            Ptr<ND::_1D, T>(data_, dims_, type_, deviceID_) {}

        inline Ptr1D<T> crop1D(const Point p, const PtrDims<ND::_1D>& newDims) const { return Ptr<ND::_1D, T>::crop(p, newDims); }
        // The elements [x, x + width)
        inline Ptr1D<T> crop1D(const int x, const uint width) const { return crop1D(Point{ x, 0, 0 }, PtrDims<ND::_1D>(width)); }
    };

    template <typename T>
    class Ptr2D : public Ptr<ND::_2D, T> {
    public:
        inline constexpr Ptr2D<T>() {}
        inline Ptr2D<T>(const Size& size, const uint& pitch_ = 0, const MemType& type_ = defaultMemType, const int& deviceID_ = 0) :
            Ptr<ND::_2D, T>(PtrDims<ND::_2D>(size.width, size.height, pitch_), type_, deviceID_) {}
        inline Ptr2D<T>(const uint& width_, const uint& height_, const uint& pitch_ = 0, const MemType& type_ = defaultMemType, const int& deviceID_ = 0) :
            Ptr<ND::_2D, T>(PtrDims<ND::_2D>(width_, height_, pitch_), type_, deviceID_) {}

        inline constexpr Ptr2D<T>(const Ptr<ND::_2D, T>& other) : Ptr<ND::_2D, T>(other) {}

        inline Ptr2D<T>(T* data_, const uint& width_, const uint& height_, const uint& pitch_, const MemType& type_ = defaultMemType, const int& deviceID_ = 0) :
            Ptr<ND::_2D, T>(data_, PtrDims<ND::_2D>(width_, height_, pitch_), type_, deviceID_) {}

        inline Ptr2D<T> crop2D(const Point p, const PtrDims<ND::_2D>& newDims) const { return Ptr<ND::_2D, T>::crop(p, newDims); }
        inline Ptr2D<T> crop2D(const Rect& rect) const { return Ptr<ND::_2D, T>::crop(rect); }
        inline void Alloc(const fk::Size& size, const uint& pitch_ = 0, const MemType& type_ = defaultMemType, const int& deviceID_ = 0) {
            this->freePtr();
            this->allocPtr(PtrDims<ND::_2D>(size.width, size.height, pitch_), type_, deviceID_);
        }
    };

    // A Ptr3D pointer
    template <typename T>
    class Ptr3D : public Ptr<ND::_3D, T> {
    public:
        inline constexpr Ptr3D<T>(const uint& width_, const uint& height_, const uint& planes_, const uint& color_planes_ = 1, const uint& pitch_ = 0, const MemType& type_ = defaultMemType, const int& deviceID_ = 0) :
            Ptr<ND::_3D, T>(PtrDims<ND::_3D>(width_, height_, planes_, color_planes_, pitch_), type_, deviceID_) {}

        inline constexpr Ptr3D<T>(const Ptr<ND::_3D, T>& other) : Ptr<ND::_3D, T>(other) {}

        inline constexpr Ptr3D<T>(T* data_, const PtrDims<ND::_3D>& dims_, const MemType& type_ = defaultMemType, const int& deviceID_ = 0) :
            Ptr<ND::_3D, T>(data_, dims_, type_, deviceID_) {}

        inline Ptr3D<T> crop3D(const Point p, const PtrDims<ND::_3D>& newDims) const { return Ptr<ND::_3D, T>::crop(p, newDims); }
    };

    // A color-plane-transposed 3D pointer PtrT3D
    template <typename T>
    class PtrT3D : public Ptr<ND::T3D, T> {
    public:
        inline constexpr PtrT3D<T>(const uint& width_, const uint& height_, const uint& planes_, const uint& color_planes_ = 1, const MemType& type_ = defaultMemType, const int& deviceID_ = 0) :
            Ptr<ND::T3D, T>(PtrDims<ND::T3D>(width_, height_, planes_, color_planes_), type_, deviceID_) {}

        inline constexpr PtrT3D<T>(const Ptr<ND::T3D, T>& other) : Ptr<ND::T3D, T>(other) {}

        inline constexpr PtrT3D<T>(T* data_, const PtrDims<ND::T3D>& dims_, const MemType& type_ = defaultMemType, const int& deviceID_ = 0) :
            Ptr<ND::T3D, T>(data_, dims_, type_, deviceID_) {}

        inline PtrT3D<T> crop3D(const Point p, const PtrDims<ND::T3D>& newDims) const { return Ptr<ND::T3D, T>::crop(p, newDims); }
    };

    // A Tensor pointer
    template <typename T>
    class Tensor : public Ptr<ND::_3D, T> {
    public:
        inline constexpr Tensor() {}

        inline constexpr Tensor(const Tensor<T>& other) : Ptr<ND::_3D, T>(other) {}

        inline constexpr Tensor(const uint& width_, const uint& height_, const uint& planes_, const uint& color_planes_ = 1, const MemType& type_ = defaultMemType, const int& deviceID_ = 0) :
            Ptr<ND::_3D, T>(PtrDims<ND::_3D>(width_, height_, planes_, color_planes_, sizeof(T)* width_), type_, deviceID_) {}

        inline constexpr Tensor(T* data, const uint& width_, const uint& height_, const uint& planes_, const uint& color_planes_ = 1, const MemType& type_ = defaultMemType, const int& deviceID_ = 0) :
            Ptr<ND::_3D, T>(data, PtrDims<ND::_3D>(width_, height_, planes_, color_planes_, sizeof(T)* width_), type_, deviceID_) {}

        inline constexpr void allocTensor(const uint& width_, const uint& height_, const uint& planes_, const uint& color_planes_ = 1, const MemType& type_ = defaultMemType, const int& deviceID_ = 0) {
            this->freePtr();
            this->allocPtr(PtrDims<ND::_3D>(width_, height_, planes_, color_planes_, sizeof(T) * width_), type_, deviceID_);
        }
    };

    // A color-plane-transposed Tensor pointer
    template <typename T>
    class TensorT : public Ptr<ND::T3D, T> {
    public:
        inline constexpr TensorT() {}

        inline constexpr TensorT(const TensorT<T>& other) : Ptr<ND::T3D, T>(other) {}

        inline constexpr TensorT(const uint& width_, const uint& height_, const uint& planes_, const uint& color_planes_ = 1, const MemType& type_ = defaultMemType, const int& deviceID_ = 0) :
            Ptr<ND::T3D, T>(PtrDims<ND::T3D>(width_, height_, planes_, color_planes_), type_, deviceID_) {}

        inline constexpr TensorT(T* data, const uint& width_, const uint& height_, const uint& planes_, const uint& color_planes_ = 1, const MemType& type_ = defaultMemType, const int& deviceID_ = 0) :
            Ptr<ND::T3D, T>(data, PtrDims<ND::T3D>(width_, height_, planes_, color_planes_), type_, deviceID_) {}

        inline constexpr void allocTensor(const uint& width_, const uint& height_, const uint& planes_, const uint& color_planes_ = 1, const MemType& type_ = defaultMemType, const int& deviceID_ = 0) {
            this->freePtr();
            this->allocPtr(PtrDims<ND::T3D>(width_, height_, planes_, color_planes_), type_, deviceID_);
        }
    };
} // namespace fk

#endif
