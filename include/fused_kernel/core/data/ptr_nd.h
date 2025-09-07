/* Copyright 2023-2025 Oscar Amoros Huguet

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
#include <fused_kernel/core/data/size.h>
#if !defined(NVRTC_COMPILER)
#include <fused_kernel/core/execution_model/stream.h>
#endif
#include <fused_kernel/core/utils/cuda_vector_utils.h>

#include <atomic>

namespace fk {
	enum class MemType { Device, Host, HostPinned, DeviceAndPinned };
#if defined(__NVCC__) || defined(__CUDA__) || defined(__HIP__) || defined(NVRTC_ENABLED)
    constexpr MemType defaultMemType = MemType::DeviceAndPinned;
#else
    constexpr MemType defaultMemType = MemType::Host;
#endif

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
            gpuErrchk(cudaMalloc(&ptr_a.data, ptr_a.dims.pitch));
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
                size_t pitch;
                gpuErrchk(cudaMallocPitch(&ptr_a.data, &pitch, sizeof(T) * ptr_a.dims.width, ptr_a.dims.height));
                ptr_a.dims.pitch = static_cast<int>(pitch);
            } else {
                gpuErrchk(cudaMalloc(&ptr_a.data, PtrImpl<ND::_2D, T>::sizeInBytes(ptr_a.dims)));
            }
        }
        FK_HOST_FUSE void h_malloc_init(PtrDims<ND::_2D>& dims) {
            if (dims.pitch == 0) {
                dims.pitch = sizeof(T) * dims.width;
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
            gpuErrchk(cudaMalloc(&ptr_a.data, PtrImpl<ND::_3D, T>::sizeInBytes(ptr_a.dims)));
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
            gpuErrchk(cudaMalloc(&ptr_a.data, PtrImpl<ND::T3D, T>::sizeInBytes(ptr_a.dims)));
        }
        FK_HOST_FUSE void h_malloc_init(PtrDims<ND::T3D>& dims) {
            if (dims.pitch == 0) {
                dims.pitch = sizeof(T) * dims.width;
            }
            dims.plane_pitch = dims.pitch * dims.height;
            dims.color_planes_pitch = dims.plane_pitch * dims.planes;
        }
    };

    struct RefPtr {
        void* ptr{nullptr};
        void* pinnedPtr{nullptr};
        std::atomic<int> cnt{1};
    };

    template <enum ND D, typename T>
    class Ptr {
        using Type = T;
        using At = PtrAccessor<D>;

    protected:
        RefPtr* ref{ nullptr };
        RawPtr<D, T> ptr_a;
        RawPtr<D, T> ptr_pinned;
        MemType type;
        int deviceID;

        inline constexpr Ptr(const RawPtr<D, T>& ptr_a_, RefPtr* ref_, const MemType& type_, const int& devID) :
            ref(ref_), ptr_a(ptr_a_), ptr_pinned(ptr_a_), type(type_), deviceID(devID) {
            if (ref) {
                ref->cnt.fetch_add(1);
            }
        }

        inline constexpr void allocDevice() {
            #if defined(__NVCC__) || defined(__CUDA__) || defined(__HIP__) || defined(NVRTC_ENABLED)
            int currentDevice;
            gpuErrchk(cudaGetDevice(&currentDevice));
            gpuErrchk(cudaSetDevice(deviceID));
            PtrImpl<D, T>::d_malloc(ptr_a);
            if (currentDevice != deviceID) {
                gpuErrchk(cudaSetDevice(currentDevice));
            }
            #else
            throw std::runtime_error("Device allocation not supported in non-CUDA compilation.");
            #endif
        }

        inline constexpr void allocHost() {
            PtrImpl<D, T>::h_malloc_init(ptr_a.dims);
            ptr_a.data = (T*)malloc(PtrImpl<D, T>::sizeInBytes(ptr_a.dims));
        }

        inline constexpr void allocHostPinned() {
            #if defined(__NVCC__) || defined(__CUDA__) || defined(__HIP__) || defined(NVRTC_ENABLED)
            int currentDevice;
            gpuErrchk(cudaGetDevice(&currentDevice));
            gpuErrchk(cudaSetDevice(deviceID));
            PtrImpl<D, T>::h_malloc_init(ptr_a.dims);
            gpuErrchk(cudaMallocHost(&ptr_a.data, PtrImpl<D, T>::sizeInBytes(ptr_a.dims)));
            if (currentDevice != deviceID) {
                gpuErrchk(cudaSetDevice(currentDevice));
            }
            #else
            throw std::runtime_error("Host pinned allocation not supported in non-CUDA compilation.");
            #endif
        }

        inline constexpr void allocDeviceAndPinned() {
            #if defined(__NVCC__) || defined(__CUDA__) || defined(__HIP__) || defined(NVRTC_ENABLED)
            int currentDevice;
            gpuErrchk(cudaGetDevice(&currentDevice));
            gpuErrchk(cudaSetDevice(deviceID));
            PtrImpl<D, T>::d_malloc(ptr_a);
            PtrImpl<D, T>::h_malloc_init(ptr_pinned.dims);
            gpuErrchk(cudaMallocHost(&ptr_pinned.data, PtrImpl<D, T>::sizeInBytes(ptr_pinned.dims)));
            if (currentDevice != deviceID) {
                gpuErrchk(cudaSetDevice(currentDevice));
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
                        #if defined(__NVCC__) || defined(__CUDA__) || defined(__HIP__) || defined(NVRTC_ENABLED)
                        gpuErrchk(cudaFree(ref->ptr));
                        #else
                        throw std::runtime_error("Device memory deallocation not supported in non-CUDA compilation.");
                        #endif
                        break;
                    }
                case MemType::Host:
                    { 
                        free(ref->ptr);
                        break;
                    }
                case MemType::HostPinned:
                    {
                        #if defined(__NVCC__) || defined(__CUDA__) || defined(__HIP__) || defined(NVRTC_ENABLED)
                        gpuErrchk(cudaFreeHost(ref->ptr));
                        #else
                        throw std::runtime_error("Host pinned memory deallocation not supported in non-CUDA compilation.");
                        #endif
                        break;
                    }
                case MemType::DeviceAndPinned:
                {
#if defined(__NVCC__) || defined(__CUDA__) || defined(__HIP__) || defined(NVRTC_ENABLED)
                    gpuErrchk(cudaFree(ref->ptr));
                    gpuErrchk(cudaFreeHost(ref->pinnedPtr));
#else
                    throw std::runtime_error("Device and Host pinned memory deallocation not supported in non-CUDA compilation.");
#endif
                    break;
                }
                default:
                    break;
                }

                delete ref;
                ref = nullptr;
            }
        }

#if defined(__NVCC__) || defined(__CUDA__) || defined(__HIP__) || defined(NVRTC_ENABLED)
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

        // Check if the compiler is specifically MSVC for VS 2017
#if VS2017_COMPILER
        template <typename... Args>
        constexpr Ptr(Args&&... args) {
            init(std::integral_constant<ND, D>{}, std::forward<Args>(args)...);
        }

    private:
        inline constexpr void init(const std::integral_constant<ND, ND::_1D>&,
                                   const uint& num_elems, const uint& size_in_bytes = 0,
                                   const MemType& type_ = defaultMemType, const int& deviceID_ = 0) {
            allocPtr(PtrDims<ND::_1D>(num_elems, size_in_bytes), type_, deviceID_);
        }
        inline constexpr void init(const std::integral_constant<ND, ND::_2D>&,
                                   const uint& width_, const uint& height_, const uint& pitch_ = 0,
                                   const MemType& type_ = defaultMemType, const int& deviceID_ = 0) {
            allocPtr(PtrDims<ND::_2D>(width_, height_, pitch_), type_, deviceID_);
        }
        inline constexpr Ptr(const std::integral_constant<ND, ND::_3D>&,
                             const uint& width_, const uint& height_, const uint& planes_,
                             const uint& color_planes_ = 1, const uint& pitch_ = 0,
                             const MemType& type_ = defaultMemType, const int& deviceID_ = 0) {
            allocPtr(PtrDims<ND::_3D>(width_, height_, planes_, color_planes_, pitch_), type_, deviceID_);
        }
        template <ND D_ = D>
        inline constexpr void init(const std::integral_constant<ND, D_>&,
                                   const PtrDims<D_>& dims,
                                   const MemType& type_ = defaultMemType, const int& deviceID_ = 0) {
            allocPtr(dims, type_, deviceID_);
        }

        template <ND D_ = D>
        inline constexpr void init(const std::integral_constant<ND, D_>&,
                                   T* data_, const PtrDims<D_>& dims, const MemType& type_, const int& deviceID_) {
            if (type_ == MemType::DeviceAndPinned) {
                throw std::runtime_error("DeviceAndPinned type requires an additional argument for the pinned pointer.");
            }
            ptr_a.data = data_;
            ptr_a.dims = dims;
            ptr_pinned = ptr_a;
            type = type_;
            deviceID = deviceID_;
        }

        template <ND D_ = D>
        inline constexpr void init(const std::integral_constant<ND, D_>&, const RawPtr<D_, T>& data_, const MemType& type_, const int& deviceID_) {
            if (type_ == MemType::DeviceAndPinned) {
                throw std::runtime_error("DeviceAndPinned type requires an additional argument for the pinned pointer.");
            }
            ptr_a = data_;
            ptr_pinned = ptr_a;
            type = type_;
            deviceID = deviceID_;
        }

        template <ND D_ = D>
        inline constexpr void init(const std::integral_constant<ND, D_>&, T* data_, T* pinned_data, const PtrDims<D_>& dims, const MemType& type_, const int& deviceID_) {
            ptr_a.data = data_;
            ptr_a.dims = dims;
            ptr_pinned.data = pinned_data;
            ptr_pinned.dims = dims;
            type = type_;
            deviceID = deviceID_;
        }

        template <ND D_ = D>
        inline constexpr void init(const std::integral_constant<ND, D_>&, RefPtr* ref_, const RawPtr<D, T>& ptr_a_, const RawPtr<D, T>& ptr_pinned_, const MemType& type_, const int& devID) {
            ref = ref_;
            ptr_a = ptr_a_;
            ptr_pinned = ptr_pinned_;
            type = type_;
            deviceID = devID;
            if (ref) {
                ref->cnt.fetch_add(1);  // Increment reference count
            }
        }

    public:
#else
        inline constexpr Ptr(RefPtr* ref_, const RawPtr<D, T>& ptr_a_, const RawPtr<D, T>& ptr_pinned_, const MemType& type_, const int& devID) :
            ref(ref_), ptr_a(ptr_a_), ptr_pinned(ptr_pinned_), type(type_), deviceID(devID) {
            if (ref) {
                ref->cnt.fetch_add(1);  // Increment reference count
            }
        }
        // Modern, more idiomatic version for all other compliant compilers (VS 2019+, GCC, Clang)
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

        inline constexpr Ptr(const PtrDims<D>& dims, const MemType& type_ = defaultMemType, const int& deviceID_ = 0) {
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
#endif
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

        inline constexpr Ptr<D, T> crop(const Point& p, const PtrDims<D>& newDims) {
            T* ptr = At::point(p, ptr_a);
            if (ref) {
                ref->cnt.fetch_add(1);
            }
            const RawPtr<D, T> newRawPtr = { ptr, newDims };
            if (type == MemType::DeviceAndPinned) {
                T* pinnedPtr = At::point(p, ptr_pinned);
                RawPtr<D, T> newPinnedRawPtr = { pinnedPtr, newDims };
                return { ref, newRawPtr, newPinnedRawPtr, type, deviceID };
            } else {
                return { ref, newRawPtr, newRawPtr, type, deviceID };
            }
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

#if !defined(NVRTC_COMPILER)
#if defined(__NVCC__) || defined(__CUDA__) || defined(__HIP__) || defined(NVRTC_ENABLED)
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
#else
        inline void upload(Stream& stream) {}
        inline void download(Stream& stream) {}
#endif // defined(__NVCC__) || defined(__HIP__) || defined(NVRTC_ENABLED)
#endif // defined(NVRTC_COMPILER)

        inline T at(const Point& p) const {
            if (type != MemType::Device) {
                return *At::cr_point(p, ptr_pinned);
            } else {
                throw std::runtime_error("Cannot access data in Device memory from host code");
                return make_set<T>(0);
            }
        }

        inline T at(const uint& x) const {
            return at(Point(x, 0, 0));
        }

        template <ND Dims = D>
        inline std::enable_if_t<(Dims == ND::_2D), T>
        at(const uint& x, const uint& y) const {
            return at(Point(x, y, 0));
        }

        template <ND Dims = D>
        inline std::enable_if_t<(Dims == ND::_3D), T>
        at(const uint& x, const uint& y, const uint& z) const {
            return at(Point(x, y, z));
        }

        inline T& at(const Point& p) {
            if (type != MemType::Device) {
                return *At::point(p, ptr_pinned);
            } else {
                throw std::runtime_error("Cannot access data in Device memory from host code");
                //return make_set<T>(0);
            }
        }

        inline T& at(const uint& x) {
            T& val = at(Point(x, 0, 0));
            return val;
        }

        template <ND Dims = D>
        inline std::enable_if_t<(Dims == ND::_2D), T&>
            at(const uint& x, const uint& y) {
            T& val = at(Point(x, y, 0));
            return val;
        }

        template <ND Dims = D>
        inline std::enable_if_t<(Dims == ND::_3D), T&>
            at(const uint& x, const uint& y, const uint& z) {
            T& val = at(Point(x, y, z));
            return val;
        }

        template <enum ND DIM = D>
        inline std::enable_if_t<(DIM > ND::_2D), Ptr<ND::_2D, T>> getPlane(const uint& plane) {
            if (plane >= this->ptr_pinned.dims.planes) {
                throw std::runtime_error("Plane index out of bounds");
            }
            const PtrDims<ND::_2D> dims_a{ ptr_a.dims.width, ptr_a.dims.height, ptr_a.dims.pitch };
            const PtrDims<ND::_2D> dims_pinned{ ptr_pinned.dims.width, ptr_pinned.dims.height, ptr_pinned.dims.pitch };
            if (ref) {
                ref->cnt.fetch_add(1);
            }
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

        inline constexpr Ptr1D<T> crop1D(const Point& p, const PtrDims<ND::_1D>& newDims) { return Ptr<ND::_1D, T>::crop(p, newDims); }
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

        inline Ptr2D<T> crop2D(const Point& p, const PtrDims<ND::_2D>& newDims) { return Ptr<ND::_2D, T>::crop(p, newDims); }
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

        inline constexpr Ptr3D<T> crop3D(const Point& p, const PtrDims<ND::_3D>& newDims) { return Ptr<ND::_3D, T>::crop(p, newDims); }
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

        inline constexpr PtrT3D<T> crop3D(const Point& p, const PtrDims<ND::T3D>& newDims) { return Ptr<ND::T3D, T>::crop(p, newDims); }
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
