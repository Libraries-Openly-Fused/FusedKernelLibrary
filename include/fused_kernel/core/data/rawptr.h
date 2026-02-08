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

#ifndef FK_RAWPTR_H
#define FK_RAWPTR_H

#include <fused_kernel/core/data/nd.h>
#include <fused_kernel/core/data/point.h>

namespace fk {
    template <enum ND D>
    struct PtrDims;

    template <>
    struct PtrDims<ND::_1D> {
        uint width{0};
        uint pitch{0};

        FK_HOST_DEVICE_CNST
            PtrDims<ND::_1D>() {}
        FK_HOST_DEVICE_CNST
            PtrDims<ND::_1D>(uint width_, uint pitch_ = 0) : width(width_), pitch(pitch_) {}
    };

    template <>
    struct PtrDims<ND::_2D> {
        uint width{ 0 };
        uint height{ 0 };
        uint pitch{ 0 };

        FK_HOST_DEVICE_CNST PtrDims<ND::_2D>() {}
        FK_HOST_DEVICE_CNST PtrDims<ND::_2D>(uint width_, uint height_, uint pitch_ = 0) :
            width(width_), height(height_), pitch(pitch_) {}
    };

    template <>
    struct PtrDims<ND::_3D> {
        // Image batch shape
        // R,G,B
        // R,G,B
        // R,G,B

        // Width and Height of one individual image
        uint width{0};
        uint height{0};
        // Number of images
        uint planes{0};
        // Number of color channels
        uint color_planes{0};

        // Pitch for each image
        uint pitch{0};

        // Pitch to jump one plane
        uint plane_pitch{0};

        FK_HOST_DEVICE_CNST PtrDims<ND::_3D>() {}
        FK_HOST_DEVICE_CNST
            PtrDims<ND::_3D>(uint width_, uint height_, uint planes_, uint color_planes_ = 1, uint pitch_ = 0) :
            width(width_), height(height_), planes(planes_), color_planes(color_planes_),
            pitch(pitch_), plane_pitch(pitch_ * height_) {}
    };

    template <>
    struct PtrDims<ND::T3D> {
        // Image batch shape
        // R,R,R
        // G,G,G
        // B,B,B

        // Width and Height of one individual image
        uint width{ 0 };
        uint height{ 0 };
        // Number of images
        uint planes{ 0 };
        // Number of color channels
        uint color_planes{ 0 };

        // Pitch for each image
        uint pitch{ 0 };

        // Pitch to jump one plane
        uint plane_pitch{ 0 };

        // Pitch to jump to the next plane of the same image
        uint color_planes_pitch{ 0 };

        FK_HOST_DEVICE_CNST PtrDims<ND::T3D>() {}
        FK_HOST_DEVICE_CNST
            PtrDims<ND::T3D>(uint width_, uint height_, uint planes_, uint color_planes_ = 1) :
            width(width_), height(height_), planes(planes_), color_planes(color_planes_),
            pitch(0), plane_pitch(0), color_planes_pitch(0) {}
    };

    template <int W>
    struct StaticPtrDims1D {
        static constexpr uint width{ W };
    };

    template <int W, int H>
    struct StaticPtrDims2D {
        static constexpr uint width{ W };
        static constexpr uint height{ H };
    };

    template<int W, int H, int P>
    struct StaticPtrDims3D {
        static constexpr uint width{ W };
        static constexpr uint height{ H };
        static constexpr uint planes{ P };
    };

    template <enum ND D, typename T>
    struct RawPtr;

    template <typename T>
    struct RawPtr<ND::_1D, T> {
        T* data{nullptr};
        PtrDims<ND::_1D> dims{};
        using type = T;
        enum { NDim = static_cast<int>(ND::_1D) };
    };

    template <typename T>
    struct RawPtr<ND::_2D, T> {
        T* data{nullptr};
        PtrDims<ND::_2D> dims{};
        using type = T;
        enum { NDim = static_cast<int>(ND::_2D) };
    };

    template <typename T>
    struct RawPtr<ND::_3D, T> {
        T* data{nullptr};
        PtrDims<ND::_3D> dims{};
        using type = T;
        enum { NDim = static_cast<int>(ND::_3D) };
    };

    template <typename T>
    struct RawPtr<ND::T3D, T> {
        T* data{nullptr};
        PtrDims<ND::T3D> dims{};
        using type = T;
        enum { NDim = static_cast<int>(ND::T3D) };
    };

    template<typename Dims, typename T>
    struct StaticRawPtr;

    template<typename T, int W>
    struct StaticRawPtr<StaticPtrDims1D<W>, T> {
        using type = T;
        T data[W];
        static constexpr StaticPtrDims1D<W> dims{};
        static constexpr ND NDim{ ND::_1D };
    };

    template<typename T, int W, int H>
    struct StaticRawPtr<StaticPtrDims2D<W, H>, T> {
        using type = T;
        T data[H][W];
        static constexpr StaticPtrDims2D<W, H> dims{};
        static constexpr ND NDim{ ND::_2D };
    };

    template<typename T, int W, int H, int P>
    struct StaticRawPtr<StaticPtrDims3D<W, H, P>, T> {
        using type = T;
        T data[P][H][W];
        static constexpr StaticPtrDims3D<W, H, P> dims{};
        static constexpr ND NDim{ ND::_3D };
    };

    template <enum ND D>
    struct PtrAccessor;

    template <>
    struct PtrAccessor<ND::_1D> {
        template <typename T, typename BiggerType = T>
        FK_HOST_DEVICE_FUSE const BiggerType* cr_point(const Point& p, const RawPtr<ND::_1D, T>& ptr) {
            return ((const BiggerType*)ptr.data) + p.x;
        }

        template <typename T, typename BiggerType = T>
        FK_HOST_DEVICE_STATIC BiggerType* point(const Point& p, const RawPtr<ND::_1D, T>& ptr) {
            return (BiggerType*)ptr.data + p.x;
        }
    };

    template <>
    struct PtrAccessor<ND::_2D> {
        template <typename T, typename BiggerType = T>
        FK_HOST_DEVICE_FUSE const BiggerType* cr_point(const Point& p, const RawPtr<ND::_2D, T>& ptr) {
            return (const BiggerType*)((const char*)ptr.data + (p.y * ptr.dims.pitch)) + p.x;
        }

        template <typename T, typename BiggerType = T>
        FK_HOST_DEVICE_STATIC BiggerType* point(const Point& p, const RawPtr<ND::_2D, T>& ptr) {
            return (BiggerType*)((char*)ptr.data + (p.y * ptr.dims.pitch)) + p.x;
        }
    };

    template <>
    struct PtrAccessor<ND::_3D> {
        template <typename T, typename BiggerType = T>
        FK_HOST_DEVICE_FUSE const BiggerType* cr_point(const Point& p, const RawPtr<ND::_3D, T>& ptr) {
            return (const BiggerType*)((const char*)ptr.data + (ptr.dims.plane_pitch * ptr.dims.color_planes * p.z) + (p.y * ptr.dims.pitch)) + p.x;
        }

        template <typename T, typename BiggerType = T>
        FK_HOST_DEVICE_STATIC BiggerType* point(const Point& p, const RawPtr<ND::_3D, T>& ptr) {
            return (BiggerType*)((char*)ptr.data + (ptr.dims.plane_pitch * ptr.dims.color_planes * p.z) + (p.y * ptr.dims.pitch)) + p.x;
        }
    };

    template <>
    struct PtrAccessor<ND::T3D> {
        template <typename T, typename BiggerType = T>
        FK_HOST_DEVICE_FUSE const BiggerType* cr_point(const Point& p, const RawPtr<ND::T3D, T>& ptr, const uint& color_plane = 0) {
            return (const BiggerType*)((const char*)ptr.data + (color_plane * ptr.dims.color_planes_pitch) + (ptr.dims.plane_pitch * p.z) + (ptr.dims.pitch * p.y)) + p.x;
        }

        template <typename T, typename BiggerType = T>
        FK_HOST_DEVICE_STATIC BiggerType* point(const Point& p, const RawPtr<ND::T3D, T>& ptr, const uint& color_plane = 0) {
            return (BiggerType*)((char*)ptr.data + (color_plane * ptr.dims.color_planes_pitch) + (ptr.dims.plane_pitch * p.z) + (ptr.dims.pitch * p.y)) + p.x;
        }
    };

    template<enum ND D>
    struct StaticPtrAccessor;

    template<>
    struct StaticPtrAccessor<ND::_1D> {
        template <int W, typename T>
        FK_HOST_DEVICE_FUSE T read(const Point& p, const StaticRawPtr<StaticPtrDims1D<W>, T>& ptr) {
            return ptr.data[p.x];
        }

        template <int W, typename T>
        FK_HOST_DEVICE_FUSE void write(const Point& p, StaticRawPtr<StaticPtrDims1D<W>, T>& ptr, const T& value) {
            ptr.data[p.x] = value;
        }
    };

    template<>
    struct StaticPtrAccessor<ND::_2D> {
        template <int W, int H, typename T>
        FK_HOST_DEVICE_FUSE T read(const Point& p, const StaticRawPtr<StaticPtrDims2D<W, H>, T>& ptr) {
            return ptr.data[p.y][p.x];
        }

        template <int W, int H, typename T>
        FK_HOST_DEVICE_FUSE void write(const Point& p, StaticRawPtr<StaticPtrDims2D<W, H>, T>& ptr, const T& value) {
            ptr.data[p.y][p.x] = value;
        }
    };

    template<>
    struct StaticPtrAccessor<ND::_3D> {
        template <int W, int H, int P, typename T>
        FK_HOST_DEVICE_FUSE T read(const Point& p, const StaticRawPtr<StaticPtrDims3D<W, H, P>, T>& ptr) {
            return ptr.data[p.z][p.y][p.x];
        }

        template <int W, int H, int P, typename T>
        FK_HOST_DEVICE_FUSE void write(const Point& p, StaticRawPtr<StaticPtrDims3D<W, H, P>, T>& ptr, const T& value) {
            ptr.data[p.z][p.y][p.x] = value;
        }
    };

    template <typename StaticRawPtr>
    struct StaticPtr {
        using Type = typename StaticRawPtr::type;
        using At = StaticPtrAccessor<static_cast<ND>(StaticRawPtr::NDim)>;
        StaticRawPtr ptr_a;
        inline constexpr StaticRawPtr ptr() const {
            return ptr_a;
        }
        inline constexpr auto dims() const {
            return ptr_a.dims;
        }
    };
}

#endif // FK_RAWPTR_H