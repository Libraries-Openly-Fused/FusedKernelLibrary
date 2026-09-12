/* Copyright 2025 Grup Mediapro S.L.U (Oscar Amoros Huguet)
   Copyright 2026 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_VECTOR_TYPES
#define FK_VECTOR_TYPES

#include <fused_kernel/core/utils/type_lists.h>
#include <fused_kernel/core/utils/utils.h>

namespace fk {

    struct Bool1 {
        bool x;
        // Making it easier to evalute in expressions that expect a bool
        FK_HOST_DEVICE_CNST operator bool() const { return x; }
    };

    struct Bool2 {
        bool x, y;
        // Making it easier to evalute in expressions that expect a bool
        FK_HOST_DEVICE_CNST operator bool() const { return x && y; }
    };

    struct Bool3 {
        bool x, y, z;
        // Making it easier to evalute in expressions that expect a bool
        FK_HOST_DEVICE_CNST operator bool() const { return x && y && z; }
    };

    struct Bool4 {
        bool x, y, z, w;
        // Making it easier to evalute in expressions that expect a bool
        FK_HOST_DEVICE_CNST operator bool() const { return x && y && z && w; }
    };

    struct Bool {
        // Utils to convert to single bool
        // Static to increase chances of inlining, specially important in GPUs
        FK_HOST_DEVICE_FUSE bool vAnd(const Bool1& bool1) { return bool1.x; }
        FK_HOST_DEVICE_FUSE bool vAnd(const Bool2& bool2) { return bool2.x && bool2.y; }
        FK_HOST_DEVICE_FUSE bool vAnd(const Bool3& bool3) { return bool3.x && bool3.y && bool3.z; }
        FK_HOST_DEVICE_FUSE bool vAnd(const Bool4& bool4) { return bool4.x && bool4.y && bool4.z && bool4.w; }
        FK_HOST_DEVICE_FUSE bool vOr(const Bool1 & bool1) { return bool1.x; }
        FK_HOST_DEVICE_FUSE bool vOr(const Bool2 & bool2) { return bool2.x || bool2.y; }
        FK_HOST_DEVICE_FUSE bool vOr(const Bool3 & bool3) { return bool3.x || bool3.y || bool3.z; }
        FK_HOST_DEVICE_FUSE bool vOr(const Bool4 & bool4) { return bool4.x || bool4.y || bool4.z || bool4.w; }
    };

    using bool1 = Bool1;
    using bool2 = Bool2;
    using bool3 = Bool3;
    using bool4 = Bool4;

    struct Char1 {
        signed char x;
    };

    struct Uchar1 {
        uchar x;
    };

    struct alignas(2) Char2 {
        signed char x, y;
    };

    struct alignas(2) Uchar2 {
        uchar x, y;
    };

    struct Char3 {
       signed char x, y, z;
    };

    struct Uchar3 {
       uchar x, y, z;
    };

    struct alignas(4) Char4 {
        signed char x, y, z, w;
    };

    struct alignas(4) Uchar4 {
        uchar x, y, z, w;
    };

    struct Short1 {
        short x;
    };

    struct Ushort1 {
        ushort x;
    };

    struct alignas(4) Short2 {
        short x, y;
    };

    struct alignas(4) Ushort2 {
        ushort x, y;
    };

    struct Short3 {
        short x, y, z;
    };

    struct Ushort3 {
        ushort x, y, z;
    };

    struct alignas(8) Short4 {
        short x, y, z, w;
    };

    struct alignas(8) Ushort4 {
        ushort x, y, z, w;
    };

    struct Int1 {
        int x;
    };

    struct Uint1 {
        unsigned int x;
    };

    struct alignas(8) Int2 {
        int x, y;
    };

    struct alignas(8) Uint2 {
        unsigned int x, y;
    };

    struct Int3 {
        int x, y, z;
    };

    struct Uint3 {
        unsigned int x, y, z;
    };

    struct alignas(16) Int4 {
        int x, y, z, w;
    };

    struct alignas(16) Uint4 {
        unsigned int x, y, z, w;
    };

    struct Long1 {
        long int x;
    };

    struct Ulong1 {
        ulong x;
    };

    struct alignas(2 * sizeof(long int)) Long2 {
        long int x, y;
    };

    struct alignas(2 * sizeof(unsigned long int)) Ulong2 {
        ulong x, y;
    };

    struct Long3 {
        long int x, y, z;
    };

    struct Ulong3 {
        ulong x, y, z;
    };

    struct alignas(16) Long4 {
        long int x, y, z, w;
    };

    struct alignas(16) Ulong4 {
        ulong x, y, z, w;
    };

    struct Float1 {
        float x;
    };

    struct alignas(8) Float2 {
        float x, y;
    };

    struct Float3 {
        float x, y, z;
    };

    struct alignas(16) Float4 {
        float x, y, z, w;
    };

    struct Longlong1 {
        long long int x;
    };

    struct Ulonglong1 {
        ulonglong x;
    };

    struct alignas(16) Longlong2 {
        long long int x, y;
    };

    struct alignas(16) Ulonglong2 {
        ulonglong x, y;
    };

    struct Longlong3 {
        long long int x, y, z;
    };

    struct Ulonglong3 {
        ulonglong x, y, z;
    };

    struct alignas(16) Longlong4 {
        long long int x, y, z, w;
    };

    struct alignas(16) Ulonglong4 {
        ulonglong x, y, z, w;
    };

    struct Double1 {
        double x;
    };

    struct alignas(16) Double2 {
        double x, y;
    };

    struct alignas(16) Double3 {
        double x, y, z;
    };

    struct alignas(16) Double4 {
        double x, y, z, w;
    };

    using char1 = Char1;
    using uchar1 = Uchar1;
    using char2 = Char2;
    using uchar2 = Uchar2;
    using char3 = Char3;
    using uchar3 = Uchar3;
    using char4 = Char4;
    using uchar4 = Uchar4;
    using short1 = Short1;
    using ushort1 = Ushort1;
    using short2 = Short2;
    using ushort2 = Ushort2;
    using short3 = Short3;
    using ushort3 = Ushort3;
    using short4 = Short4;
    using ushort4 = Ushort4;
    using int1 = Int1;
    using uint1 = Uint1;
    using int2 = Int2;
    using uint2 = Uint2;
    using int3 = Int3;
    using uint3 = Uint3;
    using int4 = Int4;
    using uint4 = Uint4;
    using long1 = Long1;
    using ulong1 = Ulong1;
    using long2 = Long2;
    using ulong2 = Ulong2;
    using long3 = Long3;
    using ulong3 = Ulong3;
    using long4 = Long4;
    using ulong4 = Ulong4;
    using float1 = Float1;
    using float2 = Float2;
    using float3 = Float3;
    using float4 = Float4;
    using longlong1 = Longlong1;
    using ulonglong1 = Ulonglong1;
    using longlong2 = Longlong2;
    using ulonglong2 = Ulonglong2;
    using longlong3 = Longlong3;
    using ulonglong3 = Ulonglong3;
    using longlong4 = Longlong4;
    using ulonglong4 = Ulonglong4;
    using double1 = Double1;
    using double2 = Double2;
    using double3 = Double3;
    using double4 = Double4;
} // namespace fk

#if !defined(__HIPCC__) && !defined(__CUDACC__) && !defined(__NVCC__) && !defined(__VECTOR_TYPES_H__) && \
    !defined(HIP_INCLUDE_HIP_HIP_VECTOR_TYPES_H)
#define FK_EXPORT_VECTOR_TYPES(BaseType) \
    using fk::BaseType##1; \
    using fk::BaseType##2; \
    using fk::BaseType##3; \
    using fk::BaseType##4;

FK_EXPORT_VECTOR_TYPES(bool)
FK_EXPORT_VECTOR_TYPES(char)
FK_EXPORT_VECTOR_TYPES(uchar)
FK_EXPORT_VECTOR_TYPES(short)
FK_EXPORT_VECTOR_TYPES(ushort)
FK_EXPORT_VECTOR_TYPES(int)
FK_EXPORT_VECTOR_TYPES(uint)
FK_EXPORT_VECTOR_TYPES(long)
FK_EXPORT_VECTOR_TYPES(ulong)
FK_EXPORT_VECTOR_TYPES(longlong)
FK_EXPORT_VECTOR_TYPES(ulonglong)
FK_EXPORT_VECTOR_TYPES(float)
FK_EXPORT_VECTOR_TYPES(double)

#undef FK_EXPORT_VECTOR_TYPES
#endif

#endif /* FK_VECTOR_TYPES */