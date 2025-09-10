/* Copyright 2025 Grup Mediapro S.L.U (Oscar Amoros Huguet)

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
        FK_HOST_CNST operator bool() const { return x; }
    };

    struct Bool2 {
        bool x, y;
        // Making it easier to evalute in expressions that expect a bool
        FK_HOST_CNST operator bool() const { return x && y; }
    };

    struct Bool3 {
        bool x, y, z;
        // Making it easier to evalute in expressions that expect a bool
        FK_HOST_CNST operator bool() const { return x && y && z; }
    };

    struct Bool4 {
        bool x, y, z, w;
        // Making it easier to evalute in expressions that expect a bool
        FK_HOST_CNST operator bool() const { return x && y && z && w; }
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

} // namespace fk

using bool1 = fk::Bool1;
using bool2 = fk::Bool2;
using bool3 = fk::Bool3;
using bool4 = fk::Bool4;

namespace fk {
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
#ifdef CUDA_ENABLE_ALIGNED_VECTOR4_TYPES
    struct alignas(16) Ulong4_16a {
        ulong x, y, z, w;
    };
    struct alignas(32) Ulong4_32a {
        ulong x, y, z, w;
    };
#else
    struct alignas(16) Long4 {
        long int x, y, z, w;
    };
#endif
#ifdef CUDA_ENABLE_ALIGNED_VECTOR4_TYPES
    struct alignas(16) Long4_16a {
        long int x, y, z, w;
    };
    struct alignas(32) Long4_32a {
        long int x, y, z, w;
    };
#else
    struct alignas(16) Ulong4 {
        ulong x, y, z, w;
    };
#endif
    struct Float1 {
        float x;
    };

    struct alignas(8) Float2 {
        float x, y;
    };

    struct Float3 {
        float x, y, z;
    };
#ifdef CUDA_ENABLE_ALIGNED_VECTOR4_TYPES
    struct alignas(16) Float4_16a {
        float x, y, z, w;
    };
    struct alignas(32) Float4_32a {
        float x, y, z, w;
    };
#else
    struct alignas(16) Float4 {
        float x, y, z, w;
    };
#endif
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
#ifdef CUDA_ENABLE_ALIGNED_VECTOR4_TYPES
    struct alignas(16) Longlong4_16a {
        long long int x, y, z, w;
    };
    struct alignas(32) Longlong4_32a {
        long long int x, y, z, w;
    };
#else
    struct alignas(16) Longlong4 {
        long long int x, y, z, w;
    };
#endif
#ifdef CUDA_ENABLE_ALIGNED_VECTOR4_TYPES
    struct alignas(16) Ulonglong4_16a {
        ulonglong x, y, z, w;
    };
    struct alignas(32) Ulonglong4_32a {
        ulonglong x, y, z, w;
    };
#else
    struct alignas(16) Ulonglong4 {
        ulonglong x, y, z, w;
    };
#endif
    struct Double1 {
        double x;
    };

    struct alignas(16) Double2 {
        double x, y;
    };

    struct alignas(16) Double3 {
        double x, y, z;
    };
#ifdef CUDA_ENABLE_ALIGNED_VECTOR4_TYPES
    struct alignas(16) Double4_16a {
        double x, y, z, w;
    };
    struct alignas(32) Double4_32a {
        double x, y, z, w;
    };
#else
    struct alignas(16) Double4 {
        double x, y, z, w;
    };
#endif
} // namespace fk

#if defined(__NVCC__) || defined(__HIP__) || defined(NVRTC_ENABLED) || defined(__NVRTC__)
#include <vector_types.h>
#elif defined(NVRTC_COMPILER)
// Nothing to include here
#else
using char1 = fk::Char1;
using uchar1 = fk::Uchar1;
using char2 = fk::Char2;
using uchar2 = fk::Uchar2;
using char3 = fk::Char3;
using uchar3 = fk::Uchar3;
using char4 = fk::Char4;
using uchar4 = fk::Uchar4;
using short1 = fk::Short1;
using ushort1 = fk::Ushort1;
using short2 = fk::Short2;
using ushort2 = fk::Ushort2;
using short3 = fk::Short3;
using ushort3 = fk::Ushort3;
using short4 = fk::Short4;
using ushort4 = fk::Ushort4;
using int1 = fk::Int1;
using uint1 = fk::Uint1;
using int2 = fk::Int2;
using uint2 = fk::Uint2;
using int3 = fk::Int3;
using uint3 = fk::Uint3;
using int4 = fk::Int4;
using uint4 = fk::Uint4;
using long1 = fk::Long1;
using ulong1 = fk::Ulong1;
using long2 = fk::Long2;
using ulong2 = fk::Ulong2;
using long3 = fk::Long3;
using ulong3 = fk::Ulong3;
#ifdef CUDA_ENABLE_ALIGNED_VECTOR4_TYPES
using long4_16a = fk::Long4_16a;
using long4_32a = fk::Long4_32a;
#else
using long4 = fk::Long4;
#endif
#ifdef CUDA_ENABLE_ALIGNED_VECTOR4_TYPES
using ulong4_16a = fk::Ulong4_16a;
using ulong4_32a = fk::Ulong4_32a;
#else
using ulong4 = fk::Ulong4;
#endif

using float1 = fk::Float1;
using float2 = fk::Float2;
using float3 = fk::Float3;
#ifdef CUDA_ENABLE_ALIGNED_VECTOR4_TYPES
using float4_16a = fk::Float4_16a;
using float4_32a = fk::Float4_32a;
#else
using float4 = fk::Float4;
#endif
using longlong1 = fk::Longlong1;
using ulonglong1 = fk::Ulonglong1;
using longlong2 = fk::Longlong2;
using ulonglong2 = fk::Ulonglong2;
using longlong3 = fk::Longlong3;
using ulonglong3 = fk::Ulonglong3;
#ifdef CUDA_ENABLE_ALIGNED_VECTOR4_TYPES
using longlong4_16a = fk::Longlong4_16a;
using longlong4_32a = fk::Longlong4_32a;
#else
using longlong4 = fk::Longlong4;
#endif
#ifdef CUDA_ENABLE_ALIGNED_VECTOR4_TYPES
using ulonglong4_16a = fk::Ulonglong4_16a;
using ulonglong4_32a = fk::Ulonglong4_32a;
#else
using ulonglong4 = fk::Ulonglong4;
#endif

using double1 = fk::Double1;
using double2 = fk::Double2;
using double3 = fk::Double3;
#ifdef CUDA_ENABLE_ALIGNED_VECTOR4_TYPES
using double4_16a = fk::Double4_16a;
using double4_32a = fk::Double4_32a
#else
using double4 = fk::Double4;
#endif
#endif /* __NVCC__ || __HIPCC__ */

#endif /* FK_VECTOR_TYPES */