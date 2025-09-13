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

#ifndef FK_UTILS
#define FK_UTILS

#if !defined(NVRTC_COMPILER)
#include <string>
#include <stdexcept>

#if defined(__NVCC__) || CLANG_HOST || defined(__HIP__) || defined(NVRTC_ENABLED)
#include <cuda_runtime.h>
#endif

#if defined(NVRTC_ENABLED)
#include <cuda.h>
#include <nvrtc.h>
#endif
#endif // NVRTC_COMPILER

#if defined(__NVCC__) || CLANG_HOST_DEVICE || defined(__HIPCC__)
#define FK_DEVICE_FUSE __device__ __forceinline__ static constexpr
#define FK_DEVICE_CNST __device__ __forceinline__ constexpr
#define FK_HOST_DEVICE_FUSE __host__ FK_DEVICE_FUSE
#define FK_HOST_DEVICE_CNST __host__ FK_DEVICE_CNST
#define FK_HOST_FUSE __host__ __forceinline__ static constexpr
#define FK_HOST_CNST __host__ __forceinline__ constexpr
#define FK_HOST_STATIC __host__ __forceinline__ static
#define FK_HOST_INLINE __host__ __forceinline__
#define FK_HOST_DEVICE_STATIC __host__ __device__ __forceinline__ static
#define FK_RESTRICT __restrict__
#elif defined(NVRTC_COMPILER)
#define FK_DEVICE_FUSE static constexpr __forceinline__
#define FK_DEVICE_CNST constexpr __forceinline__
#define FK_HOST_DEVICE_FUSE FK_DEVICE_FUSE
#define FK_HOST_DEVICE_CNST FK_DEVICE_CNST
#define FK_HOST_FUSE static constexpr __forceinline__
#define FK_HOST_CNST constexpr __forceinline__
#define FK_HOST_STATIC static constexpr __forceinline__
#define FK_HOST_INLINE constexpr __forceinline__
#define FK_HOST_DEVICE_STATIC static __forceinline__
#define FK_RESTRICT __restrict__
#else
#define FK_DEVICE_FUSE static constexpr inline
#define FK_DEVICE_CNST constexpr inline
#define FK_HOST_DEVICE_FUSE FK_DEVICE_FUSE
#define FK_HOST_DEVICE_CNST FK_DEVICE_CNST
#define FK_HOST_FUSE static constexpr inline
#define FK_HOST_CNST constexpr inline
#define FK_HOST_STATIC static inline
#define FK_HOST_INLINE inline
#define FK_HOST_DEVICE_STATIC static inline
#ifdef _MSC_VER
#define FK_RESTRICT __restrict
#else
#define FK_RESTRICT __restrict__
#endif // _MSC_VER
#endif // __CUDACC__ || __NVCC__ || NVRTC_COMPILER || __HIPCC__

#ifdef CUDART_VERSION
#define CUDART_MAJOR_VERSION CUDART_VERSION/1000
#else
#define CUDART_MAJOR_VERSION 0 // We are not compiling with nvcc
#endif

#define FK_STATIC_STRUCT(struct_name, struct_alias) \
    public: /* Ensure deletions are in a public section (conventional) */ \
        struct_name() = delete; \
        struct_name(const struct_alias&) = delete; \
        struct_name& operator=(const struct_alias&) = delete; \
        struct_name(struct_alias&&) = delete; \
        struct_name& operator=(struct_alias&&) = delete;

using uchar = unsigned char;
using schar = signed char;
using uint = unsigned int;
using longlong = long long;
using ulonglong = unsigned long long;
using ushort = unsigned short;
using ulong = unsigned long;

#if (defined(__NVCC__) || CLANG_HOST ||defined(__HIP__) || defined(NVRTC_ENABLED)) && !defined(NVRTC_COMPILER)
namespace fk {
    inline void gpuAssert(cudaError_t code,
                          const char *file,
                          int line,
                          bool abort = true) {
        if (code != cudaSuccess) {
            std::string message = "GPU Error: ";
            message.append(cudaGetErrorString(code));
            message.append(" File: ");
            message.append(file);
            message.append(" Line:");
            message.append(std::to_string(line).c_str());
            message.append("\n");
            if (abort) throw std::runtime_error(message.c_str());
        }
    }
#if defined(NVRTC_ENABLED)
    inline void gpuAssert(CUresult code,
                          const char* file,
                          int line,
                          bool abort = true) {
        if (code != CUDA_SUCCESS) {
            std::string message = "GPU Error: ";
            const char* error_str{""};
            cuGetErrorString(code, &error_str);
            message.append(error_str);
            message.append(" File: ");
            message.append(file);
            message.append(" Line:");
            message.append(std::to_string(line).c_str());
            message.append("\n");
            if (abort) throw std::runtime_error(message.c_str());
        }
    }
    inline void gpuAssert(nvrtcResult code,
                          const char* file,
                          int line,
                          bool abort = true) {
        if (code != NVRTC_SUCCESS) {
            std::string message = "GPU Error: ";
            const char* error_str = nvrtcGetErrorString(code);
            message.append(error_str);
            message.append(" File: ");
            message.append(file);
            message.append(" Line:");
            message.append(std::to_string(line).c_str());
            message.append("\n");
            if (abort) throw std::runtime_error(message.c_str());
        }
    }
#endif // NVRTC_ENABLED
} // namespace fk

#define gpuErrchk(ans) { fk::gpuAssert((ans), __FILE__, __LINE__, true); }
#endif // (__NVCC__) || __HIP__ || NVRTC_ENABLED) && !defined(NVRTC_COMPILER)

// Null type, used for Operation required aliases that can not still be known, because they are deduced
// from a backwards operation that is till not defined.
struct NullType {};

namespace fk {
    template <typename... Types>
    struct PrintTypes;
}

#if defined(_MSC_VER)
#define _MSC_VER_EXISTS 1
#else
#define _MSC_VER_EXISTS 0
#endif
#if defined(__clang__) && defined(__CUDA__) && !defined(__CUDA_ARCH__)
// clang compiling CUDA code, host mode.
#define CLANG_HOST
#endif

#if defined(__clang__) && defined(__CUDA__) && defined(__CUDA_ARCH__)
// clang compiling CUDA code, device mode.º
#define CLANG_DEVICE
#endif

#if defined(__clang__) && defined(__CUDA__)
// clang compiling CUDA code, device mode.º
#define CLANG_HOST_DEVICE
#endif

#define VS2017_COMPILER (_MSC_VER_EXISTS && _MSC_VER >= 1910 && _MSC_VER < 1920)
#define NO_VS2017_COMPILER !VS2017_COMPILER
#endif // FK_UTILS
