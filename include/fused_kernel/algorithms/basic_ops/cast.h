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

#ifndef FK_CAST
#define FK_CAST

#if defined(__NVCC__)
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#endif

#include <fused_kernel/core/execution_model/operation_model/operation_model.h>
#include <fused_kernel/core/constexpr_libs/constexpr_cmath.h>

namespace fk {
    template <typename I, typename O>
    struct Cast {
    private:
        using SelfType = Cast<I, O>;
    public:
        FK_STATIC_STRUCT(Cast, SelfType)
        using Parent = UnaryOperation<I, O, Cast<I, O>>;
        DECLARE_UNARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType input) {
            return cxp::cast<OutputType>::f(input);
        }
    };

#if defined(__NVCC__)
    template <>
    struct Cast<__half, float> {
    private:
        using SelfType = Cast<__half, float>;
    public:
        FK_STATIC_STRUCT(Cast, SelfType)
        using Parent = UnaryOperation<__half, float, SelfType>;
        DECLARE_UNARY_PARENT
        FK_HOST_DEVICE_STATIC OutputType exec(const InputType input) {
#if defined(__CUDA_ARCH__)
            return __half2float(input);
#else
            return static_cast<float>(input);
#endif
        }
    };

    template <>
    struct Cast<float, __half> {
    private:
        using SelfType = Cast<float, __half>;
    public:
        FK_STATIC_STRUCT(Cast, SelfType)
        using Parent = UnaryOperation<float, __half, SelfType>;
        DECLARE_UNARY_PARENT
        FK_HOST_DEVICE_STATIC OutputType exec(const InputType input) {
#if defined(__CUDA_ARCH__)
            return __float2half(input);
#else
            return static_cast<__half>(input);
#endif
        }
    };

    template <>
    struct Cast<__nv_bfloat16, float> {
    private:
        using SelfType = Cast<__nv_bfloat16, float>;
    public:
        FK_STATIC_STRUCT(Cast, SelfType)
        using Parent = UnaryOperation<__nv_bfloat16, float, SelfType>;
        DECLARE_UNARY_PARENT
        FK_HOST_DEVICE_STATIC OutputType exec(const InputType input) {
#if defined(__CUDA_ARCH__)
            return __bfloat162float(input);
#else
            return static_cast<float>(input);
#endif
        }
    };

    template <>
    struct Cast<float, __nv_bfloat16> {
    private:
        using SelfType = Cast<float, __nv_bfloat16>;
    public:
        FK_STATIC_STRUCT(Cast, SelfType)
        using Parent = UnaryOperation<float, __nv_bfloat16, SelfType>;
        DECLARE_UNARY_PARENT
        FK_HOST_DEVICE_STATIC OutputType exec(const InputType input) {
#if defined(__CUDA_ARCH__)
            return __float2bfloat16(input);
#else
            return static_cast<__nv_bfloat16>(input);
#endif
        }
    };
#endif
} // namespace fk

#endif
