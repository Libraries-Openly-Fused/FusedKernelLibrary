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

#ifndef FK_LOGICAL
#define FK_LOGICAL

#include <fused_kernel/core/execution_model/operation_model/operation_model.h>
#include <fused_kernel/core/data/tuple.h>
#include <fused_kernel/core/constexpr_libs/constexpr_cmath.h>

namespace fk {
    template <typename I>
    struct IsEven {
        static_assert(std::is_integral_v<VBase<I>>, "IsEven only works with integral and integral vector types");
    private:
        using SelfType = IsEven<I>;
    public:
        FK_STATIC_STRUCT(IsEven, SelfType)
        using Parent = UnaryOperation<I, bool, IsEven<I>>;
        DECLARE_UNARY_PARENT
        FK_HOST_DEVICE_FUSE auto exec(const InputType& input) {
            return cxp::is_even(input);
        }
    };

    template <typename I, typename P = I, typename O = I, typename IType = BinaryType>
    struct MaxBase {
    private:
        using SelfType = MaxBase<I, P, O, IType>;
    public:
        FK_STATIC_STRUCT(MaxBase, SelfType)
        using Parent = BinaryOperation<I, P, O, MaxBase<I, P, O, BinaryType>>;
        DECLARE_BINARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& params) {
            static_assert(!validCUDAVec<I> && !validCUDAVec<P> && !validCUDAVec<O>,
                "Max_ can't work with cuda vector types.");
            return cxp::max(input, params);
        }
    };

    template <typename I, typename P, typename O>
    struct MaxBase<I, P, O, UnaryType> {
    private:
        using SelfType = MaxBase<I, P, O, UnaryType>;
    public:
        FK_STATIC_STRUCT(MaxBase, SelfType)
        using Parent = UnaryOperation<Tuple<I, P>, O, MaxBase<I, P, O, UnaryType>>;
        DECLARE_UNARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            static_assert(!validCUDAVec<I> && !validCUDAVec<P> && !validCUDAVec<O>,
                "Max_ can't work with cuda vector types.");
            return cxp::max(get<0>(input), get<1>(input));
        }
    };

    template <typename I, typename P = I, typename O = I, typename IType = BinaryType>
    struct Max {
    private:
        using SelfType = Max<I, P, O, IType>;
    public:
        FK_STATIC_STRUCT(Max, SelfType)
        using Parent = BinaryOperation<I, P, O, Max<I, P, O, BinaryType>>;
        DECLARE_BINARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& params) {
            return BinaryV<MaxBase<VBase<I>, VBase<P>, VBase<O>>, I, P, O>::exec(input, params);
        }
    };

    template <typename I, typename P, typename O>
    struct Max<I, P, O, UnaryType> {
    private:
        using SelfType = Max<I, P, O, UnaryType>;
    public:
        FK_STATIC_STRUCT(Max, SelfType)
        using Parent = UnaryOperation<Tuple<I, P>, O, Max<I, P, O, UnaryType>>;
        DECLARE_UNARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            return UnaryV<MaxBase<VBase<I>, VBase<P>, VBase<O>, UnaryType>, Tuple<I, P>, O>::exec(input);
        }
    };

    template <typename I, typename P = I, typename O = I, typename IType = BinaryType>
    struct MinBase {
    private:
        using SelfType = MinBase<I, P, O, IType>;
    public:
        FK_STATIC_STRUCT(MinBase, SelfType)
        using Parent = BinaryOperation<I, P, O, MinBase<I, P, O, BinaryType>>;
        DECLARE_BINARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& params) {
            static_assert(!validCUDAVec<I> && !validCUDAVec<P> && !validCUDAVec<O>,
                "Min_ can't work with cuda vector types.");
            return cxp::min(input, params);
        }
    };

    template <typename I, typename P, typename O>
    struct MinBase<I, P, O, UnaryType> {
    private:
        using SelfType = MinBase<I, P, O, UnaryType>;
    public:
        FK_STATIC_STRUCT(MinBase, SelfType)
        using Parent = UnaryOperation<Tuple<I, P>, O, MinBase<I, P, O, UnaryType>>;
        DECLARE_UNARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            static_assert(!validCUDAVec<I> && !validCUDAVec<P> && !validCUDAVec<O>,
                "Min_ can't work with cuda vector types.");
            return cxp::min(get<0>(input), get<1>(input));
        }
    };

    template <typename I, typename P = I, typename O = I, typename IType = BinaryType>
    struct Min {
    private:
        using SelfType = Min<I, P, O, IType>;
    public:
        FK_STATIC_STRUCT(Min, SelfType)
        using Parent = BinaryOperation<I, P, O, Min<I, P, O, BinaryType>>;
        DECLARE_BINARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& params) {
            return BinaryV<MinBase<VBase<I>, VBase<P>, VBase<O>>, I, P, O>::exec(input, params);
        }
    };

    template <typename I, typename P, typename O>
    struct Min<I, P, O, UnaryType> {
    private:
        using SelfType = Min<I, P, O, UnaryType>;
    public:
        FK_STATIC_STRUCT(Min, SelfType)
        using Parent = UnaryOperation<Tuple<I, P>, O, Min<I, P, O, UnaryType>>;
        DECLARE_UNARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            return UnaryV<MinBase<VBase<I>, VBase<P>, VBase<O>, UnaryType>, Tuple<I, P>, O>::exec(input);
        }
    };

    template <typename I1, typename I2=I1>
    struct Equal {
    private:
        using SelfType = Equal<I1, I2>;
    public:
        FK_STATIC_STRUCT(Equal, SelfType)
        using Parent = UnaryOperation<Tuple<I1, I2>, bool, Equal<I1, I2>>;
        DECLARE_UNARY_PARENT
        template <int N = cn<I1>>
        FK_HOST_DEVICE_FUSE std::enable_if_t<N==1, OutputType> exec(const InputType& input) {
            return get<0>(input) == get<1>(input);
        }
        template <int N = cn<I1>>
        FK_HOST_DEVICE_FUSE std::enable_if_t<N == 2, OutputType> exec(const InputType& input) {
            const auto result = get<0>(input) == get<1>(input);
            return result.x && result.y;
        }
        template <int N = cn<I1>>
        FK_HOST_DEVICE_FUSE std::enable_if_t<N == 3, OutputType> exec(const InputType& input) {
            const auto result = get<0>(input) == get<1>(input);
            return result.x && result.y && result.z;
        }
        template <int N = cn<I1>>
        FK_HOST_DEVICE_FUSE std::enable_if_t<N == 4, OutputType> exec(const InputType& input) {
            const auto result = get<0>(input) == get<1>(input);
            return result.x && result.y && result.z && result.w;
        }
    };
} //namespace fk

#endif
