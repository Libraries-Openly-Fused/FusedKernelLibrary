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

#ifndef FK_CUDA_VECTOR
#define FK_CUDA_VECTOR

#include <fused_kernel/core/execution_model/operation_model/operation_model.h>
#include <fused_kernel/core/constexpr_libs/constexpr_vector.h>

namespace fk {
    template <typename I, typename O>
    struct Discard {
    private:
        using SelfType = Discard<I, O>;
    public:
        FK_STATIC_STRUCT(Discard, SelfType)
        using Parent = UnaryOperation<I, O, Discard<I, O>>;
        DECLARE_UNARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            static_assert(cn<I> > cn<O>, "Output type should at least have one channel less");
            static_assert(std::is_same_v<VBase<I>, VBase<O>>,
                "Base types should be the same");
            const auto result = cxp::discard<cn<OutputType>>::f(input);
            if constexpr (std::is_fundamental_v<OutputType>) {
                return result.x;
            } else {
                return result;
            }
        }
    };

    template <typename T, int... Idx>
    struct VectorReorder {
    private:
        using SelfType = VectorReorder<T, Idx...>;
    public:
        FK_STATIC_STRUCT(VectorReorder, SelfType)
        using Parent = UnaryOperation<T, T, VectorReorder<T, Idx...>>;
        DECLARE_UNARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            static_assert(validCUDAVec<T>, "Non valid CUDA vetor type: UnaryVectorReorder");
            static_assert(cn<T> >= 2, "Minimum number of channels is 2: UnaryVectorReorder");
            return {static_get<Idx>::f(input)...};
        }
    };

    template <typename T>
    struct VectorReorderRT {
    private:
        using SelfType = VectorReorderRT<T>;
    public:
        FK_STATIC_STRUCT(VectorReorderRT, SelfType)
        using Parent = BinaryOperation<T, VectorType_t<int, cn<T>>, T, VectorReorderRT<T>>;
        DECLARE_BINARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& params) {
            static_assert(validCUDAVec<T>, "Non valid CUDA vetor type");
            static_assert(cn<T> >= 2, "Minimum number of channels is 2");
            if constexpr (cn<T> == 2) {
                return { vector_at::f(params.x, input), vector_at::f(params.y, input) };
            } else if constexpr (cn<T> == 3) {
                return { vector_at::f(params.x, input), vector_at::f(params.y, input),
                         vector_at::f(params.z, input) };
            } else {
                return { vector_at::f(params.x, input), vector_at::f(params.y, input),
                         vector_at::f(params.z, input), vector_at::f(params.w, input) };
            }
        }
    };

    template <typename T, typename Operation>
    struct VectorReduce {
    private:
        using SelfType = VectorReduce<T, Operation>;
    public:
        FK_STATIC_STRUCT(VectorReduce, SelfType)
        using Parent = UnaryOperation<T, typename Operation::OutputType, VectorReduce<T, Operation>>;
        DECLARE_UNARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) { 
            return cxp::vector_reduce<Operation>::f(input);
        }
    };

    template <typename I, typename O>
    struct AddLast {
    private:
        using SelfType = AddLast<I, O>;
    public:
        FK_STATIC_STRUCT(AddLast, SelfType)
        using Parent = BinaryOperation<I, typename VectorTraits<I>::base, O, AddLast<I, O>>;
        DECLARE_BINARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& params) {
            static_assert(cn<I> == cn<O> -1, "Output type should have one channel more");
            static_assert(std::is_same_v<typename VectorTraits<I>::base, typename VectorTraits<O>::base>,
                "Base types should be the same");
            if constexpr (cn<I> == 1) {
                if constexpr (validCUDAVec<I>) {
                    return { input.x, params };
                } else {
                  return {input, params};
                }
            } else if constexpr (cn<I> == 2) {
              return {input.x, input.y, params};
            } else if constexpr (cn<I> == 3) {
              return {input.x, input.y, input.z, params};
            }
        }
    };

    template <typename T>
    struct VectorAnd {
    private:
        using SelfType = VectorAnd<T>;
    public:
        FK_STATIC_STRUCT(VectorAnd, SelfType)
        using Parent = UnaryOperation<T, bool, VectorAnd<T>>;
        DECLARE_UNARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
            return cxp::and::f(input);
        }
    };
} // namespace fk

#endif
