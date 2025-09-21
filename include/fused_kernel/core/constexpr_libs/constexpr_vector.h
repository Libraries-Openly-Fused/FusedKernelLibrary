/* Copyright 2025 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef CXP_CONSTEXPR_VECTOR_H
#define CXP_CONSTEXPR_VECTOR_H

#include <fused_kernel/core/utils/static_get.h>
#include <fused_kernel/core/execution_model/operation_model/operation_types.h>

namespace cxp {
    struct vector_and {
        template <typename T>
        FK_HOST_DEVICE_FUSE bool f(const T& value) {
            if constexpr (fk::validCUDAVec<T>) {
                const auto valBool = cast<fk::VectorType_t<bool, fk::cn<T>>>::f(value);
                if constexpr (fk::cn<T> == 1) {
                    return valBool;
                } else if constexpr (fk::cn<T> == 2) {
                    return valBool.x && valBool.y;
                } else if constexpr (fk::cn<T> == 3) {
                    return valBool.x && valBool.y && valBool.z;
                } else {
                    return valBool.x && valBool.y && valBool.z && valBool.w;
                }
            } else {
                return static_cast<bool>(value);
            }
        }
    };

    template <size_t NewNumChannels>
    struct discard {
    private:
        template <size_t... Idx, typename I>
        FK_HOST_DEVICE_FUSE auto f_helper(const std::index_sequence<Idx...>&,
                                          const I& input) {
            return fk::VectorType_t<fk::VBase<I>, NewNumChannels>{fk::static_get<Idx>::f(input)...};
        }

    public:
        template <typename I>
        FK_HOST_DEVICE_FUSE auto f(const I& input)
            -> std::enable_if_t<(fk::cn<I> >= 2) &&
                                (NewNumChannels < fk::cn<I>),
                                fk::VectorType_t<fk::VBase<I>, NewNumChannels>> {
            return f_helper<NewNumChannels>(std::make_index_sequence<NewNumChannels>{}, input);
        }
        template <typename I>
        FK_HOST_DEVICE_FUSE auto f(const I& input)
            -> std::enable_if_t<(NewNumChannels == fk::cn<I>), I> {
            return input;
        }
    };

    template <size_t... Idx>
    struct vector_reorder {
        template <typename VT>
        FK_HOST_DEVICE_FUSE VT f(const VT& v) {
            static_assert(fk::validCUDAVec<VT>, "Non valid CUDA vetor type: vector_reorder");
            static_assert(fk::cn<VT> >= 2, "Minimum number of channels is 2: vector_reorder");
            static_assert(sizeof...(Idx) == fk::cn<VT>, "Number of indices must match number of channels");
            return {fk::static_get<Idx>::f(v)...};
        }
    };

    template <typename Op>
    struct vector_reduce {
        template <typename VT>
        FK_HOST_DEVICE_FUSE auto f(const VT& v) {
            if constexpr (std::is_same_v<typename Op::InstanceType, fk::UnaryType>) {
                using T1 = fk::get_type_t<0, typename Op::InputType>;
                using T2 = fk::get_type_t<1, typename Op::InputType>;
                if constexpr (fk::cn<VT> == 1) {
                    if constexpr (fk::validCUDAVec<VT>) {
                        return v.x;
                    } else {
                        return v;
                    }
                } else if constexpr (fk::cn<VT> == 2) {
                    return Op::exec({ static_cast<T1>(v.x), static_cast<T2>(v.y) });
                } else if constexpr (fk::cn<VT> == 3) {
                    const auto firstR = Op::exec({ static_cast<T1>(v.x), static_cast<T2>(v.y) });
                    return Op::exec({ static_cast<T1>(firstR), static_cast<T2>(v.z) });
                } else if constexpr (fk::cn<VT> == 4) {
                    const auto firstR  = Op::exec({ static_cast<T1>(v.x),     static_cast<T2>(v.y) });
                    const auto secondR = Op::exec({ static_cast<T1>(firstR),  static_cast<T2>(v.z)});
                    return Op::exec({ static_cast<T1>(secondR), static_cast<T2>(v.w) });
                }
            } else if constexpr (std::is_same_v<typename Op::InstanceType, fk::BinaryType>) {
                if constexpr (fk::cn<VT> == 1) {
                    if constexpr (fk::validCUDAVec<VT>) {
                        return v.x;
                    } else {
                        return v;
                    }
                } else if constexpr (fk::cn<VT> == 2) {
                    return Op::exec(v.x, v.y);
                } else if constexpr (fk::cn<VT> == 3) {
                    return Op::exec(Op::exec(v.x, v.y), v.z);
                } else if constexpr (fk::cn<VT> == 4) {
                    return Op::exec(Op::exec(Op::exec(v.x, v.y), v.z), v.w);
                }
            }
        }
    };
} // namespace cxp

#endif // CXP_CONSTEXPR_VECTOR_H