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

#include <fused_kernel/core/constexpr_libs/constexpr_cmath.h>
#include <fused_kernel/core/utils/vector_utils.h>

namespace cxp {
    template <typename Op, typename = void>
    struct Exec;

    template <typename Op>
    struct Exec<Op, std::enable_if_t<std::is_same_v<typename Op::InstanceType, fk::UnaryType>>> {
        template <typename ST>
        FK_HOST_DEVICE_FUSE auto exec(const ST& s)
            -> std::enable_if_t<std::is_fundamental_v<ST>> {
            return Op::exec(s);
        }
        template <typename VT>
        FK_HOST_DEVICE_FUSE auto exec(const VT& v)
            -> std::enable_if_t<fk::validCUDAVec<VT>,
                                fk::VectorType_t<decltype(Op::exec(std::declval<fk::VBase<VT>>())), fk::cn<VT>>> {
            using OT = fk::VectorType_t<decltype(Op::exec(std::declval<fk::VBase<VT>>())), fk::cn<VT>>;
            if constexpr (fk::cn<VT> == 1) {
                return OT{ Op::exec(v.x) };
            } else if constexpr (fk::cn<VT> == 2) {
                return OT{ Op::exec(v.x), Op::exec(v.y) };
            } else if constexpr (fk::cn<VT> == 3) {
                return OT{ Op::exec(v.x), Op::exec(v.y), Op::exec(v.z) };
            } else {
                return OT{ Op::exec(v.x), Op::exec(v.y), Op::exec(v.z), Op::exec(v.w) };
            }
        }
    };

    template <typename Op>
    struct Exec<Op, std::enable_if_t<std::is_same_v<typename Op::InstanceType, fk::BinaryType>>> {
        template <typename ST1, typename ST2>
        FK_HOST_DEVICE_FUSE auto exec(const ST1& s1, const ST2& s2)
            -> std::enable_if_t<fk::AreSS<ST1, ST2>::value, decltype(Op::exec(std::declval<ST1>(), std::declval<ST2>()))> {
            return Op::exec(s1, s2);
        }
        template <typename ST1, typename ST2, typename... STs>
        FK_HOST_DEVICE_FUSE auto exec(const ST1& s1, const ST2& s2, const STs&... scals)
            -> std::enable_if_t<std::is_fundamental_v<ST1> && std::is_fundamental_v<ST2> && (std::is_fundamental_v<STs> && ...)> {
            return exec(exec(s1, s2), scals...);
        }
        template <typename VT>
        FK_HOST_DEVICE_FUSE auto exec(const VT& v) 
            -> std::enable_if_t<fk::validCUDAVec<VT>,
                                decltype(Op::exec(std::declval<fk::VBase<VT>>(), std::declval<fk::VBase<VT>>()))> {
            if constexpr (fk::cn<VT> == 1) {
                return v.x;
            } else if constexpr (fk::cn<VT> == 2) {
                return Op::exec(v.x, v.y);
            } else if constexpr (fk::cn<VT> == 3) {
                return Op::exec(Op::exec(v.x, v.y), v.z);
            } else {
                return Op::exec(Op::exec(Op::exec(v.x, v.y), v.z), v.w);
            }
        }
        template <typename VT, typename ST>
        FK_HOST_DEVICE_FUSE auto exec(const VT& v, const ST& s)
            -> std::enable_if_t<fk::AreVS<VT, ST>::value, decltype(Op::exec(std::declval<fk::VBase<VT>>(), std::declval<ST>()))> {
            using BaseO = decltype(Op::exec(std::declval<fk::VBase<VT>>(), std::declval<ST>()));
            if constexpr (fk::cn<VT> == 1) {
                return fk::VectorType_t<BaseO, 1>{ Op::exec(v.x, s) };
            } else if constexpr (fk::cn<VT> == 2) {
                return fk::VectorType_t<BaseO, 2>{ Op::exec(v.x, s), Op::exec(v.y, s) };
            } else if constexpr (fk::cn<VT> == 3) {
                return fk::VectorType_t<BaseO, 3>{ Op::exec(v.x, s), Op::exec(v.y, s), Op::exec(v.z, s) };
            } else {
                return fk::VectorType_t<BaseO, 4>{ Op::exec(v.x, s), Op::exec(v.y, s), Op::exec(v.z, s), Op::exec(v.w, s) };
            }
        }
        template <typename ST, typename VT>
        FK_HOST_DEVICE_FUSE auto exec(const ST& s, const VT& v)
            -> std::enable_if_t<fk::AreSV<ST, VT>::value, decltype(Op::exec(std::declval<ST>(), std::declval<fk::VBase<VT>>()))> {
            using BaseO = decltype(Op::exec(std::declval<ST>(), std::declval<fk::VBase<VT>>()));
            if constexpr (fk::cn<VT> == 1) {
                return fk::VectorType_t<BaseO, 1>{ Op::exec(s, v.x) };
            } else if constexpr (fk::cn<VT> == 2) {
                return fk::VectorType_t<BaseO, 2>{ Op::exec(s, v.x), Op::exec(s, v.y) };
            } else if constexpr (fk::cn<VT> == 3) {
                return fk::VectorType_t<BaseO, 3>{ Op::exec(s, v.x), Op::exec(s, v.y), Op::exec(s, v.z) };
            } else {
                return fk::VectorType_t<BaseO, 4>{ Op::exec(s, v.x), Op::exec(s, v.y), Op::exec(s, v.z), Op::exec(s, v.w) };
            }
        }
        template <typename VT1, typename VT2>
        FK_HOST_DEVICE_FUSE auto exec(const VT1& v1, const VT2& v2)
            -> std::enable_if_t<fk::AreVVEqCN<VT1, VT2>::value,
                                decltype(Op::exec(std::declval<fk::VBase<VT1>>(), std::declval<fk::VBase<VT2>>()))> {
            using BaseO = decltype(Op::exec(std::declval<fk::VBase<VT1>>(), std::declval<fk::VBase<VT2>>()));
            if constexpr (fk::cn<VT1> == 1) {
                return fk::VectorType_t<BaseO, 1>{ Op::exec(v1.x, v2.x) };
            } else if constexpr (fk::cn<VT1> == 2) {
                return fk::VectorType_t<BaseO, 2>{ Op::exec(v1.x, v2.x), Op::exec(v1.y, v2.y) };
            } else if constexpr (fk::cn<VT1> == 3) {
                return fk::VectorType_t<BaseO, 3>{ Op::exec(v1.x, v2.x), Op::exec(v1.y, v2.y), Op::exec(v1.z, v2.z) };
            } else {
                return fk::VectorType_t<BaseO, 4>{ Op::exec(v1.x, v2.x), Op::exec(v1.y, v2.y), Op::exec(v1.z, v2.z), Op::exec(v1.w, v2.w) };
            }
        }
    };

    template <size_t Idx, typename T>
    FK_HOST_DEVICE_CNST typename fk::VBase<T> v_get(const T& vecVal) {
        static_assert(fk::validCUDAVec<T>, "v_get can only be used with vector types");
        static_assert(Idx < fk::cn<T>, "Index out of bounds");
        return fk::vectorAt(Idx, vecVal);
    }

    namespace internal {
        template <typename TargetT, typename SourceT, size_t... Idx>
        FK_HOST_DEVICE_CNST TargetT v_static_cast_helper(const SourceT& source, const std::index_sequence<Idx...>&) {
            return fk::make_<TargetT>(static_cast<fk::VBase<TargetT>>(v_get<Idx>(source))...);
        }
    } // namespace internal
#define CXP_F_FUNC \
public: \
    template <typename... Types> \
    FK_HOST_DEVICE_FUSE auto f(const Types&... vals) { \
        return Exec<BaseFunc>::exec(vals...); \
    }

    struct Sum {
    private:
        struct BaseFunc {
            using InstanceType = fk::BinaryType;
            template <typename ST1, typename ST2>
            FK_HOST_DEVICE_FUSE auto exec(const ST1& s1, const ST2& s2) {
                return s1 + s2;
            }
        };
        CXP_F_FUNC
    };

#undef CXP_F_FUNC

    template <typename T>
    FK_HOST_DEVICE_CNST auto v_sum(const T& vecVal) {
        static_assert(fk::validCUDAVec<T>, "v_sum can only be used with vector types");
        return Sum::f(vecVal);
    }

    template <typename TargetT, typename SourceT>
    FK_HOST_DEVICE_CNST TargetT v_static_cast(const SourceT& source) {
        if constexpr (std::is_same_v<TargetT, SourceT>) {
            return source;
        } else if constexpr (fk::validCUDAVec<SourceT>) {
            static_assert(fk::cn<TargetT> == fk::cn<SourceT>, "Can not cast to a type with different number of channels");
            return internal::v_static_cast_helper<TargetT>(source, std::make_index_sequence<fk::cn<SourceT>>{});
        } else {
            static_assert(!fk::validCUDAVec<TargetT> || (fk::cn<TargetT> == 1),
                "Can not convert a fundamental type to a vetor type with more than one channel");
            if constexpr (fk::validCUDAVec<TargetT>) {
                return fk::make_<TargetT>(static_cast<fk::VBase<TargetT>>(source));
            } else {
                return static_cast<TargetT>(source);
            }
        }
    }

    template <typename T>
    FK_HOST_DEVICE_CNST bool v_and(const T& value) {
        const auto valBool = v_static_cast<fk::VectorType_t<bool, fk::cn<T>>>(value);
        return valBool;
    }

    namespace internal {
        template <typename T>
        FK_HOST_DEVICE_CNST auto is_even_helper(const T& value)
            -> std::enable_if_t<std::is_integral_v<T>, bool> {
            return value & 1 == 0;
        }
        template <size_t... Idx, typename T>
        FK_HOST_DEVICE_CNST auto is_even_helper(const std::index_sequence<Idx...>&, const T& value)
            -> std::enable_if_t<fk::validCUDAVec<T>, fk::VectorType_t<bool, fk::cn<T>>> {
            return fk::make_<fk::VectorType_t<bool, fk::cn<T>>>(is_even(v_get<Idx>(value))...);
        }
    }

    template <typename T>
    FK_HOST_DEVICE_CNST auto is_even(const T& value)
        -> std::enable_if_t<fk::validCUDAVec<T>
                            && std::is_integral_v<fk::VBase<T>>,
                                fk::VectorType_t<bool, fk::cn<T>>> {
        return internal::is_even_helper(std::make_index_sequence<fk::cn<T>>{}, value);
    }

    template <typename T>
    FK_HOST_DEVICE_CNST auto is_even(const T& value)
        -> std::enable_if_t<std::is_integral_v<T>, bool> {
        return internal::is_even_helper(value);
    }
 } // namespace cxp

#endif // CXP_CONSTEXPR_VECTOR_H