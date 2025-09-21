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

#ifndef CXP_CONSTEXPR_VECTOR_EXEC_H
#define CXP_CONSTEXPR_VECTOR_EXEC_H

#include <fused_kernel/core/utils/vector_utils.h>
#include <fused_kernel/core/execution_model/operation_model/operation_types.h>

namespace cxp {
    template <typename Op, typename = void>
    struct Exec;

    template <typename Op>
    struct Exec<Op, std::enable_if_t<std::is_same_v<typename Op::InstanceType, fk::UnaryType>>> {
        template <typename T>
        FK_HOST_DEVICE_FUSE auto exec(const T& val) {
            if constexpr (std::is_fundamental_v<T>) {
                return Op::exec(val);
            } else {
                static_assert(fk::validCUDAVec<T>, "Type not supported in Unary operation execution.");
                using OT = typename fk::VectorType<decltype(Op::exec(std::declval<fk::VBase<T>>())), fk::cn<T>>::type_v;
                if constexpr (fk::cn<T> == 1) {
                    return OT{ Op::exec(val.x) };
                } else if constexpr (fk::cn<T> == 2) {
                    return OT{ Op::exec(val.x), Op::exec(val.y) };
                } else if constexpr (fk::cn<T> == 3) {
                    return OT{ Op::exec(val.x), Op::exec(val.y), Op::exec(val.z) };
                } else {
                    return OT{ Op::exec(val.x), Op::exec(val.y), Op::exec(val.z), Op::exec(val.w) };
                }
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
        FK_HOST_DEVICE_FUSE auto exec(const ST1& s1, const ST2& s2, const STs&... scals) {
            return exec(Op::exec(s1, s2), scals...);
        }
        template <typename VT>
        FK_HOST_DEVICE_FUSE auto exec(const VT& v)
            -> std::enable_if_t<fk::IsCudaVector<VT>::value,
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
            -> std::enable_if_t<fk::AreVS<VT, ST>::value,
            fk::VectorType_t<decltype(Op::exec(std::declval<fk::VBase<VT>>(), std::declval<ST>())), fk::cn<VT>>> {
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
            -> std::enable_if_t<fk::AreSV<ST, VT>::value,
            fk::VectorType_t<decltype(Op::exec(std::declval<ST>(), std::declval<fk::VBase<VT>>())), fk::cn<VT>>> {
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
            fk::VectorType_t<decltype(Op::exec(std::declval<fk::VBase<VT1>>(), std::declval<fk::VBase<VT2>>())), fk::cn<VT1>>> {
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
} // namespace cxp

#endif // CXP_CONSTEXPR_VECTOR_EXEC_H
