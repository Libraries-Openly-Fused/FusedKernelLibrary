/* Copyright 2023-2026 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_MATH
#define FK_MATH

#include <fused_kernel/core/execution_model/operation_model/operation_model.h>
#include <fused_kernel/core/utils/vector_utils.h>
#include <fused_kernel/core/constexpr_libs/constexpr_cmath.h>
#include <fused_kernel/core/constexpr_libs/constexpr_vector_exec.h>
#include <fused_kernel/core/data/tuple.h>
#include <cmath>

namespace fk {
    // Per-channel unary math, applied across vector components via cxp::Exec.
    namespace math_detail {
        struct AbsFunc {
            using InstanceType = UnaryType;
            template <typename ST> FK_HOST_DEVICE_FUSE auto exec(const ST& s) {
                if constexpr (std::is_signed_v<ST>) return s < ST(0) ? static_cast<ST>(-s) : s;
                else return s;
            }
        };
        struct SqrFunc {
            using InstanceType = UnaryType;
            template <typename ST> FK_HOST_DEVICE_FUSE auto exec(const ST& s) { return s * s; }
        };
        struct SqrtFunc {
            using InstanceType = UnaryType;
            template <typename ST> FK_HOST_DEVICE_FUSE auto exec(const ST& s) {
                return ::sqrtf(static_cast<float>(s));
            }
        };
        struct LnFunc {
            using InstanceType = UnaryType;
            template <typename ST> FK_HOST_DEVICE_FUSE auto exec(const ST& s) {
                return ::logf(static_cast<float>(s));
            }
        };
        struct ExpFunc {
            using InstanceType = UnaryType;
            template <typename ST> FK_HOST_DEVICE_FUSE auto exec(const ST& s) {
                return ::expf(static_cast<float>(s));
            }
        };
    } // namespace math_detail

    // |x| per channel.
    template <typename I, typename O = I>
    struct Abs {
    private:
        using SelfType = Abs<I, O>;
    public:
        FK_STATIC_STRUCT(Abs, SelfType)
        using Parent = UnaryOperation<I, O, Abs<I, O>>;
        DECLARE_UNARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType input) {
            return cxp::Exec<math_detail::AbsFunc>::exec(input);
        }
    };

    // x*x per channel.
    template <typename I, typename O = I>
    struct Sqr {
    private:
        using SelfType = Sqr<I, O>;
    public:
        FK_STATIC_STRUCT(Sqr, SelfType)
        using Parent = UnaryOperation<I, O, Sqr<I, O>>;
        DECLARE_UNARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType input) {
            return cxp::Exec<math_detail::SqrFunc>::exec(input);
        }
    };

    // sqrt(x) per channel, computed in float.
    template <typename I, typename O = I>
    struct Sqrt {
    private:
        using SelfType = Sqrt<I, O>;
    public:
        FK_STATIC_STRUCT(Sqrt, SelfType)
        using Parent = UnaryOperation<I, O, Sqrt<I, O>>;
        DECLARE_UNARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType input) {
            return cxp::Exec<math_detail::SqrtFunc>::exec(input);
        }
    };

    // natural log per channel, computed in float.
    template <typename I, typename O = I>
    struct Ln {
    private:
        using SelfType = Ln<I, O>;
    public:
        FK_STATIC_STRUCT(Ln, SelfType)
        using Parent = UnaryOperation<I, O, Ln<I, O>>;
        DECLARE_UNARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType input) {
            return cxp::Exec<math_detail::LnFunc>::exec(input);
        }
    };

    // exp(x) per channel, computed in float.
    template <typename I, typename O = I>
    struct Exp {
    private:
        using SelfType = Exp<I, O>;
    public:
        FK_STATIC_STRUCT(Exp, SelfType)
        using Parent = UnaryOperation<I, O, Exp<I, O>>;
        DECLARE_UNARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType input) {
            return cxp::Exec<math_detail::ExpFunc>::exec(input);
        }
    };

    // |a - b| per channel. Binary form uses a constant; two-input Unary form
    // takes a Tuple of two inputs (e.g. two images).
    namespace math_detail {
        struct AbsDiffFunc {
            using InstanceType = BinaryType;
            template <typename ST1, typename ST2> FK_HOST_DEVICE_FUSE auto exec(const ST1& a, const ST2& b) {
                const auto d = a - b;
                return d < decltype(d)(0) ? -d : d;
            }
        };
    } // namespace math_detail

    template <typename I1, typename I2 = I1, typename O = I1, typename IT = BinaryType>
    struct AbsDiff;

    template <typename I, typename P, typename O>
    struct AbsDiff<I, P, O, BinaryType> {
    private:
        using SelfType = AbsDiff<I, P, O, BinaryType>;
    public:
        FK_STATIC_STRUCT(AbsDiff, SelfType)
        using Parent = BinaryOperation<I, P, O, AbsDiff<I, P, O, BinaryType>>;
        DECLARE_BINARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType input, const ParamsType& params) {
            return cxp::Exec<math_detail::AbsDiffFunc>::exec(input, params);
        }
    };

    template <typename I1, typename I2, typename O>
    struct AbsDiff<I1, I2, O, UnaryType> {
    private:
        using SelfType = AbsDiff<I1, I2, O, UnaryType>;
    public:
        FK_STATIC_STRUCT(AbsDiff, SelfType)
        using Parent = UnaryOperation<Tuple<I1, I2>, O, AbsDiff<I1, I2, O, UnaryType>>;
        DECLARE_UNARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType input) {
            return cxp::Exec<math_detail::AbsDiffFunc>::exec(get<0>(input), get<1>(input));
        }
    };
} // namespace fk

#endif
