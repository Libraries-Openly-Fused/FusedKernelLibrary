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

#ifndef FK_INTERPOLATION
#define FK_INTERPOLATION

#include <fused_kernel/core/constexpr_libs/constexpr_cmath.h>
#include <fused_kernel/core/data/size.h>
#include <fused_kernel/core/execution_model/operation_model/operation_model.h>

namespace fk {
    template <typename T>
    struct Slice2x2 {
        T _0x0;
        T _1x0;
        T _0x1;
        T _1x1;
    };

    enum class InterpolationType {
        INTER_LINEAR = 1,
        NONE = 17
    };

    template <enum InterpolationType INTER_T>
    struct InterpolationParameters;

    template <>
    struct InterpolationParameters<InterpolationType::INTER_LINEAR> {};

    template <enum InterpolationType IType, typename BackIOp_>
    struct InterpolateComplete;

    template <typename BackIOp_>
    struct InterpolateComplete<InterpolationType::INTER_LINEAR, BackIOp_> {
        static_assert(isCompleteOperation<BackIOp_>, "NewBackIOp must be a complete operation.");
    private:
        using SelfType = InterpolateComplete<InterpolationType::INTER_LINEAR, BackIOp_>;
        using BackIOpOutputType = typename BackIOp_::Operation::OutputType;
    public:
        FK_STATIC_STRUCT(InterpolateComplete, SelfType)
        using Parent = TernaryOperation<float2, InterpolationParameters<InterpolationType::INTER_LINEAR>,
                                        BackIOp_, VectorType_t<float, cn<BackIOpOutputType>>,
                                        SelfType>;
        DECLARE_TERNARY_PARENT

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return BackIOp::Operation::num_elems_x(thread, opData.backIOp);
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const OperationDataType& opData) {
            return BackIOp::Operation::num_elems_y(thread, opData.backIOp);
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const OperationDataType& opData) {
            return 1;
        }

        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const ParamsType& params, const BackIOp& backIOp) {
            const float src_x = input.x;
            const float src_y = input.y;

#ifdef __CUDA_ARCH__
            const int x1 = __float2int_rd(src_x);
            const int y1 = __float2int_rd(src_y);
#else
            const int x1 = static_cast<int>(cxp::floor::f(src_x));
            const int y1 = static_cast<int>(cxp::floor::f(src_x));
#endif
            const int x2 = x1 + 1;
            const int y2 = y1 + 1;

            const Size srcSize = NumElems::size(Point(), backIOp);
            const int x2_read = cxp::min::f(x2, srcSize.width - 1);
            const int y2_read = cxp::min::f(y2, srcSize.height - 1);

            const Slice2x2<Point> readPoints{ Point(x1, y1),
                                              Point(x2_read, y1),
                                              Point(x1, y2_read),
                                              Point(x2_read, y2_read) };

            // Read the 4 pixels from backIOp Read or ReadBack Operation
            const auto src_reg0x0 = BackIOp::Operation::exec(readPoints._0x0, backIOp);
            const auto src_reg1x0 = BackIOp::Operation::exec(readPoints._1x0, backIOp);
            const auto src_reg0x1 = BackIOp::Operation::exec(readPoints._0x1, backIOp);
            const auto src_reg1x1 = BackIOp::Operation::exec(readPoints._1x1, backIOp);

            // Compute the interpolated pixel and return it
            return (src_reg0x0 * ((x2 - src_x) * (y2 - src_y))) +
                   (src_reg1x0 * ((src_x - x1) * (y2 - src_y))) +
                   (src_reg0x1 * ((x2 - src_x) * (src_y - y1))) +
                   (src_reg1x1 * ((src_x - x1) * (src_y - y1)));
        }
    };

    template <InterpolationType IT>
    struct Interpolate;

    template <>
    struct Interpolate<InterpolationType::INTER_LINEAR> {
    private:
        using SelfType = Interpolate<InterpolationType::INTER_LINEAR>;
    public:
        FK_STATIC_STRUCT(Interpolate, SelfType)
        using InputType = float2;
        using OutputType = NullType;
        using ParamsType = InterpolationParameters<InterpolationType::INTER_LINEAR>;
        using BackIOp = NullType;
        using InstanceType = TernaryType;
        using OperationDataType = OperationData<Interpolate<InterpolationType::INTER_LINEAR>>;
        using InstantiableType = Ternary<SelfType>;
        static constexpr bool IS_FUSED_OP = false;

        FK_HOST_FUSE auto build(const OperationDataType& opData) {
            return InstantiableType{ opData };
        }
        FK_HOST_FUSE auto build(const ParamsType& params, const BackIOp& backIOp) {
            return InstantiableType{ OperationDataType{params, backIOp} };
        }

        FK_HOST_FUSE auto build() {
            return InstantiableType{};
        }

        template <typename NewBackIOp>
        FK_HOST_FUSE auto build(const NewBackIOp& newBackIOp) {
            return InterpolateComplete<InterpolationType::INTER_LINEAR, NewBackIOp>::build(ParamsType{}, newBackIOp);
        }
    };
} // namespace fk
#endif
