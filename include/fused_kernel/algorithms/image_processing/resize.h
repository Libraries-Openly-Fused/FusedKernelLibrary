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

#ifndef FK_RESIZE
#define FK_RESIZE

#include <fused_kernel/algorithms/image_processing/interpolation.h>
#include <fused_kernel/algorithms/image_processing/saturate.h>
#include <fused_kernel/algorithms/basic_ops/cast.h>
#include <fused_kernel/core/data/array.h>
#include <fused_kernel/core/execution_model/memory_operations.h>
#include <fused_kernel/core/constexpr_libs/constexpr_cmath.h>

namespace fk {
    struct ComputeResizePoint {
        FK_STATIC_STRUCT(ComputeResizePoint, ComputeResizePoint)
        using Parent = BinaryOperation<Point, float2, float2, ComputeResizePoint>;
        DECLARE_BINARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& thread, const ParamsType& params) {
            // This is what makes the interpolation a resize operation
            const float fx = params.x;
            const float fy = params.y;

            const float src_x = thread.x * fx;
            const float src_y = thread.y * fy;

            return { src_x, src_y };
        }
    };

    enum class AspectRatio { PRESERVE_AR = 0, IGNORE_AR = 1, PRESERVE_AR_RN_EVEN = 2, PRESERVE_AR_LEFT = 3 };

    template <InterpolationType IType, AspectRatio AR = AspectRatio::IGNORE_AR, typename T = NullType>
    struct ResizeParams {
        static_assert(!std::is_same_v<T, NullType>, "DefaultType must be different than NullType");
        Size dstSize; // This is the destination size used to compute the src_conv_factors
        float2 src_conv_factors;
        InterpolationParameters<IType> params;
        int x1, y1; // Top left
        int x2, y2; // Bottom right
        T defaultValue;
    };

    template <InterpolationType IType>
    struct ResizeParams<IType, AspectRatio::IGNORE_AR, NullType> {
        Size dstSize; // This is the destination size used to compute the src_conv_factors
        float2 src_conv_factors;
        InterpolationParameters<IType> params;
    };

    template <InterpolationType IType, AspectRatio AR = AspectRatio::IGNORE_AR, typename DefaultType = NullType, typename BackIOp_ = NullType, typename = void>
    struct Resize;

    template <InterpolationType IType, typename BackIOp_>
    struct Resize<IType, AspectRatio::IGNORE_AR, NullType, BackIOp_> {
        static_assert(isAnyCompleteReadType<BackIOp_>, "BackIOp must be a complete type for this specialization");
    private:
        using SelfType = Resize<IType, AspectRatio::IGNORE_AR, NullType, BackIOp_>;
    public:
        FK_STATIC_STRUCT(Resize, SelfType)
            using InterpolateOutputType = typename Interpolate<IType, BackIOp_>::OutputType;
        using Parent = ReadBackOperation<typename BackIOp_::Operation::OutputType,
                                         ResizeParams<IType, AspectRatio::IGNORE_AR>,
                                         BackIOp_,
                                         InterpolateOutputType,
                                         SelfType>;
        DECLARE_READBACK_PARENT

        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const ParamsType& params, const BackIOp& backIOp) {
            return exec_resize(thread, params, backIOp);
        }

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return opData.params.dstSize.width;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const OperationDataType& opData) {
            return opData.params.dstSize.height;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const OperationDataType& opData) {
            return 1;
        }

        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
            return { num_elems_x(Point(), opData), num_elems_y(Point(), opData), num_elems_z(Point(), opData) };
        }

    private:
        FK_HOST_DEVICE_FUSE OutputType exec_resize(const Point& thread, const ParamsType& params, const BackIOp& backIOp) {
            const float fx = params.src_conv_factors.x;
            const float fy = params.src_conv_factors.y;

            const float src_x = thread.x * fx;
            const float src_y = thread.y * fy;
            const float2 rezisePoint = { src_x, src_y };
            // We don't set Interpolate as the BackFuntion of Resize, because we won't use any other function than Interpolate
            // Therefore, we consider Interpolate to be part of the Resize implementation, and not a template variable.
            // But, it would be relatively easy to change Interpolate with anything else if needed.
            return Interpolate<IType, BackIOp>::exec(rezisePoint, { params.params, backIOp });
        }
    };

    template <InterpolationType IType, AspectRatio AR, typename BackIOp_>
    struct Resize<IType, AR, NullType, BackIOp_, std::enable_if_t<AR != AspectRatio::IGNORE_AR && isAnyCompleteReadType<BackIOp_>>> {
    private:
        using SelfType = Resize<IType, AR, NullType, BackIOp_>;
    public:
        FK_STATIC_STRUCT(Resize, SelfType)
        using InterpolateOutputType = typename Interpolate<IType, BackIOp_>::OutputType;
        using Parent = ReadBackOperation<typename BackIOp_::Operation::OutputType,
                                         ResizeParams<IType, AR, InterpolateOutputType>,
                                         BackIOp_,
                                         InterpolateOutputType,
                                         SelfType>;
        DECLARE_READBACK_PARENT

        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const ParamsType& params, const BackIOp& backIOp) {
            if (thread.x >= params.x1 && thread.x <= params.x2 &&
                thread.y >= params.y1 && thread.y <= params.y2) {
                const Point roiThread(thread.x - params.x1, thread.y - params.y1, thread.z);
                return exec_resize(roiThread, params, backIOp);
            } else {
                return params.defaultValue;
            }
        }

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return opData.params.dstSize.width;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const OperationDataType& opData) {
            return opData.params.dstSize.height;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const OperationDataType& opData) {
            return 1;
        }
 
        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
            return { num_elems_x(Point(), opData), num_elems_y(Point(), opData), num_elems_z(Point(), opData) };
        }

    private:
        FK_HOST_DEVICE_FUSE OutputType exec_resize(const Point& thread, const ParamsType& params, const BackIOp& backIOp) {
            const float fx = params.src_conv_factors.x;
            const float fy = params.src_conv_factors.y;

            const float src_x = thread.x * fx;
            const float src_y = thread.y * fy;
            const float2 rezisePoint = { src_x, src_y };
            // We don't set Interpolate as the BackFuntion of Resize, because we won't use any other function than Interpolate
            // Therefore, we consider Interpolate to be part of the Resize implementation, and not a template variable.
            // But, it would be relatively easy to change Interpolate with anything else if needed.
            return Interpolate<IType, BackIOp>::exec(rezisePoint, { params.params, backIOp });
        }
    };

    template <AspectRatio AR, typename T = NullType>
    struct IncompleteResizeParams {
        static_assert(!std::is_same_v<T, NullType>, "DefaultType must be different than NullType");
        Size dstSize;
        T defaultValue;
    };

    template <>
    struct IncompleteResizeParams<AspectRatio::IGNORE_AR, NullType> {
        Size dstSize;
    };

    template <InterpolationType IType>
    struct Resize<IType, AspectRatio::IGNORE_AR> {
    private:
        using SelfType = Resize<IType, AspectRatio::IGNORE_AR>;
    public:
        FK_STATIC_STRUCT(Resize, SelfType)
        using Parent = IncompleteReadBackOperation<NullType,
                                                   IncompleteResizeParams<AspectRatio::IGNORE_AR>,
                                                   NullType,
                                                   NullType,
                                                   SelfType>;
        DECLARE_INCOMPLETEREADBACK_PARENT
        template <typename BackIOp_>
        using NewInstantiableType = Resize<IType, AspectRatio::IGNORE_AR, NullType, BackIOp_>;

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return opData.params.dstSize.width;
        }
        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const OperationDataType& opData) {
            return opData.params.dstSize.height;
        }
        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const OperationDataType& opData) {
            return 1;
        }

        FK_HOST_FUSE auto build(const Size& dstSize) {
            return InstantiableType{ {ParamsType{dstSize}, BackIOp{}} };
        }

        template <typename NewBackIOp>
        FK_HOST_FUSE auto build(const NewBackIOp& backIOp, const Size& dstSize) {
            const Size srcSize = NumElems::size(Point(), backIOp);
            const double cfx = static_cast<double>(dstSize.width) / static_cast<double>(srcSize.width);
            const double cfy = static_cast<double>(dstSize.height) / static_cast<double>(srcSize.height);
            const ResizeParams<IType, AspectRatio::IGNORE_AR> resizeParams{
                dstSize,
                { static_cast<float>(1.0 / cfx), static_cast<float>(1.0 / cfy) },
                { srcSize }
            };

            return NewInstantiableType<NewBackIOp>::build(resizeParams, backIOp);
        }

        template <typename NewBackIOp>
        FK_HOST_FUSE auto build(const NewBackIOp& backIOp, const InstantiableType& selfIOp) {
            return build(backIOp, selfIOp.params.dstSize);
        }

        template <typename T>
        FK_HOST_FUSE auto build(const RawPtr<ND::_2D, T>& input, const Size& dSize, const double& fx, const double& fy) {
            const auto readIOp = PerThreadRead<ND::_2D, T>::build(input);
            if (dSize.width != 0 && dSize.height != 0) {
                return build(readIOp, dSize);
            } else {
                const Size computedDSize{ SaturateCast<double, int>::exec(input.dims.width * fx),
                                          SaturateCast<double, int>::exec(input.dims.height * fy) };

                return build(readIOp, computedDSize);
            }
        }
    };

    template <InterpolationType IType, AspectRatio AR, typename DefaultType>
    struct Resize<IType, AR, DefaultType, NullType, std::enable_if_t<AR != AspectRatio::IGNORE_AR && !std::is_same_v<DefaultType, NullType>>> {
    private:
        using SelfType = Resize<IType, AR, DefaultType>;
    public:
        FK_STATIC_STRUCT(Resize, SelfType)
            using Parent = IncompleteReadBackOperation<NullType,
            IncompleteResizeParams<AR, DefaultType>,
            NullType,
            DefaultType,
            SelfType>;
        DECLARE_INCOMPLETEREADBACK_PARENT

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return opData.params.dstSize.width;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const OperationDataType& opData) {
            return opData.params.dstSize.height;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const OperationDataType& opData) {
            return 1;
        }

        template <typename NewBackIOp>
        FK_HOST_FUSE auto build(const NewBackIOp& backIOp, const InstantiableType& iOp) {
            using NewOutputType = VectorType_t<float, cn<typename NewBackIOp::Operation::OutputType>>;
            return Resize<IType, AR>::build(backIOp, iOp.params.dstSize, v_static_cast<NewOutputType>(iOp.params.defaultValue));
        }
    };

    template <InterpolationType IType, AspectRatio AR>
    struct Resize<IType, AR, NullType, NullType, std::enable_if_t<AR != AspectRatio::IGNORE_AR>> {
    private:
        using SelfType = Resize<IType, AR>;
    public:
        FK_STATIC_STRUCT(Resize, SelfType)
            using Parent = IncompleteReadBackOperation<NullType,
                                                       NullType,
                                                       NullType,
                                                       NullType,
                                                       SelfType>;
        DECLARE_INCOMPLETEREADBACK_PARENT
        template <typename NewBackIOp>
        using NewInstantiableType = Resize<IType, AR, NullType, NewBackIOp>;

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return opData.params.dstSize.width;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const OperationDataType& opData) {
            return opData.params.dstSize.height;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const OperationDataType& opData) {
            return 1;
        }

        template <typename DefaultType>
        FK_HOST_FUSE auto build(const Size& dstSize, const DefaultType& backgroundValue) {
            using NewParamsType = IncompleteResizeParams<AR, DefaultType>;
            const NewParamsType resizeParams{ dstSize, backgroundValue };
            return Resize<IType, AR, DefaultType>::build(resizeParams, NullType{});
        }

        template <typename NewBackIOp>
        FK_HOST_FUSE auto build(const NewBackIOp& backIOp, const Size& dstSize,
                                const VectorType_t<float, cn<typename NewBackIOp::Operation::OutputType>>& backgroundValue) {
            const Size srcSize = NumElems::size(Point(), backIOp);
            const Size targetSize = compute_target_size(srcSize, dstSize);

            const double cfx = static_cast<double>(targetSize.width) / srcSize.width;
            const double cfy = static_cast<double>(targetSize.height) / srcSize.height;

            using NewOutputType = VectorType_t<float, cn<typename NewBackIOp::Operation::OutputType>>;
            using NewParamsType = ResizeParams<IType, AR, NewOutputType>;

            if constexpr (AR == AspectRatio::PRESERVE_AR_LEFT) {
                const int x1 = 0; // Always 0 to make sure the image is adjusted to the left
                const int y1 = static_cast<int>((dstSize.height - targetSize.height) / 2);

                const NewParamsType resizeParams{
                dstSize,
                { static_cast<float>(1.0 / cfx), static_cast<float>(1.0 / cfy) },
                { srcSize },
                x1,
                y1,
                x1 + targetSize.width - 1,
                y1 + targetSize.height - 1,
                backgroundValue};

                return NewInstantiableType<NewBackIOp>::build(resizeParams, backIOp);
            } else {
                const int x1 = static_cast<int>((dstSize.width - targetSize.width) / 2);
                const int y1 = static_cast<int>((dstSize.height - targetSize.height) / 2);

                const NewParamsType resizeParams{
                dstSize,
                { static_cast<float>(1.0 / cfx), static_cast<float>(1.0 / cfy) },
                { srcSize },
                x1,
                y1,
                x1 + targetSize.width - 1,
                y1 + targetSize.height - 1,
                backgroundValue};

                return NewInstantiableType<NewBackIOp>::build(resizeParams, backIOp);
            }
        }

    private:
        FK_HOST_FUSE Size compute_target_size(const Size& srcSize, const Size& dstSize) {
            const float scaleFactor = dstSize.height / (float)srcSize.height;
            const int targetHeight = dstSize.height;
            const int targetWidth = static_cast<int>(cxp::round(scaleFactor * srcSize.width));
            if constexpr (AR == AspectRatio::PRESERVE_AR_RN_EVEN) {
                // We round to the next even integer smaller or equal to targetWidth
                const int targetWidthTemp = targetWidth - (targetWidth % 2);
                if (targetWidthTemp > dstSize.width) {
                    const float scaleFactorTemp = dstSize.width / (float)srcSize.width;
                    const int targetWidthTemp2 = dstSize.width;
                    const int targetHeightTemp = static_cast<int> (cxp::round(scaleFactorTemp * srcSize.height));
                    return Size(targetWidthTemp2, targetHeightTemp - (targetHeightTemp % 2));
                } else {
                    return Size(targetWidthTemp, targetHeight);
                }
            } else {
                if (targetWidth > dstSize.width) {
                    const float scaleFactorTemp = dstSize.width / (float)srcSize.width;
                    const int targetWidthTemp = dstSize.width;
                    const int targetHeightTemp = static_cast<int> (cxp::round(scaleFactorTemp * srcSize.height));
                    return Size(targetWidthTemp, targetHeightTemp);
                } else {
                    return Size(targetWidth, targetHeight);
                }
            }
        }
    };
}; // namespace fk

#endif
