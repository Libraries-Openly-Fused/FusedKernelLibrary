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

    template <AspectRatio AR = AspectRatio::IGNORE_AR, typename T = NullType>
    struct ResizeParams {
        static_assert(!std::is_same_v<T, NullType>, "DefaultType must be different than NullType");
        Size dstSize; // This is the destination size used to compute the src_conv_factors
        float2 src_conv_factors;
        int x1, y1; // Top left
        int x2, y2; // Bottom right
        T defaultValue;
    };

    template <typename T>
    struct ResizeParams<AspectRatio::IGNORE_AR, T> {
        Size dstSize; // This is the destination size used to compute the src_conv_factors
        float2 src_conv_factors;
    };

    template <AspectRatio AR, typename T = NullType>
    struct IncompleteResizeParams {
        Size dstSize;
        T defaultValue;
    };

    template <typename T>
    struct IncompleteResizeParams<AspectRatio::IGNORE_AR, T> {
        Size dstSize;
    };

    template <AspectRatio AR, typename BackIOp_>
    struct ResizeComplete {
        static_assert(isTernaryType<BackIOp_>, "BackIOp must be a ternary type for this specialization");
    private:
        using SelfType = ResizeComplete<AR, BackIOp_>;
    public:
        FK_STATIC_STRUCT(ResizeComplete, SelfType)
        using DefaultType = VectorType_t<float, cn<typename BackIOp_::Operation::OutputType>>;
        using Parent = ReadBackOperation<typename BackIOp_::Operation::OutputType,
                                         ResizeParams<AR, DefaultType>,
                                         BackIOp_,
                                         DefaultType,
                                         SelfType>;
        DECLARE_READBACK_PARENT

        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const ParamsType& params, const BackIOp& backIOp) {
            if constexpr (AR == AspectRatio::IGNORE_AR) {
                return exec_resize(thread, params, backIOp);
            } else {
                if (thread.x >= params.x1 && thread.x <= params.x2 &&
                    thread.y >= params.y1 && thread.y <= params.y2) {
                    const Point roiThread(thread.x - params.x1, thread.y - params.y1, thread.z);
                    return exec_resize(roiThread, params, backIOp);
                } else {
                    return params.defaultValue;
                }
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

            // Assuming BackIOp is a TernaryType
            return BackIOp::Operation::exec(rezisePoint, backIOp);
        }
    };

    template <InterpolationType IType = InterpolationType::INTER_LINEAR, AspectRatio AR = AspectRatio::IGNORE_AR, typename DefaultType = NullType>
    struct Resize {
    private:
        using SelfType = Resize<IType, AR, DefaultType>;
    public:
        FK_STATIC_STRUCT(Resize, SelfType)
        using Parent = IncompleteReadBackOperation<NullType,
                                                   IncompleteResizeParams<AR, DefaultType>,
                                                   NullType,
                                                   NullType,
                                                   SelfType>;
        DECLARE_INCOMPLETEREADBACK_PARENT
        template <typename BackIOp_>
        using NewInstantiableType = ResizeComplete<AR, Ternary<InterpolateComplete<IType, BackIOp_>>>;

        FK_HOST_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return opData.params.dstSize.width;
        }
        FK_HOST_FUSE uint num_elems_y(const Point& thread, const OperationDataType& opData) {
            return opData.params.dstSize.height;
        }
        FK_HOST_FUSE uint num_elems_z(const Point& thread, const OperationDataType& opData) {
            return 1;
        }

        template <typename NewBackIOp>
        FK_HOST_FUSE auto build(const NewBackIOp& backIOp, const InstantiableType& iOp) {
            static_assert(isCompleteOperation<NewBackIOp>, "NewBackIOp must be a complete IOp");
            using NewDefaultType = VectorType_t<float, cn<typename NewBackIOp::Operation::OutputType>>;
            static_assert(std::is_same_v<NewDefaultType, DefaultType>, "Default value type and Op::OutputType must be the same.");
            return build(backIOp, iOp.params.dstSize, iOp.params.defaultValue);
        }

        template <typename DefaultType_>
        FK_HOST_FUSE auto build(const Size& dstSize, const DefaultType_& backgroundValue) {
            return Resize<IType, AR, DefaultType_>::build(IncompleteResizeParams<AR, DefaultType_>{dstSize, backgroundValue}, NullType{});
        }

        template <typename NewBackIOp>
        FK_HOST_FUSE auto build(const NewBackIOp& backIOp, const Size& dstSize,
                                const VectorType_t<float, cn<typename NewBackIOp::Operation::OutputType>>& backgroundValue) {
            static_assert(isCompleteOperation<NewBackIOp>, "NewBackIOp must be a complete IOp");
            const Size srcSize = NumElems::size(Point(), backIOp);
            const Size targetSize = compute_target_size(srcSize, dstSize);

            const double cfx = static_cast<double>(targetSize.width) / srcSize.width;
            const double cfy = static_cast<double>(targetSize.height) / srcSize.height;

            using NewOutputType = VectorType_t<float, cn<typename NewBackIOp::Operation::OutputType>>;

            if constexpr (AR == AspectRatio::PRESERVE_AR_LEFT) {
                const int x1 = 0; // Always 0 to make sure the image is adjusted to the left
                const int y1 = static_cast<int>((dstSize.height - targetSize.height) / 2);

                const ResizeParams<AR, NewOutputType> resizeParams{
                dstSize,
                { static_cast<float>(1.0 / cfx), static_cast<float>(1.0 / cfy) },
                x1,
                y1,
                x1 + targetSize.width - 1,
                y1 + targetSize.height - 1,
                backgroundValue };

                return NewInstantiableType<NewBackIOp>::build(resizeParams, Interpolate<IType>::build(backIOp));
            } else {
                const int x1 = static_cast<int>((dstSize.width - targetSize.width) / 2);
                const int y1 = static_cast<int>((dstSize.height - targetSize.height) / 2);

                const ResizeParams<AR, NewOutputType> resizeParams{
                dstSize,
                { static_cast<float>(1.0 / cfx), static_cast<float>(1.0 / cfy) },
                x1,
                y1,
                x1 + targetSize.width - 1,
                y1 + targetSize.height - 1,
                backgroundValue };

                return NewInstantiableType<NewBackIOp>::build(resizeParams, Interpolate<IType>::build(backIOp));
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

    template <>
    struct Resize<InterpolationType::INTER_LINEAR, AspectRatio::IGNORE_AR, NullType> {
    private:
        using SelfType = Resize<InterpolationType::INTER_LINEAR, AspectRatio::IGNORE_AR>;
        static constexpr InterpolationType IType{ InterpolationType::INTER_LINEAR };
    public:
        FK_STATIC_STRUCT(Resize, SelfType)
        using Parent = IncompleteReadBackOperation<NullType,
                                                   IncompleteResizeParams<AspectRatio::IGNORE_AR>,
                                                   NullType,
                                                   NullType,
                                                   SelfType>;
        DECLARE_INCOMPLETEREADBACK_PARENT
        template <typename NewBackIOp>
        using NewInstantiableType = ResizeComplete<AspectRatio::IGNORE_AR, Ternary<InterpolateComplete<InterpolationType::INTER_LINEAR, NewBackIOp>>>;

        FK_HOST_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return opData.params.dstSize.width;
        }
        FK_HOST_FUSE uint num_elems_y(const Point& thread, const OperationDataType& opData) {
            return opData.params.dstSize.height;
        }
        FK_HOST_FUSE uint num_elems_z(const Point& thread, const OperationDataType& opData) {
            return 1;
        }

        FK_HOST_FUSE auto build(const Size& dstSize) {
            return InstantiableType{ {ParamsType{dstSize}, BackIOp{}} };
        }

        template <typename NewBackIOp>
        FK_HOST_FUSE auto build(const NewBackIOp& backIOp, const Size& dstSize) {
            static_assert(isCompleteOperation<NewBackIOp>, "NewBackIOp must be a complete IOp");
            const Size srcSize = NumElems::size(Point(), backIOp);
            const double cfx = static_cast<double>(dstSize.width) / static_cast<double>(srcSize.width);
            const double cfy = static_cast<double>(dstSize.height) / static_cast<double>(srcSize.height);
            const typename NewInstantiableType<NewBackIOp>::ParamsType resizeParams{
                dstSize,
                { static_cast<float>(1.0 / cfx), static_cast<float>(1.0 / cfy) }
            };

            return NewInstantiableType<NewBackIOp>::build(resizeParams, Interpolate<IType>::build(backIOp));
        }

        template <typename NewBackIOp>
        FK_HOST_FUSE auto build(const NewBackIOp& backIOp, const InstantiableType& selfIOp) {
            static_assert(isCompleteOperation<NewBackIOp>, "NewBackIOp must be a complete IOp");
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
}; // namespace fk

#endif
