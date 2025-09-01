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

    template <enum InterpolationType IType, enum AspectRatio AP = AspectRatio::IGNORE_AR, typename T = void>
    struct ResizeReadParams {
        Size dstSize; // This is the destination size used to compute the src_conv_factors
        float2 src_conv_factors;
        InterpolationParameters<IType> params;
        int x1, y1; // Top left
        int x2, y2; // Bottom right
        T defaultValue;
    };

    template <enum InterpolationType IType>
    struct ResizeReadParams<IType, AspectRatio::IGNORE_AR, void> {
        Size dstSize; // This is the destination size used to compute the src_conv_factors
        float2 src_conv_factors;
        InterpolationParameters<IType> params;
    };

    template <enum InterpolationType IType, enum AspectRatio AR = AspectRatio::IGNORE_AR, typename BackIOp_ = void>
    struct Resize {
    private:
        using SelfType = Resize<IType, AR, BackIOp_>;
    public:
        FK_STATIC_STRUCT(Resize, SelfType)
        using InterpolateOutputType = typename Interpolate<IType, BackIOp_>::OutputType;
        using Parent = ReadBackOperation<typename BackIOp_::Operation::OutputType,
                                         ResizeReadParams<IType, AR, std::conditional_t<AR == AspectRatio::IGNORE_AR, void, InterpolateOutputType>>,
                                         BackIOp_,
                                         typename Interpolate<IType, BackIOp_>::OutputType,
                                         Resize<IType, AR, BackIOp_>>;
        DECLARE_READBACK_PARENT

        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const ParamsType& params, const BackIOp& backIOp) {
            if constexpr (AR == AspectRatio::IGNORE_AR) {
                return exec_resize(thread, params, backIOp);
            } else { // Assuming PRESERVE_AR or PRESERVE_AR_RN_EVEN
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

        template <enum AspectRatio AR_ = AR>
        FK_HOST_FUSE
        std::enable_if_t<AR_ == AspectRatio::IGNORE_AR, InstantiableType>
        build(const BackIOp& backIOp, const Size& dstSize) {
            const Size srcSize = Num_elems<BackIOp>::size(Point(), backIOp);
            const double cfx = static_cast<double>(dstSize.width) / static_cast<double>(srcSize.width);
            const double cfy = static_cast<double>(dstSize.height) / static_cast<double>(srcSize.height);
            const ParamsType resizeParams{
                dstSize,
                { static_cast<float>(1.0 / cfx), static_cast<float>(1.0 / cfy) },
                { srcSize }
            };

            return { {resizeParams, backIOp} };
        }

        template <enum AspectRatio AR_ = AR>
        FK_HOST_FUSE
        std::enable_if_t<AR_ != AspectRatio::IGNORE_AR, InstantiableType>
        build(const BackIOp& backIOp, const Size& dstSize, const OutputType& backgroundValue) {
            const Size srcSize = Num_elems<BackIOp>::size(Point(), backIOp);

            const Size targetSize = compute_target_size(srcSize, dstSize);

            const double cfx = static_cast<double>(targetSize.width) / srcSize.width;
            const double cfy = static_cast<double>(targetSize.height) / srcSize.height;

            if constexpr (AR_ == AspectRatio::PRESERVE_AR_LEFT) {
                const int x1 = 0; // Always 0 to make sure the image is adjusted to the left
                const int y1 = static_cast<int>((dstSize.height - targetSize.height) / 2);

                const ParamsType resizeParams{
                dstSize,
                { static_cast<float>(1.0 / cfx), static_cast<float>(1.0 / cfy) },
                { srcSize },
                /*x1*/ x1,
                /*y1*/ y1,
                /*x2*/ x1 + targetSize.width - 1,
                /*y2*/ y1 + targetSize.height - 1,
                /*defaultValue*/ backgroundValue
                };

                return { {resizeParams, backIOp} };

            } else {
                const int x1 = static_cast<int>((dstSize.width - targetSize.width) / 2);
                const int y1 = static_cast<int>((dstSize.height - targetSize.height) / 2);

                const ParamsType resizeParams{
                dstSize,
                { static_cast<float>(1.0 / cfx), static_cast<float>(1.0 / cfy) },
                { srcSize },
                /*x1*/ x1,
                /*y1*/ y1,
                /*x2*/ x1 + targetSize.width - 1,
                /*y2*/ y1 + targetSize.height - 1,
                /*defaultValue*/ backgroundValue
                };

                return { {resizeParams, backIOp} };
            }
        }

        template <typename BIOp = BackIOp_>
        FK_HOST_FUSE
        std::enable_if_t<std::is_same_v<BIOp, Read<PerThreadRead<ND::_2D, ReadDataType>>>, InstantiableType>
        build(const RawPtr<ND::_2D, ReadDataType>& input, const Size& dSize, const double& fx, const double& fy) {
            if (dSize.width != 0 && dSize.height != 0) {
                return build(BIOp{ {input} }, dSize);
            } else {
                const Size computedDSize{ SaturateCast<double, int>::exec(input.dims.width * fx),
                                          SaturateCast<double, int>::exec(input.dims.height * fy) };

                return build(BIOp{ {input} }, computedDSize);
            }
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

    template <enum AspectRatio AR, typename T = void>
    struct IncompleteResizeReadParams {
        Size dstSize;
        T defaultValue;
    };

    template <enum AspectRatio AR>
    struct IncompleteResizeReadParams<AR, void> {
        Size dstSize;
    };

    template <enum InterpolationType IType, enum AspectRatio AR, typename T>
    struct Resize<IType, AR, TypeList<void, T>> {
    private:
        using SelfType = Resize<IType, AR, TypeList<void, T>>;
    public:
        FK_STATIC_STRUCT(Resize, SelfType)
        using Parent = ReadBackOperation<NullType,
                                         IncompleteResizeReadParams<AR, T>,
                                         NullType,
                                         T,
                                         Resize<IType, AR, TypeList<void, T>>>;
        DECLARE_READBACK_PARENT_INCOMPLETE

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return opData.params.dstSize.width;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const OperationDataType& opData) {
            return opData.params.dstSize.height;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const OperationDataType& opData) {
            return 1;
        }

        template <enum AspectRatio AR_ = AR>
        FK_HOST_FUSE std::enable_if_t<AR_ != AspectRatio::IGNORE_AR, InstantiableType>
        build(const Size& dstSize, const T& backgroundValue) {
            return InstantiableType{ {{dstSize, backgroundValue}, {}} };
        }

        template <typename ReadIOp, enum AspectRatio AR_ = AR>
        FK_HOST_FUSE std::enable_if_t<AR_ != AspectRatio::IGNORE_AR, ReadBack<Resize<IType, AR_, ReadIOp>>>
        build(const ReadIOp& readIOp, const InstantiableType& iOp) {
            using ReadIOpOutputType = typename Resize<IType, AR_, ReadIOp>::OutputType;
            return Resize<IType, AR_, ReadIOp>::build(readIOp, iOp.params.dstSize, Cast<T, ReadIOpOutputType>::exec(iOp.params.defaultValue));
        }
    };

    template <enum InterpolationType IType>
    struct Resize<IType, AspectRatio::IGNORE_AR, TypeList<void, void>> {
    private:
        using SelfType = Resize<IType, AspectRatio::IGNORE_AR, TypeList<void, void>>;
    public:
        FK_STATIC_STRUCT(Resize, SelfType)
        using Parent = ReadBackOperation<NullType,
                                         IncompleteResizeReadParams<AspectRatio::IGNORE_AR, void>,
                                         NullType,
                                         NullType,
                                         Resize<IType, AspectRatio::IGNORE_AR, TypeList<void, void>>>;
        DECLARE_READBACK_PARENT_INCOMPLETE

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return opData.params.dstSize.width;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const OperationDataType& opData) {
            return opData.params.dstSize.height;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const OperationDataType& opData) {
            return 1;
        }

        FK_HOST_FUSE InstantiableType build(const Size& dstSize) {
            return { {{dstSize}, {}} };
        }

        template <typename ReadIOp>
        FK_HOST_FUSE auto build(const ReadIOp& readIOp, const InstantiableType& iOp) {
            return Resize<IType, AspectRatio::IGNORE_AR, ReadIOp>::build(readIOp, iOp.params.dstSize);
        }
    };

    template <enum InterpolationType IType, enum AspectRatio AR>
    struct Resize<IType, AR, void> {
    private:
        using SelfType = Resize<IType, AR, void>;
    public:
        FK_STATIC_STRUCT(Resize, SelfType)
        using Parent = ReadBackOperation<NullType, NullType, NullType, NullType,
                                         Resize<IType, AR, void>>;
        DECLARE_READBACK_PARENT_BATCH_INCOMPLETE
        template <typename BIOp, enum AspectRatio AR_ = AR>
        FK_HOST_FUSE std::enable_if_t<AR_ == AspectRatio::IGNORE_AR && isAnyReadType<BIOp>, ReadBack<Resize<IType, AR_, BIOp>>>
        build(const BIOp& backIOp, const Size& dstSize) {
            return Resize<IType, AR_, BIOp>::build(backIOp, dstSize);
        }

        template <typename BIOp, enum AspectRatio AR_ = AR>
        FK_HOST_FUSE std::enable_if_t<AR_ != AspectRatio::IGNORE_AR && isAnyReadType<BIOp>, ReadBack<Resize<IType, AR_, BIOp>>>
        build(const BIOp& backIOp, const Size& dstSize,
              const typename Resize<IType, AR_, BIOp>::OutputType& backgroundValue) {
            return Resize<IType, AR_, BIOp>::build(backIOp, dstSize, backgroundValue);
        }

        template <enum AspectRatio AR_ = AR>
        FK_HOST_FUSE std::enable_if_t<AR_ == AspectRatio::IGNORE_AR, ReadBack<Resize<IType, AR_, TypeList<void, void>>>>
        build(const Size& dstSize) {
            return Resize<IType, AR_, TypeList<void, void>>::build(dstSize);
        }

        template <typename T, enum AspectRatio AR_ = AR>
        FK_HOST_FUSE std::enable_if_t<AR_ != AspectRatio::IGNORE_AR, ReadBack<Resize<IType, AR_, TypeList<void, T>>>>
        build(const Size& dstSize, const T& backgroundValue) {
            return Resize<IType, AR_, TypeList<void, T>>::build(dstSize, backgroundValue);
        }

        template <typename T>
        FK_HOST_FUSE auto build(const RawPtr<ND::_2D, T>& input, const Size& dSize, const double& fx, const double& fy) {
            return Resize<IType, AR, Read<PerThreadRead<ND::_2D, T>>>::build(input, dSize, fx, fy);
        }
    };
}; // namespace fk

#endif
