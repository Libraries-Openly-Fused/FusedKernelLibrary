/* Copyright 2025 Oscar Amoros Huguet
   Copyright 2025 Grup Mediapro S.L.U (Oscar Amoros Huguet)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_DEINTERLACE
#define FK_DEINTERLACE

#include <fused_kernel/core/execution_model/operation_model/operation_model.h>
#include <fused_kernel/core/execution_model/memory_operations.h>
#include <fused_kernel/core/data/size.h>
#include <fused_kernel/algorithms/basic_ops/logical.h>

namespace fk {
    enum class DeinterlaceType { BLEND, INTER_LINEAR };

    template <enum DeinterlaceType DType>
    struct DeinterlaceParameters {};

    template <>
    struct DeinterlaceParameters<DeinterlaceType::BLEND> {};

    template <>
    struct DeinterlaceParameters<DeinterlaceType::INTER_LINEAR> {
        bool useEvenLines{true};
    };

    enum class DeinterlaceLinear : bool { USE_EVEN = true, USE_ODD = false };

    template <enum DeinterlaceType DType, typename BackIOp_ = NullType>
    struct Deinterlace {
        static_assert(isAnyCompleteReadType<BackIOp_>, "BackIOp must be a complete type for this specialization");
    private:
        using SelfType = Deinterlace<DType, BackIOp_>;
    public:
        FK_STATIC_STRUCT(Deinterlace, SelfType)
        using Parent = ReadBackOperation<typename BackIOp_::Operation::OutputType,
                                         DeinterlaceParameters<DType>,
                                         BackIOp_,
                                         VectorType_t<float, cn<typename BackIOp_::Operation::OutputType>>,
                                         Deinterlace<DType, BackIOp_>>;
        DECLARE_READBACK_PARENT

        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const ParamsType& params, const BackIOp& backIOp) {
            if constexpr (DType == DeinterlaceType::BLEND) {
                return execBlend(thread, params, backIOp);
            } else { // INTER_LINEAR
                return execInterLinear(thread, params, backIOp);
            }
        }

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return BackIOp::Operation::num_elems_x(thread, opData.backIOp);
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const OperationDataType& opData) {
            return BackIOp::Operation::num_elems_y(thread, opData.backIOp);
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const OperationDataType& opData) {
            return 1;
        }

        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
            return { num_elems_x(Point(), opData), num_elems_y(Point(), opData), num_elems_z(Point(), opData) };
        }
    private:
        FK_HOST_DEVICE_FUSE OutputType execBlend(const Point& thread, const ParamsType& params, const BackIOp& backIOp) {
            // For blend deinterlacing, we average the current line with adjacent lines
            using ReadOperation = typename BackIOp::Operation;
            
            // Read current pixel
            const auto current = ReadOperation::exec(Point(thread.x, thread.y, thread.z), backIOp);
            
            if (thread.y > 0) {
                const auto above = ReadOperation::exec(Point(thread.x, thread.y - 1, thread.z), backIOp);
                return (current + above + 1) * 0.5f;
            } else {
                return current * 1.f;
            }
        }

        FK_HOST_DEVICE_FUSE OutputType execInterLinearGetPixel(const Point& thread, const BackIOp& backIOp, const bool& interpolate) {
            using ReadOperation = typename BackIOp::Operation;
            if (interpolate) {
                // We average the above pixel with the below pixel
                const auto above = ReadOperation::exec(Point(thread.x, thread.y - 1, thread.z), backIOp);
                const auto below = ReadOperation::exec(Point(thread.x, thread.y + 1, thread.z), backIOp);
                return (above + below + 1) * 0.5f;
            } else {
                return ReadOperation::exec(thread, backIOp) * 1.f;
            }
        }

        FK_HOST_DEVICE_FUSE OutputType execInterLinear(const Point& thread, const ParamsType& params, const BackIOp& backIOp) {
            using ReadOperation = typename BackIOp::Operation;
            
            // Assuming BackFunction::Operation::num_elems_y(Point(), backIOp) is an even number
            // If useEvenLines is true, we interpolate on odd lines, otherwise we interpolate the even lines
            // useEvenLines = true, we interpolate if thread.y is odd and not the last line
            // useEvenLines = false, we interpolate if thread.y is even and not the first line
            const bool interpolate = params.useEvenLines ?
                                        !IsEven<int>::exec(thread.y) && thread.y != ReadOperation::num_elems_y(Point(), backIOp) - 1
                                        : IsEven<int>::exec(thread.y) && thread.y != 0;

            return execInterLinearGetPixel(thread, backIOp, interpolate);
        }
    };

    template <enum DeinterlaceType DType>
    struct Deinterlace<DType, NullType> {
    private:
        using SelfType = Deinterlace<DType, NullType>;
    public:
        FK_STATIC_STRUCT(Deinterlace, SelfType)
        using Parent = IncompleteReadBackOperation<NullType, DeinterlaceParameters<DType>,
                                                   NullType, NullType, Deinterlace<DType, NullType>>;
        DECLARE_INCOMPLETEREADBACK_PARENT
        template <typename BackIOp_>
        using NewInstantiableType = ReadBack<Deinterlace<DType, BackIOp_>>;

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return 1;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const OperationDataType& opData) {
            return 1;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const OperationDataType& opData) {
            return 1;
        }

        FK_HOST_FUSE InstantiableType build(const ParamsType& params) {
            return {{params, {}}};
        }

        template <DeinterlaceType DT = DType>
        FK_HOST_FUSE auto build(const DeinterlaceType& type)
            -> std::enable_if_t<DT == DeinterlaceType::INTER_LINEAR, InstantiableType> {
            const ParamsType deinterlaceParams{ static_cast<bool>(type) };
            return { {deinterlaceParams, {}} };
        }

        template <DeinterlaceType DT = DType>
        FK_HOST_FUSE auto build() 
            -> std::enable_if_t<DT == DeinterlaceType::BLEND, InstantiableType> {
            const ParamsType deinterlaceParams{};
            return InstantiableType{ {deinterlaceParams, {}} };
        }
        template <typename BackIOp_>
        FK_HOST_FUSE auto build(const BackIOp_& backIOp, const InstantiableType& iOp) {
            return NewInstantiableType<BackIOp_>{ {iOp.params, backIOp} };
        }

        template <typename BIOp, DeinterlaceType DT = DType>
        FK_HOST_FUSE auto build(const BIOp& backIOp)
            -> std::enable_if_t<DT == DeinterlaceType::BLEND, NewInstantiableType<BIOp>> {
            return NewInstantiableType<BIOp>{ {ParamsType{}, backIOp} };
        }

        template <typename BIOp, DeinterlaceType DT = DType>
        FK_HOST_FUSE auto build(const DeinterlaceLinear& lin, const BIOp& backIOp)
            -> std::enable_if_t<DT == DeinterlaceType::INTER_LINEAR, NewInstantiableType<BIOp>> {
            const DeinterlaceParameters<DeinterlaceType::INTER_LINEAR> deinterlaceParams{ static_cast<bool>(lin) };
            return NewInstantiableType<BIOp>{ {deinterlaceParams, backIOp} };
        }

    };

} // namespace fk

#endif