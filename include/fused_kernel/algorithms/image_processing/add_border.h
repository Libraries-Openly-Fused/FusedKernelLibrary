/* Copyright 2026 Oscar Amoros Huguet
   Copyright 2026 Grup Mediapro S.L.U (Oscar Amoros Huguet)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_ADD_BORDER_H
#define FK_ADD_BORDER_H

#include <fused_kernel/core/execution_model/operation_model/batch_operations.h>

namespace fk {
    enum class AddBorderType { CONSTANT, BORDER_READER };

    template <AddBorderType ABT, typename T = void>
    struct AddBorderParams;

    template <>
    struct AddBorderParams<AddBorderType::BORDER_READER, void> {
        int top, bottom, left, right;
    };

    template <typename T>
    struct AddBorderParams<AddBorderType::CONSTANT, T> {
        int top, bottom, left, right;
        T borderValue;
    };

    template <AddBorderType ABT, typename BackIOp_>
    struct AddBorderComplete;

    template <typename BackIOp_>
    struct AddBorderComplete<AddBorderType::CONSTANT, BackIOp_> {
      private:
        using SelfType = AddBorderComplete<AddBorderType::CONSTANT, BackIOp_>;
        using ParentType =
            ReadBackOperation<typename BackIOp_::Operation::OutputType,
                              AddBorderParams<AddBorderType::CONSTANT>,
                              BackIOp_,
                              typename BackIOp_::Operation::OutputType,
                              SelfType>;
      public:
        FK_STATIC_STRUCT(AddBorderComplete, SelfType)
        DECLARE_READBACK_PARENT

        FK_HOST_DEVICE_FUSE OutputType exec(const Point thread, const ParamsType& params, const BackIOp& backIOp) {
            OutputType result{};
            if (thread.x < params.left || thread.x >= params.left + BackIOp_::Operation::num_elems_x(thread, backIOp) ||
                thread.y < params.top  || thread.y >= params.top + BackIOp_::Operation::num_elems_y(thread, backIOp))
            {
                result = params.borderValue;
            } 
            else
            {
                result = BackIOp::Operation::exec(Point{thread.x - params.left, thread.y - params.top, thread.z}, backIOp);
            }
            return result;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point thread, const OperationDataType& opData) {
            return opData.params.left + opData.params.right + BackIOp_::Operation::num_elems_x(thread, opData.backIOp);
        }
        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point thread, const OperationDataType& opData) {
            return opData.params.top + opData.params.bottom + BackIOp_::Operation::num_elems_y(thread, opData.backIOp);
        }
        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point thread, const OperationDataType& opData) {
            return 1;
        }
    };

    template <typename BackIOp_>
    struct AddBorderComplete<AddBorderType::BORDER_READER, BackIOp_>
    {
        using ParamsType =  AddBorderParams<AddBorderType::BORDER_READER, typename BackIOp_::Operation::OutputType>;
        using BackIOp = BackIOp_;
    };

    // Incomplete
    template <AddBorderType ABT, typename CType = void>
    struct AddBorderIncomplete
    {
      private:
        using SelfType = AddBorderIncomplete<ABT>;
        using Parent = fk::IncompleteReadBackOperation<NullType,
                                                       AddBorderParams<ABT, CType>,
                                                       NullType,
                                                       std::conditional_t<std::is_same_v<CType, void>, NullType, CType>,
                                                       SelfType>;
      public:
        FK_STATIC_STRUCT(AddBorderIncomplete, SelfType)
        DECLARE_INCOMPLETEREADBACK_PARENT
        template <typename NewBackIOp>
        FK_HOST_FUSE auto build(const NewBackIOp& backIOp, const SelfType& selfIOp)
        {
            if constexpr (std::bool_constant<ABT == AddBorderType::CONSTANT>::value)
            {
                static_assert(std::is_same_v<OutputType, typename NewBackIOp::Operation::OutputType>, "OutputTypes must be the same for AddBorder and the BackIOp");
            }
            return AddBorderComplete<ABT, NewBackIOp>::build(selfIOp.params, backIOp);
        }
    };

    // Builder
    struct AddBorder {
        template <typename T, typename BackIOp>
        FK_HOST_FUSE auto build(const int top, const int bottom, const int left, const int right, const T borderValue, const BackIOp& iOp)
        {
            return AddBorderComplete<AddBorderType::CONSTANT, BackIOp>{{{top, bottom, left, right, borderValue}, iOp}};
        }
        template <typename BackIOp>
        FK_HOST_FUSE auto build(const int top, const int bottom, const int left, const int right, const BackIOp& iOp)
        {
            return AddBorderComplete<AddBorderType::BORDER_READER, BackIOp>{{{top, bottom, left, right}, iOp}};
        }

        template <typename T>
        FK_HOST_FUSE auto build(const int top, const int bottom, const int left, const int right, const T borderValue)
        {
            return AddBorderIncomplete
        }
        FK_HOST_FUSE auto build(const int top, const int bottom, const int left, const int right);
    };
} // namespace fk

#endif // !FK_ADD_BORDER_H
