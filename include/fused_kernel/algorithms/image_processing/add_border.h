/* Copyright 2026 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_ADD_BORDER_OP
#define FK_ADD_BORDER_OP

#include <fused_kernel/core/execution_model/operation_model/operation_model.h>
#include <fused_kernel/core/data/point.h>
#include <fused_kernel/core/constexpr_libs/constexpr_cmath.h>

namespace fk {
    // Equivalent to OpenCV's copyMakeBorder with a constant border: the output has size
    // (w + left + right, h + top + bottom), the BackIOp is read at (x - left, y - top),
    // and every pixel outside of the BackIOp gets value. With T = NullType the value is zero.
    template <typename T = NullType>
    struct AddBorderParams {
        int top;
        int bottom;
        int left;
        int right;
        T value;
    };

    template <typename T = NullType, typename BackIOp_ = NullType>
    struct AddBorder {
        static_assert(isAnyCompleteReadType<BackIOp_>, "The BackIOp_ must be a complete Read type");
        static_assert(std::is_same_v<T, typename BackIOp_::Operation::OutputType>,
                      "The border value type must be the BackIOp OutputType");
    private:
        using SelfType = AddBorder<T, BackIOp_>;
    public:
        FK_STATIC_STRUCT(AddBorder, SelfType)
        using Parent = ReadBackOperation<T, AddBorderParams<T>, BackIOp_, T, AddBorder<T, BackIOp_>>;
        DECLARE_READBACK_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const Point thread, const ParamsType& params, const BackIOp& backIOp) {
            const Point srcThread{thread.x - params.left, thread.y - params.top, thread.z};
            const int width = static_cast<int>(BackIOp::Operation::num_elems_x(srcThread, backIOp));
            const int height = static_cast<int>(BackIOp::Operation::num_elems_y(srcThread, backIOp));
            if (srcThread.x >= 0 && srcThread.x < width && srcThread.y >= 0 && srcThread.y < height) {
                return BackIOp::Operation::exec(srcThread, backIOp);
            } else {
                return params.value;
            }
        }

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point thread, const OperationDataType& opData) {
            return BackIOp::Operation::num_elems_x(thread, opData.backIOp) + opData.params.left + opData.params.right;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point thread, const OperationDataType& opData) {
            return BackIOp::Operation::num_elems_y(thread, opData.backIOp) + opData.params.top + opData.params.bottom;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point thread, const OperationDataType& opData) {
            return BackIOp::Operation::num_elems_z(thread, opData.backIOp);
        }

        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
            return { num_elems_x(Point{0,0,0}, opData), num_elems_y(Point{0,0,0}, opData), num_elems_z(Point{0,0,0}, opData) };
        }
    };

    template <typename T>
    struct AddBorder<T, NullType> {
    private:
        using SelfType = AddBorder<T, NullType>;
    public:
        FK_STATIC_STRUCT(AddBorder, SelfType)
        using Parent = IncompleteReadBackOperation<NullType, AddBorderParams<T>, NullType, NullType, AddBorder<T, NullType>>;
        DECLARE_INCOMPLETEREADBACK_PARENT

        // The size of the BackIOp is not known yet, only the border it adds
        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point thread, const OperationDataType& opData) {
            return opData.params.left + opData.params.right;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point thread, const OperationDataType& opData) {
            return opData.params.top + opData.params.bottom;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point thread, const OperationDataType& opData) {
            return 1;
        }

        FK_HOST_FUSE InstantiableType build(const ParamsType& params) {
            return InstantiableType{ { params, NullType{} } };
        }

        template <typename V>
        FK_HOST_FUSE auto build(const AddBorderParams<V>& params) {
            return AddBorder<V>::build(params);
        }

        FK_HOST_FUSE auto build(const int& top, const int& bottom, const int& left, const int& right) {
            return AddBorder<NullType>::build(AddBorderParams<NullType>{ top, bottom, left, right, NullType{} });
        }

        template <typename V>
        FK_HOST_FUSE auto build(const int& top, const int& bottom, const int& left, const int& right, const V& value) {
            return AddBorder<V>::build(AddBorderParams<V>{ top, bottom, left, right, value });
        }

        template <typename BIOp>
        FK_HOST_FUSE auto build(const BIOp& backIOp, const InstantiableType& selfIOp) {
            static_assert(isAnyCompleteReadType<BIOp>, "BIOp type is not of any complete Read Type.");
            using O = typename BIOp::Operation::OutputType;
            const ParamsType& params = selfIOp.params;
            O value{};
            if constexpr (!std::is_same_v<T, NullType>) {
                value = cxp::cast<O>::f(params.value);
            }
            const AddBorderParams<O> newParams{ params.top, params.bottom, params.left, params.right, value };
            return AddBorder<O, BIOp>::build(newParams, backIOp);
        }
    };

} // namespace fk

#endif // FK_ADD_BORDER_OP
