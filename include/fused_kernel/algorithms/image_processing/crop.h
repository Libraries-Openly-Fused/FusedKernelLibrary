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

#ifndef FK_CROP_OP
#define FK_CROP_OP

#include <fused_kernel/core/execution_model/operation_model/operation_model.h>
#include <fused_kernel/core/data/rect.h>
#include <fused_kernel/core/data/point.h>

namespace fk {
    template <typename BackIOp_ = NullType>
    struct Crop {
        static_assert(isAnyCompleteReadType<BackIOp_>, "The BackIOp_ must be a complete Read type");
    private:
        using SelfType = Crop<BackIOp_>;
    public:
        FK_STATIC_STRUCT(Crop, SelfType)
        using Parent = ReadBackOperation<typename BackIOp_::Operation::OutputType,
                                         Rect,
                                         BackIOp_,
                                         typename BackIOp_::Operation::OutputType,
                                         Crop<BackIOp_>>;
        DECLARE_READBACK_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const ParamsType& params, const BackIOp& backIOp) {
            const Point newThread(thread.x + params.x, thread.y + params.y);
            return BackIOp::Operation::exec(newThread, backIOp);
        }

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return opData.params.width;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const OperationDataType& opData) {
            return opData.params.height;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const OperationDataType& opData) {
            return 1;
        }

        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
            return { num_elems_x(Point(), opData), num_elems_y(Point(), opData), num_elems_z(Point(), opData) };
        }

        FK_HOST_FUSE InstantiableType build(const BackIOp& backIOp, const Rect& rect) {
            return InstantiableType{ { rect, backIOp } };
        }
    };

    template <>
    struct Crop<NullType> {
    private:
        using SelfType = Crop<NullType>;
    public:
        FK_STATIC_STRUCT(Crop, SelfType)
        using Parent = IncompleteReadBackOperation<NullType, Rect, NullType, NullType, Crop<NullType>>;
        DECLARE_INCOMPLETEREADBACK_PARENT

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return opData.params.width;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const OperationDataType& opData) {
            return opData.params.height;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const OperationDataType& opData) {
            return 1;
        }

        FK_HOST_FUSE auto build(const Rect& rectCrop) {
            return InstantiableType{ { rectCrop, NullType{} } };
        }

        template <typename BackIOp>
        FK_HOST_FUSE auto build(const BackIOp& bIOp, const InstantiableType& iOp) {
            return Crop<BackIOp>::build(iOp.params, bIOp);
        }
    };

} // namespace fk

#endif