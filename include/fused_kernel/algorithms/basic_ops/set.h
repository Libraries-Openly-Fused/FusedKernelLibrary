/* Copyright 2024-2025 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_SET
#define FK_SET

#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/execution_model/operation_model/operation_model.h>

namespace fk {

    template <typename T>
    struct ReadSetParams {
        T value;
        ActiveThreads size;
    };

    template <typename T>
    struct ReadSet {
    private:
        using SelfType = ReadSet<T>;
    public:
        FK_STATIC_STRUCT(ReadSet, SelfType)
        using Parent = ReadOperation<T, ReadSetParams<T>, T, TF::DISABLED, ReadSet<T>>;
        DECLARE_READ_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const ParamsType& params) {
            return params.value;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return opData.params.size.x;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const OperationDataType& opData) {
            return opData.params.size.y;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const OperationDataType& opData) {
            return opData.params.size.z;
        }

        FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
            return { num_elems_x(Point(), opData), num_elems_y(Point(), opData), num_elems_z(Point(), opData) };
        }

        FK_HOST_FUSE auto build(const T& value, const ActiveThreads& activeThreads) {
            return InstantiableType{ OperationDataType{{ value, activeThreads }} };
        }

        template <enum ND D>
        FK_HOST_FUSE auto build(const T& value, const PtrDims<D>& dims) {
            if constexpr (D == ND::_1D) {
                const ActiveThreads activeThreads(dims.width, 1, 1);
                return InstantiableType{ OperationDataType{{ value, activeThreads }} };
            } else if constexpr (D == ND::_2D) {
                const ActiveThreads activeThreads(dims.width, dims.height, 1);
                return InstantiableType{ OperationDataType{{ value, activeThreads }} };
            } else if constexpr (D == ND::_3D) {
                const ActiveThreads activeThreads(dims.width, dims.height, dims.planes);
                return InstantiableType{ OperationDataType{{ value, activeThreads }} };
            } else if constexpr (D == ND::T3D) {
                const ActiveThreads activeThreads(dims.width, dims.height, dims.planes);
                return InstantiableType{ OperationDataType{{value, activeThreads}} };
            } else {
                static_assert(D == ND::_1D || D == ND::_2D || D == ND::_3D || D == ND::T3D, "Unsupported ND type for ReadSet build.");
                return InstantiableType{};
            }
        }
    };
} // namespace fk

#endif
