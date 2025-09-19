/* Copyright 2025 Grup Mediapro S.L.U

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
   */

#ifndef FK_BORDER_READER_CUH
#define FK_BORDER_READER_CUH

#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/execution_model/operation_model/operation_model.h>
#include <fused_kernel/core/constexpr_libs/constexpr_cmath.h>

namespace fk {
    //! Border types, image boundaries are denoted with `|`
    enum class BorderType : int {
        CONSTANT = 0,    //!< `iiiiii|abcdefgh|iiiiiii`  with some specified `i`
        REPLICATE = 1,   //!< `aaaaaa|abcdefgh|hhhhhhh`
        REFLECT = 2,     //!< `fedcba|abcdefgh|hgfedcb`
        WRAP = 3,        //!< `cdefgh|abcdefgh|abcdefg`
        REFLECT_101 = 4, //!< `gfedcb|abcdefgh|gfedcba`
        TRANSPARENT_ = 5, //!< `uvwxyz|abcdefgh|ijklmno` - Treats outliers as transparent.

        DEFAULT = REFLECT_101, //!< same as BORDER_REFLECT_101
        ISOLATED = 16 //!< Interpolation restricted within the ROI boundaries.
    };
    template<BorderType BT, typename ReadType = void>
    struct BorderReaderParameters;

    template<BorderType BT>
    struct BorderReaderParameters<BT, void> {};

    template<typename ReadType>
    struct BorderReaderParameters<BorderType::CONSTANT, ReadType> {
        ReadType value;
    };

    template <BorderType BT, typename ParamsType = NullType, typename BackIOp = NullType, typename Enabler = void>
    struct BorderReader;
    
    template <>
    struct BorderReader<BorderType::CONSTANT> {
    private:
        using SelfType = BorderReader<BorderType::CONSTANT>;
    public:
        FK_STATIC_STRUCT(BorderReader, SelfType)
        using Parent = IncompleteReadBackOperation<NullType, NullType, NullType, NullType, SelfType>;
        DECLARE_INCOMPLETEREADBACK_PARENT

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return 1;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const OperationDataType& opData) {
            return 1;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const OperationDataType& opData) {
            return 1;
        }

        template <typename T>
        FK_HOST_FUSE auto build(const T& defaultValue) {
            using NewParamsType = BorderReaderParameters<BorderType::CONSTANT, T>;
            const NewParamsType params{ defaultValue };
            return BorderReader<BorderType::CONSTANT, NewParamsType>::build(params, NullType{});
        }

        template <typename BIOp>
        FK_HOST_FUSE auto build(const BIOp& backFunction, const typename BIOp::Operation::OutputType& defaultValue) {
            static_assert(isAnyCompleteReadType<BIOp>, "BIOp type is not of any complete Read Type.");
            using NewParamsType = BorderReaderParameters<BorderType::CONSTANT, typename BIOp::Operation::OutputType>;
            const NewParamsType params{ defaultValue };
            return BorderReader<BorderType::CONSTANT, NewParamsType, BIOp>::build(params, backFunction);
        }
    };

    template <typename T>
    struct BorderReader<BorderType::CONSTANT, BorderReaderParameters<BorderType::CONSTANT, T>> {
    private:
        using SelfType = BorderReader<BorderType::CONSTANT>;
    public:
        FK_STATIC_STRUCT(BorderReader, SelfType)
        using Parent = IncompleteReadBackOperation<NullType, BorderReaderParameters<BorderType::CONSTANT, T>, NullType, NullType, SelfType>;
        DECLARE_INCOMPLETEREADBACK_PARENT

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return 1;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const OperationDataType& opData) {
            return 1;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const OperationDataType& opData) {
            return 1;
        }

        template <typename BIOp>
        FK_HOST_FUSE auto build(const BIOp& backFunction, const InstantiableType& selfIOp) {
            static_assert(isAnyCompleteReadType<BIOp>, "BIOp type is not of any complete Read Type.");
            using BIOpOutputType = typename BIOp::Operation::OutputType;
            if constexpr (std::is_same_v<BIOpOutputType, T>) {
                return BorderReader<BorderType::CONSTANT, BorderReaderParameters<BorderType::CONSTANT, T>, BIOp>::build(selfIOp.params, backFunction);
            } else {
                return BorderReader<BorderType::CONSTANT, BorderReaderParameters<BorderType::CONSTANT, BIOpOutputType>, BIOp>::build(
                    { cxp::v_static_cast<BIOpOutputType>(selfIOp.params.value) }, backFunction);
            }
        }
    };

    template <BorderType BT>
    struct BorderReader<BT> {
    private:
        using SelfType = BorderReader<BT>;
    public:
        FK_STATIC_STRUCT(BorderReader, SelfType)
        using Parent = IncompleteReadBackOperation<NullType, BorderReaderParameters<BT>, NullType, NullType, SelfType>;
        DECLARE_INCOMPLETEREADBACK_PARENT

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return 1;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const OperationDataType& opData) {
            return 1;
        }

        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const OperationDataType& opData) {
            return 1;
        }

        FK_HOST_FUSE auto build() {
            return BorderReader<BT>{};
        }

        template <typename BIOp>
        FK_HOST_FUSE auto build(const BIOp& backFunction) {
            static_assert(isAnyCompleteReadType<BIOp>, "BIOp type is not of any complete Read Type.");
            return BorderReader<BT, ParamsType, BIOp>::build(ParamsType{}, backFunction);
        }

        template <typename BIOp>
        FK_HOST_FUSE auto build(const BIOp& backFunction, const InstantiableType& selfIOp) {
            static_assert(isAnyCompleteReadType<BIOp>, "BIOp type is not of any complete Read Type.");
            return BorderReader<BT, ParamsType, BIOp>::build(ParamsType{}, backFunction);
        }
    };

#define BORDER_READER_DETAILS(BT) \
private: \
    using SelfType = \
        BorderReader<BT, BorderReaderParameters<BT>, BackIOp_>; \
public: \
    FK_STATIC_STRUCT(BorderReader, SelfType) \
    using Parent = ReadBackOperation<typename BackIOp_::Operation::OutputType, BorderReaderParameters<BT>, \
        BackIOp_, typename BackIOp_::Operation::OutputType, SelfType>; \
    DECLARE_READBACK_PARENT \
FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) { \
    return BackIOp::Operation::num_elems_x(thread, opData.backIOp); \
} \
FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const OperationDataType& opData) { \
    return BackIOp::Operation::num_elems_y(thread, opData.backIOp); \
} \
FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const OperationDataType& opData) { \
    return BackIOp::Operation::num_elems_z(thread, opData.backIOp); \
}

#define BORDER_READER_EXEC \
FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const ParamsType& params, const BackIOp& backIOp) { \
    const int last_col = BackIOp::Operation::num_elems_x(thread, backIOp) - 1; \
    const int last_row = BackIOp::Operation::num_elems_y(thread, backIOp) - 1; \
    const Point new_thread(idx_col(thread.x, last_col), idx_row(thread.y, last_row), thread.z); \
    return BackIOp::Operation::exec(new_thread, backIOp); \
}

    template <typename T, typename BackIOp_>
    struct BorderReader<BorderType::CONSTANT, BorderReaderParameters<BorderType::CONSTANT, T>, BackIOp_,
                        std::enable_if_t<isAnyCompleteReadType<BackIOp_>, void>> {
    private:
        using SelfType = BorderReader<BorderType::CONSTANT, BorderReaderParameters<BorderType::CONSTANT, T>, BackIOp_,
                                        std::enable_if_t<isAnyCompleteReadType<BackIOp_>, void>>;
        using ReadAndOutputType = typename BackIOp_::Operation::OutputType;
        static_assert(std::is_same_v<T, ReadAndOutputType>, "BorderReader default value must have the same type as the OutputType in BackIOp");
    public:
        FK_STATIC_STRUCT(BorderReader, SelfType)
        using Parent = ReadBackOperation<ReadAndOutputType, BorderReaderParameters<BorderType::CONSTANT, T>,
                        BackIOp_, ReadAndOutputType, SelfType>;
        DECLARE_READBACK_PARENT

        FK_HOST_DEVICE_FUSE uint num_elems_x(const Point& thread, const OperationDataType& opData) {
            return BackIOp::Operation::num_elems_x(thread, opData.backIOp);
        }
        FK_HOST_DEVICE_FUSE uint num_elems_y(const Point& thread, const OperationDataType& opData) {
            return BackIOp::Operation::num_elems_y(thread, opData.backIOp);
        }
        FK_HOST_DEVICE_FUSE uint num_elems_z(const Point& thread, const OperationDataType& opData) {
            return BackIOp::Operation::num_elems_z(thread, opData.backIOp);
        }

        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const ParamsType& params, const BackIOp& backIOp) {
            const int width = BackIOp::Operation::num_elems_x(thread, backIOp);
            const int height = BackIOp::Operation::num_elems_y(thread, backIOp);
            if (thread.x >= 0 && thread.x < width && thread.y >= 0 && thread.y < height) {
                return BackIOp::Operation::exec(thread, backIOp);
            } else {
                return params.value;
            }
        }
    };

    template <typename BackIOp_>
    struct BorderReader<BorderType::REPLICATE, BorderReaderParameters<BorderType::REPLICATE>, BackIOp_,
                        std::enable_if_t<isAnyCompleteReadType<BackIOp_>, void>> {
        BORDER_READER_DETAILS(BorderType::REPLICATE)
        BORDER_READER_EXEC
    private:
        FK_HOST_DEVICE_FUSE int idx_row_low(const int& y) {
            return cxp::max(y, 0);
        }
        FK_HOST_DEVICE_FUSE int idx_row_high(const int& y, const int& last_row) {
            return cxp::min(y, last_row);
        }
        FK_HOST_DEVICE_FUSE int idx_row(const int& y, const int& last_row) {
            return idx_row_low(idx_row_high(y, last_row));
        }
        FK_HOST_DEVICE_FUSE int idx_col_low(const int& x) {
            return cxp::max(x, 0);
        }
        FK_HOST_DEVICE_FUSE int idx_col_high(const int& x, const int& last_col) {
            return cxp::min(x, last_col);
        }
        FK_HOST_DEVICE_FUSE int idx_col(const int& x, const int& last_col) {
            return idx_col_low(idx_col_high(x, last_col));
        }
    };

    template <typename BackIOp_>
    struct BorderReader<BorderType::REFLECT, BorderReaderParameters<BorderType::REFLECT>, BackIOp_,
                        std::enable_if_t<isAnyCompleteReadType<BackIOp_>, void>> {
        BORDER_READER_DETAILS(BorderType::REFLECT)
        BORDER_READER_EXEC
    private:
        FK_HOST_DEVICE_FUSE int idx_row_low(const int& y, const int& last_row) {
            return (cxp::abs(y) - (y < 0)) % (last_row + 1);
        }
        FK_HOST_DEVICE_FUSE int idx_row_high(const int& y, const int& last_row) {
            return (last_row - cxp::abs(last_row - y) + (y > last_row));
        }
        FK_HOST_DEVICE_FUSE int idx_row(const int& y, const int& last_row) {
            return idx_row_low(idx_row_high(y, last_row), last_row);
        }
        FK_HOST_DEVICE_FUSE int idx_col_low(const int& x, const int& last_col) {
            return (cxp::abs(x) - (x < 0)) % (last_col + 1);
        }
        FK_HOST_DEVICE_FUSE int idx_col_high(const int& x, const int& last_col) {
            return (last_col - cxp::abs(last_col - x) + (x > last_col));
        }
        FK_HOST_DEVICE_FUSE int idx_col(const int& x, const int& last_col) {
            return idx_col_low(idx_col_high(x, last_col), last_col);
        }
    };

    template <typename BackIOp_>
    struct BorderReader<BorderType::WRAP, BorderReaderParameters<BorderType::WRAP>, BackIOp_,
                        std::enable_if_t<isAnyCompleteReadType<BackIOp_>, void>> {
        BORDER_READER_DETAILS(BorderType::WRAP)
        FK_HOST_DEVICE_FUSE OutputType exec(const Point& thread, const ParamsType& params, const BackIOp& backIOp) {
            const int width = BackIOp::Operation::num_elems_x(thread, backIOp);
            const int height = BackIOp::Operation::num_elems_y(thread, backIOp);
            const Point new_thread(idx_col(thread.x, width), idx_row(thread.y, height), thread.z);
            return BackIOp::Operation::exec(new_thread, backIOp);
        }
    private:
        FK_HOST_DEVICE_FUSE int idx_row_low(const int& y, const int& height) {
            return (y >= 0) ? y : (y - ((y - height + 1) / height) * height);
        }
        FK_HOST_DEVICE_FUSE int idx_row_high(const int& y, const int& height) {
            return (y < height) ? y : (y % height);
        }
        FK_HOST_DEVICE_FUSE int idx_row(const int& y, const int& height) {
            return idx_row_low(idx_row_high(y, height), height);
        }
        FK_HOST_DEVICE_FUSE int idx_col_low(const int& x, const int& width) {
            return (x >= 0) ? x : (x - ((x - width + 1) / width) * width);
        }
        FK_HOST_DEVICE_FUSE int idx_col_high(const int& x, const int& width) {
            return (x < width) ? x : (x % width);
        }
        FK_HOST_DEVICE_FUSE int idx_col(const int& x, const int& width) {
            return idx_col_low(idx_col_high(x, width), width);
        }
    };

    template <typename BackIOp_>
    struct BorderReader<BorderType::REFLECT_101, BorderReaderParameters<BorderType::REFLECT_101>, BackIOp_,
                        std::enable_if_t<isAnyCompleteReadType<BackIOp_>, void>> {
        BORDER_READER_DETAILS(BorderType::REFLECT_101)
        BORDER_READER_EXEC
    private:
        FK_HOST_DEVICE_FUSE int idx_row_low(const int& y, const int& last_row) {
            return cxp::abs(y) % (last_row + 1);
        }
        FK_HOST_DEVICE_FUSE int idx_row_high(const int& y, const int& last_row) {
            return cxp::abs(last_row - cxp::abs(last_row - y)) % (last_row + 1);
        }
        FK_HOST_DEVICE_FUSE int idx_row(const int& y, const int& last_row) {
            return idx_row_low(idx_row_high(y, last_row), last_row);
        }
        FK_HOST_DEVICE_FUSE int idx_col_low(const int& x, const int& last_col) {
            return cxp::abs(x) % (last_col + 1);
        }
        FK_HOST_DEVICE_FUSE int idx_col_high(const int& x, const int& last_col) {
            return cxp::abs(last_col - cxp::abs(last_col - x)) % (last_col + 1);
        }
        FK_HOST_DEVICE_FUSE int idx_col(const int& x, const int& last_col) {
            return idx_col_low(idx_col_high(x, last_col), last_col);
        }
    };

#undef BORDER_READER_DETAILS
#undef BORDER_READER_EXEC

}

#endif // FK_BORDER_READER_CUH