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

#ifndef FK_PARENT_OPERATIONS_EXEC_H
#define FK_PARENT_OPERATIONS_EXEC_H

#include <fused_kernel/core/utils/utils.h>
#include <fused_kernel/core/data/vector_types.h>
#include <fused_kernel/core/execution_model/operation_model/operation_types.h>
#include <fused_kernel/core/execution_model/operation_model/operation_data.h>
#include <fused_kernel/core/utils/macro_utils.h>
#include <fused_kernel/core/data/size.h>

namespace fk {
    // PARENT OPERATIONS
    // PARENT COMPUTE OPERATIONS
#define UNARY_ALIAS(InputT, OutputT, IS_FOP)                       \
  using InputType = DEPAREN(InputT);                               \
  using OutputType = DEPAREN(OutputT);                             \
  using InstanceType = UnaryType;                                  \
  static constexpr bool IS_FUSED_OP = DEPAREN(IS_FOP);

#define UNARY_OPERATION(NAME, InputT, OutputT, IS_FOP) \
        UNARY_ALIAS(InputT, OutputT, IS_FOP)

#define TEMPLATE_UNARY_OPERATION(NAME, TEMPLATE_PARAMS, InputT, OutputT, IS_FOP) \
        UNARY_ALIAS(InputT, OutputT, IS_FOP)

#define UNARY_OPERATION_EXEC(NAME, InputT, OutputT, IS_FOP) \
        FK_STATIC_STRUCT(NAME, NAME)                        \
        UNARY_OPERATION(NAME, InputT, OutputT, IS_FOP)

#define TEMPLATE_UNARY_OPERATION_EXEC(NAME, TEMPLATE_PARAMS, InputT, OutputT, IS_FOP) \
        FK_STATIC_STRUCT_(NAME, (NAME<DEPAREN(TEMPLATE_PARAMS)>))                     \
        TEMPLATE_UNARY_OPERATION(NAME, TEMPLATE_PARAMS, InputT, OutputT, IS_FOP)

#define BINARY_ALIAS(InputT, ParamsT, OutputT, IS_FOP)            \
  using InputType = DEPAREN(InputT);                              \
  using ParamsType = DEPAREN(ParamsT);                            \
  using OutputType = DEPAREN(OutputT);                            \
  using InstanceType = BinaryType;                                \
  static constexpr bool IS_FUSED_OP = DEPAREN(IS_FOP);

#define BINARY_OPERATION(NAME, InputT, ParamsT, OutputT, IS_FOP) \
      BINARY_ALIAS(InputT, ParamsT, OutputT, IS_FOP)             \
      using OperationDataType = OperationData<NAME>;

#define TEMPLATE_BINARY_OPERATION(NAME, TEMPLATE_PARAMS, InputT, ParamsT, OutputT, IS_FOP) \
      BINARY_ALIAS(InputT, ParamsT, OutputT, IS_FOP)                                       \
      using OperationDataType = OperationData<NAME<DEPAREN(TEMPLATE_PARAMS)>>;

#define BINARY_EXEC                                                                              \
  FK_HOST_DEVICE_FUSE OutputType exec(const InputType &input, const OperationDataType &opData) { \
      return exec(input, opData.params);                                                         \
  }

#define BINARY_OPERATION_EXEC(NAME, InputT, ParamsT, OutputT, IS_FOP) \
      FK_STATIC_STRUCT(NAME, NAME)                                    \
      BINARY_OPERATION(NAME, InputT, ParamsT, OutputT, IS_FOP)        \
      BINARY_EXEC

#define TEMPLATE_BINARY_OPERATION_EXEC(NAME, TEMPLATE_PARAMS, InputT, ParamsT, OutputT, IS_FOP) \
      FK_STATIC_STRUCT_(NAME, (NAME<DEPAREN(TEMPLATE_PARAMS)>))                                 \
      TEMPLATE_BINARY_OPERATION(NAME, TEMPLATE_PARAMS, InputT, ParamsT, OutputT, IS_FOP)        \
      BINARY_EXEC

#define TERNARY_ALIAS(InputT, ParamsT, BackIOpT, OutputT)          \
  using InputType = DEPAREN(InputT);                               \
  using OutputType = DEPAREN(OutputT);                             \
  using ParamsType = DEPAREN(ParamsT);                             \
  using BackIOp = DEPAREN(BackIOpT);                               \
  using InstanceType = TernaryType;                                \
  static constexpr bool IS_FUSED_OP = false;

#define TERNARY_OPERATION(NAME, InputT, ParamsT, BackIOpT, OutputT) \
      TERNARY_ALIAS(InputT, ParamsT, BackIOpT, OutputT)             \
      using OperationDataType = OperationData<NAME>;

#define TEMPLATE_TERNARY_OPERATION(NAME, TEMPLATE_PARAMS, InputT, ParamsT, BackIOpT, OutputT) \
      TERNARY_ALIAS(InputT, ParamsT, BackIOpT, OutputT)                                       \
      using OperationDataType = OperationData<NAME<DEPAREN(TEMPLATE_PARAMS)>>;

#define TERNARY_EXEC \
  FK_HOST_DEVICE_FUSE OutputType exec(const InputType &input, const OperationDataType &opData) { \
      return exec(input, opData.params, opData.backIOp);                                         \
  }

#define TERNARY_OPERATION_EXEC(NAME, InputT, ParamsT, BackIOpT, OutputT) \
        FK_STATIC_STRUCT(NAME, NAME)                                     \
        TERNARY_OPERATION(NAME, InputT, ParamsT, BackIOpT, OutputT)      \
        TERNARY_EXEC

#define TEMPLATE_TERNARY_OPERATION_EXEC(NAME, TEMPLATE_PARAMS, InputT, ParamsT, BackIOpT, OutputT) \
        FK_STATIC_STRUCT(NAME, (NAME<DEPAREN(TEMPLATE_PARAMS)>))                                   \
        TEMPLATE_TERNARY_OPERATION(NAME, TEMPLATE_PARAMS, InputT, ParamsT, BackIOpT, OutputT)      \
        TERNARY_EXEC

    // END PARENT COMPUTE OPERATIONS
    // PARENT MEMORY OPERATIONS
#define READ_ALIAS(ReadDataT, ParamsT, OutputT, IS_FOP, TF_)                     \
  using ParamsType = DEPAREN(ParamsT);                                           \
  using ReadDataType = DEPAREN(ReadDataT);                                       \
  using InstanceType = ReadType;                                                 \
  using OutputType = DEPAREN(OutputT);                                           \
  static constexpr bool IS_FUSED_OP = DEPAREN(IS_FOP);                           \
  static constexpr bool THREAD_FUSION = DEPAREN(TF_);

#define READ_OPERATION(NAME, READ_DATA_TYPE, ParamsT, OutputT, IS_FOP, TF_) \
      READ_ALIAS(READ_DATA_TYPE, ParamsT, OutputT, IS_FOP, TF_)             \
      using OperationDataType = OperationData<NAME>;

#define TEMPLATE_READ_OPERATION(NAME, TEMPLATE_PARAMS, READ_DATA_TYPE, ParamsT, OutputT, IS_FOP, TF_) \
        READ_ALIAS(READ_DATA_TYPE, ParamsT, OutputT, IS_FOP, TF_)                                     \
        using OperationDataType = OperationData<NAME<DEPAREN(TEMPLATE_PARAMS)>>;

#define READ_EXEC                                                                          \
  template <uint ELEMS_PER_THREAD = 1>                                                     \
  FK_HOST_DEVICE_FUSE auto exec(const Point& thread, const OperationDataType& opData)      \
    -> ThreadFusionType<ReadDataType, ELEMS_PER_THREAD, OutputType> {                      \
      if constexpr (std::bool_constant<THREAD_FUSION>::value) {                            \
          return exec<ELEMS_PER_THREAD>(thread, opData.params);                            \
      } else {                                                                             \
          return exec(thread, opData.params);                                              \
      }                                                                                    \
  }

#define READ_OPERATION_EXEC(NAME, READ_DATA_TYPE, ParamsT, OutputT, IS_FOP, TF_) \
      FK_STATIC_STRUCT(NAME, NAME)                                               \
      READ_OPERATION(NAME, READ_DATA_TYPE, ParamsT, OutputT, IS_FOP, TF_)        \
      READ_EXEC

#define TEMPLATE_READ_OPERATION_EXEC(NAME, TEMPLATE_PARAMS, READ_DATA_TYPE, ParamsT, OutputT, IS_FOP, TF_) \
        FK_STATIC_STRUCT_(NAME, (NAME<DEPAREN(TEMPLATE_PARAMS)>))                                          \
        TEMPLATE_READ_OPERATION(NAME, TEMPLATE_PARAMS, READ_DATA_TYPE, ParamsT, OutputT, IS_FOP, TF_)      \
        READ_EXEC

#define WRITE_ALIAS(InputT, ParamsT, WriteDataT, IS_FOP, TF_)   \
  using ParamsType = DEPAREN(ParamsT);                          \
  using InputType = DEPAREN(InputT);                            \
  using WriteDataType = DEPAREN(WriteDataT);                    \
  using InstanceType = WriteType;                               \
  static constexpr bool IS_FUSED_OP = DEPAREN(IS_FOP);          \
  static constexpr bool THREAD_FUSION = DEPAREN(TF_);

#define WRITE_OPERATION(NAME, InputT, ParamsT, WRITE_DATA_TYPE, IS_FOP, TF_) \
    WRITE_ALIAS(InputT, ParamsT, WRITE_DATA_TYPE, IS_FOP, TF_)               \
    using OperationDataType = OperationData<NAME>;

#define TEMPLATE_WRITE_OPERATION(NAME, TEMPLATE_PARAMS, InputT, ParamsT, WRITE_DATA_TYPE, IS_FOP, TF_) \
    WRITE_ALIAS(InputT, ParamsT, WRITE_DATA_TYPE, IS_FOP, TF_)                                         \
    using OperationDataType = OperationData<NAME<DEPAREN(TEMPLATE_PARAMS)>>;

#define WRITE_EXEC                                                                                                  \
  template <uint ELEMS_PER_THREAD = 1>                                                                              \
  FK_HOST_DEVICE_FUSE void exec(const Point& thread,                                                                \
                                const ThreadFusionType<InputType, ELEMS_PER_THREAD, WriteDataType>& input,          \
                                const OperationDataType& opData) {                                                  \
      if constexpr (THREAD_FUSION) {                                                                                \
          exec<ELEMS_PER_THREAD>(thread, input, opData.params);                                                     \
      } else {                                                                                                      \
          exec(thread, input, opData.params);                                                                       \
      }                                                                                                             \
  }

#define WRITE_OPERATION_EXEC(NAME, InputT, ParamsT, WRITE_DATA_TYPE, IS_FOP, TF_) \
        FK_STATIC_STRUCT(NAME, NAME)                                              \
        WRITE_OPERATION(NAME, InputT, ParamsT, WRITE_DATA_TYPE, IS_FOP, TF_)      \
        WRITE_EXEC

#define TEMPLATE_WRITE_OPERATION_EXEC(NAME, TEMPLATE_PARAMS, InputT, ParamsT, WRITE_DATA_TYPE, IS_FOP, TF_) \
        FK_STATIC_STRUCT_(NAME, (NAME<DEPAREN(TEMPLATE_PARAMS)>))                                           \
        TEMPLATE_WRITE_OPERATION(NAME, TEMPLATE_PARAMS, InputT, ParamsT, WRITE_DATA_TYPE, IS_FOP, TF_)      \
        WRITE_EXEC

#define READBACK_ALIAS(ReadDataT, ParamsT, BackIOpT, OutputT, IS_FOP, TF_)              \
  using ReadDataType = DEPAREN(ReadDataT);                                              \
  using ParamsType = DEPAREN(ParamsT);                                                  \
  using BackIOp = DEPAREN(BackIOpT);                                                    \
  using InstanceType = ReadBackType;                                                    \
  using OutputType = DEPAREN(OutputT);                                                  \
  static constexpr bool IS_FUSED_OP = DEPAREN(IS_FOP);                                  \
  static constexpr bool THREAD_FUSION = DEPAREN(TF_);

#define READBACK_OPERATION(NAME, READ_DATA_TYPE, ParamsT, BackIOpT, OutputT, IS_FOP, TF_) \
  READBACK_ALIAS(READ_DATA_TYPE, ParamsT, BackIOpT, OutputT, IS_FOP, TF_)                 \
  using OperationDataType = OperationData<NAME>;

#define TEMPLATE_READBACK_OPERATION(NAME, TEMPLATE_PARAMS, READ_DATA_TYPE, ParamsT, BackIOpT, \
                                    OutputT, IS_FOP, TF_)                                     \
  READBACK_ALIAS(READ_DATA_TYPE, ParamsT, BackIOpT, OutputT, IS_FOP, TF_)                     \
  using OperationDataType = OperationData<NAME<DEPAREN(TEMPLATE_PARAMS)>>;

#define READBACK_EXEC                                                                        \
    template <uint ELEMS_PER_THREAD = 1>                                                     \
    FK_HOST_DEVICE_FUSE auto exec(const Point& thread, const OperationDataType& opData)      \
        -> ThreadFusionType<ReadDataType, ELEMS_PER_THREAD, OutputType> {                    \
        if constexpr (std::bool_constant<THREAD_FUSION>::value) {                            \
            return exec<ELEMS_PER_THREAD>(thread, opData.params, opData.backIOp);            \
        } else {                                                                             \
            return exec(thread, opData.params, opData.backIOp);                              \
        }                                                                                    \
    }

#define READBACK_OPERATION_EXEC(NAME, READ_DATA_TYPE, ParamsT, BackIOpT, OutputT, IS_FOP, TF_) \
        FK_STATIC_STRUCT(NAME, NAME)                                                           \
        READBACK_OPERATION(NAME, READ_DATA_TYPE, ParamsT, BackIOpT, OutputT, IS_FOP, TF_)      \
        READBACK_EXEC

#define TEMPLATE_READBACK_OPERATION_EXEC(NAME, TEMPLATE_PARAMS, READ_DATA_TYPE, ParamsT, BackIOpT, \
                                         OutputT, IS_FOP, TF_)                                     \
        FK_STATIC_STRUCT_(NAME, (NAME<DEPAREN(TEMPLATE_PARAMS)>))                                  \
        TEMPLATE_READBACK_OPERATION(NAME, TEMPLATE_PARAMS, READ_DATA_TYPE, ParamsT, BackIOpT,      \
                                    OutputT, IS_FOP, TF_)                                          \
        READBACK_EXEC

    /*struct NumElems {
        template <typename IOp>
        FK_HOST_DEVICE_FUSE uint x(const Point& thread, const IOp& iOp) {
            static_assert(isAnyReadType<IOp> || isTernaryType<IOp>, "Only Read, ReadBack, IncompleteReadBack and Ternary Types work with NumElems::x");
            return IOp::Operation::num_elems_x(thread, iOp);
        }
        template <typename IOp>
        FK_HOST_DEVICE_FUSE uint y(const Point& thread, const IOp& iOp) {
            static_assert(isAnyReadType<IOp> || isTernaryType<IOp>, "Only Read, ReadBack, IncompleteReadBack and Ternary Types work with NumElems::y");
            return IOp::Operation::num_elems_y(thread, iOp);
        }
        template <typename IOp>
        FK_HOST_DEVICE_FUSE Size size(const Point& thread, const IOp& iOp) {
            static_assert(isAnyReadType<IOp> || isTernaryType<IOp>, "Only Read, ReadBack, IncompleteReadBack and Ternary Types work with NumElems::size");
            return Size(x(thread, iOp), y(thread, iOp));
        }
        template <typename IOp>
        FK_HOST_DEVICE_FUSE uint z(const Point& thread, const IOp& iOp) {
            static_assert(isAnyReadType<IOp> || isTernaryType<IOp>, "Only Read, ReadBack, IncompleteReadBack and Ternary Types work with NumElems::z");
            return IOp::Operation::num_elems_z(thread, iOp);
        }
    };*/
} // namespace fk

#endif // FK_PARENT_OPERATIONS_EXEC_H