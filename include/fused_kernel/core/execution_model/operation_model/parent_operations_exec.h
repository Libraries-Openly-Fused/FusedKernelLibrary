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

#include <fused_kernel/core/execution_model/operation_model/instantiable_operations.h>

namespace fk {
    // PARENT OPERATIONS
    // PARENT COMPUTE OPERATIONS
    template <typename I, typename O, typename ChildImplementation, bool IS_FUSED = false>
    struct UnaryOperationExec {
    private:
        using SelfType = UnaryOperationExec<I, O, ChildImplementation, IS_FUSED>;
    public:
        FK_STATIC_STRUCT(UnaryOperationExec, SelfType)
        using Child = ChildImplementation;
        using InputType = I;
        using OutputType = O;
        using InstanceType = UnaryType;
        using InstantiableType = Unary<Child>;
        static constexpr bool IS_FUSED_OP = IS_FUSED;
    };
#define DECLARE_UNARY_PARENT_EXEC                                        \
  using InputType = typename Parent::InputType;                          \
  using OutputType = typename Parent::OutputType;                        \
  using InstanceType = typename Parent::InstanceType;                    \
  using InstantiableType = typename Parent::InstantiableType;            \
  static constexpr bool IS_FUSED_OP = Parent::IS_FUSED_OP;

    template <typename I, typename P, typename O, typename ChildImplementation, bool IS_FUSED = false>
    struct BinaryOperationExec {
    private:
        using SelfType = BinaryOperationExec<I, P, O, ChildImplementation, IS_FUSED>;
    public:
        FK_STATIC_STRUCT(BinaryOperationExec, SelfType)
        using Child = ChildImplementation;
        using InputType = I;
        using OutputType = O;
        using ParamsType = P;
        using InstanceType = BinaryType;
        using OperationDataType = OperationData<Child>;
        using InstantiableType = Binary<Child>;
        static constexpr bool IS_FUSED_OP = IS_FUSED;
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const OperationDataType& opData) {
            return Child::exec(input, opData.params);
        }
    };

#define DECLARE_BINARY_PARENT_EXEC                                                               \
  using InputType = typename Parent::InputType;                                                  \
  using OutputType = typename Parent::OutputType;                                                \
  using ParamsType = typename Parent::ParamsType;                                                \
  using InstanceType = typename Parent::InstanceType;                                            \
  using OperationDataType = typename Parent::OperationDataType;                                  \
  using InstantiableType = typename Parent::InstantiableType;                                    \
  static constexpr bool IS_FUSED_OP = Parent::IS_FUSED_OP;                                       \
  FK_HOST_DEVICE_FUSE OutputType exec(const InputType &input, const OperationDataType &opData) { \
      return Parent::exec(input, opData);                                                        \
  }

    template <typename I, typename P, typename BIOp, typename O, typename ChildImplementation, bool IS_FUSED = false>
    struct TernaryOperationExec {
    private:
        using SelfType = TernaryOperationExec<I, P, BIOp, O, ChildImplementation, IS_FUSED>;
    public:
        FK_STATIC_STRUCT(TernaryOperationExec, SelfType)
        using Child = ChildImplementation;
        using InputType = I;
        using OutputType = O;
        using ParamsType = P;
        using BackIOp = BIOp;
        using InstanceType = TernaryType;
        using OperationDataType = OperationData<Child>;
        using InstantiableType = Ternary<Child>;
        static constexpr bool IS_FUSED_OP = IS_FUSED;

        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const OperationDataType& opData) {
            return Child::exec(input, opData.params, opData.backIOp);
        }
    };

#define DECLARE_TERNARY_PARENT_EXEC                                                                                    \
  using InputType = typename Parent::InputType;                                                                        \
  using OutputType = typename Parent::OutputType;                                                                      \
  using ParamsType = typename Parent::ParamsType;                                                                      \
  using BackIOp = typename Parent::BackIOp;                                                                            \
  using InstanceType = typename Parent::InstanceType;                                                                  \
  using OperationDataType = typename Parent::OperationDataType;                                                        \
  using InstantiableType = typename Parent::InstantiableType;                                                          \
  static constexpr bool IS_FUSED_OP = Parent::IS_FUSED_OP;                                                             \
  FK_HOST_DEVICE_FUSE OutputType exec(const InputType &input, const OperationDataType &opData) {                       \
    return Parent::exec(input, opData);                                                                                \
  }

    // END PARENT COMPUTE OPERATIONS
    // PARENT MEMORY OPERATIONS
    template <typename RT, typename P, typename O, enum TF TFE, typename ChildImplementation, bool IS_FUSED = false>
    struct ReadOperationExec {
    private:
        using SelfType = ReadOperationExec<RT, P, O, TFE, ChildImplementation, IS_FUSED>;
    public:
        FK_STATIC_STRUCT(ReadOperationExec, SelfType)
        using Child = ChildImplementation;
        using ParamsType = P;
        using ReadDataType = RT;
        using InstanceType = ReadType;
        static constexpr bool THREAD_FUSION{ static_cast<bool>(TFE) };
        using OutputType = O;
        using OperationDataType = OperationData<Child>;
        using InstantiableType = Read<Child>;
        static constexpr bool IS_FUSED_OP = IS_FUSED;

        template <uint ELEMS_PER_THREAD = 1>
        FK_HOST_DEVICE_FUSE ThreadFusionType<ReadDataType, ELEMS_PER_THREAD, OutputType>
            exec(const Point& thread, const OperationDataType& opData) {
            if constexpr (std::bool_constant<THREAD_FUSION>::value) {
                return Child::template exec<ELEMS_PER_THREAD>(thread, opData.params);
            } else {
                return Child::exec(thread, opData.params);
            }
        }
    };

#define DECLARE_READ_PARENT_EXEC                                                           \
  using ParamsType = typename Parent::ParamsType;                                          \
  using ReadDataType = typename Parent::ReadDataType;                                      \
  using InstanceType = typename Parent::InstanceType;                                      \
  using OutputType = typename Parent::OutputType;                                          \
  using OperationDataType = typename Parent::OperationDataType;                            \
  using InstantiableType = typename Parent::InstantiableType;                              \
  static constexpr bool IS_FUSED_OP = Parent::IS_FUSED_OP;                                 \
  static constexpr bool THREAD_FUSION = Parent::THREAD_FUSION;                             \
  template <uint ELEMS_PER_THREAD = 1>                                                     \
  FK_HOST_DEVICE_FUSE ThreadFusionType<ReadDataType, ELEMS_PER_THREAD, OutputType>         \
  exec(const Point &thread, const OperationDataType &opData) {                             \
    if constexpr (std::bool_constant<THREAD_FUSION>::value) {                              \
      return Parent::template exec<ELEMS_PER_THREAD>(thread, opData);                      \
    } else {                                                                               \
      return Parent::exec(thread, opData);                                                 \
    }                                                                                      \
  }

    template <typename I, typename P, typename WT, enum TF TFE, typename ChildImplementation, bool IS_FUSED = false>
    struct WriteOperationExec {
    private:
        using SelfType = WriteOperationExec<I, P, WT, TFE, ChildImplementation, IS_FUSED>;
    public:
        FK_STATIC_STRUCT(WriteOperationExec, SelfType)
        using Child = ChildImplementation;
        using ParamsType = P;
        using InputType = I;
        using WriteDataType = WT;
        using InstanceType = WriteType;
        static constexpr bool THREAD_FUSION{ static_cast<bool>(TFE) };
        using OperationDataType = OperationData<Child>;
        using InstantiableType = Write<Child>;
        static constexpr bool IS_FUSED_OP = IS_FUSED;

        template <uint ELEMS_PER_THREAD = 1>
        FK_HOST_DEVICE_FUSE void exec(const Point& thread,
            const ThreadFusionType<InputType, ELEMS_PER_THREAD, InputType>& input,
            const OperationDataType& opData) {
            if constexpr (THREAD_FUSION) {
                Child::template exec<ELEMS_PER_THREAD>(thread, input, opData.params);
            } else {
                Child::exec(thread, input, opData.params);
            }
        }
    };

#define DECLARE_WRITE_PARENT_BASIC_EXEC                                                                                \
  using ParamsType = typename Parent::ParamsType;                                                                      \
  using InputType = typename Parent::InputType;                                                                        \
  using WriteDataType = typename Parent::WriteDataType;                                                                \
  using InstanceType = typename Parent::InstanceType;                                                                  \
  using OperationDataType = typename Parent::OperationDataType;                                                        \
  using InstantiableType = typename Parent::InstantiableType;                                                          \
  static constexpr bool IS_FUSED_OP = Parent::IS_FUSED_OP;                                                             \
  static constexpr bool THREAD_FUSION = Parent::THREAD_FUSION;                                                         \
  template <uint ELEMS_PER_THREAD = 1>                                                                                 \
  FK_HOST_DEVICE_FUSE void exec(const Point &thread,                                                                   \
                                const ThreadFusionType<InputType, ELEMS_PER_THREAD, InputType> &input,                 \
                                const OperationDataType &opData) {                                                     \
    Parent::template exec<ELEMS_PER_THREAD>(thread, input, opData);                                                    \
  }

    template <typename RT, typename P, typename B, typename O, typename ChildImplementation>
    struct ReadBackOperationExec {
    private:
        using SelfType = ReadBackOperationExec<RT, P, B, O, ChildImplementation>;
    public:
        FK_STATIC_STRUCT(ReadBackOperationExec, SelfType)
        using Child = ChildImplementation;
        using ReadDataType = RT;
        using OutputType = O;
        using ParamsType = P;
        using BackIOp = B;
        using InstanceType = ReadBackType;
        using OperationDataType = OperationData<Child>;
        using InstantiableType = ReadBack<Child>;
        static constexpr bool IS_FUSED_OP = false;
        static constexpr bool THREAD_FUSION = false;

        template <typename BIOp = BackIOp>
        FK_HOST_DEVICE_FUSE std::enable_if_t<!std::is_same_v<BIOp, NullType>, OutputType>
            exec(const Point& thread, const OperationDataType& opData) {
            return Child::exec(thread, opData.params, opData.backIOp);
        }
    };

#define DECLARE_READBACK_PARENT_BASIC                                                                                  \
  using ReadDataType = typename Parent::ReadDataType;                                                                  \
  using OutputType = typename Parent::OutputType;                                                                      \
  using ParamsType = typename Parent::ParamsType;                                                                      \
  using BackIOp = typename Parent::BackIOp;                                                                            \
  using InstanceType = typename Parent::InstanceType;                                                                  \
  using OperationDataType = typename Parent::OperationDataType;                                                        \
  using InstantiableType = typename Parent::InstantiableType;                                                          \
  static constexpr bool IS_FUSED_OP = Parent::IS_FUSED_OP;                                                             \
  static constexpr bool THREAD_FUSION = Parent::THREAD_FUSION;                                                         \
  FK_HOST_DEVICE_FUSE OutputType exec(const Point &thread, const OperationDataType &opData) {                          \
    return Parent::exec(thread, opData);                                                                               \
  }

    struct NumElems {
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
    };
} // namespace fk

#endif // FK_PARENT_OPERATIONS_EXEC_H