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

#ifndef FK_PARENT_OPERATIONS_CUH
#define FK_PARENT_OPERATIONS_CUH

#include <fused_kernel/core/execution_model/operation_model/instantiable_operations.h>

namespace fk {
    // PARENT OPERATIONS
    // PARENT COMPUTE OPERATIONS
    template <typename I, typename O, typename ChildImplementation, bool IS_FUSED = false>
    struct UnaryOperation {
    private:
        using SelfType = UnaryOperation<I, O, ChildImplementation, IS_FUSED>;
    public:
        FK_STATIC_STRUCT(UnaryOperation, SelfType)
        using Child = ChildImplementation;
        using InputType = I;
        using OutputType = O;
        using InstanceType = UnaryType;
        using InstantiableType = Unary<Child>;
        static constexpr bool IS_FUSED_OP = IS_FUSED;
        FK_HOST_FUSE auto build() { return typename Child::InstantiableType{}; }
    };

#define DECLARE_UNARY_PARENT                                                                                           \
  using InputType = typename Parent::InputType;                                                                        \
  using OutputType = typename Parent::OutputType;                                                                      \
  using InstanceType = typename Parent::InstanceType;                                                                  \
  using InstantiableType = typename Parent::InstantiableType;                                                          \
  static constexpr bool IS_FUSED_OP = Parent::IS_FUSED_OP;                                                             \
  FK_HOST_FUSE InstantiableType build() { return Parent::build(); }

    template <typename I, typename P, typename O, typename ChildImplementation, bool IS_FUSED = false>
    struct BinaryOperation {
    private:
        using SelfType = BinaryOperation<I, P, O, ChildImplementation, IS_FUSED>;
    public:
        FK_STATIC_STRUCT(BinaryOperation, SelfType)
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
        FK_HOST_FUSE InstantiableType build(const OperationDataType& opData) { return InstantiableType{ opData }; }
        FK_HOST_FUSE InstantiableType build(const ParamsType& params) { return InstantiableType{ {params} }; }
    };

#define DECLARE_BINARY_PARENT                                                                                          \
  using InputType = typename Parent::InputType;                                                                        \
  using OutputType = typename Parent::OutputType;                                                                      \
  using ParamsType = typename Parent::ParamsType;                                                                      \
  using InstanceType = typename Parent::InstanceType;                                                                  \
  using OperationDataType = typename Parent::OperationDataType;                                                        \
  using InstantiableType = typename Parent::InstantiableType;                                                          \
  static constexpr bool IS_FUSED_OP = Parent::IS_FUSED_OP;                                                             \
  FK_HOST_DEVICE_FUSE OutputType exec(const InputType &input, const OperationDataType &opData) {                       \
    return Parent::exec(input, opData);                                                                                \
  }                                                                                                                    \
  FK_HOST_FUSE InstantiableType build(const OperationDataType &opData) { return Parent::build(opData); }               \
  FK_HOST_FUSE InstantiableType build(const ParamsType &params) { return Parent::build(params); }

    template <typename I, typename P, typename BIOp, typename O, typename ChildImplementation, bool IS_FUSED = false>
    struct TernaryOperation {
    private:
        using SelfType = TernaryOperation<I, P, BIOp, O, ChildImplementation, IS_FUSED>;
    public:
        FK_STATIC_STRUCT(TernaryOperation, SelfType)
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
        FK_HOST_FUSE InstantiableType build(const OperationDataType& opData) { return InstantiableType{opData}; }
        FK_HOST_FUSE InstantiableType build(const ParamsType& params, const BackIOp& backFunc) {
            return InstantiableType{{params, backFunc}};
        }
    };

#define DECLARE_TERNARY_PARENT                                                                                         \
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
  }                                                                                                                    \
  FK_HOST_FUSE InstantiableType build(const OperationDataType &opData) { return Parent::build(opData); }               \
  FK_HOST_FUSE InstantiableType build(const ParamsType &params, const BackIOp &backFunc) {                             \
    return Parent::build(params, backFunc);                                                                            \
  }
    // END PARENT COMPUTE OPERATIONS
    // PARENT MEMORY OPERATIONS
    template <typename RT, typename P, typename O, enum TF TFE, typename ChildImplementation, bool IS_FUSED = false>
    struct ReadOperation {
    private:
        using SelfType = ReadOperation<RT, P, O, TFE, ChildImplementation, IS_FUSED>;
    public:
        FK_STATIC_STRUCT(ReadOperation, SelfType)
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
        FK_HOST_FUSE auto build(const OperationDataType& opData) { return InstantiableType{opData}; }
        FK_HOST_FUSE auto build(const ParamsType& params) { return InstantiableType{{params}}; }
    };

#define DECLARE_READ_PARENT_DEVICE_BASIC                                                                               \
  using ParamsType = typename Parent::ParamsType;                                                                      \
  using ReadDataType = typename Parent::ReadDataType;                                                                  \
  using InstanceType = typename Parent::InstanceType;                                                                  \
  using OutputType = typename Parent::OutputType;                                                                      \
  using OperationDataType = typename Parent::OperationDataType;                                                        \
  using InstantiableType = typename Parent::InstantiableType;                                                          \
  static constexpr bool IS_FUSED_OP = Parent::IS_FUSED_OP;                                                             \
  static constexpr bool THREAD_FUSION = Parent::THREAD_FUSION;                                                         \
  template <uint ELEMS_PER_THREAD = 1>                                                                                 \
  FK_HOST_DEVICE_FUSE ThreadFusionType<ReadDataType, ELEMS_PER_THREAD, OutputType>                                     \
  exec(const Point &thread, const OperationDataType &opData) {                                                         \
    if constexpr (std::bool_constant<THREAD_FUSION>::value) {                                                          \
      return Parent::template exec<ELEMS_PER_THREAD>(thread, opData);                                                  \
    } else {                                                                                                           \
      return Parent::exec(thread, opData);                                                                             \
    }                                                                                                                  \
  }

#ifndef NVRTC_COMPILER
#define DECLARE_READ_PARENT_BASIC                                                                                      \
  DECLARE_READ_PARENT_DEVICE_BASIC                                                                                     \
  FK_HOST_FUSE auto build(const OperationDataType &opData) { return Parent::build(opData); }                           \
  FK_HOST_FUSE auto build(const ParamsType &params) { return Parent::build(params); }
#else
#define DECLARE_READ_PARENT_BASIC                                                                                      \
DECLARE_READ_PARENT_DEVICE_BASIC
#endif // 

    template <typename I, typename P, typename WT, enum TF TFE, typename ChildImplementation, bool IS_FUSED = false>
    struct WriteOperation {
    private:
        using SelfType = WriteOperation<I, P, WT, TFE, ChildImplementation, IS_FUSED>;
    public:
        FK_STATIC_STRUCT(WriteOperation, SelfType)
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
        FK_HOST_FUSE auto build(const OperationDataType& opData) { return InstantiableType{ opData }; }
        FK_HOST_FUSE auto build(const ParamsType& params) { return InstantiableType{ {params} }; }
    };

#define DECLARE_WRITE_PARENT_BASIC                                                                                     \
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
  }                                                                                                                    \
  FK_HOST_FUSE auto build(const OperationDataType &opData) { return Parent::build(opData); }                           \
  FK_HOST_FUSE auto build(const ParamsType &params) { return Parent::build(params); }

    template <typename RT, typename P, typename B, typename O, typename ChildImplementation, bool IS_FUSED = false>
    struct ReadBackOperation {
    private:
        using SelfType = ReadBackOperation<RT, P, B, O, ChildImplementation, IS_FUSED>;
    public:
        FK_STATIC_STRUCT(ReadBackOperation, SelfType)
        using Child = ChildImplementation;
        using ReadDataType = RT;
        using OutputType = O;
        using ParamsType = P;
        using BackIOp = B;
        using InstanceType = ReadBackType;
        using OperationDataType = OperationData<Child>;
        using InstantiableType = ReadBack<Child>;
        static constexpr bool IS_FUSED_OP = IS_FUSED;
        static constexpr bool THREAD_FUSION = false;

        template <typename BIOp = BackIOp>
        FK_HOST_DEVICE_FUSE std::enable_if_t<!std::is_same_v<BIOp, NullType>, OutputType>
        exec(const Point& thread, const OperationDataType& opData) {
            return Child::exec(thread, opData.params, opData.backIOp);
        }
        FK_HOST_FUSE auto build(const OperationDataType& opData) { return InstantiableType{ opData }; }
        FK_HOST_FUSE auto build(const ParamsType& params, const BackIOp& backFunc) {
            return InstantiableType{{params, backFunc}};
        }
    };

#define DECLARE_READBACK_PARENT_ALIAS                                                                                  \
  using ReadDataType = typename Parent::ReadDataType;                                                                  \
  using OutputType = typename Parent::OutputType;                                                                      \
  using ParamsType = typename Parent::ParamsType;                                                                      \
  using BackIOp = typename Parent::BackIOp;                                                                            \
  using InstanceType = typename Parent::InstanceType;                                                                  \
  using OperationDataType = typename Parent::OperationDataType;                                                        \
  using InstantiableType = typename Parent::InstantiableType;                                                          \
  static constexpr bool IS_FUSED_OP = Parent::IS_FUSED_OP;                                                             \
  static constexpr bool THREAD_FUSION = Parent::THREAD_FUSION;
    // END PARENT OPERATIONS
} // namespace fk

#endif