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
    template <typename I, typename O, typename UOperationImpl, bool IS_FUSED = false> 
    struct UnaryOperation {
    private:
        using SelfType = UnaryOperation<I, O, UOperationImpl, IS_FUSED>;
    public:
        FK_STATIC_STRUCT_SELFTYPE(UnaryOperation, SelfType)
        using InputType = I;
        using OutputType = O;
        using InstanceType = UnaryType;
        using InstantiableType = UnaryInstantiableOperation<UOperationImpl>;
        static constexpr bool IS_FUSED_OP = IS_FUSED;
        // build() is fine, it only refers to UOperationImpl::InstantiableType
        // within the function body/return type, which is instantiated later.
        FK_HOST_DEVICE_FUSE auto build() { return typename UOperationImpl::InstantiableType{}; }
    };

#define DECLARE_UNARY_PARENT                                                                                           \
  using InputType = typename Parent::InputType;                                                                        \
  using OutputType = typename Parent::OutputType;                                                                      \
  using InstanceType = typename Parent::InstanceType;                                                                  \
  using InstantiableType = typename Parent::InstantiableType;                                                          \
  static constexpr bool IS_FUSED_OP = Parent::IS_FUSED_OP;                                                             \
  FK_HOST_DEVICE_FUSE InstantiableType build() { return Parent::build(); }

    template <typename I, typename P, typename O, typename BOperationImpl, bool IS_FUSED = false>
    struct BinaryOperation {
    private:
        using SelfType = BinaryOperation<I, P, O, BOperationImpl, IS_FUSED>;
    public:
        FK_STATIC_STRUCT_SELFTYPE(BinaryOperation, SelfType)
        // --- REMOVE using ALIASES that depend on BOperationImpl ---
        // These caused the incomplete type error during base class instantiation.
        // We will refer to BOperationImpl::TypeNeeded directly in methods.
        using InputType = I;
        using OutputType = O; // Needed for the static exec signature
        using ParamsType = P; // Needed by OperationData and build(params)
        using InstanceType = BinaryType;
        using OperationDataType = OperationData<BOperationImpl>; // Needed by exec/build(opData)
        using InstantiableType = BinaryInstantiableOperation<BOperationImpl>;
        static constexpr bool IS_FUSED_OP = IS_FUSED;
        // --- exec Method ---
        // Accesses types only in signature/body -> OK
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const OperationDataType& opData) {
            // Calls the static exec of the derived class (CRTP)
            return BOperationImpl::exec(input, opData.params);
            // Return type deduced via 'auto'
        }
        // --- build Methods ---
        // Accesses types only in signature/body -> OK
        FK_HOST_DEVICE_FUSE InstantiableType build(const OperationDataType& opData) {
            // Return type deduced via 'auto'
            return InstantiableType{ opData };
        }
        FK_HOST_DEVICE_FUSE InstantiableType build(const ParamsType& params) {
            // Return type deduced via 'auto'
            return InstantiableType{ {params} };
        }
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
  FK_HOST_DEVICE_FUSE InstantiableType build(const OperationDataType &opData) { return Parent::build(opData); }        \
  FK_HOST_DEVICE_FUSE InstantiableType build(const ParamsType &params) { return Parent::build(params); }
    template <typename I, typename P, typename BF, typename O, typename TOperationImpl, bool IS_FUSED = false>
    struct TernaryOperation {
    private:
        using SelfType = TernaryOperation<I, P, BF, O, TOperationImpl, IS_FUSED>;
    public:
        FK_STATIC_STRUCT_SELFTYPE(TernaryOperation, SelfType)
        using InputType = I;
        using OutputType = O;
        using ParamsType = P;
        using BackFunction = BF;
        using InstanceType = TernaryType;
        using OperationDataType = OperationData<TOperationImpl>;
        using InstantiableType = TernaryInstantiableOperation<TOperationImpl>;
        static constexpr bool IS_FUSED_OP = IS_FUSED;
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input, const OperationDataType& opData) {
            return TOperationImpl::exec(input, opData.params, opData.back_function);
        }
        FK_HOST_DEVICE_FUSE InstantiableType build(const OperationDataType& opData) { return InstantiableType{ opData }; }
        FK_HOST_DEVICE_FUSE InstantiableType build(const ParamsType& params, const BackFunction& backFunc) {
            return InstantiableType{ {params, backFunc} };
        }
    };

#define DECLARE_TERNARY_PARENT                                                                                         \
  using InputType = typename Parent::InputType;                                                                        \
  using OutputType = typename Parent::OutputType;                                                                      \
  using ParamsType = typename Parent::ParamsType;                                                                      \
  using BackFunction = typename Parent::BackFunction;                                                                  \
  using InstanceType = typename Parent::InstanceType;                                                                  \
  using OperationDataType = typename Parent::OperationDataType;                                                        \
  using InstantiableType = typename Parent::InstantiableType;                                                          \
  static constexpr bool IS_FUSED_OP = Parent::IS_FUSED_OP;                                                             \
  FK_HOST_DEVICE_FUSE OutputType exec(const InputType &input, const OperationDataType &opData) {                       \
    return Parent::exec(input, opData);                                                                                \
  }                                                                                                                    \
  FK_HOST_DEVICE_FUSE InstantiableType build(const OperationDataType &opData) { return Parent::build(opData); }        \
  FK_HOST_DEVICE_FUSE InstantiableType build(const ParamsType &params, const BackFunction &backFunc) {                 \
    return Parent::build(params, backFunc);                                                                            \
  }
    // END PARENT COMPUTE OPERATIONS
    // PARENT MEMORY OPERATIONS
    template <typename RT, typename P, typename O, enum TF TFE, typename ROperationImpl, bool IS_FUSED = false>
    struct ReadOperation {
    private:
        using SelfType = ReadOperation<RT, P, O, TFE, ROperationImpl, IS_FUSED>;
    public:
        FK_STATIC_STRUCT_SELFTYPE(ReadOperation, SelfType)
        using Child = ROperationImpl;
        using ParamsType = P;
        using ReadDataType = RT;
        using InstanceType = ReadType;
        static constexpr bool THREAD_FUSION{ static_cast<bool>(TFE) };
        using OutputType = O;
        using OperationDataType = OperationData<ROperationImpl>;
        using InstantiableType = Read<ROperationImpl>;
        static constexpr bool IS_FUSED_OP = IS_FUSED;

        template <uint ELEMS_PER_THREAD = 1, typename CH = Child>
        FK_DEVICE_FUSE ThreadFusionType<ReadDataType, ELEMS_PER_THREAD, OutputType> exec(const Point& thread,
            const OperationDataType& opData) {
            if constexpr (std::bool_constant<THREAD_FUSION>::value) {
                return ROperationImpl::template exec<ELEMS_PER_THREAD>(thread, opData.params);
            }
            else {
                return ROperationImpl::exec(thread, opData.params);
            }
        }

        FK_HOST_DEVICE_FUSE auto build(const OperationDataType& opData) { return InstantiableType{ opData }; }

        FK_HOST_DEVICE_FUSE auto build(const ParamsType& params) { return InstantiableType{ {params} }; };
    };

#define DECLARE_READ_PARENT_BASIC                                                                                      \
  using ParamsType = typename Parent::ParamsType;                                                                      \
  using ReadDataType = typename Parent::ReadDataType;                                                                  \
  using InstanceType = typename Parent::InstanceType;                                                                  \
  using OutputType = typename Parent::OutputType;                                                                      \
  using OperationDataType = typename Parent::OperationDataType;                                                        \
  using InstantiableType = typename Parent::InstantiableType;                                                          \
  static constexpr bool IS_FUSED_OP = Parent::IS_FUSED_OP;                                                             \
  static constexpr bool THREAD_FUSION = Parent::THREAD_FUSION;                                                         \
  template <uint ELEMS_PER_THREAD = 1>                                                                                 \
  FK_DEVICE_FUSE ThreadFusionType<ReadDataType, ELEMS_PER_THREAD, OutputType> exec(const Point &thread,                \
                                                                                   const OperationDataType &opData) {  \
    if constexpr (std::bool_constant<THREAD_FUSION>::value) {                                                          \
      return Parent::template exec<ELEMS_PER_THREAD>(thread, opData);                                                  \
    } else {                                                                                                           \
      return Parent::exec(thread, opData);                                                                             \
    }                                                                                                                  \
  }                                                                                                                    \
  FK_HOST_DEVICE_FUSE auto build(const OperationDataType &opData) { return Parent::build(opData); }                    \
  FK_HOST_DEVICE_FUSE auto build(const ParamsType &params) { return Parent::build(params); }

    template <typename I, typename P, typename WT, enum TF TFE, typename WOperationImpl, bool IS_FUSED = false>
    struct WriteOperation {
    private:
        using SelfType = WriteOperation<I, P, WT, TFE, WOperationImpl, IS_FUSED>;
    public:
        FK_STATIC_STRUCT_SELFTYPE(WriteOperation, SelfType)
        using Child = WOperationImpl;
        using ParamsType = P;
        using InputType = I;
        using WriteDataType = WT;
        using InstanceType = WriteType;
        static constexpr bool THREAD_FUSION{ static_cast<bool>(TFE) };
        using OperationDataType = OperationData<WOperationImpl>;
        using InstantiableType = Write<WOperationImpl>;
        static constexpr bool IS_FUSED_OP = IS_FUSED;
        template <uint ELEMS_PER_THREAD = 1>
        FK_HOST_DEVICE_FUSE void exec(const Point& thread,
                                      const ThreadFusionType<InputType, ELEMS_PER_THREAD, InputType>& input,
                                      const OperationDataType& opData) {
            if constexpr (THREAD_FUSION) {
                WOperationImpl::template exec<ELEMS_PER_THREAD>(thread, input, opData.params);
            } else {
                WOperationImpl::exec(thread, input, opData.params);
            }
        }
        FK_HOST_DEVICE_FUSE auto build(const OperationDataType& opData) { return InstantiableType{ opData }; }
        FK_HOST_DEVICE_FUSE auto build(const ParamsType& params) { return InstantiableType{ {params} }; };
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
  FK_HOST_DEVICE_FUSE auto build(const OperationDataType &opData) { return Parent::build(opData); }                    \
  FK_HOST_DEVICE_FUSE auto build(const ParamsType &params) { return Parent::build(params); }

    template <typename RT, typename P, typename B, typename O, typename RBOperationImpl, bool IS_FUSED = false>
    struct ReadBackOperation {
    private:
        using SelfType = ReadBackOperation<RT, P, B, O, RBOperationImpl, IS_FUSED>;
    public:
        FK_STATIC_STRUCT_SELFTYPE(ReadBackOperation, SelfType)
        using Child = RBOperationImpl;
        using ReadDataType = RT;
        using OutputType = O;
        using ParamsType = P;
        using BackFunction = B;
        using InstanceType = ReadBackType;
        using OperationDataType = OperationData<RBOperationImpl>;
        using InstantiableType = ReadBackInstantiableOperation<RBOperationImpl>;
        static constexpr bool IS_FUSED_OP = IS_FUSED;
        static constexpr bool THREAD_FUSION = false;

        template <typename BF = BackFunction>
        FK_DEVICE_FUSE std::enable_if_t<!std::is_same_v<BF, NullType>, OutputType> exec(const Point& thread,
            const OperationDataType& opData) {
            return RBOperationImpl::exec(thread, opData.params, opData.back_function);
        }
        FK_HOST_DEVICE_FUSE auto build(const OperationDataType& opData) { return InstantiableType{ opData }; }
        FK_HOST_DEVICE_FUSE auto build(const ParamsType& params, const BackFunction& backFunc) {
            return InstantiableType{ {params, backFunc} };
        };
    };

#define DECLARE_READBACK_PARENT_ALIAS                                                                                  \
  using ReadDataType = typename Parent::ReadDataType;                                                                  \
  using OutputType = typename Parent::OutputType;                                                                      \
  using ParamsType = typename Parent::ParamsType;                                                                      \
  using BackFunction = typename Parent::BackFunction;                                                                  \
  using InstanceType = typename Parent::InstanceType;                                                                  \
  using OperationDataType = typename Parent::OperationDataType;                                                        \
  using InstantiableType = typename Parent::InstantiableType;                                                          \
  static constexpr bool IS_FUSED_OP = Parent::IS_FUSED_OP;                                                             \
  static constexpr bool THREAD_FUSION = Parent::THREAD_FUSION;
    // END PARENT OPERATIONS
} // namespace fk

#endif