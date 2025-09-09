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

#ifndef FK_PARENT_OPERATIONS_BUILDERS_H
#define FK_PARENT_OPERATIONS_BUILDERS_H

#include <fused_kernel/core/execution_model/operation_model/parent_operations_exec.h>

namespace fk {
    // PARENT OPERATIONS
    // PARENT COMPUTE OPERATIONS
    template <typename I, typename O, typename ChildImplementation, bool IS_FUSED = false>
    struct UnaryOperationBuild {
    private:
        using SelfType = UnaryOperationBuild<I, O, ChildImplementation, IS_FUSED>;
    public:
        FK_STATIC_STRUCT(UnaryOperationBuild, SelfType)
        using Parent = UnaryOperationExec<I, O, ChildImplementation, IS_FUSED>;
        DECLARE_UNARY_PARENT_EXEC
        FK_HOST_FUSE auto build() { return typename Child::InstantiableType{}; }
    };

#define DECLARE_UNARY_PARENT_BUILD                                          \
  DECLARE_UNARY_PARENT_EXEC                                                 \
  FK_HOST_FUSE InstantiableType build() { return Parent::build(); }

    template <typename I, typename P, typename O, typename ChildImplementation, bool IS_FUSED = false>
    struct BinaryOperationBuild {
    private:
        using SelfType = BinaryOperationBuild<I, P, O, ChildImplementation, IS_FUSED>;
    public:
        FK_STATIC_STRUCT(BinaryOperationBuild, SelfType)
        using Parent = BinaryOperationExec<I, P, O, ChildImplementation, IS_FUSED>;
        DECLARE_BINARY_PARENT_EXEC
        FK_HOST_FUSE InstantiableType build(const OperationDataType& opData) { return InstantiableType{ opData }; }
        FK_HOST_FUSE InstantiableType build(const ParamsType& params) { return InstantiableType{ {params} }; }
    };

#define DECLARE_BINARY_PARENT_BUILD                                                                                    \
  DECLARE_BINARY_PARENT_EXEC                                                                                           \
  FK_HOST_FUSE InstantiableType build(const OperationDataType &opData) { return Parent::build(opData); }               \
  FK_HOST_FUSE InstantiableType build(const ParamsType &params) { return Parent::build(params); }

    template <typename I, typename P, typename BIOp, typename O, typename ChildImplementation, bool IS_FUSED = false>
    struct TernaryOperationExec {
    private:
        using SelfType = TernaryOperationExec<I, P, BIOp, O, ChildImplementation, IS_FUSED>;
    public:
        FK_STATIC_STRUCT(TernaryOperationExec, SelfType)
        using Parent = TernaryOperationExec<I, P, BIOp, O, ChildImplementation, IS_FUSED>;
        DECLARE_TERNARY_PARENT_EXEC

        FK_HOST_FUSE InstantiableType build(const OperationDataType& opData) { return InstantiableType{ opData }; }
        FK_HOST_FUSE InstantiableType build(const ParamsType& params, const BackIOp& backFunc) {
            return InstantiableType{ {params, backFunc} };
        }
    };

#define DECLARE_TERNARY_PARENT_BUILD                                                                                   \
  DECLARE_TERNARY_PARENT_EXEC                                                                                          \
  FK_HOST_FUSE InstantiableType build(const OperationDataType &opData) { return Parent::build(opData); }               \
  FK_HOST_FUSE InstantiableType build(const ParamsType &params, const BackIOp &backFunc) {                             \
      return Parent::build(params, backFunc);                                                                          \
  }

    template <typename RT, typename P, typename O, enum TF TFE, typename ChildImplementation, bool IS_FUSED = false>
    struct ReadOperationBuild {
    private:
        using SelfType = ReadOperationBuild<RT, P, O, TFE, ChildImplementation, IS_FUSED>;
    public:
        FK_STATIC_STRUCT(ReadOperationBuild, SelfType)
        DECLARE_READ_PARENT_EXEC
        FK_HOST_FUSE auto build(const OperationDataType& opData) { return InstantiableType{ opData }; }
        FK_HOST_FUSE auto build(const ParamsType& params) { return InstantiableType{ {params} }; }
    };

#define DECLARE_READ_PARENT_BUILD                                                             \
  DECLARE_READ_PARENT_EXEC                                                                    \
  FK_HOST_FUSE auto build(const OperationDataType &opData) { return Parent::build(opData); }  \
  FK_HOST_FUSE auto build(const ParamsType &params) { return Parent::build(params); }


} // namespace fk

#endif // FK_PARENT_OPERATIONS_BUILDERS_H