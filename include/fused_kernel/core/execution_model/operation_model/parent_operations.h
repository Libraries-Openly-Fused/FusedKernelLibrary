/* Copyright 2023-2026 Oscar Amoros Huguet
*  Copyright 2026 Grup Mediapro S.L.U (Oscar Amoros Huguet)

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

/* Notes on implementation decisions
* 
*  Having exec and build functions defined in macros:
*   - It may be inconvenient sometimes for debugging.
*   - The alternative is to have them defined in the parent operations and then call them from
*     the macro exec and build definitions. This option, we observed that increases compilation
*     times.
*   - We may decide to move the exec functions that have threa fusion management to the parent
*     operations, or do it temporarily just for debugging.
*/

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
    };

#define DECLARE_UNARY_PARENT                                    \
  using InputType = typename Parent::InputType;                 \
  using OutputType = typename Parent::OutputType;               \
  using InstanceType = typename Parent::InstanceType;           \
  using InstantiableType = typename Parent::InstantiableType;   \
  static constexpr bool IS_FUSED_OP = Parent::IS_FUSED_OP;      \
  FK_HOST_FUSE InstantiableType build() { return {}; }

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
    };

#define DECLARE_BINARY_PARENT                                                                    \
  using InputType = typename Parent::InputType;                                                  \
  using OutputType = typename Parent::OutputType;                                                \
  using ParamsType = typename Parent::ParamsType;                                                \
  using InstanceType = typename Parent::InstanceType;                                            \
  using OperationDataType = typename Parent::OperationDataType;                                  \
  using InstantiableType = typename Parent::InstantiableType;                                    \
  static constexpr bool IS_FUSED_OP = Parent::IS_FUSED_OP;                                       \
  FK_HOST_DEVICE_FUSE OutputType exec(const InputType input, const OperationDataType &opData) { \
    return exec(input, opData.params);                                                           \
  }                                                                                              \
  FK_HOST_FUSE InstantiableType build(const OperationDataType& opData) { return {opData}; }      \
  FK_HOST_FUSE InstantiableType build(const ParamsType& params) { return {{params}}; }

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
    };

#define DECLARE_TERNARY_PARENT                                                                    \
  using InputType = typename Parent::InputType;                                                   \
  using OutputType = typename Parent::OutputType;                                                 \
  using ParamsType = typename Parent::ParamsType;                                                 \
  using BackIOp = typename Parent::BackIOp;                                                       \
  using InstanceType = typename Parent::InstanceType;                                             \
  using OperationDataType = typename Parent::OperationDataType;                                   \
  using InstantiableType = typename Parent::InstantiableType;                                     \
  static constexpr bool IS_FUSED_OP = Parent::IS_FUSED_OP;                                        \
  FK_HOST_DEVICE_FUSE OutputType exec(const InputType input, const OperationDataType &opData) {  \
    return exec(input, opData.params, opData.backIOp);                                            \
  }                                                                                               \
  FK_HOST_FUSE InstantiableType build(const OperationDataType &opData) { return {opData}; }       \
  FK_HOST_FUSE InstantiableType build(const ParamsType &params, const BackIOp &backFunc) {        \
    return {{params, backFunc}};                                                                  \
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
    };

#define DECLARE_READ_PARENT_DEVICE_BASIC                                                       \
  using ParamsType = typename Parent::ParamsType;                                              \
  using ReadDataType = typename Parent::ReadDataType;                                          \
  using InstanceType = typename Parent::InstanceType;                                          \
  using OutputType = typename Parent::OutputType;                                              \
  using OperationDataType = typename Parent::OperationDataType;                                \
  using InstantiableType = typename Parent::InstantiableType;                                  \
  static constexpr bool IS_FUSED_OP = Parent::IS_FUSED_OP;                                     \
  static constexpr bool THREAD_FUSION = Parent::THREAD_FUSION;                                 \
  template <uint ELEMS_PER_THREAD = 1>                                                         \
  FK_HOST_DEVICE_FUSE auto exec(const Point thread, const OperationDataType& opData) {        \
    if constexpr (std::bool_constant<THREAD_FUSION>::value) {                                  \
      return exec<ELEMS_PER_THREAD>(thread, opData.params);                                    \
    } else {                                                                                   \
      return exec(thread, opData.params);                                                      \
    }                                                                                          \
  }

#define DECLARE_READ_PARENT_BASIC                                                              \
  DECLARE_READ_PARENT_DEVICE_BASIC                                                             \
  FK_HOST_FUSE InstantiableType build(const OperationDataType& opData) { return {opData}; }    \
  FK_HOST_FUSE InstantiableType build(const ParamsType& params) { return {{params}}; }

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
    };

#define DECLARE_WRITE_PARENT_BASIC                                                                      \
  using ParamsType = typename Parent::ParamsType;                                                       \
  using InputType = typename Parent::InputType;                                                         \
  using WriteDataType = typename Parent::WriteDataType;                                                 \
  using InstanceType = typename Parent::InstanceType;                                                   \
  using OperationDataType = typename Parent::OperationDataType;                                         \
  using InstantiableType = typename Parent::InstantiableType;                                           \
  static constexpr bool IS_FUSED_OP = Parent::IS_FUSED_OP;                                              \
  static constexpr bool THREAD_FUSION = Parent::THREAD_FUSION;                                          \
  template <uint ELEMS_PER_THREAD = 1>                                                                  \
  FK_HOST_DEVICE_FUSE void exec(const Point thread,                                                    \
                                const ThreadFusionType<InputType, ELEMS_PER_THREAD, InputType> &input,  \
                                const OperationDataType &opData) {                                      \
    if constexpr (THREAD_FUSION) {                                                                      \
        exec<ELEMS_PER_THREAD>(thread, input, opData.params);                                           \
    } else {                                                                                            \
        exec(thread, input, opData.params);                                                             \
    }                                                                                                   \
  }                                                                                                     \
  FK_HOST_FUSE InstantiableType build(const OperationDataType &opData) { return {opData}; }             \
  FK_HOST_FUSE InstantiableType build(const ParamsType &params) { return {{params}}; }

    template <typename I, typename P, typename O, typename ChildImplementation, bool IS_FUSED = false>
    struct OpenOperationParent {
    private:
        using SelfType = OpenOperationParent<I, P, O, ChildImplementation, IS_FUSED>;
    public:
        FK_STATIC_STRUCT(OpenOperationParent, SelfType)
        using Child = ChildImplementation;
        using ParamsType = P;
        using InputType = I;
        using OutputType = O;
        using InstanceType = OpenType;
        using OperationDataType = OperationData<Child>;
        using InstantiableType = Open<Child>;
        static constexpr bool IS_FUSED_OP = IS_FUSED;
    };

#define DECLARE_OPEN_PARENT                                                                 \
  using ParamsType = typename Parent::ParamsType;                                           \
  using InputType = typename Parent::InputType;                                             \
  using OutputType = typename Parent::OutputType;                                           \
  using InstanceType = typename Parent::InstanceType;                                       \
  using OperationDataType = typename Parent::OperationDataType;                             \
  using InstantiableType = typename Parent::InstantiableType;                               \
  static constexpr bool IS_FUSED_OP = Parent::IS_FUSED_OP;                                  \
  FK_HOST_DEVICE_FUSE OutputType exec(const Point thread,                                  \
                                      const InputType input,                               \
                                      const OperationDataType& opData) {                    \
    return exec(thread, input, opData.params);                                              \
  }                                                                                         \
  FK_HOST_FUSE InstantiableType build(const OperationDataType &opData) { return {opData}; } \
  FK_HOST_FUSE InstantiableType build(const ParamsType &params) { return {{params}}; }

template <typename P, typename ChildImplementation, bool IS_FUSED = false>
struct ClosedOperation {
    private:
    using SelfType = ClosedOperation<P, ChildImplementation, IS_FUSED>;

    public:
    FK_STATIC_STRUCT(ClosedOperation, SelfType)
    using Child = ChildImplementation;
    using ParamsType = P;
    using InstanceType = ClosedType;
    using OperationDataType = OperationData<Child>;
    using InstantiableType = Closed<Child>;
    static constexpr bool IS_FUSED_OP = IS_FUSED;
};

#define DECLARE_CLOSED_PARENT                                                                   \
    using ParamsType = typename Parent::ParamsType;                                             \
    using InstanceType = typename Parent::InstanceType;                                         \
    using OperationDataType = typename Parent::OperationDataType;                               \
    using InstantiableType = typename Parent::InstantiableType;                                 \
    static constexpr bool IS_FUSED_OP = Parent::IS_FUSED_OP;                                    \
    FK_HOST_DEVICE_FUSE void exec(const Point thread, const OperationDataType &opData) {       \
        exec(thread, opData.params);                                                            \
    }                                                                                           \
    FK_HOST_FUSE InstantiableType build(const OperationDataType &opData) { return { opData }; } \
    FK_HOST_FUSE InstantiableType build(const ParamsType &params) { return {{ params }}; }

    template <typename RT, typename P, typename B, typename O, typename ChildImplementation>
    struct ReadBackOperation {
    private:
        using SelfType = ReadBackOperation<RT, P, B, O, ChildImplementation>;
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
        static constexpr bool IS_FUSED_OP = false;
        static constexpr bool THREAD_FUSION = false;
    };

#define DECLARE_READBACK_PARENT_BASIC                                                          \
  using ReadDataType = typename Parent::ReadDataType;                                          \
  using OutputType = typename Parent::OutputType;                                              \
  using ParamsType = typename Parent::ParamsType;                                              \
  using BackIOp = typename Parent::BackIOp;                                                    \
  using InstanceType = typename Parent::InstanceType;                                          \
  using OperationDataType = typename Parent::OperationDataType;                                \
  using InstantiableType = typename Parent::InstantiableType;                                  \
  static constexpr bool IS_FUSED_OP = Parent::IS_FUSED_OP;                                     \
  static constexpr bool THREAD_FUSION = Parent::THREAD_FUSION;                                 \
  FK_HOST_DEVICE_FUSE OutputType exec(const Point thread, const OperationDataType &opData) {  \
    return exec(thread, opData.params, opData.backIOp);                                        \
  }                                                                                            \
  FK_HOST_FUSE InstantiableType build(const OperationDataType &opData) { return {opData}; }    \
  FK_HOST_FUSE InstantiableType build(const ParamsType &params, const BackIOp &backIOp) {      \
    return {{params, backIOp}};                                                                \
  }

    template <typename RT, typename P, typename B, typename O, typename ChildImplementation>
    struct IncompleteReadBackOperation {
    private:
        using SelfType = IncompleteReadBackOperation<RT, P, B, O, ChildImplementation>;
    public:
        FK_STATIC_STRUCT(IncompleteReadBackOperation, SelfType)
        using Child = ChildImplementation;
        using ReadDataType = RT;
        using OutputType = O;
        using ParamsType = P;
        using BackIOp = B;
        using InstanceType = IncompleteReadBackType;
        using OperationDataType = OperationData<Child>;
        using InstantiableType = IncompleteReadBack<Child>;
        static constexpr bool IS_FUSED_OP = false;
        static constexpr bool THREAD_FUSION = false;
    };
#define DECLARE_INCOMPLETEREADBACK_PARENT_BASIC                                             \
  using ReadDataType = typename Parent::ReadDataType;                                       \
  using OutputType = typename Parent::OutputType;                                           \
  using ParamsType = typename Parent::ParamsType;                                           \
  using BackIOp = typename Parent::BackIOp;                                                 \
  using InstanceType = typename Parent::InstanceType;                                       \
  using OperationDataType = typename Parent::OperationDataType;                             \
  using InstantiableType = typename Parent::InstantiableType;                               \
  static constexpr bool IS_FUSED_OP = Parent::IS_FUSED_OP;                                  \
  static constexpr bool THREAD_FUSION = Parent::THREAD_FUSION;                              \
  FK_HOST_FUSE InstantiableType build(const OperationDataType &opData) { return {opData}; } \
  FK_HOST_FUSE InstantiableType build(const ParamsType &params, const BackIOp &backFunc) {  \
    return {{params, backFunc}};                                                            \
  }
// END PARENT OPERATIONS

    struct NumElems {
        template <typename IOp>
        FK_HOST_DEVICE_FUSE uint x(const Point thread, const IOp& iOp) {
            static_assert(isAnyReadType<IOp> || opIs<TernaryType, IOp>, "Only Read, ReadBack, IncompleteReadBack and Ternary Types work with NumElems::x");
            return IOp::Operation::num_elems_x(thread, iOp);
        }
        template <typename IOp>
        FK_HOST_DEVICE_FUSE uint y(const Point thread, const IOp& iOp) {
            static_assert(isAnyReadType<IOp> || opIs<TernaryType, IOp>, "Only Read, ReadBack, IncompleteReadBack and Ternary Types work with NumElems::y");
            return IOp::Operation::num_elems_y(thread, iOp);
        }
        template <typename IOp>
        FK_HOST_DEVICE_FUSE Size size(const Point thread, const IOp& iOp) {
            static_assert(isAnyReadType<IOp> || opIs<TernaryType, IOp>, "Only Read, ReadBack, IncompleteReadBack and Ternary Types work with NumElems::size");
            return Size(x(thread, iOp), y(thread, iOp));
        }
        template <typename IOp>
        FK_HOST_DEVICE_FUSE uint z(const Point thread, const IOp& iOp) {
            static_assert(isAnyReadType<IOp> || opIs<TernaryType, IOp>, "Only Read, ReadBack, IncompleteReadBack and Ternary Types work with NumElems::z");
            return IOp::Operation::num_elems_z(thread, iOp);
        }
    };
} // namespace fk

#endif