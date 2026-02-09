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

#ifndef FK_OPERATION_TYPES
#define FK_OPERATION_TYPES

#include <fused_kernel/core/data/point.h>
#include <fused_kernel/core/utils/template_operations.h>
#include <fused_kernel/core/utils/type_lists.h>

namespace fk {
    /* Operation types are linked to the exec() function definition in the Operation. 
    *  The elements that can change accross Operation types are:
    *   - OutputType: wether the exec function returns a value or not, and which type it is. The value resides on registers.
    *   - ElementIdx (using the type Point): wether the exec function gets the thread idx as input or not.
    *     It is used to compute DRAM or Shared Memory addresses to read from or write into.
    *   - InputType: wether the exec function gets an input value or not. This value resides on registers.
    *   - ParamsType: wether the exec function gets an any additional data that is not computed inside the kernel
    *     and that is needed for the execution of the operation.
    *   - BackIOp: wether the exec function gets an additional IOp as input, that is executed as part of the operation
    *     implementation.
    * 
    *    An example of the exec function with all the types would be:
    *    OutputType exec(Point, InputType, ParamsType, BackIOp)

         +------------------------+----------+----------+----------+----------+----------+
         |                        |   Out    |   EIdx   |    In    |   Par    |   BIOp   |
         +------------------------+----------+----------+----------+----------+----------+
         | ReadType               |    X     |    X     |          |    X     |          |  OutputType exec(Point, ParamsType)
         +------------------------+----------+----------+----------+----------+----------+
         | WriteType              |          |    X     |    X     |    X     |          |  void exec(Point, InputType, ParamsType)
         +------------------------+----------+----------+----------+----------+----------+
         | UnaryType              |    X     |          |    X     |          |          |  OutputType exec(InputType)
         +------------------------+----------+----------+----------+----------+----------+
         | BinaryType             |    X     |          |    X     |    X     |          |  OutputType exec(InputType, ParamsType)
         +------------------------+----------+----------+----------+----------+----------+
         | ReadBackType           |    X     |    X     |          |    X     |    X     |  OutputType exec(Point, ParamsType, BackIOp)
         +------------------------+----------+----------+----------+----------+----------+
         | IncompleteReadBackType |          |          |          |          |          |  no exec function present
         +------------------------+----------+----------+----------+----------+----------+
         | TernaryType            |    X     |          |    X     |    X     |    X     |  OutputType exec(InputType, ParamsType, BackIOp)
         +------------------------+----------+----------+----------+----------+----------+
         | IncompleteTernaryType  |          |          |          |          |          |  no exec function present
         +------------------------+----------+----------+----------+----------+----------+
         | MidWriteType *         |    X     |    X     |    X     |    X     |          |  InputType exec(Point, InputType, ParamsType)
         +------------------------+----------+----------+----------+----------+----------+
         | OpenType **            |    X     |    X     |    X     |    X     |          |  OutputType exec(Point, InputType, ParamsType)
         +------------------------+----------+----------+----------+----------+----------+
         | ClosedType **          |          |    X     |          |    X     |          |  void exec(Point, ParamsType)
         +------------------------+----------+----------+----------+----------+----------+

         * Aplicable only to Instantiapble Operations. In and Out must be the same type and value. Operation must be of WriteType.
         ** OpenType and ClosedType are only applicable to FusedOperations. FusedOperations can also be ReadType or WriteType.
    */

    struct ReadType;
    struct WriteType;
    struct UnaryType;
    struct BinaryType;
    struct ReadBackType;
    struct IncompleteReadBackType;
    struct TernaryType;
    struct IncompleteTernaryType;
    struct MidWriteType;
    struct OpenType;
    struct ClosedType;

    template <typename T, typename = void>
    struct HasInstanceType : std::false_type {};
    template <typename T>
    struct HasInstanceType<T, std::void_t<typename T::InstanceType>> : std::true_type {};

    template <typename T, typename = void>
    struct IsReadBackType : std::false_type {};
    template <typename T>
    struct IsReadBackType<T, std::enable_if_t<!std::is_same_v<typename T::InstanceType, ReadBackType>, void>> : std::false_type {};
    template <typename T>
    struct IsReadBackType<T, std::enable_if_t<std::is_same_v<typename T::InstanceType, ReadBackType>, void>> : std::true_type {};

    template <typename T, typename = void>
    struct IsIncompleteReadBackType : std::false_type {};
    template <typename T>
    struct IsIncompleteReadBackType<T, std::enable_if_t<!std::is_same_v<typename T::InstanceType, IncompleteReadBackType>, void>> : std::false_type {};
    template <typename T>
    struct IsIncompleteReadBackType<T, std::enable_if_t<std::is_same_v<typename T::InstanceType, IncompleteReadBackType>, void>> : std::true_type {};

    template <typename T, typename = void>
    struct IsUnaryType : std::false_type {};
    template <typename T>
    struct IsUnaryType<T, std::enable_if_t<!std::is_same_v<typename T::InstanceType, UnaryType>, void>> : std::false_type {};
    template <typename T>
    struct IsUnaryType<T, std::enable_if_t<std::is_same_v<typename T::InstanceType, UnaryType>, void>> : std::true_type {};

    template <typename T, typename = void>
    struct IsBinaryType : std::false_type {};
    template <typename T>
    struct IsBinaryType<T, std::enable_if_t<!std::is_same_v<typename T::InstanceType, BinaryType>, void>> : std::false_type {};
    template <typename T>
    struct IsBinaryType<T, std::enable_if_t<std::is_same_v<typename T::InstanceType, BinaryType>, void>> : std::true_type {};

    template <typename T, typename = void>
    struct IsTernaryType : std::false_type {};
    template <typename T>
    struct IsTernaryType<T, std::enable_if_t<!std::is_same_v<typename T::InstanceType, TernaryType>, void>> : std::false_type {};
    template <typename T>
    struct IsTernaryType<T, std::enable_if_t<std::is_same_v<typename T::InstanceType, TernaryType>, void>> : std::true_type {};

    template <typename T, typename = void>
    struct IsMidWriteType : std::false_type {};
    template <typename T>
    struct IsMidWriteType<T, std::enable_if_t<!std::is_same_v<typename T::InstanceType, MidWriteType>, void>> : std::false_type {};
    template <typename T>
    struct IsMidWriteType<T, std::enable_if_t<std::is_same_v<typename T::InstanceType, MidWriteType>, void>> : std::true_type {};

    template <typename T, typename = void>
    struct IsOpenType : std::false_type {};
    template <typename T>
    struct IsOpenType<T, std::enable_if_t<!std::is_same_v<typename T::InstanceType, OpenType>, void>> : std::false_type {};
    template <typename T>
    struct IsOpenType<T, std::enable_if_t<std::is_same_v<typename T::InstanceType, OpenType>, void>> : std::true_type {};

    template <typename T, typename = void>
    struct IsWriteType : std::false_type {};
    template <typename T>
    struct IsWriteType<T, std::enable_if_t<!std::is_same_v<typename T::InstanceType, WriteType>, void>> : std::false_type {};
    template <typename T>
    struct IsWriteType<T, std::enable_if_t<std::is_same_v<typename T::InstanceType, WriteType>, void>> : std::true_type {};

    template <typename OperationType, typename OpOrIOp, typename = void>
    struct OpIs : std::false_type {};
    template <typename OperationType, typename OpOrIOp>
    struct OpIs<OperationType, OpOrIOp,
              std::enable_if_t<std::is_same_v<typename OpOrIOp::InstanceType, OperationType>, void>> : std::true_type {};

    template <typename OperationType, typename OpOrIOp>
    constexpr bool opIs = OpIs<OperationType, OpOrIOp>::value;

    template <typename T>
    constexpr bool isOperation = HasInstanceType<T>::value;

    template <typename OpORIOp>
    constexpr bool isReadType = opIs<ReadType, OpORIOp>;

    template <typename OpORIOp>
    constexpr bool isReadBackType = opIs<ReadBackType, OpORIOp>;

    template <typename OpORIOp>
    constexpr bool isIncompleteReadBackType = opIs<IncompleteReadBackType, OpORIOp>;

    template <typename OpORIOp>
    constexpr bool isAnyReadType = isReadType<OpORIOp> || isReadBackType<OpORIOp> || isIncompleteReadBackType<OpORIOp>;

    template <typename OpORIOp>
    constexpr bool isAnyCompleteReadType = isReadType<OpORIOp> || isReadBackType<OpORIOp>;

    template <typename OpORIOp>
    constexpr bool isUnaryType = opIs<UnaryType, OpORIOp>;

    template <typename OpORIOp>
    constexpr bool isBinaryType = opIs<BinaryType, OpORIOp>;

    template <typename OpORIOp>
    constexpr bool isTernaryType = opIs<TernaryType, OpORIOp>;

    template <typename OpORIOp>
    constexpr bool isOpenType = opIs<OpenType, OpORIOp>;

    template <typename OpORIOp>
    constexpr bool isWriteType = opIs<WriteType, OpORIOp>;

    template <typename OpORIOp>
    constexpr bool isMidWriteType = opIs<MidWriteType, OpORIOp>;

    template <typename OpORIOp>
    constexpr bool isComputeType = isUnaryType<OpORIOp> || isBinaryType<OpORIOp> || isTernaryType<OpORIOp>;

    using WriteTypeList = TypeList<WriteType, MidWriteType>;

    template <typename OpORIOp>
    constexpr bool isAnyWriteType = isWriteType<OpORIOp> || isMidWriteType<OpORIOp>;

    template <typename IOp>
    using GetInputType_t = typename IOp::Operation::InputType;

    template <typename IOp>
    using GetOutputType_t = typename IOp::Operation::OutputType;

    template <typename IOp>
    FK_HOST_DEVICE_CNST GetOutputType_t<IOp> compute(const GetInputType_t<IOp>& input,
                                                     const IOp& instantiableOperation) {
        static_assert(isComputeType<IOp>,
            "Function compute only works with IOp InstanceTypes UnaryType, BinaryType and TernaryType");
        if constexpr (isUnaryType<IOp>) {
            return IOp::Operation::exec(input);
        } else {
            return IOp::Operation::exec(input, instantiableOperation);
        }
    }

    template <typename... OpsOrIOps>
    constexpr bool allUnaryTypes = and_v<isUnaryType<OpsOrIOps>...>;

    template <typename... OpsOrIOps>
    constexpr bool allComputeTypes = and_v<isComputeType<OpsOrIOps>...>;

    template <typename... OpsOrIOps>
    constexpr bool atLeastOneMidWriteType = or_v<isMidWriteType<OpsOrIOps>...>;

    template <typename = void, typename... OpsOrIOps>
    struct NotAllUnary final : public std::false_type {};

    // This intermediate step is needed to avoid VS2017 crashing with an unespecified error
    template <typename... OpsOrIOps>
    constexpr bool notAllUnaryTypesNoSFINAE = ((!std::is_same_v<typename OpsOrIOps::InstanceType, UnaryType>) || ...);

    template <typename... OpsOrIOps>
    struct NotAllUnary<std::enable_if_t<notAllUnaryTypesNoSFINAE<OpsOrIOps...>, void>, OpsOrIOps...> final : public std::true_type {};

    template <typename... OpsOrIOps>
    constexpr bool notAllUnaryTypes = NotAllUnary<void, OpsOrIOps...>::value;

    template <typename Enabler, typename... OpsOrIOps>
    struct are_all_unary_types : std::false_type {};

    template <typename... OperationsOrInstantiableOperations>
    struct are_all_unary_types<std::enable_if_t<allUnaryTypes<OperationsOrInstantiableOperations...>>,
                               OperationsOrInstantiableOperations...> : std::true_type {};

    template <typename... OperationORInstantiableOperation>
    constexpr bool noneWriteType = and_v<(!isWriteType<OperationORInstantiableOperation>)...>;

    template <typename... OperationORInstantiableOperation>
    constexpr bool noneMidWriteType = and_v<(!isMidWriteType<OperationORInstantiableOperation>)...>;

    template <typename... OperationORInstantiableOperation>
    constexpr bool noneAnyWriteType = and_v<(!isAnyWriteType<OperationORInstantiableOperation>)...>;

    template <typename... OperationORInstantiableOperation>
    constexpr bool noneReadType = and_v<(!isReadType<OperationORInstantiableOperation>)...>;

    template <typename... OperationORInstantiableOperation>
    constexpr bool noneReadBackType = and_v<(!isReadBackType<OperationORInstantiableOperation>)...>;

    template <typename... OperationORInstantiableOperation>
    constexpr bool noneIncompleteReadBackType = and_v<(!isIncompleteReadBackType<OperationORInstantiableOperation>)...>;

    template <typename... OperationORInstantiableOperation>
    constexpr bool noneAnyReadType = and_v<(!isAnyReadType<OperationORInstantiableOperation>)...>;

    template <typename T, typename=void>
    struct IsCompleteOperation : std::false_type {};

    template <typename T>
    struct IsCompleteOperation<T, std::enable_if_t<isOperation<T> && !isIncompleteReadBackType<T>>> : std::true_type {};

    template <typename T>
    constexpr bool isCompleteOperation = IsCompleteOperation<T>::value;

    template <typename Enabler, typename T>
    struct is_fused_operation_ : std::false_type {};

    template <template <typename...> class FusedOperation, typename... Operations>
    struct is_fused_operation_<std::enable_if_t<FusedOperation<Operations...>::IS_FUSED_OP, void>, FusedOperation<Operations...>> : std::true_type{};

    template <typename Operation>
    using is_fused_operation = is_fused_operation_<void, Operation>;
} // namespace fk

#endif
