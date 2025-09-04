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
    struct ReadType;
    struct ReadBackType;
    struct IncompleteReadBackType;
    struct UnaryType;
    struct BinaryType;
    struct TernaryType;
    struct MidWriteType;
    struct WriteType;

    template <typename T, typename = void>
    struct HasInstanceType : std::false_type {};
    template <typename T>
    struct HasInstanceType<T, std::void_t<typename T::InstanceType>> : std::true_type {};

    template <typename T, typename = void>
    struct IsReadType : std::false_type {};
    template <typename T>
    struct IsReadType<T, std::enable_if_t<!std::is_same_v<typename T::InstanceType, ReadType>, void>> : std::false_type {};
    template <typename T>
    struct IsReadType<T, std::enable_if_t<std::is_same_v<typename T::InstanceType, ReadType>, void>> : std::true_type {};

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
    struct IsWriteType : std::false_type {};
    template <typename T>
    struct IsWriteType<T, std::enable_if_t<!std::is_same_v<typename T::InstanceType, WriteType>, void>> : std::false_type {};
    template <typename T>
    struct IsWriteType<T, std::enable_if_t<std::is_same_v<typename T::InstanceType, WriteType>, void>> : std::true_type {};

    template <typename T>
    constexpr bool isOperation = HasInstanceType<T>::value;

    template <typename OpORIOp>
    constexpr bool isReadType = IsReadType<OpORIOp>::value;

    template <typename OpORIOp>
    constexpr bool isReadBackType = IsReadBackType<OpORIOp>::value;

    template <typename OpORIOp>
    constexpr bool isIncompleteReadBackType = IsIncompleteReadBackType<OpORIOp>::value;

    template <typename OpORIOp>
    constexpr bool isAnyReadType = isReadType<OpORIOp> || isReadBackType<OpORIOp> || isIncompleteReadBackType<OpORIOp>;

    template <typename OpORIOp>
    constexpr bool isAnyCompleteReadType = isReadType<OpORIOp> || isReadBackType<OpORIOp>;

    template <typename OpORIOp>
    constexpr bool isUnaryType = IsUnaryType<OpORIOp>::value;

    template <typename OpORIOp>
    constexpr bool isBinaryType = IsBinaryType<OpORIOp>::value;

    template <typename OpORIOp>
    constexpr bool isTernaryType = IsTernaryType<OpORIOp>::value;

    template <typename OpORIOp>
    constexpr bool isWriteType = IsWriteType<OpORIOp>::value;

    template <typename OpORIOp>
    constexpr bool isMidWriteType = IsMidWriteType<OpORIOp>::value;

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
