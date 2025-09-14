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
#include <fused_kernel/core/utils/utils.h>


namespace fk {

    template <typename T = NullType>
    struct RDT;
    template <typename T = NullType>
    struct IT;
    template <typename T = NullType>
    struct PT;
    template <typename T = NullType>
    struct OT;
    template <typename T = NullType>
    struct BIOpT;
    template <typename T = NullType>
    struct WDT;

    template <typename Alias, typename... Aliases>
    struct OpAlias;

#define OP_ALIAS_MATCH(ALIAS) \
    template <typename T, typename... Aliases> \
    struct OpAlias<ALIAS<NullType>, ALIAS<T>, Aliases...> { \
        using type = T; \
    }; \
    template <typename T> \
    struct OpAlias<ALIAS<NullType>, ALIAS<T>> { \
        using type = T; \
    };

    OP_ALIAS_MATCH(RDT)
    OP_ALIAS_MATCH(IT)
    OP_ALIAS_MATCH(PT)
    OP_ALIAS_MATCH(OT)
    OP_ALIAS_MATCH(BIOpT)
    OP_ALIAS_MATCH(WDT)

#undef OP_ALIAS_MATCH

#define OP_ALIAS_NO_MATCH(ALIAS_SEARCH, ALIAS_FOUND) \
    template <typename T, typename... Aliases> \
    struct OpAlias<ALIAS_SEARCH<NullType>, ALIAS_FOUND<T>, Aliases...> { \
        using type = typename OpAlias<ALIAS_SEARCH<NullType>, Aliases...>::type; \
    }; \
    template <typename T> \
    struct OpAlias<ALIAS_SEARCH<NullType>, ALIAS_FOUND<T>> { \
        using type = NullType; \
    };

    OP_ALIAS_NO_MATCH(RDT, IT)
    OP_ALIAS_NO_MATCH(RDT, PT)
    OP_ALIAS_NO_MATCH(RDT, OT)
    OP_ALIAS_NO_MATCH(RDT, BIOpT)
    OP_ALIAS_NO_MATCH(RDT, WDT)
    OP_ALIAS_NO_MATCH(IT, RDT)
    OP_ALIAS_NO_MATCH(IT, PT)
    OP_ALIAS_NO_MATCH(IT, OT)
    OP_ALIAS_NO_MATCH(IT, BIOpT)
    OP_ALIAS_NO_MATCH(IT, WDT)
    OP_ALIAS_NO_MATCH(PT, RDT)
    OP_ALIAS_NO_MATCH(PT, IT)
    OP_ALIAS_NO_MATCH(PT, OT)
    OP_ALIAS_NO_MATCH(PT, BIOpT)
    OP_ALIAS_NO_MATCH(PT, WDT)
    OP_ALIAS_NO_MATCH(OT, RDT)
    OP_ALIAS_NO_MATCH(OT, IT)
    OP_ALIAS_NO_MATCH(OT, PT)
    OP_ALIAS_NO_MATCH(OT, BIOpT)
    OP_ALIAS_NO_MATCH(OT, WDT)
    OP_ALIAS_NO_MATCH(BIOpT, RDT)
    OP_ALIAS_NO_MATCH(BIOpT, IT)
    OP_ALIAS_NO_MATCH(BIOpT, PT)
    OP_ALIAS_NO_MATCH(BIOpT, OT)
    OP_ALIAS_NO_MATCH(BIOpT, WDT)
    OP_ALIAS_NO_MATCH(WDT, RDT)
    OP_ALIAS_NO_MATCH(WDT, IT)
    OP_ALIAS_NO_MATCH(WDT, PT)
    OP_ALIAS_NO_MATCH(WDT, OT)
    OP_ALIAS_NO_MATCH(WDT, BIOpT)

#undef OP_ALIAS_NO_MATCH

    template <typename Alias, typename... Aliases>
    using OpAlias_t = typename OpAlias<Alias, Aliases...>::type;

    // Operation types specifier, helpers to define the Operation required aliases.
    template <typename... Types>
    struct ReadOp {
        static_assert(sizeof...(Types) <= 3, "ReadOp can only have up to 3 type aliases: RDT, PT and OT");
        using ReadDataType = OpAlias_t<RDT<>, Types...>;
        using ParamsType = OpAlias_t<PT<>, Types...>;
        using OutputType = OpAlias_t<OT<>, Types...>;
        constexpr ReadOp() {};
    };
    template <>
    struct ReadOp<> {
        using ReadDataType = NullType;
        using ParamsType = NullType;
        using OutputType = NullType;

        constexpr ReadOp() {};
    };

    template <typename... Types>
    struct ReadBackOp {
        static_assert(sizeof...(Types) <= 4, "ReadBackOp can only have up to 4 type aliases: RDT, PT, OT and BIOpT");
        using ReadDataType = OpAlias_t<RDT<>, Types...>;
        using ParamsType = OpAlias_t<PT<>, Types...>;
        using OutputType = OpAlias_t<OT<>, Types...>;
        using BackIOp = OpAlias_t<BIOpT<>, Types...>;
        constexpr ReadBackOp() {};
    };
    template <>
    struct ReadBackOp<> {
        using ReadDataType = NullType;
        using ParamsType = NullType;
        using OutputType = NullType;
        using BackIOp = NullType;
        constexpr ReadBackOp() {};
    };

    template <typename... Types>
    struct UnaryOp {
        static_assert(sizeof...(Types) <= 2, "UnaryOp can only have up to 2 type aliases: IT and OT");
        using InputType = OpAlias_t<IT<>, Types...>;
        using OutputType = OpAlias_t<OT<>, Types...>;
        constexpr UnaryOp() {};
    };
    template <>
    struct UnaryOp<> {
        using InputType = NullType;
        using OutputType = NullType;
        constexpr UnaryOp() {};
    };

    template <typename... Types>
    struct BinaryOp {
        static_assert(sizeof...(Types) <= 3, "BinaryOp can only have up to 3 type aliases: IT, PT and OT");
        using InputType = OpAlias_t<IT<>, Types...>;
        using ParamsType = OpAlias_t<PT<>, Types...>;
        using OutputType = OpAlias_t<OT<>, Types...>;
        constexpr BinaryOp() {};
    };
    template <>
    struct BinaryOp<> {
        using InputType = NullType;
        using ParamsType = NullType;
        using OutputType = NullType;
        constexpr BinaryOp() {};
    };

    template <typename... Types>
    struct TernaryOp {
        static_assert(sizeof...(Types) <= 4, "TernaryOp can only have up to 4 type aliases: IT, PT, BIOpT and OT");
        using InputType = OpAlias_t<IT<>, Types...>;
        using ParamsType = OpAlias_t<PT<>, Types...>;
        using BackIOp = OpAlias_t<BIOpT<>, Types...>;
        using OutputType = OpAlias_t<OT<>, Types...>;
        constexpr TernaryOp() {};
    };
    template <>
    struct TernaryOp<> {
        using InputType = NullType;
        using ParamsType = NullType;
        using BackIOp = NullType;
        using OutputType = NullType;
        constexpr TernaryOp() {};
    };

    template <typename... Types>
    struct MidWriteOp {
        static_assert(sizeof...(Types) <= 4, "MidWriteOp can only have up to 4 type aliases: IT, PT, WDT and OT");
        using InputType = OpAlias_t<IT<>, Types... >;
        using ParamsType = OpAlias_t<PT<>, Types...>;
        using WriteDataType = OpAlias_t<WDT<>, Types...>;
        using OutputType = OpAlias_t<OT<>, Types...>;
        
        constexpr MidWriteOp() {};
    private:
        static constexpr bool sameITAndOTTypes =
            (!isNullType<InputType> && !isNullType<OutputType>) && std::is_same_v<InputType, OutputType>;
        static constexpr bool anyITOrOTIsNullType = isNullType<InputType> || isNullType<OutputType>;
        static_assert(anyITOrOTIsNullType || sameITAndOTTypes, "MidWriteOp can not have different IT and OT.");
    };
    template <>
    struct MidWriteOp<> {
        using InputType = NullType;
        using ParamsType = NullType;
        using WriteDataType = NullType;
        using OutputType = NullType;
        constexpr MidWriteOp() {};
    };

    template <typename... Types>
    struct WriteOp {
        static_assert(sizeof...(Types) <= 3, "WriteOp can only have up to 3 type aliases: IT, PT and WDT");
        using InputType = OpAlias_t<IT<>, Types...>;
        using ParamsType = OpAlias_t<PT<>, Types...>;
        using WriteDataType = OpAlias_t<WDT<>, Types...>;
        constexpr WriteOp() {};
    };
    template <>
    struct WriteOp<> {
        using InputType = NullType;
        using ParamsType = NullType;
        using WriteDataType = NullType;
        constexpr WriteOp() {};
    };

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
