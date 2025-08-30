/* Copyright 2023-2024 Oscar Amoros Huguet
   Copyright 2023 David Del Rio Astorga
   Copyright 2023 Grup Mediapro S.L.U. (Oscar Amoros Huguet)

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

#ifndef FK_TYPE_LIST
#define FK_TYPE_LIST

#include <utility>
#include <fused_kernel/core/utils/utils.h>

namespace fk { // namespace fused kernel
    /**
     * @struct TypeList
     * @brief Struct to hold a list of types, and be able to work with them at compile time.
     *
     * This the base defintion of the struct. Contains no implementation
     */
    template <typename... Types>
    struct TypeList {
    private:
        template <size_t n, typename... TypeList_t>
        struct At;
        template <typename Head>
        struct At<0, TypeList<Head>> {
            using type = Head;
        };
        template <typename Head, typename... Tail>
        struct At<0, TypeList<Head, Tail...>> {
            using type = Head;
        };
        template <size_t n, typename Head, typename... Tail>
        struct At<n, TypeList<Head, Tail...>> {
            static_assert(n < TypeList<Head, Tail...>::size, "Index out of range");
            using type = typename At<n - 1, TypeList<Tail...>>::type;
        };
    public:
        static constexpr size_t size{sizeof...(Types)};

        template <int Idx>
        using at = typename At<Idx, TypeList<Types...>>::type;
        using first = at<0>;
        using last = at<sizeof...(Types)-1>;
    };

    template <typename T>
    struct IsTypeList : std::false_type {};

    template <typename... Types>
    struct IsTypeList<TypeList<Types...>> : std::true_type {};

    template <typename T>
    constexpr bool isTypeList = IsTypeList<T>::value;

    /**
     * @struct TypeList<TypeList<Args1...>, TypeList<Args2...>, TypeList<Args3...>, TypeList<Args4...>>
     * @brief Struct to fuse 4 TypeList into a single one.
     *
     * The expansion results into a single struct that holds all the types.
     */
    template<typename... Args1, typename... Args2, typename... Args3, typename... Args4>
    struct TypeList<TypeList<Args1...>, TypeList<Args2...>, TypeList<Args3...>, TypeList<Args4...>> {
        using type = TypeList<Args1..., Args2..., Args3..., Args4...>;
    };

    template<typename... Args1, typename... Args2, typename... Args3,
             typename... Args4, typename... Args5, typename... Args6>
    struct TypeList<TypeList<Args1...>, TypeList<Args2...>, TypeList<Args3...>,
                    TypeList<Args4...>, TypeList<Args5...>, TypeList<Args6...>> {
        using type = TypeList<Args1..., Args2..., Args3..., Args4..., Args5..., Args6...>;
    };

    template<typename... Types>
    struct TypeListCat{};

    template<typename... Args1, typename... Args2>
    struct TypeListCat<TypeList<Args1...>, TypeList<Args2...>> {
        using type = TypeList<Args1..., Args2...>;
    };

    template <typename TypeList1, typename TypeList2>
    using TypeListCat_t = typename TypeListCat<TypeList1, TypeList2>::type;

    template <typename... Args>
    struct one_of {};

    template <typename T, typename... U>
    struct one_of<T, TypeList<U...>> {
        static constexpr int value = std::disjunction_v<std::is_same<T,U>...>;
    };

    template <typename T, typename TypeList_t>
    constexpr bool one_of_v = one_of<T, TypeList_t>::value;

    template <typename... Args>
    struct all_of {};

    template <typename T, typename... U>
    struct all_of<T, TypeList<U...>> {
        static constexpr bool value = std::conjunction_v<std::is_same<T, U>...>;
    };

    template <typename T, typename TypeList_t>
    constexpr bool all_of_v = all_of<T, TypeList_t>::value;

    /**
     * @struct EnumType
     * @brief Struct to convert an enum value into a type
     *
     * This the base defintion of the struct. Contains no implementation
     */
    template <typename Enum, Enum value>
    struct EnumType {};

    template <typename Enum, Enum value>
    using E_t = EnumType<Enum, value>;

    /**
     * @struct TypeIndex
     * @brief Struct to find at compile time, the index in which the type T is found
     * in the TypeList TypeList_t.
     *
     * This the base defintion of the struct. Contains no implementation
     */
    template <typename T, typename TypeList_t>
    struct TypeIndex;

    /**
     * @struct TypeIndex<T, TypeList<T, Types...>>
     * @brief TypeIndex especialization that implements the case when T is
     * the same type as the first type in TypeList.
     *
     * This the stop condition of the recursive algorithm.
     */
    template <typename T, typename... Types>
    struct TypeIndex<T, TypeList<T, Types...>> {
        static constexpr size_t value = 0;
    };

    /**
     * @struct TypeIndex<T, TypeList<U, Types...>>
     * @brief TypeIndex especialization that implements the case when T is
     * not the same type as the first type in TypeList.
     *
     * If T is not the same type as the first Type in TypeList, U, then we define value to be 1 + 
     * whatever is expanded by TypeIndex<T, TypeList<Types...>>::value which will evaluate the 
     * TypeList minus U type.
     */
    template <typename T, typename U, typename... Types>
    struct TypeIndex<T, TypeList<U, Types...>> {
        static_assert(one_of<T, TypeList<U, Types...>>::value == true, "The type is not on the type list");
        static constexpr size_t value = 1 + TypeIndex<T, TypeList<Types...>>::value;
    };

    /**
     * \var constexpr size_t TypeIndex_v
     * \brief Template variable that will hold the result of expanding TypeIndex<T, TypeList_t>::value
     */
    template <typename T, typename TypeList_t>
    constexpr size_t TypeIndex_v = TypeIndex<T, TypeList_t>::value;

    // Obtain the type found in the index Idx, in TypeList
    template <int n, typename TypeList_t>
    using TypeAt_t = std::conditional_t<(n>=0), typename TypeList_t::template at<static_cast<size_t>(n)>, typename TypeList_t::last>;

    template <typename... Types>
    using FirstType_t = TypeAt_t<0, TypeList<Types...>>;

    template <typename... Types>
    using LastType_t = TypeAt_t<sizeof...(Types)-1, TypeList<Types...>>;

    template <typename... Types>
    using FirstInstantiableOperationInputType_t = typename FirstType_t<Types...>::Operation::InputType;

    template <typename... Types>
    using LastInstantiableOperationOutputType_t = typename LastType_t<Types...>::Operation::OutputType;

    // Find the index of T in TypeList1 and obtain the tyoe for that index
    // in TypeList2. All this at compile time. This can be used when you want to automatically derive
    // a type from another type.
    template <typename T, typename TypeList1, typename TypeList2>
    struct EquivalentType {
        static_assert(one_of_v<T, TypeList1>, "The type is not in the first list");
        using type = TypeAt_t<TypeIndex_v<T, TypeList1>, TypeList2>;
    };

    template <typename T, typename TypeList1, typename TypeList2>
    using EquivalentType_t = typename EquivalentType<T, TypeList1, TypeList2>::type;

    template<typename T, typename... Ts>
    constexpr bool all_types_are_same = std::conjunction_v<std::is_same<T, Ts>...>;

    template <size_t Index, typename T, typename... Types>
    struct InsertType {};

    template <typename T>
    struct InsertType<0, T> {
        using type = TypeList<T>;
    };

    template <size_t Index, typename T, typename Head>
    struct InsertType<Index, T, TypeList<Head>> {
        using type = std::conditional_t<Index == 0,
            TypeList<T, Head>,
            TypeList<Head, T>
        >;
    };

    template <size_t Index, typename T, typename Head, typename... Tail>
    struct InsertType<Index, T, TypeList<Head, Tail...>> {
        using type = std::conditional_t<Index == 0,
                                        TypeList<T, Head, Tail...>,
                                        TypeListCat_t<TypeList<Head>, typename InsertType<Index - 1, T, TypeList<Tail...>>::type>
                                       >;
    };

    template <size_t Index, typename T, typename TypeList>
    using InsertType_t = typename InsertType<Index, T, TypeList>::type;

    template <typename TypeList, typename T>
    using InsertTypeBack_t = typename InsertType<TypeList::size, T, TypeList>::type;

    template <typename T, typename TypeList>
    using InsertTypeFront_t = typename InsertType<0, T, TypeList>::type;

    template <size_t Index, typename... Types>
    struct RemoveType;

    template <typename Head, typename... Tail>
    struct RemoveType<0, TypeList<Head, Tail...>> {
        using type = TypeList<Tail...>; // Remove the first type
    };

    template <size_t Index, typename Head, typename... Tail>
    struct RemoveType<Index, TypeList<Head, Tail...>> {
        static_assert(Index < TypeList<Head, Tail...>::size, "Index out of range");
        using type = TypeListCat_t<TypeList<Head>, typename RemoveType<Index - 1, TypeList<Tail...>>::type>;
    };

    template <size_t Index, typename TypeList>
    using RemoveType_t = typename RemoveType<Index, TypeList>::type;

    template <typename T, typename Restriction, typename TypeList, bool last, T currentInt, T... integers>
    struct RestrictedIntegerSequenceBuilder;

    template <typename T, typename Restriction, typename TypeList, T currentInt, T... integers>
    struct RestrictedIntegerSequenceBuilder<T, Restriction, TypeList, false, currentInt, integers...> {
        static_assert(TypeList::size > 0, "Can't generate an integer sequence for an empty TypeList");
        using type = std::conditional_t<(sizeof...(integers) > 0),
            std::conditional_t <TypeList::size == currentInt + 1,
                std::conditional_t<Restriction::template complies<TypeAt_t<currentInt, TypeList>>(),
                    std::integer_sequence<T, integers..., currentInt>,
                    std::integer_sequence<T, integers...>>,
                std::conditional_t<Restriction::template complies<TypeAt_t<currentInt, TypeList>>(),
                    typename RestrictedIntegerSequenceBuilder<T, Restriction, TypeList, currentInt + 1 == TypeList::size, currentInt + 1, integers..., currentInt>::type,
                    typename RestrictedIntegerSequenceBuilder<T, Restriction, TypeList, currentInt + 1 == TypeList::size, currentInt + 1, integers...>::type>>,
            std::conditional_t<TypeList::size == 1,
                std::conditional_t<Restriction::template complies<TypeAt_t<currentInt, TypeList>>(),
                    std::integer_sequence<T, currentInt>,
                    std::integer_sequence<T>>,
                std::conditional_t<Restriction::template complies<TypeAt_t<currentInt, TypeList>>(),
                    typename RestrictedIntegerSequenceBuilder<T, Restriction, TypeList, currentInt + 1 == TypeList::size, currentInt + 1, currentInt>::type,
                    typename RestrictedIntegerSequenceBuilder<T, Restriction, TypeList, currentInt + 1 == TypeList::size, currentInt + 1>::type>>>;
    };

    template <typename T, typename Restriction, typename TypeList, T currentInt, T... integers>
    struct RestrictedIntegerSequenceBuilder<T, Restriction, TypeList, true, currentInt, integers...> {
        using type = std::conditional_t<(sizeof...(integers) > 0), std::integer_sequence<T,integers...>, std::integer_sequence<T>>;
    };

    template <typename T, typename TypeRestriction, typename TypeList>
    using filtered_integer_sequence_t = typename RestrictedIntegerSequenceBuilder<T, TypeRestriction, TypeList, false, 0>::type;

    template <typename TypeRestriction, typename TypeList>
    using filtered_index_sequence_t = typename RestrictedIntegerSequenceBuilder<size_t, TypeRestriction, TypeList, false, 0>::type;

} // namespace fk

#endif
