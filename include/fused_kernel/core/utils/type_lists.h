/* Copyright 2023-2026 Oscar Amoros Huguet
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

    // Type identity implementation that avoids adding to namespace std
    // Uses std::type_identity when available, otherwise provides our own
#if defined(__cpp_lib_type_identity) && __cpp_lib_type_identity >= 201806L
    template <typename T>
    using type_identity = std::type_identity<T>;
#else
    template <typename T>
    struct type_identity {
        using type = T;
    };
#endif

    template <size_t I, typename T>
    struct TypeLeaf {
        using type = T;
    };

    namespace detail {
        // Declaration only.
        // We rely on the compiler matching the specific base class TypeLeaf<I, T>
        // to the explicitly provided index 'I'.
        template <size_t I, typename T>
        fk::type_identity<T> getter(const TypeLeaf<I, T> &);
    } // namespace detail

    template <typename IndxSeq, typename... Ts>
    struct TypeListImpl;

    template <size_t... Is, typename... Ts>
    struct TypeListImpl<std::index_sequence<Is...>, Ts...> 
        : TypeLeaf<Is, Ts>... 
    {
    };

    template <typename... Ts>
    struct TypeList : TypeListImpl<std::make_index_sequence<sizeof...(Ts)>, Ts...> {
        static constexpr size_t size = sizeof...(Ts);

        template <size_t Idx>
        using at = typename decltype(detail::getter<Idx>(std::declval<TypeList<Ts...>>()))::type;

        template <typename... Us>
        FK_HOST_DEVICE_CNST TypeList<Ts..., Us...> operator>>(const TypeList<Us...> &) const;
    };

    // The Accessor Alias
    template <size_t Idx, typename List>
    using TypeAt_t = typename decltype(detail::getter<Idx>(std::declval<List>()))::type;

    template <typename T>
    struct IsTypeList : std::false_type {};

    template <typename... Types>
    struct IsTypeList<TypeList<Types...>> : std::true_type {};

    template <typename T>
    constexpr bool isTypeList = IsTypeList<T>::value;

    // ------------------------------------------------------------------
    // TypeListCat Implementation
    // ------------------------------------------------------------------
    template <typename... Lists>
    struct TypeListCat {
        using type = decltype((std::declval<Lists>() >> ...));
    };
    
    template <typename... Lists>
    using TypeListCat_t = typename TypeListCat<Lists...>::type;

    // Primary template (default false)
    template <typename T, typename List>
    struct one_of {
        static constexpr bool value = false;
    };

    // Specialization for TypeList
    // This will match any fk::TypeList<Us...>
    template <typename T, typename... Us>
    struct one_of<T, fk::TypeList<Us...>> {
        static constexpr bool value = (std::is_same_v<T, Us> || ...);
    };

    template <typename T, typename TypeList_t>
    constexpr bool one_of_v = one_of<T, TypeList_t>::value;

    template <typename... Args>
    struct all_of {};

    template <typename T, typename... U>
    struct all_of<T, TypeList<U...>> {
        static constexpr bool value = (std::is_same_v<T, U> && ...);
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
