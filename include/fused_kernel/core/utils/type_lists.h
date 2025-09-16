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
        template <size_t Idx, typename... Types_>
        struct At;
        template <typename Head>
        struct At<0, Head> {
            using type = Head;
        };
        template <typename Head, typename... Tail>
        struct At<0, Head, Tail...> {
            using type = Head;
        };
        template <size_t Idx, typename Head, typename... Tail>
        struct At<Idx, Head, Tail...> {
            static_assert(Idx < TypeList<Types...>::size, "Index out of range");
            using type = typename At<Idx - 1, Tail...>::type;
        };

        template<typename... TypeLists>
        struct TypeListCat;

        template<typename... Args1>
        struct TypeListCat<TypeList<Args1...>> {
            using type = TypeList<Args1...>;
        };

        template<typename... Args1, typename... Args2>
        struct TypeListCat<TypeList<Args1...>, TypeList<Args2...>> {
            using type = TypeList<Args1..., Args2...>;
        };

        template<typename... Args1, typename... Args2, typename... TypeLists>
        struct TypeListCat<TypeList<Args1...>, TypeList<Args2...>, TypeLists...> {
            using type = typename TypeListCat<TypeList<Args1..., Args2...>, TypeLists...>::type;
        };

        /**
         * @struct TypeIndex
         * @brief Struct to find at compile time, the index in which the type T is found
         * for the first time in the TypeList (searching left to right).
         */
        template <typename T>
        struct TypeIndex {
            static_assert(TypeList<Types...>::template contains<T>, "The type is not in the list");
            template <size_t CurrentIdx, typename U, typename... Remaining>
            struct TypeIndexHelper{
                static constexpr size_t value =
                    std::is_same_v<T, U> ? CurrentIdx : TypeIndexHelper<CurrentIdx + 1, Remaining...>::value;
            };

            template <size_t CurrentIdx, typename U>
            struct TypeIndexHelper<CurrentIdx, U> {
                static constexpr size_t value =
                    std::is_same_v<T, U> ? CurrentIdx : CurrentIdx + 1;
            };

            static constexpr size_t value = TypeIndexHelper<0, Types...>::value;
        };

        template <size_t Idx, typename T, typename... Types>
        struct InsertType;

        template <typename T>
        struct InsertType<0, T> {
            using type = TypeList<T>;
        };

        template <size_t Idx, typename T, typename Head>
        struct InsertType<Idx, T, Head> {
            using type = std::conditional_t<Idx == 0,
                TypeList<T, Head>,
                TypeList<Head, T>
            >;
        };

        template <size_t Idx, typename T, typename Head, typename... Tail>
        struct InsertType<Idx, T, Head, Tail...> {
            using type = std::conditional_t<Idx == 0,
                TypeList<T, Head, Tail...>,
                typename TypeList<Head>::template cat<typename InsertType<Idx - 1, T, Tail...>::type>
            >;
        };

        /*
        * @struct RemoveType
         * @brief Struct to remove at compile time, the type found at Idx
        */
        template <size_t Idx, typename... Types>
        struct RemoveType;

        template <typename Head, typename... Tail>
        struct RemoveType<0, Head, Tail...> {
            using type = TypeList<Tail...>; // Remove the first type
        };

        template <size_t Idx, typename Head, typename... Tail>
        struct RemoveType<Idx, Head, Tail...> {
            static_assert(Idx < TypeList<Head, Tail...>::size, "Index out of range");
            using type = typename TypeList<Head>::template cat<typename RemoveType<Idx - 1, Tail...>::type>;
        };

    public:
        static constexpr size_t size{sizeof...(Types)};

        template <size_t Idx>
        using at = typename At<Idx, Types...>::type;
        using first = at<0>;
        using last = at<sizeof...(Types)-1>;
        template <typename T>
        using addFront = TypeList<T, Types...>;
        template <typename T>
        using addBack = TypeList<Types..., T>;
        template <size_t Idx, typename T>
        using addAt = typename InsertType<Idx, T, Types...>::type;
        template <typename... TypeLists>
        using cat = typename TypeListCat<TypeList<Types...>, TypeLists...>::type;
        template <size_t Idx>
        using removeAt = typename RemoveType<Idx, Types...>::type;

        template <typename T>
        static constexpr bool contains = (std::is_same_v<T, Types> || ...);

        template <typename T>
        static constexpr bool allAre = (std::is_same_v<T, Types> && ...);

        static constexpr bool allAreSame = (std::is_same_v<first, Types> && ...);

        template <typename T>
        static constexpr size_t idx = TypeIndex<T>::value;
    };

    template <typename T>
    struct IsTypeList : std::false_type {};

    template <typename... Types>
    struct IsTypeList<TypeList<Types...>> : std::true_type {};

    template <typename T>
    constexpr bool isTypeList = IsTypeList<T>::value;

    template <typename TypeList1, typename... TypeLists>
    using TypeListCat_t = typename TypeList1::template cat<TypeLists...>;

    template <typename T, typename TypeList_t>
    struct one_of {
        static constexpr int value = TypeList_t::template contains<T>;
    };

    template <typename T, typename TypeList_t>
    constexpr bool one_of_v = one_of<T, TypeList_t>::value;

    template <typename T, typename TypeList_t>
    constexpr bool none_of_v = !one_of_v<T, TypeList_t>;

    template <typename T, typename TypeList_t>
    struct all_of {
        static constexpr bool value = TypeList_t::template allAre<T>;
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
     */
    template <typename T, typename TypeList_t>
    struct TypeIndex {
        static_assert(isTypeList<TypeList_t>, "The second parameter must be a TypeList");
        static_assert(TypeList_t::template contains<T>, "The type is not in the list");
        static constexpr size_t value = TypeList_t::template idx<T>;
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
    using FirstType_t = typename TypeList<Types...>::first;

    template <typename... Types>
    using LastType_t = typename TypeList<Types...>::last;

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
    private:
        static constexpr size_t index = TypeList1::template idx<T>;
    public:
        using type = typename TypeList2::template at<index>;
    };

    template <typename T, typename TypeList1, typename TypeList2>
    using EquivalentType_t = typename EquivalentType<T, TypeList1, TypeList2>::type;

    template<typename T, typename... Ts>
    constexpr bool all_types_are_same = std::conjunction_v<std::is_same<T, Ts>...>;

    template <size_t Index, typename T, typename TypeList>
    using InsertType_t = typename TypeList::template addAt<Index, T>;

    template <typename TypeList, typename T>
    using InsertTypeBack_t = typename TypeList::template addBack<T>;

    template <typename T, typename TypeList>
    using InsertTypeFront_t = typename TypeList::template addFront<T>;

    template <size_t Index, typename TypeList>
    using RemoveType_t = typename TypeList::template removeAt<Index>;

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
