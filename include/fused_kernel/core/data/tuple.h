/* Copyright 2023-2026 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_TUPLE
#define FK_TUPLE

#include <fused_kernel/core/utils/type_lists.h>
#include <fused_kernel/core/utils/vector_utils.h>
#include <array>
#include <type_traits>
#include <utility>

namespace fk {
    // 1. The Leaf Node
    // Holds exactly one value.
    template <size_t I, typename T>
    struct TupleLeaf {
        T value;

        FK_HOST_DEVICE_CNST TupleLeaf() {};

        // FIX: Use a forwarding constructor that is constrained 
        // to NOT match the TupleLeaf itself (prevent copy-ctor ambiguity)
        template <typename U,
            typename = std::enable_if_t<!std::is_same_v<std::decay_t<U>, TupleLeaf>>>
        FK_HOST_DEVICE_CNST TupleLeaf(U&& v) : value(std::forward<U>(v)) {}
    };

    // 2. The Implementation Wrapper
    template <typename IndxSeq, typename... Ts>
    struct TupleImpl;

    template <size_t... Is, typename... Ts>
    struct TupleImpl<std::index_sequence<Is...>, Ts...> : TupleLeaf<Is, Ts>...
    {
        FK_HOST_DEVICE_CNST TupleImpl() {};

        // FIX: Explicitly forward arguments to the base classes.
        // The brace initialization {std::forward<Us>(args)...} ensures 
        // that the pack expansion happens in order.
        template <typename... Us>
        FK_HOST_DEVICE_CNST TupleImpl(Us&&... args)
            : TupleLeaf<Is, Ts>(std::forward<Us>(args))...
        {}
    };

    // 3. The User-Facing Tuple
    template <typename... Ts>
    struct Tuple : TupleImpl<std::make_index_sequence<sizeof...(Ts)>, Ts...> {
        using Base = TupleImpl<std::make_index_sequence<sizeof...(Ts)>, Ts...>;

        static constexpr size_t size = sizeof...(Ts);

        FK_HOST_DEVICE_CNST Tuple() {};

        // FIX: Use Universal References (U&&...) and std::forward
        // This handles:
        // 1. Tuple<int>(10)         -> moves 10
        // 2. Tuple<int&>(x)         -> references x
        // 3. Tuple<int&&>(std::move(x)) -> moves x
        template <typename... Us,
            typename = std::enable_if_t<sizeof...(Us) == sizeof...(Ts)>>
            FK_HOST_DEVICE_CNST Tuple(Us&&... args)
            : Base(std::forward<Us>(args)...) {}
    };

    // Deduction hint
    template <typename... Types> Tuple(Types...) -> Tuple<Types...>;

    // Primary template: defaults to false
    template <typename T>
    struct isTuple : std::false_type {};

    // Partial specialization: matches any specialization of Tuple
    template <typename... Types>
    struct isTuple<Tuple<Types...>> : std::true_type {};

    template <typename TypeToTest>
    constexpr bool isTuple_v = isTuple<std::decay_t<TypeToTest>>::value;

    // Assuming FK_HOST_DEVICE_FUSE expands to:
    // __host__ __device__ __forceinline__ static constexpr

    struct TupleUtil {
        // ==========================================
        // 1. Get Helpers (unchanged, strictly typed)
        // ==========================================

        // The compiler deduces 'T' by upcasting 'leaf' to the specific base class TupleLeaf<I, T>
        template <size_t I, typename T>
        FK_HOST_DEVICE_FUSE T& get_leaf_value(TupleLeaf<I, T> &leaf) {
            return leaf.value;
        }

        template <size_t I, typename T>
        FK_HOST_DEVICE_FUSE const T& get_leaf_value(const TupleLeaf<I, T> &leaf) {
            return leaf.value;
        }

        template <size_t I, typename T>
        FK_HOST_DEVICE_FUSE T&& get_leaf_value(TupleLeaf<I, T>&& leaf) {
            return static_cast<T&&>(leaf.value);
        }

        template <size_t I, typename T>
        FK_HOST_DEVICE_FUSE const T&& get_leaf_value(const TupleLeaf<I, T>&& leaf) {
            return static_cast<T&&>(leaf.value);;
        }

        // Accessor for the main Tuple
        template <size_t I, typename... Ts>
        FK_HOST_DEVICE_FUSE auto& get(Tuple<Ts...>& t) {
            return get_leaf_value<I>(t);
        }

        template <size_t I, typename... Ts>
        FK_HOST_DEVICE_FUSE const auto& get(const Tuple<Ts...>& t) {
            return get_leaf_value<I>(t);
        }

        template <size_t I, typename... Ts>
        FK_HOST_DEVICE_FUSE auto&& get(Tuple<Ts...>&& t) {
            return get_leaf_value<I>(std::forward<Tuple<Ts...>>(t));
        }

        template <size_t I, typename... Ts>
        FK_HOST_DEVICE_FUSE const auto&& get(const Tuple<Ts...>&& t) {
            return get_leaf_value<I>(std::move(t));
        }

        // ==========================================
        // 2. Concatenation (Flat Expansion)
        // ==========================================
        // Helper to extract and concatenate generic parameter packs
        template <typename T1, typename T2>
        struct TupleCatTraits;

        // Specialization for two Tuples
        template <typename... Ts, typename... Us>
        struct TupleCatTraits<Tuple<Ts...>, Tuple<Us...>> {
            using type = Tuple<Ts..., Us...>;
        };

        // Convenience alias
        template <typename T1, typename T2>
        using TupleCatResult_t = typename TupleCatTraits<std::decay_t<T1>, std::decay_t<T2>>::type;

        template <typename Tuple1, typename Tuple2, size_t... I1, size_t... I2>
        FK_HOST_DEVICE_FUSE auto cat_impl(Tuple1&& t1, Tuple2&& t2,
            std::index_sequence<I1...>,
            std::index_sequence<I2...>) {
            // 1. Calculate the exact Result Type (e.g., Tuple<int, float&, double>)
            using ResultTuple = TupleCatResult_t<Tuple1, Tuple2>;

            // 2. Construct explicitly. 
            //    We use std::forward to invoke the correct 'get' (move vs copy).
            //    The ResultTuple constructor will accept the results of get.
            return ResultTuple(
                get<I1>(std::forward<Tuple1>(t1))...,
                get<I2>(std::forward<Tuple2>(t2))...
            );
        }

        template <typename Tuple1, typename Tuple2>
        FK_HOST_DEVICE_FUSE auto cat(Tuple1&& t1, Tuple2&& t2) {
            using T1 = std::decay_t<Tuple1>;
            using T2 = std::decay_t<Tuple2>;

            return cat_impl(
                std::forward<Tuple1>(t1),
                std::forward<Tuple2>(t2),
                std::make_index_sequence<T1::size>(),
                std::make_index_sequence<T2::size>()
            );
        }

        // ==========================================
        // 3. Make, forward and tie
        // ==========================================

        template <typename... Types>
        FK_HOST_DEVICE_FUSE auto make_tuple(Types&&... args) {
            // Forwarding references allow move semantics if 'instances' are temporary
            // Store all velues in a new Tuple, effectively making a copy
            return Tuple<std::decay_t<Types>...>(std::forward<Types>(args)...);
        }

        template <typename... Types>
        FK_HOST_DEVICE_FUSE auto forward_as_tuple(Types&&... args) {
            // Constructs Tuple<T&&...> preserving exact value category (L-value vs R-value)
            // This is essentially a "view" of the arguments.
            return Tuple<Types&&...>(std::forward<Types>(args)...);
        }

        template <typename... Types>
        FK_HOST_DEVICE_FUSE auto tie(Types&... args) {
            // Constructs Tuple<T&...>
            // Only accepts L-values (variables), refuses temporaries.
            return Tuple<Types&...>(args...);
        }

        // ==========================================
        // 4. Insert (Fully Flattened)
        // ==========================================

        template <size_t InsertIdx, typename T, typename TupleT, size_t... PreIdx, size_t... PostIdx>
        FK_HOST_DEVICE_FUSE auto insert_impl(T&& val, TupleT&& t,
            std::index_sequence<PreIdx...>,
            std::index_sequence<PostIdx...>) {
            // FIX: Remove explicit template arguments <decltype(...)>.
            // rely on the Tuple(args...) deduction guide to decay types (int& -> int).
            // This creates safe storage.
            return Tuple(
                // 1. Elements before
                get<PreIdx>(std::forward<TupleT>(t))...,

                // 2. The new element
                std::forward<T>(val),

                // 3. Elements after (shifted by InsertIdx)
                get<PostIdx + InsertIdx>(std::forward<TupleT>(t))...
            );
        }

        template <size_t INDEX, typename T, typename TupleT>
        FK_HOST_DEVICE_FUSE auto tuple_insert(T&& instance, TupleT&& tuple) {
            using BareTuple = std::decay_t<TupleT>;
            constexpr size_t N = BareTuple::size;

            static_assert(INDEX <= N, "Index out of range.");

            // Split indices: [0, INDEX) and [INDEX, N)
            // Note: The second sequence length is (N - INDEX)
            return insert_impl<INDEX>(
                std::forward<T>(instance),
                std::forward<TupleT>(tuple),
                std::make_index_sequence<INDEX>{},
                std::make_index_sequence<N - INDEX>{}
            );
        }
    };

    template <size_t Idx, typename TupleLike>
    FK_HOST_DEVICE_CNST decltype(auto) get(TupleLike&& tuple) {
        static_assert(isTuple_v<TupleLike>, "fk::get can only be used with fk::Tuple");

        // decltype(auto) + std::forward preserves EXACTLY what TupleUtil returned
        // (Value category, const-ness, and reference type)
        return TupleUtil::get<Idx>(std::forward<TupleLike>(tuple));
    }

    template <typename... Types>
    FK_HOST_DEVICE_CNST auto make_tuple(Types&&... args) {
        // Store all values in a new Tuple, effectively making a copy
        return TupleUtil::make_tuple(std::forward<Types>(args)...);
    }

    template <typename... Types>
    FK_HOST_DEVICE_CNST auto forward_as_tuple(Types&&... args) {
        // Constructs Tuple<T&&...> preserving exact value category (L-value vs R-value)
        // This is essentially a "view" of the arguments.
        return TupleUtil::forward_as_tuple(std::forward<Types>(args)...);
    }

    template <typename... Types>
    FK_HOST_DEVICE_CNST auto tie(Types&... args) {
        // Constructs Tuple<T&...>
        // Only accepts L-values (variables), refuses temporaries.
        return TupleUtil::tie(args...);
    }

    template <typename TupleType>
    struct TupleTypeToTypeList;

    template <typename... Types>
    struct TupleTypeToTypeList<Tuple<Types...>> {
        using type = TypeList<Types...>;
    };

    template <typename TupleType>
    using ToTypeList = typename TupleTypeToTypeList<TupleType>::type;

    template <int INDEX, typename TupleType>
    using get_t = TypeAt_t<INDEX, ToTypeList<TupleType>>;

    template <int INDEX, typename T, typename TupleLike>
    FK_HOST_DEVICE_CNST auto tuple_insert(T&& element, TupleLike&& tuple) {
        return TupleUtil::tuple_insert<INDEX, T>(std::forward<T>(element), std::forward<TupleLike>(tuple));
    }

    template <typename T, typename TupleLike>
    FK_HOST_DEVICE_CNST auto tuple_insert_back(TupleLike&& tuple, T&& element) {
        return TupleUtil::tuple_insert<std::decay_t<TupleLike>::size, T>(std::forward<T>(element), std::forward<TupleLike>(tuple));
    }

    template <typename Tuple1, typename Tuple2>
    FK_HOST_DEVICE_CNST auto tuple_cat(Tuple1&& t1, Tuple2&& t2) {
        return TupleUtil::cat(std::forward<Tuple1>(t1), std::forward<Tuple2>(t2));
    }

    template <int INDEX, typename TupleLike>
    struct GetType {};

    template <int INDEX, template <typename...> class TupleLike, typename... Instances>
    struct GetType<INDEX, TupleLike<Instances...>> {
        using type = TypeAt_t<INDEX, TypeList<Instances...>>;
    };

    template <int INDEX, typename TupleLike>
    using get_type_t = typename GetType<INDEX, TupleLike>::type;

    // ==========================================
    // The apply Implementation (Now with Return Values)
    // ==========================================

    namespace detail {
        template <typename F, typename TupleT, size_t... Is>
        // decltype(auto) lets the compiler figure out the return type
        FK_HOST_CNST decltype(auto) apply_impl(F &&f, TupleT&&t, std::index_sequence<Is...>) {

            // We simply return the result of the function call.
            return std::forward<F>(f)(get<Is>(std::forward<TupleT>(t))...);
        }
        template <typename F, typename TupleT, size_t... Is>
        // decltype(auto) lets the compiler figure out the return type
        FK_HOST_DEVICE_CNST decltype(auto) apply_d_impl(F &&f, TupleT&&t, std::index_sequence<Is...>) {

            // We simply return the result of the function call.
            return std::forward<F>(f)(get<Is>(std::forward<TupleT>(t))...);
        }
    } // namespace detail

    template <typename F, typename TupleType>
    FK_HOST_CNST decltype(auto) apply(F&& f, TupleType&& t) {
        return detail::apply_impl(std::forward<F>(f), std::forward<TupleType>(t),
            std::make_index_sequence<std::decay_t<TupleType>::size>{});
    }

    template <typename F, typename TupleType>
    FK_HOST_DEVICE_CNST decltype(auto) apply_d(F&& f, TupleType&& t) {
        return detail::apply_d_impl(std::forward<F>(f), std::forward<TupleType>(t),
            std::make_index_sequence<std::decay_t<TupleType>::size>{});
    }

    // Struct to hold a parameter pack, and be able to pass it arround
    template <typename... IOpTypes>
    struct InstantiableOperationSequence {
        Tuple<IOpTypes...> iOps;
    };

    template <typename... IOpTypes>
    using IOpSequence = InstantiableOperationSequence<IOpTypes...>;

    // Function that fills the OperationSequence struct, from a parameter pack
    template <typename... IOpTypes>
    FK_HOST_CNST auto buildOperationSequence(const IOpTypes&... instantiableOperationInstances) {
        return InstantiableOperationSequence<IOpTypes...> {{instantiableOperationInstances...}};
    }

    template <typename... IOpTypes>
    FK_HOST_CNST auto buildOperationSequence_tup(const Tuple<IOpTypes...>& instantiableOperationInstances) {
        return apply([](const auto&... args) {
            return buildOperationSequence(args...);
            }, instantiableOperationInstances);
    }

    // Util to insert an element before the last element of a tuple
    template <typename T, typename Tuple>
    FK_HOST_CNST auto insert_before_last_tup(const T& t, const Tuple& args) {
        return tuple_insert<Tuple::size - 1>(t, args);
    }

    template<typename T, typename... Args>
    FK_HOST_CNST auto insert_before_last(const T& t, const Args&... args) {
        return tuple_insert<sizeof...(Args) - 1>(t, Tuple<Args...>{args...});
    }

    template <typename FirstType, typename SecondType>
    struct GetFirst {
        using OutputType = FirstType;
        template <int Idx>
        FK_HOST_FUSE FirstType translate(const int& usedPlanes, const std::pair<FirstType, SecondType>& a_pair) {
            return a_pair.first;
        }
    };

    template <typename FirstType, typename SecondType>
    struct GetSecond {
        using OutputType = SecondType;
        template <int Idx>
        FK_HOST_FUSE SecondType translate(const int& usedPlanes, const std::pair<FirstType, SecondType>& a_pair) {
            return a_pair.second;
        }
    };

    template <typename T>
    FK_HOST_DEVICE_CNST auto cudaVectorToTuple(const T& cudaVectType) {
        static_assert(validCUDAVec<T>, "Non valid CUDA vetor type: cudaVectorToTuple<invalid_type>()");
        if constexpr (cn<T> == 1) {
            return fk::make_tuple(cudaVectType.x);
        } else if constexpr (cn<T> == 2) {
            return fk::make_tuple(cudaVectType.x, cudaVectType.y);
        } else if constexpr (cn<T> == 3) {
            return fk::make_tuple(cudaVectType.x, cudaVectType.y, cudaVectType.z);
        } else if constexpr (cn<T> == 4) {
            return fk::make_tuple(cudaVectType.x, cudaVectType.y, cudaVectType.z, cudaVectType.w);
        }
    }
} // namespace fk

#endif
