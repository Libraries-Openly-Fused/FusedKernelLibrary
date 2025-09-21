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

#ifndef FK_TUPLE
#define FK_TUPLE

#include <fused_kernel/core/utils/type_lists.h>
#include <fused_kernel/core/utils/cuda_vector_utils.h>
#include <array>

namespace fk {
    // Generic Tuple
    template <typename... Types>
    struct Tuple {};

    template <typename T>
    struct Tuple<T> {
        T instance;
        enum { size = 1 };
    };

    template <typename T, typename... Types>
    struct Tuple<T, Types...> {
        T instance;
        Tuple<Types...> next;
        enum { size = sizeof...(Types) + 1 };
    };

    // Primary template: defaults to false
    template <typename T>
    struct isTuple : std::false_type {};

    // Partial specialization: matches any specialization of Tuple
    template <typename... Types>
    struct isTuple<Tuple<Types...>> : std::true_type {};

    template <typename TypeToTest>
    constexpr bool isTuple_v = isTuple<std::decay_t<TypeToTest>>::value;

    struct TupleUtil {
        template <typename Tuple1, typename Tuple2, int... I1, int... I2>
        FK_HOST_DEVICE_FUSE auto cat_impl(const Tuple1& t1, std::integer_sequence<int, I1...>,
                                          const Tuple2& t2, std::integer_sequence<int, I2...>) {
            return make_tuple(get<I1>(t1)..., get<I2>(t2)...);
        }

        template <int INDEX, typename... InstanceTypes>
        FK_HOST_DEVICE_FUSE auto& get(Tuple<InstanceTypes...>& instances) {
            constexpr int numberOfInstances = Tuple<InstanceTypes...>::size;
            static_assert(INDEX < numberOfInstances,
                "Index out of range. There are not so many instances in the tuple.");
            if constexpr (INDEX > 0) {
                return get<INDEX - 1>(instances.next);
            } else if constexpr (INDEX == -1) {
                if constexpr (numberOfInstances > 0) {
                    return get<numberOfInstances - 1>(instances.next);
                } else {
                    return instances.instance;
                }
            } else {
                return instances.instance;
            }
        }

        template <int INDEX, typename... InstanceTypes>
        FK_HOST_DEVICE_FUSE auto get(const Tuple<InstanceTypes...>& instances) {
            constexpr int numberOfInstances = Tuple<InstanceTypes...>::size;
            static_assert(INDEX < numberOfInstances,
                "Index out of range. There are not so many instances in the tuple.");
            if constexpr (INDEX > 0) {
                return get<INDEX - 1>(instances.next);
            } else if constexpr (INDEX == -1) {
                if constexpr (numberOfInstances > 0) {
                    return get<numberOfInstances - 1>(instances);
                } else {
                    return instances.instance;
                }
            } else {
                return instances.instance;
            }
        }

        template <typename Tuple1, typename Tuple2>
        FK_HOST_DEVICE_FUSE auto cat(Tuple1& t1, Tuple2& t2) {
            return cat_impl(t1, std::make_integer_sequence<int, Tuple1::size>(),
                            t2, std::make_integer_sequence<int, Tuple2::size>());
        }

        template <typename Tuple1, typename Tuple2>
        FK_HOST_DEVICE_FUSE auto cat(const Tuple1& t1, const Tuple2& t2) {
            return cat_impl(t1, std::make_integer_sequence<int, Tuple1::size>(),
                            t2, std::make_integer_sequence<int, Tuple2::size>());
        }

        template <typename... Types>
        FK_HOST_DEVICE_FUSE auto make_tuple(const Types&... instances) {
            return Tuple<Types...>{instances...};
        }

        template <int INDEX, typename T, typename Tuple_>
        FK_HOST_DEVICE_FUSE auto tuple_insert(const T& instance, const Tuple_& tuple) {
            constexpr int numberOfInstances = Tuple_::size;
            static_assert(INDEX <= numberOfInstances,
                "Index out of range. There are not so many instances in the tuple.");
            if constexpr (INDEX == 0) {
                return TupleUtil::cat(make_tuple(instance), tuple);
            } else {
                if constexpr (Tuple_::size > 1) {
                    const auto [head, tail] = tuple;
                    return TupleUtil::cat(make_tuple(head), tuple_insert<INDEX - 1>(instance, tail));
                } else {
                    return TupleUtil::cat(tuple, make_tuple(instance));
                }
            }
        }
    };

    template <int INDEX, typename... Types>
    FK_HOST_DEVICE_CNST auto get(const Tuple<Types...>& tuple) {
        return TupleUtil::get<INDEX>(tuple);
    }

    template <int INDEX, typename... Types>
    FK_HOST_DEVICE_CNST auto& get(Tuple<Types...>& tuple) {
        return TupleUtil::get<INDEX>(tuple);
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
    FK_HOST_DEVICE_CNST auto tuple_insert(const T& element, const TupleLike& tuple) {
        return TupleUtil::tuple_insert<INDEX,T>(element, tuple);
    }

    template <typename T, typename TupleLike>
    FK_HOST_DEVICE_CNST auto tuple_insert_back(const TupleLike& tuple, const T& element) {
        return TupleUtil::tuple_insert<TupleLike::size, T>(element, tuple);
    }

    template <typename Tuple1, typename Tuple2>
    FK_HOST_DEVICE_CNST auto cat(const Tuple1& t1, const Tuple2& t2) {
        return TupleUtil::cat(t1, t2);
    }

    template <typename... Types>
    FK_HOST_DEVICE_CNST auto make_tuple(const Types&... instances) {
        return TupleUtil::make_tuple(instances...);
    }

    template <int INDEX, typename TupleLike>
    struct GetType {};

    template <int INDEX, template <typename...> class TupleLike, typename... Instances>
    struct GetType<INDEX, TupleLike<Instances...>> {
        using type = TypeAt_t<INDEX, TypeList<Instances...>>;
    };

    template <int INDEX, typename TupleLike>
    using get_type_t = typename GetType<INDEX, TupleLike>::type;

    template <typename F, typename Tuple, size_t... I>
    FK_HOST_DEVICE_CNST auto apply_impl(F&& f, Tuple& t, std::index_sequence<I...>)
        -> decltype(std::forward<F>(f)(get<I>(std::forward<Tuple>(t))...)) {
        return std::forward<F>(f)(get<I>(std::forward<Tuple>(t))...);
    }

    template <typename F, typename Tuple>
    FK_HOST_DEVICE_CNST auto apply(F&& f, Tuple& t)
        -> decltype(apply_impl(std::forward<F>(f), std::forward<Tuple>(t),
            std::make_index_sequence<Tuple::size>())) {
        return apply_impl(std::forward<F>(f), std::forward<Tuple>(t),
            std::make_index_sequence<Tuple::size>());
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
    FK_HOST_DEVICE_CNST auto buildOperationSequence(const IOpTypes&... instantiableOperationInstances) {
        return InstantiableOperationSequence<IOpTypes...> {{instantiableOperationInstances...}};
    }

    template <typename... IOpTypes>
    FK_HOST_DEVICE_CNST auto buildOperationSequence_tup(const Tuple<IOpTypes...>& instantiableOperationInstances) {
        return apply([](const auto&... args) {
            return buildOperationSequence(args...);
            }, instantiableOperationInstances);
    }

    // Util to insert an element before the last element of a tuple
    template <typename T, typename Tuple>
    FK_HOST_DEVICE_CNST auto insert_before_last_tup(const T& t, const Tuple& args) {
        return tuple_insert<Tuple::size - 1>(t, args);
    }

    template<typename T, typename... Args>
    FK_HOST_DEVICE_CNST auto insert_before_last(const T& t, const Args&... args) {
        return tuple_insert<sizeof...(Args) - 1>(t, Tuple<Args...>{args...});
    }

    /*template <typename TranslateType, typename SourceType, size_t NElems, int... Idx, typename... ExtraParams>
    constexpr inline std::array<typename TranslateType::OutputType, NElems>
        static_translate_helper(const int& usedPlanes,
            const std::array<SourceType, NElems>& srcArray,
            const std::integer_sequence<int, Idx...>&,
            const ExtraParams&... extParams) {
        return { (TranslateType::template translate<Idx>(usedPlanes, srcArray[Idx], extParams...))... };
    }

    template <typename TranslateType, typename SourceType, size_t NElems, typename... ExtraParams>
    constexpr inline std::array<typename TranslateType::OutputType, NElems>
        static_translate(const int& usedPlanes, const std::array<SourceType, NElems>& srcArray, const ExtraParams&... extParams) {
        return static_translate_helper<TranslateType>(usedPlanes, srcArray, std::make_integer_sequence<int, NElems>{}, extParams...);
    }*/

    template <typename FirstType, typename SecondType>
    struct GetFirst {
        using OutputType = FirstType;
        template <int Idx>
        static constexpr inline FirstType translate(const int& usedPlanes, const std::pair<FirstType, SecondType>& a_pair) {
            return a_pair.first;
        }
    };

    template <typename FirstType, typename SecondType>
    struct GetSecond {
        using OutputType = SecondType;
        template <int Idx>
        static constexpr inline SecondType translate(const int& usedPlanes, const std::pair<FirstType, SecondType>& a_pair) {
            return a_pair.second;
        }
    };

    /*template <typename FT, typename ST, size_t NElems>
    constexpr inline std::array<FT, NElems>
        static_translate_get_first(const std::array<std::pair<FT, ST>, NElems>& srcArray) {
        return static_translate_helper<GetFirst<FT, ST>>(NElems, srcArray, std::make_integer_sequence<int, NElems>{});
    }

    template <typename FT, typename ST, size_t NElems>
    constexpr inline std::array<ST, NElems>
        static_translate_get_second(const std::array<std::pair<FT, ST>, NElems>& srcArray) {
        return static_translate_helper<GetSecond<FT, ST>>(NElems, srcArray, std::make_integer_sequence<int, NElems>{});
    }*/

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
