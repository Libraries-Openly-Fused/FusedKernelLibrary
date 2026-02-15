/* Copyright 2026 Oscar Amoros Huguet

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

#ifndef FK_ARRAY
#define FK_ARRAY

#include <fused_kernel/core/utils/vector_utils.h>
#include <cstddef>
#include <array>

namespace fk {
    template <typename T, size_t SIZE>
    struct Array {
        static constexpr size_t size{ SIZE };
        T values[SIZE];
        FK_HOST_DEVICE_CNST const T& operator[](const int index) const {
            return values[index];
        }
        FK_HOST_DEVICE_CNST T& operator[](const int index) {
            return values[index];
        }
    };

    template <typename T, size_t SIZE>
    union ArrayVector {
        enum { size = SIZE };
        T at[SIZE];
        FK_HOST_DEVICE_CNST ArrayVector(const T& initValue) {
            for (int i = 0; i < static_cast<int>(SIZE); i++) {
                at[i] = initValue;
            }
        }
        template <typename... Types>
        FK_HOST_DEVICE_CNST ArrayVector(const Types&... values) : at{static_cast<T>(values)...} {
            static_assert(all_of_v<T, TypeList<Types...>>, "Not all input types are the expected type T");
            static_assert(sizeof...(Types) == SIZE, "The number of elements passed to the constructor does not correspond with the ArrayVector size.");
        }
        FK_HOST_DEVICE_CNST T operator[](const size_t& index) const {
            return at[index];
        }
        FK_HOST_DEVICE_CNST T& operator[](const size_t& index) {
            return at[index];
        }
    };

    template <typename T>
    union ArrayVector<T, 0> {
        enum { size = 0 };
        FK_HOST_DEVICE_CNST ArrayVector() {}
    };

    template <typename T>
    union ArrayVector<T, 1> {
        static_assert(std::is_fundamental_v<T>, "ArrayVector<T, 1> can only be used with fundamental types");
        enum { size = 1 };
        T at[1];
        struct {
            T x;
        };
        FK_HOST_DEVICE_CNST ArrayVector(const T& x) : at{ x } {}
        FK_HOST_DEVICE_CNST ArrayVector(const typename VectorType<T, 1>::type_v& other) : x(other.x) {}
        FK_HOST_DEVICE_CNST T operator[](const size_t& index) const {
            return at[index];
        }
        FK_HOST_DEVICE_CNST T& operator[](const size_t& index) {
            return at[index];
        }
        FK_HOST_DEVICE_CNST T operator()(const int& index) const {
            return x;
        }
        FK_HOST_DEVICE_CNST ArrayVector<T, 1>& operator=(const VectorType_t<T, 1>& other) {
            x = other.x;
            return *this;
        }
    };

    template <typename T>
    union ArrayVector<T, 2> {
        static_assert(std::is_fundamental_v<T>, "ArrayVector<T, 2> can only be used with fundamental types");
        enum { size = 2 };
        T at[2];
        struct {
            T x, y;
        };
        FK_HOST_DEVICE_CNST ArrayVector(const T& x, const T& y) : at{ x, y } {}
        FK_HOST_DEVICE_CNST ArrayVector(const VectorType_t<T, 2>& other) : x(other.x), y(other.y) {}
        FK_HOST_DEVICE_CNST ArrayVector(const T& initValue) {
            for (int i = 0; i < size; i++) {
                at[i] = initValue;
            }
        }
        FK_HOST_DEVICE_CNST T operator[](const size_t& index) const {
            return at[index];
        }
        FK_HOST_DEVICE_CNST T& operator[](const size_t& index) {
            return at[index];
        }
        // This indexing method, is more costly than the previous one, but
        // if the variable is private, it will prevent the compiler from
        // placing it in local memory, which is way slower.
        FK_HOST_DEVICE_CNST T operator()(const int& index) const {
            return vector_at::f(index, make_<VectorType_t<T,size>>(x,y));
        }
        FK_HOST_DEVICE_CNST ArrayVector<T, 2>& operator=(const VectorType_t<T, 2>& other) {
            x = other.x;
            y = other.y;
            return *this;
        }
    };

    template <typename T>
    union ArrayVector<T, 3> {
        static_assert(std::is_fundamental_v<T>, "ArrayVector<T, 3> can only be used with fundamental types");
        enum { size = 3 };
        T at[3];
        struct {
            T x, y, z;
        };
        FK_HOST_DEVICE_CNST ArrayVector(const T& x, const T& y, const T& z) : at{ x, y, z } {}
        FK_HOST_DEVICE_CNST ArrayVector(const VectorType_t<T, 3>& other) : x(other.x), y(other.y), z(other.z) {}
        FK_HOST_DEVICE_CNST ArrayVector(const T& initValue) {
            for (int i = 0; i < size; i++) {
                at[i] = initValue;
            }
        }
        FK_HOST_DEVICE_CNST T operator[](const size_t& index) const {
            return at[index];
        }
        FK_HOST_DEVICE_CNST T& operator[](const size_t& index) {
            return at[index];
        }
        // This indexing method, is more costly than the previous one, but
        // if the variable is private, it will prevent the compiler from
        // placing it in local memory, which is way slower.
        FK_HOST_DEVICE_CNST T operator()(const int& index) const {
            return vector_at::f(index, make_<VectorType_t<T, size>>(x, y, z));
        }
        FK_HOST_DEVICE_CNST ArrayVector<T, 3>& operator=(const VectorType_t<T, 3>& other) {
            x = other.x;
            y = other.y;
            z = other.z;
            return *this;
        }
    };

    template <typename T>
    union ArrayVector<T, 4> {
        static_assert(std::is_fundamental_v<T>, "ArrayVector<T, 4> can only be used with fundamental types");
        enum { size = 4 };
        T at[4];
        struct {
            T x, y, z, w;
        };
        FK_HOST_DEVICE_CNST ArrayVector(const T& x, const T& y, const T& z, const T& w) : at{ x, y, z, w } {}
        FK_HOST_DEVICE_CNST ArrayVector(const VectorType_t<T, 4>& other) : x(other.x), y(other.y), z(other.z), w(other.w) {}
        FK_HOST_DEVICE_CNST ArrayVector(const T& initValue) {
            for (int i = 0; i < size; i++) {
                at[i] = initValue;
            }
        }
        FK_HOST_DEVICE_CNST T operator[](const size_t& index) const {
            return at[index];
        }
        FK_HOST_DEVICE_CNST T& operator[](const size_t& index) {
            return at[index];
        }
        // This indexing method, is more costly than the previous one, but
        // if the variable is private, it will prevent the compiler from
        // placing it in local memory, which is way slower.
        FK_HOST_DEVICE_CNST T operator()(const int& index) const {
            return vector_at::f(index, make_<VectorType_t<T, size>>(x, y, z, w));
        }
        FK_HOST_DEVICE_CNST ArrayVector<T, 4>& operator=(const VectorType_t<T, 4>& other) {
            x = other.x;
            y = other.y;
            z = other.z;
            w = other.w;
            return *this;
        }
    };

    template <typename CUDAVector>
    using ToArray = ArrayVector<VBase<CUDAVector>, cn<CUDAVector>>;

    template <typename V>
    FK_HOST_DEVICE_CNST ToArray<V> toArray(const V& vector) {
        return vector;
    }

    template <typename T, size_t SIZE, size_t... Idx>
    FK_HOST_DEVICE_CNST VectorType_t<T, SIZE> toVector_helper(const ArrayVector<T, SIZE>& array_v, const std::integer_sequence<int, Idx...>&) {
        return { array_v.at[Idx]... };
    }

    template <typename T, size_t SIZE>
    FK_HOST_DEVICE_CNST VectorType_t<T, SIZE> toVector(const ArrayVector<T, SIZE>& array_v) {
        static_assert(SIZE <= 4, "No Vector types available with size greater than 4");
        if constexpr (SIZE == 1) {
            return array_v.at[0];
        } else {
            return toVector_helper(array_v, std::index_sequence<SIZE>{});
        }
    }

    template <size_t BATCH, typename T>
    FK_HOST_CNST std::array<T, BATCH> make_set_std_array(const T& value) {
        std::array<T, BATCH> arr{};
        for (size_t i = 0; i < BATCH; i++) {
            arr[i] = value;
        }
        return arr;
    }

    template <typename ArrayLike>
    struct ArrayTraits;

    template <template <typename, size_t> class ArrayLike, typename T, size_t N>
    struct ArrayTraits<ArrayLike<T, N>> {
        using type = T;
        static constexpr size_t size = N;
    };

    template <typename ArrayLike>
    constexpr size_t arraySize = ArrayTraits<ArrayLike>::size;

    template <typename ArrayLike>
    using ArrayType = typename ArrayTraits<ArrayLike>::type;

    template <size_t BATCH, typename... ArrayTypes>
    constexpr bool allArraysSameSize_v = and_v<(arraySize<ArrayTypes> == BATCH)...>;

    template <template <typename, size_t> class ArrayLike, typename T, size_t N, typename F, std::size_t... Is>
    FK_HOST_CNST auto transformArray_impl(const ArrayLike<T, N>& input, F&& func, std::index_sequence<Is...>) {
        using ReturnType = decltype(func(std::declval<std::decay_t<decltype(input[0])>>()));
        return ArrayLike<ReturnType, N>{ { func(input[Is])... } };
    }

    template <typename ArrayLike, typename F>
    FK_HOST_CNST auto transformArray(const ArrayLike& input, F&& func) {
        return transformArray_impl(input, std::forward<F>(func), std::make_index_sequence<arraySize<ArrayLike>>{});
    }

    template <typename ArrayType, size_t... Idx>
    FK_HOST_DEVICE_CNST ArrayType getIndexArray_helper(const std::index_sequence<Idx...>&) {
        return {Idx...};
    }

    template <template <typename, size_t> class ArrayLike, typename T, size_t N>
    FK_HOST_DEVICE_CNST auto getIndexArray(const ArrayLike<T, N>&) -> ArrayLike<size_t, N> {
        return getIndexArray_helper<ArrayLike<size_t, N>>(std::make_index_sequence<N>{});
    }

    template <size_t N>
    FK_HOST_DEVICE_CNST ArrayVector<size_t, N> makeIndexArray() {
        return getIndexArray_helper<ArrayVector<size_t, N>>(std::make_index_sequence<N>{});
    }

    template <typename T, typename ArrayType, size_t... Idx>
    FK_HOST_DEVICE_CNST bool allValuesAre_helper(const T& value, const ArrayType& arrValues, const std::index_sequence<Idx...>&) {
        return ((arrValues[Idx] == value) && ...);
    }

    template <template <typename, size_t> class ArrayLike, typename T, size_t N>
    FK_HOST_DEVICE_CNST bool allValuesAre(const T& value, const ArrayLike<T, N>& arrValues) {
        return allValuesAre_helper(value, arrValues, std::make_index_sequence<N>{});
    }


} // namespace fk

#endif
