/* Copyright 2023-2025 Oscar Amoros Huguet

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
#ifndef FK_CUDA_VECTOR_UTILS
#define FK_CUDA_VECTOR_UTILS

#include <cassert>

#include <fused_kernel/core/utils/type_lists.h>
#include <fused_kernel/core/utils/template_operations.h>
#include <fused_kernel/core/data/vector_types.h>

namespace fk {

    template <typename BaseType, int Channels>
    struct VectorType {};

#define VECTOR_TYPE(BaseType) \
    template <> \
    struct VectorType<BaseType, 1> { using type = BaseType; using type_v = BaseType ## 1; }; \
    template <> \
    struct VectorType<BaseType, 2> { using type = BaseType ## 2; using type_v = type; }; \
    template <> \
    struct VectorType<BaseType, 3> { using type = BaseType ## 3; using type_v = type; }; \
    template <> \
    struct VectorType<BaseType, 4> { using type = BaseType ## 4; using type_v = type; };

    VECTOR_TYPE(uchar)
    VECTOR_TYPE(char)
    VECTOR_TYPE(short)
    VECTOR_TYPE(ushort)
    VECTOR_TYPE(int)
    VECTOR_TYPE(uint)
    VECTOR_TYPE(long)
    VECTOR_TYPE(ulong)
    VECTOR_TYPE(longlong)
    VECTOR_TYPE(ulonglong)
    VECTOR_TYPE(float)
    VECTOR_TYPE(double)
    VECTOR_TYPE(bool)
#undef VECTOR_TYPE

    template <typename BaseType, int Channels>
    using VectorType_t = typename VectorType<BaseType, Channels>::type;

    template <uint CHANNELS>
    using VectorTypeList = TypeList<VectorType_t<bool, CHANNELS>, VectorType_t<uchar, CHANNELS>, VectorType_t<char, CHANNELS>,
                                    VectorType_t<ushort, CHANNELS>, VectorType_t<short, CHANNELS>,
                                    VectorType_t<uint, CHANNELS>, VectorType_t<int, CHANNELS>,
                                    VectorType_t<ulong, CHANNELS>, VectorType_t<long, CHANNELS>,
                                    VectorType_t<ulonglong, CHANNELS>, VectorType_t<longlong, CHANNELS>,
                                    VectorType_t<float, CHANNELS>, VectorType_t<double, CHANNELS>>;
    using FloatingTypes = TypeList<float, double>;
    using IntegralTypes = TypeList<uchar, char, ushort, short, uint, int, ulong, long, ulonglong, longlong>;
    using StandardTypes = TypeListCat_t<TypeListCat_t<TypeList<bool>, IntegralTypes>, FloatingTypes>;
    using VOne = TypeList<bool1, uchar1, char1, ushort1, short1, uint1, int1, ulong1, long1, ulonglong1, longlong1, float1, double1>;
    using VTwo = VectorTypeList<2>;
    using VThree = VectorTypeList<3>;
    using VFour = VectorTypeList<4>;
    using VAll = typename TypeList<VOne, VTwo, VThree, VFour>::type;

    template <typename T>
    constexpr bool validCUDAVec = one_of<T, VAll>::value;

    template <typename T>
    struct IsCudaVector : std::conditional_t<validCUDAVec<T>, std::true_type, std::false_type> {};

    template <typename T>
    FK_HOST_DEVICE_CNST int Channels() {
        if constexpr (one_of_v<T, VOne> || !validCUDAVec<T>) {
            return 1;
        } else if constexpr (one_of_v<T, VTwo>) {
            return 2;
        } else if constexpr (one_of_v<T, VThree>) {
            return 3;
        } else if constexpr (one_of_v<T, VFour>) {
            return 4;
        }
    }

    template <typename T>
    constexpr int cn = Channels<T>();

    template <typename V>
    struct VectorTraits {};

#define VECTOR_TRAITS(BaseType) \
    template <> \
    struct VectorTraits<BaseType> { using base = BaseType; enum {bytes=sizeof(base)}; }; \
    template <> \
    struct VectorTraits<BaseType ## 1> { using base = BaseType; enum {bytes=sizeof(base)}; }; \
    template <> \
    struct VectorTraits<BaseType ## 2> { using base = BaseType; enum {bytes=sizeof(base)*2}; }; \
    template <> \
    struct VectorTraits<BaseType ## 3> { using base = BaseType; enum {bytes=sizeof(base)*3}; }; \
    template <> \
    struct VectorTraits<BaseType ## 4> { using base = BaseType; enum {bytes=sizeof(base)*4}; };

    VECTOR_TRAITS(bool)
    VECTOR_TRAITS(uchar)
    VECTOR_TRAITS(char)
    VECTOR_TRAITS(short)
    VECTOR_TRAITS(ushort)
    VECTOR_TRAITS(int)
    VECTOR_TRAITS(uint)
    VECTOR_TRAITS(long)
    VECTOR_TRAITS(ulong)
    VECTOR_TRAITS(longlong)
    VECTOR_TRAITS(ulonglong)
    VECTOR_TRAITS(float)
    VECTOR_TRAITS(double)
#undef VECTOR_TRAITS

    template <typename T>
    using VBase = typename VectorTraits<T>::base;
    
    template <int idx, typename T>
    FK_HOST_DEVICE_CNST auto VectorAt(const T& vector) {
        if constexpr (idx == 0) {
            if constexpr (validCUDAVec<T>) {
                return vector.x;
            } else {
                return vector;
            }
        } else if constexpr (idx == 1) {
            static_assert(validCUDAVec<T>, "Non valid CUDA vetor type: VectorAt<invalid_type>()");
            static_assert(cn<T> >= 2, "Vector type smaller than 2 elements has no member y");
            return vector.y;
        } else if constexpr (idx == 2) {
            static_assert(validCUDAVec<T>, "Non valid CUDA vetor type: VectorAt<invalid_type>()");
            static_assert(cn<T> >= 3, "Vector type smaller than 3 elements has no member z");
            return vector.z;
        } else if constexpr (idx == 3) {
            static_assert(validCUDAVec<T>, "Non valid CUDA vetor type: VectorAt<invalid_type>()");
            static_assert(cn<T> == 4, "Vector type smaller than 4 elements has no member w");
            return vector.w;
        }
    }

    template <typename T>
    FK_HOST_DEVICE_CNST std::enable_if_t<(cn<T> == 1), VBase<T>> VectorAt(const int& idx, const T& vector) {
        assert((idx == 0 && idx >= 0) && "Index out of range. Either the Vector type has 1 channel or the type is not a CUDA Vector type");
        if constexpr (validCUDAVec<T>) {
            return vector.x;
        } else {
            return vector;
        }
    }

    template <typename T>
    FK_HOST_DEVICE_CNST std::enable_if_t<(cn<T> == 2), VBase<T>> VectorAt(const int& idx, const T& vector) {
        assert((idx < 2 && idx >= 0) && "Index out of range. Vector type has only 2 channels.");
        assert(validCUDAVec<T> && "Non valid CUDA vetor type: VectorAt<invalid_type>()");
        if (idx == 0) {
            return vector.x;
        } else {
            return vector.y;
        }
    }

    template <typename T>
    FK_HOST_DEVICE_CNST std::enable_if_t<(cn<T> == 3), VBase<T>> VectorAt(const int& idx, const T& vector) {
        assert((idx < 3 && idx >= 0) && "Index out of range. Vector type has only 2 channels.");
        assert(validCUDAVec<T> && "Non valid CUDA vetor type: VectorAt<invalid_type>()");
        if (idx == 0) {
            return vector.x;
        } else if (idx == 1) {
            return vector.y;
        } else {
            return vector.z;
        }
    }

    template <typename T>
    FK_HOST_DEVICE_CNST std::enable_if_t<(cn<T> == 4), VBase<T>> VectorAt(const int& idx, const T& vector) {
        assert((idx < 4 && idx >= 0) && "Index out of range. Vector type has only 2 channels.");
        assert(validCUDAVec<T> && "Non valid CUDA vetor type: VectorAt<invalid_type>()");
        if (idx == 0) {
            return vector.x;
        } else if (idx == 1) {
            return vector.y;
        } else if (idx == 2) {
            return vector.z;
        } else {
            return vector.w;
        }
    }

    // Automagically making any CUDA vector type from a template type
    // It will not compile if you try to do bad things. The number of elements
    // need to conform to T, and the type of the elements will always be casted.
    struct make {
        template <typename T, typename... Numbers>
        FK_HOST_DEVICE_FUSE T type(const Numbers&... pack) {
            static_assert(validCUDAVec<T>, "Non valid CUDA vetor type: make::type<invalid_type>()");
#if defined(_MSC_VER) && _MSC_VER >= 1910 && _MSC_VER <= 1916
            return T{ static_cast<std::decay_t<decltype(T::x)>>(pack)... };
#else
            if constexpr (std::is_union_v<T>) {
                return T{ static_cast<std::decay_t<decltype(T::at[0])>>(pack)... };
            }
            else if constexpr (std::is_class_v<T>) {
                return T{ static_cast<std::decay_t<decltype(T::x)>>(pack)... };
            }
            else {
                static_assert(std::is_union_v<T> || std::is_class_v<T>,
                    "make::type can only be used with CUDA vector_types or fk vector_types");
                return T{};
            }
#endif
        }
    };

    template <typename T, typename... Numbers>
    FK_HOST_DEVICE_CNST T make_(const Numbers&... pack) {
        if constexpr (std::is_aggregate_v<T>) {
            return make::type<T>(pack...);
        } else {
            static_assert(sizeof...(pack) == 1, "make_ can only be used to create fk vector types");
            return first(pack...);
        }
    }

    template <typename T, typename Enabler = void>
    struct UnaryVectorSet;

    // This case exists to make things easier when we don't know if the type
    // is going to be a vector type or a normal type
    template <typename T>
    struct UnaryVectorSet<T, typename std::enable_if_t<!validCUDAVec<T>, void>> {
        FK_HOST_DEVICE_FUSE T exec(const T& val) {
            return val;
        }
    };

    template <typename T>
    struct UnaryVectorSet<T, typename std::enable_if_t<validCUDAVec<T>, void>> {
        FK_HOST_DEVICE_FUSE T exec(const VBase<T>& val) {
            if constexpr (cn<T> == 1) {
                return { val };
            }
            else if constexpr (cn<T> == 2) {
                return { val, val };
            }
            else if constexpr (cn<T> == 3) {
                return { val, val, val };
            }
            else {
                return { val, val, val, val };
            }
        }
    };

    template <typename T>
    FK_HOST_DEVICE_CNST T make_set(const typename VectorTraits<T>::base& val) {
        return UnaryVectorSet<T>::exec(val);
    }

    template <typename T>
    FK_HOST_DEVICE_CNST T make_set(const T& val) {
        return UnaryVectorSet<T>::exec(val);
    }

} // namespace fk

#ifdef DEBUG_MATRIX
#include <iostream>

template <typename T>
struct to_printable {
    FK_HOST_FUSE int exec(T val) {
        if constexpr (sizeof(T) == 1) {
            return static_cast<int>(val);
        }
        else if constexpr (sizeof(T) > 1) {
            return val;
        }
    }
};

template <typename T>
struct print_vector {
    FK_HOST_FUSE std::ostream& exec(std::ostream& outs, T val) {
        if constexpr (!fk::validCUDAVec<T>) {
            outs << val;
            return outs;
        }
        else if constexpr (fk::cn<T> == 1) {
            outs << "{" << to_printable<decltype(T::x)>::exec(val.x) << "}";
            return outs;
        }
        else if constexpr (fk::cn<T> == 2) {
            outs << "{" << to_printable<decltype(T::x)>::exec(val.x) <<
                ", " << to_printable<decltype(T::y)>::exec(val.y) << "}";
            return outs;
        }
        else if constexpr (fk::cn<T> == 3) {
            outs << "{" << to_printable<decltype(T::x)>::exec(val.x) <<
                ", " << to_printable<decltype(T::y)>::exec(val.y) <<
                ", " << to_printable<decltype(T::z)>::exec(val.z) << "}";
            return outs;
        }
        else {
            outs << "{" << to_printable<decltype(T::x)>::exec(val.x) <<
                ", " << to_printable<decltype(T::y)>::exec(val.y) <<
                ", " << to_printable<decltype(T::z)>::exec(val.z) <<
                ", " << to_printable<decltype(T::w)>::exec(val.w) << "}";
            return outs;
        }
    }
};

template <typename T>
inline constexpr typename std::enable_if_t<fk::validCUDAVec<T>, std::ostream&> operator<<(std::ostream& outs, const T& val) {
    return print_vector<T>::exec(outs, val);
}
#endif

// ####################### VECTOR OPERATORS ##########################
// Implemented in a way that the return types follow the c++ standard, for each vector component
// The user is responsible for knowing the type conversion hazards, inherent to the C++ language.
#define VEC_UNARY_UNIVERSAL(op) \
template <typename T> \
FK_HOST_DEVICE_CNST auto operator op(const T& a) -> std::enable_if_t<fk::validCUDAVec<T>, fk::VectorType_t<decltype(op std::declval<fk::VBase<T>>()), fk::cn<T>>> { \
    using O = fk::VectorType_t<decltype(op std::declval<fk::VBase<T>>()), fk::cn<T>>; \
    if constexpr (fk::cn<T> == 1) { \
        return fk::make_<O>(op a.x); \
    } else if constexpr (fk::cn<T> == 2) { \
        return fk::make_<O>(op a.x, op a.y); \
    } else if constexpr (fk::cn<T> == 3) { \
        return fk::make_<O>(op a.x, op a.y, op a.z); \
    } else { \
        return fk::make_<O>(op a.x, op a.y, op a.z, op a.w); \
    } \
}

VEC_UNARY_UNIVERSAL(-)
VEC_UNARY_UNIVERSAL(!)
VEC_UNARY_UNIVERSAL(~)

#undef VEC_UNARY_UNIVERSAL

#define VEC_COMPOUND_UNIVERSAL(op) \
template <typename I1, typename I2> \
FK_HOST_DEVICE_CNST auto operator op(I1& a, const I2& b) \
-> std::enable_if_t<std::is_fundamental_v<fk::VBase<I1>> && std::is_fundamental_v<fk::VBase<I2>> && !(std::is_fundamental_v<I1>&& std::is_fundamental_v<I2>), I1> { \
    static_assert(fk::validCUDAVec<I1>, "First operand must be a valid CUDA vector type. You can not store a vector type on an scalar type."); \
    if constexpr (fk::IsCudaVector<I2>::value) { \
        static_assert(fk::cn<I1> == fk::cn<I2>, "Vectors must have the same number of channels"); \
        a.x op b.x; \
        if constexpr (fk::cn<I1> >= 2) { a.y op b.y; } \
        if constexpr (fk::cn<I1> >= 3) { a.z op b.z; } \
        if constexpr (fk::cn<I1> == 4) { a.w op b.w; } \
    } else { \
        a.x op b; \
        if constexpr (fk::cn<I1> >= 2) { a.y op b; } \
        if constexpr (fk::cn<I1> >= 3) { a.z op b; } \
        if constexpr (fk::cn<I1> == 4) { a.w op b; } \
    } \
    return a; \
}

VEC_COMPOUND_UNIVERSAL(-=)
VEC_COMPOUND_UNIVERSAL(+=)
VEC_COMPOUND_UNIVERSAL(*=)
VEC_COMPOUND_UNIVERSAL(/=)
VEC_COMPOUND_UNIVERSAL(&=)
VEC_COMPOUND_UNIVERSAL(|=)

#undef VEC_COMPOUND_UNIVERSAL

// We don't need to check for I2 being a vector type, because the enable_if condition ensures it is a cuda vector if the two previous conditions are false
#define VEC_BINARY_UNIVERSAL(op) \
template <typename I1, typename I2> \
FK_HOST_DEVICE_CNST auto operator op(const I1& a, const I2& b) \
    -> std::enable_if_t<std::is_fundamental_v<fk::VBase<I1>> && std::is_fundamental_v<fk::VBase<I2>> && !(std::is_fundamental_v<I1> && std::is_fundamental_v<I2>), \
                        typename fk::VectorType<decltype(std::declval<fk::VBase<I1>>() op std::declval<fk::VBase<I2>>()), \
                                                (fk::cn<I1> > fk::cn<I2> ? fk::cn<I1> : fk::cn<I2>)>::type_v> { \
    using O = typename fk::VectorType<decltype(std::declval<fk::VBase<I1>>() op std::declval<fk::VBase<I2>>()), \
                                      (fk::cn<I1> > fk::cn<I2> ? fk::cn<I1> : fk::cn<I2>)>::type_v; \
    if constexpr (fk::validCUDAVec<I1> && fk::validCUDAVec<I2>) { \
        static_assert(fk::cn<I1> == fk::cn<I2>, "Vectors must have the same number of channels"); \
        if constexpr (fk::cn<I1> == 1) { \
            return fk::make_<O>(a.x op b.x); \
        } else if constexpr (fk::cn<I1> == 2) { \
            return fk::make_<O>(a.x op b.x, a.y op b.y); \
        } else if constexpr (fk::cn<I1> == 3) { \
            return fk::make_<O>(a.x op b.x, a.y op b.y, a.z op b.z); \
        } else { \
            return fk::make_<O>(a.x op b.x, a.y op b.y, a.z op b.z, a.w op b.w); \
        } \
    } else if constexpr (fk::validCUDAVec<I1>) { \
        if constexpr (fk::cn<I1> == 1) { \
            return fk::make_<O>(a.x op b); \
        } else if constexpr (fk::cn<I1> == 2) { \
            return fk::make_<O>(a.x op b, a.y op b); \
        } else if constexpr (fk::cn<I1> == 3) { \
            return fk::make_<O>(a.x op b, a.y op b, a.z op b); \
        } else { \
            return fk::make_<O>(a.x op b, a.y op b, a.z op b, a.w op b); \
        } \
    } else { \
        if constexpr (fk::cn<I2> == 1) { \
            return fk::make_<O>(a op b.x); \
        } else if constexpr (fk::cn<I2> == 2) { \
            return fk::make_<O>(a op b.x, a op b.y); \
        } else if constexpr (fk::cn<I2> == 3) { \
            return fk::make_<O>(a op b.x, a op b.y, a op b.z); \
        } else { \
            return fk::make_<O>(a op b.x, a op b.y, a op b.z, a op b.w); \
        } \
    } \
}

VEC_BINARY_UNIVERSAL(+)
VEC_BINARY_UNIVERSAL(-)
VEC_BINARY_UNIVERSAL(*)
VEC_BINARY_UNIVERSAL(/)
VEC_BINARY_UNIVERSAL(==)
VEC_BINARY_UNIVERSAL(!=)
VEC_BINARY_UNIVERSAL(>)
VEC_BINARY_UNIVERSAL(<)
VEC_BINARY_UNIVERSAL(>=)
VEC_BINARY_UNIVERSAL(<=)
VEC_BINARY_UNIVERSAL(&&)
VEC_BINARY_UNIVERSAL(||)
VEC_BINARY_UNIVERSAL(&)
VEC_BINARY_UNIVERSAL(|)
VEC_BINARY_UNIVERSAL(^)

#undef VEC_BINARY_UNIVERSAL
// ######################## VECTOR OPERATORS ##########################

namespace fk {
    template <typename T, typename = void>
    struct IsVectorType : std::false_type {};
    template <typename T>
    struct IsVectorType<T, std::void_t<decltype(T::x)>> : std::true_type {};
} // namespace fk

#endif
