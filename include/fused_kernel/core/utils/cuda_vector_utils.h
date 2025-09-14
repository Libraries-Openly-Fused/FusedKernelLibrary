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
#include <utility>

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

    template <>
    struct VectorType<char, 1> { using type = std::conditional_t<std::is_unsigned_v<char>, uchar, schar>; using type_v = std::conditional_t<std::is_unsigned_v<char>, uchar1, char1>; };
    template <>
    struct VectorType<char, 2> { using type = std::conditional_t<std::is_unsigned_v<char>, uchar2, char2>; using type_v = type; };
    template <>
    struct VectorType<char, 3> { using type = std::conditional_t<std::is_unsigned_v<char>, uchar3, char3>; using type_v = type; };
    template <>
    struct VectorType<char, 4> { using type = std::conditional_t<std::is_unsigned_v<char>, uchar4, char4>; using type_v = type; };

    template <>
    struct VectorType<schar, 1> { using type = schar; using type_v = char1; };
    template <>
    struct VectorType<schar, 2> { using type = char2; using type_v = type; };
    template <>
    struct VectorType<schar, 3> { using type = char3; using type_v = type; };
    template <>
    struct VectorType<schar, 4> { using type = char4; using type_v = type; };

    template <typename BaseType, int Channels>
    using VectorType_t = typename VectorType<BaseType, Channels>::type;

    template <uint CHANNELS>
    using VectorTypeList = TypeList<VectorType_t<bool, CHANNELS>, VectorType_t<uchar, CHANNELS>, VectorType_t<schar, CHANNELS>,
                                    VectorType_t<ushort, CHANNELS>, VectorType_t<short, CHANNELS>,
                                    VectorType_t<uint, CHANNELS>, VectorType_t<int, CHANNELS>,
                                    VectorType_t<ulong, CHANNELS>, VectorType_t<long, CHANNELS>,
                                    VectorType_t<ulonglong, CHANNELS>, VectorType_t<longlong, CHANNELS>,
                                    VectorType_t<float, CHANNELS>, VectorType_t<double, CHANNELS>>;

    using FloatingTypes = TypeList<float, double>;
    using IntegralTypes = TypeList<uchar, char, schar, ushort, short, uint, int, ulong, long, ulonglong, longlong>;
    using IntegralBaseTypes = TypeList<uchar, schar, ushort, short, uint, int, ulong, long, ulonglong, longlong>;
    using StandardTypes = TypeListCat_t<IntegralTypes, FloatingTypes>::addFront<bool>;
    using BaseTypes = TypeListCat_t<IntegralBaseTypes, FloatingTypes>::addFront<bool>;
    using VOne = TypeList<bool1, uchar1, char1, ushort1, short1, uint1, int1, ulong1, long1, ulonglong1, longlong1, float1, double1>;
    using VTwo = VectorTypeList<2>;
    using VThree = VectorTypeList<3>;
    using VFour = VectorTypeList<4>;
    using VAll = TypeListCat_t<VOne, VTwo, VThree, VFour>;

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

    template <>
    struct VectorTraits<schar> { using base = schar; enum { bytes = sizeof(base) }; };
    template <>
    struct VectorTraits<char> { using base = std::conditional_t<std::is_unsigned_v<char>, uchar, schar>; enum { bytes = sizeof(base) }; };
    template <>
    struct VectorTraits<char1> { using base = schar; enum { bytes = sizeof(base) }; };
    template <>
    struct VectorTraits<char2> { using base = schar; enum { bytes = sizeof(base) * 2 }; };
    template <>
    struct VectorTraits<char3> { using base = schar; enum { bytes = sizeof(base) * 3 }; };
    template <>
    struct VectorTraits<char4> { using base = schar; enum { bytes = sizeof(base) * 4 }; };

    template <typename T>
    using VBase = typename VectorTraits<T>::base;
    
    template <int idx, typename T>
    FK_HOST_DEVICE_CNST auto vectorAt(const T& vector) {
        if constexpr (idx == 0) {
            if constexpr (validCUDAVec<T>) {
                return vector.x;
            } else {
                return vector;
            }
        } else if constexpr (idx == 1) {
            static_assert(validCUDAVec<T>, "Non valid CUDA vetor type: vectorAt<invalid_type>()");
            static_assert(cn<T> >= 2, "Vector type smaller than 2 elements has no member y");
            return vector.y;
        } else if constexpr (idx == 2) {
            static_assert(validCUDAVec<T>, "Non valid CUDA vetor type: vectorAt<invalid_type>()");
            static_assert(cn<T> >= 3, "Vector type smaller than 3 elements has no member z");
            return vector.z;
        } else if constexpr (idx == 3) {
            static_assert(validCUDAVec<T>, "Non valid CUDA vetor type: vectorAt<invalid_type>()");
            static_assert(cn<T> == 4, "Vector type smaller than 4 elements has no member w");
            return vector.w;
        }
    }

    template <typename T>
    FK_HOST_DEVICE_CNST std::enable_if_t<(cn<T> == 1), VBase<T>> vectorAt(const int& idx, const T& vector) {
        assert((idx == 0 && idx >= 0) && "Index out of range. Either the Vector type has 1 channel or the type is not a CUDA Vector type");
        if constexpr (validCUDAVec<T>) {
            return vector.x;
        } else {
            return vector;
        }
    }

    template <typename T>
    FK_HOST_DEVICE_CNST std::enable_if_t<(cn<T> == 2), VBase<T>> vectorAt(const int& idx, const T& vector) {
        assert((idx < 2 && idx >= 0) && "Index out of range. Vector type has only 2 channels.");
        assert(validCUDAVec<T> && "Non valid CUDA vetor type: vectorAt<invalid_type>()");
        if (idx == 0) {
            return vector.x;
        } else {
            return vector.y;
        }
    }

    template <typename T>
    FK_HOST_DEVICE_CNST std::enable_if_t<(cn<T> == 3), VBase<T>> vectorAt(const int& idx, const T& vector) {
        assert((idx < 3 && idx >= 0) && "Index out of range. Vector type has only 2 channels.");
        assert(validCUDAVec<T> && "Non valid CUDA vetor type: vectorAt<invalid_type>()");
        if (idx == 0) {
            return vector.x;
        } else if (idx == 1) {
            return vector.y;
        } else {
            return vector.z;
        }
    }

    template <typename T>
    FK_HOST_DEVICE_CNST std::enable_if_t<(cn<T> == 4), VBase<T>> vectorAt(const int& idx, const T& vector) {
        assert((idx < 4 && idx >= 0) && "Index out of range. Vector type has only 2 channels.");
        assert(validCUDAVec<T> && "Non valid CUDA vetor type: vectorAt<invalid_type>()");
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
            static_assert(sizeof...(pack) == 1, "passing more than one argument for a non cuda vector type");
            return (pack, ...);
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

    // Utils to check detais about types and pairs of types
    template <typename I1, typename I2, typename = void>
    struct BothIntegrals : public std::false_type {};

    template <typename I1, typename I2>
    struct BothIntegrals<I1, I2, std::enable_if_t<std::is_integral_v<fk::VBase<I1>>&& std::is_integral_v<fk::VBase<I2>>, void>> : public std::true_type {};

    template <typename I1, typename I2, typename = void>
    struct AreVVEqCN : public std::false_type {};

    template <typename I1, typename I2>
    struct AreVVEqCN<I1, I2, std::enable_if_t<fk::validCUDAVec<I1>&& fk::validCUDAVec<I2> && (fk::cn<I1> == fk::cn<I2>), void>> : public std::true_type {};

    template <typename I1, typename I2, typename = void>
    struct AreSV : public std::false_type {};

    template <typename I1, typename I2>
    struct AreSV<I1, I2, std::enable_if_t<std::is_fundamental_v<I1>&& fk::validCUDAVec<I2>, void>> : public std::true_type {};

    template <typename I1, typename I2, typename = void>
    struct AreVS : public std::false_type {};

    template <typename I1, typename I2>
    struct AreVS<I1, I2, std::enable_if_t<fk::validCUDAVec<I1>&& std::is_fundamental_v<I2>, void>> : public std::true_type {};

    // Utils to check if the type or combination of types can be used with a particular operator
    template <typename T, typename = void>
    struct CanUnary : public std::false_type {};

    template <typename T>
    struct CanUnary<T, std::enable_if_t<fk::validCUDAVec<T>, void>> : public std::true_type {};

    template <typename I1, typename I2, typename = void>
    struct CanBinary : public std::false_type {};

    template <typename I1, typename I2>
    struct CanBinary<I1, I2,
        std::enable_if_t<AreVVEqCN<I1, I2>::value || AreSV<I1, I2>::value ||
        AreVS<I1, I2>::value, void>> : public std::true_type {};

    template <typename I1, typename I2, typename = void>
    struct CanBinaryBitwise : public std::false_type {};

    template <typename I1, typename I2>
    struct CanBinaryBitwise<I1, I2,
        std::enable_if_t<(AreVVEqCN<I1, I2>::value || AreSV<I1, I2>::value ||
            AreVS<I1, I2>::value) && BothIntegrals<I1, I2>::value, void>> : public std::true_type {};

    template <typename I1, typename I2, typename = void>
    struct CanCompound : public std::false_type {};

    template <typename I1, typename I2>
    struct CanCompound<I1, I2,
        std::enable_if_t<AreVVEqCN<I1, I2>::value || AreVS<I1, I2>::value, void>> : public std::true_type {};

    template <typename I1, typename I2, typename = void>
    struct CanCompoundLogical : public std::false_type {};

    template <typename I1, typename I2>
    struct CanCompoundLogical<I1, I2,
        std::enable_if_t<(AreVVEqCN<I1, I2>::value || AreVS<I1, I2>::value) && BothIntegrals<I1, I2>::value, void>> : public std::true_type {};

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
#if VS2017_COMPILER
#define VEC_UNARY_OP(op, input_type) \
FK_HOST_DEVICE_CNST auto operator op(const input_type ## 1 & a) \
{ \
    using OutputType = typename fk::VectorType<decltype(op std::declval<input_type>()), 1>::type_v; \
    return OutputType{op (a.x)}; \
} \
FK_HOST_DEVICE_CNST auto operator op(const input_type ## 2 & a) \
{ \
    using OutputType = fk::VectorType_t<decltype(op std::declval<input_type>()), 2>; \
    return OutputType{op (a.x), op (a.y)}; \
} \
FK_HOST_DEVICE_CNST auto operator op(const input_type ## 3 & a) \
{ \
    using OutputType = fk::VectorType_t<decltype(op std::declval<input_type>()), 3>; \
    return OutputType{op (a.x), op (a.y), op (a.z)}; \
} \
FK_HOST_DEVICE_CNST auto operator op(const input_type ## 4 & a) \
{ \
    using OutputType = fk::VectorType_t<decltype(op std::declval<input_type>()), 4>; \
    return OutputType{op (a.x), op (a.y), op (a.z), op (a.w)}; \
}

VEC_UNARY_OP(-, char)
VEC_UNARY_OP(-, short)
VEC_UNARY_OP(-, int)
VEC_UNARY_OP(-, float)
VEC_UNARY_OP(-, double)

VEC_UNARY_OP(!, uchar)
VEC_UNARY_OP(!, char)
VEC_UNARY_OP(!, ushort)
VEC_UNARY_OP(!, short)
VEC_UNARY_OP(!, int)
VEC_UNARY_OP(!, uint)
VEC_UNARY_OP(!, float)
VEC_UNARY_OP(!, double)

VEC_UNARY_OP(~, uchar)
VEC_UNARY_OP(~, char)
VEC_UNARY_OP(~, ushort)
VEC_UNARY_OP(~, short)
VEC_UNARY_OP(~, int)
VEC_UNARY_OP(~, uint)

#undef VEC_UNARY_OP

#define VEC_COMPOUND_OP(op, modificable_type, input_type) \
FK_HOST_DEVICE_CNST modificable_type ## 1& operator op(modificable_type ## 1 & a, const input_type ## 1 & b) { \
    a.x op b.x; \
    return a; \
} \
FK_HOST_DEVICE_CNST modificable_type ## 2& operator op(modificable_type ## 2 & a, const input_type ## 2 & b) { \
    a.x op b.x; \
    a.y op b.y; \
    return a; \
} \
FK_HOST_DEVICE_CNST modificable_type ## 3& operator op(modificable_type ## 3 & a, const input_type ## 3 & b) { \
    a.x op b.x; \
    a.y op b.y; \
    a.z op b.z; \
    return a; \
} \
FK_HOST_DEVICE_CNST modificable_type ## 4& operator op(modificable_type ## 4 & a, const input_type ## 4 & b) { \
    a.x op b.x; \
    a.y op b.y; \
    a.z op b.z; \
    a.w op b.w; \
    return a; \
} \
FK_HOST_DEVICE_CNST modificable_type ## 1& operator op(modificable_type ## 1 & a, const input_type& s) { \
    a.x op s; \
    return a; \
} \
FK_HOST_DEVICE_CNST modificable_type ## 2& operator op(modificable_type ## 2 & a, const input_type& s) { \
    a.x op s; \
    a.y op s; \
    return a; \
} \
FK_HOST_DEVICE_CNST modificable_type ## 3& operator op(modificable_type ## 3 & a, const input_type& s) { \
    a.x op s; \
    a.y op s; \
    a.z op s; \
    return a; \
} \
FK_HOST_DEVICE_CNST modificable_type ## 4& operator op(modificable_type ## 4 & a, const input_type& s) { \
    a.x op s; \
    a.y op s; \
    a.z op s; \
    a.w op s; \
    return a; \
}

VEC_COMPOUND_OP(-=, char, char)
VEC_COMPOUND_OP(-=, short, short)
VEC_COMPOUND_OP(-=, int, int)
VEC_COMPOUND_OP(-=, float, float)
VEC_COMPOUND_OP(-=, double, double)
VEC_COMPOUND_OP(-=, uchar, uchar)
VEC_COMPOUND_OP(-=, char, uchar)
VEC_COMPOUND_OP(-=, ushort, uchar)
VEC_COMPOUND_OP(-=, short, uchar)
VEC_COMPOUND_OP(-=, int, uchar)
VEC_COMPOUND_OP(-=, uint, uchar)
VEC_COMPOUND_OP(-=, float, uchar)
VEC_COMPOUND_OP(-=, double, uchar)
VEC_COMPOUND_OP(-=, uint, uint)

VEC_COMPOUND_OP(+=, char, char)
VEC_COMPOUND_OP(+=, short, short)
VEC_COMPOUND_OP(+=, int, int)
VEC_COMPOUND_OP(+=, float, float)
VEC_COMPOUND_OP(+=, double, double)
VEC_COMPOUND_OP(+=, uchar, uchar)
VEC_COMPOUND_OP(+=, char, uchar)
VEC_COMPOUND_OP(+=, ushort, uchar)
VEC_COMPOUND_OP(+=, short, uchar)
VEC_COMPOUND_OP(+=, int, uchar)
VEC_COMPOUND_OP(+=, uint, uchar)
VEC_COMPOUND_OP(+=, float, uchar)
VEC_COMPOUND_OP(+=, double, uchar)
VEC_COMPOUND_OP(+=, uint, uint)

VEC_COMPOUND_OP(*=, char, char)
VEC_COMPOUND_OP(*=, short, short)
VEC_COMPOUND_OP(*=, int, int)
VEC_COMPOUND_OP(*=, float, float)
VEC_COMPOUND_OP(*=, double, double)
VEC_COMPOUND_OP(*=, uchar, uchar)
VEC_COMPOUND_OP(*=, char, uchar)
VEC_COMPOUND_OP(*=, ushort, uchar)
VEC_COMPOUND_OP(*=, short, uchar)
VEC_COMPOUND_OP(*=, int, uchar)
VEC_COMPOUND_OP(*=, uint, uchar)
VEC_COMPOUND_OP(*=, float, uchar)
VEC_COMPOUND_OP(*=, double, uchar)
VEC_COMPOUND_OP(*=, uint, uint)

VEC_COMPOUND_OP(/=, char, char)
VEC_COMPOUND_OP(/=, short, short)
VEC_COMPOUND_OP(/=, int, int)
VEC_COMPOUND_OP(/=, float, float)
VEC_COMPOUND_OP(/=, double, double)
VEC_COMPOUND_OP(/=, uchar, uchar)
VEC_COMPOUND_OP(/=, char, uchar)
VEC_COMPOUND_OP(/=, ushort, uchar)
VEC_COMPOUND_OP(/=, short, uchar)
VEC_COMPOUND_OP(/=, int, uchar)
VEC_COMPOUND_OP(/=, uint, uchar)
VEC_COMPOUND_OP(/=, float, uchar)
VEC_COMPOUND_OP(/=, double, uchar)
VEC_COMPOUND_OP(/=, uint, uint)

#undef VEC_COMPOUND_OP

// binary operators (vec & vec)
#define VEC_BINARY_OP_DIFF_TYPES(op, input_type1, input_type2) \
FK_HOST_DEVICE_CNST auto operator op(const input_type1 ## 1 & a, const input_type2 ## 1 & b) \
{ \
    using OutputType = typename fk::VectorType<decltype(std::declval<input_type1>() op std::declval<input_type2>()), 1>::type_v; \
    return OutputType{a.x op b.x}; \
} \
FK_HOST_DEVICE_CNST auto operator op(const input_type1 ## 2 & a, const input_type2 ## 2 & b) \
{ \
    using OutputType = fk::VectorType_t<decltype(std::declval<input_type1>() op std::declval<input_type2>()), 2>; \
    return OutputType{a.x op b.x, a.y op b.y}; \
} \
FK_HOST_DEVICE_CNST auto operator op(const input_type1 ## 3 & a, const input_type2 ## 3 & b) \
{ \
    using OutputType = fk::VectorType_t<decltype(std::declval<input_type1>() op std::declval<input_type2>()), 3>; \
    return OutputType{a.x op b.x, a.y op b.y, a.z op b.z}; \
} \
FK_HOST_DEVICE_CNST auto operator op(const input_type1 ## 4 & a, const input_type2 ## 4 & b) \
{ \
    using OutputType = fk::VectorType_t<decltype(std::declval<input_type1>() op std::declval<input_type2>()), 4>; \
    return OutputType{a.x op b.x, a.y op b.y, a.z op b.z, a.w op b.w}; \
}

#define VEC_BINARY_OP_DIFF_TYPES_OP_FIRST_FLOATING(op, input_type1) \
VEC_BINARY_OP_DIFF_TYPES(op, input_type1, float) \
VEC_BINARY_OP_DIFF_TYPES(op, input_type1, double)

#define VEC_BINARY_OP_DIFF_TYPES_OP_FIRST_INTEGERS(op, input_type1) \
VEC_BINARY_OP_DIFF_TYPES(op, input_type1, char) \
VEC_BINARY_OP_DIFF_TYPES(op, input_type1, uchar) \
VEC_BINARY_OP_DIFF_TYPES(op, input_type1, short) \
VEC_BINARY_OP_DIFF_TYPES(op, input_type1, ushort) \
VEC_BINARY_OP_DIFF_TYPES(op, input_type1, int) \
VEC_BINARY_OP_DIFF_TYPES(op, input_type1, uint) \
VEC_BINARY_OP_DIFF_TYPES(op, input_type1, long) \
VEC_BINARY_OP_DIFF_TYPES(op, input_type1, ulong) \
VEC_BINARY_OP_DIFF_TYPES(op, input_type1, longlong) \
VEC_BINARY_OP_DIFF_TYPES(op, input_type1, ulonglong)

#define VEC_BINARY_OP_DIFF_TYPES_OP_FLOATING_ALL(op) \
VEC_BINARY_OP_DIFF_TYPES_OP_FIRST_INTEGERS(op, float) \
VEC_BINARY_OP_DIFF_TYPES_OP_FIRST_FLOATING(op, float) \
VEC_BINARY_OP_DIFF_TYPES_OP_FIRST_INTEGERS(op, double) \
VEC_BINARY_OP_DIFF_TYPES_OP_FIRST_FLOATING(op, double)

#define VEC_BINARY_OP_DIFF_TYPES_OP_INTEGERS_INTEGERS(op) \
VEC_BINARY_OP_DIFF_TYPES_OP_FIRST_INTEGERS(op, char) \
VEC_BINARY_OP_DIFF_TYPES_OP_FIRST_INTEGERS(op, uchar) \
VEC_BINARY_OP_DIFF_TYPES_OP_FIRST_INTEGERS(op, short) \
VEC_BINARY_OP_DIFF_TYPES_OP_FIRST_INTEGERS(op, ushort) \
VEC_BINARY_OP_DIFF_TYPES_OP_FIRST_INTEGERS(op, int) \
VEC_BINARY_OP_DIFF_TYPES_OP_FIRST_INTEGERS(op, uint) \
VEC_BINARY_OP_DIFF_TYPES_OP_FIRST_INTEGERS(op, long) \
VEC_BINARY_OP_DIFF_TYPES_OP_FIRST_INTEGERS(op, ulong) \
VEC_BINARY_OP_DIFF_TYPES_OP_FIRST_INTEGERS(op, longlong) \
VEC_BINARY_OP_DIFF_TYPES_OP_FIRST_INTEGERS(op, ulonglong)

#define VEC_BINARY_OP_DIFF_TYPES_OP_INTEGERS_FLOATING(op) \
VEC_BINARY_OP_DIFF_TYPES_OP_FIRST_FLOATING(op, char) \
VEC_BINARY_OP_DIFF_TYPES_OP_FIRST_FLOATING(op, uchar) \
VEC_BINARY_OP_DIFF_TYPES_OP_FIRST_FLOATING(op, short) \
VEC_BINARY_OP_DIFF_TYPES_OP_FIRST_FLOATING(op, ushort) \
VEC_BINARY_OP_DIFF_TYPES_OP_FIRST_FLOATING(op, int) \
VEC_BINARY_OP_DIFF_TYPES_OP_FIRST_FLOATING(op, uint) \
VEC_BINARY_OP_DIFF_TYPES_OP_FIRST_FLOATING(op, long) \
VEC_BINARY_OP_DIFF_TYPES_OP_FIRST_FLOATING(op, ulong) \
VEC_BINARY_OP_DIFF_TYPES_OP_FIRST_FLOATING(op, longlong) \
VEC_BINARY_OP_DIFF_TYPES_OP_FIRST_FLOATING(op, ulonglong)

#define VEC_BINARY_OP_DIFF_TYPES_OP(op) \
VEC_BINARY_OP_DIFF_TYPES_OP_INTEGERS_INTEGERS(op) \
VEC_BINARY_OP_DIFF_TYPES_OP_INTEGERS_FLOATING(op) \
VEC_BINARY_OP_DIFF_TYPES_OP_FLOATING_ALL(op)

VEC_BINARY_OP_DIFF_TYPES_OP(+)
VEC_BINARY_OP_DIFF_TYPES_OP(-)
VEC_BINARY_OP_DIFF_TYPES_OP(*)
VEC_BINARY_OP_DIFF_TYPES_OP(/)
VEC_BINARY_OP_DIFF_TYPES_OP(==)
VEC_BINARY_OP_DIFF_TYPES_OP(!=)
VEC_BINARY_OP_DIFF_TYPES_OP(>)
VEC_BINARY_OP_DIFF_TYPES_OP(<)
VEC_BINARY_OP_DIFF_TYPES_OP(>=)
VEC_BINARY_OP_DIFF_TYPES_OP(<=)
VEC_BINARY_OP_DIFF_TYPES_OP(&&)
VEC_BINARY_OP_DIFF_TYPES_OP(||)

VEC_BINARY_OP_DIFF_TYPES_OP_INTEGERS_INTEGERS(&)
VEC_BINARY_OP_DIFF_TYPES_OP_INTEGERS_INTEGERS(|)
VEC_BINARY_OP_DIFF_TYPES_OP_INTEGERS_INTEGERS(^)

#undef VEC_BINARY_OP_DIFF_TYPES_OP
#undef VEC_BINARY_OP_DIFF_TYPES_OP_INTEGERS_FLOATING
#undef VEC_BINARY_OP_DIFF_TYPES_OP_INTEGERS_INTEGERS
#undef VEC_BINARY_OP_DIFF_TYPES_OP_FLOATING_ALL
#undef VEC_BINARY_OP_DIFF_TYPES_OP_FIRST_INTEGERS
#undef VEC_BINARY_OP_DIFF_TYPES_OP_FIRST_FLOATING
#undef VEC_BINARY_OP_DIFF_TYPES

// binary operators (vec & scalar)
#define SCALAR_BINARY_OP(op, input_type, scalar_type) \
FK_HOST_DEVICE_CNST auto operator op(const input_type ## 1 & a, const scalar_type& s) \
{ \
    using OutputType = typename fk::VectorType<decltype(std::declval<input_type>() op std::declval<scalar_type>()), 1>::type_v; \
    return OutputType{a.x op s}; \
} \
FK_HOST_DEVICE_CNST auto operator op(const scalar_type& s, const input_type ## 1 & b) \
{ \
    using OutputType = typename fk::VectorType<decltype(std::declval<scalar_type>() op std::declval<input_type>()), 1>::type_v; \
    return OutputType{s op b.x}; \
} \
FK_HOST_DEVICE_CNST auto operator op(const input_type ## 2 & a, const scalar_type& s) \
{ \
    using OutputType = typename fk::VectorType_t<decltype(std::declval<input_type>() op std::declval<scalar_type>()), 2>; \
    return OutputType{a.x op s, a.y op s}; \
} \
FK_HOST_DEVICE_CNST auto operator op(const scalar_type& s, const input_type ## 2 & b) \
{ \
    using OutputType = typename fk::VectorType_t<decltype(std::declval<scalar_type>() op std::declval<input_type>()), 2>; \
    return OutputType{s op b.x, s op b.y}; \
} \
FK_HOST_DEVICE_CNST auto operator op(const input_type ## 3 & a, const scalar_type& s) \
{ \
    using OutputType = typename fk::VectorType_t<decltype(std::declval<input_type>() op std::declval<scalar_type>()), 3>; \
    return OutputType{a.x op s, a.y op s, a.z op s}; \
} \
FK_HOST_DEVICE_CNST auto operator op(const scalar_type& s, const input_type ## 3 & b) \
{ \
    using OutputType = typename fk::VectorType_t<decltype(std::declval<scalar_type>() op std::declval<input_type>()), 3>; \
    return OutputType{s op b.x, s op b.y, s op b.z}; \
} \
FK_HOST_DEVICE_CNST auto operator op(const input_type ## 4 & a, const scalar_type& s) \
{ \
    using OutputType = typename fk::VectorType_t<decltype(std::declval<input_type>() op std::declval<scalar_type>()), 4>; \
    return OutputType{a.x op s, a.y op s, a.z op s, a.w op s}; \
} \
FK_HOST_DEVICE_CNST auto operator op(const scalar_type& s, const input_type ## 4 & b) \
{ \
    using OutputType = typename fk::VectorType_t<decltype(std::declval<scalar_type>() op std::declval<input_type>()), 4>; \
    return OutputType{s op b.x, s op b.y, s op b.z, s op b.w}; \
}

#define SCALAR_BINARY_OP_FIRST_FLOATING(op, input_type1) \
SCALAR_BINARY_OP(op, input_type1, float) \
SCALAR_BINARY_OP(op, input_type1, double)

#define SCALAR_BINARY_OP_FIRST_INTEGERS(op, input_type1) \
SCALAR_BINARY_OP(op, input_type1, char) \
SCALAR_BINARY_OP(op, input_type1, uchar) \
SCALAR_BINARY_OP(op, input_type1, short) \
SCALAR_BINARY_OP(op, input_type1, ushort) \
SCALAR_BINARY_OP(op, input_type1, int) \
SCALAR_BINARY_OP(op, input_type1, uint) \
SCALAR_BINARY_OP(op, input_type1, long) \
SCALAR_BINARY_OP(op, input_type1, ulong) \
SCALAR_BINARY_OP(op, input_type1, longlong) \
SCALAR_BINARY_OP(op, input_type1, ulonglong)

#define SCALAR_BINARY_OP_FLOATING_ALL(op) \
SCALAR_BINARY_OP_FIRST_FLOATING(op, float) \
SCALAR_BINARY_OP_FIRST_FLOATING(op, double) \
SCALAR_BINARY_OP_FIRST_INTEGERS(op, float) \
SCALAR_BINARY_OP_FIRST_INTEGERS(op, double)

#define SCALAR_BINARY_OP_INTEGERS_INTEGERS(op) \
SCALAR_BINARY_OP_FIRST_INTEGERS(op, char) \
SCALAR_BINARY_OP_FIRST_INTEGERS(op, uchar) \
SCALAR_BINARY_OP_FIRST_INTEGERS(op, short) \
SCALAR_BINARY_OP_FIRST_INTEGERS(op, ushort) \
SCALAR_BINARY_OP_FIRST_INTEGERS(op, int) \
SCALAR_BINARY_OP_FIRST_INTEGERS(op, uint) \
SCALAR_BINARY_OP_FIRST_INTEGERS(op, long) \
SCALAR_BINARY_OP_FIRST_INTEGERS(op, ulong) \
SCALAR_BINARY_OP_FIRST_INTEGERS(op, longlong) \
SCALAR_BINARY_OP_FIRST_INTEGERS(op, ulonglong)

#define SCALAR_BINARY_OP_INTEGERS_FLOATING(op) \
SCALAR_BINARY_OP_FIRST_FLOATING(op, char) \
SCALAR_BINARY_OP_FIRST_FLOATING(op, uchar) \
SCALAR_BINARY_OP_FIRST_FLOATING(op, short) \
SCALAR_BINARY_OP_FIRST_FLOATING(op, ushort) \
SCALAR_BINARY_OP_FIRST_FLOATING(op, int) \
SCALAR_BINARY_OP_FIRST_FLOATING(op, uint) \
SCALAR_BINARY_OP_FIRST_FLOATING(op, long) \
SCALAR_BINARY_OP_FIRST_FLOATING(op, ulong) \
SCALAR_BINARY_OP_FIRST_FLOATING(op, longlong) \
SCALAR_BINARY_OP_FIRST_FLOATING(op, ulonglong)

#define SCALAR_BINARY_OP_ALL(op) \
SCALAR_BINARY_OP_FLOATING_ALL(op) \
SCALAR_BINARY_OP_INTEGERS_INTEGERS(op) \
SCALAR_BINARY_OP_INTEGERS_FLOATING(op)

SCALAR_BINARY_OP_ALL(+)
SCALAR_BINARY_OP_ALL(-)
SCALAR_BINARY_OP_ALL(*)
SCALAR_BINARY_OP_ALL(/)
SCALAR_BINARY_OP_ALL(==)
SCALAR_BINARY_OP_ALL(!=)
SCALAR_BINARY_OP_ALL(<=)
SCALAR_BINARY_OP_ALL(>=)
SCALAR_BINARY_OP_ALL(<)
SCALAR_BINARY_OP_ALL(>)
SCALAR_BINARY_OP_ALL(&&)
SCALAR_BINARY_OP_ALL(||)

SCALAR_BINARY_OP_INTEGERS_INTEGERS(&)
SCALAR_BINARY_OP_INTEGERS_INTEGERS(|)
SCALAR_BINARY_OP_INTEGERS_INTEGERS(^)

#undef SCALAR_BINARY_OP_ALL
#undef SCALAR_BINARY_OP_INTEGERS_FLOATING
#undef SCALAR_BINARY_OP_INTEGERS_INTEGERS
#undef SCALAR_BINARY_OP_FLOATING_ALL
#undef SCALAR_BINARY_OP_FIRST_INTEGERS
#undef SCALAR_BINARY_OP_FIRST_FLOATING
#undef SCALAR_BINARY_OP
#else
#define VEC_UNARY_UNIVERSAL(op) \
template <typename T> \
FK_HOST_DEVICE_CNST auto operator op(const T& a) -> \
    std::enable_if_t<fk::CanUnary<T>::value, \
                     fk::VectorType_t<decltype(op std::declval<fk::VBase<T>>()), fk::cn<T>>> { \
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

#define VEC_COMPOUND_ARITHMETICAL(op) \
template <typename I1, typename I2> \
FK_HOST_DEVICE_CNST auto operator op(I1& a, const I2& b) \
    -> std::enable_if_t<fk::CanCompound<I1, I2>::value, I1> { \
    if constexpr (fk::IsCudaVector<I2>::value) { \
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

VEC_COMPOUND_ARITHMETICAL(-=)
VEC_COMPOUND_ARITHMETICAL(+=)
VEC_COMPOUND_ARITHMETICAL(*=)
VEC_COMPOUND_ARITHMETICAL(/=)

#undef VEC_COMPOUND_ARITHMETICAL

#define VEC_COMPOUND_LOGICAL(op) \
template <typename I1, typename I2> \
FK_HOST_DEVICE_CNST auto operator op(I1& a, const I2& b) \
    -> std::enable_if_t<fk::CanCompoundLogical<I1, I2>::value, I1> { \
    if constexpr (fk::IsCudaVector<I2>::value) { \
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

VEC_COMPOUND_LOGICAL(&=)
VEC_COMPOUND_LOGICAL(|=)

#undef VEC_COMPOUND_LOGICAL

// We don't need to check for I2 being a vector type, because the enable_if condition ensures it is a cuda vector if the two previous conditions are false
#define VEC_BINARY(op) \
template <typename I1, typename I2> \
FK_HOST_DEVICE_CNST auto operator op(const I1& a, const I2& b) \
    -> std::enable_if_t<fk::CanBinary<I1, I2>::value, \
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

VEC_BINARY(+)
VEC_BINARY(-)
VEC_BINARY(*)
VEC_BINARY(/)
VEC_BINARY(==)
VEC_BINARY(!=)
VEC_BINARY(>)
VEC_BINARY(<)
VEC_BINARY(>=)
VEC_BINARY(<=)
VEC_BINARY(&&)
VEC_BINARY(||)

#undef VEC_BINARY

#define VEC_BINARY_BITWISE(op) \
template <typename I1, typename I2> \
FK_HOST_DEVICE_CNST auto operator op(const I1& a, const I2& b) \
    -> std::enable_if_t<fk::CanBinaryBitwise<I1, I2>::value, \
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

VEC_BINARY_BITWISE(&)
VEC_BINARY_BITWISE(|)
VEC_BINARY_BITWISE(^)
#undef VEC_BINARY_BITWISE
#endif // VS2017_COMPILER

namespace fk {
    namespace internal {
        template <typename TargetT, typename SourceT, size_t... Idx>
        FK_HOST_DEVICE_CNST TargetT v_static_cast_helper(const SourceT& source, const std::index_sequence<Idx...>&) {
            return make_<TargetT>(static_cast<VBase<TargetT>>(vectorAt<Idx>(source))...);
        }

        template <typename T, size_t... Idx>
        FK_HOST_DEVICE_CNST auto v_sum_helper(const T& vectorValue, const std::index_sequence<Idx...>&) {
            return (vectorAt<Idx>(vectorValue) + ...);
        }
    } // namespace fk::internal

    template <typename TargetT, typename SourceT>
    FK_HOST_DEVICE_CNST TargetT v_static_cast(const SourceT& source) {
        using namespace fk;
        if constexpr (std::is_same_v<TargetT, SourceT>) {
            return source;
        } else if constexpr (validCUDAVec<SourceT>) {
            static_assert(cn<TargetT> == cn<SourceT>, "Can not cast to a type with different number of channels");
            return internal::v_static_cast_helper<TargetT>(source, std::make_index_sequence<cn<SourceT>>{});
        } else {
            static_assert(!validCUDAVec<TargetT> || (cn<TargetT> == 1),
                "Can not convert a fundamental type to a vetor type with more than one channel");
            if constexpr (validCUDAVec<TargetT>) {
                return make_<TargetT>(static_cast<VBase<TargetT>>(source));
            } else {
                return static_cast<TargetT>(source);
            }
        }
    }

    template <typename T>
    FK_HOST_DEVICE_CNST auto v_sum(const T& vectorValue) {
        return internal::v_sum_helper(vectorValue, std::make_index_sequence<cn<T>>{});
    }
} // namespace fk
#endif
