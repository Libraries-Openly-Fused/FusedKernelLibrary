/* Copyright 2023-2026 Oscar Amoros Huguet

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
#include <fused_kernel/core/utils/utils.h>

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

    // Reduced precision types: hand-written like char/schar because their vector aliases do not
    // follow the BaseType##N token-pasting pattern for fp8/fp4.
    template <>
    struct VectorType<fp16, 1> { using type = fp16; using type_v = fp16_1; };
    template <>
    struct VectorType<fp16, 2> { using type = fp16_2; using type_v = type; };
    template <>
    struct VectorType<fp16, 3> { using type = fp16_3; using type_v = type; };
    template <>
    struct VectorType<fp16, 4> { using type = fp16_4; using type_v = type; };

    template <>
    struct VectorType<bf16, 1> { using type = bf16; using type_v = bf16_1; };
    template <>
    struct VectorType<bf16, 2> { using type = bf16_2; using type_v = type; };
    template <>
    struct VectorType<bf16, 3> { using type = bf16_3; using type_v = type; };
    template <>
    struct VectorType<bf16, 4> { using type = bf16_4; using type_v = type; };

    template <>
    struct VectorType<fp8_e4m3, 1> { using type = fp8_e4m3; using type_v = fp8_e4m3_1; };
    template <>
    struct VectorType<fp8_e4m3, 2> { using type = fp8_e4m3_2; using type_v = type; };
    template <>
    struct VectorType<fp8_e4m3, 3> { using type = fp8_e4m3_3; using type_v = type; };
    template <>
    struct VectorType<fp8_e4m3, 4> { using type = fp8_e4m3_4; using type_v = type; };

    template <>
    struct VectorType<fp8_e5m2, 1> { using type = fp8_e5m2; using type_v = fp8_e5m2_1; };
    template <>
    struct VectorType<fp8_e5m2, 2> { using type = fp8_e5m2_2; using type_v = type; };
    template <>
    struct VectorType<fp8_e5m2, 3> { using type = fp8_e5m2_3; using type_v = type; };
    template <>
    struct VectorType<fp8_e5m2, 4> { using type = fp8_e5m2_4; using type_v = type; };

    template <>
    struct VectorType<fp4_e2m1, 1> { using type = fp4_e2m1; using type_v = fp4_e2m1_1; };
    template <>
    struct VectorType<fp4_e2m1, 2> { using type = fp4_e2m1_2; using type_v = type; };
    template <>
    struct VectorType<fp4_e2m1, 3> { using type = fp4_e2m1_3; using type_v = type; };
    template <>
    struct VectorType<fp4_e2m1, 4> { using type = fp4_e2m1_4; using type_v = type; };

    template <typename BaseType, int Channels>
    using VectorType_t = typename VectorType<BaseType, Channels>::type;

    template <size_t CN> using bool_ = VectorType_t<bool, CN>;
    template <size_t CN> using uchar_ = VectorType_t<uchar, CN>;
    template <size_t CN> using char_ = VectorType_t<schar, CN>;
    template <size_t CN> using ushort_ = VectorType_t<ushort, CN>;
    template <size_t CN> using short_ = VectorType_t<short, CN>;
    template <size_t CN> using uint_ = VectorType_t<uint, CN>;
    template <size_t CN> using int_ = VectorType_t<int, CN>;
    template <size_t CN> using ulong_ = VectorType_t<ulong, CN>;
    template <size_t CN> using long_ = VectorType_t<long, CN>;
    template <size_t CN> using ulonglong_ = VectorType_t<ulonglong, CN>;
    template <size_t CN> using longlong_ = VectorType_t<longlong, CN>;
    template <size_t CN> using float_ = VectorType_t<float, CN>;
    template <size_t CN> using double_ = VectorType_t<double, CN>;
    template <size_t CN> using fp16_ = VectorType_t<fp16, CN>;
    template <size_t CN> using bf16_ = VectorType_t<bf16, CN>;
    template <size_t CN> using fp8_e4m3_ = VectorType_t<fp8_e4m3, CN>;
    template <size_t CN> using fp8_e5m2_ = VectorType_t<fp8_e5m2, CN>;
    template <size_t CN> using fp4_e2m1_ = VectorType_t<fp4_e2m1, CN>;

    template <uint CN>
    using VectorTypeList = TypeList<bool_<CN>, uchar_<CN>, char_<CN>, ushort_<CN>, short_<CN>, uint_<CN>, int_<CN>,
                                    ulong_<CN>, long_<CN>, ulonglong_<CN>, longlong_<CN>, float_<CN>, double_<CN>>;

    using FloatingTypes = TypeList<float, double>;
    using IntegralTypes = TypeList<uchar, char, schar, ushort, short, uint, int, ulong, long, ulonglong, longlong>;
    using IntegralBaseTypes = TypeList<uchar, schar, ushort, short, uint, int, ulong, long, ulonglong, longlong>;
    using StandardTypes = TypeListCat_t<TypeList<bool>, IntegralTypes, FloatingTypes>;
    using BaseTypes = TypeListCat_t<TypeList<bool>, IntegralBaseTypes, FloatingTypes>;
    using VOne = TypeList<bool1, uchar1, char1, ushort1, short1, uint1, int1, ulong1, long1, ulonglong1, longlong1, float1, double1>;
    using VTwo = VectorTypeList<2>;
    using VThree = VectorTypeList<3>;
    using VFour = VectorTypeList<4>;
    // Reduced precision vector lists are kept separate so the standard lists (and everything
    // positionally derived from them) stay untouched.
    using RFVOne = TypeList<fp16_1, bf16_1, fp8_e4m3_1, fp8_e5m2_1, fp4_e2m1_1>;
    using RFVTwo = TypeList<fp16_2, bf16_2, fp8_e4m3_2, fp8_e5m2_2, fp4_e2m1_2>;
    using RFVThree = TypeList<fp16_3, bf16_3, fp8_e4m3_3, fp8_e5m2_3, fp4_e2m1_3>;
    using RFVFour = TypeList<fp16_4, bf16_4, fp8_e4m3_4, fp8_e5m2_4, fp4_e2m1_4>;
    using VAll = TypeListCat_t<VOne, VTwo, VThree, VFour, RFVOne, RFVTwo, RFVThree, RFVFour>;

    template <typename T>
    constexpr bool validCUDAVec = one_of<T, VAll>::value;

    template <typename T>
    struct IsCudaVector : std::conditional_t<validCUDAVec<T>, std::true_type, std::false_type> {};

    template <typename T>
    FK_HOST_DEVICE_CNST int Channels() {
        if constexpr (one_of_v<T, VOne> || one_of_v<T, RFVOne> || !validCUDAVec<T>) {
            return 1;
        } else if constexpr (one_of_v<T, VTwo> || one_of_v<T, RFVTwo>) {
            return 2;
        } else if constexpr (one_of_v<T, VThree> || one_of_v<T, RFVThree>) {
            return 3;
        } else {
            static_assert(one_of_v<T, VFour> || one_of_v<T, RFVFour>,
                          "Type T must be a valid CUDA vector type (1, 2, 3, or 4 channels)");
            return 4;
        }
    }

    template <typename T>
    constexpr int cn = Channels<T>();

    template <typename V>
    struct VectorTraits;

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

#undef VECTOR_TRAITS

#define VECTOR_TRAITS_RF(BaseType, V1, V2, V3, V4) \
    template <> \
    struct VectorTraits<BaseType> { using base = BaseType; enum { bytes = sizeof(base) }; }; \
    template <> \
    struct VectorTraits<V1> { using base = BaseType; enum { bytes = sizeof(base) }; }; \
    template <> \
    struct VectorTraits<V2> { using base = BaseType; enum { bytes = sizeof(base) * 2 }; }; \
    template <> \
    struct VectorTraits<V3> { using base = BaseType; enum { bytes = sizeof(base) * 3 }; }; \
    template <> \
    struct VectorTraits<V4> { using base = BaseType; enum { bytes = sizeof(base) * 4 }; };

    VECTOR_TRAITS_RF(fp16, fp16_1, fp16_2, fp16_3, fp16_4)
    VECTOR_TRAITS_RF(bf16, bf16_1, bf16_2, bf16_3, bf16_4)
    VECTOR_TRAITS_RF(fp8_e4m3, fp8_e4m3_1, fp8_e4m3_2, fp8_e4m3_3, fp8_e4m3_4)
    VECTOR_TRAITS_RF(fp8_e5m2, fp8_e5m2_1, fp8_e5m2_2, fp8_e5m2_3, fp8_e5m2_4)
    VECTOR_TRAITS_RF(fp4_e2m1, fp4_e2m1_1, fp4_e2m1_2, fp4_e2m1_3, fp4_e2m1_4)

#undef VECTOR_TRAITS_RF

    template <typename T>
    using VBase = typename VectorTraits<T>::base;

    template <size_t Idx, typename VT>
    FK_HOST_DEVICE_CNST auto static_get(const VT& v) {
        static_assert(IsCudaVector<VT>::value, "Invalid type for static_get");
        static_assert((Idx < cn<VT>), "Index out of bounds.");
        if constexpr (Idx == 0) {
            return v.x;
        } else if constexpr (Idx == 1) {
            return v.y;
        } else if constexpr (Idx == 2) {
            return v.z;
        } else {
            return v.w;
        }
    }

    struct vector_at {
        template <typename VT>
        FK_HOST_DEVICE_FUSE auto f(const int& idx, const VT& v)
            -> std::enable_if_t<validScalar<VT>, VT> {
            return v;
        }
        template <typename VT>
        FK_HOST_DEVICE_FUSE auto f(const int& idx, const VT& v)
            -> std::enable_if_t<IsCudaVector<VT>::value && (cn<VT> == 1), VBase<VT>> {
            return v.x;
        }
        template <typename VT>
        FK_HOST_DEVICE_FUSE auto f(const int& idx, const VT& v)
            -> std::enable_if_t<IsCudaVector<VT>::value && (cn<VT> == 2), VBase<VT>> {
            if (idx == 0) {
                return v.x;
            } else {
                return v.y;
            }
        }
        template <typename VT>
        FK_HOST_DEVICE_FUSE auto f(const int& idx, const VT& v)
            -> std::enable_if_t<IsCudaVector<VT>::value && (cn<VT> == 3), VBase<VT>> {
            if (idx == 0) {
                return v.x;
            } else if (idx == 1) {
                return v.y;
            } else {
                return v.z;
            }
        }
        template <typename VT>
        FK_HOST_DEVICE_FUSE auto f(const int& idx, const VT& v)
            -> std::enable_if_t<IsCudaVector<VT>::value && (cn<VT> == 4), VBase<VT>> {
            if (idx == 0) {
                return v.x;
            } else if (idx == 1) {
                return v.y;
            } else if (idx == 2) {
                return v.z;
            } else {
                return v.w;
            }
        }
    };

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
    struct BothIntegrals<I1, I2, std::enable_if_t<std::is_integral_v<fk::VBase<I1>> && std::is_integral_v<fk::VBase<I2>>, void>> : public std::true_type {};

    template <typename I1, typename I2, typename = void>
    struct AreVVEqCN : public std::false_type {};

    template <typename I1, typename I2>
        struct AreVVEqCN<I1, I2, std::enable_if_t<fk::validCUDAVec<I1> && fk::validCUDAVec<I2>>>
        : public std::bool_constant<(fk::cn<I1> == fk::cn<I2>)> {};

    template <typename I1, typename I2, typename = void>
    struct AreSV : public std::false_type {};

    template <typename I1, typename I2>
    struct AreSV<I1, I2, std::enable_if_t<fk::validScalar<I1> && fk::validCUDAVec<I2>, void>> : public std::true_type {};

    template <typename I1, typename I2, typename = void>
    struct AreVS : public std::false_type {};

    template <typename I1, typename I2>
    struct AreVS<I1, I2, std::enable_if_t<fk::validCUDAVec<I1> && fk::validScalar<I2>, void>> : public std::true_type {};

    template <typename I1, typename I2, typename = void>
    struct AreSS : public std::false_type {};

    template <typename I1, typename I2>
    struct AreSS<I1, I2, std::enable_if_t<fk::validScalar<I1> && fk::validScalar<I2>, void>> : public std::true_type {};

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
    struct CanShift : public std::false_type {};

    template <typename I1, typename I2>
    struct CanShift<I1, I2,
        std::enable_if_t<(AreVVEqCN<I1, I2>::value || AreVS<I1, I2>::value)
                         && BothIntegrals<I1, I2>::value, void>> : public std::true_type {};

    // Closed form availability of the ELEMENT level compound operator (x op= y). Expressed as
    // a trait instead of probing the expression: an expression probe would re-enter the vector
    // operator templates and create a self dependent constraint. Fundamentals compound among
    // themselves and with the arithmetic reduced floats (through their implicit float
    // conversion); fp16/bf16 compound only with themselves; fp8/fp4 have no operators at all.
    template <typename B1, typename B2>
    constexpr bool hasCompoundBaseOp =
        (std::is_fundamental_v<B1> && std::is_fundamental_v<B2>) ||
        (std::is_fundamental_v<B1> && isArithmeticReducedFloat<B2>) ||
        (isArithmeticReducedFloat<B1> && std::is_same_v<B1, B2>);

    template <typename I1, typename I2, typename = void>
    struct CanCompound : public std::false_type {};

    // Vector op= vector: both bases are registered, so VBase is safe to evaluate on both sides.
    template <typename I1, typename I2>
    struct CanCompound<I1, I2,
        std::enable_if_t<AreVVEqCN<I1, I2>::value &&
                         hasCompoundBaseOp<VBase<I1>, VBase<I2>>, void>> : public std::true_type {};

    // Vector op= scalar: VBase<I2> must NOT be evaluated for plain fundamentals - types like
    // long double or wchar_t are fundamental (and compound with any fundamental base through
    // the built in operators) but have no VectorTraits specialization.
    template <typename I1, typename I2>
    struct CanCompound<I1, I2,
        std::enable_if_t<AreVS<I1, I2>::value &&
                         ((std::is_fundamental_v<I2> && std::is_fundamental_v<VBase<I1>>) ||
                          (isReducedFloat<I2> && hasCompoundBaseOp<VBase<I1>, I2>)), void>> : public std::true_type {};

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
// The Can* gates live in a default template argument (not in the return type): they must be
// checked BEFORE the decltype over the base types is substituted, otherwise resolving the
// element level operator for class type scalars (fp16/bf16) re-enters this same template and
// recurses infinitely.
#define VEC_UNARY_UNIVERSAL(op) \
template <typename T, typename = std::enable_if_t<fk::CanUnary<T>::value>> \
FK_HOST_DEVICE_CNST auto operator op(const T& a) -> \
    fk::VectorType_t<decltype(op std::declval<fk::VBase<T>>()), fk::cn<T>> { \
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
template <typename I1, typename I2, typename = std::enable_if_t<fk::CanBinary<I1, I2>::value>> \
FK_HOST_DEVICE_CNST auto operator op(const I1& a, const I2& b) \
    -> typename fk::VectorType<decltype(std::declval<fk::VBase<I1>>() op std::declval<fk::VBase<I2>>()), \
                               (fk::cn<I1> > fk::cn<I2> ? fk::cn<I1> : fk::cn<I2>)>::type_v { \
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
template <typename I1, typename I2, typename = std::enable_if_t<fk::CanBinaryBitwise<I1, I2>::value>> \
FK_HOST_DEVICE_CNST auto operator op(const I1& a, const I2& b) \
    -> typename fk::VectorType<decltype(std::declval<fk::VBase<I1>>() op std::declval<fk::VBase<I2>>()), \
                               (fk::cn<I1> > fk::cn<I2> ? fk::cn<I1> : fk::cn<I2>)>::type_v { \
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


template <typename I1, typename I2>
FK_HOST_DEVICE_CNST auto operator<<(const I1& a, const I2& b)
-> std::enable_if_t<fk::CanShift<I1, I2>::value, I1> {
    if constexpr (fk::validCUDAVec<I2>) {
        static_assert(fk::cn<I1> == fk::cn<I2>, "Vectors must have the same number of channels");
        if constexpr (fk::cn<I1> == 1) {
            if constexpr (fk::validCUDAVec<I1>) {
                return fk::make_<I1>(a.x << b.x);
            } else {
                return a << b.x;
            }
        } else if constexpr (fk::cn<I1> == 2) {
            return fk::make_<I1>(a.x << b.x, a.y << b.y);
        } else if constexpr (fk::cn<I1> == 3) {
            return fk::make_<I1>(a.x << b.x, a.y << b.y, a.z << b.z);
        } else {
            return fk::make_<I1>(a.x << b.x, a.y << b.y, a.z << b.z, a.w << b.w);
        }
    } else {
        if constexpr (fk::validCUDAVec<I1>) {
            if constexpr (fk::cn<I1> == 1) {
                return fk::make_<I1>(a.x << b);
            } else if constexpr (fk::cn<I1> == 2) {
                return fk::make_<I1>(a.x << b, a.y << b);
            } else if constexpr (fk::cn<I1> == 3) {
                return fk::make_<I1>(a.x << b, a.y << b, a.z << b);
            } else {
                return fk::make_<I1>(a.x << b, a.y << b, a.z << b, a.w << b);
            }
        } else {
            return a << b;
        }
    }
}

template <typename I1, typename I2>
FK_HOST_DEVICE_CNST auto operator>>(const I1& a, const I2& b)
-> std::enable_if_t<fk::CanShift<I1, I2>::value, I1> {
    if constexpr (fk::validCUDAVec<I2>) {
        static_assert(fk::cn<I1> == fk::cn<I2>, "Vectors must have the same number of channels");
        if constexpr (fk::cn<I1> == 1) {
            if constexpr (fk::validCUDAVec<I1>) {
                return fk::make_<I1>(a.x >> b.x);
            } else {
                return a >> b.x;
            }
        } else if constexpr (fk::cn<I1> == 2) {
            return fk::make_<I1>(a.x >> b.x, a.y >> b.y);
        } else if constexpr (fk::cn<I1> == 3) {
            return fk::make_<I1>(a.x >> b.x, a.y >> b.y, a.z >> b.z);
        } else {
            return fk::make_<I1>(a.x >> b.x, a.y >> b.y, a.z >> b.z, a.w >> b.w);
        }
    } else {
        if constexpr (fk::validCUDAVec<I1>) {
            if constexpr (fk::cn<I1> == 1) {
                return fk::make_<I1>(a.x >> b);
            } else if constexpr (fk::cn<I1> == 2) {
                return fk::make_<I1>(a.x >> b, a.y >> b);
            } else if constexpr (fk::cn<I1> == 3) {
                return fk::make_<I1>(a.x >> b, a.y >> b, a.z >> b);
            } else {
                return fk::make_<I1>(a.x >> b, a.y >> b, a.z >> b, a.w >> b);
            }
        } else {
            return a >> b;
        }
    }
}

#endif
