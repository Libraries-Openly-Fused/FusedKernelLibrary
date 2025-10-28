#ifndef UTEST_COMMON_H
#define UTEST_COMMON_H

#include <fused_kernel/algorithms/basic_ops/cast.h>
#include <fused_kernel/algorithms/image_processing/saturate.h>
#include <fused_kernel/core/utils/vector_utils.h>
#include <fused_kernel/core/utils/type_lists.h>
#include <fused_kernel/core/utils/type_to_string.h>
#include <fused_kernel/core/utils/vlimits.h>
#include <tests/operation_test_utils.h>

template <typename InputType, typename OutputType> constexpr OutputType expectedMinValue() {
    if constexpr (cxp::cmp_less_equal::f(fk::minValue<fk::VBase<InputType>>, fk::minValue<fk::VBase<OutputType>>)) {
        return fk::minValue<OutputType>;
    } else {
        return fk::Cast<InputType, OutputType>::exec(fk::minValue<InputType>);
    }
}

template <typename T> constexpr T halfPositiveRange() { return fk::make_set<T>(fk::maxValue<fk::VBase<T>> / 2); }

template <typename OutputType, typename InputType> constexpr OutputType expectedPositiveValue(const InputType &input) {
    if (cxp::cmp_greater::f(fk::get<0>(input), fk::maxValue<fk::VBase<OutputType>>)) {
        return fk::maxValue<OutputType>;
    } else {
        return fk::Cast<InputType, OutputType>::exec(input);
    }
}

template <typename InputType, typename OutputType> void addOneTest() {
    if constexpr (std::is_floating_point_v<InputType> && std::is_integral_v<OutputType> && std::is_signed_v<OutputType>) {
        constexpr OutputType expectedMinVal = expectedMinValue<InputType, OutputType>();

        constexpr OutputType expectedMaxVal = expectedPositiveValue<OutputType>(fk::maxValue<InputType>);

        constexpr OutputType expectedHalfMaxValue = expectedPositiveValue<OutputType>(halfPositiveRange<InputType>());

        constexpr std::array<InputType, 8> inputVals{ fk::minValue<InputType>, halfPositiveRange<InputType>(),
                                                      fk::maxValue<InputType>, static_cast<InputType>(0.5),
                                                      static_cast<InputType>(1.5),
                                                      static_cast<InputType>(2.5),
                                                      static_cast<InputType>(-1.5),
                                                      static_cast<InputType>(-2.5) };
        constexpr std::array<OutputType, 8> outputVals{ expectedMinVal, expectedHalfMaxValue, expectedMaxVal,
                                                        static_cast<OutputType>(0),
                                                        static_cast<OutputType>(2),
                                                        static_cast<OutputType>(2),
                                                        static_cast<OutputType>(-2),
                                                        static_cast<OutputType>(-2) };

        TestCaseBuilder<fk::SaturateCast<InputType, OutputType>>::addTest(testCases, inputVals, outputVals);
    } else if constexpr (std::is_floating_point_v<InputType> && std::is_integral_v<OutputType> && !std::is_signed_v<OutputType>) {
        constexpr OutputType expectedMinVal = expectedMinValue<InputType, OutputType>();

        constexpr OutputType expectedMaxVal = expectedPositiveValue<OutputType>(fk::maxValue<InputType>);

        constexpr OutputType expectedHalfMaxValue = expectedPositiveValue<OutputType>(halfPositiveRange<InputType>());

        constexpr std::array<InputType, 6> inputVals{ fk::minValue<InputType>, halfPositiveRange<InputType>(), fk::maxValue<InputType>,
                                                      static_cast<InputType>(0.5),
                                                      static_cast<InputType>(1.5),
                                                      static_cast<InputType>(2.5) };
        constexpr std::array<OutputType, 6> outputVals{ expectedMinVal, expectedHalfMaxValue, expectedMaxVal,
                                                        static_cast<OutputType>(0),
                                                        static_cast<OutputType>(2),
                                                        static_cast<OutputType>(2) };

        TestCaseBuilder<fk::SaturateCast<InputType, OutputType>>::addTest(testCases, inputVals, outputVals);
    } else {
        constexpr OutputType expectedMinVal = expectedMinValue<InputType, OutputType>();

        constexpr OutputType expectedMaxVal = expectedPositiveValue<OutputType>(fk::maxValue<InputType>);

        constexpr OutputType expectedHalfMaxValue = expectedPositiveValue<OutputType>(halfPositiveRange<InputType>());

        constexpr std::array<InputType, 3> inputVals{ fk::minValue<InputType>, halfPositiveRange<InputType>(),
                                                     fk::maxValue<InputType> };
        constexpr std::array<OutputType, 3> outputVals{ expectedMinVal, expectedHalfMaxValue, expectedMaxVal };

        TestCaseBuilder<fk::SaturateCast<InputType, OutputType>>::addTest(testCases, inputVals, outputVals);
    }
}

template <typename BaseInput, typename BaseOutput> void addOneTestAllChannels() {
    // Base Type
    addOneTest<BaseInput, BaseOutput>();

    // Vector of 1
    using Input1 = typename fk::VectorType<BaseInput, 1>::type_v;
    using Output1 = typename fk::VectorType<BaseOutput, 1>::type_v;
    addOneTest<Input1, Output1>();

    // Vector of 2
    using Input2 = fk::VectorType_t<BaseInput, 2>;
    using Output2 = fk::VectorType_t<BaseOutput, 2>;
    addOneTest<Input2, Output2>();

    // Vector of 3
    using Input3 = fk::VectorType_t<BaseInput, 3>;
    using Output3 = fk::VectorType_t<BaseOutput, 3>;
    addOneTest<Input3, Output3>();

    // Vector of 4
    using Input4 = fk::VectorType_t<BaseInput, 4>;
    using Output4 = fk::VectorType_t<BaseOutput, 4>;
    addOneTest<Input4, Output4>();
}

template <typename TypeList_, typename Type, size_t... Idx>
void addAllTestsFor_helper(const std::index_sequence<Idx...> &) {
    static_assert(fk::validCUDAVec<Type> || std::is_fundamental_v<Type>,
                  "Type must be either a cuda vector or a fundamental type.");
    static_assert(fk::isTypeList<TypeList_>, "TypeList_ must be a valid TypeList.");
    // For each type in TypeList_, add tests with Type
    (addOneTestAllChannels<fk::TypeAt_t<Idx, TypeList_>, Type>(), ...);
}

template <typename TypeList_, size_t... Idx> void addAllTestsFor(const std::index_sequence<Idx...> &) {
    // For each type in TypeList_, add tests with each type in TypeList_
    (addAllTestsFor_helper<TypeList_, fk::TypeAt_t<Idx, TypeList_>>(std::make_index_sequence<TypeList_::size>{}), ...);
}

template <typename OutputTypeList, typename InputType, size_t... Idx>
void addAllOutputTestsForInput(const std::index_sequence<Idx...> &) {
    // For each OutputType in OutputTypeList, add tests with fixed InputType
    (addOneTestAllChannels<InputType, fk::TypeAt_t<Idx, OutputTypeList>>(), ...);
}

#endif // ! UTEST_COMMON_H