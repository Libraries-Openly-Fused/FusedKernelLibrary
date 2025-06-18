﻿/* Copyright 2025 Oscar Amoros Huguet
   Copyright 2025 Albert Andaluz
   Copyright 2025 Grup Mediapro S.L.U

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include <fused_kernel/algorithms/image_processing/saturate.h>
#include <fused_kernel/algorithms/basic_ops/cast.h>
#include <fused_kernel/core/utils/cuda_vector_utils.h>
#include <fused_kernel/core/utils/type_to_string.h>
#include <fused_kernel/core/utils/vlimits.h>
#include <tests/operation_test_utils.h>

inline std::string niceType(const std::string& input) {
    // Map "unsigned type" to specific type names
    static const std::unordered_map<std::string, std::string> unsignedTypeMap = {
        {"unsigned char", "uchar"},
        {"unsigned short", "ushort"},
        {"unsigned int", "uint"},
        {"unsigned long", "ulong"},
        {"unsigned longlong", "ulonglong"},
        {"__int64", "longlong"},
        {"unsigned __int64", "ulonglong"}
    };

    // Check if the input matches any key in the map
    auto it = unsignedTypeMap.find(input);
    if (it != unsignedTypeMap.end()) {
        return it->second; // Return the mapped type name
    }
    return input;
}

template <typename InputType, typename OutputType>
constexpr OutputType expectedMinValue() {
    if constexpr (fk::minValue<fk::VBase<InputType>> <= fk::minValue<fk::VBase<OutputType>>) {
        return fk::minValue<OutputType>;
    } else {
        return fk::Cast<InputType, OutputType>::exec(fk::minValue<InputType>);
    }
}

template <typename T>
constexpr T halfPositiveRange() {
    return fk::make_set<T>(fk::maxValue<fk::VBase<T>> / 2);
}

template <typename OutputType, typename InputType>
constexpr OutputType expectedPositiveValue(const InputType& input) {
    if (fk::VectorAt<0>(input) > fk::maxValue<fk::VBase<OutputType>>) {
        return fk::maxValue<OutputType>;
    } else {
        return fk::Cast<InputType, OutputType>::exec(input);
    }
}

template <typename InputType, typename OutputType>
void addOneTest() {
    // minValue<Input> <= minValue<Output> -> output{ fk::minValue<Output>, ... }
    // minValue<Input> > minValue<Output> -> output{ fk::Cast<Input, Output>::exec(fk::minValue<Input>), ... }
    constexpr OutputType expectedMinVal = expectedMinValue<InputType, OutputType>();

    // maxValue<Input> < maxValue<Output> -> output{ ... , fk::Cast<Input, Output>::exec(fk::maxValue<Input>) }
    // maxValue<Input> >= maxValue<Output> -> output{ ... , fk::maxValue<Output> }
    constexpr OutputType expectedMaxVal = expectedPositiveValue<OutputType>(fk::maxValue<InputType>);

    // halfPositiveRange<InputType>() < maxValue<Output> -> output{ ... , fk::Cast<Input, Output>::exec(fk::maxValue<Input>) }
    // halfPositiveRange<InputType>() >= maxValue<Output> -> output{ ... , fk::maxValue<Output> }
    constexpr OutputType expectedHalfMaxValue = expectedPositiveValue<OutputType>(halfPositiveRange<InputType>());

    constexpr std::array<InputType, 3> inputVals{ fk::minValue<InputType>, halfPositiveRange<InputType>(), fk::maxValue<InputType> };
    constexpr std::array<OutputType, 3> outputVals{ expectedMinVal, expectedHalfMaxValue, expectedMaxVal};
    
    const std::string testName = "Testfk::SaturateCast_" + niceType(fk::typeToString<InputType>()) + "_" + niceType(fk::typeToString<OutputType>());
    testCases[testName] = 
        TestCaseBuilder<fk::SaturateCast<InputType, OutputType>>::build(testName, inputVals, outputVals);
}

template <typename BaseInput, typename BaseOutput>
void addOneTestAllChannels() {
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
void addAllTestsFor_helper(const std::index_sequence<Idx...>&) {
    static_assert(fk::validCUDAVec<Type> || std::is_fundamental_v<Type>, "Type must be either a cuda vector or a fundamental type.");
    static_assert(fk::isTypeList<TypeList_>, "TypeList_ must be a valid TypeList.");
    // For each type in TypeList_, add tests with Type
    (addOneTestAllChannels<fk::TypeAt_t<Idx, TypeList_>, Type>(), ...);
}

template <typename TypeList_, size_t... Idx>
void addAllTestsFor(const std::index_sequence<Idx...>&) {
    // For each type in TypeList_, add tests with each type in TypeList_
    (addAllTestsFor_helper<TypeList_, fk::TypeAt_t<Idx, TypeList_>>(std::make_index_sequence<TypeList_::size>{}), ...);
}

START_ADDING_TESTS
using Fundamental = fk::RemoveType_t<0, fk::StandardTypes>;
addAllTestsFor<Fundamental>(std::make_index_sequence<Fundamental::size>());
STOP_ADDING_TESTS

// You can add more tests for other type combinations as needed.
int launch() {
    RUN_ALL_TESTS
};