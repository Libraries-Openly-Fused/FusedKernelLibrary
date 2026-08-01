/* Copyright 2023-2025 Oscar Amoros Huguet
   Copyright 2025 Grup Mediapro S.L.U (Oscar Amoros Huguet)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include <tests/main.h>

#include <benchmarks/fkBenchmarksCommon.h>
#include <benchmarks/twoExecutionsBenchmark.h>

#include <fused_kernel/core/core.h>
#include <fused_kernel/algorithms/image_processing/saturate.h>
#include <fused_kernel/algorithms/image_processing/color_conversion.h>
#include <fused_kernel/core/data/ptr_utils.h>

#include <iostream>
#include <fused_kernel/fused_kernel.h>
#include "tests/nvtx.h"

constexpr size_t NUM_EXPERIMENTS = 1;
constexpr size_t FIRST_VALUE = 1024;
constexpr size_t INCREMENT = 1024;
constexpr std::array<size_t, NUM_EXPERIMENTS> variableDimensionValues = arrayIndexSecuence<FIRST_VALUE, INCREMENT, NUM_EXPERIMENTS>;
constexpr char VARIABLE_DIMENSION_NAME[] = "Pixels per side";
constexpr std::string_view FIRST_LABEL = "Normal";
constexpr std::string_view SECOND_LABEL = "ThreadFusion";

template <typename T, size_t RESOLUTION>
bool testThreadFusionSameTypeIO(fk::Stream& stream) {
    std::stringstream error_s;
    bool passed = true;
    bool exception = false;

    constexpr size_t BATCH = RESOLUTION;

    constexpr uint NUM_ELEMS_X = (uint)RESOLUTION;
    constexpr uint NUM_ELEMS_Y = (uint)RESOLUTION;

    T val_init = fk::make_set<T>(2);

    try {
        fk::Ptr2D<T> d_input(NUM_ELEMS_Y, NUM_ELEMS_X);
        fk::setTo(val_init, d_input, stream);
        fk::Ptr2D<T> d_output_cvGS(NUM_ELEMS_Y, NUM_ELEMS_X);
        fk::Ptr2D<T> d_output_cvGS_ThreadFusion(NUM_ELEMS_Y, NUM_ELEMS_X);

        START_FIRST_BENCHMARK(fk::defaultParArch)
        // non fusion version
        const auto read = fk::PerThreadRead<fk::ND::_2D, T>::build(d_input);
        const auto write = fk::PerThreadWrite<fk::ND::_2D, T>::build(d_output_cvGS);
        fk::executeOperations<fk::TransformDPP<fk::defaultParArch, fk::TF::DISABLED>>(stream, read, write);
        STOP_FIRST_START_SECOND_BENCHMARK
        // fusion version
        const auto readTF = fk::PerThreadRead<fk::ND::_2D, T>::build(d_input);
        const auto writeTF = fk::PerThreadWrite<fk::ND::_2D, T>::build(d_output_cvGS_ThreadFusion);
        fk::executeOperations<fk::TransformDPP<fk::defaultParArch, fk::TF::ENABLED>>(stream, readTF, writeTF);
        STOP_SECOND_BENCHMARK

        // Verify results
        d_output_cvGS_ThreadFusion.download(stream);
        d_output_cvGS.download(stream);

        stream.sync();

        passed = compareAndCheck(d_output_cvGS_ThreadFusion, d_output_cvGS);
    } catch (const std::exception& e) {
        error_s << e.what();
        passed = false;
        exception = true;
    }

    if (!passed) {
        if (!exception) {
            std::stringstream ss;
            ss << "testThreadFusionTimes<" << fk::typeToString<T>();
            std::cout << ss.str() << "> failed!! RESULT ERROR: Some results do not match baseline." << std::endl;
        } else {
            std::stringstream ss;
            ss << "testThreadFusionTimes<" << fk::typeToString<T>();
            std::cout << ss.str() << "> failed!! EXCEPTION: " << error_s.str() << std::endl;
        }
    }
    return passed;
}

template <typename I, typename O, size_t RESOLUTION>
bool testThreadFusionDifferentTypeIO(fk::Stream& stream) {
    std::stringstream error_s;
    bool passed = true;
    bool exception = false;

    constexpr size_t BATCH = RESOLUTION;

    constexpr uint NUM_ELEMS_X = static_cast<uint>(RESOLUTION);
    constexpr uint NUM_ELEMS_Y = static_cast<uint>(RESOLUTION);

    constexpr I val_init = fk::make_set<I>(2);

    try {
        fk::Ptr2D<I> d_input(NUM_ELEMS_X, NUM_ELEMS_Y);
        fk::setTo(val_init, d_input, stream);
        fk::Ptr2D<O> d_output_cvGS(NUM_ELEMS_X, NUM_ELEMS_Y);
        fk::Ptr2D<O> d_output_cvGS_ThreadFusion(NUM_ELEMS_X, NUM_ELEMS_Y);

        START_FIRST_BENCHMARK(fk::defaultParArch)
        // non fusion version
        const auto read = fk::PerThreadRead<fk::ND::_2D, I>::build(d_input);
        const auto write = fk::PerThreadWrite<fk::ND::_2D, O>::build(d_output_cvGS);
        fk::executeOperations<fk::TransformDPP<fk::defaultParArch, fk::TF::DISABLED>>(stream, read, fk::SaturateCast<I, O>::build(), write);
        STOP_FIRST_START_SECOND_BENCHMARK
        // fusion version
        const auto readTF = fk::PerThreadRead<fk::ND::_2D, I>::build(d_input);
        const auto writeTF = fk::PerThreadWrite<fk::ND::_2D, O>::build(d_output_cvGS_ThreadFusion);
        fk::executeOperations<fk::TransformDPP<fk::defaultParArch, fk::TF::ENABLED>>(stream, readTF, fk::SaturateCast<I, O>::build(), writeTF);
        STOP_SECOND_BENCHMARK

        // Verify results
        d_output_cvGS_ThreadFusion.download(stream);
        d_output_cvGS.download(stream);

        stream.sync();

        passed = compareAndCheck(d_output_cvGS_ThreadFusion, d_output_cvGS);
    } catch (const std::exception& e) {
        error_s << e.what();
        passed = false;
        exception = true;
    }

    if (!passed) {
        if (!exception) {
            std::stringstream ss;
            ss << "testThreadFusionTimes<" << fk::typeToString<I>() << ", " << fk::typeToString<O>();
            std::cout << ss.str() << "> failed!! RESULT ERROR: Some results do not match baseline." << std::endl;
        } else {
            std::stringstream ss;
            ss << "testThreadFusionTimes<" << fk::typeToString<I>() << ", " << fk::typeToString<O>();
            std::cout << ss.str() << "> failed!! EXCEPTION: " << error_s.str() << std::endl;
        }
    }
    return passed;
}

template <typename I, typename T, typename O, enum fk::ColorConversionCodes CODE, size_t RESOLUTION>
bool testThreadFusionDifferentTypeAndChannelIO(fk::Stream& stream) {
    std::stringstream error_s;
    bool passed = true;
    bool exception = false;

    constexpr size_t BATCH = RESOLUTION;

    constexpr uint NUM_ELEMS_X = static_cast<uint>(RESOLUTION);
    constexpr uint NUM_ELEMS_Y = static_cast<uint>(RESOLUTION);

    constexpr I val_init = fk::make_set<I>(2);

    try {
        fk::Ptr2D<I> d_input(NUM_ELEMS_X, NUM_ELEMS_Y);
        fk::setTo(val_init, d_input, stream);
        fk::Ptr2D<O> d_output_cvGS(NUM_ELEMS_X, NUM_ELEMS_Y);
        fk::Ptr2D<O> d_output_cvGS_ThreadFusion(NUM_ELEMS_X, NUM_ELEMS_Y);

        // In this case it's not OpenCV, it's cvGPUSpeedup without thread fusion
        START_FIRST_BENCHMARK(fk::defaultParArch)
        // non fusion version
        const auto read = fk::PerThreadRead<fk::ND::_2D, I>::build(d_input);
        const auto write = fk::PerThreadWrite<fk::ND::_2D, O>::build(d_output_cvGS);
        fk::executeOperations<fk::TransformDPP<fk::defaultParArch, fk::TF::DISABLED>>(stream, read, fk::SaturateCast<I, T>::build(), fk::ColorConversion<CODE, T, O>::build(), write);
        STOP_FIRST_START_SECOND_BENCHMARK
        // fusion version
        const auto readTF = fk::PerThreadRead<fk::ND::_2D, I>::build(d_input);
        const auto writeTF = fk::PerThreadWrite<fk::ND::_2D, O>::build(d_output_cvGS_ThreadFusion);
        fk::executeOperations<fk::TransformDPP<fk::defaultParArch, fk::TF::ENABLED>>(stream, readTF, fk::SaturateCast<I, T>::build(), fk::ColorConversion<CODE, T, O>::build(), writeTF);
        STOP_SECOND_BENCHMARK

        // Verify results
        d_output_cvGS_ThreadFusion.download(stream);
        d_output_cvGS.download(stream);

        stream.sync();

        passed = compareAndCheck(d_output_cvGS_ThreadFusion, d_output_cvGS);
    } catch (const std::exception& e) {
        error_s << e.what();
        passed = false;
        exception = true;
    }

    if (!passed) {
        if (!exception) {
            std::stringstream ss;
            ss << "testThreadFusionTimes<" << fk::typeToString<I>() << ", " << fk::typeToString<O>();
            std::cout << ss.str() << "> failed!! RESULT ERROR: Some results do not match baseline." << std::endl;
        } else {
            std::stringstream ss;
            ss << "testThreadFusionTimes<" << fk::typeToString<I>() << ", " << fk::typeToString<O>();
            std::cout << ss.str() << "> failed!! EXCEPTION: " << error_s.str() << std::endl;
        }
    }
    return passed;
}

template <size_t... IDX>
bool testThreadFusionSameTypeIO_launcher_impl(fk::Stream& stream, const std::integer_sequence<size_t, IDX...>&) {
    bool passed = true;

#define LAUNCH_testThreadFusionSameTypeIO(BASE) \
    passed &= (testThreadFusionSameTypeIO<BASE, variableDimensionValues[IDX]>(stream) && ...); \
    passed &= (testThreadFusionSameTypeIO<BASE, variableDimensionValues[IDX] + 1>(stream) && ...); \
    passed &= (testThreadFusionSameTypeIO<BASE ## 2, variableDimensionValues[IDX]>(stream) && ...); \
    passed &= (testThreadFusionSameTypeIO<BASE ## 2, variableDimensionValues[IDX] + 1>(stream) && ...); \
    passed &= (testThreadFusionSameTypeIO<BASE ## 3, variableDimensionValues[IDX]>(stream) && ...); \
    passed &= (testThreadFusionSameTypeIO<BASE ## 3, variableDimensionValues[IDX] + 1>(stream) && ...); \
    passed &= (testThreadFusionSameTypeIO<BASE ## 4, variableDimensionValues[IDX]>(stream) && ...); \
    passed &= (testThreadFusionSameTypeIO<BASE ## 4, variableDimensionValues[IDX] + 1>(stream) && ...);

    LAUNCH_testThreadFusionSameTypeIO(uchar)
    LAUNCH_testThreadFusionSameTypeIO(char)
    LAUNCH_testThreadFusionSameTypeIO(ushort)
    LAUNCH_testThreadFusionSameTypeIO(short)
    LAUNCH_testThreadFusionSameTypeIO(int)
    LAUNCH_testThreadFusionSameTypeIO(float)
    LAUNCH_testThreadFusionSameTypeIO(double)
#undef LAUNCH_testThreadFusionTimes

        return passed;
}

template <size_t... IDX>
bool testThreadFusionDifferentTypeIO_launcher_impl(fk::Stream& stream, const std::index_sequence<IDX...>&) {
    bool passed = true;

    passed &= (testThreadFusionDifferentTypeIO<uchar, float, variableDimensionValues[IDX]>(stream) && ...);
    passed &= (testThreadFusionDifferentTypeIO<uchar, float, variableDimensionValues[IDX] + 1>(stream) && ...);
    passed &= (testThreadFusionDifferentTypeIO<uchar2, float2, variableDimensionValues[IDX]>(stream) && ...);
    passed &= (testThreadFusionDifferentTypeIO<uchar2, float2, variableDimensionValues[IDX] + 1>(stream) && ...);
    passed &= (testThreadFusionDifferentTypeIO<uchar3, float3, variableDimensionValues[IDX]>(stream) && ...);
    passed &= (testThreadFusionDifferentTypeIO<uchar3, float3, variableDimensionValues[IDX] + 1>(stream) && ...);
    passed &= (testThreadFusionDifferentTypeIO<uchar4, float4, variableDimensionValues[IDX]>(stream) && ...);
    passed &= (testThreadFusionDifferentTypeIO<uchar4, float4, variableDimensionValues[IDX] + 1>(stream) && ...);
    passed &= (testThreadFusionDifferentTypeIO<ushort, float, variableDimensionValues[IDX]>(stream) && ...);
    passed &= (testThreadFusionDifferentTypeIO<ushort, float, variableDimensionValues[IDX] + 1>(stream) && ...);
    passed &= (testThreadFusionDifferentTypeIO<ushort2, float2, variableDimensionValues[IDX]>(stream) && ...);
    passed &= (testThreadFusionDifferentTypeIO<ushort2, float2, variableDimensionValues[IDX] + 1>(stream) && ...);
    passed &= (testThreadFusionDifferentTypeIO<ushort3, float3, variableDimensionValues[IDX]>(stream) && ...);
    passed &= (testThreadFusionDifferentTypeIO<ushort3, float3, variableDimensionValues[IDX] + 1>(stream) && ...);
    passed &= (testThreadFusionDifferentTypeIO<ushort4, float4, variableDimensionValues[IDX]>(stream) && ...);
    passed &= (testThreadFusionDifferentTypeIO<ushort4, float4, variableDimensionValues[IDX] + 1>(stream) && ...);

    return passed;
}

template <size_t... IDX>
bool testThreadFusionDifferentTypeAndChannelIO_launcher_impl(fk::Stream& stream, const std::index_sequence<IDX...>&) {
    bool passed = true;

    passed &= (testThreadFusionDifferentTypeAndChannelIO<uchar3, float3, float4, fk::ColorConversionCodes::COLOR_RGB2RGBA, variableDimensionValues[IDX]>(stream) && ...);
    passed &= (testThreadFusionDifferentTypeAndChannelIO<uchar3, float3, float4, fk::ColorConversionCodes::COLOR_RGB2RGBA, variableDimensionValues[IDX] + 1>(stream) && ...);
    passed &= (testThreadFusionDifferentTypeAndChannelIO<uchar4, float4, float3, fk::ColorConversionCodes::COLOR_RGBA2RGB, variableDimensionValues[IDX]>(stream) && ...);
    passed &= (testThreadFusionDifferentTypeAndChannelIO<uchar4, float4, float3, fk::ColorConversionCodes::COLOR_RGBA2RGB, variableDimensionValues[IDX] + 1>(stream) && ...);
    passed &= (testThreadFusionDifferentTypeAndChannelIO<float3, uchar3, uchar4, fk::ColorConversionCodes::COLOR_RGB2RGBA, variableDimensionValues[IDX]>(stream) && ...);
    passed &= (testThreadFusionDifferentTypeAndChannelIO<float3, uchar3, uchar4, fk::ColorConversionCodes::COLOR_RGB2RGBA, variableDimensionValues[IDX] + 1>(stream) && ...);
    passed &= (testThreadFusionDifferentTypeAndChannelIO<float4, uchar4, uchar3, fk::ColorConversionCodes::COLOR_RGBA2RGB, variableDimensionValues[IDX]>(stream) && ...);
    passed &= (testThreadFusionDifferentTypeAndChannelIO<float4, uchar4, uchar3, fk::ColorConversionCodes::COLOR_RGBA2RGB, variableDimensionValues[IDX] + 1>(stream) && ...);
    passed &= (testThreadFusionDifferentTypeAndChannelIO<float4, uchar4, uchar, fk::ColorConversionCodes::COLOR_RGBA2GRAY, variableDimensionValues[IDX]>(stream) && ...);
    passed &= (testThreadFusionDifferentTypeAndChannelIO<float4, uchar4, uchar, fk::ColorConversionCodes::COLOR_RGBA2GRAY, variableDimensionValues[IDX] + 1>(stream) && ...);

    return passed;
}

bool testThreadFusionSameTypeIO_launcher(fk::Stream& stream) {
    return testThreadFusionSameTypeIO_launcher_impl(stream, std::make_index_sequence<variableDimensionValues.size()>());
}

bool testThreadFusionDifferentTypeIO_launcher(fk::Stream& stream) {
    return testThreadFusionDifferentTypeIO_launcher_impl(stream, std::make_index_sequence<variableDimensionValues.size()>());
}

bool testThreadFusionDifferentTypeAndChannelIO_launcher(fk::Stream& stream) {
    return testThreadFusionDifferentTypeAndChannelIO_launcher_impl(stream, std::make_index_sequence<variableDimensionValues.size()>());
}

int launch() {
    fk::Stream stream;
    bool passed = true;
    {
        PUSH_RANGE_RAII p("testThreadFusionSameTypeIO");
        passed &= testThreadFusionSameTypeIO_launcher(stream);
    }
    {
        PUSH_RANGE_RAII p("testThreadFusionDifferentTypeIO");
        passed &= testThreadFusionDifferentTypeIO_launcher(stream);
    }
    {
        PUSH_RANGE_RAII p("testThreadFusionDifferentTypeAndChannelIO");
        passed &= testThreadFusionDifferentTypeAndChannelIO_launcher(stream);
    }
    CLOSE_BENCHMARK

    if (passed) {
        std::cout << "test_thread_fusion Passed!!!" << std::endl;
        return 0;
    } else {
        std::cout << "test_thread_fusion Failed!!!" << std::endl;
        return -1;
    }
}