/* Copyright 2026 Oscar Amoros Huguet, Johnny Nunez

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

// Reduced float element types in full executeOperations pipelines, on both backends:
// the same source builds as _cpp (ParArch::CPU) and _cu (ParArch::GPU_NVIDIA).

#include <tests/main.h>
#include <tests/operation_test_utils.h>

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/cast.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/saturate.h>

#include <iostream>

using namespace fk;

namespace {
    constexpr int WIDTH = 67; // odd width: exercises the non thread-divisible path too
    constexpr int HEIGHT = 23;

    // Read T -> Cast to float -> x2 -> SaturateCast back to T. The expected value is computed
    // element by element with the same scalar operations, so the comparison is bit exact.
    template <typename T>
    bool scalarPipeline() {
        Stream stream;
        Ptr2D<T> input(WIDTH, HEIGHT);
        Ptr2D<T> output(WIDTH, HEIGHT);
        for (int y = 0; y < HEIGHT; ++y) {
            for (int x = 0; x < WIDTH; ++x) {
                // Values exactly representable in every format under test
                input.at(Point(x, y)) = T(static_cast<float>((x % 3) - 1));
            }
        }
        input.upload(stream);
        executeOperations<TransformDPP<>>(stream,
                                          PerThreadRead<ND::_2D, T>::build(input.ptr()),
                                          Cast<T, float>::build(),
                                          Mul<float>::build(2.0f),
                                          SaturateCast<float, T>::build(),
                                          PerThreadWrite<ND::_2D, T>::build(output.ptr()));
        output.download(stream);
        stream.sync();
        for (int y = 0; y < HEIGHT; ++y) {
            for (int x = 0; x < WIDTH; ++x) {
                const T expected = cxp::saturate_cast<T>::f(static_cast<float>(input.at(Point(x, y))) * 2.0f);
                if (!equalValues(output.at(Point(x, y)), expected)) {
                    std::cout << "scalar pipeline mismatch at (" << x << "," << y << ") for "
                              << typeToString<T>() << std::endl;
                    return false;
                }
            }
        }
        return true;
    }

    // Same pipeline expressed as loose operations and as a .then() fused operation: the two
    // must produce bit identical results (quantization applies equally to both).
    bool fusedVsLoose() {
        Stream stream;
        Ptr2D<fp16> input(WIDTH, HEIGHT);
        Ptr2D<fp16> outLoose(WIDTH, HEIGHT);
        Ptr2D<fp16> outFused(WIDTH, HEIGHT);
        for (int y = 0; y < HEIGHT; ++y) {
            for (int x = 0; x < WIDTH; ++x) {
                input.at(Point(x, y)) = fp16(0.25f * static_cast<float>(x - 30));
            }
        }
        input.upload(stream);
        executeOperations<TransformDPP<>>(stream,
                                          PerThreadRead<ND::_2D, fp16>::build(input.ptr()),
                                          Cast<fp16, float>::build(),
                                          Mul<float>::build(3.0f),
                                          SaturateCast<float, fp16>::build(),
                                          PerThreadWrite<ND::_2D, fp16>::build(outLoose.ptr()));
        const auto fusedRead = PerThreadRead<ND::_2D, fp16>::build(input.ptr())
                                   .then(Cast<fp16, float>::build())
                                   .then(Mul<float>::build(3.0f))
                                   .then(SaturateCast<float, fp16>::build());
        executeOperations<TransformDPP<>>(stream, fusedRead,
                                          PerThreadWrite<ND::_2D, fp16>::build(outFused.ptr()));
        outLoose.download(stream);
        outFused.download(stream);
        stream.sync();
        for (int y = 0; y < HEIGHT; ++y) {
            for (int x = 0; x < WIDTH; ++x) {
                if (toBits(outLoose.at(Point(x, y))) != toBits(outFused.at(Point(x, y)))) {
                    std::cout << "fused vs loose mismatch at (" << x << "," << y << ")" << std::endl;
                    return false;
                }
            }
        }
        return true;
    }

    // Vector element type end to end
    bool vectorPipeline() {
        Stream stream;
        Ptr2D<fp16_2> input(WIDTH, HEIGHT);
        Ptr2D<fp16_2> output(WIDTH, HEIGHT);
        for (int y = 0; y < HEIGHT; ++y) {
            for (int x = 0; x < WIDTH; ++x) {
                input.at(Point(x, y)) = fp16_2{ fp16(static_cast<float>(x % 5)), fp16(-0.5f * (y % 4)) };
            }
        }
        input.upload(stream);
        executeOperations<TransformDPP<>>(stream,
                                          PerThreadRead<ND::_2D, fp16_2>::build(input.ptr()),
                                          Cast<fp16_2, float2>::build(),
                                          Mul<float2>::build(float2{ 2.0f, 4.0f }),
                                          SaturateCast<float2, fp16_2>::build(),
                                          PerThreadWrite<ND::_2D, fp16_2>::build(output.ptr()));
        output.download(stream);
        stream.sync();
        for (int y = 0; y < HEIGHT; ++y) {
            for (int x = 0; x < WIDTH; ++x) {
                const auto in = input.at(Point(x, y));
                const auto out = output.at(Point(x, y));
                const fp16 expectedX = cxp::saturate_cast<fp16>::f(static_cast<float>(in.x) * 2.0f);
                const fp16 expectedY = cxp::saturate_cast<fp16>::f(static_cast<float>(in.y) * 4.0f);
                if (!equalValues(out.x, expectedX) || !equalValues(out.y, expectedY)) {
                    std::cout << "vector pipeline mismatch at (" << x << "," << y << ")" << std::endl;
                    return false;
                }
            }
        }
        return true;
    }

    // Thread fusion enabled: fp16 maps to fp16_2 (two elements per thread), including the
    // non divisible tail path (WIDTH is odd).
    bool threadFusionPipeline() {
        Stream stream;
        Ptr2D<fp16> input(WIDTH, HEIGHT);
        Ptr2D<fp16> output(WIDTH, HEIGHT);
        for (int y = 0; y < HEIGHT; ++y) {
            for (int x = 0; x < WIDTH; ++x) {
                input.at(Point(x, y)) = fp16(0.5f * static_cast<float>(x));
            }
        }
        input.upload(stream);
        executeOperations<TransformDPP<defaultParArch, TF::ENABLED>>(
            stream,
            PerThreadRead<ND::_2D, fp16>::build(input.ptr()),
            Cast<fp16, float>::build(),
            Mul<float>::build(2.0f),
            SaturateCast<float, fp16>::build(),
            PerThreadWrite<ND::_2D, fp16>::build(output.ptr()));
        output.download(stream);
        stream.sync();
        for (int y = 0; y < HEIGHT; ++y) {
            for (int x = 0; x < WIDTH; ++x) {
                const fp16 expected = cxp::saturate_cast<fp16>::f(static_cast<float>(input.at(Point(x, y))) * 2.0f);
                if (!equalValues(output.at(Point(x, y)), expected)) {
                    std::cout << "thread fusion pipeline mismatch at (" << x << "," << y << ")" << std::endl;
                    return false;
                }
            }
        }
        return true;
    }
} // namespace

int launch() {
    bool correct = true;

    correct &= scalarPipeline<fp16>();
    correct &= scalarPipeline<bf16>();
    correct &= scalarPipeline<fp8_e4m3>();
    correct &= scalarPipeline<fp8_e5m2>();
    correct &= scalarPipeline<fp4_e2m1>();
    correct &= fusedVsLoose();
    correct &= vectorPipeline();
    correct &= threadFusionPipeline();

    // Exactly representable values: expectations hold bit exactly on every backend.
    TestCaseBuilder<Cast<fp16, float>>::addTest(testCases,
        std::array<fp16, 4>{ fp16(1.0f), fp16(-2.5f), fp16(0.0f), fp16(65504.0f) },
        std::array<float, 4>{ 1.0f, -2.5f, 0.0f, 65504.0f });
    TestCaseBuilder<SaturateCast<float, fp16>>::addTest(testCases,
        std::array<float, 4>{ 1.0f, 1e20f, -1e20f, 0.5f },
        std::array<fp16, 4>{ fp16(1.0f), fp16(65504.0f), fp16(-65504.0f), fp16(0.5f) });
    TestCaseBuilder<SaturateCast<float, fp8_e4m3>>::addTest(testCases,
        std::array<float, 4>{ 1.0f, 1000.0f, -1000.0f, 0.25f },
        std::array<fp8_e4m3, 4>{ fp8_e4m3(1.0f), fp8_e4m3(448.0f), fp8_e4m3(-448.0f), fp8_e4m3(0.25f) });
    TestCaseBuilder<SaturateCast<fp16, uchar>>::addTest(testCases,
        std::array<fp16, 4>{ fp16(1.0f), fp16(300.0f), fp16(-5.0f), fp16(254.5f) },
        std::array<uchar, 4>{ 1, 255, 0, 254 });
    for (const auto& [testName, testFunc] : testCases) {
        correct &= testFunc();
    }
    testCases.clear();

    if (correct) {
        std::cout << "utest_reduced_float_pipelines: all checks passed" << std::endl;
    }
    return correct ? 0 : -1;
}
