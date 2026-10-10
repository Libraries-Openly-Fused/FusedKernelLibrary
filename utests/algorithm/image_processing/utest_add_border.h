/* Copyright 2026 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include <tests/operation_test_utils.h>

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/add_border.h>
#include <fused_kernel/algorithms/image_processing/crop.h>

namespace fk {

// Reference implementation of copyMakeBorder with a constant border, independent from AddBorder
template <typename T>
Ptr<ND::_2D, T> referenceAddBorder(const Ptr<ND::_2D, T>& input, const int top, const int bottom, const int left,
                                   const int right, const T& value) {
    const int width = static_cast<int>(input.dims().width);
    const int height = static_cast<int>(input.dims().height);
    Ptr<ND::_2D, T> output(width + left + right, height + top + bottom, 0, MemType::Host);
    for (int y = 0; y < height + top + bottom; ++y) {
        for (int x = 0; x < width + left + right; ++x) {
            const int srcX = x - left;
            const int srcY = y - top;
            const bool inside = srcX >= 0 && srcX < width && srcY >= 0 && srcY < height;
            output.at(x, y) = inside ? input.at(srcX, srcY) : value;
        }
    }
    return output;
}

int launch_impl() {
    Stream stream;

    // 3x2 uchar3 input with distinct pixels
    Ptr2D<uchar3> inputRGB(3, 2);
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 3; ++x) {
            inputRGB.at(x, y) = make_<uchar3>(10 * y + x + 1, 100 + 10 * y + x, 200 + 10 * y + x);
        }
    }
    inputRGB.upload(stream);
    const auto readRGB = PerThreadRead<ND::_2D, uchar3>::build(inputRGB.ptr());

    // top/bottom only, left/right only, all four sides, default (zero) value and params struct
    const uchar3 value{7, 8, 9};
    const auto topBottom = readRGB.then(AddBorder<>::build(2, 1, 0, 0, value));
    const auto leftRight = readRGB.then(AddBorder<>::build(0, 0, 2, 1, value));
    const auto allSides = readRGB.then(AddBorder<>::build(AddBorderParams<uchar3>{1, 2, 3, 1, value}));
    const auto zeroValue = readRGB.then(AddBorder<>::build(1, 1, 1, 1));
    using ConstantOp = typename decltype(topBottom)::Operation;
    static_assert(std::is_same_v<ConstantOp, typename decltype(zeroValue)::Operation>,
                  "AddBorder without value must complete to the same Operation as with a value");
    TestCaseBuilder<ConstantOp>::addTest(testCases, stream,
        std::array{topBottom, leftRight, allSides, zeroValue},
        std::array{referenceAddBorder(inputRGB, 2, 1, 0, 0, value),
                   referenceAddBorder(inputRGB, 0, 0, 2, 1, value),
                   referenceAddBorder(inputRGB, 1, 2, 3, 1, value),
                   referenceAddBorder(inputRGB, 1, 1, 1, 1, uchar3{0, 0, 0})});

    // Border value of a different type than the BackIOp OutputType (int -> float)
    Ptr2D<float> inputFloat(2, 2);
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 2; ++x) {
            inputFloat.at(x, y) = 0.5f + x + 2 * y;
        }
    }
    inputFloat.upload(stream);
    const auto castValue = PerThreadRead<ND::_2D, float>::build(inputFloat.ptr()).then(AddBorder<>::build(1, 0, 0, 1, 5));
    TestCaseBuilder<typename decltype(castValue)::Operation>::addTest(testCases, stream, castValue,
        referenceAddBorder(inputFloat, 1, 0, 0, 1, 5.f));

    // Horizontal fusion: one AddBorder applied to a batch of crops, and one AddBorder per plane
    Ptr2D<uchar3> inputBig(8, 6);
    for (int y = 0; y < 6; ++y) {
        for (int x = 0; x < 8; ++x) {
            inputBig.at(x, y) = make_<uchar3>(x, y, x + 8 * y);
        }
    }
    inputBig.upload(stream);
    constexpr std::array<Rect, 3> rects{Rect(0, 0, 3, 2), Rect(4, 1, 3, 2), Rect(5, 4, 3, 2)};
    const auto batchCrop = PerThreadRead<ND::_2D, uchar3>::build(inputBig.ptr()).then(Crop<>::build(rects));
    const auto sameBorder = batchCrop.then(AddBorder<>::build(1, 1, 2, 0, value));
    const std::array<AddBorderParams<uchar3>, 3> planeParams{AddBorderParams<uchar3>{2, 0, 0, 2, value},
                                                             AddBorderParams<uchar3>{1, 1, 1, 1, value},
                                                             AddBorderParams<uchar3>{0, 2, 2, 0, uchar3{0, 0, 0}}};
    const auto perPlaneBorder = batchCrop.then(AddBorder<>::build(planeParams));
    using BatchOp = typename decltype(sameBorder)::Operation;
    static_assert(std::is_same_v<BatchOp, typename decltype(perPlaneBorder)::Operation>,
                  "Both batch forms must produce the same Operation");

    const auto expected3D = [&](const std::array<AddBorderParams<uchar3>, 3>& params) {
        Ptr<ND::_3D, uchar3> expected(5, 4, 3, 1, 0, MemType::Host);
        for (int z = 0; z < 3; ++z) {
            Ptr<ND::_2D, uchar3> crop(rects[z].width, rects[z].height, 0, MemType::Host);
            for (int y = 0; y < rects[z].height; ++y) {
                for (int x = 0; x < rects[z].width; ++x) {
                    crop.at(x, y) = inputBig.at(rects[z].x + x, rects[z].y + y);
                }
            }
            const auto& p = params[z];
            const auto plane = referenceAddBorder(crop, p.top, p.bottom, p.left, p.right, p.value);
            for (int y = 0; y < 4; ++y) {
                for (int x = 0; x < 5; ++x) {
                    expected.at(x, y, z) = plane.at(x, y);
                }
            }
        }
        return expected;
    };
    const AddBorderParams<uchar3> sameParams{1, 1, 2, 0, value};
    TestCaseBuilder<BatchOp>::addTest(testCases, stream, std::array{sameBorder, perPlaneBorder},
                                      std::array{expected3D({sameParams, sameParams, sameParams}), expected3D(planeParams)});

    bool correct{true};
    for (const auto& testCase : testCases) {
        correct &= testCase.second();
    }
    testCases.clear();
    return correct ? 0 : -1;
}

} // namespace fk

int launch() {
    return fk::launch_impl();
}
