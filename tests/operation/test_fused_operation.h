/* Copyright 2024-2025 Oscar Amoros Huguet

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

#include <fused_kernel/algorithms/basic_ops/basic_ops.h>
#include <fused_kernel/core/execution_model/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/image_processing.h>

using namespace fk;

constexpr bool test_fuseDFResultingTypes() {

    constexpr Read<PerThreadRead<ND::_2D, float>> readOp{};
    constexpr Binary<Add<float>> addOp{ 3.f };
    constexpr Unary<Cast<float, int>> castOp{};
    constexpr Write<PerThreadWrite<ND::_2D, float>> writeOp{};

    using Test = decltype(PerThreadRead<ND::_2D, float>::num_elems_y(std::declval<Point>(), std::declval<typename PerThreadRead<ND::_2D, float>::OperationDataType>()));

    static_assert(std::is_same_v<Test, uint>);

    constexpr auto fused1 = fuse(readOp, addOp, castOp);

    constexpr auto read = Read<PerThreadRead<ND::_2D, float>>{ { fk::RawPtr<ND::_2D, float>{nullptr, {128, 4}} } };
    static_assert(std::is_same_v<std::decay_t<decltype(read)>, Read<PerThreadRead<ND::_2D, float>>>, "Unexpected type after fuseIOps");

    constexpr auto readOp2 = PerThreadRead<ND::_2D, uchar3>::build(RawPtr<ND::_2D, uchar3>{nullptr, PtrDims<ND::_2D>(128,128)});
    static_assert(std::is_same_v<std::decay_t<decltype(readOp2)>, Read<PerThreadRead<ND::_2D, uchar3>>>, "Unexpected type after fuseIOps");

    constexpr auto readYUV = ReadYUV<PixelFormat::NV12>::build({ {RawPtr<ND::_2D, uchar>{nullptr, PtrDims<ND::_2D>(128, 128+64)}, 128, 128} });
    constexpr auto readRGB = readYUV.then(ConvertYUVToRGB<PixelFormat::NV12, ColorRange::Full, ColorPrimitives::bt2020, false>::build());

    constexpr auto resizeRead = Resize<InterpolationType::INTER_LINEAR>::build(readRGB, Size(64, 64));
    constexpr auto resizeReadWithMul = resizeRead.then(Mul<float>::build(3.f));

    constexpr auto resizeReadWithDiv = resizeReadWithMul.then(Div<float>::build(4.3f));
    static_assert(resizeReadWithDiv.params.next.next.instance.params == 4.3f, "Unexpected value after resizeRead");

    static_assert(std::is_same_v<typename decltype(fused1)::Operation,
        fk::FusedOperation<fk::PerThreadRead<fk::ND::_2D, float>, fk::Add<float>, fk::Cast<float, int>>>, "Unexpected type after fuseIOps");

    constexpr bool result1 = fk::is_fused_operation<fk::FusedOperation<fk::PerThreadRead<fk::ND::_2D, float>, fk::Add<float>, fk::Cast<float, int>>>::value;

    constexpr bool result2 = fk::is_fused_operation<typename decltype(fused1)::Operation>::value;

    static_assert(result1 && result2, "is_fused_operation does not work properly");

    constexpr auto fused2 = fk::fuse(readOp, addOp, writeOp);
    static_assert(std::is_same_v<typename decltype(fused2)::Operation,
        fk::FusedOperation<fk::PerThreadRead<fk::ND::_2D, float>, fk::Add<float>, fk::PerThreadWrite<fk::ND::_2D, float>>>, "Unexpected type after fuseIOps");

    return result1 && result2;
}

constexpr bool test_fuseFusedOperations() {
    const fk::Read<fk::PerThreadRead<fk::ND::_2D, float>> readOp{};
    const fk::Binary<fk::Add<float>> addOp{ 3.f };
    const fk::Unary<fk::Cast<float, int>> castOp{};

    const auto fused1 = fk::fuse(readOp, addOp);
    [[maybe_unused]] const auto fused2 = fk::fuse(fused1, castOp);

    return true;
}

int launch() {
    constexpr auto opTuple1 = fk::make_operation_tuple_<fk::Add<int, int, int, fk::UnaryType>>();

    using OpTuple1Type = decltype(opTuple1);

    static_assert(OpTuple1Type::size == 1, "Wrong operation tuple size");
    static_assert(fk::isUnaryType<typename OpTuple1Type::Operation>, "Wrong Operation Type");

    constexpr auto opTuple2 =
        fk::make_operation_tuple_<fk::Add<int, int, int, fk::UnaryType>, fk::Add<int>>
        (fk::OperationData<fk::Add<int>>{3});

    using OpTuple2Type = decltype(opTuple2);

    constexpr auto df2 = fk::Add<int, int, int, fk::UnaryType>::build().then(fk::Add<int >::build(3));
    static_assert(df2.params.next.instance.params == 3, "");

    constexpr auto result1 = decltype(df2)::Operation::exec(fk::Tuple<int, int>{4, 4}, df2);

    static_assert(result1 == 11, "Wrong result1");

    static_assert(OpTuple2Type::size == 2, "Wrong operation tuple size");
    static_assert(fk::isBinaryType<typename OpTuple2Type::Next::Operation>, "Wrong Operation Type");
    static_assert(opTuple2.next.instance.params == 3, "Wrong value");

    constexpr auto opTuple3 = fk::make_operation_tuple_<fk::Add<int, int, int, fk::UnaryType>,
    fk::Cast<int, float>, fk::Cast<float, int>>();

    using OpTuple3Type = decltype(opTuple3);

    constexpr auto df3 = fk::Add<int, int, int, fk::UnaryType>::build().then(fk::Cast<int, float>::build()).then(fk::Cast<float, int>::build());

    constexpr auto result3 = decltype(df3)::Operation::exec(fk::Tuple<int, int>{5,20});
    static_assert(result3 == 25, "Wrong result3");

    static_assert(OpTuple3Type::size == 3, "Wrong operation tuple size");
    //opTuple3.next; //must not compile
    static_assert(fk::isUnaryType<typename OpTuple3Type::Operation>, "Wrong Operation Type");

    static_assert(test_fuseDFResultingTypes(), "Something wrong with the types generated by fusedDF");
    static_assert(test_fuseFusedOperations(), "Something wrong while fusing a FusedOperation with another operation");

    using SomeFusedOp =
    fk::FusedOperation_<void,
        fk::Resize<fk::InterpolationType::INTER_LINEAR,
                   fk::AspectRatio::PRESERVE_AR,
                   fk::ReadBackInstantiableOperation<fk::Crop<fk::ReadInstantiableOperation<fk::PerThreadRead<fk::ND::_2D, uchar3>>>>>,
        fk::Mul<float3, float3, float3>,
        fk::Sub<float3, float3, float3>,
        fk::Div<float3, float3, float3>,
        fk::VectorReorder<float3, 2, 1, 0>>;

    static_assert(isCompleteOperation<SomeFusedOp>, "Something wrong with the compiler?");

    return 0;
}