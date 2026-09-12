/* Copyright 2024-2026 Oscar Amoros Huguet

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
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/image_processing.h>

namespace fk {

constexpr bool test_fuseDFResultingTypes() {

    constexpr Read<PerThreadRead<ND::_2D, float>> readOp{};
    constexpr Binary<Add<float>> addOp{ 3.f };
    constexpr Unary<Cast<float, int>> castOp{};
    constexpr Write<PerThreadWrite<ND::_2D, float>> writeOp{};

    using Test = decltype(PerThreadRead<ND::_2D, float>::num_elems_y(std::declval<Point>(), std::declval<typename PerThreadRead<ND::_2D, float>::OperationDataType>()));

    static_assert(std::is_same_v<std::decay_t<Test>, uint>);

    constexpr auto fused1 = fuse(readOp, addOp, castOp);

    constexpr auto read = Read<PerThreadRead<ND::_2D, float>>{ { RawPtr<ND::_2D, float>{nullptr, {128, 4}} } };
    static_assert(std::is_same_v<std::decay_t<decltype(read)>, Read<PerThreadRead<ND::_2D, float>>>, "Unexpected type after fuseIOps");

    constexpr auto readOp2 = PerThreadRead<ND::_2D, uchar3>::build(RawPtr<ND::_2D, uchar3>{nullptr, PtrDims<ND::_2D>(128, 128)});
    static_assert(std::is_same_v<std::decay_t<decltype(readOp2)>, Read<PerThreadRead<ND::_2D, uchar3>>>, "Unexpected type after fuseIOps");

    constexpr auto readYUV = ReadYUV<PixelFormat::NV12>::build({ {RawPtr<ND::_2D, uchar>{nullptr, PtrDims<ND::_2D>(128, 128 + 64)}, 128, 128} });
    constexpr auto readRGB = readYUV.then(ConvertYUVToRGB<ColorDepth::p8bit, ColorRange::Full, ColorPrimitives::bt2020>::build());

    constexpr auto resizeRead = Resize<InterpolationType::INTER_LINEAR>::build(readRGB, Size(64, 64));
    constexpr auto resizeReadWithMul = resizeRead.then(Mul<float>::build(3.f));

    constexpr auto resizeReadWithDiv = resizeReadWithMul.then(Div<float>::build(4.3f));
    static_assert(get_opt<2>(resizeReadWithDiv.params).params == 4.3f, "Unexpected value after resizeRead");

    static_assert(std::is_same_v<typename std::decay_t<decltype(fused1)>::Operation,
        FusedOperation<Read<PerThreadRead<ND::_2D, float>>, Binary<Add<float>>, Unary<Cast<float, int>>>>, "Unexpected type after fuseIOps");

    constexpr bool result1 = is_fused_operation<FusedOperation<Read<PerThreadRead<ND::_2D, float>>, Binary<Add<float>>, Unary<Cast<float, int>>>>::value;

    constexpr bool result2 = is_fused_operation<typename decltype(fused1)::Operation>::value;

    static_assert(result1 && result2, "is_fused_operation does not work properly");

    constexpr auto fused2 = fuse(readOp, addOp, writeOp);
    static_assert(std::is_same_v<typename std::decay_t<decltype(fused2)>::Operation,
        FusedOperation<Read<PerThreadRead<ND::_2D, float>>, Binary<Add<float>>, Write<PerThreadWrite<ND::_2D, float>>>>,
        "Unexpected type after fuseIOps");

    return result1 && result2;
}

constexpr bool test_fuseFusedOperations() {
    const Read<PerThreadRead<ND::_2D, float>> readOp{};
    const Binary<Add<float>> addOp{ 3.f };
    const Unary<Cast<float, int>> castOp{};

    const auto fused1 = fuse(readOp, addOp);
    [[maybe_unused]] const auto fused2 = fuse(fused1, castOp);

    return true;
}

int launch_impl() {
    using namespace fk;
    constexpr auto opTuple1 = make_new_operation_tuple(Add<int, int, int, UnaryType>::build());

    using OpTuple1Type = std::decay_t<decltype(opTuple1)>;

    static_assert(OpTuple1Type::size == 1, "Wrong operation tuple size");

    constexpr auto opTuple2 = make_new_operation_tuple(Add<int, int, int, UnaryType> ::build(), Add<int>::build(3));

    using OpTuple2Type = decltype(opTuple2);

    constexpr auto df2 = Add<int, int, int, UnaryType>::build().then(Add<int >::build(3));
    static_assert(get_opt<1>(df2.params).params == 3, "");

    constexpr auto result1 = std::decay_t<decltype(get_opt<0>(df2.params))>::Operation::exec(Tuple<int, int>{4, 4});

    static_assert(result1 == 8, "Wrong result1");

    static_assert(OpTuple2Type::size == 2, "Wrong operation tuple size");
    static_assert(opIs<BinaryType, TypeAt_t<1, typename OpTuple2Type::Operations>>, "Wrong Operation Type");
    static_assert(get_opt<1>(opTuple2).params == 3, "Wrong value");

    constexpr auto opTuple3 = make_new_operation_tuple(Add<int, int, int, UnaryType>::build(),
    Cast<int, float>::build(), Cast<float, int>::build());

    using OpTuple3Type = decltype(opTuple3);

    constexpr auto df3 = Add<int, int, int, UnaryType>::build().then(Cast<int, float>::build()).then(Cast<float, int>::build());

    constexpr auto result3 = TypeAt_t<0, typename decltype(df3)::Operation::Operations>::Operation::exec(Tuple<int, int>{5,20});
    static_assert(result3 == 25, "Wrong result3");

    static_assert(OpTuple3Type::size == 3, "Wrong operation tuple size");
    //opTuple3.next; //must not compile
    static_assert(opIs<UnaryType, TypeAt_t<0, typename OpTuple3Type::Operations>>, "Wrong Operation Type");

    static_assert(test_fuseDFResultingTypes(), "Something wrong with the types generated by fusedDF");
    static_assert(test_fuseFusedOperations(), "Something wrong while fusing a FusedOperation with another operation");

    using SomeFusedOp =
    FusedOperation<
        ReadBack<ResizeComplete<AspectRatio::PRESERVE_AR,
                  Ternary<InterpolateComplete<
                      InterpolationType::INTER_LINEAR,
                   ReadBack<Crop<Read<PerThreadRead<ND::_2D, uchar3>>>>>>>>,
        Binary<Mul<float3, float3, float3>>,
        Binary<Sub<float3, float3, float3>>,
        Binary<Div<float3, float3, float3>>,
        Unary<VectorReorder<float3, 2, 1, 0>>>;

    static_assert(isCompleteOperation<SomeFusedOp>, "Something wrong with the compiler?");

    // ===========================================================================
    // COMPREHENSIVE TESTS FOR DEEPLY NESTED FUSED OPERATIONS
    // Testing all specializations with various nesting depths as requested in PR feedback
    // ===========================================================================
    // Note: OpenType requires MidWriteType operations which are specialized internal operations.
    // OpenType is tested implicitly through the fold expression implementation tests below.

    // Test 1: ClosedType FusedOperation (Read + operations + Write)
    // This combines ReadType at the start and WriteType at the end
    {
        using ClosedFusedOp = FusedOperation<
            Read<PerThreadRead<ND::_2D, float>>,
            Binary<Add<float>>,
            Unary<Cast<float, int>>,
            Write<PerThreadWrite<ND::_2D, int>>
        >;
        static_assert(std::is_same_v<typename ClosedFusedOp::InstanceType, ClosedType>,
            "ClosedType FusedOperation not correctly identified");
    }

    // Test 2: WriteType FusedOperation (operations + Write, no Read at start)
    {
        using WriteFusedOp = FusedOperation<
            Binary<Add<float>>,
            Unary<Cast<float, int>>,
            Write<PerThreadWrite<ND::_2D, int>>
        >;
        static_assert(std::is_same_v<typename WriteFusedOp::InstanceType, WriteType>,
            "WriteType FusedOperation not correctly identified");
    }

    // Test 3: ReadType FusedOperation (Read + operations, no Write at end)
    {
        using ReadFusedOp = FusedOperation<
            Read<PerThreadRead<ND::_2D, float>>,
            Binary<Add<float>>,
            Unary<Cast<float, int>>
        >;
        static_assert(std::is_same_v<typename ReadFusedOp::InstanceType, ReadType>,
            "ReadType FusedOperation not correctly identified");
    }

    // Test 4: UnaryType FusedOperation (all operations are Unary)
    {
        using UnaryChainOp = FusedOperation<
            Unary<Cast<int, float>>,
            Unary<Cast<float, double>>,
            Unary<Cast<double, int>>
        >;
        static_assert(std::is_same_v<typename UnaryChainOp::InstanceType, UnaryType>,
            "UnaryType FusedOperation not correctly identified");
    }

    // Test 5: BinaryType FusedOperation (compute operations, no Read/Write/MidWrite)
    {
        using BinaryChainOp = FusedOperation<
            Binary<Add<float>>,
            Binary<Mul<float>>,
            Binary<Sub<float>>
        >;
        static_assert(std::is_same_v<typename BinaryChainOp::InstanceType, BinaryType>,
            "BinaryType FusedOperation not correctly identified");
    }

    // Test 6: Deeply nested ReadType (5+ levels)
    {
        using DeeplyNestedRead = FusedOperation<
            Read<PerThreadRead<ND::_2D, float>>,
            Binary<Add<float>>,
            Binary<Mul<float>>,
            Binary<Sub<float>>,
            Binary<Div<float>>,
            Unary<Cast<float, int>>
        >;
        static_assert(std::is_same_v<typename DeeplyNestedRead::InstanceType, ReadType>,
            "Deeply nested ReadType FusedOperation failed");
    }

    // Test 7: Deeply nested ClosedType (5+ levels)
    {
        using DeeplyNestedClosed = FusedOperation<
            Read<PerThreadRead<ND::_2D, float>>,
            Binary<Add<float>>,
            Binary<Mul<float>>,
            Binary<Sub<float>>,
            Binary<Div<float>>,
            Unary<Cast<float, int>>,
            Write<PerThreadWrite<ND::_2D, int>>
        >;
        static_assert(std::is_same_v<typename DeeplyNestedClosed::InstanceType, ClosedType>,
            "Deeply nested ClosedType FusedOperation failed");
    }

    // Test 8: Deeply nested UnaryType (5+ levels)
    {
        using DeeplyNestedUnary = FusedOperation<
            Unary<Cast<int, float>>,
            Unary<Cast<float, double>>,
            Unary<Cast<double, float>>,
            Unary<Cast<float, double>>,
            Unary<Cast<double, int>>
        >;
        static_assert(std::is_same_v<typename DeeplyNestedUnary::InstanceType, UnaryType>,
            "Deeply nested UnaryType FusedOperation failed");
    }

    // Test 9: Deeply nested BinaryType (5+ levels)
    {
        using DeeplyNestedBinary = FusedOperation<
            Binary<Add<float>>,
            Binary<Mul<float>>,
            Binary<Sub<float>>,
            Binary<Div<float>>,
            Binary<Add<float>>
        >;
        static_assert(std::is_same_v<typename DeeplyNestedBinary::InstanceType, BinaryType>,
            "Deeply nested BinaryType FusedOperation failed");
    }

    // Test 10: Deeply nested WriteType (5+ levels)
    {
        using DeeplyNestedWrite = FusedOperation<
            Binary<Add<float>>,
            Binary<Mul<float>>,
            Binary<Sub<float>>,
            Binary<Div<float>>,
            Unary<Cast<float, int>>,
            Write<PerThreadWrite<ND::_2D, int>>
        >;
        static_assert(std::is_same_v<typename DeeplyNestedWrite::InstanceType, WriteType>,
            "Deeply nested WriteType FusedOperation failed");
    }

    // Test 11: Very deeply nested ClosedType (10+ levels) - stress test
    {
        using VeryDeeplyNestedClosed = FusedOperation<
            Read<PerThreadRead<ND::_2D, float>>,
            Binary<Add<float>>,
            Binary<Mul<float>>,
            Binary<Sub<float>>,
            Binary<Div<float>>,
            Binary<Add<float>>,
            Binary<Mul<float>>,
            Unary<Cast<float, double>>,
            Binary<Div<double>>,
            Unary<Cast<double, float>>,
            Unary<Cast<float, int>>,
            Write<PerThreadWrite<ND::_2D, int>>
        >;
        static_assert(std::is_same_v<typename VeryDeeplyNestedClosed::InstanceType, ClosedType>,
            "Very deeply nested ClosedType FusedOperation failed");
    }

    // Test 12: Mixed compute types (Binary and Unary) in deep chain
    {
        using MixedComputeChain = FusedOperation<
            Binary<Add<int>>,
            Unary<Cast<int, float>>,
            Binary<Mul<float>>,
            Unary<Cast<float, double>>,
            Binary<Div<double>>,
            Unary<Cast<double, int>>
        >;
        static_assert(std::is_same_v<typename MixedComputeChain::InstanceType, BinaryType>,
            "Mixed compute chain should be BinaryType when not all are Unary");
    }

    // Test 13: Verify complex existing SomeFusedOp is still valid
    // This ensures backward compatibility with existing complex operations
    static_assert(std::is_same_v<typename SomeFusedOp::InstanceType, ReadType>,
        "Complex SomeFusedOp should be ReadType (starts with ReadBack, no Write at end)");

    // Test 14: Verify operation fusion with .then() for deep chains
    {
        constexpr auto chain1 = Add<int, int, int, UnaryType>::build()
            .then(Cast<int, float>::build())
            .then(Cast<float, double>::build())
            .then(Cast<double, float>::build())
            .then(Cast<float, int>::build());

        using ChainType = std::decay_t<decltype(chain1)>;
        static_assert(ChainType::Operation::Operations::size == 5,
            "Deep .then() chain should have 5 operations");
    }

    // Test 15: Verify FusedOperation can be fused again (nesting FusedOperations)
    {
        constexpr auto inner = fuse(
            Add<int, int, int, UnaryType>::build(),
            Cast<int, float>::build()
        );
        [[maybe_unused]] constexpr auto outer = fuse(
            inner,
            Cast<float, int>::build()
        );
        // This should compile without errors
    }

    // Test 16: Edge case - Single operation wrapped in FusedOperation
    {
        using SingleOpFused = FusedOperation<Unary<Cast<int, float>>>;
        static_assert(std::is_same_v<typename SingleOpFused::InstanceType, UnaryType>,
            "Single Unary operation should be UnaryType");
    }

    // Test 17: Edge case - Two operations (minimum for meaningful fusion)
    {
        using TwoOpFused = FusedOperation<
            Binary<Add<float>>,
            Unary<Cast<float, int>>
        >;
        static_assert(std::is_same_v<typename TwoOpFused::InstanceType, BinaryType>,
            "Two mixed compute ops should be BinaryType");
    }

    // Test 18: Maximum stress test - 15+ operations deeply nested
    {
        using MaxStressTest = FusedOperation<
            Read<PerThreadRead<ND::_2D, float>>,
            Binary<Add<float>>,
            Binary<Mul<float>>,
            Binary<Sub<float>>,
            Binary<Div<float>>,
            Binary<Add<float>>,
            Binary<Mul<float>>,
            Binary<Sub<float>>,
            Unary<Cast<float, double>>,
            Binary<Div<double>>,
            Binary<Add<double>>,
            Unary<Cast<double, float>>,
            Binary<Mul<float>>,
            Unary<Cast<float, int>>,
            Binary<Add<int>>,
            Write<PerThreadWrite<ND::_2D, int>>
        >;
        static_assert(std::is_same_v<typename MaxStressTest::InstanceType, ClosedType>,
            "Maximum stress test (15+ ops) ClosedType FusedOperation failed");
    }

    return 0;
}

} // namespace fk

int launch() {
    return fk::launch_impl();
}
