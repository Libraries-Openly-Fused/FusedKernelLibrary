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

#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/cast.h>
#include <fused_kernel/core/execution_model/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/resize.h>
#include <fused_kernel/core/utils/type_lists.h>
#include <fused_kernel/algorithms/basic_ops/set.h>
#include <fused_kernel/fused_kernel.h>

using namespace fk;

// Operation types
// Read
using RPerThrFloat = PerThreadRead<ND::_2D, float>;
// ReadBack
using RBResize = Resize<InterpolationType::INTER_LINEAR, AspectRatio::IGNORE_AR, Instantiable<RPerThrFloat>>;
// Unary
using UIntFloat = Cast<int, float>;
using UFloatInt = Cast<float, int>;
// Binary
using BAddInt = Add<int>;
using BAddFloat = Add<float>;
// Ternary
using TInterpFloat = InterpolateComplete<InterpolationType::INTER_LINEAR, Instantiable<RPerThrFloat>>;
// Write
using WPerThrFloat = PerThreadWrite<ND::_2D, float>;
// MidWrite
using MWPerThrFloat = FusedOperation<WPerThrFloat, BAddFloat>;

constexpr inline bool test_read_then_batch() {
    constexpr RawPtr<ND::_2D, float> input{ nullptr, { 64,64, 64 * sizeof(float) } };
    constexpr auto readIOp = RPerThrFloat::build(input);

    constexpr std::array<Size, 2> resizes{ Size(16, 16), Size(16, 16) };
    constexpr auto batchResize = Resize<InterpolationType::INTER_LINEAR>::build(2, 3.f, resizes);

    // start then()
    constexpr auto backOpArray = make_set_std_array<2>(readIOp);
    constexpr auto forwardOpArray = BatchUtils::toArray(batchResize);
    using BackwardIOp = typename decltype(backOpArray)::value_type;
    using ForwardIOp = typename decltype(forwardOpArray)::value_type;
    using ResultingType = decltype(ForwardIOp::Operation::build(std::declval<BackwardIOp>(), std::declval<ForwardIOp>()));
    constexpr auto fusedArray = std::array<ResultingType, 2>{ForwardIOp::Operation::build(backOpArray[0], forwardOpArray[0]),
                                                         ForwardIOp::Operation::build(backOpArray[1], forwardOpArray[1])};
    using ContinuationIOpNewType = typename decltype(fusedArray)::value_type;
    constexpr auto fusedBatchIOp = ContinuationIOpNewType::Operation::build(batchResize.params.usedPlanes, batchResize.params.default_value, fusedArray);
    using ResultingType_ = decltype(fusedBatchIOp);
    // end then()

    constexpr auto fusedBatchIOp2 = readIOp.then(batchResize);
    using ResultingType2 = decltype(fusedBatchIOp2);

    static_assert(std::is_same_v<ResultingType_, ResultingType2>);
    static_assert(fusedBatchIOp.params.usedPlanes == 2);
    static_assert(fusedBatchIOp.params.default_value == 3.f);
    static_assert(isReadType<ResultingType2>);
    static_assert(isReadBackType<typename ResultingType2::Operation::Operation>);
    static_assert(isTernaryType<typename ResultingType2::Operation::Operation::BackIOp>);

    return true;
}

constexpr inline bool test_readback_then_batch() {

    constexpr RawPtr<ND::_2D, float> input{ nullptr, { 64, 64, 64 * sizeof(float) } };
    constexpr auto readIOp = RPerThrFloat::build(input);
    constexpr auto oneResize = Resize<InterpolationType::INTER_LINEAR>::build(readIOp, Size(32, 32));
    using ResizeType = decltype(oneResize);

    constexpr std::array<Size, 2> resizes{ Size(16, 16), Size(16, 16) };
    constexpr auto batchResize = Resize<InterpolationType::INTER_LINEAR>::build(2, 3.f, resizes);

    constexpr auto fusedBatch = oneResize.then(batchResize);
    static_assert(decltype(fusedBatch)::Operation::BATCH == 2, "Unexpected BATCH size for fusedBATCH");

    return true;
}

constexpr inline bool test_batch_then_readback() {
    constexpr std::array<RawPtr<ND::_2D, float>, 2> inputs{ RawPtr<ND::_2D, float>{nullptr, {64,64, 64 * sizeof(float)}},
                                                        RawPtr<ND::_2D, float>{nullptr, {128,64, 64 * (sizeof(float))}} };

    constexpr auto readBatchOp = RPerThrFloat::build(inputs);
    using ReadIOp = decltype(readBatchOp);

    constexpr auto oneResize = Resize<InterpolationType::INTER_LINEAR>::build(Size(32, 32));
    using ResizeType = decltype(oneResize);

    constexpr auto fusedBatch = readBatchOp.then(oneResize);
    static_assert(decltype(fusedBatch)::Operation::BATCH == 2, "Unexpected BATCH size for fusedBATCH");

    return true;
}

constexpr inline bool test_batch_then_compute() {
    constexpr std::array<RawPtr<ND::_2D, float>, 2> inputs{ RawPtr<ND::_2D, float>{nullptr, {64,64, 64 * sizeof(float)}},
                                                        RawPtr<ND::_2D, float>{nullptr, {64,64, 64 * (sizeof(float))}} };

    constexpr auto readBatchOp = RPerThrFloat::build(inputs);
    using ReadIOp = decltype(readBatchOp);

    constexpr auto fusedBatch = readBatchOp.then(Add<float>::build(4.5));
    static_assert(decltype(fusedBatch)::Operation::BATCH == 2, "Unexpected BATCH size for fusedBATCH");

    return true;
}

constexpr inline bool test_read_then_readback() {
    constexpr RawPtr<ND::_2D, float> input{ nullptr, { 64, 64, 64 * sizeof(float) } };
    constexpr auto readIOp = RPerThrFloat::build(input);

    constexpr auto fusedOp = readIOp.then(Resize<InterpolationType::INTER_LINEAR, AspectRatio::PRESERVE_AR>::build(Size(16,32), 0.5f));
    static_assert(isReadBackType<decltype(fusedOp)>, "The IOp should be a ReadBack type");

    return true;
}

constexpr inline bool test_batched() {

    constexpr std::array<RawPtr<ND::_2D, float>, 2> inputs{ RawPtr<ND::_2D, float>{nullptr, {64,64, 64*sizeof(float)}},
                                                        RawPtr<ND::_2D, float>{nullptr, {64,64, 64*(sizeof(float))}}};

    constexpr auto readBatchOp = RPerThrFloat::build(inputs);
    using ReadIOp = decltype(readBatchOp);

    constexpr auto oneResize = Resize<InterpolationType::INTER_LINEAR>::build(Size(32, 32));
    using ResizeType = decltype(oneResize);

    constexpr std::array<Size, 2> resizes{ Size(32, 32), Size(32, 32) };

    constexpr auto batchResize = Resize<InterpolationType::INTER_LINEAR>::build(2, 3.f, resizes);

    // start then()
    constexpr auto bkArray = BatchUtils::toArray(readBatchOp);
    constexpr auto fwdArray = BatchUtils::toArray(batchResize);
    using BackwardIOp = std::decay_t<decltype(bkArray[0])>;
    using ForwardIOp = std::decay_t<decltype(fwdArray[0])>;

    using ResultingType = decltype(ForwardIOp::Operation::build(std::declval<BackwardIOp>(), std::declval<ForwardIOp>()));
    constexpr std::array<ResultingType, 2> fusedArray{ForwardIOp::Operation::build(bkArray[0], fwdArray[0]),
                                                      ForwardIOp::Operation::build(bkArray[1], fwdArray[1])};

    constexpr auto batcheResize = ResultingType::Operation::build(batchResize.params.usedPlanes,
                                                                  batchResize.params.default_value,
                                                                  fusedArray);
    // end then()
    constexpr auto fusedBatchesOp = readBatchOp.then(batchResize);

    constexpr std::array<RawPtr<ND::_2D, float>, 2> outputs{ RawPtr<ND::_2D, float>{nullptr, {64,64, 64 * sizeof(float)}},
                                                         RawPtr<ND::_2D, float>{nullptr, {64,64, 64 * (sizeof(float))}} };

    [[maybe_unused]] constexpr auto writeBatchOp = PerThreadWrite<ND::_2D, float>::build(outputs);

    return std::is_same_v<decltype(batcheResize), decltype(fusedBatchesOp)>;
}

int launch() {
    constexpr Instantiable<RPerThrFloat> func1{};
    func1.then(UFloatInt::build());
    fuse(func1, UFloatInt::build());
    constexpr auto func2 =
        func1.then(Instantiable<UFloatInt>{}).
        then(Instantiable<BAddInt>{4}).
        then(Instantiable<UIntFloat>{}).
        then(Instantiable<BAddFloat>{5.3});

    static_assert(func2.is<ReadType>);
    static_assert(func2.params.size == 5);
    using ResType = decltype(func2);
    static_assert(is_fused_operation<typename ResType::Operation>::value);
    using ResOperationTuple = typename ResType::Operation::ParamsType;
    constexpr bool noIntermediateFusedOperation =
        and_v<!is_fused_operation<ResOperationTuple::Operation>::value,
        !is_fused_operation<ResOperationTuple::Next::Operation>::value,
        !is_fused_operation<ResOperationTuple::Next::Next::Operation>::value,
        !is_fused_operation<ResOperationTuple::Next::Next::Next::Operation>::value,
        !is_fused_operation<ResOperationTuple::Next::Next::Next::Next::Operation>::value>;
    static_assert(noIntermediateFusedOperation);

    // All Unary
    constexpr auto func = Instantiable<UFloatInt>{}.then(Instantiable<UIntFloat>{}).then(Instantiable<UFloatInt>{});

    using Operations = decltype(func)::Operation::Operations;
    static_assert(Operations::size == 3);
    static_assert(std::is_same_v<TypeAt_t<0,Operations>, UFloatInt>);
    static_assert(std::is_same_v<TypeAt_t<1, Operations>, UIntFloat>);
    static_assert(std::is_same_v<TypeAt_t<2, Operations>, UFloatInt>);
    static_assert(decltype(func)::Operation::exec(5.5f) == 5);

    constexpr auto op = BAddInt::build(45);

    static_assert(op.params == 45);
    static_assert(decltype(op)::Operation::exec(10, op.params) == 55);

    Ptr2D<uint3> outputAlt(32, 32);
    constexpr RawPtr<ND::_2D, uchar3> input{nullptr, PtrDims<ND::_2D>(128, 128)};
    constexpr Size dstSize(32, 32);
    Stream stream;

    constexpr auto someReadOp =
        PerThreadRead<ND::_2D, uchar3>::build(input).then(Cast<uchar3, float3>::build()).then(Resize<InterpolationType::INTER_LINEAR>::build(dstSize));
    static_assert(isReadBackType<decltype(someReadOp)>, "Unexpected Operation Type for someReadOp");
    static_assert(std::is_same_v<decltype(someReadOp.backIOp.backIOp.params), OperationTuple<PerThreadRead<ND::_2D, uchar3>, Cast<uchar3, float3>>>, "Unexpected type for params");

    constexpr bool correct =
        std::is_same_v<OperationTuple<PerThreadRead<ND::_2D, uchar3>, Cast<uchar3, float3>>, decltype(someReadOp.backIOp.backIOp.params)>;
    static_assert(correct, "Unexpected resulting type");

    constexpr auto finalOp = someReadOp.then(Mul<float3>::build(make_<float3>(3.f, 1.f, 32.f)));
    static_assert(!isReadBackType<std::decay_t<decltype(finalOp)>>, "Unexpected type for finalOp");

    constexpr auto inputAlt = ReadSet<uchar3>::build({ { { 0,0,0 }, {128,128,1} } });

    constexpr ActiveThreads activeThreads = inputAlt.getActiveThreads();

    static_assert(activeThreads.x == 128, "Incorrect size in x");
    static_assert(activeThreads.y == 128, "Incorrect size in y");
    static_assert(activeThreads.z == 1, "Incorrect size in z");

    constexpr uchar3 value = make_<uchar3>(0, 0, 0);
    constexpr ActiveThreads threads = ActiveThreads(128, 128, 1);

    constexpr float3 addValue = make_<float3>(3.f, 1.f, 32.f);

    constexpr auto someReadOpAlt =
        ReadSet<uchar3>::build(value, threads)
                        .then(Cast<uchar3, float3>::build())
                        .then(Resize<InterpolationType::INTER_LINEAR>::build(dstSize))
                        .then(Add<float3>::build(addValue))
                        .then(Cast<float3, uint3>::build());

    static_assert(someReadOpAlt.getActiveThreads().x == 32, "Wrong width");
    static_assert(someReadOpAlt.getActiveThreads().y == 32, "Wrong height");
    static_assert(someReadOpAlt.getActiveThreads().z == 1, "Wrong depth");

    executeOperations<fk::TransformDPP<>>(stream, someReadOpAlt, PerThreadWrite<ND::_2D, uint3>::build(outputAlt));

    outputAlt.download(stream);
    stream.sync();

    bool correct2{ true };

    for (int y = 0; y < 32; ++y) {
        for (int x = 0; x < 32; ++x) {
            const uint3 temp = *PtrAccessor<ND::_2D>::cr_point({x, y}, outputAlt.ptrPinned());
            correct2 &= (temp.x == 3 && temp.y == 1 && temp.z == 32);
        }
    }

    constexpr bool correct3 = and_v<test_batched(),
                                    test_read_then_batch(),
                                    test_readback_then_batch(),
                                    test_batch_then_readback(),
                                    test_batch_then_compute(),
                                    test_read_then_readback()>;

    return (correct && correct2 && correct3) ? 0 : -1;
}