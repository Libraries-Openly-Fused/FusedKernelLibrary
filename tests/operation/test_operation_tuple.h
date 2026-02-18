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

#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/execution_model/operation_model/operation_model.h>
#include <fused_kernel/core/execution_model/stream.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/cast.h>
#include <fused_kernel/algorithms/image_processing/saturate.h>

bool test_OTInitialization() {
    constexpr uint X = 64;
    constexpr uint Y = 64;

    const fk::Ptr2D<uchar> input(X, Y);
    using IOp = typename fk::PerThreadRead<fk::ND::_2D, uchar>::InstantiableType;
    const IOp read{ {input} };

    [[maybe_unused]] const fk::OperationTuple<IOp> testing{ {read} };

    const auto test2 = fk::make_new_operation_tuple(read);
    //const fk::Read<fk::FusedOperation<Op>> test3 = fk::fuse(read); //Should not compile

    using Op2 = fk::SaturateCast<uchar, uint>;
    constexpr fk::Unary<Op2> cast = {};

    const auto ot1 = fk::make_new_operation_tuple(read);
    constexpr auto ot2 = fk::make_new_operation_tuple(cast);

    const auto test4 = fk::make_new_operation_tuple(fk::get_opt<0>(ot1));

    constexpr auto filtered1 =
        fk::filtered_integer_sequence_t<int, fk::NotIsUnaryRestriction, fk::TypeList<typename IOp::InstanceType>>{};
    static_assert(filtered1.size() == 1, "Wrong filtered integer sequence size");

    const auto test6 = fk::cat(ot1, ot2);

    const auto test7 = fk::make_new_operation_tuple(read, cast);

    const auto test8 = fk::fuse(read, cast);

    const auto test9 = fk::Instantiable<fk::FusedOperation<std::decay_t<decltype(read)>,
                                                           std::decay_t<decltype(cast)>>>
    { fk::make_new_operation_tuple(read, cast) };

    return true;
}

bool testNewOperationTuple() {
    using namespace fk;

    constexpr auto op1 = Div<uchar>::build(1u);
    constexpr auto op2 = SaturateCast<uchar, uint>::build();
    constexpr auto op3 = Mul<uint>::build(5u);
    constexpr auto op4 = Cast<uint, float>::build();
    constexpr auto op5 = Add<float>::build(0.5f);
    constexpr auto op6 = Div<float>::build(0.6f);
    constexpr auto op7 = Mul<float>::build(0.8f);

    // Some unary
    constexpr auto opTuple1 = make_new_operation_tuple(op1, op2, op3);
    constexpr auto gotOp1 = get_opt<0>(opTuple1);
    constexpr auto gotOp2 = get_opt<1>(opTuple1);
    constexpr auto gotOp3 = get_opt<2>(opTuple1);

    static_assert(opTuple1.instances.size == 2, "Wrong Tuple size in OperationTuple");
    static_assert(gotOp1.params == 1u, "Wrong value in op1");
    static_assert(opIs<UnaryType, std::decay_t<decltype(gotOp2)>>, "Op2 must be Unary");
    static_assert(gotOp3.params == 5u, "Wrong value in op3");

    // All unary
    constexpr auto opTuple2 = make_new_operation_tuple(op2, op4);
    constexpr auto gotT2Op2 = get_opt<0>(opTuple2);
    constexpr auto gotT2Op4 = get_opt<1>(opTuple2);

    static_assert(opIs<UnaryType, std::decay_t<decltype(gotT2Op2)>>, "Op2 must be Unary");
    static_assert(opIs<UnaryType, std::decay_t<decltype(gotT2Op4)>>, "Op4 must be Unary");

    // None unary
    constexpr auto opTuple3 = make_new_operation_tuple(op5, op6, op7);
    constexpr auto gotT3Op5 = get_opt<0>(opTuple3);
    constexpr auto gotT3Op6 = get_opt<1>(opTuple3);
    constexpr auto gotT3Op7 = get_opt<2>(opTuple3);

    static_assert(gotT3Op5.params == 0.5f, "Wrong value in op5");
    static_assert(gotT3Op6.params == 0.6f, "Wrong value in op6");
    static_assert(gotT3Op7.params == 0.8f, "Wrong value in op7");

    return true;
}

int launch() {
    constexpr auto opTuple1 = fk::make_new_operation_tuple(fk::Add<int, int, int, fk::UnaryType>::build());

    using OpTuple1Type = std::decay_t<decltype(opTuple1)>;

    static_assert(OpTuple1Type::size == 1, "Wrong operation tuple size");
    static_assert(fk::opIs<fk::UnaryType, fk::TypeAt_t<0, typename OpTuple1Type::Operations>>, "Wrong Operation Type");

    constexpr fk::OperationData<fk::Add<int>> data{ 3 };
    static_assert(data.params == 3, "Wrong value");

    constexpr auto opTuple2 =
        fk::make_new_operation_tuple(fk::Add<int, int, int, fk::UnaryType>::build(), fk::Add<int>::build(3));

    using OpTuple2Type = decltype(opTuple2);

    static_assert(OpTuple2Type::size == 2, "Wrong operation tuple size");
    static_assert(fk::opIs<fk::BinaryType, fk::TypeAt_t<1, typename OpTuple2Type::Operations>>, "Wrong Operation Type");
    static_assert(fk::get_opt<1>(opTuple2).params == 3, "Wrong value");

    constexpr auto opTuple3 = fk::make_new_operation_tuple(fk::Add<int, int, int, fk::UnaryType>::build(),
    fk::Cast<int, float>::build(), fk::Cast<float, int>::build());

    using OpTuple3Type = decltype(opTuple3);

    static_assert(OpTuple3Type::size == 3, "Wrong operation tuple size");
    //opTuple3.next; must not compile
    static_assert(fk::opIs<fk::UnaryType, fk::TypeAt_t<0, typename OpTuple3Type::Operations>>, "Wrong Operation Type");
   
    if (!test_OTInitialization() || !testNewOperationTuple()) {
        return -1;
    }

    return 0;
}
