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

// The check of the IOps that the executors of ReduceDPP make (reduce_detail::iopsAreValid and
// reduce_detail::reportIOps, in include/fused_kernel/core/execution_model/reduce_dpp.h). The mistakes of this test:
//   1. an IOp of a reduction that is not a compute IOp (a Write where the AccumulationIOp goes)
//   2. a FirstIOp that does not give the accumulator type (a float, for an accumulator that is a double)
//   3. the reductions are not a Tuple of Tuple<FirstIOp, AccumulationIOp, FinalIOp> (one reduction, not in a Tuple)
//   4. the output type of a FinalIOp does not convert to the input type of the write (a float3, for a write of floats)
//   5. an accumulator that can not be stored (a struct with a const member, which can not be assigned)
//   6. a read that is not complete (a batch of Crops without the read that they crop, which is a ReadType IOp)
//   7. the output type of the read does not convert to the input type of a FirstIOp (a float, for a float3)
//   8. an AccumulationIOp that does not take a Tuple of two accumulated values (a Cast)
//   9. the accumulator type does not convert to the input type of its FinalIOp (a float, for a float3)
//  10. what writes is not a Write IOp (a Cast)
// They are one per static_assert of the report: an executor discards the execution of IOps that are not valid, so a
// condition of iopsAreValid whose static_assert were missing from the report would compile, and do nothing.
//
// Without a macro this file is the test of the correct version of that execution, on the architecture of the target
// (the check as a predicate on the types of the IOps is in utest_reduce_dpp.h). With FKL_REDUCE_CHECK_WRONG=1 to 10
// the file holds only the WRONG version, which must not compile: CMakeLists.txt of this folder builds each variant
// as a target that is not part of "all", with a ctest test that passes only if the build fails with the message of
// the check and with no other error. With FKL_REDUCE_CHECK_CPU_MT the executor of ParArch::CPU_MULTITHREAD is used
// instead of the one of the target.
#include <tests/main.h>

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/core/execution_model/reduce_dpp.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/cast.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/crop.h>
#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/execution_model/stream.h>
#ifdef FKL_REDUCE_CHECK_CPU_MT
#include <fused_kernel/core/execution_model/cpu_multithread.h>
#endif

#include <array>

#ifndef FKL_REDUCE_CHECK_WRONG
#define FKL_REDUCE_CHECK_WRONG 0
#endif

#ifdef FKL_REDUCE_CHECK_CPU_MT
constexpr fk::ParArch REDUCE_CHECK_ARCH = fk::ParArch::CPU_MULTITHREAD;
#else
constexpr fk::ParArch REDUCE_CHECK_ARCH = fk::defaultParArch;
#endif
using ReduceCheckStream = fk::Stream_<REDUCE_CHECK_ARCH>;
using ReduceCheckDPP = fk::ReduceDPP<REDUCE_CHECK_ARCH>;

constexpr int REDUCE_CHECK_W = 8;
constexpr int REDUCE_CHECK_H = 4;

#if FKL_REDUCE_CHECK_WRONG == 0
// That an executor given wrong IOps instantiates the report and nothing else is something that the WRONG variants only
// show with the compilers that go on after an error (clang, nvcc). g++ does not instantiate the function templates
// that an instantiation uses after its first error, so with it the execution that follows a failed report gives no
// second error. Here the report of three "IOps" that are not IOps is specialized to say nothing: executing them
// compiles only if the executor discards the execution, with every compiler, because any other code of ReduceDPP
// instantiated with them is an error.
struct ReduceCheckNotAnIOp {};
namespace fk {
    namespace reduce_detail {
        template <>
        struct IOpsReport<ReduceCheckNotAnIOp, ReduceCheckNotAnIOp, ReduceCheckNotAnIOp, false> {
            static constexpr bool reported = true;
        };
    } // namespace reduce_detail
} // namespace fk
#endif

#if FKL_REDUCE_CHECK_WRONG == 5
// An accumulator with a const member, and the IOps of a reduction that uses it
struct ReduceCheckConst {
    const float value = 0.f;
};
struct ReduceCheckToConst {
private:
    using SelfType = ReduceCheckToConst;
public:
    FK_STATIC_STRUCT(ReduceCheckToConst, SelfType)
    using Parent = fk::UnaryOperation<float, ReduceCheckConst, ReduceCheckToConst>;
    DECLARE_UNARY_PARENT
    FK_HOST_DEVICE_FUSE OutputType exec(const InputType input) {
        return ReduceCheckConst{input};
    }
};
struct ReduceCheckAddConst {
private:
    using SelfType = ReduceCheckAddConst;
public:
    FK_STATIC_STRUCT(ReduceCheckAddConst, SelfType)
    using Parent =
        fk::UnaryOperation<fk::Tuple<ReduceCheckConst, ReduceCheckConst>, ReduceCheckConst, ReduceCheckAddConst>;
    DECLARE_UNARY_PARENT
    FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
        return ReduceCheckConst{fk::get<0>(input).value + fk::get<1>(input).value};
    }
};
struct ReduceCheckFromConst {
private:
    using SelfType = ReduceCheckFromConst;
public:
    FK_STATIC_STRUCT(ReduceCheckFromConst, SelfType)
    using Parent = fk::UnaryOperation<ReduceCheckConst, float, ReduceCheckFromConst>;
    DECLARE_UNARY_PARENT
    FK_HOST_DEVICE_FUSE OutputType exec(const InputType& input) {
        return input.value;
    }
};
#endif

int launch() {
    ReduceCheckStream stream;
    fk::Ptr2D<float> in(REDUCE_CHECK_W, REDUCE_CHECK_H);
    fk::Ptr1D<float> out(1);
    const auto read = fk::PerThreadRead<fk::ND::_2D, float>::build(in);
    const auto write = fk::PerThreadWrite<fk::ND::_1D, float>::build(out);
#if FKL_REDUCE_CHECK_WRONG == 0
    // Wrong IOps with a silent report: nothing else is instantiated with them (see ReduceCheckNotAnIOp)
    fk::executeOperations<ReduceCheckDPP>(stream, ReduceCheckNotAnIOp{}, ReduceCheckNotAnIOp{},
                                          ReduceCheckNotAnIOp{});
    // The correct execution: the sum of the floats, accumulated in a double and written as a float
    for (int y = 0; y < REDUCE_CHECK_H; ++y) {
        for (int x = 0; x < REDUCE_CHECK_W; ++x) {
            in.at(fk::Point{x, y, 0}) = static_cast<float>(x + y * REDUCE_CHECK_W);
        }
    }
    out.at(fk::Point{0, 0, 0}) = -1.f;
    in.upload(stream);
    out.upload(stream);
    fk::executeOperations<ReduceCheckDPP>(stream, read,
        fk::make_tuple(fk::make_tuple(fk::Cast<float, double>::build(),
                                      fk::Add<double, double, double, fk::UnaryType>::build(),
                                      fk::Cast<double, float>::build())),
        write);
    out.download(stream);
    stream.sync();
    constexpr int ELEMENTS = REDUCE_CHECK_W * REDUCE_CHECK_H;
    return out.at(fk::Point{0, 0, 0}) == static_cast<float>(ELEMENTS * (ELEMENTS - 1) / 2) ? 0 : -1;
#elif FKL_REDUCE_CHECK_WRONG == 1
    // A Write where the AccumulationIOp goes
    fk::executeOperations<ReduceCheckDPP>(stream, read,
        fk::make_tuple(fk::make_tuple(fk::Cast<float, float>::build(), write, fk::Cast<float, float>::build())),
        write);
#elif FKL_REDUCE_CHECK_WRONG == 2
    // The FirstIOp gives a float and the accumulator is a double: the Cast<float, double> is missing
    fk::executeOperations<ReduceCheckDPP>(stream, read,
        fk::make_tuple(fk::make_tuple(fk::Cast<float, float>::build(),
                                      fk::Add<double, double, double, fk::UnaryType>::build(),
                                      fk::Cast<double, float>::build())),
        write);
#elif FKL_REDUCE_CHECK_WRONG == 3
    // One reduction that is not inside the Tuple of the reductions
    fk::executeOperations<ReduceCheckDPP>(stream, read,
        fk::make_tuple(fk::Cast<float, float>::build(), fk::Add<float, float, float, fk::UnaryType>::build(),
                       fk::Cast<float, float>::build()),
        write);
#elif FKL_REDUCE_CHECK_WRONG == 4
    // The FinalIOp gives a float3 and the write takes a float
    fk::executeOperations<ReduceCheckDPP>(stream, read,
        fk::make_tuple(fk::make_tuple(fk::Cast<float, float>::build(),
                                      fk::Add<float, float, float, fk::UnaryType>::build(),
                                      fk::Cast<float, fk::float3>::build())),
        write);
#elif FKL_REDUCE_CHECK_WRONG == 5
    // An accumulator with a const member: it can not be assigned, so it can not be stored
    fk::executeOperations<ReduceCheckDPP>(stream, read,
        fk::make_tuple(fk::make_tuple(ReduceCheckToConst::build(), ReduceCheckAddConst::build(),
                                      ReduceCheckFromConst::build())),
        write);
#elif FKL_REDUCE_CHECK_WRONG == 6
    // A batch of Crops without the read that they crop: a ReadType IOp that is not a complete read
    fk::executeOperations<ReduceCheckDPP>(stream,
        fk::Crop<>::build(std::array<fk::Rect, 2>{fk::Rect(0, 0, 4, 2), fk::Rect(2, 1, 4, 2)}),
        fk::make_tuple(fk::make_tuple(fk::Cast<float, float>::build(),
                                      fk::Add<float, float, float, fk::UnaryType>::build(),
                                      fk::Cast<float, float>::build())),
        write);
#elif FKL_REDUCE_CHECK_WRONG == 7
    // The read gives a float and the FirstIOp takes a float3
    fk::executeOperations<ReduceCheckDPP>(stream, read,
        fk::make_tuple(fk::make_tuple(fk::Cast<fk::float3, float>::build(),
                                      fk::Add<float, float, float, fk::UnaryType>::build(),
                                      fk::Cast<float, float>::build())),
        write);
#elif FKL_REDUCE_CHECK_WRONG == 8
    // An AccumulationIOp that takes one value, not a Tuple of two accumulated values
    fk::executeOperations<ReduceCheckDPP>(stream, read,
        fk::make_tuple(fk::make_tuple(fk::Cast<float, float>::build(), fk::Cast<float, float>::build(),
                                      fk::Cast<float, float>::build())),
        write);
#elif FKL_REDUCE_CHECK_WRONG == 9
    // The accumulator is a float and the FinalIOp takes a float3
    fk::executeOperations<ReduceCheckDPP>(stream, read,
        fk::make_tuple(fk::make_tuple(fk::Cast<float, float>::build(),
                                      fk::Add<float, float, float, fk::UnaryType>::build(),
                                      fk::Cast<fk::float3, float>::build())),
        write);
#elif FKL_REDUCE_CHECK_WRONG == 10
    // A compute IOp where the write goes
    fk::executeOperations<ReduceCheckDPP>(stream, read,
        fk::make_tuple(fk::make_tuple(fk::Cast<float, float>::build(),
                                      fk::Add<float, float, float, fk::UnaryType>::build(),
                                      fk::Cast<float, float>::build())),
        fk::Cast<float, float>::build());
#else
#error "FKL_REDUCE_CHECK_WRONG must be from 0 to 10"
#endif
#if FKL_REDUCE_CHECK_WRONG != 0
    stream.sync();
    return 0;
#endif
}
