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

// The check of the chain that the executors of TransformDPP make
// (include/fused_kernel/core/execution_model/operation_model/chain_check.h). The mistakes it catches:
//   1. the output type of an operation does not convert to the input type of the next one (Mul<float3> then Add<float>)
//   2. a complete Read that is not the first operation of the chain
//   3. batch operations of different BATCH size (4 images read, 5 written)
//   4. as 1, after a MidWrite (it gives its input to the next operation)
//
// Without a macro this file is the test: the check as a predicate on the types of the chains (the correct and the
// wrong version of each, and chains that have to stay valid), and correct chains executed and compared with the
// expected values, on the architecture of the target. With FKL_CHAIN_CHECK_WRONG=1, 2, 3 or 4 the file holds only the
// WRONG version of that chain, which must not compile: CMakeLists.txt of this folder builds each variant as a target
// that is not part of "all", with a ctest test that passes only if the build fails with the message of the check and
// with no other error. With FKL_CHAIN_CHECK_CPU_MT the executor of ParArch::CPU_MULTITHREAD is used instead of the one
// of the target.
#include <tests/main.h>

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/logical.h>
#include <fused_kernel/algorithms/basic_ops/vector_ops.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/crop.h>
#include <fused_kernel/core/data/ptr_utils.h>
#include <fused_kernel/core/execution_model/executors.h>
#include <fused_kernel/core/execution_model/operation_model/chain_check.h>

#include <array>
#include <iostream>

#ifndef FKL_CHAIN_CHECK_WRONG
#define FKL_CHAIN_CHECK_WRONG 0
#endif

#ifdef FKL_CHAIN_CHECK_CPU_MT
constexpr fk::ParArch CHAIN_CHECK_ARCH = fk::ParArch::CPU_MULTITHREAD;
#else
constexpr fk::ParArch CHAIN_CHECK_ARCH = fk::defaultParArch;
#endif
using ChainCheckStream = fk::Stream_<CHAIN_CHECK_ARCH>;
using ChainCheckDPP = fk::TransformDPP<CHAIN_CHECK_ARCH>;

constexpr int CHAIN_CHECK_W = 8;
constexpr int CHAIN_CHECK_H = 4;
constexpr size_t CHAIN_CHECK_BATCH = 4;

#if FKL_CHAIN_CHECK_WRONG == 0

// The check as a predicate on the types of the chains
namespace chain_check_types {
    using ReadF3 = decltype(fk::PerThreadRead<fk::ND::_2D, fk::float3>::build(std::declval<fk::Ptr2D<fk::float3>&>()));
    using ReadF = decltype(fk::PerThreadRead<fk::ND::_2D, float>::build(std::declval<fk::Ptr2D<float>&>()));
    using ReadI3 = decltype(fk::PerThreadRead<fk::ND::_2D, fk::int3>::build(std::declval<fk::Ptr2D<fk::int3>&>()));
    using MulF3 = decltype(fk::Mul<fk::float3>::build(fk::float3{}));
    using MulF = decltype(fk::Mul<float>::build(1.f));
    using AddF3 = decltype(fk::Add<fk::float3>::build(fk::float3{}));
    using AddF = decltype(fk::Add<float>::build(1.f));
    using WriteF3 =
        decltype(fk::PerThreadWrite<fk::ND::_2D, fk::float3>::build(std::declval<fk::Ptr2D<fk::float3>&>()));
    using WriteF = decltype(fk::PerThreadWrite<fk::ND::_2D, float>::build(std::declval<fk::Ptr2D<float>&>()));
    using WriteB = decltype(fk::PerThreadWrite<fk::ND::_2D, bool>::build(std::declval<fk::Ptr2D<bool>&>()));

    template <size_t N>
    using BatchPtrs = std::array<fk::Ptr2D<float>, N>;
    template <size_t N>
    using BatchReadN = decltype(fk::PerThreadRead<fk::ND::_2D, float>::build(std::declval<BatchPtrs<N>&>()));
    template <size_t N>
    using BatchWriteN = decltype(fk::PerThreadWrite<fk::ND::_2D, float>::build(std::declval<BatchPtrs<N>&>()));

    // 1. the output type of an operation converts to the input type of the next one
    static_assert(fk::chainIsValid<ReadF3, MulF3, AddF3, WriteF3>, "float3 chain: all the types match");
    static_assert(!fk::chainIsValid<ReadF3, MulF3, AddF, WriteF>, "a float3 into an Add<float> must be rejected");
    static_assert(fk::chainIsValid<ReadF, MulF, AddF, WriteF>, "float chain: all the types match");
    // 2. only the first operation can be a complete Read
    static_assert(!fk::chainIsValid<ReadF, MulF, ReadF, AddF, WriteF>,
                  "a complete Read in the middle must be rejected");
    // 3. the batch operations have the same BATCH
    static_assert(fk::chainIsValid<BatchReadN<CHAIN_CHECK_BATCH>, MulF, BatchWriteN<CHAIN_CHECK_BATCH>>,
                  "a batch of 4 read and 4 written is correct");
    static_assert(!fk::chainIsValid<BatchReadN<CHAIN_CHECK_BATCH>, MulF, BatchWriteN<CHAIN_CHECK_BATCH + 1>>,
                  "a batch of 4 read and 5 written must be rejected");
    static_assert(!fk::chainIsValid<BatchReadN<CHAIN_CHECK_BATCH + 1>, MulF, BatchWriteN<CHAIN_CHECK_BATCH>>,
                  "a batch of 5 read and 4 written must be rejected");
    // The IOps of the executors are plain types or const references to them: both give the same answer
    static_assert(fk::chainIsValid<const ReadF3&, const MulF3&, const AddF3&, const WriteF3&> ==
                      fk::chainIsValid<ReadF3, MulF3, AddF3, WriteF3>,
                  "references and plain types must give the same answer");

    // A MidWrite writes its input and gives it to the next operation: the pair after it is compared too (4)
    using MidWriteF3 = fk::MidWrite<fk::PerThreadWrite<fk::ND::_2D, fk::float3>>;
    static_assert(fk::chainIsValid<ReadF3, MidWriteF3, AddF3, WriteF3>, "a MidWrite passes its input on");
    static_assert(!fk::chainIsValid<ReadF3, MidWriteF3, AddF, WriteF>, "a float3 from a MidWrite into an Add<float>");
    static_assert(!fk::chainIsValid<ReadF, MidWriteF3, AddF3, WriteF3>, "a float into a MidWrite<float3>");

    // The declared types are the ones that count. IsEven of a vector gives a vector of bool, and has to say so
    using IsEvenI3 = decltype(fk::IsEven<fk::int3>::build());
    using VectorAndB3 = decltype(fk::VectorAnd<fk::bool3>::build());
    static_assert(std::is_same_v<fk::IsEven<fk::int3>::OutputType, fk::bool3>, "IsEven<int3> gives a bool3");
    static_assert(std::is_same_v<fk::IsEven<int>::OutputType, bool>, "IsEven<int> gives a bool");
    static_assert(fk::chainIsValid<ReadI3, IsEvenI3, VectorAndB3, WriteB>, "a bool3 into a VectorAnd<bool3>");

    // A fused read has no BATCH member, but it runs the planes of its first operation
    using FusedBatchRead4 = decltype(fk::FusedOperation<>::build(std::declval<const BatchReadN<CHAIN_CHECK_BATCH>&>(),
                                                                 std::declval<const MulF&>()));
    static_assert(fk::chainIsValid<FusedBatchRead4, BatchWriteN<CHAIN_CHECK_BATCH>>,
                  "a fused batch of 4 read and 4 written is correct");
    static_assert(!fk::chainIsValid<FusedBatchRead4, BatchWriteN<CHAIN_CHECK_BATCH + 1>>,
                  "a fused batch of 4 read and 5 written must be rejected");

    // A read built from an array of Crops is an incomplete read: it is not a complete Read in the middle of a chain
    using CropArray = decltype(fk::Crop<>::build(std::declval<const std::array<fk::Rect, 2>&>()));
    static_assert(!fk::chainIsCompleteRead<CropArray>, "a read of an array of Crops is not complete");
    static_assert(fk::chainIsCompleteRead<ReadF>, "a PerThreadRead is a complete Read");

    // BatchWrite exposes its BATCH like BatchRead. This does not make it a batch operation for the Fuser
    // (isBatchOperation also needs an Operation member, which BatchWrite does not have): what the Fuser does with it
    // is the same as before the BATCH member existed
    using WriteOp5 = fk::BatchWrite<5, fk::PerThreadWrite<fk::ND::_2D, float>>;
    static_assert(WriteOp5::BATCH == 5, "BatchWrite must expose its BATCH");
    static_assert(fk::has_batch<WriteOp5>::value, "BatchWrite must have a BATCH member");
    static_assert(!fk::isBatchOperation<WriteOp5>, "BatchWrite is not a batch operation for the Fuser");
    static_assert(fk::isBatchOperation<BatchReadN<5>::Operation>, "BatchRead is a batch operation");
} // namespace chain_check_types

inline void chain_check_fill(fk::Ptr2D<float>& ptr, const float value) {
    for (int y = 0; y < CHAIN_CHECK_H; ++y) {
        for (int x = 0; x < CHAIN_CHECK_W; ++x) {
            ptr.at(x, y) = value;
        }
    }
}

// 1. float3 through Mul<float3> and Add<float3>: out = in * 2 + 1
inline bool chain_check_run_types(ChainCheckStream& stream) {
    fk::Ptr2D<fk::float3> in(CHAIN_CHECK_W, CHAIN_CHECK_H);
    fk::Ptr2D<fk::float3> out(CHAIN_CHECK_W, CHAIN_CHECK_H);
    for (int y = 0; y < CHAIN_CHECK_H; ++y) {
        for (int x = 0; x < CHAIN_CHECK_W; ++x) {
            in.at(x, y) = fk::make_<fk::float3>(static_cast<float>(x), static_cast<float>(y), 3.f);
        }
    }
    in.upload(stream);
    fk::executeOperations<ChainCheckDPP>(stream,
        fk::PerThreadRead<fk::ND::_2D, fk::float3>::build(in),
        fk::Mul<fk::float3>::build(fk::make_set<fk::float3>(2.f)),
        fk::Add<fk::float3>::build(fk::make_set<fk::float3>(1.f)),
        fk::PerThreadWrite<fk::ND::_2D, fk::float3>::build(out));
    out.download(stream);
    stream.sync();
    bool correct{true};
    for (int y = 0; y < CHAIN_CHECK_H; ++y) {
        for (int x = 0; x < CHAIN_CHECK_W; ++x) {
            const fk::float3 v = out.at(x, y);
            correct = correct && v.x == 2.f * static_cast<float>(x) + 1.f &&
                      v.y == 2.f * static_cast<float>(y) + 1.f && v.z == 7.f;
        }
    }
    if (!correct) {
        std::cout << "chain 1 (types): wrong values" << std::endl;
    }
    return correct;
}

// 2. float through Mul<float> and Add<float>, a chain with only one Read: out = in * 2 + 1
inline bool chain_check_run_one_read(ChainCheckStream& stream) {
    fk::Ptr2D<float> in(CHAIN_CHECK_W, CHAIN_CHECK_H);
    fk::Ptr2D<float> out(CHAIN_CHECK_W, CHAIN_CHECK_H);
    chain_check_fill(in, 5.f);
    in.upload(stream);
    fk::executeOperations<ChainCheckDPP>(stream,
        fk::PerThreadRead<fk::ND::_2D, float>::build(in),
        fk::Mul<float>::build(2.f),
        fk::Add<float>::build(1.f),
        fk::PerThreadWrite<fk::ND::_2D, float>::build(out));
    out.download(stream);
    stream.sync();
    bool correct{true};
    for (int y = 0; y < CHAIN_CHECK_H; ++y) {
        for (int x = 0; x < CHAIN_CHECK_W; ++x) {
            correct = correct && out.at(x, y) == 11.f;
        }
    }
    if (!correct) {
        std::cout << "chain 2 (one Read): wrong values" << std::endl;
    }
    return correct;
}

// 3. a batch of 4 images read and 4 written: every output image is its input image * 2
inline bool chain_check_run_batch(ChainCheckStream& stream) {
    std::array<fk::Ptr2D<float>, CHAIN_CHECK_BATCH> ins;
    std::array<fk::Ptr2D<float>, CHAIN_CHECK_BATCH> outs;
    for (size_t i = 0; i < CHAIN_CHECK_BATCH; ++i) {
        ins[i] = fk::Ptr2D<float>(CHAIN_CHECK_W, CHAIN_CHECK_H);
        outs[i] = fk::Ptr2D<float>(CHAIN_CHECK_W, CHAIN_CHECK_H);
        chain_check_fill(ins[i], static_cast<float>(i + 1));
        chain_check_fill(outs[i], -1.f);
        ins[i].upload(stream);
        outs[i].upload(stream);
    }
    fk::executeOperations<ChainCheckDPP>(stream,
        fk::PerThreadRead<fk::ND::_2D, float>::build(ins),
        fk::Mul<float>::build(2.f),
        fk::PerThreadWrite<fk::ND::_2D, float>::build(outs));
    for (auto& out : outs) {
        out.download(stream);
    }
    stream.sync();
    bool correct{true};
    for (size_t i = 0; i < CHAIN_CHECK_BATCH; ++i) {
        for (int y = 0; y < CHAIN_CHECK_H; ++y) {
            for (int x = 0; x < CHAIN_CHECK_W; ++x) {
                correct = correct && outs[i].at(x, y) == 2.f * static_cast<float>(i + 1);
            }
        }
    }
    if (!correct) {
        std::cout << "chain 3 (batch): wrong values" << std::endl;
    }
    return correct;
}

// 4. a MidWrite in the middle of the chain: mid = in, out = in + 10
inline bool chain_check_run_midwrite(ChainCheckStream& stream) {
    fk::Ptr2D<fk::float3> in(CHAIN_CHECK_W, CHAIN_CHECK_H);
    fk::Ptr2D<fk::float3> mid(CHAIN_CHECK_W, CHAIN_CHECK_H);
    fk::Ptr2D<fk::float3> out(CHAIN_CHECK_W, CHAIN_CHECK_H);
    for (int y = 0; y < CHAIN_CHECK_H; ++y) {
        for (int x = 0; x < CHAIN_CHECK_W; ++x) {
            in.at(x, y) = fk::make_<fk::float3>(static_cast<float>(x), static_cast<float>(y), 3.f);
        }
    }
    in.upload(stream);
    chain_check_types::MidWriteF3 midWrite;
    midWrite.params = mid.ptr();
    fk::executeOperations<ChainCheckDPP>(stream,
        fk::PerThreadRead<fk::ND::_2D, fk::float3>::build(in),
        midWrite,
        fk::Add<fk::float3>::build(fk::make_set<fk::float3>(10.f)),
        fk::PerThreadWrite<fk::ND::_2D, fk::float3>::build(out));
    mid.download(stream);
    out.download(stream);
    stream.sync();
    bool correct{true};
    for (int y = 0; y < CHAIN_CHECK_H; ++y) {
        for (int x = 0; x < CHAIN_CHECK_W; ++x) {
            const fk::float3 m = mid.at(x, y);
            const fk::float3 o = out.at(x, y);
            correct = correct && m.x == static_cast<float>(x) && m.y == static_cast<float>(y) && m.z == 3.f &&
                      o.x == static_cast<float>(x) + 10.f && o.y == static_cast<float>(y) + 10.f && o.z == 13.f;
        }
    }
    if (!correct) {
        std::cout << "chain 4 (MidWrite): wrong values" << std::endl;
    }
    return correct;
}

// 5. IsEven of an int3 gives a bool3, and VectorAnd<bool3> takes it: out = (x is even) and (y is even)
inline bool chain_check_run_iseven(ChainCheckStream& stream) {
    fk::Ptr2D<fk::int3> in(CHAIN_CHECK_W, CHAIN_CHECK_H);
    fk::Ptr2D<bool> out(CHAIN_CHECK_W, CHAIN_CHECK_H);
    for (int y = 0; y < CHAIN_CHECK_H; ++y) {
        for (int x = 0; x < CHAIN_CHECK_W; ++x) {
            in.at(x, y) = fk::make_<fk::int3>(x, y, 6);
        }
    }
    in.upload(stream);
    fk::executeOperations<ChainCheckDPP>(stream,
        fk::PerThreadRead<fk::ND::_2D, fk::int3>::build(in),
        fk::IsEven<fk::int3>::build(),
        fk::VectorAnd<fk::bool3>::build(),
        fk::PerThreadWrite<fk::ND::_2D, bool>::build(out));
    out.download(stream);
    stream.sync();
    bool correct{true};
    for (int y = 0; y < CHAIN_CHECK_H; ++y) {
        for (int x = 0; x < CHAIN_CHECK_W; ++x) {
            const bool expected = (x % 2 == 0) && (y % 2 == 0);
            correct = correct && out.at(x, y) == expected;
        }
    }
    if (!correct) {
        std::cout << "chain 5 (IsEven): wrong values" << std::endl;
    }
    return correct;
}

int launch() {
    ChainCheckStream stream;
    bool correct{true};
    correct = chain_check_run_types(stream) && correct;
    correct = chain_check_run_one_read(stream) && correct;
    correct = chain_check_run_batch(stream) && correct;
    correct = chain_check_run_midwrite(stream) && correct;
    correct = chain_check_run_iseven(stream) && correct;
    return correct ? 0 : -1;
}

#else // FKL_CHAIN_CHECK_WRONG: one wrong chain, which must not compile

int launch() {
    ChainCheckStream stream;
#if FKL_CHAIN_CHECK_WRONG == 1
    // The Mul<float3> gives a float3 and the Add<float> takes a float
    fk::Ptr2D<fk::float3> in(CHAIN_CHECK_W, CHAIN_CHECK_H);
    fk::Ptr2D<float> out(CHAIN_CHECK_W, CHAIN_CHECK_H);
    fk::executeOperations<ChainCheckDPP>(stream,
        fk::PerThreadRead<fk::ND::_2D, fk::float3>::build(in),
        fk::Mul<fk::float3>::build(fk::make_set<fk::float3>(2.f)),
        fk::Add<float>::build(1.f),
        fk::PerThreadWrite<fk::ND::_2D, float>::build(out));
#elif FKL_CHAIN_CHECK_WRONG == 2
    // A complete Read in the middle of the chain
    fk::Ptr2D<float> in(CHAIN_CHECK_W, CHAIN_CHECK_H);
    fk::Ptr2D<float> in2(CHAIN_CHECK_W, CHAIN_CHECK_H);
    fk::Ptr2D<float> out(CHAIN_CHECK_W, CHAIN_CHECK_H);
    fk::executeOperations<ChainCheckDPP>(stream,
        fk::PerThreadRead<fk::ND::_2D, float>::build(in),
        fk::Mul<float>::build(2.f),
        fk::PerThreadRead<fk::ND::_2D, float>::build(in2),
        fk::Add<float>::build(1.f),
        fk::PerThreadWrite<fk::ND::_2D, float>::build(out));
#elif FKL_CHAIN_CHECK_WRONG == 3
    // 4 images read and 5 written
    std::array<fk::Ptr2D<float>, CHAIN_CHECK_BATCH> ins;
    std::array<fk::Ptr2D<float>, CHAIN_CHECK_BATCH + 1> outs;
    fk::executeOperations<ChainCheckDPP>(stream,
        fk::PerThreadRead<fk::ND::_2D, float>::build(ins),
        fk::Mul<float>::build(2.f),
        fk::PerThreadWrite<fk::ND::_2D, float>::build(outs));
#elif FKL_CHAIN_CHECK_WRONG == 4
    // The MidWrite gives a float3 to the next operation, which takes a float
    fk::Ptr2D<fk::float3> in(CHAIN_CHECK_W, CHAIN_CHECK_H);
    fk::Ptr2D<fk::float3> mid(CHAIN_CHECK_W, CHAIN_CHECK_H);
    fk::Ptr2D<float> out(CHAIN_CHECK_W, CHAIN_CHECK_H);
    fk::MidWrite<fk::PerThreadWrite<fk::ND::_2D, fk::float3>> midWrite;
    midWrite.params = mid.ptr();
    fk::executeOperations<ChainCheckDPP>(stream,
        fk::PerThreadRead<fk::ND::_2D, fk::float3>::build(in),
        midWrite,
        fk::Add<float>::build(1.f),
        fk::PerThreadWrite<fk::ND::_2D, float>::build(out));
#else
#error "FKL_CHAIN_CHECK_WRONG must be 0, 1, 2, 3 or 4"
#endif
    stream.sync();
    return 0;
}

#endif // FKL_CHAIN_CHECK_WRONG
