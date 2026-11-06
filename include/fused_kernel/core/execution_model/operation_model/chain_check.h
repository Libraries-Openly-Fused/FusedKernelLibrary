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

#ifndef FK_CHAIN_CHECK_H
#define FK_CHAIN_CHECK_H

#include <type_traits>

#include <fused_kernel/core/utils/utils.h>
#include <fused_kernel/core/utils/template_operations.h>
#include <fused_kernel/core/execution_model/operation_model/operation_types.h>
#include <fused_kernel/core/execution_model/operation_model/batch_operations.h>

namespace fk {
    /*
     * Compile time checks of the chain of IOps that the executors of TransformDPP get. They are made on the chain as
     * the executor receives it: the IncompleteReadBack IOps are already fused with what precedes them, and an IOp that
     * was fused before (then(), fuse(), FusedOperation<>::build) is one IOp, so its inner IOps are not checked. When
     * the check fails the kernel and the rest of the chain are not instantiated, so a wrong chain ends with
     * static_asserts that name the IOps involved, instead of an instantiation backtrace through the library:
     *  - the output of each IOp converts to the input of the next one (chainOutputsConvertToInputs)
     *  - no IOp after the first is a complete Read (chainHasNoLaterCompleteRead)
     *  - the batch operations of the chain have the BATCH of the first IOp, when that one is a batch operation
     *    (chainBatchSizesMatch)
     * chainIsValid<IOps...> is the check: a plain bool, without errors inside. The executors discard the chain with
     * if constexpr when it is false and use reportChain<IOps...> instead, which has one static_assert per wrong pair or
     * IOp, so that the IOps are in the instantiation context of the failed static_assert. The IOps can be references
     * (the argument pack of the executors) or plain types.
     * Only public information of the IOps is used: the InputType and OutputType of their Operation, the traits of
     * operation_types.h and the BATCH member of the batch operations.
     */

    // The traits below detect members with std::void_t used directly as the argument of a partial specialization, as
    // has_batch does. nvcc drops the std::void_t (and with it the detection) when it is nested in another template,
    // for example in std::enable_if_t<condition, std::void_t<...>>, so that form is not used

    // What an IOp gives to the next one: its OutputType, or its InputType for a MidWrite, which writes its input and
    // passes it on. Nothing to compare ("has" is false) for an IOp without OutputType (a Write), and for an IOp whose
    // OutputType is NullType (an IncompleteReadBack, or an incomplete read built from an array, like
    // Crop<>::build(std::array<Rect, N>))
    template <typename IOp, typename = void>
    struct ChainOperationOutput {
        using type = NullType;
        static constexpr bool has = false;
    };

    template <typename IOp>
    struct ChainOperationOutput<IOp, std::void_t<GetOutputType_t<IOp>>> {
        using type = GetOutputType_t<IOp>;
        static constexpr bool has = !std::is_same_v<type, NullType>;
    };

    template <typename IOp, bool IsMidWrite = opIs<MidWriteType, IOp>>
    struct ChainOutput : ChainOperationOutput<IOp> {};

    template <typename IOp>
    struct ChainOutput<IOp, true> {
        using type = GetInputType_t<IOp>;
        static constexpr bool has = true;
    };

    // What an IOp takes from the previous one: its InputType. A Read and the NullType at the end of the chain have none
    template <typename IOp, typename = void>
    struct ChainInput {
        using type = NullType;
        static constexpr bool has = false;
    };

    template <typename IOp>
    struct ChainInput<IOp, std::void_t<GetInputType_t<IOp>>> {
        using type = GetInputType_t<IOp>;
        static constexpr bool has = true;
    };

    // The output of an IOp must be convertible to the input of the next one. A pair is compared only when the previous
    // IOp gives something and the next one takes something
    template <typename PrevIOp, typename NextIOp, bool Compare = ChainOutput<PrevIOp>::has && ChainInput<NextIOp>::has>
    struct ChainLinkTypes : std::true_type {};

    template <typename PrevIOp, typename NextIOp>
    struct ChainLinkTypes<PrevIOp, NextIOp, true>
        : std::bool_constant<std::is_convertible_v<typename ChainOutput<PrevIOp>::type,
                                                   typename ChainInput<NextIOp>::type>> {};

    template <typename...>
    struct ChainPack; // only carries a pack, never defined

    template <typename PrevIOps, typename NextIOps>
    struct ChainLinks;
    template <typename... PrevIOps, typename... NextIOps>
    struct ChainLinks<ChainPack<PrevIOps...>, ChainPack<NextIOps...>> {
        static constexpr bool valid = and_v<ChainLinkTypes<PrevIOps, NextIOps>::value...>;
    };

    // The pairs (IOps[i], IOps[i + 1]) of the chain. The last IOp is paired with NullType, which has nothing to compare
    template <typename FirstIOp, typename... IOps>
    inline constexpr bool chainOutputsConvertToInputs =
        ChainLinks<ChainPack<std::decay_t<FirstIOp>, std::decay_t<IOps>...>,
                   ChainPack<std::decay_t<IOps>..., NullType>>::valid;

    // One instance per pair: the compiler names both IOps when it reports the instantiation of ChainLinkReport
    template <typename PrevIOp, typename NextIOp>
    struct ChainLinkReport {
        static_assert(ChainLinkTypes<PrevIOp, NextIOp>::value,
                      "FKL: the OutputType of the previous Op is not convertible to the InputType of the next Op");
        static constexpr bool reported = true;
    };

    template <typename PrevIOps, typename NextIOps>
    struct ChainLinksReport;
    template <typename... PrevIOps, typename... NextIOps>
    struct ChainLinksReport<ChainPack<PrevIOps...>, ChainPack<NextIOps...>> {
        static constexpr bool reported = and_v<ChainLinkReport<PrevIOps, NextIOps>::reported...>;
    };

    template <typename FirstIOp, typename... IOps>
    inline constexpr bool reportChainLinks =
        ChainLinksReport<ChainPack<std::decay_t<FirstIOp>, std::decay_t<IOps>...>,
                         ChainPack<std::decay_t<IOps>..., NullType>>::reported;

    // A complete Read (Read or ReadBack). A read built from an array of incomplete reads has no OutputType to give
    // (see ChainOutput) and is not complete
    template <typename IOp>
    inline constexpr bool chainIsCompleteRead = isAnyCompleteReadType<IOp> && ChainOutput<IOp>::has;

    // The IOps after the first one: none of them is a complete Read
    template <typename... IOps>
    inline constexpr bool chainHasNoLaterCompleteRead = and_v<(!chainIsCompleteRead<std::decay_t<IOps>>)...>;

    template <typename IOp>
    struct ChainReadReport {
        static_assert(!chainIsCompleteRead<IOp>,
                      "FKL: Complete Read Operations as continuations are not allowed. "
                      "It has to be an IncompleteReadBackOperation.");
        static constexpr bool reported = true;
    };

    // The BATCH of an Operation: the BATCH member of the batch operations (the BatchRead specializations and
    // BatchWrite). A fused operation (FusedOperation<>::build(batchRead, ...)) has no BATCH member, but the number of
    // planes it runs is the one of its first IOp
    template <typename List>
    struct ChainFirstOf;
    template <typename First, typename... Rest>
    struct ChainFirstOf<TypeList<First, Rest...>> {
        using type = std::decay_t<First>;
    };

    template <typename Op, bool HasBatch = has_batch<Op>::value>
    struct ChainBatchOp;

    template <typename Op, typename = void>
    struct ChainFusedBatchOp {
        static constexpr bool isBatch = false;
        static constexpr size_t size = 0;
    };

    template <typename Op>
    struct ChainFusedBatchOp<Op, std::void_t<typename Op::Operations>>
        : ChainBatchOp<typename ChainFirstOf<typename Op::Operations>::type::Operation> {};

    template <typename Op, bool HasBatch>
    struct ChainBatchOp : ChainFusedBatchOp<Op> {};

    template <typename Op>
    struct ChainBatchOp<Op, true> {
        static constexpr bool isBatch = true;
        static constexpr size_t size = Op::BATCH;
    };

    template <typename IOp>
    using ChainBatch = ChainBatchOp<typename IOp::Operation>;

    // Every batch operation of the chain has the BATCH of the first IOp, when that one is a batch operation
    template <typename FirstIOp, typename IOp>
    inline constexpr bool chainBatchSizeMatches = !ChainBatch<FirstIOp>::isBatch || !ChainBatch<IOp>::isBatch ||
                                                  ChainBatch<FirstIOp>::size == ChainBatch<IOp>::size;

    template <typename FirstIOp, typename... IOps>
    inline constexpr bool chainBatchSizesMatch =
        and_v<chainBatchSizeMatches<std::decay_t<FirstIOp>, std::decay_t<IOps>>...>;

    template <typename FirstIOp, typename IOp>
    struct ChainBatchReport {
        static_assert(chainBatchSizeMatches<FirstIOp, IOp>,
                      "FKL: Fusing two batch operations of different BATCH size is not allowed.");
        static constexpr bool reported = true;
    };

    // All the checks
    template <typename FirstIOp, typename... IOps>
    inline constexpr bool chainIsValid = and_v<chainOutputsConvertToInputs<FirstIOp, IOps...>,
                                               chainHasNoLaterCompleteRead<IOps...>,
                                               chainBatchSizesMatch<FirstIOp, IOps...>>;

    template <typename FirstIOp, typename... IOps>
    inline constexpr bool reportChain = and_v<reportChainLinks<FirstIOp, IOps...>,
                                              and_v<ChainReadReport<std::decay_t<IOps>>::reported...>,
                                              and_v<ChainBatchReport<std::decay_t<FirstIOp>,
                                                                     std::decay_t<IOps>>::reported...>>;
} // namespace fk

#endif
