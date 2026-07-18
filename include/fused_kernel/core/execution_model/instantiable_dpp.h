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

#ifndef FK_INSTANTIABLE_DPP_H
#define FK_INSTANTIABLE_DPP_H

/* InstantiableDPP: Data Parallel Patterns as instantiable values.
 *
 * Mirrors the InstantiableOperation (IOp) design: an Operation is a stateless
 * static struct and Op::build(params) produces an IOp (Operation type + runtime
 * OperationData). Likewise, a DPP is a stateless static struct and
 * DPP::build(iOps...) produces an InstantiableDPP (DPP type + runtime details +
 * the IOps it will execute). Types define the kernel; build() values are runtime
 * parameters.
 *
 * Every DPP declares its IO API (what goes in and what comes out) as a
 * compile-time contract: a `static constexpr DPPIOSpec IO_SPEC` member stating
 * how many input (Read/ReadBack) IOps it consumes, how many output (Write) IOps
 * it produces into, whether it accepts a chain of compute IOps in between, and
 * whether it consumes whole InstantiableOperationSequences (one per divergent
 * branch) instead of loose IOps. Building or executing an InstantiableDPP
 * static_asserts conformance with that contract.
 *
 * A DPP that conforms to this model needs NO hand-written Executor
 * specialization and NO dedicated __global__ kernel: the generic execution path
 * in executors.h (fk::execute(stream, instantiableDPP)) launches it, provided
 * the DPP implements:
 *   - static constexpr ParArch PAR_ARCH             : the backend it implements.
 *   - static constexpr DPPIOSpec IO_SPEC            : the IO contract.
 *   - build_details(iOps...)                        : runtime, non-data parameters (may return an empty struct).
 *   - exec(details, iOps...)                        : the pattern itself (device code on GPU backends,
 *                                                     a plain sequential loop on the CPU backend).
 *   - getLaunchConfig(details, iOps...) (GPU only)  : grid/block sizes as a DPPLaunchConfig.
 */

#include <fused_kernel/core/utils/utils.h>
#include <fused_kernel/core/utils/type_lists.h>
#include <fused_kernel/core/data/tuple.h>
#include <fused_kernel/core/execution_model/parallel_architectures.h>
#include <fused_kernel/core/execution_model/operation_model/operation_model.h>
#include <fused_kernel/core/execution_model/executor_details/launch_config.h>

#include <type_traits>
#include <utility>

namespace fk {

    /**
     * @brief DPPIOSpec: the compile-time IO contract of a Data Parallel Pattern.
     * It formally answers "what goes into and what comes out of this DPP".
     * All data enters and leaves a DPP through IOps: inputs through complete
     * Read/ReadBack IOps and outputs through Write IOps. Never through raw pointers.
     */
    struct DPPIOSpec {
        /** Number of leading input IOps. They must be complete Read or ReadBack IOps. */
        size_t inputIOps;
        /** Number of trailing output IOps. They must be Write IOps. */
        size_t outputIOps;
        /** Whether a chain of compute (Unary/Binary/Ternary) or MidWrite IOps is
         *  accepted between the input and output IOps. */
        bool acceptsComputeIOps;
        /** Whether the DPP consumes InstantiableOperationSequences (one per divergent
         *  branch, see buildOperationSequence()) instead of loose IOps. In that case
         *  inputIOps/outputIOps/acceptsComputeIOps apply to EACH sequence. */
        bool argsAreIOpSequences;
    };

    // hasDPPIOSpec trait: detects whether a DPP declares its IO contract
    template <typename T, typename = void>
    struct HasDPPIOSpec : std::false_type {};
    template <typename T>
    struct HasDPPIOSpec<T, std::void_t<decltype(T::IO_SPEC)>>
        : std::bool_constant<std::is_same_v<std::decay_t<decltype(T::IO_SPEC)>, DPPIOSpec>> {};
    template <typename T>
    constexpr bool hasDPPIOSpec_v = HasDPPIOSpec<T>::value;

    // isIOpSequence trait
    template <typename T>
    struct IsIOpSequence : std::false_type {};
    template <typename... IOps>
    struct IsIOpSequence<InstantiableOperationSequence<IOps...>> : std::true_type {};
    template <typename T>
    constexpr bool isIOpSequence_v = IsIOpSequence<std::decay_t<T>>::value;

    namespace dpp_contract_detail {
        template <typename IOpsTL, size_t OFFSET, typename IdxSeq>
        struct AllSlotsAreInputs;
        template <typename IOpsTL, size_t OFFSET, size_t... Idx>
        struct AllSlotsAreInputs<IOpsTL, OFFSET, std::index_sequence<Idx...>> {
            static constexpr bool value = (isAnyCompleteReadType<TypeAt_t<OFFSET + Idx, IOpsTL>> && ...);
        };

        template <typename IOpsTL, size_t OFFSET, typename IdxSeq>
        struct AllSlotsAreCompute;
        template <typename IOpsTL, size_t OFFSET, size_t... Idx>
        struct AllSlotsAreCompute<IOpsTL, OFFSET, std::index_sequence<Idx...>> {
            static constexpr bool value = ((isComputeType<TypeAt_t<OFFSET + Idx, IOpsTL>> ||
                                            opIs<MidWriteType, TypeAt_t<OFFSET + Idx, IOpsTL>>) && ...);
        };

        template <typename IOpsTL, size_t OFFSET, typename IdxSeq>
        struct AllSlotsAreOutputs;
        template <typename IOpsTL, size_t OFFSET, size_t... Idx>
        struct AllSlotsAreOutputs<IOpsTL, OFFSET, std::index_sequence<Idx...>> {
            static constexpr bool value = (opIs<WriteType, TypeAt_t<OFFSET + Idx, IOpsTL>> && ...);
        };
    } // namespace dpp_contract_detail

    /**
     * @brief IOPackConformance: checks a pack of IOps against the counts and kinds
     * required by a DPPIOSpec. Exposes granular flags so that contract violations
     * can be reported with precise static_assert messages.
     */
    template <size_t INPUT_IOPS, size_t OUTPUT_IOPS, bool ACCEPTS_COMPUTE, typename... IOps>
    struct IOPackConformance {
    private:
        using IOpsTL = TypeList<IOps...>;
        static constexpr size_t N = sizeof...(IOps);
    public:
        static constexpr bool ENOUGH_IOPS = N >= INPUT_IOPS + OUTPUT_IOPS;
        static constexpr size_t COMPUTE_IOPS = ENOUGH_IOPS ? (N - INPUT_IOPS - OUTPUT_IOPS) : 0;
        static constexpr bool COMPUTE_CHAIN_OK = ACCEPTS_COMPUTE || (COMPUTE_IOPS == 0);
        static constexpr bool INPUTS_ARE_READS =
            std::conditional_t<ENOUGH_IOPS,
                dpp_contract_detail::AllSlotsAreInputs<IOpsTL, 0, std::make_index_sequence<INPUT_IOPS>>,
                std::false_type>::value;
        static constexpr bool MIDDLE_ARE_COMPUTE =
            std::conditional_t<ENOUGH_IOPS,
                dpp_contract_detail::AllSlotsAreCompute<IOpsTL, INPUT_IOPS, std::make_index_sequence<COMPUTE_IOPS>>,
                std::false_type>::value;
        static constexpr bool OUTPUTS_ARE_WRITES =
            std::conditional_t<ENOUGH_IOPS,
                dpp_contract_detail::AllSlotsAreOutputs<IOpsTL, INPUT_IOPS + COMPUTE_IOPS, std::make_index_sequence<OUTPUT_IOPS>>,
                std::false_type>::value;
        static constexpr bool CONFORMS =
            ENOUGH_IOPS && COMPUTE_CHAIN_OK && INPUTS_ARE_READS && MIDDLE_ARE_COMPUTE && OUTPUTS_ARE_WRITES;
    };

    // Per-sequence conformance (for DPPs with IO_SPEC.argsAreIOpSequences == true)
    template <typename DPP, typename Seq>
    struct IOpSequenceConformance;
    template <typename DPP, typename... IOps>
    struct IOpSequenceConformance<DPP, InstantiableOperationSequence<IOps...>> {
        using Check = IOPackConformance<DPP::IO_SPEC.inputIOps, DPP::IO_SPEC.outputIOps,
                                        DPP::IO_SPEC.acceptsComputeIOps, IOps...>;
    };

    /**
     * @brief dppIOContractSatisfied: soft (non-asserting) check of a pack of build/exec
     * arguments against the IO contract of DPP. Usable in static_asserts and SFINAE.
     */
    template <typename DPP, typename... Args>
    FK_HOST_DEVICE_CNST bool dppIOContractSatisfied() {
        if constexpr (!hasDPPIOSpec_v<DPP>) {
            return false;
        } else if constexpr (DPP::IO_SPEC.argsAreIOpSequences) {
            if constexpr (sizeof...(Args) == 0) {
                return false;
            } else if constexpr (!(isIOpSequence_v<Args> && ...)) {
                return false;
            } else {
                return (IOpSequenceConformance<DPP, std::decay_t<Args>>::Check::CONFORMS && ...);
            }
        } else {
            return IOPackConformance<DPP::IO_SPEC.inputIOps, DPP::IO_SPEC.outputIOps,
                                     DPP::IO_SPEC.acceptsComputeIOps, std::decay_t<Args>...>::CONFORMS;
        }
    }

    namespace dpp_contract_detail {
        template <typename DPP, typename... IOps>
        FK_HOST_DEVICE_CNST void assertIOPackContract() {
            using Check = IOPackConformance<DPP::IO_SPEC.inputIOps, DPP::IO_SPEC.outputIOps,
                                            DPP::IO_SPEC.acceptsComputeIOps, IOps...>;
            static_assert(Check::ENOUGH_IOPS,
                "DPP IO contract violation: not enough IOps. The DPP consumes IO_SPEC.inputIOps Read/ReadBack IOps "
                "and produces into IO_SPEC.outputIOps Write IOps.");
            if constexpr (Check::ENOUGH_IOPS) {
                static_assert(Check::INPUTS_ARE_READS,
                    "DPP IO contract violation: the first IO_SPEC.inputIOps IOps must be complete Read or ReadBack "
                    "IOps (the data going INTO the DPP).");
                static_assert(Check::COMPUTE_CHAIN_OK,
                    "DPP IO contract violation: this DPP does not accept compute IOps between its input (Read/ReadBack) "
                    "and output (Write) IOps. Fuse them into the input IOp instead (readIOp.then(computeIOp...)).");
                static_assert(Check::MIDDLE_ARE_COMPUTE,
                    "DPP IO contract violation: the IOps between the inputs and the outputs must be compute "
                    "(Unary/Binary/Ternary) or MidWrite IOps.");
                static_assert(Check::OUTPUTS_ARE_WRITES,
                    "DPP IO contract violation: the last IO_SPEC.outputIOps IOps must be Write IOps (the data coming "
                    "OUT of the DPP).");
            }
        }

        template <typename DPP, typename Seq>
        FK_HOST_DEVICE_CNST void assertIOpSequenceContract() {
            static_assert(IOpSequenceConformance<DPP, Seq>::Check::CONFORMS,
                "DPP IO contract violation: an InstantiableOperationSequence does not conform to the DPP IO contract. "
                "Each sequence must contain IO_SPEC.inputIOps complete Read/ReadBack IOps, then compute IOps (if "
                "IO_SPEC.acceptsComputeIOps), then IO_SPEC.outputIOps Write IOps.");
        }
    } // namespace dpp_contract_detail

    /**
     * @brief assertDPPIOContract: hard (static_assert) check of a pack of build/exec
     * arguments against the IO contract of DPP, with granular error messages.
     */
    template <typename DPP, typename... Args>
    FK_HOST_DEVICE_CNST void assertDPPIOContract() {
        static_assert(hasDPPIOSpec_v<DPP>,
            "InstantiableDPP: the DPP does not declare its IO contract. "
            "Add 'static constexpr DPPIOSpec IO_SPEC{ inputIOps, outputIOps, acceptsComputeIOps, argsAreIOpSequences };' "
            "to the DPP struct.");
        if constexpr (hasDPPIOSpec_v<DPP>) {
            if constexpr (DPP::IO_SPEC.argsAreIOpSequences) {
                static_assert(sizeof...(Args) > 0,
                    "DPP IO contract violation: this DPP requires at least one InstantiableOperationSequence "
                    "(see buildOperationSequence()).");
                static_assert((isIOpSequence_v<Args> && ...),
                    "DPP IO contract violation: this DPP consumes InstantiableOperationSequences "
                    "(see buildOperationSequence()), not loose IOps.");
                if constexpr (sizeof...(Args) > 0 && (isIOpSequence_v<Args> && ...)) {
                    ((void)dpp_contract_detail::assertIOpSequenceContract<DPP, std::decay_t<Args>>(), ...);
                }
            } else {
                static_assert((!isIOpSequence_v<Args> && ...),
                    "DPP IO contract violation: this DPP consumes loose IOps, not InstantiableOperationSequences. "
                    "Pass the IOps directly instead of wrapping them with buildOperationSequence().");
                if constexpr ((!isIOpSequence_v<Args> && ...)) {
                    dpp_contract_detail::assertIOPackContract<DPP, std::decay_t<Args>...>();
                }
            }
        }
    }

    /**
     * @brief InstantiableDPP: a Data Parallel Pattern bundled with the runtime values
     * it needs in order to be executed: the DPP details (runtime parameters not related
     * to the data being processed) plus the IOps (or IOpSequences) that read, transform
     * and write the data. Produced by DPP::build(iOps...). Execute it with
     * fk::execute(stream, instantiableDPP) (see executors.h / fused_kernel.h).
     */
    template <typename DPPType, typename DetailsType, typename... IOps>
    struct InstantiableDPP {
        using DPP = DPPType;
        using Details = DetailsType;
        static constexpr ParArch PAR_ARCH = DPPType::PAR_ARCH;
        Details details;
        Tuple<IOps...> iOps;
    };

    // isInstantiableDPP trait
    template <typename T>
    struct IsInstantiableDPP : std::false_type {};
    template <typename DPP, typename Details, typename... IOps>
    struct IsInstantiableDPP<InstantiableDPP<DPP, Details, IOps...>> : std::true_type {};
    template <typename T>
    constexpr bool isInstantiableDPP_v = IsInstantiableDPP<std::decay_t<T>>::value;

    namespace dpp_contract_detail {
        // Fuses the ReadBack stack (Backwards Vertical Fusion) of an IOpSequence,
        // returning a new IOpSequence in canonical (fused) form.
        template <typename... IOps>
        FK_HOST_CNST auto fuseBackIOpSequence(const InstantiableOperationSequence<IOps...>& iOpSequence) {
            return buildOperationSequence_tup(
                apply([](const auto&... iOps) {
                    return BackFuser::fuse_back(iOps...);
                }, iOpSequence.iOps));
        }

        template <typename DPP, typename... Args>
        FK_HOST_CNST auto buildInstantiableDPPFromCanonical(const Args&... args) {
            assertDPPIOContract<DPP, Args...>();
            if constexpr (dppIOContractSatisfied<DPP, Args...>()) {
                const auto details = DPP::build_details(args...);
                using Details = std::decay_t<decltype(details)>;
                return InstantiableDPP<DPP, Details, Args...>{ details, { args... } };
            } else {
                // The static_asserts above already failed: return a dummy to avoid error cascades.
                return NullType{};
            }
        }
    } // namespace dpp_contract_detail

    /**
     * @brief buildInstantiableDPP: generic implementation of DPP::build(iOps...).
     * Brings the IOps to canonical form (fusing ReadBack stacks via Backwards Vertical
     * Fusion), enforces the DPP IO contract at compile time, builds the DPP details and
     * returns the resulting InstantiableDPP value.
     */
    template <typename DPP, typename... Args>
    FK_HOST_CNST auto buildInstantiableDPP(const Args&... args) {
        constexpr bool ARGS_ARE_SEQUENCES = (sizeof...(Args) > 0) && (isIOpSequence_v<Args> && ...);
        if constexpr (ARGS_ARE_SEQUENCES) {
            return dpp_contract_detail::buildInstantiableDPPFromCanonical<DPP>(
                dpp_contract_detail::fuseBackIOpSequence(args)...);
        } else {
            const auto fusedIOps = BackFuser::fuse_back(args...);
            return apply([](const auto&... fused) {
                return dpp_contract_detail::buildInstantiableDPPFromCanonical<DPP>(fused...);
            }, fusedIOps);
        }
    }

} // namespace fk

#endif // FK_INSTANTIABLE_DPP_H
