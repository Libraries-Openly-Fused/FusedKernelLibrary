/* Copyright 2023-2025 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef IOP_FUSER_CUH
#define IOP_FUSER_CUH

#include <fused_kernel/core/execution_model/fused_operation.cuh>

namespace fk {

    struct Fuser {
        template <typename FirstIOp, typename SecondIOp>
        FK_HOST_FUSE auto fuse(const FirstIOp& firstIOp, const SecondIOp& secondIOp) {
            return fuseAllIOps(firstIOp, secondIOp);
        }

        template <typename FirstIOp, typename SecondIOp, typename... RestIOps>
        FK_HOST_FUSE auto fuse(const FirstIOp& firstIOp, const SecondIOp& secondIOp, const RestIOps&... restIOps) {
            return fuse(fuse(firstIOp, secondIOp), restIOps...);
        }
    private:
        /** @brief fuseIOps: function that creates either a Read or a Binary IOp, composed of a
        * FusedOperation, where the operations are the ones found in the InstantiableOperations in the
        * instantiableOperations parameter pack.
        * This is a convenience function to simplify the implementation of ReadBack and Ternary InstantiableOperations
        * and Operations.
        */
        template <typename... InstantiableOperations>
        FK_HOST_FUSE auto fuseNonBatchForwardIOps(const InstantiableOperations&... instantiableOperations) {
            return operationTupleToIOp(iOpsToOperationTuple(instantiableOperations...));
        }

        template <typename SelfType, typename ContinuationIOp>
        FK_HOST_FUSE auto fuseAllIOps(const SelfType& selfIOp, const ContinuationIOp& cIOp) {
            using Operation = typename SelfType::Operation;
            if constexpr (isBatchOperation<Operation> && isBatchOperation<typename ContinuationIOp::Operation>) {
                static_assert(Operation::BATCH == ContinuationIOp::Operation::BATCH,
                    "Fusing two batch operations of different BATCH size is not allowed.");
                static_assert(isReadBackType<typename ContinuationIOp::Operation::Operation>,
                    "ReadOperation as continuation is not allowed. It has to be a ReadBackOperation.");
                const auto backOpArray = BatchOperation::toArray(selfIOp);
                const auto forwardOpArray = BatchOperation::toArray(cIOp);
                using BuilderType = typename ContinuationIOp::Operation::Operation;
                if constexpr (Operation::PP == PROCESS_ALL && ContinuationIOp::Operation::PP == PROCESS_ALL) {
                    return BuilderType::build(backOpArray, forwardOpArray);
                } else if constexpr ((Operation::PP == PROCESS_ALL && ContinuationIOp::Operation::PP == CONDITIONAL_WITH_DEFAULT) ||
                    (Operation::PP == CONDITIONAL_WITH_DEFAULT && ContinuationIOp::Operation::PP == CONDITIONAL_WITH_DEFAULT)) {
                    // We assume that the output type of the forward operation does not change
                    // We will take the default value of the continuation operation
                    return BuilderType::build(cIOp.params.usedPlanes, cIOp.params.default_value, backOpArray, forwardOpArray);
                } else {
                    static_assert((Operation::PP == CONDITIONAL_WITH_DEFAULT && ContinuationIOp::Operation::PP == PROCESS_ALL), "We should not be here");
                    using BackType = std::decay_t<decltype(backOpArray)>;
                    using ForType = std::decay_t<decltype(forwardOpArray)>;
                    constexpr size_t BATCH = static_cast<size_t>(ContinuationIOp::Operation::BATCH);
                    using FusedType = typename decltype(make_fusedArray(std::declval<std::make_index_sequence<BATCH>>(), std::declval<BackType>(), std::declval<ForType>()))::value_type;
                    using DefaultValueType = typename FusedType::Operation::OutputType;
                    if constexpr (std::is_same_v<typename Operation::OutputType, DefaultValueType>) {
                        return BuilderType::build(selfIOp.params.usedPlanes, selfIOp.params.default_value, backOpArray, forwardOpArray);
                    } else {
                        using Original = typename BackType::value_type::Operation::OutputType;
                        const auto defaultValue = UnaryV<CastBase<VBase<Original>, VBase<DefaultValueType>>, Original, DefaultValueType>::exec(selfIOp.params.default_value);
                        return BuilderType::build(selfIOp.params.usedPlanes, defaultValue, backOpArray, forwardOpArray);
                    }
                }
            } else if constexpr (!isBatchOperation<Operation> && isBatchOperation<typename ContinuationIOp::Operation>) {
                static_assert(isReadBackType<typename ContinuationIOp::Operation::Operation>,
                    "ReadOperation as continuation is not allowed. It has to be a ReadBackOperation.");
                constexpr size_t BATCH = static_cast<size_t>(ContinuationIOp::Operation::BATCH);
                const auto backOpArray = make_set_std_array<BATCH>(selfIOp);
                const auto forwardOpArray = BatchOperation::toArray(cIOp);
                using BuilderType = typename ContinuationIOp::Operation::Operation;
                if constexpr (ContinuationIOp::Operation::PP == CONDITIONAL_WITH_DEFAULT) {
                    return BuilderType::build(cIOp.params.usedPlanes, cIOp.params.default_value, backOpArray, forwardOpArray);
                } else {
                    return BuilderType::build(backOpArray, forwardOpArray);
                }
            } else if constexpr (isBatchOperation<Operation> && !isBatchOperation<typename ContinuationIOp::Operation>) {
                static_assert(!isAnyReadType<ContinuationIOp> || isReadBackType<ContinuationIOp>,
                    "ReadOperation as continuation is not allowed. It has to be a ReadBackOperation.");
                constexpr size_t BATCH = static_cast<size_t>(Operation::BATCH);
                if constexpr (isReadBackType<ContinuationIOp>) {
                    const auto backOpArray = BatchOperation::toArray(selfIOp);
                    const auto forwardOpArray = make_set_std_array<BATCH>(cIOp);
                    using BuilderType = typename ContinuationIOp::Operation;
                    if constexpr (Operation::PP == CONDITIONAL_WITH_DEFAULT) {
                        using BackType = std::decay_t<decltype(backOpArray)>;
                        using ForType = std::decay_t<decltype(forwardOpArray)>;
                        using FusedType = typename decltype(make_fusedArray<BATCH>(std::declval<std::make_index_sequence<BATCH>>(), std::declval<BackType>(), std::declval<ForType>()))::value_type;
                        using DefaultValueType = typename FusedType::Operation::OutputType;
                        if constexpr (std::is_same_v<typename Operation::OutputType, DefaultValueType>) {
                            return BuilderType::build(selfIOp.params.usedPlanes, selfIOp.params.default_value, backOpArray, forwardOpArray);
                        } else {
                            using Original = typename BackType::value_type::Operation::OutputType;
                            const auto defaultValue = UnaryV<CastBase<VBase<Original>, VBase<DefaultValueType>>, Original, DefaultValueType>::exec(selfIOp.params.default_value);
                            return BuilderType::build(selfIOp.params.usedPlanes, defaultValue, backOpArray, forwardOpArray);
                        }
                    } else {
                        return BuilderType::build(backOpArray, forwardOpArray);
                    }
                } else {
                    const auto backOpArray = BatchOperation::toArray(selfIOp);
                    const auto forwardOpArray = make_set_std_array<BATCH>(cIOp);
                    const auto iOpsArray = make_fusedArray(std::make_index_sequence<BATCH>{}, backOpArray, forwardOpArray);
                    using BuilderType = typename decltype(iOpsArray)::value_type::Operation;
                    return BuilderType::build(iOpsArray);
                }
            } else if constexpr (!isBatchOperation<Operation> && !isBatchOperation<typename ContinuationIOp::Operation>) {
                static_assert(!isAnyReadType<ContinuationIOp> || isReadBackType<ContinuationIOp>,
                    "ReadOperation as continuation is not allowed. It has to be a ReadBackOperation.");
                if constexpr (isReadBackType<ContinuationIOp>) {
                    using BuilderType = typename ContinuationIOp::Operation;
                    return BuilderType::build(selfIOp, cIOp);
                } else {
                    return fuseNonBatchForwardIOps(selfIOp, cIOp);
                }
            }
        }

        template <size_t BATCH, size_t... Idx, typename BackwardIOp, typename ForwardIOp>
        FK_HOST_FUSE auto make_fusedArrayBack(const std::index_sequence<Idx...>&,
            const std::array<BackwardIOp, BATCH>& bkArray,
            const std::array<ForwardIOp, BATCH>& fwdArray) {
            using ResultingType =
                decltype(ForwardIOp::Operation::build(std::declval<BackwardIOp>(), std::declval<ForwardIOp>()));
            return std::array<ResultingType, BATCH>{ForwardIOp::Operation::build(bkArray[Idx], fwdArray[Idx])...};
        }
        template <size_t BATCH, size_t... Idx, typename ThisIOp, typename ForwardIOp>
        FK_HOST_FUSE auto make_fusedArray(const std::index_sequence<Idx...>&,
            const std::array<ThisIOp, BATCH>& thisArray,
            const std::array<ForwardIOp, BATCH>& fwdArray) {
            using ResultingType = decltype(fuseNonBatchForwardIOps(std::declval<ThisIOp>(), std::declval<ForwardIOp>()));
            return std::array<ResultingType, BATCH>{fuseNonBatchForwardIOps(thisArray[Idx], fwdArray[Idx])...};
        }
    };

    template <typename... IOps>
    FK_HOST_CNST auto fuse(const IOps&... iOps) {
        return Fuser::fuse(iOps...);
    }
} // namespace fk
#endif // IOP_FUSER_CUH