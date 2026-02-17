/* Copyright 2023-2026 Oscar Amoros Huguet

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

#include <fused_kernel/core/execution_model/operation_model/fused_operation.h>

namespace fk {

    struct Fuser {
        template <typename SelfType, typename ContinuationIOp>
        FK_HOST_FUSE auto fuse(const SelfType& selfIOp, const ContinuationIOp& cIOp) {
            static_assert(isOperation<SelfType> && isOperation<ContinuationIOp>, "SelfType and ContinuationIOp should be IOps");
            using Operation = typename SelfType::Operation;
            if constexpr (isBatchOperation<Operation> && isBatchOperation<typename ContinuationIOp::Operation>) {
                static_assert(Operation::BATCH == ContinuationIOp::Operation::BATCH,
                    "Fusing two batch operations of different BATCH size is not allowed.");
                static_assert(opIs<IncompleteReadBackType, typename ContinuationIOp::Operation::Operation>,
                    "Read or ReadBack Operation as continuation is not allowed. It has to be an IncompleteReadBackOperation.");
                const auto backOpArray = BatchUtils::toArray(selfIOp);
                const auto forwardOpArray = BatchUtils::toArray(cIOp);
                using BuilderType = typename ContinuationIOp::Operation::Operation;
                if constexpr (Operation::PP == PlanePolicy::PROCESS_ALL && ContinuationIOp::Operation::PP == PlanePolicy::PROCESS_ALL) {
                    return BuilderType::build(backOpArray, forwardOpArray);
                } else if constexpr ((Operation::PP == PlanePolicy::PROCESS_ALL && ContinuationIOp::Operation::PP == PlanePolicy::CONDITIONAL_WITH_DEFAULT) ||
                    (Operation::PP == PlanePolicy::CONDITIONAL_WITH_DEFAULT && ContinuationIOp::Operation::PP == PlanePolicy::CONDITIONAL_WITH_DEFAULT)) {
                    // We assume that the output type of the forward operation does not change
                    // We will take the default value of the continuation operation
                    return BuilderType::build(cIOp.params.usedPlanes, cIOp.params.default_value, backOpArray, forwardOpArray);
                } else {
                    static_assert((Operation::PP == PlanePolicy::CONDITIONAL_WITH_DEFAULT && ContinuationIOp::Operation::PP == PlanePolicy::PROCESS_ALL), "We should not be here");
                    using BackType = std::decay_t<decltype(backOpArray)>;
                    using ForType = std::decay_t<decltype(forwardOpArray)>;
                    using FusedType = typename decltype(make_fusedArray(std::declval<BackType>(), std::declval<ForType>()))::value_type;
                    using DefaultValueType = typename FusedType::Operation::OutputType;
                    if constexpr (std::is_same_v<typename Operation::OutputType, DefaultValueType>) {
                        return BuilderType::build(selfIOp.params.usedPlanes, selfIOp.params.default_value, backOpArray, forwardOpArray);
                    } else {
                        using Original = typename BackType::value_type::Operation::OutputType;
                        const auto defaultValue = cxp::cast<DefaultValueType>::f(selfIOp.params.default_value);
                        return BuilderType::build(selfIOp.params.usedPlanes, defaultValue, backOpArray, forwardOpArray);
                    }
                }
            } else if constexpr (!isBatchOperation<Operation> && isBatchOperation<typename ContinuationIOp::Operation>) {
                static_assert(opIs<IncompleteReadBackType, typename ContinuationIOp::Operation::Operation>,
                    "ReadOperation as continuation is not allowed. It has to be a ReadBackOperation.");
                constexpr size_t BATCH = static_cast<size_t>(ContinuationIOp::Operation::BATCH);
                const auto backOpArray = make_set_std_array<BATCH>(selfIOp);
                const auto forwardOpArray = BatchUtils::toArray(cIOp);
                using BuilderType = typename ContinuationIOp::Operation::Operation;
                if constexpr (ContinuationIOp::Operation::PP == PlanePolicy::CONDITIONAL_WITH_DEFAULT) {
                    return BuilderType::build(cIOp.params.usedPlanes, cIOp.params.default_value, backOpArray, forwardOpArray);
                } else {
                    return BuilderType::build(backOpArray, forwardOpArray);
                }
            } else if constexpr (isBatchOperation<Operation> && !isBatchOperation<typename ContinuationIOp::Operation>) {
                static_assert(!isAnyReadType<ContinuationIOp> || opIs<IncompleteReadBackType, ContinuationIOp>,
                    "ReadOperation as continuation is not allowed. It has to be a ReadBackOperation.");
                constexpr size_t BATCH = static_cast<size_t>(Operation::BATCH);
                if constexpr (opIs<IncompleteReadBackType, ContinuationIOp>) {
                    const auto backOpArray = BatchUtils::toArray(selfIOp);
                    const auto forwardOpArray = make_set_std_array<BATCH>(cIOp);
                    using BuilderType = typename ContinuationIOp::Operation;
                    if constexpr (Operation::PP == PlanePolicy::CONDITIONAL_WITH_DEFAULT) {
                        using BackType = std::decay_t<decltype(backOpArray)>;
                        using ForType = std::decay_t<decltype(forwardOpArray)>;
                        using FusedType = typename decltype(make_fusedArray<BATCH>(std::declval<BackType>(), std::declval<ForType>()))::value_type;
                        using DefaultValueType = typename FusedType::Operation::OutputType;
                        if constexpr (std::is_same_v<typename Operation::OutputType, DefaultValueType>) {
                            return BuilderType::build(selfIOp.params.usedPlanes, selfIOp.params.default_value, backOpArray, forwardOpArray);
                        } else {
                            using Original = typename BackType::value_type::Operation::OutputType;
                            const auto defaultValue = cxp::cast<DefaultValueType>::f(selfIOp.params.default_value);
                            return BuilderType::build(selfIOp.params.usedPlanes, defaultValue, backOpArray, forwardOpArray);
                        }
                    } else {
                        return BuilderType::build(backOpArray, forwardOpArray);
                    }
                } else {
                    const auto backOpArray = BatchUtils::toArray(selfIOp);
                    const auto forwardOpArray = make_set_std_array<BATCH>(cIOp);
                    const auto iOpsArray = make_fusedArray(backOpArray, forwardOpArray);
                    using BuilderType = typename decltype(iOpsArray)::value_type::Operation;
                    return BuilderType::build(iOpsArray);
                }
            } else if constexpr (!isBatchOperation<Operation> && !isBatchOperation<typename ContinuationIOp::Operation>) {
                static_assert(!isAnyCompleteReadType<ContinuationIOp>,
                    "Complete Read Operations as continuations are not allowed. It has to be an IncompleteReadBackOperation.");
                if constexpr (opIs<IncompleteReadBackType, ContinuationIOp>) {
                    using BuilderType = typename ContinuationIOp::Operation;
                    return BuilderType::build(selfIOp, cIOp);
                } else {
                    return FusedOperation<>::build(selfIOp, cIOp);
                }
            }
        }
        private:
        template <size_t BATCH, typename ThisIOp, typename ForwardIOp>
        FK_HOST_FUSE auto make_fusedArray(const std::array<ThisIOp, BATCH>& thisArray,
                                          const std::array<ForwardIOp, BATCH>& fwdArray) {
            if constexpr (opIs<IncompleteReadBackType, ForwardIOp>) {
                using BuilderType = typename ForwardIOp::Operation;
                using ResultingType = decltype(BuilderType::build(std::declval<ThisIOp>(), std::declval<ForwardIOp>()));
                std::array<ResultingType, BATCH> resultArray{};
                for (size_t i = 0; i < BATCH; ++i) {
                    resultArray[i] = BuilderType::build(thisArray[i], fwdArray[i]);
                }
                return resultArray;
            } else {
                using ResultingType = decltype(FusedOperation<>::build(std::declval<ThisIOp>(), std::declval<ForwardIOp>()));
                std::array<ResultingType, BATCH> resultArray{};
                for (size_t i = 0; i < BATCH; ++i) {
                    resultArray[i] = FusedOperation<>::build(thisArray[i], fwdArray[i]);
                }
                return resultArray;
            }
        }
    };

    template <typename FirstIOp, typename... IOps>
    FK_HOST_CNST decltype(auto) fuse(FirstIOp&& firstIOp, IOps&&... iOps) {
        return (std::forward<FirstIOp>(firstIOp) & ... & std::forward<IOps>(iOps));
    }

    struct BackFuser {
      protected:
        template <typename... IOps>
        FK_HOST_FUSE size_t idxFirstNonBack() {
            size_t index = 0;
            size_t result = 0;

            // Iterate through every type in Ts...
            // The comma operator ensures left-to-right evaluation.
            ((index++, // Increment index for every type (1-based)
              (opIs<ReadBackType, IOps> || opIs<IncompleteReadBackType, IOps>) 
                  ? result = index // If it's a Back type, update the result
                  : 0              // Otherwise do nothing
             ), ...);

            return result;
        }

        template <typename T, size_t... Is>
        FK_HOST_FUSE auto get_head(T&& t, const std::index_sequence<Is...>&) {
            // Forward the specific elements to your function
            return fuse(get<Is>(std::forward<T>(t))...);
        }

        // Helper to unpack the Tail (indices split_idx to End)
        // Creates a new tuple starting from Offset
        template <size_t Offset, typename T, size_t... Is>
        FK_HOST_FUSE auto get_tail(T&& t, const std::index_sequence<Is...>&) {
            // Apply Offset to grab the correct elements from the end of the tuple
            return make_tuple(get<Offset + Is>(std::forward<T>(t))...);
        }

      public:
        template <typename... IOps>
        FK_HOST_FUSE auto fuse_back(IOps&&... iOps) {
            // 1. Calculate the split point at compile time
            constexpr size_t split_idx = idxFirstNonBack<std::decay_t<IOps>...>();
            if constexpr (split_idx < 2) {
                return forward_as_tuple(std::forward<IOps>(iOps)...);
            } else {
                constexpr size_t total_size = sizeof...(IOps);

                // 2. Pack arguments into a tuple so we can access them by index
                //    Using a reference tuple prevents unnecessary copies.
                auto full_tuple = forward_as_tuple(std::forward<IOps>(iOps)...);

                // 3. Execute the split
                //    Head: sequence 0..split_idx
                //    Tail: sequence 0..(total - split_idx)
                auto new_element = get_head(full_tuple, std::make_index_sequence<split_idx>{});
                auto tail_tuple = get_tail<split_idx>(full_tuple, std::make_index_sequence<total_size - split_idx>{});

                // 4. Concatenate the result
                return tuple_cat(make_tuple(new_element), tail_tuple);
            }
        }
    };
} // namespace fk
#endif // IOP_FUSER_CUH