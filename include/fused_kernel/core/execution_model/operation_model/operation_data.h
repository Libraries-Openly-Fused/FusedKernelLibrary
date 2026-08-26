/* Copyright 2025-2026 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_OPERATION_DATA_H
#define FK_OPERATION_DATA_H

#include <type_traits>
#include <fused_kernel/core/execution_model/operation_model/operation_types.h>

namespace fk {
    //Operation type traits
    // hasParams trait
    template <typename, typename = std::void_t<>>
    struct hasParams : std::false_type {};

    template <typename T>
    struct hasParams<T, std::void_t<typename T::ParamsType>> : std::true_type {};

    template <typename T>
    constexpr bool hasParams_v = hasParams<T>::value;

    // Primary template: assumes T does not have member 'next'
    template <typename, typename = std::void_t<>>
    struct has_next : std::false_type {};

    // Specialized template: this will be chosen if T has member 'next'
    template <typename T>
    struct has_next<T, std::void_t<decltype(std::declval<T>().next)>> : std::true_type {};

    // Helper variable template for easier usage
    template <typename T>
    constexpr bool has_next_v = has_next<T>::value;

    using BIOpList = TypeList<ReadBackType, TernaryType, IncompleteReadBackType>;
    template <typename OpOrDF>
    constexpr bool hasNoBackIOp_v = !one_of_v<typename OpOrDF::InstanceType, BIOpList>;

    // hasBackIOp trait
    template <typename, typename = std::void_t<>>
    struct hasBackIOp : std::false_type {};

    template <typename T>
    struct hasBackIOp<T, std::void_t<typename T::BackIOp>> : std::true_type {};

    template <typename T>
    constexpr bool hasBackIOp_v = hasBackIOp<T>::value;

    // hasParamsAndBackIOp trait
    template <typename, typename = std::void_t<>>
    struct hasParamsAndBackIOp : std::false_type {};

    template <typename T>
    struct hasParamsAndBackIOp<T, std::void_t<typename T::ParamsType,
        typename T::BackIOp>> : std::true_type {};

    template <typename T>
    constexpr bool hasParamsAndBackIOp_v = hasParamsAndBackIOp<T>::value;

    // OperationData implementation selectors
    template <typename Operation>
    constexpr bool hasParamsNoArray =
        hasParams_v<Operation> && !std::is_array_v<typename Operation::ParamsType>;
    template <typename Operation>
    constexpr bool hasParamsArray =
        hasParams_v<Operation> && std::is_array_v<typename Operation::ParamsType>;

    template <typename Operation>
    constexpr bool hasParamsAndBackIOpNoArray =
        hasParamsAndBackIOp_v<Operation> && !std::is_array_v<typename Operation::ParamsType> &&
        !std::is_array_v<typename Operation::BackIOp>;

    template <typename Operation>
    constexpr bool hasParamsAndBackIOpArray =
        hasParamsAndBackIOp_v<Operation> && std::is_array_v<typename Operation::ParamsType> &&
        std::is_array_v<typename Operation::BackIOp>;

    // OperationData implementations
    template <typename Operation>
    struct OperationData;

    using ParamsTypes = TypeList<BinaryType, ReadType, WriteType, MidWriteType, OpenType, ClosedType>;
    using ParamsAndBackIOpTypes = TypeList<ReadBackType, IncompleteReadBackType, TernaryType>;

    template <typename Operation>
    requires (one_of_v<typename Operation::InstanceType, ParamsTypes>)
    struct OperationData<Operation> {
        FK_HOST_DEVICE_CNST OperationData() {};
        FK_HOST_DEVICE_CNST OperationData(const typename Operation::ParamsType &params_)
            requires(hasParamsNoArray<Operation>) : params(params_) {}
        FK_HOST_DEVICE_CNST OperationData(const typename Operation::ParamsType& params_)
            requires(hasParamsArray<Operation>) : params{} {
            for (size_t i = 0; i < std::extent_v<typename Operation::ParamsType>; ++i) {
                params[i] = params_[i];
            }
        }
        typename Operation::ParamsType params{};
    };

    template <typename Operation>
    requires (one_of_v<typename Operation::InstanceType, ParamsAndBackIOpTypes>)
    struct OperationData<Operation> {
        FK_HOST_DEVICE_CNST OperationData() {};
        FK_HOST_DEVICE_CNST OperationData(const typename Operation::ParamsType& params_, const typename Operation::BackIOp& backIOp_)
            requires (hasParamsAndBackIOpNoArray<Operation>) : params(params_), backIOp(backIOp_) {}
        FK_HOST_DEVICE_CNST OperationData(const typename Operation::ParamsType& params_, const typename Operation::BackIOp& backIOp_)
            requires (hasParamsAndBackIOpArray<Operation>) : params{}, backIOp{} {
            for (size_t i = 0; i < std::extent_v<typename Operation::ParamsType>; ++i) {
                params[i] = params_[i];
            }
            for (size_t i = 0; i < std::extent_v<typename Operation::BackIOp>; ++i) {
                backIOp[i] = backIOp_[i];
            }
        }
        typename Operation::ParamsType params{};
        typename Operation::BackIOp backIOp{};
    };
} // namespace fk

#endif // FK_OPERATION_DATA_H