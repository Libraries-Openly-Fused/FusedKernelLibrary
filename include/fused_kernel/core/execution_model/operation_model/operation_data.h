/* Copyright 2025 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_OPERATION_DATA_CUH
#define FK_OPERATION_DATA_CUH

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

    // OperationData implementations
    template <typename Operation, typename Enabler = void>
    struct OperationData;

    template <typename Operation>
    struct OperationData<Operation, std::enable_if_t<hasParamsNoArray<Operation>&& hasNoBackIOp_v<Operation>, void>> {
#ifdef COPYABLE_IOPS
        FK_HOST_DEVICE_CNST OperationData() {};
        FK_HOST_DEVICE_CNST OperationData(const typename Operation::ParamsType& params_) : params(params_) {}
#endif
        typename Operation::ParamsType params;
    };

    template <typename Operation>
    struct OperationData<Operation, std::enable_if_t<hasParamsArray<Operation>&& hasNoBackIOp_v<Operation>, void>> {
#ifdef COPYABLE_IOPS
        FK_HOST_DEVICE_CNST OperationData() {};
        __host__ __forceinline__ OperationData(const typename Operation::ParamsType& params_) {
            std::copy(std::begin(params_), std::end(params_), std::begin(params));
        }
        __host__ __forceinline__ OperationData<Operation>& operator=(const OperationData<Operation>& other) {
            if (this != &other) {
                std::copy(std::begin(other.params), std::end(other.params), std::begin(params));
            }
            return *this;
        }
#endif
        typename Operation::ParamsType params;
    };

    template <typename Operation>
    struct OperationData<Operation, std::enable_if_t<hasParamsAndBackIOp_v<Operation> &&
        !std::is_array_v<typename Operation::ParamsType> &&
        !std::is_array_v<typename Operation::BackIOp>, void>> {
#ifdef COPYABLE_IOPS
        FK_HOST_DEVICE_CNST OperationData() {};
        FK_HOST_DEVICE_CNST OperationData(const typename Operation::ParamsType& params_,
            const typename Operation::BackIOp& backIOp_) :
            params(params_), backIOp(backIOp_) {}
#endif
        typename Operation::ParamsType params;
        typename Operation::BackIOp backIOp;
    };

    template <typename Operation>
    struct OperationData<Operation, std::enable_if_t<hasParamsAndBackIOp_v<Operation> &&
        (std::is_array_v<typename Operation::ParamsType> ||
            std::is_array_v<typename Operation::BackIOp>), void>> {
#ifdef COPYABLE_IOPS
        __host__ __forceinline__ OperationData() {};
        __host__ __forceinline__ OperationData(const typename Operation::ParamsType& params_,
            const typename Operation::BackIOp& backIOp_) {
            if constexpr (std::is_array_v<typename Operation::ParamsType>) {
                std::copy(std::begin(params_), std::end(params_), std::begin(params));
            } else {
                params = params_;
            }
            if constexpr (std::is_array_v<typename Operation::BackIOp>) {
                std::copy(std::begin(backIOp_), std::end(backIOp_), std::begin(backIOp));
            } else {
                backIOp = backIOp_;
            }
        }
        __host__ __forceinline__ OperationData<Operation>& operator=(const OperationData<Operation>& other) {
            if (this != &other) {
                if constexpr (std::is_array_v<typename Operation::ParamsType>) {
                    std::copy(std::begin(other.params), std::end(other.params), std::begin(params));
                } else {
                    params = other.params;
                }
                if constexpr (std::is_array_v<typename Operation::BackIOp>) {
                    std::copy(std::begin(other.backIOp), std::end(other.backIOp), std::begin(backIOp));
                } else {
                    backIOp = other.backIOp;
                }
            }
            return *this;
        }
#endif
        typename Operation::ParamsType params;
        typename Operation::BackIOp backIOp;
    };
} // namespace fk

#endif