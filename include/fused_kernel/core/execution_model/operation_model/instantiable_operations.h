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

#ifndef FK_INSTANTIABLE_OPERATIONS
#define FK_INSTANTIABLE_OPERATIONS

#include <fused_kernel/core/data/vector_types.h>
#include <fused_kernel/core/execution_model/operation_model/operation_data.h>
#include <fused_kernel/core/execution_model/thread_fusion.h>
#include <fused_kernel/core/utils/parameter_pack_utils.h>
#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/data/size.h>
#include <fused_kernel/core/data/array.h>
#include <fused_kernel/core/constexpr_libs/constexpr_cmath.h>
#include <fused_kernel/core/execution_model/operation_model/vector_operations.h>
#include <fused_kernel/algorithms/basic_ops/cast_base.h>
#include <fused_kernel/core/execution_model/active_threads.h>

namespace fk { // namespace FusedKernel
#define DEVICE_FUNCTION_DETAILS_IS(instance_type) \
    using Operation = Operation_t; \
    using InstanceType = instance_type; \
    template <typename IT> \
    static constexpr bool is{ std::is_same_v<IT, InstanceType> };

#define DEVICE_FUNCTION_DETAILS_IS_ASSERT(instance_type) \
    DEVICE_FUNCTION_DETAILS_IS(instance_type) \
    static_assert(std::is_same_v<typename Operation::InstanceType, instance_type>, "Operation is not " #instance_type );

    struct Fuser;

    template <typename Operation_t>
    struct ReadInstantiableOperation final : public OperationData<Operation_t> {
        DEVICE_FUNCTION_DETAILS_IS_ASSERT(ReadType)

        template <typename ContinuationIOp, typename Fuser_t = Fuser>
        FK_HOST_CNST auto then(const ContinuationIOp& cIOp) const {
            return Fuser_t::fuse(*this, cIOp);
        }

        FK_HOST_DEVICE_CNST ActiveThreads getActiveThreads() const {
            return Operation::getActiveThreads(*this);
        }
    };

    template <typename Operation_t>
    struct ReadBackInstantiableOperation final : public OperationData<Operation_t> {
        DEVICE_FUNCTION_DETAILS_IS_ASSERT(ReadBackType)

        template <typename ContinuationIOp, typename Fuser_t = Fuser>
        FK_HOST_CNST auto then(const ContinuationIOp& cIOp) const {
            return Fuser_t::fuse(*this, cIOp);
        }

        FK_HOST_DEVICE_CNST ActiveThreads getActiveThreads() const {
            return Operation::getActiveThreads(*this);
        }
    };

    /**
    * @brief BinaryInstantiableOperation: represents a IOp that takes the result of the previous IOp as input
    * (which will reside on GPU registers) and an additional parameter that contains data not generated during the execution
    * of the current kernel.
    * It generates an output and returns it in register memory.
    * It can be composed of a single Operation or of a chain of Operations, in which case it wraps them into an
    * FusedOperation.
    * Expects Operation_t to have an static __device__ function member with the following parameters:
    * OutputType exec(const InputType& input, const OperationData<Operation_t>& opDat)
    */
    template <typename Operation_t>
    struct BinaryInstantiableOperation final : public OperationData<Operation_t> {
        DEVICE_FUNCTION_DETAILS_IS_ASSERT(BinaryType)

        template <typename ContinuationIOp, typename Fuser_t = Fuser>
        FK_HOST_CNST auto then(const ContinuationIOp& cIOp) const {
            return Fuser_t::fuse(*this, cIOp);
        }

        template <typename ContinuationIOp, typename... ContinuationIOps>
        FK_HOST_CNST auto then(const ContinuationIOp& cIOp, const ContinuationIOps&... cIOps) const {
            return then(cIOp).then(cIOps...);
        }
    };

    /**
    * @brief TernaryInstantiableOperation: represents a IOp that takes the result of the previous IOp as input
    * (which will reside on GPU registers) plus two additional parameters.
    * Second parameter (params): represents the same as in a BinaryFunction, data thas was not generated during the execution
    * of this kernel.
    * Third parameter (back_function): it's a IOp that will be used at some point in the implementation of the
    * Operation. It can be any kind of IOp.
    * Expects Operation_t to have an static __device__ function member with the following parameters:
    * OutputType exec(const InputType& input, const OperationData<Operation_t>& opData)
    */
    template <typename Operation_t>
    struct TernaryInstantiableOperation final : public OperationData<Operation_t> {
        DEVICE_FUNCTION_DETAILS_IS_ASSERT(TernaryType)

        template <typename ContinuationIOp, typename Fuser_t = Fuser>
        FK_HOST_CNST auto then(const ContinuationIOp& cIOp) const {
            return Fuser_t::fuse(*this, cIOp);
        }

        template <typename ContinuationIOp, typename... ContinuationIOps>
        FK_HOST_CNST auto then(const ContinuationIOp& cIOp, const ContinuationIOps&... cIOps) const {
            return then(cIOp).then(cIOps...);
        }
    };

    /**
    * @brief UnaryInstantiableOperation: represents a IOp that takes the result of the previous IOp as input
    * (which will reside on GPU registers).
    * It allows to execute the Operation (or chain of Unary Operations) on the input, and returns the result as output
    * in register memory.
    * Expects Operation_t to have an static __device__ function member with the following parameters:
    * OutputType exec(const InputType& input)
    */
    template <typename Operation_t>
    struct UnaryInstantiableOperation {
        DEVICE_FUNCTION_DETAILS_IS_ASSERT(UnaryType)

        template <typename ContinuationIOp, typename Fuser_t = Fuser>
        FK_HOST_CNST auto then(const ContinuationIOp& cIOp) const {
            return Fuser_t::fuse(*this, cIOp);
        }

        template <typename ContinuationIOp, typename... ContinuationIOps>
        FK_HOST_CNST auto then(const ContinuationIOp& cIOp, const ContinuationIOps&... cIOps) const {
            return then(cIOp).then(cIOps...);
        }
    };

    /**
    * @brief MidWriteInstantiableOperation: represents a IOp that takes the result of the previous IOp as input
    * (which will reside on GPU registers) and writes it into device memory. The way that the data is written, is definded
    * by the implementation of Operation_t.
    * It returns the input data without modification, so that another IOp can be executed after it, using the same data.
    */
    template <typename Operation_t>
    struct MidWriteInstantiableOperation final : public OperationData<Operation_t> {
        DEVICE_FUNCTION_DETAILS_IS(MidWriteType)
            static_assert(std::is_same_v<typename Operation::InstanceType, WriteType> ||
                std::is_same_v<typename Operation::InstanceType, MidWriteType>,
                "Operation is not WriteType or MidWriteType");

        template <typename ContinuationIOp, typename Fuser_t = Fuser>
        FK_HOST_CNST auto then(const ContinuationIOp& cIOp) const {
            return Fuser_t::fuse(*this, cIOp);
        }

        template <typename ContinuationIOp, typename... ContinuationIOps>
        FK_HOST_CNST auto then(const ContinuationIOp& cIOp, const ContinuationIOps&... cIOps) const {
            return then(cIOp).then(cIOps...);
        }
    };

    /**
    * @brief WriteInstantiableOperation: represents a IOp that takes the result of the previous IOp as input
    * (which will reside on GPU registers) and writes it into device memory. The way that the data is written, is definded
    * by the implementation of Operation_t.
    * It can only be the last IOp in a sequence of InstantiableOperations.
    */
    template <typename Operation_t>
    struct WriteInstantiableOperation final : public OperationData<Operation_t> {
        DEVICE_FUNCTION_DETAILS_IS_ASSERT(WriteType)
    };

#undef DEVICE_FUNCTION_DETAILS_IS_ASSERT
#undef IS_ASSERT
#undef DEVICE_FUNCTION_DETAILS_IS
#undef DEVICE_FUNCTION_DETAILS
#undef ASSERT
#undef IS

    template <typename Operation>
    using Read = ReadInstantiableOperation<Operation>;
    template <typename Operation>
    using ReadBack = ReadBackInstantiableOperation<Operation>;
    template <typename Operation>
    using Unary = UnaryInstantiableOperation<Operation>;
    template <typename Operation>
    using Binary = BinaryInstantiableOperation<Operation>;
    template <typename Operation>
    using Ternary = TernaryInstantiableOperation<Operation>;
    template <typename Operation>
    using MidWrite = MidWriteInstantiableOperation<Operation>;
    template <typename Operation>
    using Write = WriteInstantiableOperation<Operation>;

    template <typename Operation, typename Enabler = void>
    struct InstantiableOperationType;

    // Single Operation cases
    template <typename Operation>
    struct InstantiableOperationType<Operation, std::enable_if_t<isReadType<Operation>>> {
        using type = Read<Operation>;
    };

    template <typename Operation>
    struct InstantiableOperationType<Operation, std::enable_if_t<isReadBackType<Operation>>> {
        using type = ReadBack<Operation>;
    };

    template <typename Operation>
    struct InstantiableOperationType<Operation, std::enable_if_t<isUnaryType<Operation>>> {
        using type = Unary<Operation>;
    };

    template <typename Operation>
    struct InstantiableOperationType<Operation, std::enable_if_t<isBinaryType<Operation>>> {
        using type = Binary<Operation>;
    };

    template <typename Operation>
    struct InstantiableOperationType<Operation, std::enable_if_t<isTernaryType<Operation>>> {
        using type = Ternary<Operation>;
    };

    template <typename Operation>
    struct InstantiableOperationType<Operation, std::enable_if_t<isWriteType<Operation>>> {
        using type = Write<Operation>;
    };

    template <typename Operation>
    using Instantiable = typename InstantiableOperationType<Operation>::type;
} // namespace fk

#endif
