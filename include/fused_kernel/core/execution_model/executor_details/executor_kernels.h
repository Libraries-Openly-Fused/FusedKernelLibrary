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

#ifndef FK_EXECUTOR_KERNELS_H
#define FK_EXECUTOR_KERNELS_H

#include <fused_kernel/core/utils/compiler_macros.h>

#if defined(FK_SYCL_ENABLED)
#include <fused_kernel/core/execution_model/stream.h>
#include <fused_kernel/core/data/tuple.h>

#include <array>
#include <cstddef>

// The total size of the arguments of a kernel is limited: 2048 bytes on Intel GPUs, and 1024 bytes is the least a
// device can have (sycl::info::device::max_parameter_size). A kernel that exceeds the limit is only rejected when the
// device compiles it, at run time. The parameters of a DPP (its details and IOps) are the arguments of its kernel
// while they add up to this number of bytes at most; if they are larger (for instance, the IOps of a large batch)
// they are copied to device memory, and the kernel gets a pointer to them (see launchDPP_Kernel).
#ifndef FK_SYCL_MAX_KERNEL_ARGUMENTS_BYTES
#define FK_SYCL_MAX_KERNEL_ARGUMENTS_BYTES 1024
#endif
#endif // defined(FK_SYCL_ENABLED)

#if defined(__NVCC__) || defined(__HIPCC__)
namespace fk {
// The kernel of every DPP: executeOperations<DPP>(stream, ...) launches launchDPP_Kernel<DPP> with the DPP's details (or
// Params) and IOps, and the kernel only calls DPP::exec with them. The DPP is the first template argument, so the kernel's
// name says which DPP it runs; DPPDetails and IOps are deduced from the launch arguments. Every parameter is
// __grid_constant__: read in place from the parameter space, never copied to a local (Oscar's decision, 2026-11-26).
template <typename DPP, typename DPPDetails, typename... IOps>
__global__ void launchDPP_Kernel(const __grid_constant__ DPPDetails details, const __grid_constant__ IOps... iOps) {
    DPP::exec(details, iOps...);
}

} // namespace fk
#endif

#if defined(FK_SYCL_ENABLED)
namespace fk {
// The SYCL counterpart of the CUDA launchDPP_Kernel<DPP><<<grid, block, 0, stream>>>(details, iOps...). SYCL has no
// kernel functions: launchDPP_Kernel<DPP>(stream, grid, details, iOps...) enqueues, in the queue of the stream, a
// kernel that runs DPP::exec(item, details, iOps...) in every thread of the grid (syclGrid in executors.h), where
// item is the sycl::nd_item<3> of the thread: the DPP reads its position in the grid from it (ThreadPosition and
// globalThread in data_parallel_patterns.h). The parameters are copied: the kernel does not use them after this
// returns, but the memory the IOps point to must stay valid until it has finished.
//
// A DPP that uses shared memory gets it with launchDPP_KernelWithLocalMemory<DPP, LocalType>(stream, grid,
// DPP::LOCAL_ELEMENTS, details, iOps...): the arrays of LocalType in work-group local memory are allocated when the
// kernel is enqueued, and exec gets a pointer to each one after the nd_item, exec(item, local..., details, iOps...).
namespace executor_detail {
template <typename... Params>
inline constexpr bool fitInKernelArguments =
    (sizeof(Params) + ... + std::size_t{ 0 }) <= FK_SYCL_MAX_KERNEL_ARGUMENTS_BYTES;

template <typename T>
inline T* localPointer(const sycl::local_accessor<T, 1>& local) {
    return local.template get_multi_ptr<sycl::access::decorated::no>().get();
}
} // namespace executor_detail

// LOCALS arrays of LocalType (0, 1 or 2), with the number of elements given in localElements
template <typename DPP, typename LocalType, std::size_t LOCALS, typename... Params>
inline void launchDPP_KernelWithLocalMemory(Stream_<ParArch::GPU_INTEL>& stream, const sycl::nd_range<3>& grid,
                                            const std::array<std::size_t, LOCALS>& localElements,
                                            const Params&... params) {
    static_assert(LOCALS <= 2, "launchDPP_KernelWithLocalMemory supports up to 2 arrays of local memory");
    if (grid.get_global_range().size() == 0) {
        return; // No threads: nothing to execute
    }
    sycl::queue& queue = stream.getSYCLQueue();
    // threadCode(item, local...) is what every thread runs. The kernel has a copy of it, which holds the parameters of
    // the DPP or the pointer to them. The captures are copies that are not const (x = x), so that the SYCL runtime can
    // move the kernel: moving a const fk::Tuple of one element does not compile.
    const auto submit = [&](const auto& threadCode) -> sycl::event {
        if constexpr (LOCALS == 0) {
            return queue.parallel_for(grid, [threadCode = threadCode](const sycl::nd_item<3> item) {
                threadCode(item);
            });
        } else if constexpr (LOCALS == 1) {
            return queue.submit([&](sycl::handler& commandGroup) {
                sycl::local_accessor<LocalType, 1> local0(sycl::range<1>(localElements[0]), commandGroup);
                commandGroup.parallel_for(grid, [threadCode = threadCode, local0](const sycl::nd_item<3> item) {
                    threadCode(item, executor_detail::localPointer(local0));
                });
            });
        } else {
            return queue.submit([&](sycl::handler& commandGroup) {
                sycl::local_accessor<LocalType, 1> local0(sycl::range<1>(localElements[0]), commandGroup);
                sycl::local_accessor<LocalType, 1> local1(sycl::range<1>(localElements[1]), commandGroup);
                commandGroup.parallel_for(
                    grid, [threadCode = threadCode, local0, local1](const sycl::nd_item<3> item) {
                        threadCode(item, executor_detail::localPointer(local0), executor_detail::localPointer(local1));
                    });
            });
        }
    };
    if constexpr (executor_detail::fitInKernelArguments<Params...>) {
        submit([... p = params](const sycl::nd_item<3>& item, auto* const... local) {
            DPP::exec(item, local..., p...);
        });
    } else {
        // Too large for the arguments of a kernel: the parameters go to device memory, in a copy enqueued in the
        // stream before the kernel, and the kernel gets the pointer
        using Block = Tuple<Params...>;
        const Block* const block = stream.kernelParams(Block(params...));
        const sycl::event kernel = submit([block](const sycl::nd_item<3>& item, auto* const... local) {
            apply_d([&](const Params&... p) { DPP::exec(item, local..., p...); }, *block);
        });
        stream.kernelParamsUsedBy(block, kernel);
    }
}

template <typename DPP, typename... Params>
inline void launchDPP_Kernel(Stream_<ParArch::GPU_INTEL>& stream, const sycl::nd_range<3>& grid,
                             const Params&... params) {
    launchDPP_KernelWithLocalMemory<DPP, char, 0>(stream, grid, std::array<std::size_t, 0>{}, params...);
}

} // namespace fk
#endif // defined(FK_SYCL_ENABLED)

#endif // FK_EXECUTOR_KERNELS_H
