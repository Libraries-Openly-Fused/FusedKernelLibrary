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

#ifndef FK_SYCL_KERNELS_H
#define FK_SYCL_KERNELS_H

// How the DPPs of the SYCL backend (ParArch::GPU_INTEL) are launched, and what they use to know which thread they
// are. Available when FK_SYCL_ENABLED is defined (see compiler_macros.h).
//
// A CUDA DPP is called from a __global__ function, launched on a grid of blocks of threads, and reads its position in
// the grid from threadIdx, blockIdx and blockDim. Its shared memory is a __shared__ array, and its parameters are the
// arguments of the kernel. In SYCL:
// - The grid is a sycl::nd_range<3> (gridOf): work groups are the blocks, and work items the threads.
// - There is no kernel function to write: gpu_intel::launch enqueues, in the stream, a kernel that calls
//   DPP::exec(item, ...) in every thread.
// - A DPP gets its position in the grid in the sycl::nd_item<3> that is the first parameter of its exec. ThreadPosition
//   reads it with the names of CUDA.
// - The arrays in work-group local memory (the shared memory) are allocated when the kernel is enqueued
//   (gpu_intel::launchWithLocalMemory), and exec gets a pointer to each one after the nd_item.
// - The parameters of the DPP (details and IOps) are the arguments of the kernel, or are in device memory when they
//   are too large for that (FK_SYCL_MAX_KERNEL_ARGUMENTS_BYTES).

#include <fused_kernel/core/utils/compiler_macros.h>

#if defined(FK_SYCL_ENABLED)

#include <fused_kernel/core/execution_model/stream.h>
#include <fused_kernel/core/data/point.h>
#include <fused_kernel/core/data/tuple.h>

#include <array>
#include <cstddef>

namespace fk {
    namespace gpu_intel {

        // A number of blocks, or of threads of a block, in each dimension: dim3 in CUDA
        struct Dim3 {
            std::size_t x{ 1 };
            std::size_t y{ 1 };
            std::size_t z{ 1 };
        };

        // The range of a grid of grid.x * grid.y * grid.z blocks of block.x * block.y * block.z threads. In SYCL the
        // last dimension of a range is the innermost one: x is dimension 2, y is dimension 1 and z is dimension 0.
        inline sycl::nd_range<3> gridOf(const Dim3& grid, const Dim3& block) {
            return sycl::nd_range<3>(sycl::range<3>(grid.z * block.z, grid.y * block.y, grid.x * block.x),
                                     sycl::range<3>(block.z, block.y, block.x));
        }

        // The position of a thread in the grid of its kernel (gridOf), with the names it has in CUDA
        struct ThreadPosition {
            Point threadIdx; // Position of the thread in its block
            Point blockIdx;  // Position of the block in the grid
            Point blockDim;  // Threads of a block

            explicit ThreadPosition(const sycl::nd_item<3>& item)
                : threadIdx{ static_cast<int>(item.get_local_id(2)), static_cast<int>(item.get_local_id(1)),
                             static_cast<int>(item.get_local_id(0)) },
                  blockIdx{ static_cast<int>(item.get_group(2)), static_cast<int>(item.get_group(1)),
                            static_cast<int>(item.get_group(0)) },
                  blockDim{ static_cast<int>(item.get_local_range(2)), static_cast<int>(item.get_local_range(1)),
                            static_cast<int>(item.get_local_range(0)) } {}
        };

        // The thread of a grid whose blocks have one thread in z, as the ones of TransformDPP: its position in x and
        // y, and the plane it belongs to in z
        inline Point globalThread(const sycl::nd_item<3>& item) {
            return Point{ static_cast<int>(item.get_global_id(2)), static_cast<int>(item.get_global_id(1)),
                          static_cast<int>(item.get_global_id(0)) };
        }

        // __syncthreads() in CUDA: every thread of the block waits here for the others
        inline void syncThreads(const sycl::nd_item<3>& item) {
            sycl::group_barrier(item.get_group());
        }

        namespace detail {
            template <typename... Params>
            inline constexpr bool fitInKernelArguments =
                (sizeof(Params) + ... + std::size_t{ 0 }) <= FK_SYCL_MAX_KERNEL_ARGUMENTS_BYTES;

            template <typename T>
            inline T* localPointer(const sycl::local_accessor<T, 1>& local) {
                return local.template get_multi_ptr<sycl::access::decorated::no>().get();
            }
        } // namespace detail

        // Enqueues in the stream a kernel that runs DPP::exec(item, local..., params...) in every thread of the grid,
        // where item is the sycl::nd_item<3> of the thread, and local... are pointers to LOCALS arrays of LocalType
        // in work-group local memory, the shared memory of the block, with the number of elements given in
        // localElements (LOCALS is 0, 1 or 2). params... are copied: the kernel does not use them after this returns.
        template <typename DPP, typename LocalType, std::size_t LOCALS, typename... Params>
        inline void launchWithLocalMemory(Stream_<ParArch::GPU_INTEL>& stream, const sycl::nd_range<3>& grid,
                                          const std::array<std::size_t, LOCALS>& localElements,
                                          const Params&... params) {
            static_assert(LOCALS <= 2, "gpu_intel::launchWithLocalMemory supports up to 2 arrays of local memory");
            if (grid.get_global_range().size() == 0) {
                return; // No threads: nothing to execute
            }
            sycl::queue& queue = stream.getSYCLQueue();
            // threadCode(item, local...) is what every thread runs. The kernel has a copy of it, which holds the
            // parameters of the DPP or the pointer to them. The captures are copies that are not const (x = x), so
            // that the SYCL runtime can move the kernel: moving a const fk::Tuple of one element does not compile.
            const auto submit = [&](const auto& threadCode) -> sycl::event {
                if constexpr (LOCALS == 0) {
                    return queue.parallel_for(grid, [threadCode = threadCode](const sycl::nd_item<3> item) {
                        threadCode(item);
                    });
                } else if constexpr (LOCALS == 1) {
                    return queue.submit([&](sycl::handler& commandGroup) {
                        sycl::local_accessor<LocalType, 1> local0(sycl::range<1>(localElements[0]), commandGroup);
                        commandGroup.parallel_for(grid, [threadCode = threadCode, local0](const sycl::nd_item<3> item) {
                            threadCode(item, detail::localPointer(local0));
                        });
                    });
                } else {
                    return queue.submit([&](sycl::handler& commandGroup) {
                        sycl::local_accessor<LocalType, 1> local0(sycl::range<1>(localElements[0]), commandGroup);
                        sycl::local_accessor<LocalType, 1> local1(sycl::range<1>(localElements[1]), commandGroup);
                        commandGroup.parallel_for(
                            grid, [threadCode = threadCode, local0, local1](const sycl::nd_item<3> item) {
                                threadCode(item, detail::localPointer(local0), detail::localPointer(local1));
                            });
                    });
                }
            };
            if constexpr (detail::fitInKernelArguments<Params...>) {
                submit([... p = params](const sycl::nd_item<3>& item, auto* const... local) {
                    DPP::exec(item, local..., p...);
                });
            } else {
                // Too large for the arguments of a kernel: the parameters go to device memory, in a copy enqueued in
                // the stream before the kernel, and the kernel gets the pointer
                using Block = Tuple<Params...>;
                const Block* const block = stream.kernelParams(Block(params...));
                const sycl::event kernel = submit([block](const sycl::nd_item<3>& item, auto* const... local) {
                    apply_d([&](const Params&... p) { DPP::exec(item, local..., p...); }, *block);
                });
                stream.kernelParamsUsedBy(block, kernel);
            }
        }

        // Enqueues in the stream a kernel that runs DPP::exec(item, params...) in every thread of the grid, where
        // item is the sycl::nd_item<3> of the thread. params... are copied: the kernel does not use them after this
        // returns. The memory the IOps in params point to must stay valid until the kernel has finished.
        template <typename DPP, typename... Params>
        inline void launch(Stream_<ParArch::GPU_INTEL>& stream, const sycl::nd_range<3>& grid,
                           const Params&... params) {
            launchWithLocalMemory<DPP, char, 0>(stream, grid, std::array<std::size_t, 0>{}, params...);
        }
    } // namespace gpu_intel
} // namespace fk

#endif // defined(FK_SYCL_ENABLED)

#endif // FK_SYCL_KERNELS_H
