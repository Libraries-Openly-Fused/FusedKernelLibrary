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

#ifndef FK_REDUCTIONS_ROW_REDUCE_H
#define FK_REDUCTIONS_ROW_REDUCE_H

/* RowReduceDPP: row-wise reduction as a Data Parallel Pattern, implemented purely
 * against the InstantiableDPP protocol (see core/execution_model/instantiable_dpp.h):
 * it needs NO Executor specialization and NO dedicated __global__ kernel.
 *
 * For each row y of the input, it combines all the elements of the row with a
 * BinaryType compute Operation (the ReduceOp, e.g. Add<int>) and writes the single
 * per-row result at Point{ y, 0, 0 } of the output Write IOp (so the natural output
 * container is a Ptr1D with one element per input row).
 *
 * ReduceOp requirements: the combination ORDER is backend-dependent. The CPU backend
 * is a strict sequential left fold over each row, while the GPU backend interleaves a
 * strided per-lane accumulation across lanes with a shared-memory tree reduction. The
 * ReduceOp must therefore be ASSOCIATIVE and COMMUTATIVE (e.g. Add, Max, Min): a
 * non-commutative BinaryType Operation such as Sub compiles cleanly but produces
 * backend-dependent results. For floating-point types both backends are correct, but
 * reassociation means their rounding may differ, so exact bitwise equality between
 * backends is not guaranteed.
 *
 * 2D inputs only: the input IOp's ActiveThreads.z must be 1. build() throws
 * std::invalid_argument for batched/3D inputs, which would otherwise silently reduce
 * only plane z == 0.
 *
 * IO contract (IO_SPEC): exactly one complete Read/ReadBack IOp in and one Write IOp
 * out; NO compute IOp chain is accepted (it would be ambiguous whether it applies
 * before or after the reduction). Per-element preprocessing must be fused into the
 * input IOp instead: readIOp.then(computeIOp...) runs in-register at load time.
 *
 * Not a TransformDPP: the row reduction requires cross-thread cooperation on the GPU
 * backend (shared memory + barriers). Like every DPP, it is a stateless struct that
 * touches memory exclusively through the IOps it receives, and it has a single-thread
 * CPU implementation.
 *
 * Usage:
 *   const auto iDPP = RowReduceDPP<Add<int>>::build(readIOp, writeIOp);
 *   fk::execute(stream, iDPP);
 */

#include <fused_kernel/core/execution_model/instantiable_dpp.h>

#if !defined(NVRTC_COMPILER)
#include <stdexcept>
#endif

namespace fk {

// Cooperative-DPP exec bodies use __shared__ + barriers: they cannot be
// constexpr (FK_DEVICE_FUSE). Plain static device inline qualifier:
#ifndef FK_COOP_DEVICE_FUSE
#define FK_COOP_DEVICE_FUSE static __device__ __forceinline__ void
#endif

    struct RowReduceDPPDetails {}; // Stateless: all geometry comes from the IOps

    namespace row_reduce_detail {
        // RowReduceDPP reduces 2D inputs only: both backends fix the Point z coordinate
        // to 0, so a batched/3D input would silently reduce only plane z == 0.
        template <typename InIOp>
        FK_HOST_CNST void assertInputIs2D(const InIOp& input) {
#if !defined(NVRTC_COMPILER)
            if (input.getActiveThreads().z != 1) {
                throw std::invalid_argument(
                    "RowReduceDPP: 2D inputs only (the input IOp's ActiveThreads.z must be 1). "
                    "A batched/3D input would silently reduce only plane z == 0.");
            }
#endif
        }
    } // namespace row_reduce_detail

#define ROW_REDUCE_DPP_STATIC_INTERFACE \
    /* The ReduceOp must also be associative and commutative: the GPU backend reorders \
     * the combination (see the header comment). That is a semantic property the type \
     * system cannot check, so it is a documented requirement, not a static_assert. */ \
    static_assert(opIs<BinaryType, ReduceOp>, \
        "RowReduceDPP: ReduceOp must be a BinaryType compute Operation (e.g. Add<int>)"); \
    static_assert(BLOCK_SIZE > 0 && BLOCK_SIZE <= 1024, \
        "RowReduceDPP: BLOCK_SIZE must be in (0, 1024]"); \
    using Details = RowReduceDPPDetails; \
    /** IO contract: one Read/ReadBack IOp in, one Write IOp out, no compute IOp chain. */ \
    static constexpr DPPIOSpec IO_SPEC{ /*inputIOps*/ 1, /*outputIOps*/ 1, \
                                        /*acceptsComputeIOps*/ false, /*argsAreIOpSequences*/ false }; \
    template <typename InIOp, typename OutIOp> \
    FK_HOST_FUSE Details build_details(const InIOp& input, const OutIOp&) { \
        row_reduce_detail::assertInputIs2D(input); \
        return Details{}; \
    } \
    /** build: creates an InstantiableDPP, enforcing IO_SPEC at compile time. \
     *  Execute the result with fk::execute(stream, instantiableDPP). */ \
    template <typename... IOps> \
    FK_HOST_FUSE auto build(const IOps&... iOps) { \
        return buildInstantiableDPP<SelfType>(iOps...); \
    }

    template <typename ReduceOp, enum ParArch PA = defaultParArch, int BLOCK_SIZE = 256>
    struct RowReduceDPP;

    template <typename ReduceOp, int BLOCK_SIZE>
    struct RowReduceDPP<ReduceOp, ParArch::CPU, BLOCK_SIZE> {
    private:
        using SelfType = RowReduceDPP<ReduceOp, ParArch::CPU, BLOCK_SIZE>;
    public:
        FK_STATIC_STRUCT(RowReduceDPP, SelfType)
        static constexpr ParArch PAR_ARCH = ParArch::CPU;
        ROW_REDUCE_DPP_STATIC_INTERFACE

        template <typename InIOp, typename OutIOp>
        FK_HOST_FUSE void exec(const Details&, const InIOp& input, const OutIOp& output) {
            using ValueType = typename InIOp::Operation::OutputType;
            static_assert(std::is_same_v<ValueType, typename ReduceOp::InputType> &&
                          std::is_same_v<ValueType, typename ReduceOp::ParamsType> &&
                          std::is_same_v<ValueType, typename ReduceOp::OutputType>,
                "RowReduceDPP: the ReduceOp Input/Params/Output types must match the input IOp OutputType");
            const ActiveThreads activeThreads = input.getActiveThreads();
            const int width = static_cast<int>(activeThreads.x);
            const int rows = static_cast<int>(activeThreads.y);
            if (width == 0) {
                return;
            }
            for (int y = 0; y < rows; ++y) {
                ValueType accumulator = InIOp::Operation::exec(Point{ 0, y, 0 }, input);
                for (int x = 1; x < width; ++x) {
                    accumulator = ReduceOp::exec(accumulator, InIOp::Operation::exec(Point{ x, y, 0 }, input));
                }
                OutIOp::Operation::exec(Point{ y, 0, 0 }, accumulator, output);
            }
        }
    };

#if defined(__NVCC__)
    template <typename ReduceOp, int BLOCK_SIZE>
    struct RowReduceDPP<ReduceOp, ParArch::GPU_NVIDIA, BLOCK_SIZE> {
    private:
        using SelfType = RowReduceDPP<ReduceOp, ParArch::GPU_NVIDIA, BLOCK_SIZE>;
    public:
        FK_STATIC_STRUCT(RowReduceDPP, SelfType)
        static constexpr ParArch PAR_ARCH = ParArch::GPU_NVIDIA;
        ROW_REDUCE_DPP_STATIC_INTERFACE

        // One thread block per row
        template <typename InIOp, typename OutIOp>
        FK_HOST_FUSE DPPLaunchConfig getLaunchConfig(const Details&, const InIOp& input, const OutIOp&) {
            const ActiveThreads activeThreads = input.getActiveThreads();
            return { ActiveThreads(activeThreads.y, 1u, 1u),
                     ActiveThreads(static_cast<uint>(BLOCK_SIZE), 1u, 1u) };
        }

        template <typename InIOp, typename OutIOp>
        FK_COOP_DEVICE_FUSE exec(const Details&, const InIOp& input, const OutIOp& output) {
            using ValueType = typename InIOp::Operation::OutputType;
            static_assert(std::is_same_v<ValueType, typename ReduceOp::InputType> &&
                          std::is_same_v<ValueType, typename ReduceOp::ParamsType> &&
                          std::is_same_v<ValueType, typename ReduceOp::OutputType>,
                "RowReduceDPP: the ReduceOp Input/Params/Output types must match the input IOp OutputType");
            __shared__ ValueType partials[BLOCK_SIZE];

            const int row = static_cast<int>(blockIdx.x);
            const int tid = static_cast<int>(threadIdx.x);
            const int width = static_cast<int>(input.getActiveThreads().x);
            if (width == 0) {
                return;
            }
            // Each active lane sequentially combines a strided slice of the row (every
            // lane owns at least one element, so no identity element is required).
            const int activeLanes = width < BLOCK_SIZE ? width : BLOCK_SIZE;
            if (tid < activeLanes) {
                ValueType accumulator = InIOp::Operation::exec(Point{ tid, row, 0 }, input);
                for (int x = tid + BLOCK_SIZE; x < width; x += BLOCK_SIZE) {
                    accumulator = ReduceOp::exec(accumulator, InIOp::Operation::exec(Point{ x, row, 0 }, input));
                }
                partials[tid] = accumulator;
            }
            __syncthreads();
            // Shared memory tree reduction over the activeLanes partials
            for (int stride = 1; stride < activeLanes; stride *= 2) {
                if ((tid % (2 * stride) == 0) && (tid + stride < activeLanes)) {
                    partials[tid] = ReduceOp::exec(partials[tid], partials[tid + stride]);
                }
                __syncthreads();
            }
            if (tid == 0) {
                OutIOp::Operation::exec(Point{ row, 0, 0 }, partials[0], output);
            }
        }
    };
#endif // defined(__NVCC__)

#undef ROW_REDUCE_DPP_STATIC_INTERFACE

} // namespace fk

#endif // FK_REDUCTIONS_ROW_REDUCE_H
