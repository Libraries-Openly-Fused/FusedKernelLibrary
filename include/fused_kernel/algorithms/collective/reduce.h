/* Copyright 2026 Johnny Nunez

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_COLLECTIVE_REDUCE_H
#define FK_COLLECTIVE_REDUCE_H

#include <fused_kernel/core/data/point.h>
#include <fused_kernel/core/data/tuple.h>
#include <fused_kernel/core/execution_model/operation_model/operation_model.h>
#include <fused_kernel/core/execution_model/parallel_architectures.h>
#include <fused_kernel/core/execution_model/stream.h>
#include <fused_kernel/core/utils/utils.h>

#include <type_traits>

namespace fk {

template <typename T, typename ComputeIOp>
struct AtomicReduceWriteParams {
    RawPtr<ND::_1D, T> output;
    ComputeIOp compute;
};

template <typename AtomicWriteIOp, typename = void>
struct AtomicReduceWriteTraits {
    static constexpr bool VALID = false;
    using ComputeType = void;
};

template <typename AtomicWriteIOp>
struct AtomicReduceWriteTraits<AtomicWriteIOp,
    std::void_t<typename AtomicWriteIOp::Operation::AtomicReduceWriteTag>> {
    static constexpr bool VALID = true;
    using ComputeType =
        typename AtomicWriteIOp::Operation::AtomicReduceWriteTag;
};

/**
 * Atomic grid-reduction epilogue. The destination must be initialized to the
 * neutral identity of ComputeIOp before launch.
 */
template <typename T, typename ComputeIOp>
struct AtomicReduceWrite {
private:
    using SelfType = AtomicReduceWrite<T, ComputeIOp>;
    using Parent = WriteOperation<
        T, AtomicReduceWriteParams<T, ComputeIOp>, T,
        TF::DISABLED, SelfType>;

public:
    FK_STATIC_STRUCT(AtomicReduceWrite, SelfType)
    DECLARE_WRITE_PARENT
    using AtomicReduceWriteTag = ComputeIOp;

    FK_DEVICE_FUSE void exec(const Point thread, const InputType input,
                             const ParamsType& params) {
#if defined(__CUDA_ARCH__)
        static_assert(sizeof(T) == sizeof(unsigned int),
                      "AtomicReduceWrite CAS supports 32-bit value types");
        static_assert(std::is_trivially_copyable_v<T>,
                      "AtomicReduceWrite requires a trivially copyable type");

        T* const address = PtrAccessor<ND::_1D>::template point<T, T>(
            thread, params.output);
        auto* const bitsAddress = reinterpret_cast<unsigned int*>(address);
        unsigned int oldBits = atomicCAS(bitsAddress, 0u, 0u);
        unsigned int assumedBits;
        do {
            assumedBits = oldBits;
            T current;
            __builtin_memcpy(&current, &assumedBits, sizeof(T));
            const T next = make_tuple(current, input) | params.compute;
            unsigned int nextBits;
            __builtin_memcpy(&nextBits, &next, sizeof(T));
            oldBits = atomicCAS(bitsAddress, assumedBits, nextBits);
        } while (oldBits != assumedBits);
#else
        (void)thread;
        (void)input;
        (void)params;
#endif
    }

    FK_HOST_FUSE InstantiableType build(const Ptr<ND::_1D, T>& ptr,
                                         const ComputeIOp& compute) {
        return {{{ptr.ptr(), compute}}};
    }
};

template <typename T, int BLOCK_SIZE = 256>
struct ReduceDPPDetails {
    static_assert(BLOCK_SIZE > 0, "ReduceDPP block size must be positive");

    using ValueType = T;
    static constexpr int BLOCK_THREADS = BLOCK_SIZE;

    int rows;
    int width;
    // Must be the neutral element of the supplied compute IOp.
    T identity;
    int rowOffset{0};
};

/**
 * Parameters for simultaneous row reductions over one shared input read.
 * Each compute entry is Tuple<transform IOp, reduce IOp>; each result is
 * emitted through the Write IOp at the same tuple index.
 */
template <typename T, int REDUCTIONS, int BLOCK_SIZE = 256>
struct MultiReduceDPPDetails {
    static_assert(REDUCTIONS > 1,
                  "MultiReduceDPP requires at least two reductions");
    static_assert(BLOCK_SIZE > 0,
                  "MultiReduceDPP block size must be positive");

    using ValueType = T;
    static constexpr int REDUCTION_COUNT = REDUCTIONS;
    static constexpr int BLOCK_THREADS = BLOCK_SIZE;

    int rows;
    int width;
    // Each value must be the neutral element of its reduction IOp.
    T identities[REDUCTIONS];
    int rowOffset{0};
};

template <ParArch PA, typename DPPDetails>
struct ReduceWarpDPP;

template <typename DPPDetails>
struct ReduceWarpDPP<ParArch::CPU, DPPDetails> {
private:
    using SelfType = ReduceWarpDPP<ParArch::CPU, DPPDetails>;
    using T = typename DPPDetails::ValueType;

public:
    FK_STATIC_STRUCT(ReduceWarpDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::CPU;

    template <typename ComputeIOp>
    FK_HOST_FUSE T exec(const DPPDetails&, const T value,
                        const ComputeIOp&) {
        return value;
    }
};

template <ParArch PA, typename DPPDetails>
struct ReduceBlockDPP;

template <typename DPPDetails>
struct ReduceBlockDPP<ParArch::CPU, DPPDetails> {
private:
    using SelfType = ReduceBlockDPP<ParArch::CPU, DPPDetails>;
    using T = typename DPPDetails::ValueType;

public:
    FK_STATIC_STRUCT(ReduceBlockDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::CPU;

    template <typename ComputeIOp>
    FK_HOST_FUSE T exec(const DPPDetails&, const T value,
                        const ComputeIOp&, T*) {
        return value;
    }
};

template <ParArch PA, typename DPPDetails>
struct ReduceDPP;

template <ParArch PA, typename DPPDetails>
struct MultiReduceDPP;

template <typename DPPDetails>
struct ReduceDPP<ParArch::CPU, DPPDetails> {
private:
    using SelfType = ReduceDPP<ParArch::CPU, DPPDetails>;
    using Details = DPPDetails;
    using T = typename Details::ValueType;

public:
    FK_STATIC_STRUCT(ReduceDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::CPU;

    template <typename ReadIOp, typename ComputeIOp, typename WriteIOp>
    FK_HOST_FUSE void exec(const Details& details,
                           const ReadIOp& input,
                           const ComputeIOp& compute,
                           const WriteIOp& output) {
        static_assert(isAnyReadType<ReadIOp>,
                      "ReduceDPP input must be a Read or ReadBack IOp");
        static_assert(isAnyWriteType<WriteIOp>,
                      "ReduceDPP output must be a Write IOp");
        static_assert(Details::BLOCK_THREADS > 0,
                      "ReduceDPP block size must be positive");

        for (int localRow = 0; localRow < details.rows; ++localRow) {
            const int row = details.rowOffset + localRow;
            T accumulator = details.identity;
            for (int x = 0; x < details.width; ++x) {
                const T value = static_cast<T>(
                    ReadIOp::Operation::exec(Point{x, row, 0}, input));
                accumulator = make_tuple(accumulator, value) | compute;
            }
            WriteIOp::Operation::exec(Point{row, 0, 0}, accumulator, output);
        }
    }
};

template <typename DPPDetails>
struct MultiReduceDPP<ParArch::CPU, DPPDetails> {
private:
    using SelfType = MultiReduceDPP<ParArch::CPU, DPPDetails>;
    using Details = DPPDetails;
    using T = typename Details::ValueType;
    static constexpr int N = Details::REDUCTION_COUNT;

    template <size_t I, typename ComputeIOps>
    FK_HOST_FUSE void accumulateOne(T (&accumulators)[N],
                                    const T value,
                                    const ComputeIOps& computes) {
        const auto& pipeline = get<I>(computes);
        using Pipeline = std::decay_t<decltype(pipeline)>;
        static_assert(isTuple_v<Pipeline> && Pipeline::size == 2,
                      "Each MultiReduce compute pipeline is Tuple<transform, reduce>");
        const T transformed = static_cast<T>(value | get<0>(pipeline));
        accumulators[I] = make_tuple(accumulators[I], transformed) |
                          get<1>(pipeline);
    }

    template <typename ComputeIOps, size_t... Is>
    FK_HOST_FUSE void accumulateAll(T (&accumulators)[N],
                                    const T value,
                                    const ComputeIOps& computes,
                                    std::index_sequence<Is...>) {
        (accumulateOne<Is>(accumulators, value, computes), ...);
    }

    template <size_t I, typename WriteIOps>
    FK_HOST_FUSE void writeOne(const int row,
                               const T (&accumulators)[N],
                               const WriteIOps& outputs) {
        const auto& output = get<I>(outputs);
        using Output = std::decay_t<decltype(output)>;
        static_assert(isAnyWriteType<Output>,
                      "Every MultiReduce output must be a Write IOp");
        Output::Operation::exec(Point{row, 0, 0}, accumulators[I], output);
    }

    template <typename WriteIOps, size_t... Is>
    FK_HOST_FUSE void writeAll(const int row,
                               const T (&accumulators)[N],
                               const WriteIOps& outputs,
                               std::index_sequence<Is...>) {
        (writeOne<Is>(row, accumulators, outputs), ...);
    }

public:
    FK_STATIC_STRUCT(MultiReduceDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::CPU;

    template <typename ReadIOp, typename ComputeIOps, typename WriteIOps>
    FK_HOST_FUSE void exec(const Details& details,
                           const ReadIOp& input,
                           const ComputeIOps& computes,
                           const WriteIOps& outputs) {
        static_assert(isAnyReadType<ReadIOp>,
                      "MultiReduceDPP input must be a Read or ReadBack IOp");
        static_assert(isTuple_v<ComputeIOps> &&
                      std::decay_t<ComputeIOps>::size == N,
                      "MultiReduceDPP needs one compute pipeline per accumulator");
        static_assert(isTuple_v<WriteIOps> &&
                      std::decay_t<WriteIOps>::size == N,
                      "MultiReduceDPP needs one Write IOp per accumulator");

        for (int localRow = 0; localRow < details.rows; ++localRow) {
            const int row = details.rowOffset + localRow;
            T accumulators[N];
            for (int i = 0; i < N; ++i) {
                accumulators[i] = details.identities[i];
            }
            for (int x = 0; x < details.width; ++x) {
                const T value = static_cast<T>(
                    ReadIOp::Operation::exec(Point{x, row, 0}, input));
                accumulateAll(accumulators, value, computes,
                              std::make_index_sequence<N>{});
            }
            writeAll(row, accumulators, outputs,
                     std::make_index_sequence<N>{});
        }
    }
};

template <ParArch PA, typename DPPDetails>
struct ReduceGridDPP;

template <typename DPPDetails>
struct ReduceGridDPP<ParArch::CPU, DPPDetails> {
private:
    using SelfType = ReduceGridDPP<ParArch::CPU, DPPDetails>;
    using T = typename DPPDetails::ValueType;

public:
    FK_STATIC_STRUCT(ReduceGridDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::CPU;

    template <typename ComputeIOp, typename ReadIOp, typename WriteIOp>
    FK_HOST_FUSE void exec(const DPPDetails&, const T value,
                           const ComputeIOp& compute,
                           const ReadIOp& accumulatorInput,
                           const WriteIOp& accumulatorOutput) {
        static_assert(isAnyReadType<ReadIOp>,
                      "CPU ReduceGridDPP accumulator input must be a Read IOp");
        static_assert(isAnyWriteType<WriteIOp>,
                      "CPU ReduceGridDPP accumulator output must be a Write IOp");
        const Point point{0, 0, 0};
        const T current = static_cast<T>(
            ReadIOp::Operation::exec(point, accumulatorInput));
        const T next = make_tuple(current, value) | compute;
        WriteIOp::Operation::exec(point, next, accumulatorOutput);
    }
};

#if defined(__NVCC__)
template <typename DPPDetails>
struct ReduceWarpDPP<ParArch::GPU_NVIDIA, DPPDetails> {
private:
    using SelfType = ReduceWarpDPP<ParArch::GPU_NVIDIA, DPPDetails>;
    using T = typename DPPDetails::ValueType;

public:
    FK_STATIC_STRUCT(ReduceWarpDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::GPU_NVIDIA;

    template <typename ComputeIOp>
    FK_DEVICE_FUSE T exec(const DPPDetails&, T value,
                          const ComputeIOp& compute) {
#if defined(__CUDA_ARCH__)
        constexpr unsigned int FULL_WARP_MASK = 0xffffffffu;
        const int lane = static_cast<int>(threadIdx.x) & 31;
        for (int offset = 16; offset > 0; offset >>= 1) {
            const T peer = __shfl_down_sync(FULL_WARP_MASK, value, offset);
            if (lane + offset < 32) {
                value = make_tuple(value, peer) | compute;
            }
        }
#else
        (void)compute;
#endif
        return value;
    }
};

template <typename DPPDetails>
struct ReduceBlockDPP<ParArch::GPU_NVIDIA, DPPDetails> {
private:
    using SelfType = ReduceBlockDPP<ParArch::GPU_NVIDIA, DPPDetails>;
    using T = typename DPPDetails::ValueType;

public:
    FK_STATIC_STRUCT(ReduceBlockDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::GPU_NVIDIA;

    template <typename ComputeIOp>
    FK_DEVICE_FUSE T exec(const DPPDetails& details, T value,
                          const ComputeIOp& compute, T* warpScratch) {
#if defined(__CUDA_ARCH__)
        constexpr int WARP_SIZE = 32;
        constexpr int NUM_WARPS =
            (DPPDetails::BLOCK_THREADS + WARP_SIZE - 1) / WARP_SIZE;
        const int lane = static_cast<int>(threadIdx.x) & (WARP_SIZE - 1);
        const int warp = static_cast<int>(threadIdx.x) / WARP_SIZE;

        value = ReduceWarpDPP<ParArch::GPU_NVIDIA, DPPDetails>::exec(
            details, value, compute);
        if (lane == 0) warpScratch[warp] = value;
        __syncthreads();

        if (warp == 0) {
            value = lane < NUM_WARPS ? warpScratch[lane] : details.identity;
            value = ReduceWarpDPP<ParArch::GPU_NVIDIA, DPPDetails>::exec(
                details, value, compute);
        }
#else
        (void)details;
        (void)compute;
        (void)warpScratch;
#endif
        return value;
    }
};

template <typename DPPDetails>
struct ReduceDPP<ParArch::GPU_NVIDIA, DPPDetails> {
private:
    using SelfType = ReduceDPP<ParArch::GPU_NVIDIA, DPPDetails>;
    using Details = DPPDetails;
    using T = typename Details::ValueType;

public:
    FK_STATIC_STRUCT(ReduceDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::GPU_NVIDIA;

    // Deliberately not FK_DEVICE_FUSE: CUDA rejects a function-local
    // __shared__ declaration inside the macro's constexpr function.
    template <typename ReadIOp, typename ComputeIOp, typename WriteIOp>
    static __device__ __forceinline__ void exec(const Details& details,
                                                const ReadIOp& input,
                                                const ComputeIOp& compute,
                                                const WriteIOp& output) {
        static_assert(isAnyReadType<ReadIOp>,
                      "ReduceDPP input must be a Read or ReadBack IOp");
        static_assert(isAnyWriteType<WriteIOp>,
                      "ReduceDPP output must be a Write IOp");
        static_assert(Details::BLOCK_THREADS >= 32 &&
                      Details::BLOCK_THREADS <= 1024 &&
                      Details::BLOCK_THREADS % 32 == 0,
                      "ReduceDPP block size must be a warp multiple in [32, 1024]");

        constexpr int NUM_WARPS = Details::BLOCK_THREADS / 32;
        __shared__ T warpScratch[NUM_WARPS];

        const int localRow = static_cast<int>(blockIdx.x);
        if (localRow >= details.rows) return;
        const int row = details.rowOffset + localRow;
        const int tid = static_cast<int>(threadIdx.x);
        T accumulator = details.identity;
        for (int x = tid; x < details.width; x += Details::BLOCK_THREADS) {
            const T value = static_cast<T>(
                ReadIOp::Operation::exec(Point{x, row, 0}, input));
            accumulator = make_tuple(accumulator, value) | compute;
        }

        const T result =
            ReduceBlockDPP<ParArch::GPU_NVIDIA, Details>::exec(
                details, accumulator, compute, warpScratch);
        if (tid == 0) {
            WriteIOp::Operation::exec(Point{row, 0, 0}, result, output);
        }
    }
};

template <typename DPPDetails>
struct MultiReduceDPP<ParArch::GPU_NVIDIA, DPPDetails> {
private:
    using SelfType = MultiReduceDPP<ParArch::GPU_NVIDIA, DPPDetails>;
    using Details = DPPDetails;
    using T = typename Details::ValueType;
    static constexpr int N = Details::REDUCTION_COUNT;
    static constexpr int NUM_WARPS = Details::BLOCK_THREADS / 32;

    template <size_t I, typename ComputeIOps>
    FK_DEVICE_FUSE void accumulateOne(T (&accumulators)[N],
                                      const T value,
                                      const ComputeIOps& computes) {
        const auto& pipeline = get<I>(computes);
        using Pipeline = std::decay_t<decltype(pipeline)>;
        static_assert(isTuple_v<Pipeline> && Pipeline::size == 2,
                      "Each MultiReduce compute pipeline is Tuple<transform, reduce>");
        const T transformed = static_cast<T>(value | get<0>(pipeline));
        accumulators[I] = make_tuple(accumulators[I], transformed) |
                          get<1>(pipeline);
    }

    template <typename ComputeIOps, size_t... Is>
    FK_DEVICE_FUSE void accumulateAll(T (&accumulators)[N],
                                      const T value,
                                      const ComputeIOps& computes,
                                      std::index_sequence<Is...>) {
        (accumulateOne<Is>(accumulators, value, computes), ...);
    }

    template <size_t I, typename ComputeIOps>
    FK_DEVICE_FUSE void reduceOne(const Details& details,
                                  T (&accumulators)[N],
                                  const ComputeIOps& computes,
                                  T (&warpScratch)[N][NUM_WARPS]) {
        const auto& pipeline = get<I>(computes);
        using ScalarDetails = ReduceDPPDetails<T, Details::BLOCK_THREADS>;
        const ScalarDetails scalarDetails{
            details.rows, details.width, details.identities[I], details.rowOffset};
        accumulators[I] =
            ReduceBlockDPP<ParArch::GPU_NVIDIA, ScalarDetails>::exec(
                scalarDetails, accumulators[I], get<1>(pipeline), warpScratch[I]);
    }

    template <typename ComputeIOps, size_t... Is>
    FK_DEVICE_FUSE void reduceAll(const Details& details,
                                  T (&accumulators)[N],
                                  const ComputeIOps& computes,
                                  T (&warpScratch)[N][NUM_WARPS],
                                  std::index_sequence<Is...>) {
        // Left-to-right fold keeps every block-wide barrier in the same order
        // for all threads while each accumulator uses disjoint scratch.
        (reduceOne<Is>(details, accumulators, computes, warpScratch), ...);
    }

    template <size_t I, typename WriteIOps>
    FK_DEVICE_FUSE void writeOne(const int row,
                                 const T (&accumulators)[N],
                                 const WriteIOps& outputs) {
        const auto& output = get<I>(outputs);
        using Output = std::decay_t<decltype(output)>;
        static_assert(isAnyWriteType<Output>,
                      "Every MultiReduce output must be a Write IOp");
        Output::Operation::exec(Point{row, 0, 0}, accumulators[I], output);
    }

    template <typename WriteIOps, size_t... Is>
    FK_DEVICE_FUSE void writeAll(const int row,
                                 const T (&accumulators)[N],
                                 const WriteIOps& outputs,
                                 std::index_sequence<Is...>) {
        (writeOne<Is>(row, accumulators, outputs), ...);
    }

public:
    FK_STATIC_STRUCT(MultiReduceDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::GPU_NVIDIA;

    // Deliberately not FK_DEVICE_FUSE: CUDA rejects function-local shared
    // storage inside the macro's constexpr function.
    template <typename ReadIOp, typename ComputeIOps, typename WriteIOps>
    static __device__ __forceinline__ void exec(
            const Details& details, const ReadIOp& input,
            const ComputeIOps& computes, const WriteIOps& outputs) {
        static_assert(isAnyReadType<ReadIOp>,
                      "MultiReduceDPP input must be a Read or ReadBack IOp");
        static_assert(isTuple_v<ComputeIOps> &&
                      std::decay_t<ComputeIOps>::size == N,
                      "MultiReduceDPP needs one compute pipeline per accumulator");
        static_assert(isTuple_v<WriteIOps> &&
                      std::decay_t<WriteIOps>::size == N,
                      "MultiReduceDPP needs one Write IOp per accumulator");
        static_assert(Details::BLOCK_THREADS >= 32 &&
                      Details::BLOCK_THREADS <= 1024 &&
                      Details::BLOCK_THREADS % 32 == 0,
                      "MultiReduceDPP block size must be a warp multiple in [32, 1024]");

        __shared__ T warpScratch[N][NUM_WARPS];
        const int localRow = static_cast<int>(blockIdx.x);
        if (localRow >= details.rows) return;
        const int row = details.rowOffset + localRow;
        const int tid = static_cast<int>(threadIdx.x);
        T accumulators[N];
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            accumulators[i] = details.identities[i];
        }

        for (int x = tid; x < details.width; x += Details::BLOCK_THREADS) {
            const T value = static_cast<T>(
                ReadIOp::Operation::exec(Point{x, row, 0}, input));
            accumulateAll(accumulators, value, computes,
                          std::make_index_sequence<N>{});
        }
        reduceAll(details, accumulators, computes, warpScratch,
                  std::make_index_sequence<N>{});
        if (tid == 0) {
            writeAll(row, accumulators, outputs,
                     std::make_index_sequence<N>{});
        }
    }
};

template <typename DPPDetails>
struct ReduceGridDPP<ParArch::GPU_NVIDIA, DPPDetails> {
private:
    using SelfType = ReduceGridDPP<ParArch::GPU_NVIDIA, DPPDetails>;
    using T = typename DPPDetails::ValueType;

public:
    FK_STATIC_STRUCT(ReduceGridDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::GPU_NVIDIA;

    template <typename ComputeIOp, typename AtomicWriteIOp>
    FK_DEVICE_FUSE void exec(const DPPDetails& details, const T value,
                             const ComputeIOp& compute, T* warpScratch,
                             const AtomicWriteIOp& atomicOutput) {
        static_assert(isAnyWriteType<AtomicWriteIOp>,
                      "ReduceGridDPP output must be a Write IOp");
        using AtomicTraits = AtomicReduceWriteTraits<AtomicWriteIOp>;
        static_assert(AtomicTraits::VALID,
                      "ReduceGridDPP output must be AtomicReduceWrite");
        static_assert(std::is_same_v<
                          typename AtomicTraits::ComputeType, ComputeIOp>,
                      "ReduceGridDPP compute and atomic output must match");
        const T blockPartial =
            ReduceBlockDPP<ParArch::GPU_NVIDIA, DPPDetails>::exec(
                details, value, compute, warpScratch);
        if (threadIdx.x == 0) {
            AtomicWriteIOp::Operation::exec(
                Point{0, 0, 0}, blockPartial, atomicOutput);
        }
    }
};

template <typename DPPDetails, typename ReadIOp,
          typename ComputeIOp, typename WriteIOp>
__global__ void launchReduceDPPKernel(
        const __grid_constant__ DPPDetails details,
        const __grid_constant__ ReadIOp input,
        const __grid_constant__ ComputeIOp compute,
        const __grid_constant__ WriteIOp output) {
    ReduceDPP<ParArch::GPU_NVIDIA, DPPDetails>::exec(
        details, input, compute, output);
}

template <typename DPPDetails, typename ReadIOp,
          typename ComputeIOp, typename WriteIOp>
inline void executeReduce(const DPPDetails& details,
                          const ReadIOp& input,
                          const ComputeIOp& compute,
                          const WriteIOp& output,
                          Stream_<ParArch::GPU_NVIDIA>& stream) {
    static_assert(std::is_trivially_copyable_v<DPPDetails>,
                  "ReduceDPP Details must be trivially copyable");
    if (details.rows <= 0) return;
    launchReduceDPPKernel<<<details.rows, DPPDetails::BLOCK_THREADS, 0,
                           stream.getCUDAStream()>>>(
        details, input, compute, output);
    gpuErrchk(cudaGetLastError());
}

template <typename DPPDetails, typename ReadIOp,
          typename ComputeIOps, typename WriteIOps>
__global__ void launchMultiReduceDPPKernel(
        const __grid_constant__ DPPDetails details,
        const __grid_constant__ ReadIOp input,
        const __grid_constant__ ComputeIOps computes,
        const __grid_constant__ WriteIOps outputs) {
    MultiReduceDPP<ParArch::GPU_NVIDIA, DPPDetails>::exec(
        details, input, computes, outputs);
}

template <typename DPPDetails, typename ReadIOp,
          typename ComputeIOps, typename WriteIOps>
inline void executeMultiReduce(
        const DPPDetails& details, const ReadIOp& input,
        const ComputeIOps& computes, const WriteIOps& outputs,
        Stream_<ParArch::GPU_NVIDIA>& stream) {
    static_assert(std::is_trivially_copyable_v<DPPDetails>,
                  "MultiReduceDPP Details must be trivially copyable");
    if (details.rows <= 0) return;
    launchMultiReduceDPPKernel<<<
        details.rows, DPPDetails::BLOCK_THREADS, 0,
        stream.getCUDAStream()>>>(details, input, computes, outputs);
    gpuErrchk(cudaGetLastError());
}
#endif // defined(__NVCC__)

} // namespace fk

#endif // FK_COLLECTIVE_REDUCE_H
