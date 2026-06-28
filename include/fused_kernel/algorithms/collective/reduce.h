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

    template <typename ComputeIOp>
    FK_HOST_FUSE void exec(const DPPDetails&, const T value,
                           const ComputeIOp& compute, T*, T* accumulator) {
        if (accumulator != nullptr) {
            *accumulator = make_tuple(*accumulator, value) | compute;
        }
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
        const unsigned int mask = __activemask();
        const int lane = static_cast<int>(threadIdx.x) & 31;
        for (int offset = 16; offset > 0; offset >>= 1) {
            const T peer = __shfl_down_sync(mask, value, offset);
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
struct ReduceGridDPP<ParArch::GPU_NVIDIA, DPPDetails> {
private:
    using SelfType = ReduceGridDPP<ParArch::GPU_NVIDIA, DPPDetails>;
    using T = typename DPPDetails::ValueType;

    template <typename ComputeIOp>
    static __device__ __forceinline__ void atomicApply(
            T* address, const T value, const ComputeIOp& compute) {
        static_assert(sizeof(T) == sizeof(unsigned int),
                      "ReduceGridDPP CAS supports 32-bit value types");
        static_assert(std::is_trivially_copyable_v<T>,
                      "ReduceGridDPP CAS requires a trivially copyable type");

        auto* bitsAddress = reinterpret_cast<unsigned int*>(address);
        unsigned int oldBits = atomicCAS(bitsAddress, 0u, 0u);
        unsigned int assumedBits;
        do {
            assumedBits = oldBits;
            T current;
            __builtin_memcpy(&current, &assumedBits, sizeof(T));
            const T next = make_tuple(current, value) | compute;
            unsigned int nextBits;
            __builtin_memcpy(&nextBits, &next, sizeof(T));
            oldBits = atomicCAS(bitsAddress, assumedBits, nextBits);
        } while (oldBits != assumedBits);
    }

public:
    FK_STATIC_STRUCT(ReduceGridDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::GPU_NVIDIA;

    template <typename ComputeIOp>
    FK_DEVICE_FUSE void exec(const DPPDetails& details, const T value,
                             const ComputeIOp& compute, T* warpScratch,
                             T* accumulator) {
        const T blockPartial =
            ReduceBlockDPP<ParArch::GPU_NVIDIA, DPPDetails>::exec(
                details, value, compute, warpScratch);
        if (threadIdx.x == 0) {
            atomicApply(accumulator, blockPartial, compute);
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
#endif // defined(__NVCC__)

} // namespace fk

#endif // FK_COLLECTIVE_REDUCE_H
