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

#ifndef FK_REDUCE_DPP_H
#define FK_REDUCE_DPP_H

// ReduceDPP: from 1 to N reductions of the same data, reading every element once.
//
//     executeOperations<ReduceDPP<>>(stream, readIOp, make_tuple(make_tuple(firstIOp, accumulationIOp, finalIOp), ...),
//                                    writeIOp);
//
// - readIOp is any complete read: a Read, a ReadBack, a batch of them or a read fused with the operations that follow
//   it (read.then(...)). It gives the elements: for every plane z of its active threads (the batch of a batched read,
//   the planes of a 3D read, 1 otherwise) the width x height elements that the read reports for that plane, read at
//   Point{x, y, z}. A 1D source is a plane of height 1. Every plane is reduced on its own, at its own size.
// - Each reduction is a Tuple<FirstIOp, AccumulationIOp, FinalIOp> of compute IOps (Unary, Binary or Ternary, or a
//   fusion of them):
//     - FirstIOp is applied to every element read, and gives a value of the accumulator type of the reduction;
//     - AccumulationIOp takes a Tuple of two accumulated values and gives the accumulated value of both, of the same
//       type (the accumulator type): Add<A, A, A, UnaryType>, Max<A, A, A, UnaryType>... It is applied to values held
//       in registers, in shared memory and, when the blocks of a plane synchronize through device memory, in device
//       memory. The order in which the elements are accumulated is not the order of the data, so it must be
//       associative and commutative. There is no neutral element: every accumulator starts as the first value of one
//       element.
//     - FinalIOp is applied once, to the accumulated value of the whole plane.
// - writeIOp receives the N final values of every plane: the value of reduction i of plane z at Point{i, 0, z}. A
//   float3 matrix with one reduction becomes one float3 value, and with three reductions three float3 values. The
//   write needs N elements in x and as many planes as the read: PerThreadWrite<ND::_1D, T> of N elements for a single
//   plane, a TensorWrite or a batch of writes for several. The executors check it and throw std::invalid_argument,
//   as they do for a read without elements. A plane without elements, among planes that have them, has nothing to
//   reduce: nothing is written for it.
// - The types of the elements and of the accumulators can be scalars, vectors or any other trivially copyable type:
//   the DPP does not operate on them, it only passes them from one IOp to the next, and the IOps are the ones that
//   know how to operate on them (Cast, PerThreadRead, TensorRead, PerThreadWrite and TensorWrite only work with
//   scalars and vectors). An accumulator also has to be default constructible, copy constructible and copy
//   assignable: it is stored, and the executors check it. Accumulators are meant to be small: see
//   ReduceDPPPolicy::BLOCK_THREADS for what the large ones cost on a GPU. FirstIOp must give exactly the accumulator
//   type (a Cast is an operation of the FirstIOp); the other IOps take what the previous one gives as any function
//   takes an argument, as in TransformDPP.
// - The buffers that the IOps read and write must stay valid until the execution has finished (sync()). The values of
//   a plane are written when all its elements have been read, so the write of a plane can be the memory that the same
//   plane reads. It must not overlap what another plane reads: the GPU versions reduce all the planes at the same
//   time.
//
// ReduceDPP<ParArch::CPU>: every plane is split in chunks of Details::chunkElements consecutive elements (in row major
// order). Each chunk is accumulated in order, and the chunks of a plane are then accumulated in order. The split only
// depends on the sizes of the data, so the result does not depend on the threads of the machine. Every plane is read
// before any is written.
// ReduceDPP<ParArch::CPU_MULTITHREAD>: runs the chunks of the CPU version on cpuThreadPool() and accumulates them in
// the same order, so both give the same results: bit for bit with integers and with plain additions of floats, and
// up to the last bits where a multiplication of the FirstIOp feeds the accumulation, because the compiler can
// contract a * b + c differently in each of them (see cpu_multithread.h).
//
// ReduceDPP<ParArch::GPU_NVIDIA> and ReduceDPP<ParArch::GPU_AMD>: one kernel (launchDPP_Kernel<ReduceDPP<...>>). The
// grid has one z plane per plane of the read (65535 at most, the planes of a grid: a read of more throws
// std::invalid_argument), and in x the blocks of the largest plane; block b of a plane reduces blockElements
// consecutive elements of it. The accumulated values go up through as many of these levels as the size of the data
// needs:
//   1. thread: each thread reads one of every BLOCK_THREADS elements of its block and accumulates them in registers;
//   2. block: the threads of the block accumulate their values in shared memory;
//   3. cluster (GPU_NVIDIA, compute capability 9.0 and newer): the blocks of a thread block cluster read the shared
//      memory of the other blocks of the cluster (distributed shared memory), so the first block of the cluster
//      accumulates the values of all of them, still without device memory;
//   4. grid: when a plane has more than one cluster (or block, without clusters), each one stores its value in device
//      memory and counts itself in an arrival counter. The last one to arrive finds the values of all the others
//      there (global synchronization), accumulates them, and is the one that applies FinalIOp and writes.
// A plane that one block reduces, or one cluster, never touches device memory before the write. The device memory
// of level 4 is the workspace of the execution, which the caller owns: the GPU executors allocate no memory. (The CPU
// versions allocate the values of their chunks, in host memory, in every execution.)
// Which levels an execution uses (ReduceDPPPolicy has the sizes, and what was measured to choose them):
// - planes of up to SINGLE_BLOCK_ELEMENTS elements (4096): one block each, and no workspace. (Also the larger planes
//   of a read of so many planes that the policy gives one block to each: workspaceBytes() is 0 for it.)
// - larger planes, with a workspace: blocks that synchronize through it (levels 1, 2 and 4), on every device;
// - larger planes, without a workspace: on a device with thread block clusters, planes of up to 8 blocks of
//   SINGLE_BLOCK_ELEMENTS elements (32768) are reduced by one cluster each (levels 1, 2 and 3). Any other execution
//   without a workspace throws.
// The grid covers the largest plane of the read: every plane costs the launch of the blocks of the largest one, and
// the ones that a smaller plane does not use return at once. The blocks of a plane are the ones of the read divided
// by its planes, as if every plane had the elements of the largest one, so in a read of many planes of very
// different sizes the large ones are reduced by few blocks: an image of 2048 x 1024 has 128 blocks alone, 32 among
// 31 images of 10 x 10 and 8 among 127, and those reads took 2 and 5.8 times longer than two executions (the large
// image, and the small ones) on an RTX 4090, and 1.1 and 2.4 times on an RTX 5050. Such planes are better reduced
// in one execution per size.
//
//     const size_t bytes = ReduceDPP<>::workspaceBytes(readIOp, reductions);
//     Ptr1D<uchar> workspace(static_cast<uint>(bytes));
//     if (bytes > 0) {
//         setTo(uchar{0}, workspace, stream);   // once, after allocating it
//     }
//     executeOperations<ReduceDPP<>>(stream, ReduceDPP<>::build_details(workspace), readIOp, reductions, writeIOp);
//
// - workspaceBytes(readIOp, reductions) is the device memory that the planes of that read need on any device, and 0
//   when they need none: one block reduces every plane, or the DPP is one of the CPU versions (which do not use a
//   workspace that they are given, so this code is the same for every backend). The call without details,
//   executeOperations<ReduceDPP<>>(stream, readIOp, reductions, writeIOp), has no workspace. An execution that needs a
//   workspace and has none, or a smaller one, throws std::invalid_argument, which says the bytes that it needs.
// - The workspace must be zero when it is used for the first time: its first bytes are the arrival counters of the
//   planes, and every execution leaves them at zero. After that, any execution of any read and reductions can use
//   it, as long as two that use it never run at the same time: the executions of one stream can share one.
// - An execution is asynchronous and allocates nothing, so it can be captured in a CUDA graph as any other.
//
// The implementation for each compute capability is chosen at compile time, with a single ParArch::GPU_NVIDIA:
// - the kernel is compiled once for each architecture of the build, and __CUDA_ARCH__ selects in each of those images
//   whether level 3 exists;
// - the host code, which is compiled once, knows the architectures of the build from __CUDA_ARCH_LIST__, which nvcc
//   also defines for it: in a build without architectures with clusters it has no code for them. Which of the images
//   a device runs can only be known at run time, so in a build with them the driver is asked, and only by the
//   executions that would launch clusters (with the default policy, the ones without a workspace of planes that one
//   block does not reduce and one cluster does): whether the image of the kernel that it runs on the device has them
//   (a build with architectures of both kinds), and the largest cluster that the stream accepts. With the default
//   policy, an execution with a workspace asks the driver nothing, and launches the same grid on every device. (With
//   ALWAYS, every execution of planes of more than one block asks, with or without a workspace; and with blocks of
//   more than 256 threads every execution asks whether the kernel can be launched with them.)
//
// The headers that this one needs are the ones of the executors. executeOperations<DPP>(stream, ...) is declared in
// fused_kernel.h, which the code that calls it includes too (before or after this header); without it, the same call
// is Executor<ReduceDPP<...>>::executeOperations(stream, ...).

#include <fused_kernel/core/execution_model/executors.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#if defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)
#include <algorithm>
#include <exception>
#include <memory>
#endif

namespace fk {
    // When ReduceDPP<ParArch::GPU_NVIDIA> launches the blocks of a plane as thread block clusters, on the devices and
    // builds that have them (see ReduceDPPPolicy::CLUSTERS)
    enum class ReduceDPPClusters {
        AUTO,   // when the execution has no workspace and every plane fits in one cluster, which needs none
        NEVER,  // never: the blocks only synchronize through the workspace
        ALWAYS  // always: one cluster when every plane fits in one, and otherwise clusters that synchronize through
                // the workspace
    };

    // What ReduceDPP does not take from the data: how it splits the elements. It is the default of the second template
    // parameter of ReduceDPP; another struct with these members (or derived from this one, redefining some) can be
    // given instead. The results of a reduction whose accumulation is not exact (a sum of floats) depend on it. On a
    // GPU the blocks of a plane also depend on the other planes of the read (how many they are, and the largest one),
    // so such a result can differ in its last bits between a plane reduced alone and the same plane in a batch; the
    // threads of a block depend on the bytes of all the accumulators of the execution (reduce_detail::blockThreads),
    // so it can also differ when reductions are added to an execution; and where the device has clusters, an
    // execution without a workspace launches other blocks than one with it. The CPU versions split every plane by
    // its own size only.
    // The values of the GPU are a compromise for two devices, an RTX 4090 (128 multiprocessors) and an RTX 5050 (20),
    // measured with cheap elements (the sum and five statistics of a uchar matrix) and expensive ones (two sums of
    // every pixel of a bilinear resize, in float and in double), for planes of 1K to 33M elements and batches of 4,
    // 30 and 128 planes: the time of an execution (the launch and the synchronization of the stream) is within 2.4%
    // and 4.2% of the best number of blocks measured for each case (geometric means; 24% and 20% in the worst case
    // of each device).
    struct ReduceDPPPolicy {
        // GPU: threads of a block, a power of two. A block has BLOCK_THREADS accumulated values in shared memory: with
        // accumulators that do not fit there, fewer threads are used (reduce_detail::blockThreads). Blocks of up to
        // 256 threads can always be launched. Larger ones only if the kernel needs no more than 65536 / BLOCK_THREADS
        // registers per thread, which depends on the IOps and on the build (a build with device debug information,
        // -G, needs more): the execution throws otherwise.
        // Fewer threads do not make large accumulators cheap. The ones that the compiler can not keep in registers
        // (structs with arrays, like a histogram) are also in the stack of every thread, from 1 to 10 times their
        // size in the kernels measured, and a kernel that needs more stack than the 1024 bytes that a context starts
        // with makes CUDA reserve its stack for every thread that the device can run at a time, whatever the threads
        // of a block are, until cudaDeviceSetLimit(cudaLimitStackSize, ...) lowers it again. Measured with one
        // accumulator of 2 KB on an RTX 4090 (196608 threads), 2 GiB of device memory; with 4 KB, 4.3 GiB; with 8 KB,
        // 13.4 GiB. With 16 KB the launch throws there (an RTX 5050, 30720 threads, reserves 4.2 GiB), and with 32 KB
        // on both. The accumulators without arrays that were tried (scalars, vectors, Tuples and structs of them)
        // stay in registers: their kernels have no stack. That is so in an optimized build: one with device debug
        // information (-O0 -g -G) keeps every accumulated value in the stack, and needs 3144 bytes per thread for one
        // float3 reduction, 11224 for five and 78448 for seventeen in the image for sm_89 (1280, 3904 and 23344 in
        // the one for sm_120), which the RTX 4090 turns into 0.4 GiB, 1.9 GiB and 14 GiB of device memory.
        static constexpr uint BLOCK_THREADS = 256;
        // GPU: the largest plane that one block reduces alone. (Measured: one block is the fastest up to some 3000
        // elements on the RTX 5050 and some 5000 on the RTX 4090.)
        static constexpr uint SINGLE_BLOCK_ELEMENTS = 4096;
        // GPU: the blocks of larger planes, which synchronize through the workspace. They are counted for all the
        // planes of the read together, as if every plane had the elements of the largest one and all of them were
        // one plane: blocks of BLOCK_ELEMENTS elements until there are BLOCKS of them; then the BLOCKS blocks grow,
        // up to LARGE_BLOCK_ELEMENTS elements each; then there are more blocks of that size, up to MAX_BLOCKS, which
        // grow again. With the defaults: blocks of 512 elements up to 64K elements, 128 blocks up to 2M, blocks of
        // 16384 elements up to 16.7M, and 1024 blocks from there.
        // Each plane has its part of those blocks, and one at least: a read of 30 planes of 65536 elements has 5
        // blocks per plane (128 / 30). 128 planes of up to 16384 elements have one block each (the 128 blocks of up
        // to 2M elements), and 128 planes of 65536 elements, which are 8.4M elements, have 4 (512 blocks of 16384
        // elements); from MAX_BLOCKS planes on, every plane has one block, whatever its size. (See the top of this
        // header for the reads whose planes have very different sizes.)
        // The best number of blocks depends on the device more than on the elements: 64 or 128 on the RTX 4090 for
        // 16K to 262K elements (256 took from 0% to 4% longer there) and 256 for 524K to 4M, and 16 to 64 on the RTX
        // 5050 for 4K to 500K. So a policy for one device can do better than these values, and BLOCKS is the member
        // to change: with 2 per multiprocessor (256 and 40), the times of each device were within 1.3% and 2.0% of
        // the best ones (geometric means). Many small blocks are what an expensive element needs (the resize in
        // double of a plane of 50K elements took 1.7 times longer in 13 blocks than in 128, on the RTX 4090), and
        // too many blocks cost as much as too few (30 planes of 65536 elements: 1.8 times longer with 128 blocks per
        // plane than with 8).
        static constexpr uint BLOCK_ELEMENTS = 512;
        static constexpr uint BLOCKS = 128;
        static constexpr uint LARGE_BLOCK_ELEMENTS = 16384;
        static constexpr uint MAX_BLOCKS = 1024;
        // GPU_NVIDIA: use of thread block clusters, and blocks of a cluster at most: from 1 to 8, the largest
        // portable size. Where the stream accepts less (few multiprocessors that are scheduled together, as in a
        // small MIG instance), the largest cluster that it accepts is used. One cluster reduces planes of up to
        // CLUSTER_BLOCKS * SINGLE_BLOCK_ELEMENTS elements, and needs no workspace: with AUTO it is what an execution
        // without a workspace launches for them, where there are clusters. With a workspace AUTO launches none,
        // because they measured no faster on an RTX 5050: one cluster of 8 blocks took the same time as the same 8
        // blocks synchronized through the workspace (10.3 us both, for 16K elements, in a build for that
        // architecture alone; 4% to 5% longer in a build that also has an architecture without clusters, where the
        // launch of a cluster asks the driver which image runs), the blocks that this policy gives a plane reduced
        // alone with the workspace are more than 8, and faster (the cluster took 4% to 15% longer in that build, for
        // 8K to 32K elements of a uchar matrix and of the resize in float and in double), and the blocks of larger
        // planes in clusters of 8 (ALWAYS) took from 1.3 to 3.4 times longer than out of clusters (49K to 33M
        // elements of a uchar matrix), because the device runs few clusters at a time.
        static constexpr ReduceDPPClusters CLUSTERS = ReduceDPPClusters::AUTO;
        static constexpr uint CLUSTER_BLOCKS = 8;
        // CPU and CPU_MULTITHREAD: elements of a chunk
        static constexpr uint CHUNK_ELEMENTS = 16384;
    };

    // The default policy of ReduceDPP<ParArch::GPU_AMD>: ReduceDPPPolicy with blocks of 128 threads, everything else the same. Measured on
    // one AMD device, an RX 9060 (gfx1200, 28 compute units, wave32, ROCm 10.1), with the kinds of data that chose ReduceDPPPolicy: a uchar
    // matrix of 4K to 33M elements (its sum, maximum and minimum), batches of 30 planes of 65536 uchar and of 128 of 262144, and the resize
    // of N crops of 280 x 280 to 224 x 224 with five statistics in float and in double (N = 1, 30, 150). The time of an execution (the launch
    // and the synchronization of the stream) is within 4.5% of the best number of blocks and threads measured for each case with 128 threads,
    // and within 10.6% with ReduceDPPPolicy's 256 (geometric means of the 13 cases). 128 threads took 16% to 17% less time for 16M and 33M
    // elements and for 128 planes of 262144, 8% to 9% less for 1M elements and for 30 planes of 65536, 6% more (1.7 us) for a plane that
    // one block reduces (4096 elements) and 2% more for the resize of 30 crops; for 30 and 150 crops every policy measured was within 1%.
    // Another device may prefer other values. A policy derived from ReduceDPPPolicy keeps its 256 threads on GPU_AMD; with blocks of 128
    // threads, accumulators of up to 256 bytes (32768 / 128) fit in the shared memory of a block without fewer threads.
    struct ReduceDPPPolicyAMD : ReduceDPPPolicy {
        static constexpr uint BLOCK_THREADS = 128;
    };

    // The default policy of each backend: ReduceDPPPolicy, and ReduceDPPPolicyAMD for ParArch::GPU_AMD
    template <enum ParArch PA>
    struct ReduceDPPDefaultPolicy {
        using type = ReduceDPPPolicy;
    };
    template <>
    struct ReduceDPPDefaultPolicy<ParArch::GPU_AMD> {
        using type = ReduceDPPPolicyAMD;
    };
    template <enum ParArch PA>
    using ReduceDPPDefaultPolicy_t = typename ReduceDPPDefaultPolicy<PA>::type;

    namespace reduce_detail {
        // The accumulated values of the reductions of one execution, one per reduction: what a thread, a block, a
        // cluster or a chunk has accumulated so far. A plain aggregate: it is stored in shared memory and in device
        // memory.
        template <typename... Accumulators>
        struct Partial;

        template <typename Accumulator>
        struct Partial<Accumulator> {
            Accumulator value;
        };

        template <typename Accumulator, typename Next, typename... Rest>
        struct Partial<Accumulator, Next, Rest...> {
            Accumulator value;
            Partial<Next, Rest...> rest;
        };

        template <typename Accumulator>
        FK_HOST_DEVICE_CNST Partial<Accumulator> makePartial(const Accumulator& value) {
            return {value};
        }

        template <typename Accumulator, typename Next, typename... Rest>
        FK_HOST_DEVICE_CNST Partial<Accumulator, Next, Rest...> makePartial(const Accumulator& value, const Next& next,
                                                                            const Rest&... rest) {
            return {value, makePartial(next, rest...)};
        }

        template <size_t Idx, typename Accumulator, typename... Rest>
        FK_HOST_DEVICE_CNST const auto& partialValue(const Partial<Accumulator, Rest...>& partial) {
            if constexpr (Idx == 0) {
                return partial.value;
            } else {
                return partialValue<Idx - 1>(partial.rest);
            }
        }

        // The reductions of an execution: a Tuple of Tuple<FirstIOp, AccumulationIOp, FinalIOp>
        template <typename Reductions>
        struct ReductionsTraits {
            static constexpr bool valid = false;
        };

        // The accumulator type of a reduction is what its AccumulationIOp gives: its OutputType. NullType for what has
        // none (a Write, something that is not an IOp), which the checks below report. (std::void_t directly as the
        // argument of the partial specialization, as in chain_check.h.)
        template <typename AccumulationIOp, typename = void>
        struct AccumulatorOfIOp {
            using type = NullType;
        };

        template <typename AccumulationIOp>
        struct AccumulatorOfIOp<AccumulationIOp, std::void_t<GetOutputType_t<AccumulationIOp>>> {
            using type = GetOutputType_t<AccumulationIOp>;
        };

        template <typename... FirstIOps, typename... AccumulationIOps, typename... FinalIOps>
        struct ReductionsTraits<Tuple<Tuple<FirstIOps, AccumulationIOps, FinalIOps>...>> {
            static constexpr bool valid = sizeof...(AccumulationIOps) > 0;
            static constexpr size_t size = sizeof...(AccumulationIOps);
            using PartialType = Partial<typename AccumulatorOfIOp<AccumulationIOps>::type...>;
            using Indexes = std::index_sequence_for<AccumulationIOps...>;
        };

        template <typename Reductions>
        using PartialOf = typename ReductionsTraits<std::decay_t<Reductions>>::PartialType;

        template <typename Reductions>
        using IndexesOf = typename ReductionsTraits<std::decay_t<Reductions>>::Indexes;

        // The compile time checks of the IOps of an execution, made as chain_check.h makes the ones of TransformDPP:
        // iopsAreValid is a plain bool, without errors inside, and reportIOps has the static_asserts. The executors
        // discard the execution with if constexpr when the IOps are not valid and use the report instead, so a wrong
        // execution ends with static_asserts that have the IOps involved in their instantiation context, and nothing
        // else is instantiated.

        // The types that the IOps of one reduction have to connect: what the read gives to FirstIOp, FirstIOp to the
        // accumulator, two accumulators to AccumulationIOp, the accumulator to FinalIOp and FinalIOp to the write. The
        // types are only looked at when the three IOps are compute IOps, which have them.
        template <typename ReadIOp, typename FirstIOp, typename AccumulationIOp, typename FinalIOp, typename WriteIOp,
                  bool Compute = isComputeType<FirstIOp> && isComputeType<AccumulationIOp> && isComputeType<FinalIOp>>
        struct ReductionTypes {
            static constexpr bool compute = false;
            static constexpr bool readToFirst = true;
            static constexpr bool accumulation = true;
            static constexpr bool firstToAccumulator = true;
            static constexpr bool accumulatorToFinal = true;
            static constexpr bool finalToWrite = true;
            static constexpr bool storable = true;
            static constexpr bool valid = false;
        };

        template <typename ReadIOp, typename FirstIOp, typename AccumulationIOp, typename FinalIOp, typename WriteIOp>
        struct ReductionTypes<ReadIOp, FirstIOp, AccumulationIOp, FinalIOp, WriteIOp, true> {
            using Accumulator = GetOutputType_t<AccumulationIOp>;
            static constexpr bool compute = true;
            static constexpr bool readToFirst =
                std::is_convertible_v<GetOutputType_t<ReadIOp>, GetInputType_t<FirstIOp>>;
            static constexpr bool accumulation =
                std::is_same_v<GetInputType_t<AccumulationIOp>, Tuple<Accumulator, Accumulator>>;
            static constexpr bool firstToAccumulator = std::is_same_v<GetOutputType_t<FirstIOp>, Accumulator>;
            static constexpr bool accumulatorToFinal = std::is_convertible_v<Accumulator, GetInputType_t<FinalIOp>>;
            static constexpr bool finalToWrite =
                std::is_convertible_v<GetOutputType_t<FinalIOp>, GetInputType_t<WriteIOp>>;
            static constexpr bool storable = std::is_trivially_copyable_v<Accumulator> &&
                                             std::is_default_constructible_v<Accumulator> &&
                                             std::is_copy_constructible_v<Accumulator> &&
                                             std::is_copy_assignable_v<Accumulator>;
            static constexpr bool valid = readToFirst && accumulation && firstToAccumulator && accumulatorToFinal &&
                                          finalToWrite && storable;
        };

        // One instance per reduction: the compiler names its IOps when it reports the static_assert that fails
        template <typename ReadIOp, typename FirstIOp, typename AccumulationIOp, typename FinalIOp, typename WriteIOp>
        struct ReductionReport {
            using Types = ReductionTypes<ReadIOp, FirstIOp, AccumulationIOp, FinalIOp, WriteIOp>;
            static_assert(Types::compute,
                          "ReduceDPP needs compute IOps (Unary, Binary or Ternary) as FirstIOp, AccumulationIOp and "
                          "FinalIOp");
            static_assert(Types::readToFirst,
                          "ReduceDPP: the OutputType of the read is not convertible to the InputType of a FirstIOp");
            static_assert(Types::accumulation,
                          "ReduceDPP: an AccumulationIOp must take a Tuple of two values of its OutputType, the "
                          "accumulator type of the reduction (for instance Add<A, A, A, UnaryType>)");
            static_assert(Types::firstToAccumulator,
                          "ReduceDPP: the OutputType of a FirstIOp must be the accumulator type, the OutputType of "
                          "its AccumulationIOp: the DPP converts nothing (a Cast is an operation of the FirstIOp)");
            static_assert(Types::accumulatorToFinal,
                          "ReduceDPP: the accumulator type, the OutputType of an AccumulationIOp, is not convertible "
                          "to the InputType of its FinalIOp");
            static_assert(Types::finalToWrite,
                          "ReduceDPP: the OutputType of a FinalIOp is not convertible to the InputType of the write");
            static_assert(Types::storable,
                          "ReduceDPP: the accumulator type, the OutputType of an AccumulationIOp, must be trivially "
                          "copyable, default constructible, copy constructible and copy assignable (so not const, "
                          "and without const members): it is stored in shared memory and in device memory");
            static constexpr bool reported = true;
        };

        // The IOps of an execution: a complete read, a write and a Tuple of reductions (Shape), whose types connect.
        // A complete read is the one of chain_check.h (chainIsCompleteRead): a Read or a ReadBack that gives
        // something, which a batch of incomplete reads (Crop<>::build(std::array<Rect, N>), without the read that it
        // crops) does not, although it is a ReadType IOp
        template <typename ReadIOp, typename Reductions, typename WriteIOp,
                  bool Shape = chainIsCompleteRead<ReadIOp> && isAnyWriteType<WriteIOp> &&
                               ReductionsTraits<Reductions>::valid>
        struct IOpsCheck {
            static constexpr bool valid = false;
        };

        template <typename ReadIOp, typename... FirstIOps, typename... AccumulationIOps, typename... FinalIOps,
                  typename WriteIOp>
        struct IOpsCheck<ReadIOp, Tuple<Tuple<FirstIOps, AccumulationIOps, FinalIOps>...>, WriteIOp, true> {
            static constexpr bool valid =
                and_v<ReductionTypes<ReadIOp, FirstIOps, AccumulationIOps, FinalIOps, WriteIOp>::valid...>;
        };

        template <typename ReadIOp, typename Reductions, typename WriteIOp,
                  bool Shape = chainIsCompleteRead<ReadIOp> && isAnyWriteType<WriteIOp> &&
                               ReductionsTraits<Reductions>::valid>
        struct IOpsReport {
            static_assert(chainIsCompleteRead<ReadIOp>,
                          "ReduceDPP needs a complete Read or ReadBack IOp to read from");
            static_assert(isAnyWriteType<WriteIOp>, "ReduceDPP needs a Write IOp to write the results with");
            static_assert(ReductionsTraits<Reductions>::valid,
                          "ReduceDPP needs its reductions as a Tuple of one or more Tuple<FirstIOp, AccumulationIOp, "
                          "FinalIOp>: make_tuple(make_tuple(firstIOp, accumulationIOp, finalIOp), ...)");
            static constexpr bool reported = true;
        };

        template <typename ReadIOp, typename... FirstIOps, typename... AccumulationIOps, typename... FinalIOps,
                  typename WriteIOp>
        struct IOpsReport<ReadIOp, Tuple<Tuple<FirstIOps, AccumulationIOps, FinalIOps>...>, WriteIOp, true> {
            static constexpr bool reported =
                and_v<ReductionReport<ReadIOp, FirstIOps, AccumulationIOps, FinalIOps, WriteIOp>::reported...>;
        };

        // The IOps can be references (the arguments of the executors) or plain types
        template <typename ReadIOp, typename Reductions, typename WriteIOp>
        inline constexpr bool iopsAreValid =
            IOpsCheck<std::decay_t<ReadIOp>, std::decay_t<Reductions>, std::decay_t<WriteIOp>>::valid;

        template <typename ReadIOp, typename Reductions, typename WriteIOp>
        inline constexpr bool reportIOps =
            IOpsReport<std::decay_t<ReadIOp>, std::decay_t<Reductions>, std::decay_t<WriteIOp>>::reported;

        // The first value of every reduction for one element read
        template <typename Value, typename Reductions, size_t... Idx>
        FK_HOST_DEVICE_CNST PartialOf<Reductions> firstValues(const Value& value, const Reductions& reductions,
                                                              const std::index_sequence<Idx...>&) {
            return makePartial((value | fk::get<0>(fk::get<Idx>(reductions)))...);
        }

        // The accumulated values of two sets of accumulated values
        template <typename Reductions, size_t... Idx>
        FK_HOST_DEVICE_CNST PartialOf<Reductions> accumulate(const PartialOf<Reductions>& a,
                                                             const PartialOf<Reductions>& b,
                                                             const Reductions& reductions,
                                                             const std::index_sequence<Idx...>&) {
            return makePartial(
                (fk::make_tuple(partialValue<Idx>(a), partialValue<Idx>(b)) | fk::get<1>(fk::get<Idx>(reductions)))...);
        }

        // The final value of every reduction of plane z, written at Point{reduction, 0, z}
        template <typename Reductions, typename WriteIOp, size_t... Idx>
        FK_HOST_DEVICE_CNST void finalValues(const int z, const PartialOf<Reductions>& partial,
                                             const Reductions& reductions, const WriteIOp& writeIOp,
                                             const std::index_sequence<Idx...>&) {
            (WriteIOp::Operation::exec(Point{static_cast<int>(Idx), 0, z},
                                       partialValue<Idx>(partial) | fk::get<2>(fk::get<Idx>(reductions)), writeIOp),
             ...);
        }

        // The elements of plane z of a read
        template <typename ReadIOp>
        FK_HOST_DEVICE_CNST ulonglong planeElements(const int z, const ReadIOp& readIOp) {
            return static_cast<ulonglong>(NumElems::x(Point{0, 0, z}, readIOp)) *
                   static_cast<ulonglong>(NumElems::y(Point{0, 0, z}, readIOp));
        }

        // The planes of a read: the z of its active threads (the batch of a batched read, the planes of a 3D read),
        // asked as its elements in z, which every complete read has (a BorderReader has no active threads of its own)
        template <typename ReadIOp>
        FK_HOST_DEVICE_CNST uint planesOf(const ReadIOp& readIOp) {
            return NumElems::z(Point{0, 0, 0}, readIOp);
        }

        // The elements of the largest plane of a read. The grid of the GPU versions covers it: the other planes use the
        // part of it that their own elements need.
        template <typename ReadIOp>
        FK_HOST_CNST ulonglong largestPlaneElements(const ReadIOp& readIOp) {
            const uint planes = planesOf(readIOp);
            ulonglong largest = 0;
            for (uint z = 0; z < planes; ++z) {
                const ulonglong elements = planeElements(static_cast<int>(z), readIOp);
                largest = elements > largest ? elements : largest;
            }
            return largest;
        }

        // The groups of size elements that cover count elements
        FK_HOST_DEVICE_CNST ulonglong groupsOf(const ulonglong count, const ulonglong size) {
            return size == 0 ? 0ull : count / size + (count % size != 0 ? 1ull : 0ull);
        }

        // What one write needs: the planes of the read and, in each of them, a row with room for one value per
        // reduction
        template <typename WriteIOp>
        FK_HOST_STATIC void checkWrite(const uint planes, const uint reductions, const WriteIOp& writeIOp) {
            executor_detail::requireArguments(WriteIOp::Operation::num_elems_z(Point{0, 0, 0}, writeIOp) == planes,
                                              "ReduceDPP",
                                              "the write's planes differ from the read's planes: the reductions of "
                                              "plane z are written at Point{reduction, 0, z}");
            for (uint z = 0; z < planes; ++z) {
                const Point plane{0, 0, static_cast<int>(z)};
                executor_detail::requireArguments(WriteIOp::Operation::num_elems_x(plane, writeIOp) >= reductions &&
                                                  WriteIOp::Operation::num_elems_y(plane, writeIOp) >= 1,
                                                  "ReduceDPP",
                                                  "a plane of the write has no row with one element per reduction");
            }
        }

        // The writes of a fused write: its last IOp, and the MidWrites fused before it, which write at the same points
        template <typename IOp>
        FK_HOST_STATIC void checkWriteOf(const uint planes, const uint reductions, const IOp& iOp) {
            if constexpr (isAnyWriteType<IOp>) {
                checkWrite(planes, reductions, iOp);
            }
        }

        template <typename WriteIOp, size_t... Idx>
        FK_HOST_STATIC void checkFusedWrite(const uint planes, const uint reductions, const WriteIOp& writeIOp,
                                            const std::index_sequence<Idx...>&) {
            (checkWriteOf(planes, reductions, get_opt<Idx>(writeIOp.params)), ...);
        }

        // What every executor checks at run time before it launches or enqueues anything: the read has elements, and
        // every write of the write IOp has the planes of the read and, in each of them, a row with room for one value
        // per reduction. Invalid arguments throw std::invalid_argument (executor_detail::requireArguments).
        template <typename ReadIOp, typename Reductions, typename WriteIOp>
        FK_HOST_STATIC void checkArguments(const ReadIOp& readIOp, const Reductions&, const WriteIOp& writeIOp) {
            const uint planes = planesOf(readIOp);
            executor_detail::requireArguments(planes > 0 && largestPlaneElements(readIOp) > 0, "ReduceDPP",
                                              "the read is empty: there is nothing to reduce");
            const uint reductions = static_cast<uint>(ReductionsTraits<Reductions>::size);
            if constexpr (is_fused_operation<typename WriteIOp::Operation>::value) {
                checkFusedWrite(planes, reductions, writeIOp,
                                std::make_index_sequence<WriteIOp::Operation::ParamsType::size>{});
            } else {
                checkWrite(planes, reductions, writeIOp);
            }
        }

        // What the executors of the CPU versions check of their details: chunks without elements would reduce
        // nothing, and write nothing (the ones of build_details are the ones of the policy, never 0)
        FK_HOST_STATIC void checkChunks(const uint chunkElements) {
            executor_detail::requireArguments(chunkElements > 0, "ReduceDPP",
                                              "the details have chunks of 0 elements: ReduceDPP<...>::build_details "
                                              "gives the ones of the policy");
        }
    } // namespace reduce_detail

    // The details of an execution of ReduceDPP, which its caller can give to executeOperations after the stream
    // (ReduceDPP<...>::build_details). The ones of the CPU are also the ones of CPU_MULTITHREAD.
    template <enum ParArch PA>
    struct ReduceDPPDetails;

    template <>
    struct ReduceDPPDetails<ParArch::CPU> {
        // Elements of a chunk: consecutive elements of a plane, in row major order
        uint chunkElements;
    };

    // GPU: the workspace of the execution, device memory of the caller (see the top of this header). Without one
    // (data == nullptr), every plane has to fit in one block, or in one cluster.
    template <>
    struct ReduceDPPDetails<ParArch::GPU_NVIDIA> {
        RawPtr<ND::_1D, uchar> workspace;
    };

    template <>
    struct ReduceDPPDetails<ParArch::GPU_AMD> {
        RawPtr<ND::_1D, uchar> workspace;
    };

    template <enum ParArch PA = defaultParArch, typename Policy = ReduceDPPDefaultPolicy_t<PA>>
    struct ReduceDPP;

    template <typename Policy>
    struct ReduceDPP<ParArch::CPU, Policy> {
    private:
        using SelfType = ReduceDPP<ParArch::CPU, Policy>;
    public:
        FK_STATIC_STRUCT(ReduceDPP, SelfType)
        static constexpr ParArch PAR_ARCH = ParArch::CPU;
        using Details = ReduceDPPDetails<ParArch::CPU>;

        // The details of an execution: the chunks of the policy. The CPU versions need no workspace: one given to
        // them, as to the GPU versions, is not used. (Details{} are not these: chunks of 0 elements, which the
        // executors reject.)
        FK_HOST_FUSE Details build_details() {
            static_assert(Policy::CHUNK_ELEMENTS > 0, "ReduceDPP: Policy::CHUNK_ELEMENTS must be greater than 0");
            return Details{Policy::CHUNK_ELEMENTS};
        }
        FK_HOST_FUSE Details build_details(const RawPtr<ND::_1D, uchar>&) {
            return build_details();
        }
        FK_HOST_STATIC Details build_details(const Ptr<ND::_1D, uchar>&) {
            return build_details();
        }

        // The bytes of the workspace that the executions of a read and some reductions need: none
        template <typename ReadIOp, typename Reductions>
        FK_HOST_FUSE size_t workspaceBytes(const ReadIOp&, const Reductions&) {
            return 0;
        }

        // The chunks of plane z
        template <typename ReadIOp>
        FK_HOST_FUSE size_t chunks(const int z, const Details& details, const ReadIOp& readIOp) {
            return static_cast<size_t>(reduce_detail::groupsOf(reduce_detail::planeElements(z, readIOp),
                                                               details.chunkElements));
        }

        // The chunks of an execution are the ones of its planes, one plane after the other. This gives the first
        // chunk of every plane, and after them the number of chunks: plane z has the chunks firsts[z] ...
        // firsts[z + 1] - 1 of the execution.
        template <typename ReadIOp>
        FK_HOST_STATIC std::vector<size_t> chunkFirsts(const Details& details, const ReadIOp& readIOp) {
            const size_t planes = static_cast<size_t>(reduce_detail::planesOf(readIOp));
            std::vector<size_t> firsts(planes + 1, size_t{0});
            for (size_t z = 0; z < planes; ++z) {
                firsts[z + 1] = firsts[z] + chunks(static_cast<int>(z), details, readIOp);
            }
            return firsts;
        }

        // Reads the elements of chunk `chunk` of plane z, which must be one of its chunks(z, ...), and accumulates
        // them in order. Used by exec() and by the CPU_MULTITHREAD version.
        template <typename ReadIOp, typename... FirstIOps, typename... AccumulationIOps, typename... FinalIOps>
        FK_HOST_STATIC auto exec_chunk(const int z, const size_t chunk, const Details& details, const ReadIOp& readIOp,
                                       const Tuple<Tuple<FirstIOps, AccumulationIOps, FinalIOps>...>& reductions) {
            using Reductions = Tuple<Tuple<FirstIOps, AccumulationIOps, FinalIOps>...>;
            using PartialType = reduce_detail::PartialOf<Reductions>;
            constexpr reduce_detail::IndexesOf<Reductions> indexes{};
            const uint width = NumElems::x(Point{0, 0, z}, readIOp);
            const ulonglong elements = reduce_detail::planeElements(z, readIOp);
            const ulonglong first = static_cast<ulonglong>(chunk) * details.chunkElements;
            ulonglong left = elements - first < details.chunkElements ? elements - first : details.chunkElements;
            uint y = static_cast<uint>(first / width);
            uint x = static_cast<uint>(first - static_cast<ulonglong>(y) * width);
            PartialType accumulated = reduce_detail::firstValues(
                ReadIOp::Operation::exec(Point{static_cast<int>(x), static_cast<int>(y), z}, readIOp), reductions,
                indexes);
            ++x;
            --left;
            while (left > 0) {
                if (x == width) {
                    x = 0;
                    ++y;
                }
                const uint rowLeft = width - x;
                const uint run = left < rowLeft ? static_cast<uint>(left) : rowLeft;
                for (uint i = 0; i < run; ++i) {
                    accumulated = reduce_detail::accumulate<Reductions>(
                        accumulated,
                        reduce_detail::firstValues(
                            ReadIOp::Operation::exec(Point{static_cast<int>(x + i), static_cast<int>(y), z}, readIOp),
                            reductions, indexes),
                        reductions, indexes);
                }
                x += run;
                left -= run;
            }
            return accumulated;
        }

        // Accumulates, in order, the values of the chunks of plane z, which are the first chunks(z, ...) values of
        // partials, and writes the final values of the plane. Used by exec() and by the CPU_MULTITHREAD version.
        template <typename ReadIOp, typename... FirstIOps, typename... AccumulationIOps, typename... FinalIOps,
                  typename WriteIOp>
        FK_HOST_STATIC void exec_plane(const int z,
                                       const reduce_detail::PartialOf<
                                           Tuple<Tuple<FirstIOps, AccumulationIOps, FinalIOps>...>>* const partials,
                                       const Details& details, const ReadIOp& readIOp,
                                       const Tuple<Tuple<FirstIOps, AccumulationIOps, FinalIOps>...>& reductions,
                                       const WriteIOp& writeIOp) {
            using Reductions = Tuple<Tuple<FirstIOps, AccumulationIOps, FinalIOps>...>;
            using PartialType = reduce_detail::PartialOf<Reductions>;
            constexpr reduce_detail::IndexesOf<Reductions> indexes{};
            const size_t planeChunks = chunks(z, details, readIOp);
            if (planeChunks == 0) {
                return;
            }
            PartialType accumulated = partials[0];
            for (size_t chunk = 1; chunk < planeChunks; ++chunk) {
                accumulated = reduce_detail::accumulate<Reductions>(accumulated, partials[chunk], reductions, indexes);
            }
            reduce_detail::finalValues(z, accumulated, reductions, writeIOp, indexes);
        }

        // The whole execution, in the two phases of the CPU_MULTITHREAD version, so that both accumulate the same
        // values in the same way: the values of the chunks of every plane are stored, and then every plane
        // accumulates its own and is written.
        template <typename ReadIOp, typename... FirstIOps, typename... AccumulationIOps, typename... FinalIOps,
                  typename WriteIOp>
        FK_HOST_STATIC void exec(const Details& details, const ReadIOp& readIOp,
                                 const Tuple<Tuple<FirstIOps, AccumulationIOps, FinalIOps>...>& reductions,
                                 const WriteIOp& writeIOp) {
            using Reductions = Tuple<Tuple<FirstIOps, AccumulationIOps, FinalIOps>...>;
            using PartialType = reduce_detail::PartialOf<Reductions>;
            [[maybe_unused]] constexpr bool reported = reduce_detail::reportIOps<ReadIOp, Reductions, WriteIOp>;
            const std::vector<size_t> firsts = chunkFirsts(details, readIOp);
            const size_t planes = firsts.size() - 1;
            std::vector<PartialType> partials(firsts.back());
            for (size_t z = 0; z < planes; ++z) {
                for (size_t chunk = firsts[z]; chunk < firsts[z + 1]; ++chunk) {
                    partials[chunk] = exec_chunk(static_cast<int>(z), chunk - firsts[z], details, readIOp, reductions);
                }
            }
            for (size_t z = 0; z < planes; ++z) {
                exec_plane(static_cast<int>(z), partials.data() + firsts[z], details, readIOp, reductions, writeIOp);
            }
        }
    };

#if defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)
    // Runs the chunks of ReduceDPP<ParArch::CPU> on cpuThreadPool(), and then accumulates and writes every plane with
    // the code of the CPU version, so both give the same results. The buffers that the IOps read and write must stay
    // valid until the execution has finished.
    template <typename Policy>
    struct ReduceDPP<ParArch::CPU_MULTITHREAD, Policy> {
    private:
        using SelfType = ReduceDPP<ParArch::CPU_MULTITHREAD, Policy>;
        using CPUDPP = ReduceDPP<ParArch::CPU, Policy>;
    public:
        FK_STATIC_STRUCT(ReduceDPP, SelfType)
        static constexpr ParArch PAR_ARCH = ParArch::CPU_MULTITHREAD;
        using Details = typename CPUDPP::Details;

        FK_HOST_FUSE Details build_details() {
            return CPUDPP::build_details();
        }
        FK_HOST_FUSE Details build_details(const RawPtr<ND::_1D, uchar>&) {
            return CPUDPP::build_details();
        }
        FK_HOST_STATIC Details build_details(const Ptr<ND::_1D, uchar>&) {
            return CPUDPP::build_details();
        }

        template <typename ReadIOp, typename Reductions>
        FK_HOST_FUSE size_t workspaceBytes(const ReadIOp&, const Reductions&) {
            return 0;
        }

        // Enqueues one chunk of work per chunk of every plane and returns; done is called when the results have been
        // written. If this throws (allocating the values of the chunks, or enqueueing), nothing was enqueued and done
        // is not called.
        template <typename ReadIOp, typename... FirstIOps, typename... AccumulationIOps, typename... FinalIOps,
                  typename WriteIOp>
        FK_HOST_STATIC void exec(const cpu_multithread::Completion& done, const Details& details,
                                 const ReadIOp& readIOp,
                                 const Tuple<Tuple<FirstIOps, AccumulationIOps, FinalIOps>...>& reductions,
                                 const WriteIOp& writeIOp) {
            using Reductions = Tuple<Tuple<FirstIOps, AccumulationIOps, FinalIOps>...>;
            using PartialType = reduce_detail::PartialOf<Reductions>;
            [[maybe_unused]] constexpr bool reported = reduce_detail::reportIOps<ReadIOp, Reductions, WriteIOp>;
            // The first chunk of every plane, and the values of the chunks, plane after plane
            const auto firsts = std::make_shared<const std::vector<size_t>>(CPUDPP::chunkFirsts(details, readIOp));
            const auto partials = std::make_shared<std::vector<PartialType>>(firsts->back());
            // The reductions are captured with an init-capture, a copy that is not const (as MorphologyDPP does):
            // a plain capture of the const reference would be a const member, and a const Tuple of one element could
            // not be moved before the constructor of fk::Tuple was constrained (tuple.h)
            cpu_multithread::enqueueChunks(firsts->back(),
                [details, readIOp, reductions = reductions, firsts, partials](const size_t work) {
                    // The plane of the chunk: the last one whose first chunk is not after it
                    const size_t z =
                        static_cast<size_t>(std::upper_bound(firsts->begin(), firsts->end(), work) - firsts->begin()) -
                        1;
                    (*partials)[work] = CPUDPP::exec_chunk(static_cast<int>(z), work - (*firsts)[z], details, readIOp,
                                                           reductions);
                },
                [done, details, readIOp, reductions = reductions, writeIOp, firsts,
                 partials](std::exception_ptr error) noexcept {
                    if (!error) {
                        try {
                            for (size_t z = 0; z + 1 < firsts->size(); ++z) {
                                CPUDPP::exec_plane(static_cast<int>(z), partials->data() + (*firsts)[z], details,
                                                   readIOp, reductions, writeIOp);
                            }
                        } catch (...) {
                            error = std::current_exception();
                        }
                    }
                    done(error);
                });
        }
    };
#endif // defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)

#if defined(__NVCC__) || defined(FK_HIP_BACKEND)
    namespace reduce_detail {
        // Read and Write of the values that the clusters of a plane store in the device memory of an execution (its
        // workspace): the value of cluster x of plane y. They are the IOps of the DPP itself, which, as any other
        // DPP, only reaches device memory through IOps. They declare their types by hand, and not through the parents
        // of the memory operations and the DECLARE_*_PARENT macros: the exec that DECLARE_WRITE_PARENT declares takes
        // a ThreadFusionType, which only exists for scalars and vectors (the values here are of any type that can be
        // stored), and Arrival only exists on the device.
        template <typename PartialType>
        struct PartialRead {
        private:
            using SelfType = PartialRead<PartialType>;
        public:
            FK_STATIC_STRUCT(PartialRead, SelfType)
            using ParamsType = RawPtr<ND::_2D, PartialType>;
            using ReadDataType = PartialType;
            using OutputType = PartialType;
            using InstanceType = ReadType;
            using OperationDataType = OperationData<SelfType>;
            using InstantiableType = Read<SelfType>;
            static constexpr bool IS_FUSED_OP = false;
            static constexpr bool THREAD_FUSION = false;
            FK_HOST_DEVICE_FUSE OutputType exec(const Point thread, const ParamsType& params) {
                return *PtrAccessor<ND::_2D>::cr_point(thread, params);
            }
            FK_HOST_DEVICE_FUSE OutputType exec(const Point thread, const OperationDataType& opData) {
                return exec(thread, opData.params);
            }
            FK_HOST_DEVICE_FUSE uint num_elems_x(const Point, const OperationDataType& opData) {
                return opData.params.dims.width;
            }
            FK_HOST_DEVICE_FUSE uint num_elems_y(const Point, const OperationDataType& opData) {
                return opData.params.dims.height;
            }
            FK_HOST_DEVICE_FUSE uint num_elems_z(const Point, const OperationDataType&) {
                return 1;
            }
            FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
                return {opData.params.dims.width, opData.params.dims.height, 1};
            }
            FK_HOST_FUSE InstantiableType build(const ParamsType& params) {
                return {{params}};
            }
        };

        template <typename PartialType>
        struct PartialWrite {
        private:
            using SelfType = PartialWrite<PartialType>;
        public:
            FK_STATIC_STRUCT(PartialWrite, SelfType)
            using ParamsType = RawPtr<ND::_2D, PartialType>;
            using InputType = PartialType;
            using WriteDataType = PartialType;
            using InstanceType = WriteType;
            using OperationDataType = OperationData<SelfType>;
            using InstantiableType = Write<SelfType>;
            static constexpr bool IS_FUSED_OP = false;
            static constexpr bool THREAD_FUSION = false;
            FK_HOST_DEVICE_STATIC void exec(const Point thread, const InputType& input, const ParamsType& params) {
                *PtrAccessor<ND::_2D>::point(thread, params) = input;
            }
            FK_HOST_DEVICE_STATIC void exec(const Point thread, const InputType& input,
                                            const OperationDataType& opData) {
                exec(thread, input, opData.params);
            }
            FK_HOST_DEVICE_FUSE uint num_elems_x(const Point, const OperationDataType& opData) {
                return opData.params.dims.width;
            }
            FK_HOST_DEVICE_FUSE uint num_elems_y(const Point, const OperationDataType& opData) {
                return opData.params.dims.height;
            }
            FK_HOST_DEVICE_FUSE uint num_elems_z(const Point, const OperationDataType&) {
                return 1;
            }
            FK_HOST_FUSE InstantiableType build(const ParamsType& params) {
                return {{params}};
            }
        };

        // The arrival of a cluster to the grid level of its plane, thread.x: counts it in the counter of the plane,
        // and gives the number of clusters of the plane that had arrived before. The counters are 0 when the
        // execution starts. It is a Read with a side effect (it counts), an IOp of this DPP for its own workspace:
        // it must not be fused, nor used by any other code.
        struct Arrival {
        private:
            using SelfType = Arrival;
        public:
            FK_STATIC_STRUCT(Arrival, SelfType)
            using ParamsType = RawPtr<ND::_1D, uint>;
            using ReadDataType = uint;
            using OutputType = uint;
            using InstanceType = ReadType;
            using OperationDataType = OperationData<SelfType>;
            using InstantiableType = Read<SelfType>;
            static constexpr bool IS_FUSED_OP = false;
            static constexpr bool THREAD_FUSION = false;
            FK_DEVICE_STATIC OutputType exec(const Point thread, const ParamsType& params) {
                return atomicAdd(PtrAccessor<ND::_1D>::point(thread, params), 1u);
            }
            FK_DEVICE_STATIC OutputType exec(const Point thread, const OperationDataType& opData) {
                return exec(thread, opData.params);
            }
            FK_HOST_DEVICE_FUSE uint num_elems_x(const Point, const OperationDataType& opData) {
                return opData.params.dims.width;
            }
            FK_HOST_DEVICE_FUSE uint num_elems_y(const Point, const OperationDataType&) {
                return 1;
            }
            FK_HOST_DEVICE_FUSE uint num_elems_z(const Point, const OperationDataType&) {
                return 1;
            }
            FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
                return {opData.params.dims.width, 1, 1};
            }
            FK_HOST_FUSE InstantiableType build(const ParamsType& params) {
                return {{params}};
            }
        };

        // The workspace of one execution, as the IOps that access it. Its values are not data of the user: they are
        // what the clusters of a plane exchange when they synchronize, as the shared memory of a block is for its
        // threads.
        template <typename PartialType>
        struct Workspace {
            // The value of cluster x of plane y
            Read<PartialRead<PartialType>> read;
            Write<PartialWrite<PartialType>> write;
            // The arrival counter of plane x: counted by every cluster, and set to 0 again by the last one to arrive
            Read<Arrival> arrival;
            Write<PerThreadWrite<ND::_1D, uint>> reset;
        };

        // The shared memory that the accumulated values of a block may use. A kernel can have 48 KB without asking
        // for more.
        constexpr size_t SHARED_MEMORY_BYTES = 32768;

        // The threads of a block: the ones of the policy, or the largest power of two below them whose accumulated
        // values fit in the shared memory
        template <typename Policy, typename PartialType>
        FK_HOST_DEVICE_CNST uint blockThreads() {
            static_assert(Policy::BLOCK_THREADS > 0 && Policy::BLOCK_THREADS <= 1024 &&
                          (Policy::BLOCK_THREADS & (Policy::BLOCK_THREADS - 1)) == 0,
                          "ReduceDPP: Policy::BLOCK_THREADS must be a power of two, of 1024 at most");
            static_assert(sizeof(PartialType) <= SHARED_MEMORY_BYTES,
                          "ReduceDPP: the accumulators of the reductions do not fit in the shared memory of a block");
            uint threads = Policy::BLOCK_THREADS;
            while (threads > 1 && threads * sizeof(PartialType) > SHARED_MEMORY_BYTES) {
                threads /= 2;
            }
            return threads;
        }

        // The accumulated values of one thread, as the shared memory of its block has them. A __shared__ variable is
        // never constructed, and the compilers only accept one of a type whose default constructor does nothing
        // (clang, the compiler of HIP, with an error), which the accumulators do not need to have (fk::Rect, a struct
        // with default member initializers): the constructor of a union constructs none of its members. (Not
        // constexpr: a constexpr constructor that initializes no member of a union is C++20.)
        template <typename PartialType>
        union SharedPartial {
            PartialType value;
            __host__ __device__ SharedPartial() {}
        };

        // The grid of one execution
        struct Launch {
            ulonglong blockElements; // elements of a block
            uint blocks;             // blocks of a plane in the grid
            uint clusterBlocks;      // blocks of a cluster; 1 without clusters
            uint clusters;           // clusters of a plane (its blocks, without clusters): more than 1 need a workspace
        };

        // The grid for `planes` planes of up to `elements` elements. clusterBlocks is the largest cluster that can be
        // launched, 1 when there are no clusters, and withWorkspace says whether the execution has a workspace.
        template <typename Policy>
        FK_HOST_CNST Launch launchOf(const ulonglong elements, const uint planes, const uint clusterBlocks,
                                     const bool withWorkspace) {
            static_assert(Policy::BLOCK_ELEMENTS > 0 && Policy::LARGE_BLOCK_ELEMENTS >= Policy::BLOCK_ELEMENTS,
                          "ReduceDPP: Policy::BLOCK_ELEMENTS must be greater than 0, and Policy::LARGE_BLOCK_ELEMENTS "
                          "not less than it");
            static_assert(Policy::BLOCKS > 0 && Policy::MAX_BLOCKS >= Policy::BLOCKS &&
                          Policy::MAX_BLOCKS <= (1u << 24),
                          "ReduceDPP: Policy::BLOCKS must be greater than 0, and Policy::MAX_BLOCKS between it and "
                          "2^24");
            if (elements <= Policy::SINGLE_BLOCK_ELEMENTS || planes == 0) {
                return {elements, 1, 1, 1};
            }
            // The blocks of all the planes together, as if `planes` planes of `elements` elements were one plane:
            // blocks of BLOCK_ELEMENTS elements, BLOCKS at most; unless they would have more than
            // LARGE_BLOCK_ELEMENTS elements: then, blocks of that size, MAX_BLOCKS at most
            constexpr ulonglong MOST = ~0ull;
            const ulonglong all = elements > MOST / planes ? MOST : elements * planes;
            ulonglong blocks = groupsOf(all, Policy::BLOCK_ELEMENTS);
            blocks = blocks < Policy::BLOCKS ? blocks : Policy::BLOCKS;
            const ulonglong largeBlocks = groupsOf(all, Policy::LARGE_BLOCK_ELEMENTS);
            blocks = blocks < largeBlocks ? largeBlocks : blocks;
            blocks = blocks < Policy::MAX_BLOCKS ? blocks : Policy::MAX_BLOCKS;
            // The part of each plane: one block at least, and no more blocks than its elements fill
            ulonglong planeBlocks = groupsOf(blocks, planes);
            const ulonglong fullBlocks = groupsOf(elements, Policy::BLOCK_ELEMENTS);
            planeBlocks = planeBlocks < fullBlocks ? planeBlocks : fullBlocks;
            const ulonglong blockElements = groupsOf(elements, planeBlocks);
            planeBlocks = groupsOf(elements, blockElements);
            if (planeBlocks <= 1) {
                // A read of so many planes that each one has one block
                return {elements, 1, 1, 1};
            }
            const bool always = Policy::CLUSTERS == ReduceDPPClusters::ALWAYS;
            const bool clustersAllowed = clusterBlocks > 1 && Policy::CLUSTERS != ReduceDPPClusters::NEVER;
            if (clustersAllowed && (always || !withWorkspace) &&
                elements <= static_cast<ulonglong>(clusterBlocks) * Policy::SINGLE_BLOCK_ELEMENTS) {
                // One cluster per plane, of blocks no larger than the one that reduces a plane alone: no workspace
                const ulonglong clusterBlockElements = groupsOf(elements, clusterBlocks);
                const uint used = static_cast<uint>(groupsOf(elements, clusterBlockElements));
                return {clusterBlockElements, used, used, 1};
            }
            if (clustersAllowed && always) {
                // The grid has whole clusters: its last blocks can have no elements
                const ulonglong clusters = groupsOf(planeBlocks, clusterBlocks);
                return {blockElements, static_cast<uint>(clusters * clusterBlocks), clusterBlocks,
                        static_cast<uint>(clusters)};
            }
            return {blockElements, static_cast<uint>(planeBlocks), 1, static_cast<uint>(planeBlocks)};
        }

        // The planes of a grid, at most
        constexpr uint MAX_PLANES = 65535;

        // The workspace: first the arrival counters, one per plane that a grid can have, and then the values of the
        // clusters of every plane. The counters are always in the same bytes, which no value is ever written to: zero
        // when the workspace is first used, and zero again after every execution, whatever its planes.
        constexpr size_t WORKSPACE_COUNTERS_BYTES = (size_t{MAX_PLANES} + 1) * sizeof(uint);

        template <typename PartialType>
        FK_HOST_CNST size_t workspaceBytesOf(const uint planes, const uint clusters) {
            return WORKSPACE_COUNTERS_BYTES + static_cast<size_t>(planes) * clusters * sizeof(PartialType);
        }

        // The bytes of the workspace that the executions of a read and some reductions need: what the grid without
        // clusters needs, which is the one with most values to store. 0 when every plane is reduced by one block.
        template <typename Policy, typename ReadIOp, typename Reductions>
        FK_HOST_CNST size_t workspaceBytes(const ReadIOp& readIOp, const Reductions&) {
            const Launch launch = launchOf<Policy>(largestPlaneElements(readIOp), planesOf(readIOp), 1, true);
            return launch.clusters > 1 ? workspaceBytesOf<PartialOf<Reductions>>(planesOf(readIOp), launch.clusters)
                                       : size_t{0};
        }

        // The workspace of a caller, as the details of the GPU versions have it. A Ptr has to be device memory.
        FK_HOST_STATIC RawPtr<ND::_1D, uchar> workspaceOf(const Ptr<ND::_1D, uchar>& workspace) {
            const RawPtr<ND::_1D, uchar> memory = workspace.ptr();
            executor_detail::requireArguments(memory.data == nullptr || workspace.getMemType() == MemType::Device ||
                                              workspace.getMemType() == MemType::DeviceAndPinned,
                                              "ReduceDPP", "the workspace must be device memory");
            return memory;
        }

        // The IOps of the workspace of an execution whose planes have `clusters` clusters, in the memory of the
        // caller, which has to have the `bytes` that workspaceBytes gives for the read and the reductions (the same
        // on every device, although a grid with clusters stores fewer values). Throws std::invalid_argument when
        // there is no memory, or not enough. A workspace has less than 4 GB, the largest size of a 1D pointer: a read
        // that needs more can not be executed.
        template <typename PartialType>
        FK_HOST_STATIC Workspace<PartialType> workspaceIOps(const RawPtr<ND::_1D, uchar>& memory, const uint planes,
                                                            const uint clusters, const size_t bytes) {
            if (memory.data == nullptr || memory.dims.width < bytes) {
                const std::string what =
                    "the planes of the read need a workspace of " + std::to_string(bytes) +
                    " bytes of device memory to synchronize their blocks (ReduceDPP<...>::workspaceBytes; less than "
                    "4 GB at most), and the execution has " +
                    (memory.data == nullptr ? std::string("none") : std::to_string(memory.dims.width) + " bytes");
                executor_detail::requireArguments(false, "ReduceDPP", what.c_str());
            }
            const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(memory.data);
            executor_detail::requireArguments(address % alignof(PartialType) == 0 && address % alignof(uint) == 0,
                                              "ReduceDPP", "the workspace is not aligned for the accumulated values");
            static_assert(WORKSPACE_COUNTERS_BYTES % alignof(PartialType) == 0,
                          "ReduceDPP: the alignment of the accumulated values must divide the bytes of the counters");
            const RawPtr<ND::_1D, uint> counters{reinterpret_cast<uint*>(memory.data),
                                                 PtrDims<ND::_1D>(planes, planes * static_cast<uint>(sizeof(uint)))};
            const RawPtr<ND::_2D, PartialType> values{
                reinterpret_cast<PartialType*>(memory.data + WORKSPACE_COUNTERS_BYTES),
                PtrDims<ND::_2D>(clusters, planes, static_cast<uint>(clusters * sizeof(PartialType)))};
            return {PartialRead<PartialType>::build(values), PartialWrite<PartialType>::build(values),
                    Arrival::build(counters), PerThreadWrite<ND::_1D, uint>::build(counters)};
        }
    } // namespace reduce_detail

    // What the kernel gets from its executor: how the elements of a plane are split in blocks, and the blocks in
    // clusters, and the workspace of the planes that need more than one cluster.
    template <typename PartialType>
    struct ReduceDPPKernelDetails {
        // Elements that a block reduces: block b of a plane, the elements from b * blockElements on, in row major
        // order
        ulonglong blockElements;
        // Blocks of the thread block clusters that the kernel is launched with, consecutive in x; 1 without clusters
        uint clusterBlocks;
        // Only used when a plane has more than one cluster
        reduce_detail::Workspace<PartialType> workspace;
    };

    // The code of a thread of the kernel, shared by the GPU versions
    template <typename Policy>
    struct ReduceDPPBase {
        friend struct ReduceDPP<ParArch::GPU_NVIDIA, Policy>;
        friend struct ReduceDPP<ParArch::GPU_AMD, Policy>;
    private:
        template <typename PartialType, typename ReadIOp, typename... FirstIOps, typename... AccumulationIOps,
                  typename... FinalIOps, typename WriteIOp>
        FK_DEVICE_STATIC void exec(const ReduceDPPKernelDetails<PartialType>& details, const ReadIOp& readIOp,
                                   const Tuple<Tuple<FirstIOps, AccumulationIOps, FinalIOps>...>& reductions,
                                   const WriteIOp& writeIOp) {
            using Reductions = Tuple<Tuple<FirstIOps, AccumulationIOps, FinalIOps>...>;
            static_assert(std::is_same_v<PartialType, reduce_detail::PartialOf<Reductions>>,
                          "ReduceDPP: the details are not the ones of these reductions");
            [[maybe_unused]] constexpr bool reported = reduce_detail::reportIOps<ReadIOp, Reductions, WriteIOp>;
            constexpr reduce_detail::IndexesOf<Reductions> indexes{};
            constexpr uint THREADS = reduce_detail::blockThreads<Policy, PartialType>();

            // The accumulated values of the threads of the block and, in [0], the one of the block (and then the one
            // of its cluster, and the one of its plane)
            __shared__ reduce_detail::SharedPartial<PartialType> partials[THREADS];

            const uint tid = threadIdx.x;
            const uint block = blockIdx.x;
            const int z = static_cast<int>(blockIdx.z);
            const uint width = NumElems::x(Point{0, 0, z}, readIOp);
            const ulonglong elements = reduce_detail::planeElements(z, readIOp);

            const auto accumulate = [&reductions, &indexes](const PartialType& a, const PartialType& b) {
                return reduce_detail::accumulate<Reductions>(a, b, reductions, indexes);
            };
            // The first `entries` values of partials, accumulated in partials[0]. The same for every thread of the
            // block, which all get here. (A lambda uses a __shared__ variable without capturing it.)
            const auto accumulateShared = [&accumulate, tid](const uint entries) {
                for (uint stride = THREADS / 2; stride > 0; stride /= 2) {
                    if (tid < stride && tid + stride < entries) {
                        partials[tid].value = accumulate(partials[tid].value, partials[tid + stride].value);
                    }
                    __syncthreads();
                }
            };
            // The values value(0) ... value(count - 1) accumulated in partials[0]: each thread accumulates one of
            // every THREADS of them, and then all do in shared memory. value(0) can be the partials[0] of this block
            // (the cluster level): only thread 0 reads it, before it writes partials[0].
            const auto accumulateValues = [&accumulate, &accumulateShared, tid](const ulonglong count,
                                                                                const auto& value) {
                if (tid < count) {
                    PartialType accumulated = value(static_cast<ulonglong>(tid));
                    for (ulonglong i = static_cast<ulonglong>(tid) + THREADS; i < count; i += THREADS) {
                        accumulated = accumulate(accumulated, value(i));
                    }
                    partials[tid].value = accumulated;
                }
                __syncthreads();
                accumulateShared(count < THREADS ? static_cast<uint>(count) : THREADS);
            };

#if defined(__NVCC__) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
            // This image of the kernel is for an architecture with thread block clusters: it forms the ones that the
            // kernel is launched with
            const uint clusterBlocks = details.clusterBlocks;
            const ulonglong blockElements = details.blockElements;
            const ulonglong blocks = reduce_detail::groupsOf(elements, blockElements);
#else
            // An image without clusters: every block is alone. The executor does not launch it with clusters (it asks
            // the driver for the image that runs); if it ever is, there is no workspace for the blocks to synchronize
            // with, and the first block of every plane reduces it whole.
            const uint clusterBlocks = 1;
            const ulonglong blockElements = details.clusterBlocks > 1 ? elements : details.blockElements;
            const ulonglong blocks = details.clusterBlocks > 1 ? (elements > 0 ? 1ull : 0ull)
                                                               : reduce_detail::groupsOf(elements, blockElements);
#endif
            const ulonglong cluster = block / clusterBlocks;
            const ulonglong clusters = reduce_detail::groupsOf(blocks, clusterBlocks);
            // The grid covers the largest plane: the clusters past the elements of this one have nothing to do. All
            // the blocks of a cluster return here, or none.
            if (cluster >= clusters) {
                return;
            }

            // 1. Thread: the elements first + tid, first + tid + THREADS... of the block, in registers. The blocks at
            // the end of the last cluster of a plane can have no elements.
            const ulonglong first = block * blockElements;
            const ulonglong count = block >= blocks ? 0ull
                                                    : (elements - first < blockElements ? elements - first
                                                                                        : blockElements);
            if (tid < count) {
                const ulonglong element = first + tid;
                uint y = static_cast<uint>(element / width);
                uint x = static_cast<uint>(element - static_cast<ulonglong>(y) * width);
                // The position of the element THREADS elements after (x, y): stepX columns and stepY rows further
                const uint stepX = THREADS % width;
                const uint stepY = THREADS / width;
                PartialType accumulated = reduce_detail::firstValues(
                    ReadIOp::Operation::exec(Point{static_cast<int>(x), static_cast<int>(y), z}, readIOp), reductions,
                    indexes);
                for (ulonglong i = static_cast<ulonglong>(tid) + THREADS; i < count; i += THREADS) {
                    y += stepY;
                    if (x >= width - stepX) {
                        x -= width - stepX;
                        ++y;
                    } else {
                        x += stepX;
                    }
                    accumulated = accumulate(
                        accumulated,
                        reduce_detail::firstValues(
                            ReadIOp::Operation::exec(Point{static_cast<int>(x), static_cast<int>(y), z}, readIOp),
                            reductions, indexes));
                }
                partials[tid].value = accumulated;
            }
            __syncthreads();
            // 2. Block: the values of its threads, in shared memory
            accumulateShared(count < THREADS ? static_cast<uint>(count) : THREADS);

#if defined(__NVCC__) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
            // 3. Cluster: the values of its blocks, which its first block reads from their shared memory
            if (clusterBlocks > 1) {
                const cg::cluster_group clusterGroup = cg::this_cluster();
                // Every block of the cluster has its value in its partials[0]
                clusterGroup.sync();
                if (clusterGroup.block_rank() == 0) {
                    // The blocks of this cluster that have elements
                    const ulonglong used = blocks - cluster * clusterBlocks < clusterBlocks
                                               ? blocks - cluster * clusterBlocks
                                               : static_cast<ulonglong>(clusterBlocks);
                    accumulateValues(used, [&clusterGroup](const ulonglong rank) {
                        return clusterGroup.map_shared_rank(&partials[0], static_cast<int>(rank))->value;
                    });
                }
                // The shared memory of a block lasts until the first block of its cluster has read it
                clusterGroup.sync();
                if (clusterGroup.block_rank() != 0) {
                    return;
                }
            }
#endif
            if (clusters == 1) {
                // The value of the plane never left the block, or the cluster
                if (tid == 0) {
                    reduce_detail::finalValues(z, partials[0].value, reductions, writeIOp, indexes);
                }
                return;
            }

            // 4. Grid: the values of the clusters of the plane, which the last one to store its own finds in the
            // workspace
            __shared__ bool lastToArrive;
            using PartialReadOp = reduce_detail::PartialRead<PartialType>;
            using PartialWriteOp = reduce_detail::PartialWrite<PartialType>;
            const int slot = static_cast<int>(cluster);
            if (tid == 0) {
                PartialWriteOp::exec(Point{slot, z, 0}, partials[0].value, details.workspace.write);
                // The value is visible to every thread of the device before this cluster counts itself as arrived
                __threadfence();
                lastToArrive = reduce_detail::Arrival::exec(Point{z, 0, 0}, details.workspace.arrival) ==
                               static_cast<uint>(clusters - 1);
                if (lastToArrive) {
                    // Every cluster of the plane has counted itself: the counter is zero again for the next execution
                    PerThreadWrite<ND::_1D, uint>::exec(Point{z, 0, 0}, 0u, details.workspace.reset);
                }
            }
            __syncthreads();
            if (!lastToArrive) {
                return;
            }
            accumulateValues(clusters, [&details, z](const ulonglong other) {
                return PartialReadOp::exec(Point{static_cast<int>(other), z, 0}, details.workspace.read);
            });
            if (tid == 0) {
                reduce_detail::finalValues(z, partials[0].value, reductions, writeIOp, indexes);
            }
        }
    };
#endif // defined(__NVCC__) || defined(FK_HIP_BACKEND)

#if defined(__NVCC__)
    template <typename Policy>
    struct ReduceDPP<ParArch::GPU_NVIDIA, Policy> {
    private:
        using SelfType = ReduceDPP<ParArch::GPU_NVIDIA, Policy>;
        using Parent = ReduceDPPBase<Policy>;
    public:
        FK_STATIC_STRUCT(ReduceDPP, SelfType)
        static constexpr ParArch PAR_ARCH = ParArch::GPU_NVIDIA;
        using Details = ReduceDPPDetails<ParArch::GPU_NVIDIA>;
        template <typename PartialType>
        using KernelDetails = ReduceDPPKernelDetails<PartialType>;

        // The details of an execution: without a workspace, or with the one of the caller
        FK_HOST_FUSE Details build_details() {
            return Details{};
        }
        FK_HOST_FUSE Details build_details(const RawPtr<ND::_1D, uchar>& workspace) {
            return Details{workspace};
        }
        FK_HOST_STATIC Details build_details(const Ptr<ND::_1D, uchar>& workspace) {
            return Details{reduce_detail::workspaceOf(workspace)};
        }

        // The bytes of the workspace that the executions of a read and some reductions need on any device; 0 when
        // they need none
        template <typename ReadIOp, typename Reductions>
        FK_HOST_FUSE size_t workspaceBytes(const ReadIOp& readIOp, const Reductions& reductions) {
            return reduce_detail::workspaceBytes<Policy>(readIOp, reductions);
        }

        // The code of every thread of the kernel
        template <typename PartialType, typename ReadIOp, typename... FirstIOps, typename... AccumulationIOps,
                  typename... FinalIOps, typename WriteIOp>
        FK_DEVICE_STATIC void exec(const KernelDetails<PartialType>& details, const ReadIOp& readIOp,
                                   const Tuple<Tuple<FirstIOps, AccumulationIOps, FinalIOps>...>& reductions,
                                   const WriteIOp& writeIOp) {
            Parent::exec(details, readIOp, reductions, writeIOp);
        }
    };
#elif defined(FK_HIP_BACKEND)
    // The GPU_NVIDIA version, without clusters
    template <typename Policy>
    struct ReduceDPP<ParArch::GPU_AMD, Policy> {
    private:
        using SelfType = ReduceDPP<ParArch::GPU_AMD, Policy>;
        using Parent = ReduceDPPBase<Policy>;
    public:
        FK_STATIC_STRUCT(ReduceDPP, SelfType)
        static constexpr ParArch PAR_ARCH = ParArch::GPU_AMD;
        using Details = ReduceDPPDetails<ParArch::GPU_AMD>;
        template <typename PartialType>
        using KernelDetails = ReduceDPPKernelDetails<PartialType>;

        // The details of an execution: without a workspace, or with the one of the caller
        FK_HOST_FUSE Details build_details() {
            return Details{};
        }
        FK_HOST_FUSE Details build_details(const RawPtr<ND::_1D, uchar>& workspace) {
            return Details{workspace};
        }
        FK_HOST_STATIC Details build_details(const Ptr<ND::_1D, uchar>& workspace) {
            return Details{reduce_detail::workspaceOf(workspace)};
        }

        // The bytes of the workspace that the executions of a read and some reductions need; 0 when they need none
        template <typename ReadIOp, typename Reductions>
        FK_HOST_FUSE size_t workspaceBytes(const ReadIOp& readIOp, const Reductions& reductions) {
            return reduce_detail::workspaceBytes<Policy>(readIOp, reductions);
        }

        // The code of every thread of the kernel
        template <typename PartialType, typename ReadIOp, typename... FirstIOps, typename... AccumulationIOps,
                  typename... FinalIOps, typename WriteIOp>
        FK_DEVICE_STATIC void exec(const KernelDetails<PartialType>& details, const ReadIOp& readIOp,
                                   const Tuple<Tuple<FirstIOps, AccumulationIOps, FinalIOps>...>& reductions,
                                   const WriteIOp& writeIOp) {
            Parent::exec(details, readIOp, reductions, writeIOp);
        }
    };
#endif // defined(__NVCC__) || defined(FK_HIP_BACKEND)

    // The executors of ReduceDPP, one per back-end:
    //   executeOperations<ReduceDPP<PA, Policy>>(stream, [details,] readIOp, reductions, writeIOp)
    // with reductions = make_tuple(make_tuple(firstIOp, accumulationIOp, finalIOp), ...), and details =
    // ReduceDPP<PA, Policy>::build_details(workspace) for the executions that need a workspace. A read without
    // elements, a write without the planes of the read or without room for the reductions, a workspace that is
    // missing or too small and details of the CPU versions with chunks of 0 elements throw std::invalid_argument
    // before anything is launched or enqueued.
    template <typename Policy>
    struct Executor<ReduceDPP<ParArch::CPU, Policy>> {
    private:
        using DPP = ReduceDPP<ParArch::CPU, Policy>;
        using SelfType = Executor<DPP>;
    public:
        FK_STATIC_STRUCT(Executor, SelfType)
        FK_HOST_FUSE ParArch parArch() {
            return ParArch::CPU;
        }

        // The results are written when this returns
        template <typename ReadIOp, typename Reductions, typename WriteIOp>
        FK_HOST_STATIC void executeOperations(Stream_<ParArch::CPU>&, const typename DPP::Details& details,
                                              const ReadIOp& readIOp, const Reductions& reductions,
                                              const WriteIOp& writeIOp) {
            if constexpr (reduce_detail::iopsAreValid<ReadIOp, Reductions, WriteIOp>) {
                reduce_detail::checkArguments(readIOp, reductions, writeIOp);
                reduce_detail::checkChunks(details.chunkElements);
                DPP::exec(details, readIOp, reductions, writeIOp);
            } else {
                // Wrong IOps: the report names them, and nothing else is instantiated
                [[maybe_unused]] constexpr bool reported = reduce_detail::reportIOps<ReadIOp, Reductions, WriteIOp>;
            }
        }
        template <typename ReadIOp, typename Reductions, typename WriteIOp>
        FK_HOST_STATIC void executeOperations(Stream_<ParArch::CPU>& stream, const ReadIOp& readIOp,
                                              const Reductions& reductions, const WriteIOp& writeIOp) {
            executeOperations(stream, DPP::build_details(), readIOp, reductions, writeIOp);
        }
    };

#if defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)
    // Enqueues the execution on the stream and returns. The execution keeps copies of the IOps: the buffers that they
    // read and write must stay valid until it has finished (sync()).
    template <typename Policy>
    struct Executor<ReduceDPP<ParArch::CPU_MULTITHREAD, Policy>> {
    private:
        using DPP = ReduceDPP<ParArch::CPU_MULTITHREAD, Policy>;
        using SelfType = Executor<DPP>;
    public:
        FK_STATIC_STRUCT(Executor, SelfType)
        FK_HOST_FUSE ParArch parArch() {
            return ParArch::CPU_MULTITHREAD;
        }

        template <typename ReadIOp, typename Reductions, typename WriteIOp>
        FK_HOST_STATIC void executeOperations(Stream_<ParArch::CPU_MULTITHREAD>& stream,
                                              const typename DPP::Details& details, const ReadIOp& readIOp,
                                              const Reductions& reductions, const WriteIOp& writeIOp) {
            if constexpr (reduce_detail::iopsAreValid<ReadIOp, Reductions, WriteIOp>) {
                reduce_detail::checkArguments(readIOp, reductions, writeIOp);
                reduce_detail::checkChunks(details.chunkElements);
                // The reductions are captured with an init-capture, as in exec
                stream.enqueue([details, readIOp, reductions = reductions,
                                writeIOp](const cpu_multithread::Completion& done) {
                    DPP::exec(done, details, readIOp, reductions, writeIOp);
                });
            } else {
                // Wrong IOps: the report names them, and nothing else is instantiated
                [[maybe_unused]] constexpr bool reported = reduce_detail::reportIOps<ReadIOp, Reductions, WriteIOp>;
            }
        }
        template <typename ReadIOp, typename Reductions, typename WriteIOp>
        FK_HOST_STATIC void executeOperations(Stream_<ParArch::CPU_MULTITHREAD>& stream, const ReadIOp& readIOp,
                                              const Reductions& reductions, const WriteIOp& writeIOp) {
            executeOperations(stream, DPP::build_details(), readIOp, reductions, writeIOp);
        }
    };
#endif // defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)

#if defined(__NVCC__)
    namespace reduce_detail {
        // The architectures that the kernels of this translation unit are compiled for, as values of __CUDA_ARCH__
        // (890 for compute capability 8.9): nvcc defines the list for the host code too. A compiler that does not
        // define it gives no clusters.
#if defined(__CUDA_ARCH_LIST__)
        constexpr int CUDA_ARCHITECTURES[] = {__CUDA_ARCH_LIST__};
#else
        constexpr int CUDA_ARCHITECTURES[] = {0};
#endif
        // The first architecture with thread block clusters
        constexpr int CLUSTERS_ARCHITECTURE = 900;

        FK_HOST_CNST bool anyImageHasClusters() {
            bool any = false;
            for (const int architecture : CUDA_ARCHITECTURES) {
                any = any || architecture >= CLUSTERS_ARCHITECTURE;
            }
            return any;
        }

        FK_HOST_CNST bool everyImageHasClusters() {
            bool every = true;
            for (const int architecture : CUDA_ARCHITECTURES) {
                every = every && architecture >= CLUSTERS_ARCHITECTURE;
            }
            return every;
        }

        // The error of a call that returns it, used as gpuErrchk(clearingLastError(call)). Such an error also stays as
        // the last error of the thread: it is cleared there, because the other launches of the library are checked
        // with cudaGetLastError(), which would report it for the next execution, one that has nothing wrong.
        inline cudaError_t clearingLastError(const cudaError_t code) {
            if (code != cudaSuccess) {
                (void)cudaGetLastError();
            }
            return code;
        }

        // The blocks of the largest cluster that a kernel can be launched with on a stream, in blocks of `threads`
        // threads: `wanted` at most, and 1 when it can not be launched with clusters. Only for builds that have
        // images with clusters (anyImageHasClusters()). The driver says:
        // - in a build with images of both kinds, which one it runs on the device of the stream: the architecture
        //   that the image of the kernel was compiled for, as major * 10 + minor. (It can not be deduced from the
        //   compute capability of the device: the driver prefers a binary for the device to the code of a newer
        //   architecture that it would have to compile, and a binary for another device does not run on this one.)
        // - the largest cluster that the stream accepts: 8 blocks is the largest portable size, and a context with
        //   few multiprocessors that are scheduled together accepts less.
        template <typename Kernel>
        inline uint largestCluster(const Kernel kernel, const uint threads, const cudaStream_t stream,
                                   const uint wanted) {
            if constexpr (!everyImageHasClusters()) {
                cudaFuncAttributes attributes{};
                gpuErrchk(clearingLastError(cudaFuncGetAttributes(&attributes, kernel)));
                if (attributes.ptxVersion * 10 < CLUSTERS_ARCHITECTURE) {
                    return 1;
                }
            }
            cudaLaunchConfig_t config = {};
            config.gridDim = dim3(wanted, 1, 1);
            config.blockDim = dim3(threads, 1, 1);
            config.stream = stream;
            int largest = 0;
            gpuErrchk(clearingLastError(cudaOccupancyMaxPotentialClusterSize(&largest, kernel, &config)));
            if (largest < 2) {
                return 1;
            }
            return static_cast<uint>(largest) < wanted ? static_cast<uint>(largest) : wanted;
        }

        // Throws, with the reason, when the device can not launch the kernel in blocks of `threads` threads, which
        // only depends on the policy and on the build. Asked before the launch by the policies with blocks of more
        // than 256 threads, and by every policy for a launch that the device rejected for its resources.
        template <typename Kernel>
        inline void throwIfTooManyThreads(const Kernel kernel, const uint threads) {
            cudaFuncAttributes attributes{};
            if (clearingLastError(cudaFuncGetAttributes(&attributes, kernel)) == cudaSuccess &&
                attributes.maxThreadsPerBlock < static_cast<int>(threads)) {
                throw std::runtime_error("ReduceDPP: this kernel needs " + std::to_string(attributes.numRegs) +
                                         " registers per thread, so the device launches it in blocks of " +
                                         std::to_string(attributes.maxThreadsPerBlock) +
                                         " threads at most, and Policy::BLOCK_THREADS gives blocks of " +
                                         std::to_string(threads) + " (blocks of 256 threads can always be launched)");
            }
        }

        // For a launch that the device rejected for its memory: throws the reason when it is the stack of the kernel.
        // The accumulated values that the compiler can not keep in registers (the ones with arrays) are in the stack
        // of each thread, several times, and a kernel that needs more stack than the context has makes it reserve
        // that stack for every thread that the device can run at a time, whatever the threads of the block are.
        template <typename Kernel>
        inline void throwIfNoMemoryForTheStack(const Kernel kernel, const size_t accumulatorsBytes) {
            cudaFuncAttributes attributes{};
            size_t stackBytes = 0;
            if (clearingLastError(cudaFuncGetAttributes(&attributes, kernel)) == cudaSuccess &&
                clearingLastError(cudaDeviceGetLimit(&stackBytes, cudaLimitStackSize)) == cudaSuccess &&
                attributes.localSizeBytes > stackBytes) {
                throw std::runtime_error("ReduceDPP: the device has no memory for the stack of this kernel, " +
                                         std::to_string(attributes.localSizeBytes) +
                                         " bytes for every thread that it can run at a time: the accumulated values "
                                         "of the reductions have " + std::to_string(accumulatorsBytes) +
                                         " bytes, and the kernel can not keep them in registers");
            }
        }

    } // namespace reduce_detail

    // Launches the kernel on the stream and returns
    template <typename Policy>
    struct Executor<ReduceDPP<ParArch::GPU_NVIDIA, Policy>> {
    private:
        using DPP = ReduceDPP<ParArch::GPU_NVIDIA, Policy>;
        using SelfType = Executor<DPP>;

        template <typename ReadIOp, typename Reductions, typename WriteIOp>
        FK_HOST_STATIC void executeOperations_checked(Stream_<ParArch::GPU_NVIDIA>& stream_,
                                                      const typename DPP::Details& details, const ReadIOp& readIOp,
                                                      const Reductions& reductions, const WriteIOp& writeIOp) {
            using PartialType = reduce_detail::PartialOf<Reductions>;
            using KernelDetails = typename DPP::template KernelDetails<PartialType>;
            static_assert(Policy::CLUSTER_BLOCKS >= 1 && Policy::CLUSTER_BLOCKS <= 8,
                          "ReduceDPP: Policy::CLUSTER_BLOCKS must be between 1 and 8, the largest portable cluster "
                          "size");
            reduce_detail::checkArguments(readIOp, reductions, writeIOp);
            const uint planes = reduce_detail::planesOf(readIOp);
            executor_detail::requireArguments(planes <= reduce_detail::MAX_PLANES, "ReduceDPP",
                                              "the read has more than 65535 planes, the most that a grid has");
            const cudaStream_t stream = stream_.getCUDAStream();
            const uint threads = reduce_detail::blockThreads<Policy, PartialType>();
            const ulonglong elements = reduce_detail::largestPlaneElements(readIOp);
            // The kernel of every DPP, launched here with cudaLaunchKernelEx (the launch that takes the clusters),
            // which needs its address. The cast is a C style one, to the parameter types with their const, as in the
            // launch code that nvcc generates: g++ does not find the kernel through any other conversion.
            const auto kernel = (void (*)(const KernelDetails, const ReadIOp, const Reductions, const WriteIOp))
                launchDPP_Kernel<DPP, KernelDetails, ReadIOp, Reductions, WriteIOp>;
            // Blocks of more than 256 threads, which the kernel may not be launchable with, are checked before the
            // launch: one that is being captured in a graph is not rejected (the instantiation of the graph is)
            if constexpr (reduce_detail::blockThreads<Policy, PartialType>() > 256) {
                reduce_detail::throwIfTooManyThreads(kernel, threads);
            }

            // The grid. A build without images with clusters has no code for them. In the others, a grid that has
            // clusters asks the driver for the largest one that can really be launched here: the executions that
            // launch no clusters ask nothing (with the default policy, the ones that have a workspace, and the ones
            // of planes that one block reduces).
            const bool withWorkspace = details.workspace.data != nullptr;
            uint clusterBlocks = 1;
            if constexpr (reduce_detail::anyImageHasClusters()) {
                clusterBlocks = Policy::CLUSTERS != ReduceDPPClusters::NEVER ? Policy::CLUSTER_BLOCKS : 1u;
            }
            reduce_detail::Launch launch =
                reduce_detail::launchOf<Policy>(elements, planes, clusterBlocks, withWorkspace);
            if constexpr (reduce_detail::anyImageHasClusters()) {
                if (launch.clusterBlocks > 1) {
                    const uint largest = reduce_detail::largestCluster(kernel, threads, stream, clusterBlocks);
                    if (largest < clusterBlocks) {
                        launch = reduce_detail::launchOf<Policy>(elements, planes, largest, withWorkspace);
                    }
                }
            }

            KernelDetails kernelDetails{};
            kernelDetails.blockElements = launch.blockElements;
            kernelDetails.clusterBlocks = launch.clusterBlocks;
            if (launch.clusters > 1) {
                // The clusters of a plane synchronize through the workspace of the caller
                kernelDetails.workspace =
                    reduce_detail::workspaceIOps<PartialType>(details.workspace, planes, launch.clusters,
                                                              DPP::workspaceBytes(readIOp, reductions));
            }

            cudaLaunchConfig_t config = {};
            config.gridDim = dim3(launch.blocks, 1, planes);
            config.blockDim = dim3(threads, 1, 1);
            config.dynamicSmemBytes = 0;
            config.stream = stream;
            cudaLaunchAttribute attribute[1];
            if (launch.clusterBlocks > 1) {
                attribute[0].id = cudaLaunchAttributeClusterDimension;
                attribute[0].val.clusterDim.x = launch.clusterBlocks;
                attribute[0].val.clusterDim.y = 1;
                attribute[0].val.clusterDim.z = 1;
                config.attrs = attribute;
                config.numAttrs = 1;
            }
            const cudaError_t code = reduce_detail::clearingLastError(
                cudaLaunchKernelEx(&config, kernel, kernelDetails, readIOp, reductions, writeIOp));
            if (code == cudaErrorLaunchOutOfResources) {
                reduce_detail::throwIfTooManyThreads(kernel, threads);
            }
            if (code == cudaErrorMemoryAllocation) {
                reduce_detail::throwIfNoMemoryForTheStack(kernel, sizeof(PartialType));
            }
            gpuErrchk(code);
        }

    public:
        FK_STATIC_STRUCT(Executor, SelfType)
        FK_HOST_FUSE ParArch parArch() {
            return ParArch::GPU_NVIDIA;
        }

        template <typename ReadIOp, typename Reductions, typename WriteIOp>
        FK_HOST_STATIC void executeOperations(Stream_<ParArch::GPU_NVIDIA>& stream,
                                              const typename DPP::Details& details, const ReadIOp& readIOp,
                                              const Reductions& reductions, const WriteIOp& writeIOp) {
            if constexpr (reduce_detail::iopsAreValid<ReadIOp, Reductions, WriteIOp>) {
                executeOperations_checked(stream, details, readIOp, reductions, writeIOp);
            } else {
                // Wrong IOps: the report names them, and nothing else is instantiated
                [[maybe_unused]] constexpr bool reported = reduce_detail::reportIOps<ReadIOp, Reductions, WriteIOp>;
            }
        }
        template <typename ReadIOp, typename Reductions, typename WriteIOp>
        FK_HOST_STATIC void executeOperations(Stream_<ParArch::GPU_NVIDIA>& stream, const ReadIOp& readIOp,
                                              const Reductions& reductions, const WriteIOp& writeIOp) {
            executeOperations(stream, DPP::build_details(), readIOp, reductions, writeIOp);
        }
    };
#elif defined(FK_HIP_BACKEND)
    // Launches the kernel on the stream and returns
    template <typename Policy>
    struct Executor<ReduceDPP<ParArch::GPU_AMD, Policy>> {
    private:
        using DPP = ReduceDPP<ParArch::GPU_AMD, Policy>;
        using SelfType = Executor<DPP>;

    public:
        FK_STATIC_STRUCT(Executor, SelfType)
        FK_HOST_FUSE ParArch parArch() {
            return ParArch::GPU_AMD;
        }

        template <typename ReadIOp, typename Reductions, typename WriteIOp>
        FK_HOST_STATIC void executeOperations(Stream_<ParArch::GPU_AMD>& stream,
                                              const typename DPP::Details& details, const ReadIOp& readIOp,
                                              const Reductions& reductions, const WriteIOp& writeIOp) {
            if constexpr (reduce_detail::iopsAreValid<ReadIOp, Reductions, WriteIOp>) {
                using PartialType = reduce_detail::PartialOf<Reductions>;
                using KernelDetails = typename DPP::template KernelDetails<PartialType>;
                reduce_detail::checkArguments(readIOp, reductions, writeIOp);
                const uint planes = reduce_detail::planesOf(readIOp);
                executor_detail::requireArguments(planes <= reduce_detail::MAX_PLANES, "ReduceDPP",
                                                  "the read has more than 65535 planes, the most that a grid has");
                // No clusters: the blocks of a plane synchronize through the workspace of the caller
                const reduce_detail::Launch launch =
                    reduce_detail::launchOf<Policy>(reduce_detail::largestPlaneElements(readIOp), planes, 1, true);
                KernelDetails kernelDetails{};
                kernelDetails.blockElements = launch.blockElements;
                kernelDetails.clusterBlocks = launch.clusterBlocks;
                if (launch.clusters > 1) {
                    kernelDetails.workspace =
                        reduce_detail::workspaceIOps<PartialType>(details.workspace, planes, launch.clusters,
                                                                  DPP::workspaceBytes(readIOp, reductions));
                }
                const dim3 block(reduce_detail::blockThreads<Policy, PartialType>(), 1, 1);
                const dim3 grid(launch.blocks, 1, planes);
                launchDPP_Kernel<DPP><<<grid, block, 0, stream.getHIPStream()>>>(kernelDetails, readIOp, reductions,
                                                                                 writeIOp);
                gpuErrchk(hipGetLastError());
            } else {
                // Wrong IOps: the report names them, and nothing else is instantiated
                [[maybe_unused]] constexpr bool reported = reduce_detail::reportIOps<ReadIOp, Reductions, WriteIOp>;
            }
        }
        template <typename ReadIOp, typename Reductions, typename WriteIOp>
        FK_HOST_STATIC void executeOperations(Stream_<ParArch::GPU_AMD>& stream, const ReadIOp& readIOp,
                                              const Reductions& reductions, const WriteIOp& writeIOp) {
            executeOperations(stream, DPP::build_details(), readIOp, reductions, writeIOp);
        }
    };
#endif // defined(__NVCC__) || defined(FK_HIP_BACKEND)
} // namespace fk

#endif // FK_REDUCE_DPP_H
