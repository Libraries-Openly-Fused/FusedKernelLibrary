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

// ReduceDPP: from 1 to N reductions of the same data, reading every element once.
// - float, float3, float4 and double2 matrices of many sizes (1x1, rows, columns, the sizes around those of a block
//   and of a chunk, images): the sum, the maximum and the minimum of each channel, with accumulators of the type of
//   the elements, are three values of that type, and nothing else is written;
// - uchar, ushort and uchar3 matrices, with a sum that accumulates in uint, and a maximum and a minimum that
//   accumulate in the type of the elements: reductions with accumulators of different types, all written as doubles;
// - one reduction (a Tuple of one Tuple), also copied and passed by value, and 1D sources;
// - a read with operations fused after it, and a write with operations fused before it;
// - a ReadBack: a Crop, against the reduction of the same region, and a Crop and a Resize against the reduction of
//   the image that TransformDPP writes with the same read;
// - a batch of crops of very different sizes, written by a TensorWrite and by a batch of writes: every plane is
//   reduced on its own, and gives what its single execution gives, bit for bit (the sums of these cases are exact:
//   on a GPU the blocks of a plane depend on the other planes of the read, so a sum that is not exact can differ in
//   its last bits between a batch and a single execution); a 3D read; a batch with planes that are not used; a
//   constant read;
// - an accumulator that is not a scalar nor a vector: the mean of every plane of a batch, as a Tuple<sum, count>;
// - a write without room for the reductions, without rows or with other planes than the read, and a read without
//   elements, throw std::invalid_argument and write nothing; a plane without elements among others is not written;
// - many executions enqueued on one stream and on two streams before any synchronization;
// - the example of the README: the mean, the maximum and the minimum of every channel of an image, on every backend
//   and, with the lines of the README, as a user compiles them (outside namespace fk);
// - sums of floats that are not exact: ParArch::CPU and CPU_MULTITHREAD give the same bytes, and every backend the
//   sum of a double, closely;
// - every reduction applies its own FirstIOp to every element, and the read is executed once per element;
// - an IOp that throws on the host backends; more accumulated values than fit in the shared memory of a block of the
//   policy; a plane of more than 2^32 elements; an accumulator with a default constructor that is not trivial;
// - the output of a plane in the memory that the plane reads; a complete BorderReader as the read;
// - the compile time checks of the IOps (reduce_detail::iopsAreValid), with accumulators that can not be stored and
//   reads that are not complete;
// - the workspace of the GPU versions, which the caller owns: its size, the executions that need none, the ones that
//   throw without it or with a small one, and one workspace used by executions of different reads and reductions;
//   the workspace of a batch, of exactly workspaceBytes() bytes, after which nothing is written; its alignment;
// - a read of 65535 planes, the most that the GPU versions reduce, and one of 65536, which they reject;
// - the host backends: every plane is read before any is written, and the elements are accumulated in the order of
//   the data (an accumulation that is not commutative); where the elements past 2^32 of a plane are;
// - GPU_NVIDIA: the grid that every policy launches and the threads of its blocks (static_asserts), the blocks that
//   really read the elements on the device of the run, with a workspace and without one, and the ones that read
//   every plane of a batch, the executions of the default policy without a workspace (one block, one cluster or an
//   exception), executions captured in a graph, or made while another thread captures one, blocks of more threads
//   than the kernel can be launched with, outside a capture and inside one, and the last error of CUDA after a call
//   that fails.
// Every case runs on ParArch::CPU, on CPU_MULTITHREAD when it is enabled, and on the GPU of the build. The cases of
// splitCases (every matrix size, a batch, the Tuple accumulator, the enqueued executions and the cases about how the
// data is split) also run on the first two with chunks of 37 elements, and on GPU_NVIDIA with policies that make
// every level of the kernel work with few elements: without clusters, with clusters of 3 blocks for every size and
// with blocks of 32 threads. Where the device has no clusters those policies launch plain blocks:
// levelsOfTheKernel() says which blocks read the elements of an execution on the device of the run, and the
// static_asserts above it what every policy launches.

#include <tests/main.h>

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/core/execution_model/reduce_dpp.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/cast.h>
#include <fused_kernel/algorithms/basic_ops/logical.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/basic_ops/set.h>
#include <fused_kernel/algorithms/image_processing/border_reader.h>
#include <fused_kernel/algorithms/image_processing/crop.h>
#include <fused_kernel/algorithms/image_processing/resize.h>
#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/data/ptr_utils.h>
#include <fused_kernel/core/execution_model/stream.h>
#ifdef FK_CPU_MULTITHREAD_ENABLED
#include <fused_kernel/core/execution_model/cpu_multithread.h>
#endif

#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <future>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// The example of README.md with the lines of the README, as a user compiles them: outside namespace fk (at the end of
// this file)
bool reduceDPPReadmeExample();

namespace fk {
namespace {

template <ParArch PA>
constexpr bool IS_GPU = PA == ParArch::GPU_NVIDIA || PA == ParArch::GPU_AMD;

#if defined(__NVCC__)
// The ParArch of the GPU of the build
constexpr ParArch GPU_PA = ParArch::GPU_NVIDIA;
#elif defined(FK_HIP_BACKEND)
constexpr ParArch GPU_PA = ParArch::GPU_AMD;
#endif

// A DPP (its ParArch and its policy), a stream for it, and what the cases do with both
template <typename DPP>
struct Backend {
    static constexpr ParArch PA = DPP::PAR_ARCH;
    static constexpr MemType MEMORY = IS_GPU<PA> ? MemType::DeviceAndPinned : MemType::Host;
    const char* name;
    Stream_<PA> stream;

    explicit Backend(const char* const name_) : name(name_) {}

    template <typename P>
    void upload(P& ptr) {
#if defined(__NVCC__) || defined(FK_HIP_BACKEND)
        if constexpr (IS_GPU<PA>) {
            ptr.upload(stream);
        }
#endif
        (void)ptr;
    }
    template <typename P>
    void download(P& ptr) {
#if defined(__NVCC__) || defined(FK_HIP_BACKEND)
        if constexpr (IS_GPU<PA>) {
            ptr.download(stream);
        }
#endif
        (void)ptr;
    }
    void sync() {
        stream.sync();
    }
    // The executions of the stream share one workspace, as the caller of the GPU versions has to provide it: a larger
    // one is allocated, and zeroed on the stream, when an execution needs more. The ones before stay alive: the
    // executions that were enqueued with them may not have finished.
    std::vector<Ptr1D<uchar>> workspaces;
    Ptr1D<uchar> workspace;
    size_t workspaceBytes = 0;

    void needWorkspace(const size_t bytes) {
        if (bytes > workspaceBytes) {
            workspace = Ptr1D<uchar>(static_cast<uint>(bytes), 0, IS_GPU<PA> ? MemType::Device : MemType::Host);
            setTo(uchar{0}, workspace, stream);
            workspaces.push_back(workspace);
            workspaceBytes = bytes;
        }
    }
    // With the workspace that the read and the reductions need (none on the CPU, and for planes of one block)
    template <typename ReadIOp, typename Reductions, typename WriteIOp>
    void reduce(const ReadIOp& read, const Reductions& reductions, const WriteIOp& write) {
        needWorkspace(DPP::workspaceBytes(read, reductions));
        executeOperations<DPP>(stream, DPP::build_details(workspace), read, reductions, write);
    }
    // Without details: no workspace
    template <typename ReadIOp, typename Reductions, typename WriteIOp>
    void reduceWithoutWorkspace(const ReadIOp& read, const Reductions& reductions, const WriteIOp& write) {
        executeOperations<DPP>(stream, read, reductions, write);
    }
};

// Policies that make the levels of the GPU version, and the chunks of the CPU ones, work with few elements
struct SmallChunks : ReduceDPPPolicy {
    static constexpr uint CHUNK_ELEMENTS = 37;
};
// The chunks of the case that looks at one of them alone (lastChunkOfMoreThan4GElements)
struct ChunksOf16384 : ReduceDPPPolicy {
    static constexpr uint CHUNK_ELEMENTS = 16384;
};
// (Blocks of BLOCK_ELEMENTS elements, up to BLOCKS = MAX_BLOCKS of them, which then grow.)
struct NoClusters : ReduceDPPPolicy {
    static constexpr ReduceDPPClusters CLUSTERS = ReduceDPPClusters::NEVER;
    static constexpr uint SINGLE_BLOCK_ELEMENTS = 1024;
    static constexpr uint BLOCK_ELEMENTS = 1024;
    static constexpr uint BLOCKS = 300;
    static constexpr uint LARGE_BLOCK_ELEMENTS = 1024;
    static constexpr uint MAX_BLOCKS = 300;
};
struct ClustersOf3 : ReduceDPPPolicy {
    static constexpr ReduceDPPClusters CLUSTERS = ReduceDPPClusters::ALWAYS;
    static constexpr uint CLUSTER_BLOCKS = 3;
    static constexpr uint SINGLE_BLOCK_ELEMENTS = 700;
    static constexpr uint BLOCK_ELEMENTS = 700;
    static constexpr uint BLOCKS = 200;
    static constexpr uint LARGE_BLOCK_ELEMENTS = 700;
    static constexpr uint MAX_BLOCKS = 200;
};
struct BlocksOf32 : ReduceDPPPolicy {
    static constexpr uint BLOCK_THREADS = 32;
    static constexpr uint SINGLE_BLOCK_ELEMENTS = 50;
    static constexpr uint BLOCK_ELEMENTS = 50;
    static constexpr uint BLOCKS = 5000;
    static constexpr uint LARGE_BLOCK_ELEMENTS = 50;
    static constexpr uint MAX_BLOCKS = 5000;
};

// Channel c of the element (x, y) of the data `salt`. The floating point values are multiples of 1 / 8 from -7 / 8 to
// 7 / 8, so the sum of up to 2^24 / 7 (2396745) of them is exact in a float in whatever order they are added (the
// largest float matrix of the test has 1921 x 1081 = 2076601 elements); the unsigned ones go from 0 to 250, and the
// signed ones from -7 to 7.
template <typename Base>
Base channelValue(const int x, const int y, const int salt, const int c) {
    if constexpr (std::is_floating_point_v<Base>) {
        return static_cast<Base>((x * 5 + y * 7 + salt * 3 + c * 11) % 15 - 7) * static_cast<Base>(0.125);
    } else if constexpr (std::is_unsigned_v<Base>) {
        return static_cast<Base>((x * 31 + y * 17 + salt * 7 + c * 13) % 251);
    } else {
        return static_cast<Base>((x * 5 + y * 7 + salt * 3 + c * 11) % 15 - 7);
    }
}

template <typename T>
T element(const int x, const int y, const int salt) {
    using Base = VBase<T>;
    if constexpr (cn<T> == 1) {
        return static_cast<T>(channelValue<Base>(x, y, salt, 0));
    } else if constexpr (cn<T> == 2) {
        return make_<T>(channelValue<Base>(x, y, salt, 0), channelValue<Base>(x, y, salt, 1));
    } else if constexpr (cn<T> == 3) {
        return make_<T>(channelValue<Base>(x, y, salt, 0), channelValue<Base>(x, y, salt, 1),
                        channelValue<Base>(x, y, salt, 2));
    } else {
        return make_<T>(channelValue<Base>(x, y, salt, 0), channelValue<Base>(x, y, salt, 1),
                        channelValue<Base>(x, y, salt, 2), channelValue<Base>(x, y, salt, 3));
    }
}

// Channel c of a scalar or a vector, as a double
template <typename V>
double channel(const V& value, const int c) {
    if constexpr (cn<V> == 1) {
        (void)c;
        return static_cast<double>(value);
    } else if constexpr (cn<V> == 2) {
        return static_cast<double>(c == 0 ? value.x : value.y);
    } else if constexpr (cn<V> == 3) {
        return static_cast<double>(c == 0 ? value.x : (c == 1 ? value.y : value.z));
    } else {
        return static_cast<double>(c == 0 ? value.x : (c == 1 ? value.y : (c == 2 ? value.z : value.w)));
    }
}

// What the reductions of the cases have to give for some elements, per channel
struct Expected {
    double sum[4]{0., 0., 0., 0.};
    double maximum[4]{0., 0., 0., 0.};
    double minimum[4]{0., 0., 0., 0.};
    ulonglong count{0};

    template <typename T>
    void add(const T& value) {
        for (int c = 0; c < cn<T>; ++c) {
            const double v = channel(value, c);
            sum[c] += v;
            maximum[c] = count == 0 || v > maximum[c] ? v : maximum[c];
            minimum[c] = count == 0 || v < minimum[c] ? v : minimum[c];
        }
        ++count;
    }
};

// The elements of a width x height region of the data `salt`, whose first element is (x0, y0)
template <typename T>
Expected expectedOf(const int width, const int height, const int salt, const int x0 = 0, const int y0 = 0) {
    Expected expected;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            expected.add(element<T>(x0 + x, y0 + y, salt));
        }
    }
    return expected;
}

template <typename V>
bool sameChannels(const V& got, const double* const expected, const double scale = 1.) {
    bool same = true;
    for (int c = 0; c < cn<V>; ++c) {
        same = same && channel(got, c) == expected[c] * scale;
    }
    return same;
}

// Channel c of a scalar or a vector, in its own type
template <typename V>
VBase<V> baseChannel(const V& value, const int c) {
    if constexpr (cn<V> == 1) {
        (void)c;
        return static_cast<VBase<V>>(value);
    } else if constexpr (cn<V> == 2) {
        return c == 0 ? value.x : value.y;
    } else if constexpr (cn<V> == 3) {
        return c == 0 ? value.x : (c == 1 ? value.y : value.z);
    } else {
        return c == 0 ? value.x : (c == 1 ? value.y : (c == 2 ? value.z : value.w));
    }
}

// Whether every channel of a and b has the same bytes. Not the bytes of the whole values: some vector types have
// padding (fk::double3 is 32 bytes on the CPU).
template <typename V>
bool sameBits(const V& a, const V& b) {
    bool same = true;
    for (int c = 0; c < cn<V>; ++c) {
        const VBase<V> channelA = baseChannel(a, c);
        const VBase<V> channelB = baseChannel(b, c);
        same = same && std::memcmp(&channelA, &channelB, sizeof(VBase<V>)) == 0;
    }
    return same;
}

template <typename T, typename DPP>
Ptr2D<T> makeImage(const int width, const int height, const int salt) {
    Ptr2D<T> image(static_cast<uint>(width), static_cast<uint>(height), 0, Backend<DPP>::MEMORY);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            image.at(Point{x, y, 0}) = element<T>(x, y, salt);
        }
    }
    return image;
}

constexpr double SENTINEL = -999.;

template <typename V>
V sentinel() {
    return make_set<V>(static_cast<VBase<V>>(SENTINEL));
}

template <typename V, typename DPP>
Ptr1D<V> makeOutput(const int elements) {
    Ptr1D<V> output(static_cast<uint>(elements), 0, Backend<DPP>::MEMORY);
    for (int i = 0; i < elements; ++i) {
        output.at(Point{i, 0, 0}) = sentinel<V>();
    }
    return output;
}

template <typename V, typename DPP>
Tensor<V> makeTensor(const int width, const int planes) {
    Tensor<V> tensor(static_cast<uint>(width), 1, static_cast<uint>(planes), 1, Backend<DPP>::MEMORY);
    for (int z = 0; z < planes; ++z) {
        for (int x = 0; x < width; ++x) {
            tensor.at(Point{x, 0, z}) = sentinel<V>();
        }
    }
    return tensor;
}

bool report(const bool ok, const char* const what, const char* const backend) {
    if (!ok) {
        std::printf("ReduceDPP %s %s FAIL\n", what, backend);
    }
    return ok;
}

// The reductions of most cases: the sum, the maximum and twice the minimum of the elements, accumulated in their own
// type T
template <typename T>
auto sumMaxMin() {
    const auto same = Cast<T, T>::build();
    return make_tuple(make_tuple(same, Add<T, T, T, UnaryType>::build(), same),
                      make_tuple(same, Max<T, T, T, UnaryType>::build(), same),
                      make_tuple(same, Min<T, T, T, UnaryType>::build(), Mul<T>::build(make_set<T>(2))));
}

template <typename V>
bool isSumMaxMin(const V& sum, const V& maximum, const V& twiceMinimum, const Expected& expected) {
    return sameChannels(sum, expected.sum) && sameChannels(maximum, expected.maximum) &&
           sameChannels(twiceMinimum, expected.minimum, 2.);
}

// 1. A width x height matrix of T, a floating point scalar or vector: three reductions that accumulate in T give three
//    values of T, and the element after them is not written
template <typename T, typename DPP>
bool floatMatrix(Backend<DPP>& backend, const int width, const int height) {
    auto input = makeImage<T, DPP>(width, height, 1);
    auto output = makeOutput<T, DPP>(4);
    backend.upload(input);
    backend.upload(output);
    backend.reduce(PerThreadRead<ND::_2D, T>::build(input), sumMaxMin<T>(), PerThreadWrite<ND::_1D, T>::build(output));
    backend.download(output);
    backend.sync();
    const bool ok = isSumMaxMin(output.at(Point{0, 0, 0}), output.at(Point{1, 0, 0}), output.at(Point{2, 0, 0}),
                                expectedOf<T>(width, height, 1)) &&
                    sameBits(output.at(Point{3, 0, 0}), sentinel<T>());
    if (!ok) {
        std::printf("ReduceDPP matrix of %d x %d elements of %d bytes %s FAIL\n", width, height,
                    static_cast<int>(sizeof(T)), backend.name);
    }
    return ok;
}

// 2. A matrix of T, an unsigned integer scalar or vector: the sum accumulates in uint, the maximum and the minimum in
//    T, and the three are written as doubles
template <typename T, typename DPP>
bool integerMatrix(Backend<DPP>& backend, const int width, const int height) {
    using Sum = VectorType_t<uint, cn<T>>;
    using Out = VectorType_t<double, cn<T>>;
    auto input = makeImage<T, DPP>(width, height, 2);
    auto output = makeOutput<Out, DPP>(4);
    backend.upload(input);
    backend.upload(output);
    backend.reduce(PerThreadRead<ND::_2D, T>::build(input),
                   make_tuple(make_tuple(Cast<T, Sum>::build(), Add<Sum, Sum, Sum, UnaryType>::build(),
                                         Cast<Sum, Out>::build()),
                              make_tuple(Cast<T, T>::build(), Max<T, T, T, UnaryType>::build(), Cast<T, Out>::build()),
                              make_tuple(Cast<T, T>::build(), Min<T, T, T, UnaryType>::build(),
                                         Cast<T, Out>::build())),
                   PerThreadWrite<ND::_1D, Out>::build(output));
    backend.download(output);
    backend.sync();
    const Expected expected = expectedOf<T>(width, height, 2);
    const bool ok = sameChannels(output.at(Point{0, 0, 0}), expected.sum) &&
                    sameChannels(output.at(Point{1, 0, 0}), expected.maximum) &&
                    sameChannels(output.at(Point{2, 0, 0}), expected.minimum) &&
                    sameBits(output.at(Point{3, 0, 0}), sentinel<Out>());
    if (!ok) {
        std::printf("ReduceDPP integer matrix of %d x %d elements of %d bytes %s FAIL\n", width, height,
                    static_cast<int>(sizeof(T)), backend.name);
    }
    return ok;
}

// 3. One reduction, and a 1D source: the sum of `elements` ints, written as one int
template <typename DPP>
bool oneReduction1D(Backend<DPP>& backend, const int elements) {
    Ptr1D<int> input(static_cast<uint>(elements), 0, Backend<DPP>::MEMORY);
    long long sum = 0;
    for (int x = 0; x < elements; ++x) {
        input.at(Point{x, 0, 0}) = element<int>(x, 3, 0);
        sum += element<int>(x, 3, 0);
    }
    auto output = makeOutput<int, DPP>(2);
    backend.upload(input);
    backend.upload(output);
    backend.reduce(PerThreadRead<ND::_1D, int>::build(input),
                   make_tuple(make_tuple(Cast<int, int>::build(), Add<int, int, int, UnaryType>::build(),
                                         Cast<int, int>::build())),
                   PerThreadWrite<ND::_1D, int>::build(output));
    backend.download(output);
    backend.sync();
    const bool ok = output.at(Point{0, 0, 0}) == static_cast<int>(sum) &&
                    output.at(Point{1, 0, 0}) == static_cast<int>(SENTINEL);
    if (!ok) {
        std::printf("ReduceDPP one reduction of %d ints in 1D %s FAIL\n", elements, backend.name);
    }
    return ok;
}

// 33. One reduction is a Tuple of one Tuple, which can be copied as any other value: from a variable that is not
//     const, as an argument passed by value, and inside a function object that is copied and moved (fk::Tuple took
//     those copies of a Tuple of one element for the construction of its element, and did not compile)
template <typename DPP>
bool copiedSingleReduction(Backend<DPP>& backend) {
    constexpr int W = 100, H = 3;
    auto input = makeImage<float, DPP>(W, H, 60);
    auto output = makeOutput<float, DPP>(2);
    backend.upload(input);
    backend.upload(output);
    const auto same = Cast<float, float>::build();
    auto reductions = make_tuple(make_tuple(same, Add<float, float, float, UnaryType>::build(), same));
    auto copy = reductions;
    const auto byValue = [](decltype(reductions) taken) { return taken; };
    const auto returned = byValue(copy);
    const std::function<void()> execute = [returned, &backend, &input, &output]() {
        backend.reduce(PerThreadRead<ND::_2D, float>::build(input), returned,
                       PerThreadWrite<ND::_1D, float>::build(output));
    };
    const std::function<void()> moved = std::function<void()>(execute);
    moved();
    backend.download(output);
    backend.sync();
    const Expected expected = expectedOf<float>(W, H, 60);
    return report(static_cast<double>(output.at(Point{0, 0, 0})) == expected.sum[0] &&
                      output.at(Point{1, 0, 0}) == static_cast<float>(SENTINEL),
                  "one reduction, copied and passed by value", backend.name);
}

// 4. Operations fused after the read (* 2 + 1, in float) and before the write (* 0.5): the sum accumulates in double
template <typename DPP>
bool fusedReadAndWrite(Backend<DPP>& backend) {
    constexpr int W = 211, H = 53;
    auto input = makeImage<float, DPP>(W, H, 4);
    auto output = makeOutput<float, DPP>(3);
    backend.upload(input);
    backend.upload(output);
    const auto read =
        PerThreadRead<ND::_2D, float>::build(input).then(Mul<float>::build(2.f)).then(Add<float>::build(1.f));
    backend.reduce(read,
                   make_tuple(make_tuple(Cast<float, double>::build(), Add<double, double, double, UnaryType>::build(),
                                         Cast<double, float>::build()),
                              make_tuple(Cast<float, float>::build(), Max<float, float, float, UnaryType>::build(),
                                         Cast<float, float>::build())),
                   Mul<float>::build(0.5f).then(PerThreadWrite<ND::_1D, float>::build(output)));
    backend.download(output);
    backend.sync();
    const Expected expected = expectedOf<float>(W, H, 4);
    const double sum = expected.sum[0] * 2. + static_cast<double>(W * H);
    const bool ok = static_cast<double>(output.at(Point{0, 0, 0})) == sum * 0.5 &&
                    static_cast<double>(output.at(Point{1, 0, 0})) == (expected.maximum[0] * 2. + 1.) * 0.5 &&
                    output.at(Point{2, 0, 0}) == static_cast<float>(SENTINEL);
    return report(ok, "read and write with fused operations", backend.name);
}

// 5. A ReadBack. A Crop is the reduction of that region of the image. A Crop and a Resize, which interpolates, give
//    (closely: the two kernels can round the interpolation differently) what the reduction of the image that
//    TransformDPP writes with the same read gives.
template <typename DPP>
bool readBack(Backend<DPP>& backend) {
    constexpr int W = 97, H = 61;
    constexpr int CX = 13, CY = 7, CW = 71, CH = 43;
    constexpr int RW = 150, RH = 90;
    auto input = makeImage<float3, DPP>(W, H, 5);
    auto cropOutput = makeOutput<float3, DPP>(3);
    auto resizeOutput = makeOutput<float3, DPP>(3);
    auto resized = makeImage<float3, DPP>(RW, RH, 0);
    auto resizedOutput = makeOutput<float3, DPP>(3);
    backend.upload(input);
    backend.upload(cropOutput);
    backend.upload(resizeOutput);
    backend.upload(resized);
    backend.upload(resizedOutput);
    const auto crop = PerThreadRead<ND::_2D, float3>::build(input).then(Crop<>::build(Rect(CX, CY, CW, CH)));
    const auto cropAndResize = crop.then(Resize<InterpolationType::INTER_LINEAR>::build(Size(RW, RH)));
    backend.reduce(crop, sumMaxMin<float3>(), PerThreadWrite<ND::_1D, float3>::build(cropOutput));
    backend.reduce(cropAndResize, sumMaxMin<float3>(), PerThreadWrite<ND::_1D, float3>::build(resizeOutput));
    executeOperations<TransformDPP<Backend<DPP>::PA>>(backend.stream, cropAndResize,
                                                      PerThreadWrite<ND::_2D, float3>::build(resized));
    backend.reduce(PerThreadRead<ND::_2D, float3>::build(resized), sumMaxMin<float3>(),
                   PerThreadWrite<ND::_1D, float3>::build(resizedOutput));
    backend.download(cropOutput);
    backend.download(resizeOutput);
    backend.download(resizedOutput);
    backend.sync();
    bool ok = isSumMaxMin(cropOutput.at(Point{0, 0, 0}), cropOutput.at(Point{1, 0, 0}), cropOutput.at(Point{2, 0, 0}),
                          expectedOf<float3>(CW, CH, 5, CX, CY));
    for (int i = 0; i < 3; ++i) {
        for (int c = 0; c < 3; ++c) {
            const double direct = channel(resizeOutput.at(Point{i, 0, 0}), c);
            const double stored = channel(resizedOutput.at(Point{i, 0, 0}), c);
            // A sum of RW x RH interpolated values, each within 1e-6 of the stored one; a maximum or a minimum
            ok = ok && std::fabs(direct - stored) <= (i == 0 ? 1e-6 * RW * RH : 1e-3);
        }
    }
    return report(ok, "ReadBack: Crop, and Crop and Resize", backend.name);
}

// The crops of the frame of the batch cases: one element, a few, a row, a column and two sizes that need a different
// number of blocks and of chunks. The largest one is neither the first nor the last: the grid of the kernel is the
// one of the largest plane, wherever it is.
constexpr int FRAME_W = 701;
constexpr int FRAME_H = 487;
constexpr int CROPS = 6;
struct CropRect {
    int x, y, w, h;
};
constexpr std::array<CropRect, CROPS> CROP_RECTS{{{700, 486, 1, 1}, {0, 0, 701, 487}, {3, 5, 7, 4},
                                                  {0, 100, 701, 1}, {350, 0, 1, 487}, {10, 20, 300, 111}}};

// 6. A batch of crops of different sizes: every plane is reduced on its own, at its own size. Written by a
//    TensorWrite and by a batch of 1D writes; every plane gives what the host computes for it and, bit for bit, what
//    its own execution gives. The tensor has one element more than reductions in x, which is not written.
template <typename T, typename DPP>
bool batchOfCrops(Backend<DPP>& backend) {
    auto frame = makeImage<T, DPP>(FRAME_W, FRAME_H, 6);
    backend.upload(frame);
    std::array<Ptr2D<T>, CROPS> crops;
    std::array<Ptr1D<T>, CROPS> batchOutputs;
    std::array<Ptr1D<T>, CROPS> singleOutputs;
    for (size_t i = 0; i < CROPS; ++i) {
        const CropRect& r = CROP_RECTS[i];
        crops[i] = frame.crop2D(Point{r.x, r.y, 0}, PtrDims<ND::_2D>(static_cast<uint>(r.w), static_cast<uint>(r.h)));
        batchOutputs[i] = makeOutput<T, DPP>(3);
        singleOutputs[i] = makeOutput<T, DPP>(3);
        backend.upload(batchOutputs[i]);
        backend.upload(singleOutputs[i]);
    }
    auto tensor = makeTensor<T, DPP>(4, CROPS);
    backend.upload(tensor);
    const auto batchRead = PerThreadRead<ND::_2D, T>::build(crops);
    backend.reduce(batchRead, sumMaxMin<T>(), TensorWrite<T>::build(tensor.ptr()));
    backend.reduce(batchRead, sumMaxMin<T>(), PerThreadWrite<ND::_1D, T>::build(batchOutputs));
    for (size_t i = 0; i < CROPS; ++i) {
        backend.reduce(PerThreadRead<ND::_2D, T>::build(crops[i]), sumMaxMin<T>(),
                       PerThreadWrite<ND::_1D, T>::build(singleOutputs[i]));
    }
    backend.download(tensor);
    for (size_t i = 0; i < CROPS; ++i) {
        backend.download(batchOutputs[i]);
        backend.download(singleOutputs[i]);
    }
    backend.sync();
    bool ok = true;
    for (size_t i = 0; i < CROPS; ++i) {
        const CropRect& r = CROP_RECTS[i];
        const int z = static_cast<int>(i);
        const auto& single = singleOutputs[i];
        ok = ok && isSumMaxMin(single.at(Point{0, 0, 0}), single.at(Point{1, 0, 0}), single.at(Point{2, 0, 0}),
                               expectedOf<T>(r.w, r.h, 6, r.x, r.y));
        for (int n = 0; n < 3; ++n) {
            ok = ok && sameBits(tensor.at(Point{n, 0, z}), single.at(Point{n, 0, 0})) &&
                 sameBits(batchOutputs[i].at(Point{n, 0, 0}), single.at(Point{n, 0, 0}));
        }
        ok = ok && sameBits(tensor.at(Point{3, 0, z}), sentinel<T>());
    }
    return report(ok, "batch of crops of different sizes", backend.name);
}

// 7. A 3D read (the planes of a Tensor), a batch whose last planes are not used and give the default value, a read
//    that reads no memory, and a complete BorderReader
template <typename DPP>
bool otherReads(Backend<DPP>& backend) {
    constexpr int W = 67, H = 29, P = 3;
    Tensor<float> input(W, H, P, 1, Backend<DPP>::MEMORY);
    std::array<Ptr2D<float>, P> images;
    for (int z = 0; z < P; ++z) {
        images[static_cast<size_t>(z)] = makeImage<float, DPP>(W, H, 10 + z);
        backend.upload(images[static_cast<size_t>(z)]);
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                input.at(Point{x, y, z}) = element<float>(x, y, 10 + z);
            }
        }
    }
    auto tensorOutput = makeTensor<float, DPP>(3, P);
    auto defaultOutput = makeTensor<float, DPP>(3, P);
    auto setOutput = makeOutput<float, DPP>(3);
    backend.upload(input);
    backend.upload(tensorOutput);
    backend.upload(defaultOutput);
    backend.upload(setOutput);
    backend.reduce(TensorRead<float>::build(input.ptr()), sumMaxMin<float>(),
                   TensorWrite<float>::build(tensorOutput.ptr()));
    // Only the first plane is used: the other two read 0.25
    backend.reduce(PerThreadRead<ND::_2D, float>::build(1, 0.25f, images), sumMaxMin<float>(),
                   TensorWrite<float>::build(defaultOutput.ptr()));
    backend.reduce(ReadSet<float>::build(0.5f, ActiveThreads{1000, 37, 1}), sumMaxMin<float>(),
                   PerThreadWrite<ND::_1D, float>::build(setOutput));
    // A complete BorderReader as the read: its elements are the ones of the read behind it
    auto borderOutput = makeOutput<float, DPP>(3);
    backend.upload(borderOutput);
    backend.reduce(BorderReader<BorderType::REPLICATE>::build(PerThreadRead<ND::_2D, float>::build(images[1])),
                   sumMaxMin<float>(), PerThreadWrite<ND::_1D, float>::build(borderOutput));
    backend.download(tensorOutput);
    backend.download(defaultOutput);
    backend.download(setOutput);
    backend.download(borderOutput);
    backend.sync();
    bool ok = isSumMaxMin(borderOutput.at(Point{0, 0, 0}), borderOutput.at(Point{1, 0, 0}),
                          borderOutput.at(Point{2, 0, 0}), expectedOf<float>(W, H, 11));
    for (int z = 0; z < P; ++z) {
        const Expected expected = expectedOf<float>(W, H, 10 + z);
        ok = ok && isSumMaxMin(tensorOutput.at(Point{0, 0, z}), tensorOutput.at(Point{1, 0, z}),
                               tensorOutput.at(Point{2, 0, z}), expected);
        if (z == 0) {
            ok = ok && isSumMaxMin(defaultOutput.at(Point{0, 0, z}), defaultOutput.at(Point{1, 0, z}),
                                   defaultOutput.at(Point{2, 0, z}), expected);
        } else {
            ok = ok && defaultOutput.at(Point{0, 0, z}) == 0.25f * W * H && defaultOutput.at(Point{1, 0, z}) == 0.25f &&
                 defaultOutput.at(Point{2, 0, z}) == 0.5f;
        }
    }
    ok = ok && setOutput.at(Point{0, 0, 0}) == 18500.f && setOutput.at(Point{1, 0, 0}) == 0.5f &&
         setOutput.at(Point{2, 0, 0}) == 1.f;
    return report(ok, "3D read, batch with planes not used, constant read, BorderReader", backend.name);
}

// An accumulator that is a Tuple: the sum of the values and how many they are. The DPP only moves it between these
// operations, which are the ones that know what it is.
using Counted = Tuple<float, uint>;

struct CountValue {
private:
    using SelfType = CountValue;
public:
    FK_STATIC_STRUCT(CountValue, SelfType)
    using Parent = UnaryOperation<float, Counted, CountValue>;
    DECLARE_UNARY_PARENT
    FK_HOST_DEVICE_FUSE OutputType exec(const InputType input) {
        return Counted(input, 1u);
    }
};

struct AddCounted {
private:
    using SelfType = AddCounted;
public:
    FK_STATIC_STRUCT(AddCounted, SelfType)
    using Parent = UnaryOperation<Tuple<Counted, Counted>, Counted, AddCounted>;
    DECLARE_UNARY_PARENT
    FK_HOST_DEVICE_FUSE OutputType exec(const InputType input) {
        return Counted(get<0>(get<0>(input)) + get<0>(get<1>(input)), get<1>(get<0>(input)) + get<1>(get<1>(input)));
    }
};

struct MeanOfCounted {
private:
    using SelfType = MeanOfCounted;
public:
    FK_STATIC_STRUCT(MeanOfCounted, SelfType)
    using Parent = UnaryOperation<Counted, float, MeanOfCounted>;
    DECLARE_UNARY_PARENT
    FK_HOST_DEVICE_FUSE OutputType exec(const InputType input) {
        return get<0>(input) / static_cast<float>(get<1>(input));
    }
};

// 8. The mean of every crop of the batch, each with its own number of elements, and its maximum
template <typename DPP>
bool tupleAccumulator(Backend<DPP>& backend) {
    auto frame = makeImage<float, DPP>(FRAME_W, FRAME_H, 7);
    backend.upload(frame);
    std::array<Ptr2D<float>, CROPS> crops;
    for (size_t i = 0; i < CROPS; ++i) {
        const CropRect& r = CROP_RECTS[i];
        crops[i] = frame.crop2D(Point{r.x, r.y, 0}, PtrDims<ND::_2D>(static_cast<uint>(r.w), static_cast<uint>(r.h)));
    }
    auto tensor = makeTensor<float, DPP>(2, CROPS);
    backend.upload(tensor);
    backend.reduce(PerThreadRead<ND::_2D, float>::build(crops),
                   make_tuple(make_tuple(CountValue::build(), AddCounted::build(), MeanOfCounted::build()),
                              make_tuple(Cast<float, float>::build(), Max<float, float, float, UnaryType>::build(),
                                         Cast<float, float>::build())),
                   TensorWrite<float>::build(tensor.ptr()));
    backend.download(tensor);
    backend.sync();
    bool ok = true;
    for (size_t i = 0; i < CROPS; ++i) {
        const CropRect& r = CROP_RECTS[i];
        const Expected expected = expectedOf<float>(r.w, r.h, 7, r.x, r.y);
        const float mean = static_cast<float>(expected.sum[0]) / static_cast<float>(expected.count);
        ok = ok && tensor.at(Point{0, 0, static_cast<int>(i)}) == mean &&
             static_cast<double>(tensor.at(Point{1, 0, static_cast<int>(i)})) == expected.maximum[0];
    }
    return report(ok, "accumulator of type Tuple<float, uint>", backend.name);
}

template <typename DPP, typename ReadIOp, typename Reductions, typename WriteIOp>
bool throwsInvalidArgument(Backend<DPP>& backend, const ReadIOp& read, const Reductions& reductions,
                           const WriteIOp& write) {
    try {
        backend.reduce(read, reductions, write);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

// 9. What can not be executed: a write with fewer elements than reductions, without rows or with other planes than
//    the read, a read without elements, and details of the CPU versions with chunks of 0 elements, throw
//    std::invalid_argument and write nothing
template <typename DPP>
bool rejected(Backend<DPP>& backend) {
    constexpr int W = 40, H = 30;
    auto input = makeImage<float, DPP>(W, H, 8);
    std::array<Ptr2D<float>, 2> images{input, input};
    auto twoElements = makeOutput<float, DPP>(2);
    auto threeElements = makeOutput<float, DPP>(3);
    auto threePlanes = makeTensor<float, DPP>(3, 3);
    // A batch of writes whose second output has no room, and a batch read whose second plane has no elements
    std::array<Ptr1D<float>, 2> secondTooSmall{makeOutput<float, DPP>(3), makeOutput<float, DPP>(2)};
    auto emptyPlane = makeTensor<float, DPP>(4, 3);
    const std::array<float, 3> values{1.f, 2.f, 3.f};
    const std::array<ActiveThreads, 3> sizes{ActiveThreads{5, 4, 1}, ActiveThreads{0, 3, 1}, ActiveThreads{7, 2, 1}};
    backend.upload(secondTooSmall[0]);
    backend.upload(secondTooSmall[1]);
    backend.upload(emptyPlane);
    backend.upload(input);
    backend.upload(twoElements);
    backend.upload(threeElements);
    backend.upload(threePlanes);
    const auto read = PerThreadRead<ND::_2D, float>::build(input);
    bool ok = throwsInvalidArgument(backend, read, sumMaxMin<float>(),
                                    PerThreadWrite<ND::_1D, float>::build(twoElements));
    ok = throwsInvalidArgument(backend, read, sumMaxMin<float>(), TensorWrite<float>::build(threePlanes.ptr())) && ok;
    ok = throwsInvalidArgument(backend, PerThreadRead<ND::_2D, float>::build(images), sumMaxMin<float>(),
                               PerThreadWrite<ND::_1D, float>::build(threeElements)) && ok;
    ok = throwsInvalidArgument(backend, PerThreadRead<ND::_2D, float>::build(images), sumMaxMin<float>(),
                               TensorWrite<float>::build(threePlanes.ptr())) && ok;
    ok = throwsInvalidArgument(backend, PerThreadRead<ND::_2D, float>::build(images), sumMaxMin<float>(),
                               PerThreadWrite<ND::_1D, float>::build(secondTooSmall)) && ok;
    // A 2D write with room in x and no rows
    Ptr2D<float> oneRow(3, 1, 0, Backend<DPP>::MEMORY);
    RawPtr<ND::_2D, float> noRows = oneRow.ptr();
    noRows.dims.height = 0;
    ok = throwsInvalidArgument(backend, read, sumMaxMin<float>(), PerThreadWrite<ND::_2D, float>::build(noRows)) &&
         ok;
    // A write fused after a MidWrite writes the values in two places: both are checked
    MidWrite<PerThreadWrite<ND::_1D, float>> smallMidWrite;
    smallMidWrite.params = twoElements.ptr();
    ok = throwsInvalidArgument(backend, read, sumMaxMin<float>(),
                               smallMidWrite.then(Mul<float>::build(0.5f))
                                   .then(PerThreadWrite<ND::_1D, float>::build(threeElements))) && ok;
    // A plane without elements between two planes that have them: nothing is written for it
    backend.reduce(ReadSet<float>::build(values, sizes), sumMaxMin<float>(),
                   TensorWrite<float>::build(emptyPlane.ptr()));
    // No elements: nothing to reduce
    ok = throwsInvalidArgument(backend, ReadSet<float>::build(1.f, ActiveThreads{0, 5, 1}), sumMaxMin<float>(),
                               PerThreadWrite<ND::_1D, float>::build(threeElements)) && ok;
    ok = throwsInvalidArgument(backend, ReadSet<float>::build(1.f, ActiveThreads{5, 0, 1}), sumMaxMin<float>(),
                               PerThreadWrite<ND::_1D, float>::build(threeElements)) && ok;
    // Details of the CPU versions with chunks of 0 elements, as a value initialized Details{} has them (on the GPU
    // versions those are the details of an execution without a workspace): there is nothing to reduce with
    if constexpr (!IS_GPU<Backend<DPP>::PA>) {
        bool thrown = false;
        try {
            executeOperations<DPP>(backend.stream, typename DPP::Details{}, read, sumMaxMin<float>(),
                                   PerThreadWrite<ND::_1D, float>::build(threeElements));
        } catch (const std::invalid_argument&) {
            thrown = true;
        }
        ok = thrown && ok;
    }
    backend.download(twoElements);
    backend.download(threeElements);
    backend.download(threePlanes);
    backend.download(secondTooSmall[0]);
    backend.download(secondTooSmall[1]);
    backend.download(emptyPlane);
    backend.sync();
    for (int i = 0; i < 3; ++i) {
        ok = ok && secondTooSmall[0].at(Point{i, 0, 0}) == static_cast<float>(SENTINEL) &&
             (i == 2 || secondTooSmall[1].at(Point{i, 0, 0}) == static_cast<float>(SENTINEL));
    }
    for (int z = 0; z < 3; ++z) {
        const float elements = static_cast<float>(sizes[static_cast<size_t>(z)].x * sizes[static_cast<size_t>(z)].y);
        const float value = values[static_cast<size_t>(z)];
        if (z == 1) {
            for (int i = 0; i < 4; ++i) {
                ok = ok && emptyPlane.at(Point{i, 0, z}) == static_cast<float>(SENTINEL);
            }
        } else {
            ok = ok && emptyPlane.at(Point{0, 0, z}) == value * elements && emptyPlane.at(Point{1, 0, z}) == value &&
                 emptyPlane.at(Point{2, 0, z}) == 2.f * value &&
                 emptyPlane.at(Point{3, 0, z}) == static_cast<float>(SENTINEL);
        }
    }
    for (int i = 0; i < 3; ++i) {
        ok = ok && (i == 2 || twoElements.at(Point{i, 0, 0}) == static_cast<float>(SENTINEL)) &&
             threeElements.at(Point{i, 0, 0}) == static_cast<float>(SENTINEL);
        for (int z = 0; z < 3; ++z) {
            ok = ok && threePlanes.at(Point{i, 0, z}) == static_cast<float>(SENTINEL);
        }
    }
    return report(ok, "writes without room or with other planes, reads without elements", backend.name);
}

// 10. Many executions enqueued before any synchronization, on one stream and alternating between two: each one with
//     its own input size and its own output
template <typename DPP>
bool enqueuedExecutions(Backend<DPP>& backend) {
    constexpr int EXECUTIONS = 48;
    Backend<DPP> other(backend.name);
    std::vector<Ptr2D<float>> inputs;
    std::vector<Ptr1D<float>> outputs;
    std::vector<Ptr1D<float>> otherOutputs;
    const auto widthOf = [](const int i) { return 1 + i * 37; };
    const auto heightOf = [](const int i) { return 1 + (i * 11) % 29; };
    for (int i = 0; i < EXECUTIONS; ++i) {
        inputs.push_back(makeImage<float, DPP>(widthOf(i), heightOf(i), 20 + i));
        outputs.push_back(makeOutput<float, DPP>(3));
        otherOutputs.push_back(makeOutput<float, DPP>(3));
        backend.upload(inputs.back());
        backend.upload(outputs.back());
        backend.upload(otherOutputs.back());
    }
    // The inputs are uploaded before the other stream reads them
    backend.sync();
    for (int i = 0; i < EXECUTIONS; ++i) {
        const auto read = PerThreadRead<ND::_2D, float>::build(inputs[static_cast<size_t>(i)]);
        backend.reduce(read, sumMaxMin<float>(),
                       PerThreadWrite<ND::_1D, float>::build(outputs[static_cast<size_t>(i)]));
        other.reduce(read, sumMaxMin<float>(),
                     PerThreadWrite<ND::_1D, float>::build(otherOutputs[static_cast<size_t>(i)]));
    }
    for (int i = 0; i < EXECUTIONS; ++i) {
        backend.download(outputs[static_cast<size_t>(i)]);
        other.download(otherOutputs[static_cast<size_t>(i)]);
    }
    backend.sync();
    other.sync();
    bool ok = true;
    for (int i = 0; i < EXECUTIONS; ++i) {
        const Expected expected = expectedOf<float>(widthOf(i), heightOf(i), 20 + i);
        const auto& output = outputs[static_cast<size_t>(i)];
        const auto& otherOutput = otherOutputs[static_cast<size_t>(i)];
        ok = ok && isSumMaxMin(output.at(Point{0, 0, 0}), output.at(Point{1, 0, 0}), output.at(Point{2, 0, 0}),
                               expected) &&
             isSumMaxMin(otherOutput.at(Point{0, 0, 0}), otherOutput.at(Point{1, 0, 0}),
                         otherOutput.at(Point{2, 0, 0}), expected);
    }
    return report(ok, "executions enqueued on one stream and on two", backend.name);
}

// 11. The example of the README and of the fkl-using-the-library skill: the mean, the maximum and the minimum of every
//     channel of an image. The mean accumulates in float3 a sum that is not exact, so it is compared closely: how
//     closely depends on the policy, and this is the accuracy of the default one (with chunks of a million elements
//     the CPU versions would be 1.5e-3 from the mean).
template <typename DPP>
bool documentedExample(Backend<DPP>& backend) {
    constexpr int W = 640, H = 480;
    auto image = makeImage<uchar3, DPP>(W, H, 9);
    auto results = makeOutput<float3, DPP>(3);
    backend.upload(image);
    backend.upload(results);
    auto& stream = backend.stream;
    const float pixels = static_cast<float>(image.dims().width * image.dims().height);
    const auto read = PerThreadRead<ND::_2D, uchar3>::build(image);
    const auto reductions = make_tuple(
        make_tuple(Cast<uchar3, float3>::build(),
                   Add<float3, float3, float3, UnaryType>::build(),
                   Div<float3>::build(make_set<float3>(pixels))),
        make_tuple(Cast<uchar3, uchar3>::build(),
                   Max<uchar3, uchar3, uchar3, UnaryType>::build(),
                   Cast<uchar3, float3>::build()),
        make_tuple(Cast<uchar3, uchar3>::build(),
                   Min<uchar3, uchar3, uchar3, UnaryType>::build(),
                   Cast<uchar3, float3>::build()));
    const auto write = PerThreadWrite<ND::_1D, float3>::build(results);

    // Once: the workspace, device memory where the thread blocks of a plane leave their values. It starts at zero.
    const size_t bytes = DPP::workspaceBytes(read, reductions); // 0 when no workspace is needed
    Ptr1D<uchar> workspace(static_cast<uint>(bytes));
    if (bytes > 0) {
        setTo(uchar{0}, workspace, stream);
    }

    // Every time
    executeOperations<DPP>(stream, DPP::build_details(workspace), read, reductions, write);

    backend.download(results);
    backend.sync();
    const Expected expected = expectedOf<uchar3>(W, H, 9);
    bool ok = sameChannels(results.at(Point{1, 0, 0}), expected.maximum) &&
              sameChannels(results.at(Point{2, 0, 0}), expected.minimum);
    for (int c = 0; c < 3; ++c) {
        const double mean = expected.sum[c] / static_cast<double>(W * H);
        ok = ok && std::fabs(channel(results.at(Point{0, 0, 0}), c) - mean) <= 1e-4 * mean;
    }
    return report(ok, "documented example: mean, maximum and minimum of an image", backend.name);
}

// 13. Three reductions with the same accumulation and different FirstIOps of the same types (the identity, a Binary
//     * 2 and a Binary + 1), on ints: the sum, twice the sum and the sum plus the number of elements, exactly. Every
//     reduction applies its own FirstIOp, and to every element (also to the first one of a chunk, or of a thread).
template <typename DPP>
bool distinctFirstIOps(Backend<DPP>& backend, const int width, const int height) {
    auto input = makeImage<int, DPP>(width, height, 11);
    auto output = makeOutput<int, DPP>(4);
    backend.upload(input);
    backend.upload(output);
    const auto same = Cast<int, int>::build();
    const auto add = Add<int, int, int, UnaryType>::build();
    backend.reduce(PerThreadRead<ND::_2D, int>::build(input),
                   make_tuple(make_tuple(same, add, same), make_tuple(Mul<int>::build(2), add, same),
                              make_tuple(Add<int>::build(1), add, same)),
                   PerThreadWrite<ND::_1D, int>::build(output));
    backend.download(output);
    backend.sync();
    const Expected expected = expectedOf<int>(width, height, 11);
    const bool ok = static_cast<double>(output.at(Point{0, 0, 0})) == expected.sum[0] &&
                    static_cast<double>(output.at(Point{1, 0, 0})) == 2. * expected.sum[0] &&
                    static_cast<double>(output.at(Point{2, 0, 0})) ==
                        expected.sum[0] + static_cast<double>(expected.count) &&
                    output.at(Point{3, 0, 0}) == static_cast<int>(SENTINEL);
    if (!ok) {
        std::printf("ReduceDPP three FirstIOps on %d x %d ints %s FAIL\n", width, height, backend.name);
    }
    return ok;
}

// A read without memory that counts its executions: on the host in HOST_READS, and on a GPU_NVIDIA device in the
// counter of its parameters
std::atomic<ulonglong> HOST_READS{0};

struct CountedReadParams {
    uint* deviceReads;
    ActiveThreads size;
};

struct CountedRead {
private:
    using SelfType = CountedRead;
public:
    FK_STATIC_STRUCT(CountedRead, SelfType)
    using ParamsType = CountedReadParams;
    using ReadDataType = int;
    using OutputType = int;
    using InstanceType = ReadType;
    using OperationDataType = OperationData<SelfType>;
    using InstantiableType = Read<SelfType>;
    static constexpr bool IS_FUSED_OP = false;
    static constexpr bool THREAD_FUSION = false;
    FK_HOST_DEVICE_STATIC OutputType exec(const Point thread, const ParamsType& params) {
#if defined(__CUDA_ARCH__)
        atomicAdd(params.deviceReads, 1u);
#elif !defined(__HIP_DEVICE_COMPILE__)
        (void)params;
        HOST_READS.fetch_add(1, std::memory_order_relaxed);
#else
        (void)params;
#endif
        return (thread.x * 5 + thread.y * 7) % 15 - 7;
    }
    FK_HOST_DEVICE_STATIC OutputType exec(const Point thread, const OperationDataType& opData) {
        return exec(thread, opData.params);
    }
    FK_HOST_DEVICE_FUSE uint num_elems_x(const Point, const OperationDataType& opData) {
        return opData.params.size.x;
    }
    FK_HOST_DEVICE_FUSE uint num_elems_y(const Point, const OperationDataType& opData) {
        return opData.params.size.y;
    }
    FK_HOST_DEVICE_FUSE uint num_elems_z(const Point, const OperationDataType& opData) {
        return opData.params.size.z;
    }
    FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
        return opData.params.size;
    }
    FK_HOST_FUSE InstantiableType build(const ParamsType& params) {
        return {{params}};
    }
};

// 14. Reading every element once: three reductions execute the read once per element, not once per reduction
template <typename DPP>
bool readsOnce(Backend<DPP>& backend, const int width, const int height) {
    Ptr1D<uint> deviceReads(1, 0, Backend<DPP>::MEMORY);
    deviceReads.at(Point{0, 0, 0}) = 0u;
    auto output = makeOutput<int, DPP>(4);
    backend.upload(deviceReads);
    backend.upload(output);
    backend.sync();
    HOST_READS.store(0);
    const auto same = Cast<int, int>::build();
    backend.reduce(CountedRead::build(CountedReadParams{deviceReads.ptr().data,
                                                        ActiveThreads{static_cast<uint>(width),
                                                                      static_cast<uint>(height), 1}}),
                   make_tuple(make_tuple(same, Add<int, int, int, UnaryType>::build(), same),
                              make_tuple(same, Max<int, int, int, UnaryType>::build(), same),
                              make_tuple(same, Min<int, int, int, UnaryType>::build(), same)),
                   PerThreadWrite<ND::_1D, int>::build(output));
    backend.download(deviceReads);
    backend.download(output);
    backend.sync();
    const ulonglong elements = static_cast<ulonglong>(width) * static_cast<ulonglong>(height);
    ulonglong reads = HOST_READS.load();
    if constexpr (IS_GPU<Backend<DPP>::PA>) {
#if defined(__NVCC__)
        reads = deviceReads.at(Point{0, 0, 0});
#else
        reads = elements; // not counted on this device
#endif
    }
    long long sum = 0;
    int maximum = -8;
    int minimum = 8;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int value = (x * 5 + y * 7) % 15 - 7;
            sum += value;
            maximum = value > maximum ? value : maximum;
            minimum = value < minimum ? value : minimum;
        }
    }
    const bool ok = reads == elements && output.at(Point{0, 0, 0}) == static_cast<int>(sum) &&
                    output.at(Point{1, 0, 0}) == maximum && output.at(Point{2, 0, 0}) == minimum &&
                    output.at(Point{3, 0, 0}) == static_cast<int>(SENTINEL);
    if (!ok) {
        std::printf("ReduceDPP reading every element once, %d x %d, %s FAIL: %llu reads of %llu elements\n", width,
                    height, backend.name, reads, elements);
    }
    return ok;
}

// Throws on the host for one value, as ThrowOn13 of tests/image_processing/test_morphology_dpp.h
struct ThrowOn5 {
private:
    using SelfType = ThrowOn5;
public:
    FK_STATIC_STRUCT(ThrowOn5, SelfType)
    using Parent = UnaryOperation<float, float, ThrowOn5>;
    DECLARE_UNARY_PARENT
    FK_HOST_DEVICE_FUSE OutputType exec(const InputType input) {
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
        if (input == 5.f) {
            throw std::runtime_error("ThrowOn5");
        }
#endif
        return input;
    }
};

// 15. An IOp that throws (host backends): in FirstIOp, in a chunk, or in FinalIOp, when the plane is written. The
//     error reaches the caller once (from executeOperations on ParArch::CPU, from sync() on CPU_MULTITHREAD), nothing
//     is written, and the stream works afterwards. (One plane and one reduction: a throw while the planes are read
//     never writes anything, and one of the final phase leaves written the planes before the one that throws.)
template <typename DPP>
bool iopThrows(Backend<DPP>& backend, const bool inFinalIOp) {
    constexpr int W = 300, H = 200;
    Ptr2D<float> input(W, H, 0, Backend<DPP>::MEMORY);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            input.at(Point{x, y, 0}) = inFinalIOp || (x == 123 && y == 150) ? 5.f : 1.f;
        }
    }
    auto output = makeOutput<float, DPP>(2);
    const auto same = Cast<float, float>::build();
    const auto maximum = Max<float, float, float, UnaryType>::build();
    const auto read = PerThreadRead<ND::_2D, float>::build(input);
    const auto write = PerThreadWrite<ND::_1D, float>::build(output);
    bool threw = false;
    try {
        if (inFinalIOp) {
            backend.reduce(read, make_tuple(make_tuple(same, maximum, ThrowOn5::build())), write);
        } else {
            backend.reduce(read, make_tuple(make_tuple(ThrowOn5::build(), maximum, same)), write);
        }
        backend.sync();
    } catch (const std::runtime_error&) {
        threw = true;
    }
    bool threwAgain = false;
    try {
        backend.sync();
    } catch (...) {
        threwAgain = true;
    }
    const bool untouched = output.at(Point{0, 0, 0}) == static_cast<float>(SENTINEL) &&
                           output.at(Point{1, 0, 0}) == static_cast<float>(SENTINEL);
    backend.reduce(ReadSet<float>::build(2.f, ActiveThreads{50, 3, 1}), make_tuple(make_tuple(same, maximum, same)),
                   write);
    backend.sync();
    const bool ok = threw && !threwAgain && untouched && output.at(Point{0, 0, 0}) == 2.f;
    if (!ok) {
        std::printf("ReduceDPP IOp that throws in %s %s FAIL: threw %d, again %d, untouched %d\n",
                    inFinalIOp ? "FinalIOp" : "FirstIOp", backend.name, threw, threwAgain, untouched);
    }
    return ok;
}

// 16. Seventeen reductions that accumulate in float3: 204 bytes of accumulated values per thread, which do not fit in
//     the shared memory of a block of 256 threads, so the GPU versions use blocks of fewer threads
//     (reduce_detail::blockThreads). In a build without optimization they are eleven (132 bytes, which do not fit
//     either): such a build keeps every accumulated value in the stack of the thread, and a device reserves the stack
//     of a kernel for every thread that it can run. Measured with device debug information (-O0 -g -G), in the image
//     for sm_89: 78448 bytes per thread with seventeen reductions, which are 15 GB on an RTX 4090, and 26752 with
//     eleven, 5 GB (an optimized build needs no stack for either).
#if defined(__OPTIMIZE__) || defined(_MSC_VER)
constexpr int MANY_ACCUMULATORS = 17;
#else
constexpr int MANY_ACCUMULATORS = 11;
#endif

template <typename DPP>
bool manyAccumulators(Backend<DPP>& backend, const int width, const int height) {
    auto input = makeImage<float3, DPP>(width, height, 12);
    auto output = makeOutput<float3, DPP>(MANY_ACCUMULATORS + 1);
    backend.upload(input);
    backend.upload(output);
    const auto same = Cast<float3, float3>::build();
    const auto sum = make_tuple(same, Add<float3, float3, float3, UnaryType>::build(), same);
    const auto maximum = make_tuple(same, Max<float3, float3, float3, UnaryType>::build(), same);
    const auto read = PerThreadRead<ND::_2D, float3>::build(input);
    const auto write = PerThreadWrite<ND::_1D, float3>::build(output);
#if defined(__OPTIMIZE__) || defined(_MSC_VER)
    const auto reductions = make_tuple(sum, maximum, sum, maximum, sum, maximum, sum, maximum, sum, maximum, sum,
                                       maximum, sum, maximum, sum, maximum, sum);
#else
    const auto reductions = make_tuple(sum, maximum, sum, maximum, sum, maximum, sum, maximum, sum, maximum, sum);
#endif
    static_assert(sizeof(reduce_detail::PartialOf<std::decay_t<decltype(reductions)>>) ==
                      static_cast<size_t>(MANY_ACCUMULATORS) * 12 &&
                  MANY_ACCUMULATORS * 12 * 256 > 32768,
                  "the accumulated values of this case do not fit in the shared memory of a block of 256 threads");
    backend.reduce(read, reductions, write);
    backend.download(output);
    backend.sync();
    const Expected expected = expectedOf<float3>(width, height, 12);
    bool ok = sameBits(output.at(Point{MANY_ACCUMULATORS, 0, 0}), sentinel<float3>());
    for (int i = 0; i < MANY_ACCUMULATORS; ++i) {
        ok = ok && sameChannels(output.at(Point{i, 0, 0}), i % 2 == 0 ? expected.sum : expected.maximum);
    }
    if (!ok) {
        std::printf("ReduceDPP %d float3 accumulators, %d x %d, %s FAIL\n", MANY_ACCUMULATORS, width, height,
                    backend.name);
    }
    return ok;
}

// 17. A plane of more than 2^32 elements (a constant read, which needs no memory): how many they are. The count of
//     the elements of a plane is a constant expression, checked in every build; the execution visits 2^32 elements,
//     which an optimized build folds (milliseconds) and a build without optimization does not (minutes on
//     ParArch::CPU), so allCases only runs it when NDEBUG is defined.
static_assert(reduce_detail::planeElements(0, ReadSet<uchar>::build(uchar{1}, ActiveThreads{65536, 65537, 1})) ==
                  4295032832ull,
              "ReduceDPP counts the elements of a plane in 64 bits");

template <typename DPP>
bool moreThan4GElements(Backend<DPP>& backend) {
    constexpr uint W = 65536, H = 65537;
    Ptr1D<ulonglong> output(2, 0, Backend<DPP>::MEMORY);
    output.at(Point{0, 0, 0}) = 7ull;
    output.at(Point{1, 0, 0}) = 7ull;
    backend.upload(output);
    backend.reduce(ReadSet<ulonglong>::build(1ull, ActiveThreads{W, H, 1}),
                   make_tuple(make_tuple(Cast<ulonglong, ulonglong>::build(),
                                         Add<ulonglong, ulonglong, ulonglong, UnaryType>::build(),
                                         Cast<ulonglong, ulonglong>::build())),
                   PerThreadWrite<ND::_1D, ulonglong>::build(output));
    backend.download(output);
    backend.sync();
    const bool ok = output.at(Point{0, 0, 0}) == static_cast<ulonglong>(W) * H && output.at(Point{1, 0, 0}) == 7ull;
    return report(ok, "plane of 65536 x 65537 elements", backend.name);
}

// 20. The output of a plane in the memory that the plane reads: the three values of every plane of a Tensor are written
//     in its first three elements, when all the elements of the plane have been read
template <typename DPP>
bool inPlace(Backend<DPP>& backend, const int width) {
    constexpr int P = 3;
    Tensor<int> tensor(static_cast<uint>(width), 1, P, 1, Backend<DPP>::MEMORY);
    for (int z = 0; z < P; ++z) {
        for (int x = 0; x < width; ++x) {
            tensor.at(Point{x, 0, z}) = element<int>(x, z, 13);
        }
    }
    backend.upload(tensor);
    backend.reduce(TensorRead<int>::build(tensor.ptr()), sumMaxMin<int>(), TensorWrite<int>::build(tensor.ptr()));
    backend.download(tensor);
    backend.sync();
    bool ok = true;
    for (int z = 0; z < P; ++z) {
        ok = ok && isSumMaxMin(tensor.at(Point{0, 0, z}), tensor.at(Point{1, 0, z}), tensor.at(Point{2, 0, z}),
                               expectedOf<int>(width, 1, 13, 0, z)) &&
             (width < 4 || tensor.at(Point{width - 1, 0, z}) == element<int>(width - 1, z, 13));
    }
    if (!ok) {
        std::printf("ReduceDPP output in the memory that is read, %d elements per plane, %s FAIL\n", width,
                    backend.name);
    }
    return ok;
}

// 27. ParArch::CPU reads every plane before it writes any, as CPU_MULTITHREAD does (host backends only: the GPU ones
//     reduce all the planes at the same time). The values of plane z are written in the memory that plane z + 1
//     reads, and every plane gives the values of the data that it had.
template <typename DPP>
bool everyPlaneReadFirst(Backend<DPP>& backend, const int width) {
    constexpr int P = 2;
    Tensor<int> tensor(static_cast<uint>(width), 1, P + 1, 1, Backend<DPP>::MEMORY);
    for (int z = 0; z < P + 1; ++z) {
        for (int x = 0; x < width; ++x) {
            tensor.at(Point{x, 0, z}) = element<int>(x, z, 15);
        }
    }
    RawPtr<ND::_3D, int> read = tensor.ptr();
    read.dims.planes = P;
    RawPtr<ND::_3D, int> write = read;
    write.data = reinterpret_cast<int*>(reinterpret_cast<char*>(write.data) + write.dims.plane_pitch);
    backend.reduce(TensorRead<int>::build(read), sumMaxMin<int>(), TensorWrite<int>::build(write));
    backend.sync();
    bool ok = true;
    for (int z = 0; z < P; ++z) {
        ok = ok && isSumMaxMin(tensor.at(Point{0, 0, z + 1}), tensor.at(Point{1, 0, z + 1}),
                               tensor.at(Point{2, 0, z + 1}), expectedOf<int>(width, 1, 15, 0, z));
    }
    if (!ok) {
        std::printf("ReduceDPP every plane read before any is written, %d elements per plane, %s FAIL\n", width,
                    backend.name);
    }
    return ok;
}

// An accumulation that is associative and not commutative: the polynomial hash of a sequence of values, which the
// concatenation of two sequences combines
struct SequenceHash {
    ulonglong hash;
    ulonglong power;
};
constexpr ulonglong HASH_BASE = 1000003ull;

struct HashOfValue {
private:
    using SelfType = HashOfValue;
public:
    FK_STATIC_STRUCT(HashOfValue, SelfType)
    using Parent = UnaryOperation<int, SequenceHash, HashOfValue>;
    DECLARE_UNARY_PARENT
    FK_HOST_DEVICE_FUSE OutputType exec(const InputType input) {
        return SequenceHash{static_cast<ulonglong>(input + 100), HASH_BASE};
    }
};

struct Concatenate {
private:
    using SelfType = Concatenate;
public:
    FK_STATIC_STRUCT(Concatenate, SelfType)
    using Parent = UnaryOperation<Tuple<SequenceHash, SequenceHash>, SequenceHash, Concatenate>;
    DECLARE_UNARY_PARENT
    FK_HOST_DEVICE_FUSE OutputType exec(const InputType input) {
        const SequenceHash a = get<0>(input);
        const SequenceHash b = get<1>(input);
        return SequenceHash{a.hash * b.power + b.hash, a.power * b.power};
    }
};

struct HashValue {
private:
    using SelfType = HashValue;
public:
    FK_STATIC_STRUCT(HashValue, SelfType)
    using Parent = UnaryOperation<SequenceHash, ulonglong, HashValue>;
    DECLARE_UNARY_PARENT
    FK_HOST_DEVICE_FUSE OutputType exec(const InputType input) {
        return input.hash;
    }
};

// 28. The host backends accumulate the elements of a chunk in order, and then the chunks of the plane in order: the
//     hash of the elements of a matrix is the one of its elements in row major order
template <typename DPP>
bool accumulatedInOrder(Backend<DPP>& backend, const int width, const int height) {
    auto input = makeImage<int, DPP>(width, height, 16);
    Ptr1D<ulonglong> output(2, 0, Backend<DPP>::MEMORY);
    output.at(Point{0, 0, 0}) = 7ull;
    output.at(Point{1, 0, 0}) = 7ull;
    ulonglong expected = 0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            expected = expected * HASH_BASE + static_cast<ulonglong>(element<int>(x, y, 16) + 100);
        }
    }
    backend.reduce(PerThreadRead<ND::_2D, int>::build(input),
                   make_tuple(make_tuple(HashOfValue::build(), Concatenate::build(), HashValue::build())),
                   PerThreadWrite<ND::_1D, ulonglong>::build(output));
    backend.sync();
    const bool ok = output.at(Point{0, 0, 0}) == expected && output.at(Point{1, 0, 0}) == 7ull;
    if (!ok) {
        std::printf("ReduceDPP accumulation in the order of the data, %d x %d, %s FAIL\n", width, height,
                    backend.name);
    }
    return ok;
}

// A read without memory that gives the row of the element
struct RowRead {
private:
    using SelfType = RowRead;
public:
    FK_STATIC_STRUCT(RowRead, SelfType)
    using ParamsType = ActiveThreads;
    using ReadDataType = uint;
    using OutputType = uint;
    using InstanceType = ReadType;
    using OperationDataType = OperationData<SelfType>;
    using InstantiableType = Read<SelfType>;
    static constexpr bool IS_FUSED_OP = false;
    static constexpr bool THREAD_FUSION = false;
    FK_HOST_DEVICE_STATIC OutputType exec(const Point thread, const ParamsType&) {
        return static_cast<uint>(thread.y);
    }
    FK_HOST_DEVICE_STATIC OutputType exec(const Point thread, const OperationDataType& opData) {
        return exec(thread, opData.params);
    }
    FK_HOST_DEVICE_FUSE uint num_elems_x(const Point, const OperationDataType& opData) {
        return opData.params.x;
    }
    FK_HOST_DEVICE_FUSE uint num_elems_y(const Point, const OperationDataType& opData) {
        return opData.params.y;
    }
    FK_HOST_DEVICE_FUSE uint num_elems_z(const Point, const OperationDataType& opData) {
        return opData.params.z;
    }
    FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
        return opData.params;
    }
    FK_HOST_FUSE InstantiableType build(const ParamsType& params) {
        return {{params}};
    }
};

// 29. Where the elements past 2^32 of a plane are: the last chunk of the CPU versions, alone (16384 elements, in
//     every build), has the elements of the last row
bool lastChunkOfMoreThan4GElements() {
    using DPP = ReduceDPP<ParArch::CPU, ChunksOf16384>;
    constexpr uint W = 65536, H = 65537;
    const auto same = Cast<uint, uint>::build();
    const auto read = RowRead::build(ActiveThreads{W, H, 1});
    const auto reductions = make_tuple(make_tuple(same, Max<uint, uint, uint, UnaryType>::build(), same),
                                       make_tuple(same, Min<uint, uint, uint, UnaryType>::build(), same));
    const auto details = DPP::build_details();
    const auto last = DPP::exec_chunk(0, DPP::chunks(0, details, read) - 1, details, read, reductions);
    return report(last.value == H - 1 && last.rest.value == H - 1, "last chunk of a plane of 65536 x 65537 elements",
                  "CPU");
}

// An accumulator whose default constructor is not trivial (default member initializers): the lowest and the highest
// of some values
struct Range {
    float lowest = 0.f;
    float highest = 0.f;
};

struct RangeOfValue {
private:
    using SelfType = RangeOfValue;
public:
    FK_STATIC_STRUCT(RangeOfValue, SelfType)
    using Parent = UnaryOperation<float, Range, RangeOfValue>;
    DECLARE_UNARY_PARENT
    FK_HOST_DEVICE_FUSE OutputType exec(const InputType input) {
        return Range{input, input};
    }
};

struct JoinRanges {
private:
    using SelfType = JoinRanges;
public:
    FK_STATIC_STRUCT(JoinRanges, SelfType)
    using Parent = UnaryOperation<Tuple<Range, Range>, Range, JoinRanges>;
    DECLARE_UNARY_PARENT
    FK_HOST_DEVICE_FUSE OutputType exec(const InputType input) {
        const Range a = get<0>(input);
        const Range b = get<1>(input);
        return Range{a.lowest < b.lowest ? a.lowest : b.lowest, a.highest > b.highest ? a.highest : b.highest};
    }
};

struct WidthOfRange {
private:
    using SelfType = WidthOfRange;
public:
    FK_STATIC_STRUCT(WidthOfRange, SelfType)
    using Parent = UnaryOperation<Range, float, WidthOfRange>;
    DECLARE_UNARY_PARENT
    FK_HOST_DEVICE_FUSE OutputType exec(const InputType input) {
        return input.highest - input.lowest;
    }
};

// 21. An accumulator with a default constructor that is not trivial: the difference between the highest and the lowest
//     value of a matrix, next to its sum
template <typename DPP>
bool rangeAccumulator(Backend<DPP>& backend, const int width, const int height) {
    auto input = makeImage<float, DPP>(width, height, 14);
    auto output = makeOutput<float, DPP>(3);
    backend.upload(input);
    backend.upload(output);
    const auto same = Cast<float, float>::build();
    backend.reduce(PerThreadRead<ND::_2D, float>::build(input),
                   make_tuple(make_tuple(RangeOfValue::build(), JoinRanges::build(), WidthOfRange::build()),
                              make_tuple(same, Add<float, float, float, UnaryType>::build(), same)),
                   PerThreadWrite<ND::_1D, float>::build(output));
    backend.download(output);
    backend.sync();
    const Expected expected = expectedOf<float>(width, height, 14);
    const bool ok = static_cast<double>(output.at(Point{0, 0, 0})) == expected.maximum[0] - expected.minimum[0] &&
                    static_cast<double>(output.at(Point{1, 0, 0})) == expected.sum[0] &&
                    output.at(Point{2, 0, 0}) == static_cast<float>(SENTINEL);
    if (!ok) {
        std::printf("ReduceDPP accumulator with a default constructor, %d x %d, %s FAIL\n", width, height,
                    backend.name);
    }
    return ok;
}

// 22. The compile time checks of the IOps of an execution. reduce_detail::iopsAreValid is what the executors ask
//     before they instantiate anything else: when it is false they only instantiate reduce_detail::reportIOps, whose
//     static_asserts name the IOps that are wrong.
namespace iop_checks {
using ReadFloat = Read<PerThreadRead<ND::_2D, float>>;
using WriteFloat = Write<PerThreadWrite<ND::_1D, float>>;
using SameFloat = decltype(Cast<float, float>::build());
using AddFloat = decltype(Add<float, float, float, UnaryType>::build());
using ToDouble = decltype(Cast<float, double>::build());
using AddDouble = decltype(Add<double, double, double, UnaryType>::build());
using ToFloat = decltype(Cast<double, float>::build());
using ToRange = decltype(RangeOfValue::build());
using JoinTwoRanges = decltype(JoinRanges::build());
using FromRange = decltype(WidthOfRange::build());
using Sum = Tuple<SameFloat, AddFloat, SameFloat>;
// The reductions, between a read and a write of floats
template <typename... Reductions>
constexpr bool valid = reduce_detail::iopsAreValid<ReadFloat, Tuple<Reductions...>, WriteFloat>;

static_assert(valid<Sum>, "one reduction");
static_assert(valid<Sum, Tuple<ToDouble, AddDouble, ToFloat>, Tuple<ToRange, JoinTwoRanges, FromRange>>,
              "three reductions with accumulators of three types");
static_assert(reduce_detail::iopsAreValid<const ReadFloat&, const Tuple<Sum>&, const WriteFloat&>,
              "the IOps as the executors receive them");
static_assert(!reduce_detail::iopsAreValid<SameFloat, Tuple<Sum>, WriteFloat>, "what reads is not a read");
static_assert(!reduce_detail::iopsAreValid<ReadFloat, Tuple<Sum>, SameFloat>, "what writes is not a write");
static_assert(!reduce_detail::iopsAreValid<ReadFloat, Sum, WriteFloat>,
              "a reduction that is not inside a Tuple of reductions");
static_assert(!valid<Tuple<SameFloat, WriteFloat, SameFloat>>, "an AccumulationIOp that is a Write");
static_assert(!valid<Tuple<int, AddFloat, SameFloat>>, "a FirstIOp that is not an IOp");
static_assert(!valid<Tuple<FromRange, AddFloat, SameFloat>>, "the read gives a float and the FirstIOp takes a Range");
static_assert(!valid<Tuple<SameFloat, AddDouble, ToFloat>>,
              "the FirstIOp gives a float and the accumulator is a double: the DPP converts nothing");
static_assert(!valid<Tuple<SameFloat, SameFloat, SameFloat>>,
              "an AccumulationIOp that does not take a Tuple of two accumulated values");
static_assert(!valid<Tuple<SameFloat, AddFloat, FromRange>>,
              "the accumulator is a float and the FinalIOp takes a Range");
static_assert(!valid<Tuple<SameFloat, AddFloat, ToRange>>, "the FinalIOp gives a Range and the write takes a float");
static_assert(!valid<Sum, Tuple<SameFloat, AddDouble, ToFloat>, Sum>, "one wrong reduction between two valid ones");

// A compute IOp that only has types, for the checks, which look at nothing else: it is never executed
template <typename I, typename O>
struct TypesOnly {
private:
    using SelfType = TypesOnly<I, O>;
public:
    FK_STATIC_STRUCT(TypesOnly, SelfType)
    using Parent = UnaryOperation<I, O, TypesOnly<I, O>>;
    DECLARE_UNARY_PARENT
    FK_HOST_DEVICE_STATIC OutputType exec(const InputType& input);
};
// One reduction whose accumulator is A, between a read and a write of floats
template <typename A>
using ReductionOf = Tuple<Unary<TypesOnly<float, A>>, Unary<TypesOnly<Tuple<A, A>, A>>, Unary<TypesOnly<A, float>>>;
// Accumulators that can not be stored, each of them for one reason only
struct NotTriviallyCopyable {
    float value;
    NotTriviallyCopyable() = default;
    NotTriviallyCopyable(const NotTriviallyCopyable& other) : value(other.value) {}
    NotTriviallyCopyable& operator=(const NotTriviallyCopyable&) = default;
};
struct NotDefaultConstructible {
    float value;
    explicit NotDefaultConstructible(const float value_) : value(value_) {}
};
struct NotCopyConstructible {
    float value;
    NotCopyConstructible() = default;
    NotCopyConstructible(const NotCopyConstructible&) = delete;
    NotCopyConstructible(NotCopyConstructible&&) = default;
    NotCopyConstructible& operator=(const NotCopyConstructible&) = default;
};
struct NotCopyAssignable {
    const float value = 0.f;
};
template <typename A>
constexpr int STORABLE = (std::is_trivially_copyable_v<A> ? 1 : 0) + (std::is_default_constructible_v<A> ? 2 : 0) +
                         (std::is_copy_constructible_v<A> ? 4 : 0) + (std::is_copy_assignable_v<A> ? 8 : 0);
static_assert(STORABLE<Range> == 15 && STORABLE<NotTriviallyCopyable> == 14 &&
                  STORABLE<NotDefaultConstructible> == 13 && STORABLE<NotCopyConstructible> == 11 &&
                  STORABLE<NotCopyAssignable> == 7,
              "each of these accumulators lacks one property, and only one");
static_assert(valid<ReductionOf<Range>>, "an accumulator that can be stored");
static_assert(!valid<ReductionOf<NotTriviallyCopyable>>, "an accumulator that is not trivially copyable");
static_assert(!valid<ReductionOf<NotDefaultConstructible>>, "an accumulator that is not default constructible");
static_assert(!valid<ReductionOf<NotCopyConstructible>>, "an accumulator that is not copy constructible");
static_assert(!valid<ReductionOf<NotCopyAssignable>>, "an accumulator that is not copy assignable (a const member)");
static_assert(!valid<Sum, ReductionOf<NotCopyAssignable>>, "one accumulator that can not be stored after one that can");
// A read that is not complete: a ReadBack without the read behind it, and a batch of them, which is a ReadType IOp
static_assert(!reduce_detail::iopsAreValid<decltype(Crop<>::build(Rect(0, 0, 1, 1))), Tuple<Sum>, WriteFloat>,
              "a Crop without the read that it crops");
static_assert(!reduce_detail::iopsAreValid<decltype(Crop<>::build(std::array<Rect, 2>{})), Tuple<Sum>, WriteFloat>,
              "a batch of Crops without the read that they crop");
} // namespace iop_checks

template <typename DPP, typename ReadIOp, typename Reductions, typename WriteIOp, typename Details>
std::string invalidArgumentOf(Backend<DPP>& backend, const Details& details, const ReadIOp& read,
                              const Reductions& reductions, const WriteIOp& write) {
    try {
        executeOperations<DPP>(backend.stream, details, read, reductions, write);
    } catch (const std::invalid_argument& error) {
        return error.what();
    }
    return "";
}

// 25. The workspace of the GPU versions, which the caller owns. The CPU versions need none. On a GPU, a plane of one
//     block needs none; an execution that needs one and has none, a smaller one, one that is not device memory or one
//     that is not aligned throws std::invalid_argument, which says the bytes that are needed, and writes nothing; and
//     one workspace serves executions of different reads and reductions, one after the other.
template <typename DPP>
bool workspaceCases(Backend<DPP>& backend) {
    constexpr int W = 640, H = 480;
    auto image = makeImage<float, DPP>(W, H, 15);
    auto small = makeImage<float, DPP>(7, 5, 16);
    auto output = makeOutput<float, DPP>(4);
    backend.upload(image);
    backend.upload(small);
    backend.upload(output);
    const auto read = PerThreadRead<ND::_2D, float>::build(image);
    const auto smallRead = PerThreadRead<ND::_2D, float>::build(small);
    const auto write = PerThreadWrite<ND::_1D, float>::build(output);
    const size_t bytes = DPP::workspaceBytes(read, sumMaxMin<float>());
    bool ok = DPP::workspaceBytes(smallRead, sumMaxMin<float>()) == 0;
    if constexpr (IS_GPU<Backend<DPP>::PA>) {
#if defined(__NVCC__) || defined(FK_HIP_BACKEND)
        ok = ok && bytes > reduce_detail::WORKSPACE_COUNTERS_BYTES;
        // None
        std::string message;
        try {
            backend.reduceWithoutWorkspace(read, sumMaxMin<float>(), write);
        } catch (const std::invalid_argument& error) {
            message = error.what();
        }
        ok = ok && message.find(std::to_string(bytes) + " bytes") != std::string::npos;
        // One byte less than needed
        Ptr1D<uchar> tooSmall(static_cast<uint>(bytes - 1), 0, MemType::Device);
        ok = ok && !invalidArgumentOf(backend, DPP::build_details(tooSmall), read, sumMaxMin<float>(), write).empty();
        // Host memory
        Ptr1D<uchar> host(static_cast<uint>(bytes), 0, MemType::Host);
        bool hostRejected = false;
        try {
            (void)DPP::build_details(host);
        } catch (const std::invalid_argument&) {
            hostRejected = true;
        }
        ok = ok && hostRejected;
        // Not aligned for the accumulated values
        Ptr1D<uchar> large(static_cast<uint>(bytes + 16), 0, MemType::Device);
        RawPtr<ND::_1D, uchar> unaligned = large.ptr();
        unaligned.data += 1;
        unaligned.dims.width -= 1;
        ok = ok && !invalidArgumentOf(backend, DPP::build_details(unaligned), read, sumMaxMin<float>(), write).empty();
#endif
    } else {
        ok = ok && bytes == 0;
        // A workspace, which the GPU versions would use, is accepted and not used
        Ptr1D<uchar> unused(64, 0, MemType::Host);
        ok = ok && invalidArgumentOf(backend, DPP::build_details(unused), read, sumMaxMin<float>(), write).empty();
        backend.sync();
        ok = ok && isSumMaxMin(output.at(Point{0, 0, 0}), output.at(Point{1, 0, 0}), output.at(Point{2, 0, 0}),
                               expectedOf<float>(W, H, 15));
        output.at(Point{0, 0, 0}) = static_cast<float>(SENTINEL);
        output.at(Point{1, 0, 0}) = static_cast<float>(SENTINEL);
        output.at(Point{2, 0, 0}) = static_cast<float>(SENTINEL);
    }
    backend.download(output);
    backend.sync();
    for (int i = 0; i < 4; ++i) {
        ok = ok && output.at(Point{i, 0, 0}) == static_cast<float>(SENTINEL);
    }
    // A plane of one block, without a workspace
    backend.reduceWithoutWorkspace(smallRead, sumMaxMin<float>(), write);
    backend.download(output);
    backend.sync();
    ok = ok && isSumMaxMin(output.at(Point{0, 0, 0}), output.at(Point{1, 0, 0}), output.at(Point{2, 0, 0}),
                           expectedOf<float>(7, 5, 16));
    // One workspace, large enough for all of them, for executions of other planes, sizes and reductions
    if constexpr (IS_GPU<Backend<DPP>::PA>) {
        backend.needWorkspace(size_t{8} << 20);
    }
    const size_t allocated = backend.workspaces.size();
    for (int round = 0; round < 2; ++round) {
        ok = floatMatrix<float3, DPP>(backend, W, H) && ok;
        ok = batchOfCrops<float, DPP>(backend) && ok;
        ok = oneReduction1D(backend, 100000) && ok;
        ok = integerMatrix<uchar3, DPP>(backend, 333, 77) && ok;
        ok = tupleAccumulator(backend) && ok;
    }
    ok = ok && backend.workspaces.size() == allocated;
    return report(ok, "workspace of the caller", backend.name);
}

// 35. A grid has 65535 planes at most. The GPU versions reduce a read of 65535 planes, and for one of 65536 they throw
//     std::invalid_argument, which says so, before anything is launched; the host versions reduce both.
template <typename DPP>
bool planesOfAGrid(Backend<DPP>& backend) {
    constexpr uint ELEMENTS = 6;
    bool ok = true;
    for (const uint planes : {65535u, 65536u}) {
        auto output = makeTensor<float, DPP>(4, static_cast<int>(planes));
        backend.upload(output);
        const bool mustThrow = IS_GPU<Backend<DPP>::PA> && planes > 65535u;
        std::string message;
        bool threw = false;
        bool threwAnother = false;
        try {
            backend.reduce(ReadSet<float>::build(0.5f, ActiveThreads{ELEMENTS, 1, planes}), sumMaxMin<float>(),
                           TensorWrite<float>::build(output.ptr()));
        } catch (const std::invalid_argument& error) {
            threw = true;
            message = error.what();
        } catch (const std::exception&) {
            threwAnother = true;
        }
        backend.download(output);
        backend.sync();
        bool right = !threwAnother && threw == mustThrow && (!mustThrow || message.find("65535") != std::string::npos);
        for (int z = 0; z < static_cast<int>(planes); ++z) {
            if (mustThrow) {
                for (int i = 0; i < 4; ++i) {
                    right = right && output.at(Point{i, 0, z}) == static_cast<float>(SENTINEL);
                }
            } else {
                right = right && output.at(Point{0, 0, z}) == 0.5f * static_cast<float>(ELEMENTS) &&
                        output.at(Point{1, 0, z}) == 0.5f && output.at(Point{2, 0, z}) == 1.f &&
                        output.at(Point{3, 0, z}) == static_cast<float>(SENTINEL);
            }
        }
        if (!right) {
            std::printf("ReduceDPP read of %u planes %s FAIL: %s\n", planes, backend.name,
                        mustThrow ? "must throw std::invalid_argument" : "must be reduced");
        }
        ok = ok && right;
    }
    return ok;
}

#if defined(__NVCC__) || defined(FK_HIP_BACKEND)
// What the executors launch for a size is a constant expression (reduce_detail::launchOf(elements, the blocks of the
// largest cluster that can be launched, whether the execution has a workspace)); a grid whose planes have more than
// one cluster needs a workspace. Policies with every value written here, so that the launches expected below do not
// depend on the defaults.
struct LaunchAuto : ReduceDPPPolicy {
    static constexpr uint BLOCK_THREADS = 256;
    static constexpr uint SINGLE_BLOCK_ELEMENTS = 4096;
    static constexpr uint BLOCK_ELEMENTS = 512;
    static constexpr uint BLOCKS = 128;
    static constexpr uint LARGE_BLOCK_ELEMENTS = 16384;
    static constexpr uint MAX_BLOCKS = 1024;
    static constexpr ReduceDPPClusters CLUSTERS = ReduceDPPClusters::AUTO;
    static constexpr uint CLUSTER_BLOCKS = 8;
};
struct LaunchNever : LaunchAuto {
    static constexpr ReduceDPPClusters CLUSTERS = ReduceDPPClusters::NEVER;
};
struct LaunchAlways : LaunchAuto {
    static constexpr ReduceDPPClusters CLUSTERS = ReduceDPPClusters::ALWAYS;
};

constexpr bool isLaunch(const reduce_detail::Launch& launch, const ulonglong blockElements, const uint blocks,
                        const uint clusterBlocks, const uint clusters) {
    return launch.blockElements == blockElements && launch.blocks == blocks &&
           launch.clusterBlocks == clusterBlocks && launch.clusters == clusters;
}
constexpr bool WITH_WORKSPACE = true;
constexpr bool NO_WORKSPACE = false;
// One block, with and without a workspace
static_assert(isLaunch(reduce_detail::launchOf<LaunchAuto>(4096, 1, 8, NO_WORKSPACE), 4096, 1, 1, 1), "one block");
static_assert(isLaunch(reduce_detail::launchOf<LaunchAuto>(4096, 1, 8, WITH_WORKSPACE), 4096, 1, 1, 1), "one block");
// AUTO without a workspace: one cluster, of blocks of up to SINGLE_BLOCK_ELEMENTS elements
static_assert(isLaunch(reduce_detail::launchOf<LaunchAuto>(4097, 1, 8, NO_WORKSPACE), 513, 8, 8, 1), "one cluster");
static_assert(isLaunch(reduce_detail::launchOf<LaunchAuto>(20000, 1, 8, NO_WORKSPACE), 2500, 8, 8, 1), "one cluster");
static_assert(isLaunch(reduce_detail::launchOf<LaunchAuto>(32768, 1, 8, NO_WORKSPACE), 4096, 8, 8, 1), "one cluster");
static_assert(isLaunch(reduce_detail::launchOf<LaunchAuto>(12288, 1, 3, NO_WORKSPACE), 4096, 3, 3, 1),
              "one cluster of 3 blocks, the largest of the device");
// AUTO without a workspace, and planes that one cluster can not reduce: blocks that need a workspace
static_assert(isLaunch(reduce_detail::launchOf<LaunchAuto>(32769, 1, 8, NO_WORKSPACE), 505, 65, 1, 65), "65 blocks");
static_assert(isLaunch(reduce_detail::launchOf<LaunchAuto>(20000, 1, 3, NO_WORKSPACE), 500, 40, 1, 40), "40 blocks");
static_assert(isLaunch(reduce_detail::launchOf<LaunchAuto>(20000, 1, 1, NO_WORKSPACE), 500, 40, 1, 40), "no clusters");
// AUTO with a workspace: never a cluster. Blocks of BLOCK_ELEMENTS elements up to BLOCKS blocks, which then grow up to
// LARGE_BLOCK_ELEMENTS elements; then more blocks of that size, up to MAX_BLOCKS, which grow again
static_assert(isLaunch(reduce_detail::launchOf<LaunchAuto>(4097, 1, 8, WITH_WORKSPACE), 456, 9, 1, 9), "9 blocks");
static_assert(isLaunch(reduce_detail::launchOf<LaunchAuto>(20000, 1, 8, WITH_WORKSPACE), 500, 40, 1, 40), "40 blocks");
static_assert(isLaunch(reduce_detail::launchOf<LaunchAuto>(65536, 1, 8, WITH_WORKSPACE), 512, 128, 1, 128), "BLOCKS");
static_assert(isLaunch(reduce_detail::launchOf<LaunchAuto>(1000000, 1, 8, WITH_WORKSPACE), 7813, 128, 1, 128),
              "BLOCKS blocks that grow");
static_assert(isLaunch(reduce_detail::launchOf<LaunchAuto>(2097152, 1, 8, WITH_WORKSPACE), 16384, 128, 1, 128),
              "BLOCKS blocks of LARGE_BLOCK_ELEMENTS");
static_assert(isLaunch(reduce_detail::launchOf<LaunchAuto>(5000000, 1, 8, WITH_WORKSPACE), 16340, 306, 1, 306),
              "more blocks of up to LARGE_BLOCK_ELEMENTS");
static_assert(isLaunch(reduce_detail::launchOf<LaunchAuto>(33177600, 1, 8, WITH_WORKSPACE), 32400, 1024, 1, 1024),
              "MAX_BLOCKS");
// NEVER: never a cluster. ALWAYS: one cluster also with a workspace, and the blocks of larger planes in clusters
static_assert(isLaunch(reduce_detail::launchOf<LaunchNever>(20000, 1, 8, NO_WORKSPACE), 500, 40, 1, 40), "NEVER");
static_assert(isLaunch(reduce_detail::launchOf<LaunchAlways>(20000, 1, 8, WITH_WORKSPACE), 2500, 8, 8, 1), "ALWAYS");
static_assert(isLaunch(reduce_detail::launchOf<LaunchAlways>(40000, 1, 8, WITH_WORKSPACE), 507, 80, 8, 10),
              "ALWAYS, ten clusters: 79 blocks with elements and one without");
static_assert(isLaunch(reduce_detail::launchOf<LaunchAlways>(40000, 1, 1, WITH_WORKSPACE), 507, 79, 1, 79),
              "ALWAYS on a device without clusters");
// The blocks are counted for all the planes of the read, and each plane has its part of them
static_assert(isLaunch(reduce_detail::launchOf<LaunchAuto>(65536, 30, 8, WITH_WORKSPACE), 13108, 5, 1, 5),
              "30 planes: 128 blocks, 5 per plane");
static_assert(isLaunch(reduce_detail::launchOf<LaunchAuto>(262144, 128, 8, WITH_WORKSPACE), 32768, 8, 1, 8),
              "128 planes: MAX_BLOCKS blocks, 8 per plane");
static_assert(isLaunch(reduce_detail::launchOf<LaunchAuto>(5000, 128, 8, NO_WORKSPACE), 5000, 1, 1, 1),
              "128 planes: one block per plane, which needs no workspace");
static_assert(isLaunch(reduce_detail::launchOf<LaunchAuto>(5000, 4, 8, WITH_WORKSPACE), 500, 10, 1, 10),
              "4 planes of 5000 elements: 40 blocks of 512 elements, 10 per plane");
static_assert(isLaunch(reduce_detail::launchOf<LaunchAuto>(20000, 30, 8, NO_WORKSPACE), 2500, 8, 8, 1),
              "30 planes without a workspace: one cluster per plane");
static_assert(isLaunch(reduce_detail::launchOf<LaunchAuto>(~0ull, 2, 8, WITH_WORKSPACE), 36028797018963968ull, 512,
                       1, 512),
              "two planes with more elements than a 64 bit integer counts: MAX_BLOCKS blocks, half for each plane");
static_assert(isLaunch(reduce_detail::launchOf<LaunchAuto>(1ull << 63, 2, 8, WITH_WORKSPACE), 1ull << 54, 512, 1, 512),
              "two planes of 2^63 elements, whose product is 0 in 64 bits: the same, and not one block per plane");
// The threads of a block are the ones of the policy while their accumulated values fit in the shared memory that the
// kernel uses (32 KB), and the largest power of two that fits otherwise
template <size_t BYTES>
struct BytesOf {
    char bytes[BYTES];
};
template <typename Policy, size_t BYTES>
constexpr uint THREADS_FOR = reduce_detail::blockThreads<Policy, reduce_detail::Partial<BytesOf<BYTES>>>();
static_assert(THREADS_FOR<LaunchAuto, 36> == 256, "three float3: the threads of the policy");
static_assert(THREADS_FOR<BlocksOf32, 36> == 32, "the threads of another policy");
static_assert(THREADS_FOR<LaunchAuto, 128> == 256, "256 x 128 bytes are the shared memory");
static_assert(THREADS_FOR<LaunchAuto, 129> == 128, "half the threads");
static_assert(THREADS_FOR<LaunchAuto, 204> == 128, "seventeen float3 (manyAccumulators)");
static_assert(THREADS_FOR<LaunchAuto, 1100> == 16, "halved four times");
static_assert(THREADS_FOR<LaunchAuto, 32768> == 1, "one thread");
// The policies of the split cases launch the levels that their names say, for the largest plane of the batch cases
static_assert(isLaunch(reduce_detail::launchOf<NoClusters>(701 * 487, 1, 8, WITH_WORKSPACE), 1138, 300, 1, 300),
              "NoClusters");
static_assert(isLaunch(reduce_detail::launchOf<NoClusters>(701 * 487, 6, 8, WITH_WORKSPACE), 6828, 50, 1, 50),
              "NoClusters, the 6 planes of a batch");
static_assert(isLaunch(reduce_detail::launchOf<ClustersOf3>(701 * 487, 1, 3, WITH_WORKSPACE), 1707, 201, 3, 67),
              "ClustersOf3");
static_assert(isLaunch(reduce_detail::launchOf<ClustersOf3>(701 * 487, 6, 3, WITH_WORKSPACE), 10041, 36, 3, 12),
              "ClustersOf3, the 6 planes of a batch");
static_assert(isLaunch(reduce_detail::launchOf<ClustersOf3>(2100, 1, 3, WITH_WORKSPACE), 700, 3, 3, 1),
              "ClustersOf3, one cluster");
static_assert(isLaunch(reduce_detail::launchOf<BlocksOf32>(1000, 1, 8, WITH_WORKSPACE), 50, 20, 1, 20), "BlocksOf32");
// The workspace: 65536 arrival counters, and then one value per block (the grid without clusters) of every plane
static_assert(reduce_detail::WORKSPACE_COUNTERS_BYTES == 262144, "the counters of the workspace");
static_assert(reduce_detail::workspaceBytesOf<float3>(6, 300) == 262144 + 6 * 300 * 12, "the bytes of a workspace");
#endif // defined(__NVCC__) || defined(FK_HIP_BACKEND)

#if defined(__NVCC__) || defined(FK_HIP_BACKEND)
// 19. A read whose planes need more workspace than a workspace can have (4 GB) can not be executed
struct ManyBlocks : ReduceDPPPolicy {
    static constexpr ReduceDPPClusters CLUSTERS = ReduceDPPClusters::NEVER;
    static constexpr uint SINGLE_BLOCK_ELEMENTS = 1;
    static constexpr uint BLOCK_ELEMENTS = 1;
    static constexpr uint BLOCKS = 1u << 24;
    static constexpr uint LARGE_BLOCK_ELEMENTS = 1;
    static constexpr uint MAX_BLOCKS = 1u << 24;
};
bool workspaceOfMoreThan4GB() {
    using DPP = ReduceDPP<defaultParArch, ManyBlocks>;
    Backend<DPP> backend("GPU");
    auto output = makeOutput<double2, DPP>(18);
    backend.upload(output);
    // One plane of 2^24 elements: 2^24 blocks x 288 bytes of accumulated values (eighteen double2) = 4.5 GiB
    const auto read = ReadSet<double2>::build(make_set<double2>(1.), ActiveThreads{1u << 24, 1, 1});
    const auto same = Cast<double2, double2>::build();
    const auto sum = make_tuple(same, Add<double2, double2, double2, UnaryType>::build(), same);
    const auto reductions = make_tuple(sum, sum, sum, sum, sum, sum, sum, sum, sum, sum, sum, sum, sum, sum, sum, sum,
                                       sum, sum);
    static_assert(sizeof(reduce_detail::PartialOf<std::decay_t<decltype(reductions)>>) == 288,
                  "the accumulated values of eighteen double2");
    const auto write = PerThreadWrite<ND::_1D, double2>::build(output);
    const size_t bytes = DPP::workspaceBytes(read, reductions);
    Ptr1D<uchar> workspace(1u << 20, 0, MemType::Device);
    const bool ok = bytes > (size_t{1} << 32) &&
                    !invalidArgumentOf(backend, DPP::build_details(workspace), read, reductions, write).empty();
    backend.sync();
    return report(ok, "read that needs a workspace of more than 4 GB", "GPU");
}
#endif // defined(__NVCC__) || defined(FK_HIP_BACKEND)

#if defined(__NVCC__) || defined(FK_HIP_BACKEND)
// 34. The workspace of a batch: the counters, and one value per block of every plane, with the blocks that the launch
//     rule gives each plane of that batch (they are counted for all the planes together). An execution that has
//     exactly those bytes writes nothing after them, leaves the counters at zero and gives the right values: the
//     memory after the workspace is a guard that has to stay as it was.
template <typename Policy>
bool workspaceOfABatch(const char* const policy, const uint elements, const uint planes) {
    using DPP = ReduceDPP<defaultParArch, Policy>;
    Backend<DPP> backend("GPU");
    constexpr size_t GUARD = size_t{1} << 20;
    constexpr uchar GUARD_VALUE = 0xA5;
    const auto read = ReadSet<float>::build(0.5f, ActiveThreads{elements, 1, planes});
    const auto reductions = sumMaxMin<float>();
    using PartialType = reduce_detail::PartialOf<std::decay_t<decltype(reductions)>>;
    const reduce_detail::Launch launch = reduce_detail::launchOf<Policy>(elements, planes, 1, true);
    const size_t bytes = DPP::workspaceBytes(read, reductions);
    const bool rightBytes = launch.clusters > 1 &&
                            bytes == reduce_detail::WORKSPACE_COUNTERS_BYTES +
                                         static_cast<size_t>(planes) * launch.clusters * sizeof(PartialType);
    Ptr1D<uchar> memory(static_cast<uint>(bytes + GUARD), 0, Backend<DPP>::MEMORY);
    for (size_t i = 0; i < bytes + GUARD; ++i) {
        memory.at(Point{static_cast<int>(i), 0, 0}) = i < bytes ? uchar{0} : GUARD_VALUE;
    }
    auto output = makeTensor<float, DPP>(4, static_cast<int>(planes));
    backend.upload(memory);
    backend.upload(output);
    RawPtr<ND::_1D, uchar> workspace = memory.ptr();
    workspace.dims.width = static_cast<uint>(bytes);
    bool executed = true;
    try {
        executeOperations<DPP>(backend.stream, DPP::build_details(workspace), read, reductions,
                               TensorWrite<float>::build(output.ptr()));
    } catch (const std::exception&) {
        executed = false;
    }
    backend.download(memory);
    backend.download(output);
    backend.sync();
    bool rightValues = executed;
    for (int z = 0; z < static_cast<int>(planes); ++z) {
        rightValues = rightValues && output.at(Point{0, 0, z}) == 0.5f * static_cast<float>(elements) &&
                      output.at(Point{1, 0, z}) == 0.5f && output.at(Point{2, 0, z}) == 1.f &&
                      output.at(Point{3, 0, z}) == static_cast<float>(SENTINEL);
    }
    bool countersAtZero = true;
    for (size_t i = 0; i < reduce_detail::WORKSPACE_COUNTERS_BYTES && i < bytes; ++i) {
        countersAtZero = countersAtZero && memory.at(Point{static_cast<int>(i), 0, 0}) == uchar{0};
    }
    bool guardIntact = true;
    for (size_t i = bytes; i < bytes + GUARD; ++i) {
        guardIntact = guardIntact && memory.at(Point{static_cast<int>(i), 0, 0}) == GUARD_VALUE;
    }
    const bool ok = rightBytes && rightValues && countersAtZero && guardIntact;
    if (!ok) {
        std::printf("ReduceDPP GPU, workspace of %u planes of %u elements with %s FAIL: bytes %d, values %d, counters "
                    "at zero %d, nothing written after the workspace %d\n", planes, elements, policy, rightBytes,
                    rightValues, countersAtZero, guardIntact);
    }
    return ok;
}

// A read of so many planes that the policy gives one block to each needs no workspace, although one block does not
// reduce a plane of that size alone
bool manyPlanesNeedNoWorkspace() {
    using DPP = ReduceDPP<>;
    Backend<DPP> backend("GPU");
    constexpr uint ELEMENTS = 5000, PLANES = 128;
    const auto read = ReadSet<float>::build(0.5f, ActiveThreads{ELEMENTS, 1, PLANES});
    auto output = makeTensor<float, DPP>(4, PLANES);
    backend.upload(output);
    bool ok = DPP::workspaceBytes(read, sumMaxMin<float>()) == 0 &&
              DPP::workspaceBytes(ReadSet<float>::build(0.5f, ActiveThreads{ELEMENTS, 1, 1}), sumMaxMin<float>()) > 0;
    backend.reduceWithoutWorkspace(read, sumMaxMin<float>(), TensorWrite<float>::build(output.ptr()));
    backend.download(output);
    backend.sync();
    for (int z = 0; z < static_cast<int>(PLANES); ++z) {
        ok = ok && output.at(Point{0, 0, z}) == 0.5f * static_cast<float>(ELEMENTS) &&
             output.at(Point{1, 0, z}) == 0.5f && output.at(Point{2, 0, z}) == 1.f &&
             output.at(Point{3, 0, z}) == static_cast<float>(SENTINEL);
    }
    return report(ok, "128 planes of 5000 elements, without a workspace", "GPU");
}

// The workspace has to be aligned for its counters (uint) and for the accumulated values: with accumulators of one
// byte, memory that is not aligned for the counters is rejected, and with accumulators of 8 bytes, memory that is
// aligned for the counters and not for them
static_assert(alignof(reduce_detail::Partial<uchar>) == 1 && alignof(reduce_detail::Partial<double>) == 8,
              "the alignments of the accumulated values of workspaceAlignment()");
bool workspaceAlignment() {
    using DPP = ReduceDPP<>;
    Backend<DPP> backend("GPU");
    constexpr int W = 100, H = 100;
    auto bytesImage = makeImage<uchar, DPP>(W, H, 61);
    auto doublesImage = makeImage<double, DPP>(W, H, 62);
    // (Not makeOutput<uchar>: the sentinel, -999, is not a value of an unsigned type, and its conversion to one is
    // undefined.)
    constexpr uchar NOT_WRITTEN = 77;
    Ptr1D<uchar> bytesOutput(1, 0, Backend<DPP>::MEMORY);
    bytesOutput.at(Point{0, 0, 0}) = NOT_WRITTEN;
    auto doublesOutput = makeOutput<double, DPP>(1);
    backend.upload(bytesImage);
    backend.upload(doublesImage);
    backend.upload(bytesOutput);
    backend.upload(doublesOutput);
    Ptr1D<uchar> memory(1u << 20, 0, MemType::Device);
    const auto offsetBy = [&memory](const uint offset) {
        RawPtr<ND::_1D, uchar> workspace = memory.ptr();
        workspace.data += offset;
        workspace.dims.width -= offset;
        return workspace;
    };
    const auto bytesRead = PerThreadRead<ND::_2D, uchar>::build(bytesImage);
    const auto bytesReductions = make_tuple(make_tuple(Cast<uchar, uchar>::build(),
                                                       Max<uchar, uchar, uchar, UnaryType>::build(),
                                                       Cast<uchar, uchar>::build()));
    const auto bytesWrite = PerThreadWrite<ND::_1D, uchar>::build(bytesOutput);
    const auto doublesRead = PerThreadRead<ND::_2D, double>::build(doublesImage);
    const auto doublesReductions = make_tuple(make_tuple(Cast<double, double>::build(),
                                                         Add<double, double, double, UnaryType>::build(),
                                                         Cast<double, double>::build()));
    const auto doublesWrite = PerThreadWrite<ND::_1D, double>::build(doublesOutput);
    bool ok = DPP::workspaceBytes(bytesRead, bytesReductions) > 0 &&
              DPP::workspaceBytes(doublesRead, doublesReductions) > 0;
    // Whether the execution with the workspace `offset` bytes after the start of the memory throws
    const auto rejects = [&backend, &offsetBy](const uint offset, const auto& read, const auto& reductions,
                                               const auto& write) {
        return !invalidArgumentOf(backend, DPP::build_details(offsetBy(offset)), read, reductions, write).empty();
    };
    ok = ok && rejects(1, bytesRead, bytesReductions, bytesWrite);
    ok = ok && rejects(2, bytesRead, bytesReductions, bytesWrite);
    ok = ok && rejects(4, doublesRead, doublesReductions, doublesWrite);
    backend.download(bytesOutput);
    backend.download(doublesOutput);
    backend.sync();
    ok = ok && bytesOutput.at(Point{0, 0, 0}) == NOT_WRITTEN && doublesOutput.at(Point{0, 0, 0}) == SENTINEL;
    // Aligned for both: offsets of 4 and of 8 bytes
    setTo(uchar{0}, memory, backend.stream);
    ok = ok && !rejects(4, bytesRead, bytesReductions, bytesWrite);
    ok = ok && !rejects(8, doublesRead, doublesReductions, doublesWrite);
    backend.download(bytesOutput);
    backend.download(doublesOutput);
    backend.sync();
    const Expected bytesExpected = expectedOf<uchar>(W, H, 61);
    const Expected doublesExpected = expectedOf<double>(W, H, 62);
    ok = ok && static_cast<double>(bytesOutput.at(Point{0, 0, 0})) == bytesExpected.maximum[0] &&
         doublesOutput.at(Point{0, 0, 0}) == doublesExpected.sum[0];
    return report(ok, "alignment of the workspace", "GPU");
}
#endif // defined(__NVCC__) || defined(FK_HIP_BACKEND)

#if defined(__NVCC__) || defined(FK_HIP_BACKEND)
// A read without memory that gives, on the device, the index in x of the block that executes it, and the architecture
// that the image of the kernel that executes it was compiled for (__CUDA_ARCH__: 890 for compute capability 8.9; 0 on
// GPU_AMD, whose kernels have no clusters)
struct BlockAndImageRead {
private:
    using SelfType = BlockAndImageRead;
public:
    FK_STATIC_STRUCT(BlockAndImageRead, SelfType)
    using ParamsType = ActiveThreads;
    using ReadDataType = uint2;
    using OutputType = uint2;
    using InstanceType = ReadType;
    using OperationDataType = OperationData<SelfType>;
    using InstantiableType = Read<SelfType>;
    static constexpr bool IS_FUSED_OP = false;
    static constexpr bool THREAD_FUSION = false;
    FK_HOST_DEVICE_STATIC OutputType exec(const Point, const ParamsType&) {
#if defined(__CUDA_ARCH__)
        return make_<uint2>(blockIdx.x, static_cast<uint>(__CUDA_ARCH__));
#elif defined(__HIP_DEVICE_COMPILE__)
        return make_<uint2>(blockIdx.x, 0u);
#else
        return make_<uint2>(0u, 0u);
#endif
    }
    FK_HOST_DEVICE_STATIC OutputType exec(const Point thread, const OperationDataType& opData) {
        return exec(thread, opData.params);
    }
    FK_HOST_DEVICE_FUSE uint num_elems_x(const Point, const OperationDataType& opData) {
        return opData.params.x;
    }
    FK_HOST_DEVICE_FUSE uint num_elems_y(const Point, const OperationDataType& opData) {
        return opData.params.y;
    }
    FK_HOST_DEVICE_FUSE uint num_elems_z(const Point, const OperationDataType& opData) {
        return opData.params.z;
    }
    FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const OperationDataType& opData) {
        return opData.params;
    }
    FK_HOST_FUSE InstantiableType build(const ParamsType& params) {
        return {{params}};
    }
};

// The blocks of the grid that read elements of a plane of `elements` elements (x), and the architecture of the image
// of the kernel that ran (y), in an execution with the workspace that the plane needs, or without one. {0, 0} for an
// execution that throws std::invalid_argument.
template <typename Policy>
uint2 blocksThatRead(const uint elements, const bool withWorkspace) {
    using DPP = ReduceDPP<GPU_PA, Policy>;
    Backend<DPP> backend("GPU");
    auto output = makeOutput<uint2, DPP>(1);
    backend.upload(output);
    const auto read = BlockAndImageRead::build(ActiveThreads{elements, 1, 1});
    const auto reductions = make_tuple(make_tuple(Cast<uint2, uint2>::build(),
                                                  Max<uint2, uint2, uint2, UnaryType>::build(),
                                                  Cast<uint2, uint2>::build()));
    const auto write = PerThreadWrite<ND::_1D, uint2>::build(output);
    if (withWorkspace) {
        backend.reduce(read, reductions, write);
    } else {
        try {
            backend.reduceWithoutWorkspace(read, reductions, write);
        } catch (const std::invalid_argument&) {
            backend.sync();
            return make_<uint2>(0u, 0u);
        }
    }
    backend.download(output);
    backend.sync();
    const uint2 seen = output.at(Point{0, 0, 0});
    return make_<uint2>(seen.x + 1u, seen.y);
}

#if defined(__NVCC__)
// A kernel that does nothing: what the driver says about the clusters of the device does not depend on the kernels of
// ReduceDPP
template <typename T>
__global__ void clusterProbe_Kernel(const T) {}
#endif

// What the device of the run launches: the architecture of the image of the kernels that the driver runs on it, as
// the image itself says, and the blocks of the largest cluster that can be launched there, as the driver says for a
// kernel that is not one of ReduceDPP (8 at most, and 1 when the image or the device have no clusters). (A plane of
// one element is reduced by one block, without clusters and without a workspace.)
struct Device {
    uint architecture;
    uint clusterBlocks;

    // The grid that the executor has to launch on this device for one plane of `elements` elements, in an execution
    // with a workspace or without one
    template <typename Policy>
    constexpr reduce_detail::Launch launch(const ulonglong elements, const bool withWorkspace) const {
        return reduce_detail::launchOf<Policy>(elements, 1, clusterBlocks, withWorkspace);
    }
};

Device deviceOfTheRun() {
#if defined(FK_HIP_BACKEND)
    // GPU_AMD has no clusters: every plane that needs more than one block has them synchronize through the workspace
    return {blocksThatRead<LaunchAuto>(1, NO_WORKSPACE).y, 1u};
#else
    Backend<ReduceDPP<>> backend("GPU");
    const uint architecture = blocksThatRead<LaunchAuto>(1, NO_WORKSPACE).y;
    uint clusterBlocks = 1;
    if (architecture >= 900) {
        cudaLaunchConfig_t config = {};
        config.gridDim = dim3(8, 1, 1);
        config.blockDim = dim3(256, 1, 1);
        config.stream = backend.stream.getCUDAStream();
        int largest = 0;
        if (cudaOccupancyMaxPotentialClusterSize(&largest, clusterProbe_Kernel<int>, &config) != cudaSuccess) {
            (void)cudaGetLastError();
            largest = 0;
        }
        clusterBlocks = largest < 2 ? 1u : (largest > 8 ? 8u : static_cast<uint>(largest));
    }
    return {architecture, clusterBlocks};
#endif
}

// Whether the blocks that read `elements` elements with a policy, in an execution with a workspace or without one,
// are the ones of the grid that the executor has to launch on the device: the blocks of that grid that have elements,
// and none (the execution throws) when that grid needs a workspace and there is none
template <typename Policy>
bool readByBlocks(const char* const policy, const uint elements, const bool withWorkspace, const Device& device) {
    const reduce_detail::Launch launch = device.template launch<Policy>(elements, withWorkspace);
    const uint blocks = launch.clusters > 1 && !withWorkspace
                            ? 0u
                            : static_cast<uint>(reduce_detail::groupsOf(elements, launch.blockElements));
    const uint2 seen = blocksThatRead<Policy>(elements, withWorkspace);
    if (seen.x != blocks) {
        std::printf("ReduceDPP GPU, levels of the kernel: %u elements with %s, %s a workspace, were read by %u blocks, "
                    "not by %u FAIL\n", elements, policy, withWorkspace ? "with" : "without", seen.x, blocks);
    }
    return seen.x == blocks;
}

// The blocks that read the elements of every plane of a batch of `planes` planes of `elements` elements, with the
// workspace that it needs: the ones that the launch rule gives a plane of that batch, which are the part of each plane
// of the blocks counted for all of them
template <typename Policy>
bool batchReadByBlocks(const char* const policy, const uint elements, const uint planes, const Device& device) {
    using DPP = ReduceDPP<GPU_PA, Policy>;
    Backend<DPP> backend("GPU");
    Tensor<uint2> output(1, 1, planes, 1, Backend<DPP>::MEMORY);
    for (int z = 0; z < static_cast<int>(planes); ++z) {
        output.at(Point{0, 0, z}) = make_<uint2>(0u, 0u);
    }
    backend.upload(output);
    backend.reduce(BlockAndImageRead::build(ActiveThreads{elements, 1, planes}),
                   make_tuple(make_tuple(Cast<uint2, uint2>::build(), Max<uint2, uint2, uint2, UnaryType>::build(),
                                         Cast<uint2, uint2>::build())),
                   TensorWrite<uint2>::build(output.ptr()));
    backend.download(output);
    backend.sync();
    const reduce_detail::Launch launch = reduce_detail::launchOf<Policy>(elements, planes, device.clusterBlocks, true);
    const uint blocks = static_cast<uint>(reduce_detail::groupsOf(elements, launch.blockElements));
    bool ok = true;
    for (int z = 0; z < static_cast<int>(planes); ++z) {
        ok = ok && output.at(Point{0, 0, z}).x + 1u == blocks;
    }
    if (!ok) {
        std::printf("ReduceDPP GPU, levels of the kernel: the planes of a batch of %u planes of %u elements with %s "
                    "were read by %u blocks (the first one), not by %u FAIL\n", planes, elements, policy,
                    output.at(Point{0, 0, 0}).x + 1u, blocks);
    }
    return ok;
}

// 18. Which levels of the kernel an execution uses. The executor has to launch clusters exactly where the image of the
//     kernels that runs on the device has them, with the size that the device accepts (it finds out from
//     __CUDA_ARCH_LIST__ and from the driver), and only in the executions that the policy says. The blocks that read
//     elements say whether a plane was reduced by one block, by the blocks of one cluster, or by all the blocks that
//     the policy gives it: without a workspace, a plane of 20000 elements is read by the 8 blocks of one cluster
//     where there are clusters of 8, and the execution throws where there are none; with a workspace, by 40 blocks.
bool levelsOfTheKernel(const Device& device) {
#if defined(FK_HIP_BACKEND)
    std::printf("ReduceDPP GPU: GPU_AMD, without thread block clusters: the cluster level is not run\n");
#else
    if (device.clusterBlocks > 1) {
        std::printf("ReduceDPP GPU: this device runs the kernels compiled for the architecture %u, with thread block "
                    "clusters of up to %u blocks\n", device.architecture, device.clusterBlocks);
    } else {
        std::printf("ReduceDPP GPU: this device runs the kernels compiled for the architecture %u, WITHOUT thread "
                    "block clusters: the cluster level is not run\n", device.architecture);
    }
#endif
    bool ok = true;
    for (const bool withWorkspace : {NO_WORKSPACE, WITH_WORKSPACE}) {
        // AUTO: one block; without a workspace, one cluster where the device has them and the plane fits, and an
        // exception otherwise; with a workspace, blocks that synchronize through it
        for (const uint elements : {4096u, 4097u, 12000u, 20000u, 32768u, 32769u, 40000u, 5000000u}) {
            ok = readByBlocks<LaunchAuto>("AUTO", elements, withWorkspace, device) && ok;
        }
        // NEVER: never a cluster
        ok = readByBlocks<LaunchNever>("NEVER", 4096, withWorkspace, device) && ok;
        ok = readByBlocks<LaunchNever>("NEVER", 20000, withWorkspace, device) && ok;
        // ALWAYS: one cluster also with a workspace, and the blocks of larger planes in clusters
        ok = readByBlocks<LaunchAlways>("ALWAYS", 20000, withWorkspace, device) && ok;
        ok = readByBlocks<LaunchAlways>("ALWAYS", 40000, withWorkspace, device) && ok;
    }
    // The blocks are counted for all the planes of a batch: 128 for 30 planes of 65536 elements, 5 for each plane
    ok = batchReadByBlocks<LaunchAuto>("AUTO", 65536, 30, device) && ok;
    ok = batchReadByBlocks<LaunchAuto>("AUTO", 262144, 128, device) && ok;
    ok = batchReadByBlocks<LaunchNever>("NEVER", 20000, 7, device) && ok;
    ok = batchReadByBlocks<LaunchAlways>("ALWAYS", 300000, 6, device) && ok;
    std::printf("ReduceDPP GPU, levels of the kernel %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// 26. The executions of the default policy without a workspace: one block reduces a plane of up to 4096 elements; one
//     cluster a plane of up to 8 times that, where the device has clusters of 8 blocks; and every other execution
//     throws std::invalid_argument, which says the bytes of the workspace that it needs, and writes nothing.
bool withoutWorkspace(const Device& device) {
    using DPP = ReduceDPP<>;
    Backend<DPP> backend("GPU");
    bool ok = true;
    for (const int elements : {1, 4096, 4097, 12288, 12289, 20000, 32768, 32769, 300000}) {
        auto input = makeImage<float, DPP>(elements, 1, 50);
        auto output = makeOutput<float, DPP>(4);
        backend.upload(input);
        backend.upload(output);
        const auto read = PerThreadRead<ND::_2D, float>::build(input);
        const bool executes =
            device.template launch<ReduceDPPPolicy>(static_cast<ulonglong>(elements), NO_WORKSPACE).clusters == 1;
        std::string message;
        try {
            backend.reduceWithoutWorkspace(read, sumMaxMin<float>(), PerThreadWrite<ND::_1D, float>::build(output));
        } catch (const std::invalid_argument& error) {
            message = error.what();
        }
        backend.download(output);
        backend.sync();
        bool right = output.at(Point{3, 0, 0}) == static_cast<float>(SENTINEL);
        if (executes) {
            right = right && message.empty() &&
                    isSumMaxMin(output.at(Point{0, 0, 0}), output.at(Point{1, 0, 0}), output.at(Point{2, 0, 0}),
                                expectedOf<float>(elements, 1, 50));
        } else {
            const size_t bytes = DPP::workspaceBytes(read, sumMaxMin<float>());
            right = right && bytes > 0 && message.find(std::to_string(bytes) + " bytes") != std::string::npos;
            for (int i = 0; i < 3; ++i) {
                right = right && output.at(Point{i, 0, 0}) == static_cast<float>(SENTINEL);
            }
        }
        if (!right) {
            std::printf("ReduceDPP GPU, without a workspace: %d elements %s FAIL\n", elements,
                        executes ? "must be reduced" : "must throw");
        }
        ok = ok && right;
    }
    return report(ok, "executions without a workspace", backend.name);
}

#endif // defined(__NVCC__) || defined(FK_HIP_BACKEND)

#if defined(__NVCC__)
// A FirstIOp that needs many registers: a chain of values, each one computed from the previous one, that are all
// used at the end in the opposite order
struct ManyRegisters {
private:
    using SelfType = ManyRegisters;
public:
    FK_STATIC_STRUCT(ManyRegisters, SelfType)
    using Parent = UnaryOperation<float, float, ManyRegisters>;
    DECLARE_UNARY_PARENT
    FK_HOST_DEVICE_FUSE OutputType exec(const InputType input) {
        constexpr int N = 200;
        float chain[N] = {};
        chain[0] = input;
        for (int i = 1; i < N; ++i) {
            chain[i] = chain[i - 1] * chain[i - 1] + 0.001f * static_cast<float>(i);
        }
        float result = 0.f;
        for (int i = 0; i < N; ++i) {
            result += chain[i] * chain[N - 1 - i];
        }
        return result * 0.f + 1.f;
    }
};

// 30. Blocks of more than 256 threads: when the kernel needs too many registers for them the execution throws an
//     error that says so, before anything is launched (whether it does depends on the build: the device is asked),
//     and the next execution works
struct BlocksOf1024 : ReduceDPPPolicy {
    static constexpr uint BLOCK_THREADS = 1024;
};
bool tooManyThreads() {
    using DPP = ReduceDPP<ParArch::GPU_NVIDIA, BlocksOf1024>;
    Backend<DPP> backend("GPU");
    auto output = makeOutput<float, DPP>(2);
    backend.upload(output);
    const auto same = Cast<float, float>::build();
    const auto add = Add<float, float, float, UnaryType>::build();
    const auto read = ReadSet<float>::build(0.5f, ActiveThreads{640, 480, 1});
    const auto reductions = make_tuple(make_tuple(ManyRegisters::build(), add, same));
    const auto write = PerThreadWrite<ND::_1D, float>::build(output);
    using ReadIOp = std::decay_t<decltype(read)>;
    using Reductions = std::decay_t<decltype(reductions)>;
    using WriteIOp = std::decay_t<decltype(write)>;
    using KernelDetails = typename DPP::template KernelDetails<reduce_detail::PartialOf<Reductions>>;
    const auto kernel = (void (*)(const KernelDetails, const ReadIOp, const Reductions, const WriteIOp))
        launchDPP_Kernel<DPP, KernelDetails, ReadIOp, Reductions, WriteIOp>;
    cudaFuncAttributes attributes{};
    gpuErrchk(cudaFuncGetAttributes(&attributes, kernel));
    const bool fits = attributes.maxThreadsPerBlock >= 1024;
    std::string message;
    try {
        backend.reduce(read, reductions, write);
    } catch (const std::runtime_error& error) {
        message = error.what();
    }
    bool ok = fits ? message.empty() : message.find("registers per thread") != std::string::npos;
    // The next execution, with the blocks of the default policy, which can always be launched (in a build with
    // device debug information not even this kernel can be launched in blocks of 1024 threads)
    executeOperations<ReduceDPP<>>(backend.stream, ReadSet<float>::build(2.f, ActiveThreads{50, 40, 1}),
                                   make_tuple(make_tuple(same, add, same)), write);
    backend.download(output);
    backend.sync();
    ok = ok && output.at(Point{0, 0, 0}) == 4000.f && output.at(Point{1, 0, 0}) == static_cast<float>(SENTINEL);
    return report(ok, "blocks of 1024 threads for a kernel that needs many registers", "GPU");
}

// 31. Where the elements past 2^32 of a plane are, on the device: the last blocks of a plane of 65536 x 65700
//     elements start after element 2^32, and read the last row
bool lastRowsOfMoreThan4GElements() {
    using DPP = ReduceDPP<>;
    constexpr uint W = 65536, H = 65700;
    Backend<DPP> backend("GPU");
    Ptr1D<uint> output(3, 0, Backend<DPP>::MEMORY);
    for (int i = 0; i < 3; ++i) {
        output.at(Point{i, 0, 0}) = 7u;
    }
    backend.upload(output);
    const auto same = Cast<uint, uint>::build();
    backend.reduce(RowRead::build(ActiveThreads{W, H, 1}),
                   make_tuple(make_tuple(same, Max<uint, uint, uint, UnaryType>::build(), same),
                              make_tuple(same, Min<uint, uint, uint, UnaryType>::build(), same)),
                   PerThreadWrite<ND::_1D, uint>::build(output));
    backend.download(output);
    backend.sync();
    return report(output.at(Point{0, 0, 0}) == H - 1 && output.at(Point{1, 0, 0}) == 0u &&
                      output.at(Point{2, 0, 0}) == 7u,
                  "rows of a plane of 65536 x 65700 elements", "GPU");
}

// 32. The error of a CUDA call that the executor makes is not left as the last error of the thread, which the other
//     DPPs of the library check after their launches: a call that fails, as the executor takes its error, and the
//     TransformDPP execution after it
bool errorNotLeftBehind() {
    using DPP = ReduceDPP<>;
    Backend<DPP> backend("GPU");
    auto image = makeImage<float, DPP>(16, 16, 51);
    backend.upload(image);
    cudaFuncAttributes attributes{};
    const cudaError_t code = reduce_detail::clearingLastError(cudaFuncGetAttributes(&attributes, nullptr));
    bool nextWorks = true;
    try {
        executeOperations<TransformDPP<ParArch::GPU_NVIDIA>>(backend.stream,
                                                             ReadSet<float>::build(3.f, ActiveThreads{16, 16, 1}),
                                                             PerThreadWrite<ND::_2D, float>::build(image));
        backend.download(image);
        backend.sync();
    } catch (const std::runtime_error&) {
        nextWorks = false;
    }
    return report(code != cudaSuccess && nextWorks && image.at(Point{5, 5, 0}) == 3.f,
                  "CUDA call that fails, and the execution after it", "GPU");
}

// The kernel nodes, and the nodes that are not kernels, of a graph
struct GraphNodes {
    int kernels;
    int others;
};

GraphNodes nodesOf(const cudaGraph_t graph) {
    size_t count = 0;
    gpuErrchk(cudaGraphGetNodes(graph, nullptr, &count));
    std::vector<cudaGraphNode_t> nodes(count);
    if (count > 0) {
        gpuErrchk(cudaGraphGetNodes(graph, nodes.data(), &count));
    }
    GraphNodes found{0, 0};
    for (const cudaGraphNode_t node : nodes) {
        cudaGraphNodeType type{};
        gpuErrchk(cudaGraphNodeGetType(node, &type));
        if (type == cudaGraphNodeTypeKernel) {
            ++found.kernels;
        } else {
            ++found.others;
        }
    }
    return found;
}

// 36. Blocks of more than 256 threads in an execution that is being captured in a graph. CUDA does not reject the
//     launch of a kernel that is being captured (it rejects the instantiation of the graph, later), so the executor
//     asks the device before the launch: the execution throws the error of tooManyThreads() here too, and the capture
//     ends valid and without nodes.
bool tooManyThreadsInACapture() {
    using DPP = ReduceDPP<ParArch::GPU_NVIDIA, BlocksOf1024>;
    Backend<DPP> backend("GPU");
    auto output = makeOutput<float, DPP>(2);
    backend.upload(output);
    const auto same = Cast<float, float>::build();
    const auto read = ReadSet<float>::build(0.5f, ActiveThreads{640, 480, 1});
    const auto reductions = make_tuple(make_tuple(ManyRegisters::build(),
                                                  Add<float, float, float, UnaryType>::build(), same));
    const auto write = PerThreadWrite<ND::_1D, float>::build(output);
    using ReadIOp = std::decay_t<decltype(read)>;
    using Reductions = std::decay_t<decltype(reductions)>;
    using WriteIOp = std::decay_t<decltype(write)>;
    using KernelDetails = typename DPP::template KernelDetails<reduce_detail::PartialOf<Reductions>>;
    const auto kernel = (void (*)(const KernelDetails, const ReadIOp, const Reductions, const WriteIOp))
        launchDPP_Kernel<DPP, KernelDetails, ReadIOp, Reductions, WriteIOp>;
    cudaFuncAttributes attributes{};
    gpuErrchk(cudaFuncGetAttributes(&attributes, kernel));
    const bool fits = attributes.maxThreadsPerBlock >= 1024;
    backend.needWorkspace(DPP::workspaceBytes(read, reductions));
    backend.sync();
    const cudaStream_t stream = backend.stream.getCUDAStream();
    cudaGraph_t graph = nullptr;
    gpuErrchk(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    std::string message;
    try {
        backend.reduce(read, reductions, write);
    } catch (const std::runtime_error& error) {
        message = error.what();
    }
    bool ok = cudaStreamEndCapture(stream, &graph) == cudaSuccess && graph != nullptr;
    if (ok) {
        const GraphNodes nodes = nodesOf(graph);
        ok = fits ? message.empty() && nodes.kernels == 1
                  : message.find("registers per thread") != std::string::npos && nodes.kernels == 0 &&
                        nodes.others == 0;
    } else {
        (void)cudaGetLastError();
    }
    if (graph != nullptr) {
        (void)cudaGraphDestroy(graph);
    }
    return report(ok, "blocks of 1024 threads in an execution that is being captured", "GPU");
}

// 23. An execution captured in a graph, in the global mode, and the graph launched three times with other data: an
//     execution is one kernel and nothing else, also when its planes synchronize through the workspace (which is
//     allocated and zeroed before the capture, and whose counters every launch leaves at zero for the next one).
template <typename Policy>
bool capturedExecution(const int width, const int height, const char* const what) {
    using DPP = ReduceDPP<ParArch::GPU_NVIDIA, Policy>;
    Backend<DPP> backend("GPU");
    auto input = makeImage<float, DPP>(width, height, 30);
    auto output = makeOutput<float, DPP>(4);
    backend.upload(input);
    backend.upload(output);
    const auto read = PerThreadRead<ND::_2D, float>::build(input);
    const auto write = PerThreadWrite<ND::_1D, float>::build(output);
    backend.needWorkspace(DPP::workspaceBytes(read, sumMaxMin<float>()));
    backend.sync();
    const cudaStream_t stream = backend.stream.getCUDAStream();
    cudaGraph_t graph = nullptr;
    gpuErrchk(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    bool captured = true;
    try {
        backend.reduce(read, sumMaxMin<float>(), write);
    } catch (const std::exception& error) {
        std::printf("ReduceDPP %s: %s\n", what, error.what());
        captured = false;
    }
    captured = cudaStreamEndCapture(stream, &graph) == cudaSuccess && captured;
    if (!captured) {
        (void)cudaGetLastError();
        if (graph != nullptr) {
            (void)cudaGraphDestroy(graph);
        }
        return report(false, what, backend.name);
    }
    const GraphNodes nodes = nodesOf(graph);
    bool ok = nodes.kernels == 1 && nodes.others == 0;
    cudaGraphExec_t executable = nullptr;
    gpuErrchk(cudaGraphInstantiate(&executable, graph, 0));
    for (int launch = 0; launch < 3; ++launch) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                input.at(Point{x, y, 0}) = element<float>(x, y, 31 + launch);
            }
        }
        backend.upload(input);
        gpuErrchk(cudaGraphLaunch(executable, stream));
        backend.download(output);
        backend.sync();
        ok = ok && isSumMaxMin(output.at(Point{0, 0, 0}), output.at(Point{1, 0, 0}), output.at(Point{2, 0, 0}),
                               expectedOf<float>(width, height, 31 + launch)) &&
             output.at(Point{3, 0, 0}) == static_cast<float>(SENTINEL);
    }
    gpuErrchk(cudaGraphExecDestroy(executable));
    gpuErrchk(cudaGraphDestroy(graph));
    return report(ok, what, backend.name);
}

// 24. An execution made while another thread captures its own stream in the global mode, in which CUDA refuses, to
//     every thread, the calls that allocate memory or synchronize: the execution works, and the capture of the other
//     thread stays valid
template <typename Policy>
bool executionWhileAnotherThreadCaptures(const int width, const int height, const char* const what) {
    using DPP = ReduceDPP<ParArch::GPU_NVIDIA, Policy>;
    Backend<DPP> backend("GPU");
    auto input = makeImage<float, DPP>(width, height, 40);
    auto output = makeOutput<float, DPP>(4);
    backend.upload(input);
    backend.upload(output);
    const auto read = PerThreadRead<ND::_2D, float>::build(input);
    const auto write = PerThreadWrite<ND::_1D, float>::build(output);
    backend.needWorkspace(DPP::workspaceBytes(read, sumMaxMin<float>()));
    backend.sync();
    std::promise<void> capturing;
    std::promise<void> executed;
    cudaError_t begun = cudaErrorUnknown;
    cudaError_t ended = cudaErrorUnknown;
    std::thread other([&capturing, &executed, &begun, &ended]() {
        cudaStream_t stream = nullptr;
        cudaGraph_t graph = nullptr;
        begun = cudaStreamCreate(&stream);
        if (begun == cudaSuccess) {
            begun = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        }
        capturing.set_value();
        executed.get_future().wait();
        if (begun == cudaSuccess) {
            ended = cudaStreamEndCapture(stream, &graph);
        }
        if (graph != nullptr) {
            (void)cudaGraphDestroy(graph);
        }
        if (stream != nullptr) {
            (void)cudaStreamDestroy(stream);
        }
        (void)cudaGetLastError();
    });
    capturing.get_future().wait();
    bool threw = false;
    try {
        backend.reduce(read, sumMaxMin<float>(), write);
    } catch (const std::exception& error) {
        std::printf("ReduceDPP %s: %s\n", what, error.what());
        threw = true;
    }
    executed.set_value();
    other.join();
    backend.download(output);
    backend.sync();
    const bool ok = !threw && begun == cudaSuccess && ended == cudaSuccess &&
                    isSumMaxMin(output.at(Point{0, 0, 0}), output.at(Point{1, 0, 0}), output.at(Point{2, 0, 0}),
                                expectedOf<float>(width, height, 40)) &&
                    output.at(Point{3, 0, 0}) == static_cast<float>(SENTINEL);
    return report(ok, what, backend.name);
}
#endif // defined(__NVCC__)

// The sizes of the matrices: one element, rows, columns and the sizes around those of a block (256 threads, 4096
// elements), of a cluster (8 blocks), of a chunk (16384 elements), and of the blocks of the default policy (128 blocks
// of 512 elements from 65536 elements, and more than 128 blocks from 2097152)
struct MatrixSize {
    int width, height;
};
constexpr std::array<MatrixSize, 26> MATRIX_SIZES{{{1, 1}, {2, 1}, {1, 2}, {255, 1}, {256, 1}, {257, 1}, {1, 300},
                                                   {31, 33}, {4095, 1}, {4096, 1}, {4097, 1}, {64, 64}, {3, 2731},
                                                   {8193, 1}, {12289, 1}, {16385, 1}, {181, 181}, {32768, 1},
                                                   {32769, 1}, {257, 129}, {65537, 1}, {640, 480}, {1000, 1001},
                                                   {1920, 1080}, {2048, 1100}, {1, 70001}}};

// The cases that every backend runs
template <typename DPP>
bool allCases(const char* const name) {
    Backend<DPP> backend(name);
    bool ok = true;
    for (const MatrixSize& size : MATRIX_SIZES) {
        ok = floatMatrix<float3, DPP>(backend, size.width, size.height) && ok;
    }
    ok = floatMatrix<float, DPP>(backend, 1921, 1081) && ok;
    ok = floatMatrix<float4, DPP>(backend, 333, 77) && ok;
    ok = floatMatrix<double2, DPP>(backend, 3840, 2160) && ok;
    ok = integerMatrix<uchar, DPP>(backend, 3840, 2160) && ok;
    ok = integerMatrix<ushort, DPP>(backend, 1023, 513) && ok;
    ok = integerMatrix<uchar3, DPP>(backend, 1920, 1080) && ok;
    ok = integerMatrix<uchar3, DPP>(backend, 5, 3) && ok;
    for (const int elements : {1, 255, 4097, 100000, (1 << 22) + 3}) {
        ok = oneReduction1D(backend, elements) && ok;
    }
    ok = copiedSingleReduction(backend) && ok;
    ok = fusedReadAndWrite(backend) && ok;
    ok = readBack(backend) && ok;
    ok = batchOfCrops<float3, DPP>(backend) && ok;
    ok = batchOfCrops<float, DPP>(backend) && ok;
    ok = otherReads(backend) && ok;
    ok = tupleAccumulator(backend) && ok;
    ok = rejected(backend) && ok;
    ok = enqueuedExecutions(backend) && ok;
    ok = documentedExample(backend) && ok;
    for (const MatrixSize& size : MATRIX_SIZES) {
        ok = distinctFirstIOps(backend, size.width, size.height) && ok;
    }
    ok = readsOnce(backend, 1000, 1001) && ok;
    ok = readsOnce(backend, 7, 1) && ok;
    if constexpr (!IS_GPU<Backend<DPP>::PA>) {
        ok = iopThrows(backend, false) && ok;
        ok = iopThrows(backend, true) && ok;
    }
    for (const MatrixSize& size : {MatrixSize{1, 1}, MatrixSize{200, 1}, MatrixSize{640, 480}}) {
        ok = manyAccumulators(backend, size.width, size.height) && ok;
        ok = rangeAccumulator(backend, size.width, size.height) && ok;
    }
    for (const int width : {3, 5000, 40000}) {
        ok = inPlace(backend, width) && ok;
    }
    if constexpr (!IS_GPU<Backend<DPP>::PA>) {
        for (const int width : {3, 100, 40000}) {
            ok = everyPlaneReadFirst(backend, width) && ok;
        }
        for (const MatrixSize& size : MATRIX_SIZES) {
            ok = accumulatedInOrder(backend, size.width, size.height) && ok;
        }
    }
    ok = workspaceCases(backend) && ok;
    ok = planesOfAGrid(backend) && ok;
#if defined(NDEBUG) && (defined(__OPTIMIZE__) || defined(_MSC_VER))
    // (Only in an optimized build: the host backends visit every element of the plane.)
    ok = moreThan4GElements(backend) && ok;
#endif
    std::printf("ReduceDPP %s %s\n", name, ok ? "PASS" : "FAIL");
    return ok;
}

// The cases of the policies that split the data in many small pieces: every size, a batch, the Tuple accumulator and
// the executions enqueued together
template <typename DPP>
bool splitCases(const char* const name) {
    Backend<DPP> backend(name);
    bool ok = true;
    for (const MatrixSize& size : MATRIX_SIZES) {
        ok = floatMatrix<float3, DPP>(backend, size.width, size.height) && ok;
    }
    ok = integerMatrix<uchar3, DPP>(backend, 1920, 1080) && ok;
    ok = oneReduction1D(backend, 100000) && ok;
    ok = batchOfCrops<float3, DPP>(backend) && ok;
    ok = tupleAccumulator(backend) && ok;
    ok = enqueuedExecutions(backend) && ok;
    for (const MatrixSize& size : MATRIX_SIZES) {
        ok = distinctFirstIOps(backend, size.width, size.height) && ok;
    }
    ok = readsOnce(backend, 1000, 1001) && ok;
    if constexpr (!IS_GPU<Backend<DPP>::PA>) {
        ok = iopThrows(backend, false) && ok;
        ok = iopThrows(backend, true) && ok;
    }
    ok = manyAccumulators(backend, 640, 480) && ok;
    ok = rangeAccumulator(backend, 640, 480) && ok;
    ok = inPlace(backend, 5000) && ok;
    if constexpr (!IS_GPU<Backend<DPP>::PA>) {
        ok = everyPlaneReadFirst(backend, 5000) && ok;
        for (const MatrixSize& size : MATRIX_SIZES) {
            ok = accumulatedInOrder(backend, size.width, size.height) && ok;
        }
    }
    ok = workspaceCases(backend) && ok;
    std::printf("ReduceDPP %s %s\n", name, ok ? "PASS" : "FAIL");
    return ok;
}

// The square of a float: a multiplication, whose result the reductions add
struct Square {
private:
    using SelfType = Square;
public:
    FK_STATIC_STRUCT(Square, SelfType)
    using Parent = UnaryOperation<float, float, Square>;
    DECLARE_UNARY_PARENT
    FK_HOST_DEVICE_FUSE OutputType exec(const InputType input) {
        return input * input;
    }
};

constexpr int RANDOM_W = 1237;
constexpr int RANDOM_H = 811;

// Floats whose sums are not exact: the results depend on the order of the additions
template <typename DPP>
Ptr2D<float> randomImage() {
    Ptr2D<float> image(RANDOM_W, RANDOM_H, 0, Backend<DPP>::MEMORY);
    std::mt19937 generator(7);
    std::uniform_real_distribution<float> value(0.f, 1.f);
    for (int y = 0; y < RANDOM_H; ++y) {
        for (int x = 0; x < RANDOM_W; ++x) {
            image.at(Point{x, y, 0}) = value(generator);
        }
    }
    return image;
}

// The sum of the floats and the sum of their squares, both accumulated in float
template <typename DPP>
std::array<float, 2> inexactSums(const char* const name) {
    Backend<DPP> backend(name);
    auto input = randomImage<DPP>();
    auto output = makeOutput<float, DPP>(2);
    backend.upload(input);
    backend.upload(output);
    const auto same = Cast<float, float>::build();
    const auto add = Add<float, float, float, UnaryType>::build();
    backend.reduce(PerThreadRead<ND::_2D, float>::build(input),
                   make_tuple(make_tuple(same, add, same), make_tuple(Square::build(), add, same)),
                   PerThreadWrite<ND::_1D, float>::build(output));
    backend.download(output);
    backend.sync();
    return {output.at(Point{0, 0, 0}), output.at(Point{1, 0, 0})};
}

bool closeTo(const float value, const double reference, const double tolerance) {
    return std::fabs(static_cast<double>(value) - reference) <= tolerance * std::fabs(reference);
}

// 12. Every backend gives, closely, the sums that a double gives, with its default policy (this is its accuracy).
//     ParArch::CPU and CPU_MULTITHREAD accumulate the same chunks in the same order: their sums are the same bytes
//     (the sums of the squares, where the compiler can contract a multiplication and an addition differently in each
//     of them, are compared closely).
bool inexactSumsOfEveryBackend() {
    const auto input = randomImage<ReduceDPP<ParArch::CPU>>();
    double sum = 0.;
    double squares = 0.;
    for (int y = 0; y < RANDOM_H; ++y) {
        for (int x = 0; x < RANDOM_W; ++x) {
            const double value = static_cast<double>(input.at(Point{x, y, 0}));
            sum += value;
            squares += value * value;
        }
    }
    const auto cpu = inexactSums<ReduceDPP<ParArch::CPU>>("CPU");
    bool ok = closeTo(cpu[0], sum, 1e-4) && closeTo(cpu[1], squares, 1e-4);
#ifdef FK_CPU_MULTITHREAD_ENABLED
    const auto multithread = inexactSums<ReduceDPP<ParArch::CPU_MULTITHREAD>>("CPU_MULTITHREAD");
    ok = ok && sameBits(multithread[0], cpu[0]) && closeTo(multithread[1], static_cast<double>(cpu[1]), 1e-5);
#endif
#if defined(__NVCC__) || defined(FK_HIP_BACKEND)
    const auto gpu = inexactSums<ReduceDPP<>>("GPU");
    ok = ok && closeTo(gpu[0], sum, 1e-4) && closeTo(gpu[1], squares, 1e-4);
#endif
    std::printf("ReduceDPP sums of floats that are not exact %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

} // namespace

int launch_impl() {
    bool ok = allCases<ReduceDPP<ParArch::CPU>>("CPU");
    ok = lastChunkOfMoreThan4GElements() && ok;
    ok = splitCases<ReduceDPP<ParArch::CPU, SmallChunks>>("CPU, chunks of 37 elements") && ok;
#ifdef FK_CPU_MULTITHREAD_ENABLED
    ok = allCases<ReduceDPP<ParArch::CPU_MULTITHREAD>>("CPU_MULTITHREAD") && ok;
    ok = splitCases<ReduceDPP<ParArch::CPU_MULTITHREAD, SmallChunks>>("CPU_MULTITHREAD, chunks of 37 elements") && ok;
#endif
#if defined(__NVCC__)
    const Device device = deviceOfTheRun();
    ok = capturedExecution<LaunchNever>(640, 480, "execution with a workspace, captured in a graph") && ok;
    ok = capturedExecution<LaunchAlways>(20000, 1,
                                         "execution of one cluster (or with a workspace), captured in a graph") && ok;
    ok = executionWhileAnotherThreadCaptures<LaunchNever>(640, 480,
                                                          "execution with a workspace while another thread captures") &&
         ok;
    ok = executionWhileAnotherThreadCaptures<LaunchAlways>(20000, 1,
                                                           "execution of one cluster while another thread captures") &&
         ok;
#endif
#if defined(__NVCC__) || defined(FK_HIP_BACKEND)
    ok = allCases<ReduceDPP<>>("GPU") && ok;
    ok = workspaceOfMoreThan4GB() && ok;
    ok = workspaceOfABatch<ReduceDPPPolicy>("the default policy", 65536, 30) && ok;
    ok = workspaceOfABatch<ReduceDPPPolicy>("the default policy", 5000, 64) && ok;
    ok = workspaceOfABatch<ReduceDPPPolicy>("the default policy", 300000, 6) && ok;
    ok = workspaceOfABatch<NoClusters>("NoClusters", 300000, 6) && ok;
    ok = workspaceOfABatch<ClustersOf3>("ClustersOf3", 300000, 6) && ok;
    ok = workspaceOfABatch<BlocksOf32>("BlocksOf32", 100000, 7) && ok;
    ok = manyPlanesNeedNoWorkspace() && ok;
    ok = workspaceAlignment() && ok;
#endif
#if defined(FK_HIP_BACKEND)
    // GPU_AMD has no clusters: ClustersOf3 launches plain blocks of 700 elements, which synchronize through the
    // workspace
    const Device device = deviceOfTheRun();
    ok = splitCases<ReduceDPP<ParArch::GPU_AMD, NoClusters>>("GPU, without clusters") && ok;
    ok = splitCases<ReduceDPP<ParArch::GPU_AMD, ClustersOf3>>("GPU, the policy of clusters of 3 blocks") && ok;
    ok = splitCases<ReduceDPP<ParArch::GPU_AMD, BlocksOf32>>("GPU, blocks of 32 threads") && ok;
    ok = levelsOfTheKernel(device) && ok;
    ok = withoutWorkspace(device) && ok;
#endif
#if defined(__NVCC__)
    ok = splitCases<ReduceDPP<ParArch::GPU_NVIDIA, NoClusters>>("GPU, without clusters") && ok;
    ok = splitCases<ReduceDPP<ParArch::GPU_NVIDIA, ClustersOf3>>("GPU, clusters of 3 blocks") && ok;
    ok = splitCases<ReduceDPP<ParArch::GPU_NVIDIA, BlocksOf32>>("GPU, blocks of 32 threads") && ok;
    ok = levelsOfTheKernel(device) && ok;
    ok = withoutWorkspace(device) && ok;
    ok = tooManyThreads() && ok;
    ok = tooManyThreadsInACapture() && ok;
    ok = errorNotLeftBehind() && ok;
#if defined(NDEBUG) && (defined(__OPTIMIZE__) || defined(_MSC_VER))
    ok = lastRowsOfMoreThan4GElements() && ok;
#endif
#endif
    ok = inexactSumsOfEveryBackend() && ok;
    ok = ::reduceDPPReadmeExample() && ok;
    std::printf("ReduceDPP: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : -1;
}

} // namespace fk

// 37. The example of README.md (and of the fkl-using-the-library skill, which has the same lines without the stream
//     and the image) as a user compiles it: outside namespace fk, after `using namespace fk;`, with ReduceDPP<>, the
//     DPP of the default backend of the build, and with every name written as the README writes it. The lines between
//     the two marks are the ones of the README, and have to stay the same as there: a name that the documents qualify
//     with fk:: and that fk does not have (the scalar aliases uchar and uint are global) does not compile here.
//     documentedExample() has the same example for every backend, written inside namespace fk.
namespace reduce_dpp_readme {
using namespace fk;

constexpr int README_W = 640;
constexpr int README_H = 480;

// What the README does not show: where its image comes from
Ptr2D<fk::uchar3> getGPUSourceImage(Stream& stream) {
    auto image = makeImage<fk::uchar3, ReduceDPP<>>(README_W, README_H, 9);
#if defined(__NVCC__) || defined(FK_HIP_BACKEND)
    image.upload(stream);
#endif
    (void)stream;
    return image;
}

bool example() {
// ---- README.md, "Reductions: ReduceDPP": the lines after `using namespace fk;` ----
Stream stream;
const Ptr2D<fk::uchar3> image = getGPUSourceImage(stream);
Ptr1D<fk::float3> results(3); // one element per reduction

// The mean, the maximum and the minimum of every channel of the image, in one kernel
const float pixels = static_cast<float>(image.dims().width * image.dims().height);
const auto read = PerThreadRead<ND::_2D, fk::uchar3>::build(image);
const auto reductions = make_tuple(
    make_tuple(Cast<fk::uchar3, fk::float3>::build(),                           // FirstIOp
               Add<fk::float3, fk::float3, fk::float3, UnaryType>::build(),     // AccumulationIOp
               Div<fk::float3>::build(make_set<fk::float3>(pixels))),           // FinalIOp
    make_tuple(Cast<fk::uchar3, fk::uchar3>::build(),
               Max<fk::uchar3, fk::uchar3, fk::uchar3, UnaryType>::build(),
               Cast<fk::uchar3, fk::float3>::build()),
    make_tuple(Cast<fk::uchar3, fk::uchar3>::build(),
               Min<fk::uchar3, fk::uchar3, fk::uchar3, UnaryType>::build(),
               Cast<fk::uchar3, fk::float3>::build()));
const auto write = PerThreadWrite<ND::_1D, fk::float3>::build(results);

// Once: the workspace, device memory where the thread blocks of a plane leave their values. It starts at zero.
const size_t bytes = ReduceDPP<>::workspaceBytes(read, reductions); // 0 when no workspace is needed
Ptr1D<uchar> workspace(static_cast<uint>(bytes));
if (bytes > 0) {
    setTo(uchar{0}, workspace, stream);
}

// Every time
executeOperations<ReduceDPP<>>(stream, ReduceDPP<>::build_details(workspace), read, reductions, write);

stream.sync();
// ---- end of the lines of README.md ----
#if defined(__NVCC__) || defined(FK_HIP_BACKEND)
    results.download(stream);
    stream.sync();
#endif
    const Expected expected = expectedOf<fk::uchar3>(README_W, README_H, 9);
    bool ok = sameChannels(results.at(Point{1, 0, 0}), expected.maximum) &&
              sameChannels(results.at(Point{2, 0, 0}), expected.minimum);
    for (int c = 0; c < 3; ++c) {
        const double mean = expected.sum[c] / static_cast<double>(README_W * README_H);
        ok = ok && std::fabs(channel(results.at(Point{0, 0, 0}), c) - mean) <= 1e-4 * mean;
    }
    return report(ok, "example of the README, with the lines of the README", "default backend");
}
} // namespace reduce_dpp_readme

bool reduceDPPReadmeExample() {
    return reduce_dpp_readme::example();
}

int launch() {
    return fk::launch_impl();
}
