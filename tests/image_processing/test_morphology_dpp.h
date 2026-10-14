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

#include <tests/main.h>

#include <fused_kernel/fused_kernel.h>

#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/logical.h>
#include <fused_kernel/algorithms/basic_ops/math.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/crop.h>
#include <fused_kernel/algorithms/image_processing/morphology.h>
#include <fused_kernel/algorithms/image_processing/resize.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace fk {

namespace {

using Details = MorphologyDPPDetails<float, 16, 8, 7, 7>;
using MinReducer = Min<float, float, float, UnaryType>;
using MaxReducer = Max<float, float, float, UnaryType>;
using MinIOp = decltype(MinReducer::build());
using MaxIOp = decltype(MaxReducer::build());

static_assert(MorphologyDPP<ParArch::CPU, Details>::
              template validReducerCount<Tuple<MinIOp>>);
static_assert(!MorphologyDPP<ParArch::CPU, Details>::
              template validReducerCount<Tuple<>>);
static_assert(!MorphologyDPP<ParArch::CPU, Details>::
              template validReducerCount<
                  Tuple<MinIOp, MaxIOp, MinIOp, MaxIOp, MinIOp>>);
static_assert(Details::MAX_MASK_WIDTH == 7 && Details::MAX_MASK_HEIGHT == 7);
#ifdef FK_CPU_MULTITHREAD_ENABLED
static_assert(MorphologyDPP<ParArch::CPU_MULTITHREAD, Details>::PAR_ARCH ==
              ParArch::CPU_MULTITHREAD);
static_assert(MorphologyDPP<ParArch::CPU_MULTITHREAD, Details>::
              template validReducerCount<Tuple<MinIOp, MaxIOp, MinIOp, MaxIOp>>);
static_assert(!MorphologyDPP<ParArch::CPU_MULTITHREAD, Details>::
              template validReducerCount<Tuple<>>);
#endif

size_t pixelIndex(const int x, const int y, const int width) {
    return static_cast<size_t>(y * width + x);
}

float inputValue(const int x, const int y) {
    return static_cast<float>((x * 17 + y * 29) % 101 - 50) * 0.125f;
}

std::vector<float> makeInput(const Details& details) {
    std::vector<float> input(
        static_cast<size_t>(details.width * details.height));
    for (int y = 0; y < details.height; ++y)
        for (int x = 0; x < details.width; ++x)
            input[pixelIndex(x, y, details.width)] = inputValue(x, y);
    return input;
}

// A host image of details' size, read and written through RawPtr
RawPtr<ND::_2D, float> hostPtr(std::vector<float>& data,
                               const Details& details) {
    const auto width = static_cast<uint>(details.width);
    return RawPtr<ND::_2D, float>{
        data.data(), PtrDims<ND::_2D>(width,
                                      static_cast<uint>(details.height),
                                      width * static_cast<uint>(
                                                  sizeof(float)))};
}

template <typename InputPtr>
auto makeRead(const InputPtr& input) {
    return PerThreadRead<ND::_2D, float>::build(input)
        .then(Mul<float>::build(2.f))
        .then(Add<float>::build(1.f));
}

template <typename OutputPtr>
auto makeWrite(const OutputPtr& output) {
    return Mul<float>::build(0.5f)
        .then(PerThreadWrite<ND::_2D, float>::build(output));
}

std::vector<float> oracle(const std::vector<float>& original,
                          const int width, const int height,
                          const int maskW, const int maskH,
                          const int anchorX, const int anchorY,
                          const std::vector<bool>& erodePasses) {
    std::vector<float> current(original.size());
    for (size_t i = 0; i < original.size(); ++i)
        current[i] = original[i] * 2.f + 1.f;
    std::vector<float> next(original.size());
    for (const bool erode : erodePasses) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                bool first = true;
                float value = 0.f;
                for (int my = 0; my < maskH; ++my) {
                    const int sy = std::clamp(y + my - anchorY, 0, height - 1);
                    for (int mx = 0; mx < maskW; ++mx) {
                        const int sx = std::clamp(x + mx - anchorX, 0, width - 1);
                        const float sample = current[pixelIndex(sx, sy, width)];
                        if (first) {
                            value = sample;
                            first = false;
                        } else {
                            value = erode ? std::min(value, sample)
                                          : std::max(value, sample);
                        }
                    }
                }
                next[pixelIndex(x, y, width)] = value;
            }
        }
        current.swap(next);
    }
    for (float& value : current) value *= 0.5f;
    return current;
}

std::vector<float> oracle(const std::vector<float>& original,
                          const Details& details,
                          const std::vector<bool>& erodePasses) {
    return oracle(original, details.width, details.height,
                  details.maskW, details.maskH,
                  details.anchorX, details.anchorY, erodePasses);
}

bool matchesOracle(const char* backend, const char* name,
                   const std::vector<float>& got,
                   const std::vector<float>& expected) {
    for (size_t i = 0; i < got.size(); ++i) {
        if (std::fabs(got[i] - expected[i]) > 1e-6f) {
            std::printf("%s %s index=%zu got=%g expected=%g\n",
                        backend, name, i, got[i], expected[i]);
            return false;
        }
    }
    return true;
}

// Runs the case with MorphologyDPP<ParArch::CPU> and returns its output in
// result. With inPlace, the output is written over the input.
template <typename Reducers>
bool runCpuCase(const char* name, const Details& details,
                const Reducers& reducers,
                const std::vector<bool>& erodePasses, const bool inPlace,
                std::vector<float>& result) {
    const std::vector<float> original = makeInput(details);
    std::vector<float> input = original;
    std::vector<float> output(input.size(), -777.f);
    std::vector<float>& target = inPlace ? input : output;

    MorphologyDPP<ParArch::CPU, Details>::exec(
        details, makeRead(hostPtr(input, details)),
        makeWrite(hostPtr(target, details)), reducers);
    result = target;
    return matchesOracle("CPU", name, target,
                         oracle(original, details, erodePasses));
}

#if defined(__NVCC__) || defined(__HIPCC__) || \
    defined(FK_CPU_MULTITHREAD_ENABLED)
// Launches the case on stream through the entry point a user would call for
// these reducers
template <typename Reducers, typename ReadIOp, typename WriteIOp,
          ParArch PA>
bool executeCase(const Details& details, const ReadIOp& read,
                 const WriteIOp& write, const Reducers& reducers,
                 Stream_<PA>& stream) {
    if constexpr (std::is_same_v<Reducers, Tuple<MinIOp>>) {
        return executeErode(details, read, write, stream);
    } else if constexpr (std::is_same_v<Reducers, Tuple<MaxIOp>>) {
        return executeDilate(details, read, write, stream);
    } else if constexpr (std::is_same_v<Reducers,
                                        Tuple<MinIOp, MaxIOp>>) {
        return executeOpen(details, read, write, stream);
    } else if constexpr (std::is_same_v<Reducers,
                                        Tuple<MaxIOp, MinIOp>>) {
        return executeClose(details, read, write, stream);
    } else {
        return executeMorphology(details, read, write, reducers, stream);
    }
}
#endif

#ifdef FK_CPU_MULTITHREAD_ENABLED
using MTStream = Stream_<ParArch::CPU_MULTITHREAD>;

bool sameBytes(const char* name, const std::vector<float>& got,
               const std::vector<float>& cpuResult) {
    if (got.size() != cpuResult.size()) {
        std::printf("CPU_MULTITHREAD %s size differs from ParArch::CPU\n",
                    name);
        return false;
    }
    for (size_t i = 0; i < got.size(); ++i) {
        if (std::memcmp(&got[i], &cpuResult[i], sizeof(float)) != 0) {
            std::printf("CPU_MULTITHREAD %s index=%zu got=%g, ParArch::CPU "
                        "got=%g\n", name, i, got[i], cpuResult[i]);
            return false;
        }
    }
    return true;
}

// Runs the case on a CPU_MULTITHREAD stream, whose output must be
// bit-identical to the ParArch::CPU one
template <typename Reducers>
bool runMtCase(const char* name, const Details& details,
               const Reducers& reducers,
               const std::vector<bool>& erodePasses, const bool inPlace,
               const std::vector<float>& cpuResult) {
    const std::vector<float> original = makeInput(details);
    std::vector<float> input = original;
    std::vector<float> output(input.size(), -777.f);
    std::vector<float>& target = inPlace ? input : output;

    MTStream stream;
    const bool launched = executeCase(
        details, makeRead(hostPtr(input, details)),
        makeWrite(hostPtr(target, details)), reducers, stream);
    stream.sync();
    if (!launched) {
        std::printf("CPU_MULTITHREAD %s launcher rejected valid details\n",
                    name);
        return false;
    }
    return sameBytes(name, target, cpuResult) &&
           matchesOracle("CPU_MULTITHREAD", name, target,
                         oracle(original, details, erodePasses));
}

// Three executions on one stream, each one reading what the previous one
// wrote (the last one in place), enqueued without waiting: each one must
// start only after the last pass of the previous one has written its output
bool runMtChained() {
    const Details details{97, 61, 5, 3, 2, 1};
    const Details closeDetails{97, 61, 3, 5, 1, 2};
    const std::vector<float> original = makeInput(details);
    const auto minOp = MinReducer::build();
    const auto maxOp = MaxReducer::build();

    std::vector<float> cpuA = original;
    std::vector<float> cpuB(original.size(), -777.f);
    std::vector<float> cpuC(original.size(), -777.f);
    using CPUDPP = MorphologyDPP<ParArch::CPU, Details>;
    CPUDPP::exec(details, makeRead(hostPtr(cpuA, details)),
                 makeWrite(hostPtr(cpuB, details)), make_tuple(minOp));
    CPUDPP::exec(details, makeRead(hostPtr(cpuB, details)),
                 makeWrite(hostPtr(cpuC, details)), make_tuple(maxOp));
    CPUDPP::exec(closeDetails, makeRead(hostPtr(cpuC, details)),
                 makeWrite(hostPtr(cpuC, details)),
                 make_tuple(maxOp, minOp));

    std::vector<float> a = original;
    std::vector<float> b(original.size(), -777.f);
    std::vector<float> c(original.size(), -777.f);
    MTStream stream;
    bool launched = executeErode(details, makeRead(hostPtr(a, details)),
                                 makeWrite(hostPtr(b, details)), stream);
    launched = executeDilate(details, makeRead(hostPtr(b, details)),
                             makeWrite(hostPtr(c, details)), stream) &&
               launched;
    launched = executeClose(closeDetails, makeRead(hostPtr(c, details)),
                            makeWrite(hostPtr(c, details)), stream) &&
               launched;
    stream.sync();
    if (!launched) {
        std::printf("CPU_MULTITHREAD chained launcher rejected valid "
                    "details\n");
        return false;
    }
    const auto expectedB = oracle(original, details, {true});
    const auto expectedC = oracle(oracle(expectedB, details, {false}),
                                  closeDetails, {false, true});
    return sameBytes("chained-erode", b, cpuB) &&
           sameBytes("chained-dilate-close", c, cpuC) &&
           matchesOracle("CPU_MULTITHREAD", "chained-erode", b, expectedB) &&
           matchesOracle("CPU_MULTITHREAD", "chained-dilate-close", c,
                         expectedC);
}

// Invalid details are rejected by every entry point, enqueueing nothing
bool runMtInvalid() {
    const Details invalid{10, 10, 9, 3, 4, 1};
    std::vector<float> input(100, 1.f);
    std::vector<float> output(100, -777.f);
    const auto read = makeRead(hostPtr(input, invalid));
    const auto write = makeWrite(hostPtr(output, invalid));
    MTStream stream;
    bool launched = executeErode(invalid, read, write, stream);
    launched = executeDilate(invalid, read, write, stream) || launched;
    launched = executeOpen(invalid, read, write, stream) || launched;
    launched = executeClose(invalid, read, write, stream) || launched;
    launched = executeMorphology(
                   invalid, read, write,
                   make_tuple(MinReducer::build(), MaxReducer::build(),
                              MinReducer::build()), stream) || launched;
    stream.sync();
    const bool untouched = std::all_of(
        output.begin(), output.end(),
        [](const float value) { return value == -777.f; });
    if (launched || !untouched) {
        std::printf("CPU_MULTITHREAD invalid morphology details accepted\n");
        return false;
    }
    return true;
}

// Throws for one value, to test the errors of the work of an execution. Host
// and device, as the IOp folds that call it, but it only throws on the host.
struct ThrowOn13 {
private:
    using SelfType = ThrowOn13;
public:
    FK_STATIC_STRUCT(ThrowOn13, SelfType)
    using Parent = UnaryOperation<float, float, ThrowOn13>;
    DECLARE_UNARY_PARENT
    FK_HOST_DEVICE_FUSE OutputType exec(const InputType input) {
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
        if (input == 13.f) {
            throw std::runtime_error("ThrowOn13");
        }
#endif
        return input;
    }
};

bool syncThrowsOnce(MTStream& stream) {
    bool thrown = false;
    try {
        stream.sync();
    } catch (const std::runtime_error&) {
        thrown = true;
    }
    bool thrownAgain = false;
    try {
        stream.sync();
    } catch (...) {
        thrownAgain = true;
    }
    return thrown && !thrownAgain;
}

// An exception thrown in the load phase skips the passes (nothing is
// written), one thrown in the last pass ends the execution, both reach
// sync(), and the stream keeps working
bool runMtErrors() {
    const Details details{41, 23, 3, 3, 1, 1};
    MTStream stream;

    std::vector<float> input = makeInput(details);
    input[pixelIndex(20, 11, details.width)] = 13.f;
    std::vector<float> output(input.size(), -777.f);
    executeOpen(details,
                PerThreadRead<ND::_2D, float>::build(hostPtr(input, details))
                    .then(ThrowOn13::build()),
                makeWrite(hostPtr(output, details)), stream);
    const bool loadThrew = syncThrowsOnce(stream);
    const bool untouched = std::all_of(
        output.begin(), output.end(),
        [](const float value) { return value == -777.f; });

    std::vector<float> thirteens(input.size(), 13.f);
    executeErode(details,
                 PerThreadRead<ND::_2D, float>::build(
                     hostPtr(thirteens, details)),
                 ThrowOn13::build().then(PerThreadWrite<ND::_2D, float>::build(
                     hostPtr(output, details))),
                 stream);
    const bool writeThrew = syncThrowsOnce(stream);

    std::vector<float> cpuResult;
    const bool cpuOk = runCpuCase("after-errors", details,
                                  make_tuple(MaxReducer::build()), {false},
                                  false, cpuResult);
    const std::vector<float> original = makeInput(details);
    std::vector<float> source = original;
    std::fill(output.begin(), output.end(), -777.f);
    executeDilate(details, makeRead(hostPtr(source, details)),
                  makeWrite(hostPtr(output, details)), stream);
    stream.sync();
    if (!loadThrew || !untouched || !writeThrew) {
        std::printf("CPU_MULTITHREAD morphology errors: load threw=%d, "
                    "output untouched=%d, write threw=%d\n",
                    loadThrew, untouched, writeThrew);
        return false;
    }
    return cpuOk && sameBytes("after-errors", output, cpuResult);
}
#endif // FK_CPU_MULTITHREAD_ENABLED

#if defined(__NVCC__) || defined(__HIPCC__)
template <typename Reducers>
bool runGpuCase(const char* name, const Details& details,
                const Reducers& reducers,
                const std::vector<bool>& erodePasses) {
    const std::vector<float> hostInput = makeInput(details);

    Ptr2D<float> input(details.width, details.height);
    Ptr2D<float> output(details.width, details.height);
    for (int y = 0; y < details.height; ++y)
        for (int x = 0; x < details.width; ++x)
            input.at(Point{x, y, 0}) =
                hostInput[pixelIndex(x, y, details.width)];

    Stream stream;
    input.upload(stream);
    const bool launched = executeCase(details, makeRead(input),
                                      makeWrite(output), reducers, stream);
    if (!launched) {
        std::printf("GPU %s launcher rejected valid details\n", name);
        return false;
    }
    output.download(stream);
    stream.sync();

    const auto expected = oracle(hostInput, details, erodePasses);
    for (int y = 0; y < details.height; ++y) {
        for (int x = 0; x < details.width; ++x) {
            const size_t i = pixelIndex(x, y, details.width);
            const float got = output.at(Point{x, y, 0});
            if (std::fabs(got - expected[i]) > 1e-6f) {
                std::printf("GPU %s (%d,%d) got=%g expected=%g\n",
                            name, x, y, got, expected[i]);
                return false;
            }
        }
    }
    return true;
}
#endif

// Batched input: Crop of an array of rectangles of different sizes, each one
// resized to BATCH_W x BATCH_H, read through the same Mul and Add as the
// other cases. One execution processes every plane (a grid z plane each on
// the GPU, a loop on ParArch::CPU, a chunk per row of every plane on
// CPU_MULTITHREAD) and writes plane z of a Tensor, each plane with its own
// border.
constexpr int BATCH_SOURCE_W = 97;
constexpr int BATCH_SOURCE_H = 61;
constexpr int BATCH_W = 24;
constexpr int BATCH_H = 16;
constexpr size_t BATCH = 3;
constexpr std::array<Rect, BATCH> batchRects{
    Rect(3, 5, 20, 10), Rect(40, 20, 32, 16), Rect(60, 30, 37, 31)};
constexpr size_t BATCH_PLANE = static_cast<size_t>(BATCH_W) * BATCH_H;

std::vector<float> makeBatchSource() {
    std::vector<float> source(
        static_cast<size_t>(BATCH_SOURCE_W) * BATCH_SOURCE_H);
    for (int y = 0; y < BATCH_SOURCE_H; ++y)
        for (int x = 0; x < BATCH_SOURCE_W; ++x)
            source[pixelIndex(x, y, BATCH_SOURCE_W)] = inputValue(x, y);
    return source;
}

RawPtr<ND::_2D, float> batchSourcePtr(std::vector<float>& source) {
    return RawPtr<ND::_2D, float>{
        source.data(),
        PtrDims<ND::_2D>(static_cast<uint>(BATCH_SOURCE_W),
                         static_cast<uint>(BATCH_SOURCE_H),
                         static_cast<uint>(BATCH_SOURCE_W * sizeof(float)))};
}

template <typename SourcePtr>
auto makeBatchResize(const SourcePtr& source) {
    return PerThreadRead<ND::_2D, float>::build(source)
        .then(Crop<>::build(batchRects))
        .then(Resize<InterpolationType::INTER_LINEAR,
                     AspectRatio::IGNORE_AR>::build(Size(BATCH_W, BATCH_H)));
}

template <typename SourcePtr>
auto makeBatchRead(const SourcePtr& source) {
    return makeBatchResize(source)
        .then(Mul<float>::build(2.f))
        .then(Add<float>::build(1.f));
}

template <typename OutputPtr>
auto makeBatchWrite(const OutputPtr& output) {
    return Mul<float>::build(0.5f)
        .then(TensorWrite<float>::build(output));
}

// The expected planes: the resized planes computed by TransformDPP on the
// CPU, then the oracle of each one
std::vector<float> batchOracle(const Details& details,
                               const std::vector<bool>& erodePasses) {
    std::vector<float> source = makeBatchSource();
    Tensor<float> resized(BATCH_W, BATCH_H, BATCH, 1, MemType::Host);
    Stream_<ParArch::CPU> cpuStream;
    executeOperations<TransformDPP<ParArch::CPU>>(
        cpuStream, makeBatchResize(batchSourcePtr(source)),
        TensorWrite<float>::build(resized.ptr()));
    std::vector<float> expected;
    for (size_t z = 0; z < BATCH; ++z) {
        std::vector<float> plane(BATCH_PLANE);
        for (int y = 0; y < BATCH_H; ++y)
            for (int x = 0; x < BATCH_W; ++x)
                plane[pixelIndex(x, y, BATCH_W)] =
                    resized.at(x, y, static_cast<int>(z));
        const auto planeExpected =
            oracle(plane, BATCH_W, BATCH_H, details.maskW, details.maskH,
                   details.anchorX, details.anchorY, erodePasses);
        expected.insert(expected.end(), planeExpected.begin(),
                        planeExpected.end());
    }
    return expected;
}

std::vector<float> tensorValues(const Tensor<float>& tensor) {
    std::vector<float> values;
    for (size_t z = 0; z < BATCH; ++z)
        for (int y = 0; y < BATCH_H; ++y)
            for (int x = 0; x < BATCH_W; ++x)
                values.push_back(tensor.at(x, y, static_cast<int>(z)));
    return values;
}

// The resize multiplies and adds, so the backends may differ in the last
// bits (fused multiply-add contraction): compare within 1e-5, relative
bool batchMatches(const char* backend, const char* name,
                  const std::vector<float>& got,
                  const std::vector<float>& expected) {
    for (size_t i = 0; i < got.size(); ++i) {
        const float tolerance =
            1e-5f * std::max(1.f, std::fabs(expected[i]));
        if (!(std::fabs(got[i] - expected[i]) <= tolerance)) {
            std::printf("%s batched %s plane=%zu index=%zu got=%g "
                        "expected=%g\n", backend, name, i / BATCH_PLANE,
                        i % BATCH_PLANE, got[i], expected[i]);
            return false;
        }
    }
    return true;
}

template <typename Reducers>
bool runBatchedCase(const char* name, const Details& details,
                    const Reducers& reducers,
                    const std::vector<bool>& erodePasses) {
    const std::vector<float> expected = batchOracle(details, erodePasses);
    std::vector<float> source = makeBatchSource();
    const auto read = makeBatchRead(batchSourcePtr(source));
    if (read.getActiveThreads().z != BATCH) {
        std::printf("batched %s: the read has %u planes, not %zu\n", name,
                    read.getActiveThreads().z, BATCH);
        return false;
    }

    Tensor<float> cpuOut(BATCH_W, BATCH_H, BATCH, 1, MemType::Host);
    MorphologyDPP<ParArch::CPU, Details>::exec(
        details, read, makeBatchWrite(cpuOut.ptr()), reducers);
    const std::vector<float> cpuResult = tensorValues(cpuOut);
    bool ok = batchMatches("CPU", name, cpuResult, expected);

#ifdef FK_CPU_MULTITHREAD_ENABLED
    Tensor<float> mtOut(BATCH_W, BATCH_H, BATCH, 1, MemType::Host);
    MTStream mtStream;
    const bool mtLaunched = executeCase(
        details, read, makeBatchWrite(mtOut.ptr()), reducers, mtStream);
    mtStream.sync();
    if (!mtLaunched) {
        std::printf("CPU_MULTITHREAD batched %s launcher rejected valid "
                    "details\n", name);
        ok = false;
    } else {
        const std::vector<float> mtResult = tensorValues(mtOut);
        ok = batchMatches("CPU_MULTITHREAD", name, mtResult, cpuResult) &&
             batchMatches("CPU_MULTITHREAD", name, mtResult, expected) && ok;
    }
#endif

#if defined(__NVCC__) || defined(__HIPCC__)
    Ptr2D<float> gpuSource(BATCH_SOURCE_W, BATCH_SOURCE_H);
    for (int y = 0; y < BATCH_SOURCE_H; ++y)
        for (int x = 0; x < BATCH_SOURCE_W; ++x)
            gpuSource.at(Point{x, y, 0}) =
                source[pixelIndex(x, y, BATCH_SOURCE_W)];
    Tensor<float> gpuOut(BATCH_W, BATCH_H, BATCH);
    Stream stream;
    gpuSource.upload(stream);
    const bool gpuLaunched = executeCase(
        details, makeBatchRead(gpuSource.ptr()),
        makeBatchWrite(gpuOut.ptr()), reducers, stream);
    gpuOut.download(stream);
    stream.sync();
    if (!gpuLaunched) {
        std::printf("GPU batched %s launcher rejected valid details\n",
                    name);
        ok = false;
    } else {
        ok = batchMatches("GPU", name, tensorValues(gpuOut), expected) && ok;
    }
#endif
    return ok;
}

// Runs the case on ParArch::CPU, then on CPU_MULTITHREAD (bit-identical to
// CPU) if enabled, and on the GPU if compiled for it. With inPlace the
// output is written over the input, which the GPU version does not support.
template <typename Reducers>
bool runCase(const char* name, const Details& details,
             const Reducers& reducers,
             const std::vector<bool>& erodePasses,
             const bool inPlace = false) {
    std::vector<float> cpuResult;
    bool ok = runCpuCase(name, details, reducers, erodePasses, inPlace,
                         cpuResult);
#ifdef FK_CPU_MULTITHREAD_ENABLED
    ok = runMtCase(name, details, reducers, erodePasses, inPlace,
                   cpuResult) && ok;
#endif
#if defined(__NVCC__) || defined(__HIPCC__)
    if (!inPlace)
        ok = runGpuCase(name, details, reducers, erodePasses) && ok;
#endif
    return ok;
}

// Box blur (executeBoxBlur): PASSES passes of the mean of the window, each
// replicating the border of the image it reads, through the same read
// (2 x + 1) and write (x / 2) as the other cases; sums in double
std::vector<float> boxOracle(const std::vector<float>& original,
                             const int width, const int height,
                             const int maskW, const int maskH,
                             const int anchorX, const int anchorY,
                             const int passes) {
    std::vector<double> current(original.size());
    for (size_t i = 0; i < original.size(); ++i)
        current[i] = original[i] * 2. + 1.;
    std::vector<double> next(original.size());
    for (int pass = 0; pass < passes; ++pass) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                double sum = 0.;
                for (int my = 0; my < maskH; ++my) {
                    const int sy = std::clamp(y + my - anchorY, 0, height - 1);
                    for (int mx = 0; mx < maskW; ++mx) {
                        const int sx = std::clamp(x + mx - anchorX, 0, width - 1);
                        sum += current[pixelIndex(sx, sy, width)];
                    }
                }
                next[pixelIndex(x, y, width)] =
                    sum / (static_cast<double>(maskW) * maskH);
            }
        }
        current.swap(next);
    }
    std::vector<float> result(current.size());
    for (size_t i = 0; i < current.size(); ++i)
        result[i] = static_cast<float>(current[i] * 0.5);
    return result;
}

// The box sums in float, in another order than the oracle: compare within
// 1e-5, relative
bool boxMatches(const char* backend, const char* name,
                const std::vector<float>& got,
                const std::vector<float>& expected) {
    for (size_t i = 0; i < got.size(); ++i) {
        const float tolerance =
            1e-5f * std::max(1.f, std::fabs(expected[i]));
        if (!(std::fabs(got[i] - expected[i]) <= tolerance)) {
            std::printf("%s box %s index=%zu got=%g expected=%g\n",
                        backend, name, i, got[i], expected[i]);
            return false;
        }
    }
    return true;
}

// The box blur on ParArch::CPU (MorphologyDPP with the reducers and the write
// of executeBoxBlur), on CPU_MULTITHREAD (executeBoxBlur, bit-identical to
// CPU) if enabled, and on the GPU (executeBoxBlur) if compiled for it
template <int PASSES>
bool runBoxCase(const char* name, const Details& details) {
    const std::vector<float> original = makeInput(details);
    const auto expected =
        boxOracle(original, details.width, details.height, details.maskW,
                  details.maskH, details.anchorX, details.anchorY, PASSES);
    std::vector<float> input = original;
    std::vector<float> cpuOut(input.size(), -777.f);
    MorphologyDPP<ParArch::CPU, Details>::exec(
        details, makeRead(hostPtr(input, details)),
        makeWrite(hostPtr(cpuOut, details)),
        morphology_detail::boxBlurReducers<float, PASSES>(details));
    bool ok = boxMatches("CPU", name, cpuOut, expected);

#ifdef FK_CPU_MULTITHREAD_ENABLED
    std::vector<float> mtOut(input.size(), -777.f);
    MTStream mtStream;
    const bool mtLaunched = executeBoxBlur<PASSES>(
        details, makeRead(hostPtr(input, details)),
        makeWrite(hostPtr(mtOut, details)), mtStream);
    mtStream.sync();
    if (!mtLaunched) {
        std::printf("CPU_MULTITHREAD box %s launcher rejected valid "
                    "details\n", name);
        ok = false;
    } else {
        ok = sameBytes(name, mtOut, cpuOut) && ok;
    }
#endif

#if defined(__NVCC__) || defined(__HIPCC__)
    Ptr2D<float> gpuIn(details.width, details.height);
    Ptr2D<float> gpuOut(details.width, details.height);
    for (int y = 0; y < details.height; ++y)
        for (int x = 0; x < details.width; ++x)
            gpuIn.at(Point{x, y, 0}) = original[pixelIndex(x, y, details.width)];
    Stream stream;
    gpuIn.upload(stream);
    const bool gpuLaunched = executeBoxBlur<PASSES>(
        details, makeRead(gpuIn), makeWrite(gpuOut), stream);
    gpuOut.download(stream);
    stream.sync();
    if (!gpuLaunched) {
        std::printf("GPU box %s launcher rejected valid details\n", name);
        ok = false;
    } else {
        std::vector<float> got(original.size());
        for (int y = 0; y < details.height; ++y)
            for (int x = 0; x < details.width; ++x)
                got[pixelIndex(x, y, details.width)] = gpuOut.at(Point{x, y, 0});
        ok = boxMatches("GPU", name, got, expected) && ok;
    }
#endif
    return ok;
}

// The box blur of the batched input (Crop of rectangles of different sizes,
// each resized to BATCH_W x BATCH_H), every plane with its own border
template <int PASSES>
bool runBatchedBoxCase(const char* name, const Details& details) {
    std::vector<float> source = makeBatchSource();
    Tensor<float> resized(BATCH_W, BATCH_H, BATCH, 1, MemType::Host);
    Stream_<ParArch::CPU> cpuStream;
    executeOperations<TransformDPP<ParArch::CPU>>(
        cpuStream, makeBatchResize(batchSourcePtr(source)),
        TensorWrite<float>::build(resized.ptr()));
    std::vector<float> expected;
    for (size_t z = 0; z < BATCH; ++z) {
        std::vector<float> plane(BATCH_PLANE);
        for (int y = 0; y < BATCH_H; ++y)
            for (int x = 0; x < BATCH_W; ++x)
                plane[pixelIndex(x, y, BATCH_W)] =
                    resized.at(x, y, static_cast<int>(z));
        const auto planeExpected =
            boxOracle(plane, BATCH_W, BATCH_H, details.maskW, details.maskH,
                      details.anchorX, details.anchorY, PASSES);
        expected.insert(expected.end(), planeExpected.begin(),
                        planeExpected.end());
    }

    const auto read = makeBatchRead(batchSourcePtr(source));
    Tensor<float> cpuOut(BATCH_W, BATCH_H, BATCH, 1, MemType::Host);
    MorphologyDPP<ParArch::CPU, Details>::exec(
        details, read, makeBatchWrite(cpuOut.ptr()),
        morphology_detail::boxBlurReducers<float, PASSES>(details));
    const std::vector<float> cpuResult = tensorValues(cpuOut);
    bool ok = batchMatches("CPU", name, cpuResult, expected);

#ifdef FK_CPU_MULTITHREAD_ENABLED
    Tensor<float> mtOut(BATCH_W, BATCH_H, BATCH, 1, MemType::Host);
    MTStream mtStream;
    const bool mtLaunched = executeBoxBlur<PASSES>(
        details, read, makeBatchWrite(mtOut.ptr()), mtStream);
    mtStream.sync();
    if (!mtLaunched) {
        std::printf("CPU_MULTITHREAD batched box %s launcher rejected valid "
                    "details\n", name);
        ok = false;
    } else {
        ok = sameBytes(name, tensorValues(mtOut), cpuResult) && ok;
    }
#endif

#if defined(__NVCC__) || defined(__HIPCC__)
    Ptr2D<float> gpuSource(BATCH_SOURCE_W, BATCH_SOURCE_H);
    for (int y = 0; y < BATCH_SOURCE_H; ++y)
        for (int x = 0; x < BATCH_SOURCE_W; ++x)
            gpuSource.at(Point{x, y, 0}) =
                source[pixelIndex(x, y, BATCH_SOURCE_W)];
    Tensor<float> gpuOut(BATCH_W, BATCH_H, BATCH);
    Stream stream;
    gpuSource.upload(stream);
    const bool gpuLaunched = executeBoxBlur<PASSES>(
        details, makeBatchRead(gpuSource.ptr()),
        makeBatchWrite(gpuOut.ptr()), stream);
    gpuOut.download(stream);
    stream.sync();
    if (!gpuLaunched) {
        std::printf("GPU batched box %s launcher rejected valid details\n",
                    name);
        ok = false;
    } else {
        ok = batchMatches("GPU", name, tensorValues(gpuOut), expected) && ok;
    }
#endif
    return ok;
}

#if defined(__NVCC__) || defined(__HIPCC__) || \
    defined(FK_CPU_MULTITHREAD_ENABLED)
// executeBoxBlur rejects invalid details without launching or enqueueing
bool runBoxInvalid() {
    const Details invalid{10, 10, 9, 3, 4, 1};
    bool ok = true;
#ifdef FK_CPU_MULTITHREAD_ENABLED
    std::vector<float> input(100, 1.f);
    std::vector<float> output(100, -777.f);
    MTStream mtStream;
    bool launched = executeBoxBlur(invalid, makeRead(hostPtr(input, invalid)),
                                   makeWrite(hostPtr(output, invalid)),
                                   mtStream);
    launched = executeBoxBlur<2>(invalid, makeRead(hostPtr(input, invalid)),
                                 makeWrite(hostPtr(output, invalid)),
                                 mtStream) || launched;
    mtStream.sync();
    const bool untouched = std::all_of(
        output.begin(), output.end(),
        [](const float value) { return value == -777.f; });
    if (launched || !untouched) {
        std::printf("CPU_MULTITHREAD invalid box details accepted\n");
        ok = false;
    }
#endif
#if defined(__NVCC__) || defined(__HIPCC__)
    Ptr2D<float> gpuIn(10, 10);
    Ptr2D<float> gpuOut(10, 10);
    Stream stream;
    if (executeBoxBlur(invalid, makeRead(gpuIn), makeWrite(gpuOut), stream)) {
        std::printf("GPU invalid box details accepted\n");
        ok = false;
    }
    stream.sync();
#endif
    return ok;
}
#endif

// The box blur of double values keeps double precision, scale included: a
// 3x3 box in two passes against a double oracle within 1e-12, relative (a
// float scale is off by about 1e-8)
bool runBoxDouble() {
    using DoubleDetails = MorphologyDPPDetails<double, 16, 8, 7, 7>;
    const DoubleDetails details{23, 13, 3, 3, 1, 1};
    std::vector<double> input(static_cast<size_t>(details.width * details.height));
    for (int y = 0; y < details.height; ++y)
        for (int x = 0; x < details.width; ++x)
            input[pixelIndex(x, y, details.width)] =
                1. + 0.001 * static_cast<double>((x * 17 + y * 29) % 101);
    std::vector<double> expected(input);
    std::vector<double> next(input.size());
    for (int pass = 0; pass < 2; ++pass) {
        for (int y = 0; y < details.height; ++y) {
            for (int x = 0; x < details.width; ++x) {
                double sum = 0.;
                for (int my = -1; my <= 1; ++my)
                    for (int mx = -1; mx <= 1; ++mx)
                        sum += expected[pixelIndex(
                            std::clamp(x + mx, 0, details.width - 1),
                            std::clamp(y + my, 0, details.height - 1),
                            details.width)];
                next[pixelIndex(x, y, details.width)] = sum / 9.;
            }
        }
        expected.swap(next);
    }
    const auto closeEnough = [&](const char* backend,
                                 const std::vector<double>& got) {
        for (size_t i = 0; i < got.size(); ++i) {
            if (!(std::fabs(got[i] - expected[i]) <=
                  1e-12 * std::fabs(expected[i]))) {
                std::printf("%s box double index=%zu got=%.17g "
                            "expected=%.17g\n", backend, i, got[i],
                            expected[i]);
                return false;
            }
        }
        return true;
    };
    const auto width = static_cast<uint>(details.width);
    const auto height = static_cast<uint>(details.height);
    std::vector<double> cpuOut(input.size(), -777.);
    const RawPtr<ND::_2D, double> inPtr{
        input.data(), PtrDims<ND::_2D>(width, height, width * sizeof(double))};
    const RawPtr<ND::_2D, double> outPtr{
        cpuOut.data(),
        PtrDims<ND::_2D>(width, height, width * sizeof(double))};
    MorphologyDPP<ParArch::CPU, DoubleDetails>::exec(
        details, PerThreadRead<ND::_2D, double>::build(inPtr),
        PerThreadWrite<ND::_2D, double>::build(outPtr),
        morphology_detail::boxBlurReducers<double, 2>(details));
    bool ok = closeEnough("CPU", cpuOut);
#ifdef FK_CPU_MULTITHREAD_ENABLED
    std::vector<double> mtOut(input.size(), -777.);
    const RawPtr<ND::_2D, double> mtPtr{
        mtOut.data(), PtrDims<ND::_2D>(width, height, width * sizeof(double))};
    MTStream mtStream;
    const bool mtLaunched = executeBoxBlur<2>(
        details, PerThreadRead<ND::_2D, double>::build(inPtr),
        PerThreadWrite<ND::_2D, double>::build(mtPtr), mtStream);
    mtStream.sync();
    if (!mtLaunched) {
        std::printf("CPU_MULTITHREAD box double launcher rejected valid "
                    "details\n");
        ok = false;
    } else {
        ok = closeEnough("CPU_MULTITHREAD", mtOut) && ok;
    }
#endif
#if defined(__NVCC__) || defined(__HIPCC__)
    Ptr2D<double> gpuIn(details.width, details.height);
    Ptr2D<double> gpuOut(details.width, details.height);
    for (int y = 0; y < details.height; ++y)
        for (int x = 0; x < details.width; ++x)
            gpuIn.at(Point{x, y, 0}) = input[pixelIndex(x, y, details.width)];
    Stream stream;
    gpuIn.upload(stream);
    const bool launched = executeBoxBlur<2>(
        details, PerThreadRead<ND::_2D, double>::build(gpuIn.ptr()),
        PerThreadWrite<ND::_2D, double>::build(gpuOut.ptr()), stream);
    gpuOut.download(stream);
    stream.sync();
    if (!launched) {
        std::printf("GPU box double launcher rejected valid details\n");
        ok = false;
    } else {
        std::vector<double> got(input.size());
        for (int y = 0; y < details.height; ++y)
            for (int x = 0; x < details.width; ++x)
                got[pixelIndex(x, y, details.width)] = gpuOut.at(Point{x, y, 0});
        ok = closeEnough("GPU", got) && ok;
    }
#endif
    return ok;
}

static_assert(std::is_same_v<
    decltype(morphology_detail::boxBlurReducers<float, 2>(Details{})),
    Tuple<MorphologyPass<decltype(Add<float, float, float, UnaryType>::build()),
                         decltype(Mul<float>::build(1.f))>,
          MorphologyPass<decltype(Add<float, float, float, UnaryType>::build()),
                         decltype(Mul<float>::build(1.f))>>>);

// Runs reducers (MorphologyPasses and plain reducers) with MorphologyDPP on
// ParArch::CPU, and through executeMorphology on CPU_MULTITHREAD
// (bit-identical to CPU) and the GPU, against expected
template <typename Reducers>
bool runPassCase(const char* name, const Details& details,
                 const Reducers& reducers, const std::vector<float>& expected) {
    const std::vector<float> original = makeInput(details);
    std::vector<float> input = original;
    std::vector<float> cpuOut(input.size(), -777.f);
    MorphologyDPP<ParArch::CPU, Details>::exec(
        details, makeRead(hostPtr(input, details)),
        makeWrite(hostPtr(cpuOut, details)), reducers);
    bool ok = matchesOracle("CPU", name, cpuOut, expected);
#ifdef FK_CPU_MULTITHREAD_ENABLED
    std::vector<float> mtOut(input.size(), -777.f);
    MTStream mtStream;
    executeMorphology(details, makeRead(hostPtr(input, details)),
                      makeWrite(hostPtr(mtOut, details)), reducers, mtStream);
    mtStream.sync();
    ok = sameBytes(name, mtOut, cpuOut) && ok;
#endif
#if defined(__NVCC__) || defined(__HIPCC__)
    Ptr2D<float> gpuIn(details.width, details.height);
    Ptr2D<float> gpuOut(details.width, details.height);
    for (int y = 0; y < details.height; ++y)
        for (int x = 0; x < details.width; ++x)
            gpuIn.at(Point{x, y, 0}) = original[pixelIndex(x, y, details.width)];
    Stream stream;
    gpuIn.upload(stream);
    executeMorphology(details, makeRead(gpuIn), makeWrite(gpuOut), reducers,
                      stream);
    gpuOut.download(stream);
    stream.sync();
    std::vector<float> got(original.size());
    for (int y = 0; y < details.height; ++y)
        for (int x = 0; x < details.width; ++x)
            got[pixelIndex(x, y, details.width)] = gpuOut.at(Point{x, y, 0});
    ok = matchesOracle("GPU", name, got, expected) && ok;
#endif
    return ok;
}

// The minimum (erode) or maximum of the 3x3 window, replicating the border
std::vector<float> window3x3(const std::vector<float>& source, const int width,
                             const int height, const bool erode) {
    std::vector<float> out(source.size());
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float value = source[pixelIndex(x, y, width)];
            for (int my = -1; my <= 1; ++my)
                for (int mx = -1; mx <= 1; ++mx) {
                    const float sample = source[pixelIndex(
                        std::clamp(x + mx, 0, width - 1),
                        std::clamp(y + my, 0, height - 1), width)];
                    value = erode ? std::min(value, sample)
                                  : std::max(value, sample);
                }
            out[pixelIndex(x, y, width)] = value;
        }
    }
    return out;
}

// MorphologyPasses next to plain reducers: an opening whose first pass has a
// finalizer, Binary (x2: the maximum commutes with it, so the result is
// twice the opening) or Unary (Abs: |erosion|, then the dilation)
bool runFinalizerCases() {
    const Details details{31, 18, 3, 3, 1, 1};
    const std::vector<float> original = makeInput(details);
    auto doubled = oracle(original, details, {true, false});
    for (float& v : doubled) v *= 2.f;
    bool ok = runPassCase(
        "open-first-pass-x2", details,
        make_tuple(makeMorphologyPass(MinReducer::build(), Mul<float>::build(2.f)),
                   MaxReducer::build()),
        doubled);

    std::vector<float> read(original.size());
    for (size_t i = 0; i < original.size(); ++i) read[i] = original[i] * 2.f + 1.f;
    auto eroded = window3x3(read, details.width, details.height, true);
    for (float& v : eroded) v = std::fabs(v);
    auto absOpened = window3x3(eroded, details.width, details.height, false);
    for (float& v : absOpened) v *= 0.5f;
    ok = runPassCase(
        "open-first-pass-abs", details,
        make_tuple(makeMorphologyPass(MinReducer::build(), Abs<float>::build()),
                   MaxReducer::build()),
        absOpened) && ok;
    return ok;
}

// Every pass of the box blur is a mean, the staged ones included, so values
// that a mean keeps finite stay finite however many passes run: a constant
// image near max() / 100, 7x7 in four passes (sums of 49 such values would
// overflow), must come out as the constant
bool runBoxNoOverflow() {
    const Details details{29, 17, 7, 7, 3, 3};
    const float big = std::numeric_limits<float>::max() / 100.f;
    std::vector<float> input(static_cast<size_t>(details.width * details.height), big);
    const auto read = PerThreadRead<ND::_2D, float>::build(hostPtr(input, details));
    const auto exact = [&](const char* backend, const std::vector<float>& got) {
        for (size_t i = 0; i < got.size(); ++i) {
            if (!(std::fabs(got[i] - big) <= 1e-5f * big)) {
                std::printf("%s box overflow index=%zu got=%g expected=%g\n",
                            backend, i, got[i], big);
                return false;
            }
        }
        return true;
    };
    std::vector<float> cpuOut(input.size(), -777.f);
    MorphologyDPP<ParArch::CPU, Details>::exec(
        details, read,
        PerThreadWrite<ND::_2D, float>::build(hostPtr(cpuOut, details)),
        morphology_detail::boxBlurReducers<float, 4>(details));
    bool ok = exact("CPU", cpuOut);
#ifdef FK_CPU_MULTITHREAD_ENABLED
    std::vector<float> mtOut(input.size(), -777.f);
    MTStream mtStream;
    executeBoxBlur<4>(details, read,
                      PerThreadWrite<ND::_2D, float>::build(hostPtr(mtOut, details)),
                      mtStream);
    mtStream.sync();
    ok = exact("CPU_MULTITHREAD", mtOut) && ok;
#endif
#if defined(__NVCC__) || defined(__HIPCC__)
    Ptr2D<float> gpuIn(details.width, details.height);
    Ptr2D<float> gpuOut(details.width, details.height);
    for (int y = 0; y < details.height; ++y)
        for (int x = 0; x < details.width; ++x) gpuIn.at(Point{x, y, 0}) = big;
    Stream stream;
    gpuIn.upload(stream);
    executeBoxBlur<4>(details, PerThreadRead<ND::_2D, float>::build(gpuIn.ptr()),
                      PerThreadWrite<ND::_2D, float>::build(gpuOut.ptr()), stream);
    gpuOut.download(stream);
    stream.sync();
    std::vector<float> got(input.size());
    for (int y = 0; y < details.height; ++y)
        for (int x = 0; x < details.width; ++x)
            got[pixelIndex(x, y, details.width)] = gpuOut.at(Point{x, y, 0});
    ok = exact("GPU", got) && ok;
#endif
    return ok;
}

} // namespace

int launch_impl() {
    using namespace fk;
    const auto minOp = MinReducer::build();
    const auto maxOp = MaxReducer::build();
    bool ok = true;
    ok = runCase("erode-3x3", Details{37, 19, 3, 3, 1, 1},
                 make_tuple(minOp), {true}) && ok;
    ok = runCase("dilate-5x5", Details{35, 21, 5, 5, 2, 2},
                 make_tuple(maxOp), {false}) && ok;
    ok = runCase("erode-7x3", Details{39, 17, 7, 3, 3, 1},
                 make_tuple(minOp), {true}) && ok;
    ok = runCase("open", Details{31, 18, 3, 3, 1, 1},
                 make_tuple(minOp, maxOp), {true, false}) && ok;
    ok = runCase("close", Details{33, 15, 5, 3, 2, 1},
                 make_tuple(maxOp, minOp), {false, true}) && ok;
    ok = runCase("three-pass", Details{29, 17, 3, 5, 1, 2},
                 make_tuple(minOp, maxOp, minOp),
                 {true, false, true}) && ok;
    ok = runCase("four-pass", Details{27, 13, 3, 3, 1, 1},
                 make_tuple(maxOp, minOp, minOp, maxOp),
                 {false, true, true, false}) && ok;
    ok = runCase("four-pass-large", Details{301, 203, 7, 5, 3, 2},
                 make_tuple(maxOp, minOp, maxOp, minOp),
                 {false, true, false, true}) && ok;
    ok = runCase("erode-3x3-in-place", Details{37, 19, 3, 3, 1, 1},
                 make_tuple(minOp), {true}, true) && ok;
    ok = runCase("close-in-place", Details{33, 15, 5, 3, 2, 1},
                 make_tuple(maxOp, minOp), {false, true}, true) && ok;
    ok = runCase("four-pass-large-in-place", Details{301, 203, 7, 5, 3, 2},
                 make_tuple(maxOp, minOp, maxOp, minOp),
                 {false, true, false, true}, true) && ok;
    ok = runBatchedCase("erode-3x3", Details{BATCH_W, BATCH_H, 3, 3, 1, 1},
                        make_tuple(minOp), {true}) && ok;
    ok = runBatchedCase("open-5x3", Details{BATCH_W, BATCH_H, 5, 3, 1, 2},
                        make_tuple(minOp, maxOp), {true, false}) && ok;
    ok = runBatchedCase("four-pass-7x5",
                        Details{BATCH_W, BATCH_H, 7, 5, 3, 2},
                        make_tuple(maxOp, minOp, maxOp, minOp),
                        {false, true, false, true}) && ok;
    ok = runBoxCase<1>("box-3x3", Details{37, 19, 3, 3, 1, 1}) && ok;
    ok = runBoxCase<1>("box-5x5", Details{35, 21, 5, 5, 2, 2}) && ok;
    ok = runBoxCase<1>("box-7x3-anchor-off-center", Details{39, 17, 7, 3, 1, 2}) && ok;
    ok = runBoxCase<2>("box-5x5-two-pass", Details{33, 15, 5, 5, 2, 2}) && ok;
    ok = runBoxCase<4>("box-3x5-four-pass-large", Details{301, 203, 3, 5, 1, 2}) && ok;
    ok = runBatchedBoxCase<1>("box-5x5", Details{BATCH_W, BATCH_H, 5, 5, 2, 2}) && ok;
    ok = runBatchedBoxCase<2>("box-7x7-two-pass", Details{BATCH_W, BATCH_H, 7, 7, 3, 3}) && ok;
    ok = runBoxDouble() && ok;
    ok = runBoxNoOverflow() && ok;
    ok = runFinalizerCases() && ok;
#if defined(__NVCC__) || defined(__HIPCC__) || \
    defined(FK_CPU_MULTITHREAD_ENABLED)
    ok = runBoxInvalid() && ok;
#endif
#ifdef FK_CPU_MULTITHREAD_ENABLED
    ok = runMtChained() && ok;
    ok = runMtInvalid() && ok;
    ok = runMtErrors() && ok;
#endif

    if (Details::valid(Details{10, 10, 9, 3, 4, 1}) ||
        Details::valid(Details{10, 10, 3, 3, 3, 1}) ||
        Details::valid(Details{0, 10, 3, 3, 1, 1})) {
        std::printf("invalid morphology details accepted\n");
        ok = false;
    }
    if (ok) std::printf("MorphologyDPP 1-4 pass contracts and box blur: PASS\n");
    return ok ? 0 : -1;
}

} // namespace fk

int launch() {
    return fk::launch_impl();
}
