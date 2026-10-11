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

// ParArch::CPU_MULTITHREAD is a host backend, so this test only has a _cpp target
#define __ONLY_CPU__

#include <tests/main.h>

#include <fused_kernel/fused_kernel.h>

#ifdef FK_CPU_MULTITHREAD_ENABLED
#include <fused_kernel/core/execution_model/cpu_multithread.h>
#include <fused_kernel/core/data/ptr_utils.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/cast.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/add_border.h>
#include <fused_kernel/algorithms/image_processing/color_conversion.h>
#include <fused_kernel/algorithms/image_processing/crop.h>
#include <fused_kernel/algorithms/image_processing/image.h>
#include <fused_kernel/algorithms/image_processing/resize.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <future>
#include <memory>
#include <iostream>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

namespace fk {

// The CPU_MULTITHREAD DPPs run the per row code of the CPU ones, but the results of float arithmetic may differ in the
// last bits: the compiler can contract a * b + c into a fused multiply-add differently where that code is inlined
inline bool closeFloats(const void* a, const void* b, const size_t bytes) {
    const float* x = static_cast<const float*>(a);
    const float* y = static_cast<const float*>(b);
    for (size_t i = 0; i < bytes / sizeof(float); ++i) {
        const float scale = std::max(1.f, std::fabs(x[i]));
        if (!(std::fabs(x[i] - y[i]) <= 1e-5f * scale)) {
            return false;
        }
    }
    return true;
}

using MTStream = Stream_<ParArch::CPU_MULTITHREAD>;
using MTDPP = TransformDPP<ParArch::CPU_MULTITHREAD>;
using CPUDPP = TransformDPP<ParArch::CPU>;

// The Figure 1 chain of the BVF paper (NV12 crops resized, padded and normalized) gives the same result on the single
// and the multithreaded CPU backends, since both run the same per thread code (up to fused multiply-add rounding)
bool testSameResultAsCPU() {
    constexpr int W = 640, H = 360;
    constexpr size_t N = 16;
    Image<PixelFormat::NV12> frame(W, H, MemType::Host);
    auto data = frame.getData();
    std::mt19937 gen(1);
    std::uniform_int_distribution<int> byte(0, 255);
    for (int y = 0; y < H * 3 / 2; ++y) {
        for (int x = 0; x < W; ++x) {
            data.at(x, y) = static_cast<uchar>(byte(gen));
        }
    }
    std::array<Rect, N> crops{};
    std::uniform_int_distribution<int> distX(0, W - 20), distY(0, H - 10);
    for (auto& crop : crops) {
        const uint x = static_cast<uint>(distX(gen));
        const uint y = static_cast<uint>(distY(gen));
        crop = Rect(x, y, 20, 10);
    }
    Tensor<float3> single(64, 64, N, 1, MemType::Host);
    Tensor<float3> multi(64, 64, N, 1, MemType::Host);
    const auto run = [&](auto dpp, auto& stream, Tensor<float3>& out) {
        executeOperations<decltype(dpp)>(stream,
            ReadYUV<PixelFormat::NV12>::build(frame),
            ConvertYUVToRGB<ColorDepth::p8bit, ColorRange::Full, ColorPrimitives::bt2020>::build(),
            Crop<>::build(crops),
            Resize<InterpolationType::INTER_LINEAR, AspectRatio::IGNORE_AR>::build(Size(64, 32)),
            AddBorder<>::build(16, 16, 0, 0, float3{0.f, 0.f, 0.f}),
            Mul<float3>::build(float3{1.f / 255.f, 1.f / 255.f, 1.f / 255.f}),
            Sub<float3>::build(float3{0.485f, 0.456f, 0.406f}),
            Div<float3>::build(float3{0.229f, 0.224f, 0.225f}),
            TensorWrite<float3>::build(out.ptr()));
        stream.sync();
    };
    Stream_<ParArch::CPU> cpuStream;
    MTStream mtStream;
    run(CPUDPP{}, cpuStream, single);
    run(MTDPP{}, mtStream, multi);
    const size_t bytes = N * 64 * 64 * sizeof(float3);
    return closeFloats(single.ptr().data, multi.ptr().data, bytes);
}

// Executions on one stream run in the order they were enqueued: a chain of dependent executions, enqueued without
// synchronizing, gives the expected result
bool testStreamOrder() {
    constexpr int W = 257, H = 129, STEPS = 200;
    Ptr2D<float> a(W, H, 0, MemType::Host);
    Ptr2D<float> b(W, H, 0, MemType::Host);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            a.at(x, y) = static_cast<float>(x + y);
        }
    }
    MTStream stream;
    for (int i = 0; i < STEPS; ++i) {
        // a -> b: +1, then b -> a: +1
        executeOperations<MTDPP>(stream, PerThreadRead<ND::_2D, float>::build(a.ptr()), Add<float>::build(1.f),
                                 PerThreadWrite<ND::_2D, float>::build(b.ptr()));
        executeOperations<MTDPP>(stream, PerThreadRead<ND::_2D, float>::build(b.ptr()), Add<float>::build(1.f),
                                 PerThreadWrite<ND::_2D, float>::build(a.ptr()));
    }
    stream.sync();
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            if (a.at(x, y) != static_cast<float>(x + y + 2 * STEPS)) {
                return false;
            }
        }
    }
    return true;
}

// Several streams in flight at the same time on the shared pool, a copied stream sharing its queue, and a stream
// destroyed with work still enqueued (its destructor waits for it)
bool testConcurrentStreams() {
    constexpr int W = 512, H = 256, STEPS = 50;
    constexpr size_t STREAMS = 4;
    std::vector<Ptr2D<int>> data;
    for (size_t s = 0; s < STREAMS; ++s) {
        data.emplace_back(W, H, 0, MemType::Host);
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                data[s].at(x, y) = static_cast<int>(s);
            }
        }
    }
    {
        std::vector<MTStream> streams(STREAMS);
        for (int i = 0; i < STEPS; ++i) {
            for (size_t s = 0; s < STREAMS; ++s) {
                MTStream copy = streams[s];
                executeOperations<MTDPP>(copy, PerThreadRead<ND::_2D, int>::build(data[s].ptr()), Add<int>::build(1),
                                         PerThreadWrite<ND::_2D, int>::build(data[s].ptr()));
            }
        }
        // No sync: the streams are destroyed here and their destructors wait for the enqueued executions
    }
    for (size_t s = 0; s < STREAMS; ++s) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                if (data[s].at(x, y) != static_cast<int>(s) + STEPS) {
                    return false;
                }
            }
        }
    }
    return true;
}

// Fails the test instead of hanging it when a future is not ready in time (a deadlock)
template <typename T>
bool readyInTime(std::future<T>& future, const char* what) {
    if (future.wait_for(std::chrono::seconds(60)) == std::future_status::ready) {
        return true;
    }
    std::cout << "CPU_MULTITHREAD: " << what << " did not finish in 60 s (deadlock)" << std::endl;
    std::_Exit(1);
}

Ptr2D<float> filled(const int width, const int height, const float value) {
    Ptr2D<float> ptr(static_cast<uint>(width), static_cast<uint>(height), 0, MemType::Host);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            ptr.at(x, y) = value + static_cast<float>(x);
        }
    }
    return ptr;
}

bool allEqual(const Ptr2D<float>& ptr, const float value) {
    for (int y = 0; y < static_cast<int>(ptr.dims().height); ++y) {
        for (int x = 0; x < static_cast<int>(ptr.dims().width); ++x) {
            if (ptr.at(x, y) != value + static_cast<float>(x)) {
                return false;
            }
        }
    }
    return true;
}

// Streams used and synchronized (explicitly and by their destructors) from inside workers of the pool, more of them
// than there are workers: the waiting workers run queued chunks instead of blocking
bool testSyncOnWorkers() {
    constexpr int W = 300, H = 200;
    const size_t tasks = 2 * cpuThreadPool().size();
    std::vector<Ptr2D<float>> data;
    for (size_t i = 0; i < tasks; ++i) {
        data.push_back(filled(W, H, 0.f));
    }
    std::vector<std::future<bool>> futures;
    for (size_t i = 0; i < tasks; ++i) {
        futures.push_back(cpuThreadPool().submit([&data, i] {
            MTStream stream;
            for (int step = 0; step < 3; ++step) {
                executeOperations<MTDPP>(data[i], data[i], stream, Add<float>::build(1.f));
            }
            stream.sync();
            const bool synced = allEqual(data[i], 3.f);
            executeOperations<MTDPP>(data[i], data[i], stream, Add<float>::build(1.f));
            return synced; // the stream's destructor waits for the last execution, on this worker
        }));
    }
    bool correct = true;
    for (size_t i = 0; i < tasks; ++i) {
        correct &= readyInTime(futures[i], "sync on pool workers") && futures[i].get() && allEqual(data[i], 4.f);
    }
    return correct;
}

// Holds its value until gate GATE is opened, so that a test can keep an execution in flight
inline std::atomic<bool>& gate(const int id) {
    static std::atomic<bool> gates[2]{};
    return gates[id];
}

template <int GATE>
struct WaitGate {
private:
    using SelfType = WaitGate<GATE>;
public:
    FK_STATIC_STRUCT(WaitGate, SelfType)
    using Parent = UnaryOperation<float, float, WaitGate<GATE>>;
    DECLARE_UNARY_PARENT
    static inline OutputType exec(const InputType input) {
        while (!gate(GATE).load()) {
            std::this_thread::yield();
        }
        return input;
    }
};

// sync() waits for the executions enqueued before it, not for those enqueued later by other threads: A is enqueued
// and synced, then B is enqueued while the sync waits; sync must return once A finishes, with B still blocked
bool testSyncWaitsOnlyForEarlierWork() {
    gate(0).store(false);
    gate(1).store(false);
    // One row each: one chunk, so each execution blocks a single worker
    Ptr2D<float> a = filled(16, 1, 0.f);
    Ptr2D<float> b = filled(16, 1, 0.f);
    MTStream stream;
    std::atomic<bool> aboutToSync{false};
    auto waiter = std::async(std::launch::async, [&] {
        executeOperations<MTDPP>(a, a, stream, WaitGate<0>::build(), Add<float>::build(1.f));
        aboutToSync.store(true);
        stream.sync();
        return allEqual(a, 1.f);
    });
    while (!aboutToSync.load()) {
        std::this_thread::yield();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    MTStream copy = stream;
    executeOperations<MTDPP>(b, b, copy, WaitGate<1>::build(), Add<float>::build(1.f));
    gate(0).store(true);
    const bool returned = waiter.wait_for(std::chrono::seconds(30)) == std::future_status::ready;
    if (!returned) {
        std::cout << "CPU_MULTITHREAD: sync() waited for an execution enqueued after it" << std::endl;
    }
    gate(1).store(true);
    const bool waiterCorrect = waiter.get();
    stream.sync();
    return returned && waiterCorrect && allEqual(b, 1.f);
}

// A thread that enqueues as fast as it can is throttled, like with a CUDA launch queue: the stream never holds more
// than about MAX_PENDING executions, so the work left when it stops finishes quickly
bool testEnqueueIsThrottled() {
    constexpr int W = 64, H = 64;
    Ptr2D<float> data = filled(W, H, 0.f);
    MTStream stream;
    long enqueued = 0;
    const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(300);
    while (std::chrono::steady_clock::now() < end) {
        executeOperations<MTDPP>(data, data, stream, Add<float>::build(1.f));
        ++enqueued;
    }
    const auto drainStart = std::chrono::steady_clock::now();
    stream.sync();
    const double drainSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - drainStart).count();
    const bool quick = drainSeconds < 10.;
    if (!quick) {
        std::cout << "CPU_MULTITHREAD: " << enqueued << " executions enqueued in 300 ms took " << drainSeconds
                  << " s to finish after the producer stopped" << std::endl;
    }
    return quick && allEqual(data, static_cast<float>(enqueued));
}

// The Ptr / Tensor entry points keep their buffers alive until the execution has finished, even if the caller drops
// them before sync(); and every executeOperations entry point works on this backend
void addThenDouble(MTStream& stream, const Ptr2D<float>& input, const Ptr2D<float>& output) {
    Ptr2D<float> tmp(input.dims().width, input.dims().height, 0, MemType::Host);
    executeOperations<MTDPP>(input, tmp, stream, Add<float>::build(1.f));
    executeOperations<MTDPP>(tmp, output, stream, Mul<float>::build(2.f));
    // tmp is released here, before the executions that use it have run
}

bool testEntryPointsAndBufferLifetime() {
    constexpr int W = 640, H = 480;
    MTStream stream;
    bool correct = true;

    // (input, output, stream, ops) with a temporary buffer released before sync()
    Ptr2D<float> input = filled(W, H, 1.f);
    Ptr2D<float> output(W, H, 0, MemType::Host);
    for (int i = 0; i < 20; ++i) {
        addThenDouble(stream, input, output);
    }
    stream.sync();
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            correct &= output.at(x, y) == (input.at(x, y) + 1.f) * 2.f;
        }
    }

    // (input, stream, ops..., write)
    Ptr2D<float> out2(W, H, 0, MemType::Host);
    executeOperations<MTDPP>(input, stream, Add<float>::build(3.f), PerThreadWrite<ND::_2D, float>::build(out2.ptr()));
    stream.sync();
    correct &= allEqual(out2, 4.f);

    // Batches: std::array of Ptr2D, with and without active batch, with a Tensor output or a write IOp
    constexpr size_t BATCH = 3;
    const std::array<Ptr2D<float>, BATCH> inputs{filled(W, H, 10.f), filled(W, H, 20.f), filled(W, H, 30.f)};
    Tensor<float> t1(W, H, BATCH, 1, MemType::Host);
    Tensor<float> t2(W, H, BATCH, 1, MemType::Host);
    Tensor<float> t3(W, H, BATCH, 1, MemType::Host);
    Tensor<float> t4(W, H, BATCH, 1, MemType::Host);
    executeOperations<MTDPP>(inputs, t1, stream, Add<float>::build(1.f));
    executeOperations<MTDPP>(inputs, 2, -100.f, t2, stream, Add<float>::build(1.f));
    executeOperations<MTDPP>(inputs, stream, Add<float>::build(1.f), PerThreadWrite<ND::_3D, float>::build(t3.ptr()));
    executeOperations<MTDPP>(inputs, 2, -100.f, stream, Add<float>::build(1.f),
                             PerThreadWrite<ND::_3D, float>::build(t4.ptr()));
    stream.sync();
    for (int z = 0; z < static_cast<int>(BATCH); ++z) {
        const float base = 10.f * static_cast<float>(z + 1) + 1.f;
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const float expected = base + static_cast<float>(x);
                const float expectedActive = z < 2 ? expected : -99.f;
                correct &= t1.at(x, y, z) == expected && t3.at(x, y, z) == expected;
                correct &= t2.at(x, y, z) == expectedActive && t4.at(x, y, z) == expectedActive;
            }
        }
    }
    return correct;
}

// Thread fusion (TF::ENABLED): the same result as the single thread CPU backend, with a width divisible by the
// elements per thread and one that is not, so that both THREAD_DIVISIBLE versions run. The rows are padded to 64
// bytes: thread fusion reads and writes vectors, which must be aligned (host Ptr2D rows are not padded by default).
bool testThreadFusion() {
    bool correct = true;
    for (const int W : {1024, 1001}) {
        constexpr int H = 77;
        const auto padded = [](const int bytes) { return static_cast<uint>((bytes + 63) / 64 * 64); };
        Ptr2D<uchar> input(static_cast<uint>(W), H, padded(W), MemType::Host);
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                input.at(x, y) = static_cast<uchar>((x * 7 + y * 13) & 0xFF);
            }
        }
        Ptr2D<float> single(static_cast<uint>(W), H, padded(W * 4), MemType::Host);
        Ptr2D<float> multi(static_cast<uint>(W), H, padded(W * 4), MemType::Host);
        const auto run = [&](auto dpp, auto& stream, Ptr2D<float>& output) {
            executeOperations<decltype(dpp)>(stream, PerThreadRead<ND::_2D, uchar>::build(input.ptr()),
                                             Cast<uchar, float>::build(), Mul<float>::build(0.5f),
                                             PerThreadWrite<ND::_2D, float>::build(output.ptr()));
            stream.sync();
        };
        Stream_<ParArch::CPU> cpuStream;
        MTStream mtStream;
        run(TransformDPP<ParArch::CPU, TF::ENABLED>{}, cpuStream, single);
        run(TransformDPP<ParArch::CPU_MULTITHREAD, TF::ENABLED>{}, mtStream, multi);
        for (int y = 0; y < H; ++y) {
            correct &= std::memcmp(&single.at(0, y), &multi.at(0, y), static_cast<size_t>(W) * sizeof(float)) == 0;
        }
    }
    return correct;
}

// Throws for one value, to test how errors in the work of an execution are reported
struct ThrowOn13 {
private:
    using SelfType = ThrowOn13;
public:
    FK_STATIC_STRUCT(ThrowOn13, SelfType)
    using Parent = UnaryOperation<float, float, ThrowOn13>;
    DECLARE_UNARY_PARENT
    static inline OutputType exec(const InputType input) {
        if (input == 13.f) {
            throw std::runtime_error("ThrowOn13");
        }
        return input;
    }
};

// An exception thrown by the work of an execution is rethrown by the next sync(), once; the stream keeps working
bool testErrorsReachSync() {
    Ptr2D<float> data = filled(64, 16, 0.f);
    MTStream stream;
    executeOperations<MTDPP>(data, data, stream, ThrowOn13::build());
    bool thrown = false;
    try {
        stream.sync();
    } catch (const std::runtime_error&) {
        thrown = true;
    }
    bool secondThrows = false;
    try {
        stream.sync();
    } catch (...) {
        secondThrows = true;
    }
    executeOperations<MTDPP>(data, data, stream, Add<float>::build(1.f));
    stream.sync();
    return thrown && !secondThrows && allEqual(data, 1.f);
}

// Copies of a stream made and destroyed concurrently on several threads (tasks of the pool that capture it by value
// and are never waited for): the stream's reference count stays right, and all their executions run in order
bool testStreamCopiesAcrossThreads() {
    constexpr int TASKS = 2000;
    Ptr2D<float> data = filled(32, 4, 0.f);
    MTStream stream;
    std::atomic<int> finished{0};
    for (int i = 0; i < TASKS; ++i) {
        (void)cpuThreadPool().submit([stream, &data, &finished]() mutable {
            MTStream copy = stream;
            executeOperations<MTDPP>(copy, PerThreadRead<ND::_2D, float>::build(data.ptr()), Add<float>::build(1.f),
                                     PerThreadWrite<ND::_2D, float>::build(data.ptr()));
            finished.fetch_add(1);
        });
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while ((finished.load() < TASKS || stream.getRefCount() != 1) && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    const bool countRight = stream.getRefCount() == 1;
    stream.sync();
    return countRight && finished.load() == TASKS && allEqual(data, static_cast<float>(TASKS));
}

// setTo, upload and download on a CPU_MULTITHREAD stream. setTo keeps its Ptr alive until the fill has run, so the
// caller can release it before sync(). upload and download of a Ptr or an Image are no-ops ordered with the executions
// of the stream. All of them need host memory: with Device or DeviceAndPinned memory they throw and enqueue nothing.
bool testMemoryHelpers() {
    constexpr int W = 333, H = 77;
    const auto allValue = [](const Ptr2D<float>& ptr, const float value) {
        for (int y = 0; y < static_cast<int>(ptr.dims().height); ++y) {
            for (int x = 0; x < static_cast<int>(ptr.dims().width); ++x) {
                if (ptr.at(x, y) != value) {
                    return false;
                }
            }
        }
        return true;
    };
    MTStream stream;
    Stream_<ParArch::CPU> cpuStream;
    bool correct = true;

    // setTo on buffers the caller releases before the fills have run: the stream is held busy by an execution that
    // waits for gate 0, so the fills are still pending when the buffers are released
    gate(0).store(false);
    Ptr2D<float> busy = filled(16, 1, 0.f);
    executeOperations<MTDPP>(busy, busy, stream, WaitGate<0>::build());
    Ptr2D<float> output(W, H, 0, MemType::Host);
    int refsWhilePending = 0;
    {
        Ptr2D<float> tmp(W, H, 0, MemType::Host);
        setTo(5.f, tmp, stream);
        refsWhilePending = tmp.getRefCount(); // the caller and the pending fill
        executeOperations<MTDPP>(tmp, output, stream, Add<float>::build(1.f));
    }
    {
        // Once released, the pending fill is the only owner of this one
        Ptr2D<float> released(W, H, 0, MemType::Host);
        setTo(7.f, released, stream);
    }
    // The same with the ParArch given explicitly
    Ptr2D<float> outputExplicit(W, H, 0, MemType::Host);
    int refsWhilePendingExplicit = 0;
    {
        Ptr2D<float> tmp(W, H, 0, MemType::Host);
        setTo<ParArch::CPU_MULTITHREAD>(9.f, tmp, stream);
        refsWhilePendingExplicit = tmp.getRefCount();
        executeOperations<MTDPP>(tmp, outputExplicit, stream, Add<float>::build(1.f));
    }
    {
        Ptr2D<float> released(W, H, 0, MemType::Host);
        setTo<ParArch::CPU_MULTITHREAD>(11.f, released, stream);
    }
    gate(0).store(true);
    stream.sync();
    correct &= refsWhilePending == 2 && allValue(output, 6.f) && allEqual(busy, 0.f);
    correct &= refsWhilePendingExplicit == 2 && allValue(outputExplicit, 10.f);

    // setTo gives the same result as on ParArch::CPU
    Tensor<float> singleFill(W, H, 4, 1, MemType::Host);
    Tensor<float> multiFill(W, H, 4, 1, MemType::Host);
    setTo(-2.5f, singleFill, cpuStream);
    setTo(-2.5f, multiFill, stream);
    stream.sync();
    correct &= std::memcmp(singleFill.ptr().data, multiFill.ptr().data, singleFill.sizeInBytes()) == 0;

    // upload and download of a Ptr around an execution that uses it
    Ptr2D<float> input = filled(W, H, 1.f);
    Ptr2D<float> result(W, H, 0, MemType::Host);
    input.upload(stream);
    executeOperations<MTDPP>(input, result, stream, Mul<float>::build(2.f));
    result.download(stream);
    stream.sync();
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            correct &= result.at(x, y) == (1.f + static_cast<float>(x)) * 2.f;
        }
    }

    // upload and download of Images: NV12 to RGB gives the same result as on ParArch::CPU, and the image data copied
    // into another image is downloaded intact
    constexpr int IW = 64, IH = 32;
    Image<PixelFormat::NV12> image(IW, IH, MemType::Host);
    Image<PixelFormat::NV12> imageCopy(IW, IH, MemType::Host);
    auto imageData = image.getData();
    for (int y = 0; y < IH * 3 / 2; ++y) {
        for (int x = 0; x < IW; ++x) {
            imageData.at(x, y) = static_cast<uchar>((x * 7 + y * 13) & 0xFF);
        }
    }
    Tensor<float3> rgbSingle(IW, IH, 1, 1, MemType::Host);
    Tensor<float3> rgbMulti(IW, IH, 1, 1, MemType::Host);
    const auto toRGB = [&](auto dpp, auto& anyStream, Tensor<float3>& rgb) {
        executeOperations<decltype(dpp)>(anyStream, ReadYUV<PixelFormat::NV12>::build(image),
            ConvertYUVToRGB<ColorDepth::p8bit, ColorRange::Full, ColorPrimitives::bt709>::build(),
            TensorWrite<float3>::build(rgb.ptr()));
    };
    image.upload(stream);
    toRGB(MTDPP{}, stream, rgbMulti);
    executeOperations<MTDPP>(image.getData(), imageCopy.getData(), stream);
    imageCopy.download(stream);
    stream.sync();
    toRGB(CPUDPP{}, cpuStream, rgbSingle);
    const auto copyData = imageCopy.getData();
    for (int y = 0; y < IH * 3 / 2; ++y) {
        for (int x = 0; x < IW; ++x) {
            correct &= copyData.at(x, y) == imageData.at(x, y);
        }
    }
    correct &= closeFloats(rgbSingle.ptr().data, rgbMulti.ptr().data, rgbSingle.sizeInBytes());

    // Device and DeviceAndPinned memory (wrapping host vectors, which they never touch)
    const auto throwsRuntimeError = [](const auto& function) {
        try {
            function();
        } catch (const std::runtime_error&) {
            return true;
        }
        return false;
    };
    std::vector<float> fakeDevice(W * H, 0.f);
    std::vector<float> fakePinned(W * H, 0.f);
    const PtrDims<ND::_2D> dims(W, H, static_cast<uint>(W * sizeof(float)));
    Ptr<ND::_2D, float> device(fakeDevice.data(), dims, MemType::Device, 0);
    Ptr<ND::_2D, float> deviceAndPinned(fakeDevice.data(), fakePinned.data(), dims, MemType::DeviceAndPinned, 0);
    for (Ptr<ND::_2D, float>* ptr : {&device, &deviceAndPinned}) {
        correct &= throwsRuntimeError([&] { ptr->upload(stream); });
        correct &= throwsRuntimeError([&] { ptr->download(stream); });
        correct &= throwsRuntimeError([&] { setTo(1.f, *ptr, stream); });
        correct &= throwsRuntimeError([&] { setTo<ParArch::CPU_MULTITHREAD>(1.f, *ptr, stream); });
    }
    std::vector<uchar> fakeImage(IW * IH * 3 / 2, 0);
    Image<PixelFormat::NV12> deviceImage(
        Ptr<ND::_2D, uchar>(fakeImage.data(), PtrDims<ND::_2D>(IW, IH * 3 / 2, IW), MemType::Device, 0), IW, IH);
    correct &= throwsRuntimeError([&] { deviceImage.upload(stream); });
    correct &= throwsRuntimeError([&] { deviceImage.download(stream); });
    // The stream keeps working, and nothing wrote the "device" memory
    executeOperations<MTDPP>(input, result, stream, Add<float>::build(1.f));
    stream.sync();
    correct &= allEqual(result, 2.f);
    for (const float value : fakeDevice) {
        correct &= value == 0.f;
    }
    for (const float value : fakePinned) {
        correct &= value == 0.f;
    }
    return correct;
}

// An execution that fails to start (as a DPP's exec does when it can not allocate) while an earlier one is running:
// the stream drops the pending executions and the next sync() reports the error every time, also when the stream is
// destroyed right after it (the worker that started it must not use the stream any more)
bool testStartFailureReachesSync() {
    constexpr int ITERATIONS = 2000;
    int reported = 0;
    for (int i = 0; i < ITERATIONS; ++i) {
        gate(0).store(false);
        Ptr2D<float> data = filled(16, 1, 0.f);
        auto stream = std::make_unique<MTStream>();
        executeOperations<MTDPP>(data, data, *stream, WaitGate<0>::build(), Add<float>::build(1.f));
        stream->enqueue([](const cpu_multithread::Completion&) { throw std::runtime_error("start failed"); });
        gate(0).store(true);
        try {
            stream->sync();
        } catch (const std::runtime_error&) {
            ++reported;
        }
        stream.reset();
    }
    return reported == ITERATIONS;
}

// sync() reports only errors of executions enqueued before it: an execution enqueued after the call, which fails as
// soon as the earlier one finishes, is left for the next sync()
bool testSyncIgnoresLaterErrors() {
    constexpr int ITERATIONS = 100;
    bool correct = true;
    for (int i = 0; i < ITERATIONS && correct; ++i) {
        gate(0).store(false);
        Ptr2D<float> data = filled(16, 1, 0.f);
        MTStream stream;
        executeOperations<MTDPP>(data, data, stream, WaitGate<0>::build(), Add<float>::build(1.f));
        std::atomic<bool> aboutToSync{false};
        auto first = std::async(std::launch::async, [&] {
            aboutToSync.store(true);
            try {
                stream.sync();
                return true;
            } catch (...) {
                return false;
            }
        });
        while (!aboutToSync.load()) {
            std::this_thread::yield();
        }
        // Let the first sync() take its target before the failing execution is enqueued
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        stream.enqueue([](const cpu_multithread::Completion&) { throw std::runtime_error("later execution"); });
        gate(0).store(true);
        if (!first.get()) {
            std::cout << "CPU_MULTITHREAD: sync() reported the error of an execution enqueued after it" << std::endl;
            correct = false;
        }
        bool laterReported = false;
        try {
            stream.sync();
        } catch (const std::runtime_error&) {
            laterReported = true;
        }
        correct &= laterReported;
    }
    return correct;
}

// Several threads enqueueing on copies of one stream at the same time never get more than MAX_PENDING executions
// pending: the first one waits on a gate, so none finishes while they fill the stream
bool testConcurrentEnqueueBound() {
    constexpr int THREADS = 8;
    constexpr auto MAX_PENDING = cpu_multithread::detail::ExecutionQueue::MAX_PENDING;
    gate(0).store(false);
    Ptr2D<float> data = filled(16, 1, 0.f);
    MTStream stream;
    std::atomic<std::uint64_t> enqueued{0};
    std::atomic<bool> stop{false};
    std::vector<std::thread> threads;
    for (int t = 0; t < THREADS; ++t) {
        threads.emplace_back([&] {
            MTStream copy = stream;
            while (!stop.load()) {
                executeOperations<MTDPP>(copy, PerThreadRead<ND::_2D, float>::build(data.ptr()), WaitGate<0>::build(),
                                         PerThreadWrite<ND::_2D, float>::build(data.ptr()));
                enqueued.fetch_add(1);
            }
        });
    }
    // Wait until the stream is full (slow builds take longer), then a bit more, in case it went past the limit
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (enqueued.load() < MAX_PENDING && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    const std::uint64_t pending = enqueued.load();
    stop.store(true);
    gate(0).store(true);
    for (auto& thread : threads) {
        thread.join();
    }
    stream.sync();
    if (pending > MAX_PENDING) {
        std::cout << "CPU_MULTITHREAD: " << pending << " executions pending, more than " << MAX_PENDING << std::endl;
    }
    return pending == MAX_PENDING;
}

int launch_impl() {
    const bool same = testSameResultAsCPU();
    const bool order = testStreamOrder();
    const bool concurrent = testConcurrentStreams();
    const bool workers = testSyncOnWorkers();
    const bool earlier = testSyncWaitsOnlyForEarlierWork();
    const bool entryPoints = testEntryPointsAndBufferLifetime();
    const bool throttled = testEnqueueIsThrottled();
    const bool threadFusion = testThreadFusion();
    const bool errors = testErrorsReachSync();
    const bool copies = testStreamCopiesAcrossThreads();
    const bool startFailure = testStartFailureReachesSync();
    const bool laterErrors = testSyncIgnoresLaterErrors();
    const bool bound = testConcurrentEnqueueBound();
    const bool memoryHelpers = testMemoryHelpers();
    std::cout << "CPU_MULTITHREAD (" << cpuThreadPool().size() << " threads): same result as CPU " << same
              << ", stream order " << order << ", concurrent streams " << concurrent << ", sync on workers "
              << workers << ", sync waits only for earlier work " << earlier << ", entry points and buffer lifetime "
              << entryPoints << ", enqueue throttled " << throttled << ", thread fusion " << threadFusion
              << ", errors reach sync " << errors << ", stream copies across threads " << copies << ", start failure reaches sync " << startFailure
              << ", sync ignores later errors " << laterErrors << ", concurrent enqueue bound " << bound
              << ", setTo upload download " << memoryHelpers << std::endl;
    return same && order && concurrent && workers && earlier && entryPoints && throttled && threadFusion &&
                   errors && copies && memoryHelpers && startFailure &&
                   laterErrors && bound ? 0 : -1;
}

} // namespace fk

int launch() {
    return fk::launch_impl();
}

#else

#include <iostream>

int launch() {
    std::cout << "ENABLE_CPU_MULTITHREAD is OFF (thread-pool submodule not checked out): CPU_MULTITHREAD not tested"
              << std::endl;
    return 0;
}

#endif
