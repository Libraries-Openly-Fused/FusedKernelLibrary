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
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
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
#include <cstring>
#include <future>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

namespace fk {

using MTStream = Stream_<ParArch::CPU_MULTITHREAD>;
using MTDPP = TransformDPP<ParArch::CPU_MULTITHREAD>;
using CPUDPP = TransformDPP<ParArch::CPU>;

// The Figure 1 chain of the BVF paper (NV12 crops resized, padded and normalized) gives exactly the same result on
// the single and the multithreaded CPU backends, since both run the same per thread code
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
        const int x = distX(gen);
        const int y = distY(gen);
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
    return std::memcmp(single.ptr().data, multi.ptr().data, bytes) == 0;
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
    constexpr int W = 512, H = 256, STREAMS = 4, STEPS = 50;
    std::vector<Ptr2D<int>> data;
    for (int s = 0; s < STREAMS; ++s) {
        data.emplace_back(W, H, 0, MemType::Host);
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                data[s].at(x, y) = s;
            }
        }
    }
    {
        std::vector<MTStream> streams(STREAMS);
        for (int i = 0; i < STEPS; ++i) {
            for (int s = 0; s < STREAMS; ++s) {
                MTStream copy = streams[s];
                executeOperations<MTDPP>(copy, PerThreadRead<ND::_2D, int>::build(data[s].ptr()), Add<int>::build(1),
                                         PerThreadWrite<ND::_2D, int>::build(data[s].ptr()));
            }
        }
        // No sync: the streams are destroyed here and their destructors wait for the enqueued executions
    }
    for (int s = 0; s < STREAMS; ++s) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                if (data[s].at(x, y) != s + STEPS) {
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
    Ptr2D<float> ptr(width, height, 0, MemType::Host);
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
    constexpr int W = 256, H = 256;
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

int launch_impl() {
    const bool same = testSameResultAsCPU();
    const bool order = testStreamOrder();
    const bool concurrent = testConcurrentStreams();
    const bool workers = testSyncOnWorkers();
    const bool earlier = testSyncWaitsOnlyForEarlierWork();
    const bool entryPoints = testEntryPointsAndBufferLifetime();
    const bool throttled = testEnqueueIsThrottled();
    std::cout << "CPU_MULTITHREAD (" << cpuThreadPool().size() << " threads): same result as CPU " << same
              << ", stream order " << order << ", concurrent streams " << concurrent << ", sync on workers "
              << workers << ", sync waits only for earlier work " << earlier << ", entry points and buffer lifetime "
              << entryPoints << ", enqueue throttled " << throttled << std::endl;
    return same && order && concurrent && workers && earlier && entryPoints && throttled ? 0 : -1;
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
