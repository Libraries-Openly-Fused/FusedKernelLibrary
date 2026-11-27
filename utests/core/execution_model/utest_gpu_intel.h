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

// The SYCL backend (ParArch::GPU_INTEL): what is specific to it, which the tests every backend shares do not cover.
// - Its devices, the Intel GPUs and nothing else, and the primary context of each one, which its streams share.
// - Device memory: the rows of a 2D pointer allocated without a pitch are aligned, so the device and the pinned
//   buffers of a pointer have different pitches, and the transfers between them copy row by row.
// - The parameters of a kernel that do not fit in its arguments (FK_SYCL_MAX_KERNEL_ARGUMENTS_BYTES) go through device
//   memory: each kernel has its own copy, released when it has finished.
// - Memory can be released while the kernels that use it are enqueued: releasing it waits for them.
// - A stream that uses a queue of the user.
// The other targets have nothing to test.

#include <tests/main.h>

#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/cast.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/fused_kernel.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#if defined(FK_SYCL_ENABLED)
namespace fk {
namespace gpu_intel_test {

using SYCLStream = Stream_<ParArch::GPU_INTEL>;

bool report(const char* const name, const bool correct) {
    std::cout << "GPU_INTEL " << name << ": " << (correct ? "OK" : "FAILED") << std::endl;
    return correct;
}

float pattern(const int x, const int y, const int z) {
    return static_cast<float>((x * 7 + y * 13 + z * 29) % 251);
}

bool isIntelGPU(const sycl::device& device) {
    return device.is_gpu() && device.get_info<sycl::info::device::vendor_id>() == 0x8086 &&
           device.has(sycl::aspect::usm_device_allocations);
}

// The devices of the backend are the Intel GPUs of one platform, each one once, and nothing else (not the CPU, not the
// GPUs of other vendors). Device 0 is the first Intel GPU the runtime offers through Level Zero, if it offers one
// through it.
bool testDevices() {
    const sycl::device& device = gpu_intel::device();
    bool correct = gpu_intel::device(0) == device && gpu_intel::context(0) == gpu_intel::context();

    bool levelZero = false;
    for (const sycl::device& gpu : sycl::device::get_devices(sycl::info::device_type::gpu)) {
        levelZero |= isIntelGPU(gpu) && gpu.get_backend() == sycl::backend::ext_oneapi_level_zero;
    }
    correct &= !levelZero || device.get_backend() == sycl::backend::ext_oneapi_level_zero;

    std::vector<sycl::device> devices;
    for (int deviceID = 0;; ++deviceID) {
        try {
            devices.push_back(gpu_intel::device(deviceID));
        } catch (const std::out_of_range&) {
            break;
        }
    }
    std::size_t intelGPUs = 0;
    for (const sycl::device& gpu : device.get_platform().get_devices(sycl::info::device_type::gpu)) {
        intelGPUs += isIntelGPU(gpu) ? 1 : 0;
    }
    correct &= !devices.empty() && devices.size() == intelGPUs;
    for (std::size_t deviceID = 0; deviceID < devices.size(); ++deviceID) {
        const sycl::device& gpu = devices[deviceID];
        std::cout << "GPU_INTEL device " << deviceID << ": " << gpu.get_info<sycl::info::device::name>() << " ("
                  << gpu.get_platform().get_info<sycl::info::platform::name>() << ")" << std::endl;
        correct &= isIntelGPU(gpu) && gpu.get_platform() == device.get_platform();
        const std::vector<sycl::device> inContext = gpu_intel::context(static_cast<int>(deviceID)).get_devices();
        correct &= inContext.size() == 1 && inContext[0] == gpu;
        for (std::size_t other = 0; other < deviceID; ++other) {
            correct &= devices[other] != gpu;
        }
    }

    bool noSuchDevice = false;
    try {
        (void)gpu_intel::device(static_cast<int>(devices.size()));
    } catch (const std::out_of_range&) {
        noSuchDevice = true;
    }
    bool noNegativeDevice = false;
    try {
        (void)gpu_intel::context(-1);
    } catch (const std::out_of_range&) {
        noNegativeDevice = true;
    }
    bool noSuchPtrDevice = false;
    try {
        Ptr2D<float> wrong(8, 8, 0, MemType::Device, 1000);
    } catch (const std::out_of_range&) {
        noSuchPtrDevice = true;
    }
    return correct && noSuchDevice && noNegativeDevice && noSuchPtrDevice;
}

// The rows of a 2D device buffer allocated without a pitch are aligned, and the pinned buffer has no padding. The
// transfers and the kernels walk each buffer with its own pitch.
bool testPitches(SYCLStream& stream) {
    constexpr int W = 33, H = 7; // 99 bytes per row
    constexpr uint ROW_BYTES = static_cast<uint>(W * sizeof(uchar3));
    Ptr2D<uchar3> input(W, H);
    Ptr2D<uchar3> output(W, H);
    constexpr size_t ALIGNMENT = Ptr2D<uchar3>::SYCL_MEMORY_ALIGNMENT;
    const uint devicePitch = input.ptr().dims.pitch;
    bool correct = devicePitch % ALIGNMENT == 0 && devicePitch > ROW_BYTES && devicePitch < ROW_BYTES + ALIGNMENT &&
                   input.ptrPinned().dims.pitch == ROW_BYTES;
    correct &= reinterpret_cast<std::uintptr_t>(input.ptr().data) % ALIGNMENT == 0 &&
               reinterpret_cast<std::uintptr_t>(input.ptrPinned().data) % ALIGNMENT == 0;

    // A pitch given by the caller is kept, and a row that is a multiple of the alignment has no padding
    const Ptr2D<uchar3> pitched(W, H, 128, MemType::Device);
    const Ptr2D<float> exact(64, H, 0, MemType::Device);
    correct &= pitched.ptr().dims.pitch == 128 && exact.ptr().dims.pitch == 64 * sizeof(float);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const uchar value = static_cast<uchar>(pattern(x, y, 0));
            input.at(Point{ x, y, 0 }) = uchar3{ value, static_cast<uchar>(value + 1), static_cast<uchar>(value + 2) };
            output.at(Point{ x, y, 0 }) = uchar3{ 0, 0, 0 };
        }
    }
    input.upload(stream);
    executeOperations<TransformDPP<>>(stream, PerThreadRead<ND::_2D, uchar3>::build(input),
                                      PerThreadWrite<ND::_2D, uchar3>::build(output));
    output.download(stream);
    stream.sync();
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const uchar value = static_cast<uchar>(pattern(x, y, 0));
            const uchar3 result = output.at(Point{ x, y, 0 });
            correct &= result.x == value && result.y == static_cast<uchar>(value + 1) &&
                       result.z == static_cast<uchar>(value + 2);
        }
    }
    return correct;
}

// Transfers from and to memory that is not pinned (MemType::Host) have finished when uploadTo and downloadTo return
bool testHostTransfers(SYCLStream& stream) {
    constexpr int W = 37, H = 5;
    Ptr2D<float> device(W, H, 0, MemType::Device);
    bool correct = true;
    {
        Ptr2D<float> source(W, H, 0, MemType::Host);
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                source.at(Point{ x, y, 0 }) = pattern(x, y, 1);
            }
        }
        source.uploadTo(device, stream);
        // source is released here: the copy has already read it
    }
    Ptr2D<float> destination(W, H, 0, MemType::Host);
    device.downloadTo(destination, stream);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            correct &= destination.at(Point{ x, y, 0 }) == pattern(x, y, 1);
        }
    }
    stream.sync();
    return correct;
}

// A batch whose read IOp is larger than the arguments of a kernel can be: the parameters of the kernel are in device
// memory. Every launch has its own parameters, also when many launches are enqueued before any of them runs.
bool testLargeKernelParameters(SYCLStream& stream) {
    constexpr size_t BATCH = 64;
    constexpr int W = 19, H = 11, LAUNCHES = 24;
    std::array<Ptr2D<float>, BATCH> inputs;
    for (size_t z = 0; z < BATCH; ++z) {
        inputs[z] = Ptr2D<float>(W, H);
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                inputs[z].at(Point{ x, y, 0 }) = pattern(x, y, static_cast<int>(z));
            }
        }
        inputs[z].upload(stream);
    }
    const auto read = PerThreadRead<ND::_2D, float>::build(inputs);
    static_assert(sizeof(read) > FK_SYCL_MAX_KERNEL_ARGUMENTS_BYTES,
                  "This test needs a read IOp that does not fit in the arguments of a kernel");

    std::vector<Tensor<float>> outputs;
    for (int launch = 0; launch < LAUNCHES; ++launch) {
        outputs.emplace_back(W, H, static_cast<uint>(BATCH));
    }
    SYCLStream sameQueue = stream;
    for (int launch = 0; launch < LAUNCHES; ++launch) {
        // Through two copies of the stream, which share the queue and the parameters that are pending
        SYCLStream& used = launch % 2 == 0 ? stream : sameQueue;
        executeOperations<TransformDPP<>>(used, read, Mul<float>::build(static_cast<float>(launch + 1)),
                                          TensorWrite<float>::build(outputs[static_cast<size_t>(launch)]));
    }
    for (Tensor<float>& output : outputs) {
        output.download(stream);
    }
    stream.sync();

    bool correct = true;
    for (int launch = 0; launch < LAUNCHES; ++launch) {
        for (int z = 0; z < static_cast<int>(BATCH); ++z) {
            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                    correct &= outputs[static_cast<size_t>(launch)].at(Point{ x, y, z }) ==
                               pattern(x, y, z) * static_cast<float>(launch + 1);
                }
            }
        }
    }
    return correct;
}

// A pointer is released while kernels that read it are still enqueued: releasing it waits for them
bool testReleaseWithPendingWork(SYCLStream& stream) {
    constexpr int W = 640, H = 360, LAUNCHES = 16;
    Ptr2D<float> output(W, H);
    {
        Ptr2D<float> input(W, H);
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                input.at(Point{ x, y, 0 }) = pattern(x, y, 2);
            }
        }
        input.upload(stream);
        for (int launch = 0; launch < LAUNCHES; ++launch) {
            executeOperations<TransformDPP<>>(stream, PerThreadRead<ND::_2D, float>::build(input),
                                              Mul<float>::build(2.f), Add<float>::build(1.f),
                                              PerThreadWrite<ND::_2D, float>::build(output));
        }
    }
    output.download(stream);
    stream.sync();
    bool correct = true;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            correct &= output.at(Point{ x, y, 0 }) == pattern(x, y, 2) * 2.f + 1.f;
        }
    }
    return correct;
}

// The streams of a device share its primary context, as CUDA streams share the primary context of their device: the
// memory of a Ptr is usable from all of them. Work is ordered within each stream, not between them (sync() here).
bool testStreamsShareContext() {
    constexpr int W = 64, H = 16;
    SYCLStream first;
    SYCLStream second;
    bool correct = first.getSYCLQueue().get_context() == gpu_intel::context() &&
                   second.getSYCLQueue().get_context() == gpu_intel::context() &&
                   first.getSYCLQueue() != second.getSYCLQueue();
    Ptr2D<float> input(W, H);
    Ptr2D<float> middle(W, H);
    Ptr2D<float> output(W, H);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            input.at(Point{ x, y, 0 }) = pattern(x, y, 4);
        }
    }
    input.upload(first);
    executeOperations<TransformDPP<>>(first, PerThreadRead<ND::_2D, float>::build(input), Mul<float>::build(2.f),
                                      PerThreadWrite<ND::_2D, float>::build(middle));
    first.sync();
    executeOperations<TransformDPP<>>(second, PerThreadRead<ND::_2D, float>::build(middle), Add<float>::build(1.f),
                                      PerThreadWrite<ND::_2D, float>::build(output));
    output.download(second);
    second.sync();
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            correct &= output.at(Point{ x, y, 0 }) == pattern(x, y, 4) * 2.f + 1.f;
        }
    }
    return correct;
}

// A stream on a queue the user made with FKL's context, which can use the memory of Ptr. The user also uses the
// queue directly, and what they enqueue is ordered with what FKL enqueues.
bool testUserQueue() {
    constexpr int W = 50, H = 20;
    sycl::queue queue(gpu_intel::context(), gpu_intel::device(), sycl::property::queue::in_order());
    SYCLStream stream(queue);
    Ptr2D<float> input(W, H);
    Ptr2D<float> output(W, H);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            input.at(Point{ x, y, 0 }) = pattern(x, y, 3);
        }
    }
    input.upload(stream);
    executeOperations<TransformDPP<>>(stream, PerThreadRead<ND::_2D, float>::build(input), Mul<float>::build(3.f),
                                      PerThreadWrite<ND::_2D, float>::build(output));
    // The user adds 1 to the first element of the device buffer, between FKL's kernel and FKL's download
    float* const first = output.ptr().data;
    queue.single_task([first]() { *first += 1.f; });
    output.download(stream);
    queue.wait_and_throw();
    bool correct = true;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const float expected = pattern(x, y, 3) * 3.f + (x == 0 && y == 0 ? 1.f : 0.f);
            correct &= output.at(Point{ x, y, 0 }) == expected;
        }
    }
    return correct;
}

int run() {
    bool correct = report("devices", testDevices());
    SYCLStream stream;
    correct &= report("pitches of device and pinned memory", testPitches(stream));
    correct &= report("transfers of host memory", testHostTransfers(stream));
    correct &= report("kernel parameters in device memory", testLargeKernelParameters(stream));
    correct &= report("memory released with pending work", testReleaseWithPendingWork(stream));
    correct &= report("streams of a device share its context", testStreamsShareContext());
    correct &= report("stream on a queue of the user", testUserQueue());
    return correct ? 0 : -1;
}

} // namespace gpu_intel_test
} // namespace fk

int launch() {
    try {
        return fk::gpu_intel_test::run();
    } catch (const std::exception& error) {
        std::cout << "GPU_INTEL test threw: " << error.what() << std::endl;
        return -1;
    }
}
#else

int launch() {
    std::cout << "utest_gpu_intel: the SYCL backend (ParArch::GPU_INTEL) is tested in the _sycl target only"
              << std::endl;
    return 0;
}
#endif
