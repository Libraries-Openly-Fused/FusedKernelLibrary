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

// GPU streams (CUDA and HIP) own the native stream they create: copies share it, assignment adopts the other stream,
// and the last copy destroys it (never before: a copy that outlives the original keeps it usable). A stream given by
// the user is never destroyed. The CPU streams own no native stream, so the _cpp target has nothing to test.

#include <tests/main.h>

#include <fused_kernel/core/execution_model/stream.h>

#include <algorithm>
#include <iostream>
#include <memory>

#if defined(__NVCC__) || defined(__HIPCC__)
namespace fk {

#if defined(__NVCC__)
using GPUStream = Stream_<ParArch::GPU_NVIDIA>;
using NativeStream = cudaStream_t;
inline bool createNative(NativeStream* stream) { return cudaStreamCreate(stream) == cudaSuccess; }
inline bool destroyNative(const NativeStream stream) { return cudaStreamDestroy(stream) == cudaSuccess; }
inline bool usable(const NativeStream stream) {
    return cudaStreamQuery(stream) == cudaSuccess && cudaStreamSynchronize(stream) == cudaSuccess;
}
inline bool initDevice() { return cudaFree(0) == cudaSuccess; }
inline bool freeMemory(size_t* free) {
    size_t total = 0;
    return cudaMemGetInfo(free, &total) == cudaSuccess;
}
#else
using GPUStream = Stream_<ParArch::GPU_AMD>;
using NativeStream = hipStream_t;
inline bool createNative(NativeStream* stream) { return hipStreamCreate(stream) == hipSuccess; }
inline bool destroyNative(const NativeStream stream) { return hipStreamDestroy(stream) == hipSuccess; }
inline bool usable(const NativeStream stream) {
    return hipStreamQuery(stream) == hipSuccess && hipStreamSynchronize(stream) == hipSuccess;
}
inline bool initDevice() { return hipFree(0) == hipSuccess; }
inline bool freeMemory(size_t* free) {
    size_t total = 0;
    return hipMemGetInfo(free, &total) == hipSuccess;
}
#endif

inline NativeStream native(const GPUStream& stream) {
    return static_cast<NativeStream>(stream);
}

// Free device memory lost after creating, copying, assigning and destroying many streams. A created CUDA stream holds
// about 160 KB of device memory, so leaking them shows here. Other processes can use the GPU at the same time: the
// smallest of three rounds is reported, which they can only make smaller.
double megabytesLostToStreams() {
    double smallest = 1e30;
    for (int round = 0; round < 3; ++round) {
        size_t freeBefore = 0, freeAfter = 0;
        if (!freeMemory(&freeBefore)) {
            return 1e30;
        }
        for (int i = 0; i < 1000; ++i) {
            GPUStream a;
            GPUStream b;
            GPUStream c = b;
            b = a;
            c = a;
        }
        if (!freeMemory(&freeAfter)) {
            return 1e30;
        }
        smallest = std::min(smallest, (static_cast<double>(freeBefore) - static_cast<double>(freeAfter)) / 1048576.0);
    }
    return smallest;
}

int launch_impl() {
    bool correct = initDevice();

    // Copies and assignment share the native stream
    GPUStream a;
    GPUStream b;
    const GPUStream copy = a;
    correct &= native(copy) == native(a) && native(b) != native(a);
    b = a;
    correct &= native(b) == native(a);

    // The stream stays usable while any copy is alive: a copy that outlives the original, and a copy that shares it
    // with an object that is assigned another stream
    std::unique_ptr<GPUStream> survivor;
    {
        GPUStream original;
        survivor = std::make_unique<GPUStream>(original);
    }
    correct &= usable(native(*survivor));
    {
        GPUStream shared;
        GPUStream sharer = shared;
        shared = a;
        correct &= usable(native(sharer)) && native(sharer) != native(a);
    }

    // A stream given by the user is never destroyed by its wrappers
    NativeStream userStream{};
    correct &= createNative(&userStream);
    {
        GPUStream wrapper(userStream);
        GPUStream wrapperCopy = wrapper;
        GPUStream other;
        other = wrapperCopy;
        correct &= native(other) == userStream;
    }
    correct &= usable(userStream) && destroyNative(userStream);

    // Owned streams are destroyed with their last copy, also when replaced by assignment (2000 leaked CUDA streams
    // would be over 300 MB)
    const double lost = megabytesLostToStreams();
    if (lost > 64.) {
        std::cout << "GPU streams: " << lost << " MB of device memory lost to streams that were not destroyed"
                  << std::endl;
        correct = false;
    }

    std::cout << "GPU stream ownership " << (correct ? "OK" : "FAILED") << std::endl;
    return correct ? 0 : -1;
}

} // namespace fk

int launch() {
    return fk::launch_impl();
}
#elif defined(FK_SYCL_ENABLED)
#include <stdexcept>

// The SYCL stream (ParArch::GPU_INTEL) is a sycl::queue, which is itself a reference counted handle: copies share the
// queue, assignment adopts the queue of the other stream, and the queue stays usable while a copy is alive. A stream
// given a queue of the user uses that queue, which the user keeps. The queue has to be in order.
namespace fk {

using GPUStream = Stream_<ParArch::GPU_INTEL>;

inline sycl::queue& native(const GPUStream& stream) {
    return stream.getSYCLQueue();
}

// Enqueues a copy in the queue and waits for it
inline bool usable(sycl::queue& queue) {
    try {
        int source = 42;
        int destination = 0;
        queue.memcpy(&destination, &source, sizeof(int));
        queue.wait_and_throw();
        return destination == 42;
    } catch (const std::exception&) {
        return false;
    }
}

int launch_impl() {
    bool correct = true;

    // A stream has an in-order queue of FKL's device and context. Copies and assignment share the queue
    GPUStream a;
    GPUStream b;
    const GPUStream copy = a;
    correct &= native(a).is_in_order() && native(a).get_device() == gpu_intel::device() &&
               native(a).get_context() == gpu_intel::context();
    correct &= native(copy) == native(a) && native(b) != native(a);
    b = a;
    correct &= native(b) == native(a);
    correct &= static_cast<sycl::queue&>(a) == native(a) && a() == native(a);

    // The queue stays usable while any copy is alive: a copy that outlives the original, and a copy that shares it
    // with an object that is assigned another stream
    std::unique_ptr<GPUStream> survivor;
    {
        GPUStream original;
        survivor = std::make_unique<GPUStream>(original);
    }
    correct &= usable(native(*survivor));
    survivor->sync();
    {
        GPUStream shared;
        GPUStream sharer = shared;
        shared = a;
        correct &= usable(native(sharer)) && native(sharer) != native(a);
    }

    // A stream of a device given by its deviceID, and a device that does not exist
    {
        GPUStream onDevice(0);
        correct &= native(onDevice).get_device() == gpu_intel::device(0) && usable(native(onDevice));
    }
    bool noSuchDevice = false;
    try {
        GPUStream wrong(1000);
    } catch (const std::out_of_range&) {
        noSuchDevice = true;
    }
    correct &= noSuchDevice;

    // A queue given by the user is the queue of its wrappers, and the user can use it after they are destroyed
    sycl::queue userQueue(gpu_intel::context(), gpu_intel::device(), sycl::property::queue::in_order());
    {
        GPUStream wrapper(userQueue);
        GPUStream wrapperCopy = wrapper;
        GPUStream other;
        other = wrapperCopy;
        correct &= native(other) == userQueue;
        other.sync();
    }
    correct &= usable(userQueue);

    // A queue that is not in order is not a stream
    bool rejected = false;
    try {
        sycl::queue outOfOrder(gpu_intel::context(), gpu_intel::device());
        GPUStream wrong(outOfOrder);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    correct &= rejected;

    // Nor is a queue of another context: it could not use the memory of Ptr, which is in FKL's context of the device
    bool otherContextRejected = false;
    try {
        sycl::queue otherContext(sycl::context(gpu_intel::device()), gpu_intel::device(),
                                 sycl::property::queue::in_order());
        GPUStream wrong(otherContext);
    } catch (const std::invalid_argument&) {
        otherContextRejected = true;
    }
    correct &= otherContextRejected;

    // Creating, copying, assigning and destroying many streams
    for (int i = 0; i < 200; ++i) {
        GPUStream c;
        GPUStream d;
        GPUStream e = d;
        d = c;
        e = c;
    }
    correct &= usable(native(a));

    std::cout << "GPU stream ownership " << (correct ? "OK" : "FAILED") << std::endl;
    return correct ? 0 : -1;
}

} // namespace fk

int launch() {
    return fk::launch_impl();
}
#else
#include <iostream>

int launch() {
    std::cout << "CPU streams own no native stream: nothing to test" << std::endl;
    return 0;
}
#endif
