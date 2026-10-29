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
// the user is never destroyed. The CPU streams own no native stream, so there is no _cpp target.

#define __ONLY_GPU__
#include <tests/main.h>

#include <fused_kernel/core/execution_model/stream.h>

#include <algorithm>
#include <iostream>
#include <memory>

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
#endif

inline NativeStream native(const GPUStream& stream) {
    return static_cast<NativeStream>(stream);
}

#if defined(__NVCC__)
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
#endif

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

#if defined(__NVCC__)
    // Owned streams are destroyed with their last copy, also when replaced by assignment (2000 leaked CUDA streams
    // would be over 300 MB)
    const double lost = megabytesLostToStreams();
    if (lost > 64.) {
        std::cout << "GPU streams: " << lost << " MB of device memory lost to streams that were not destroyed"
                  << std::endl;
        correct = false;
    }
#else
    // Not checked under HIP: ROCm maps the streams onto a few hardware queues and reuses their memory, so free device
    // memory does not show leaked streams (ROCm 10.1 on an RX 9060: 4000 streams that are never destroyed cost 0 MB)
#endif

    std::cout << "GPU stream ownership " << (correct ? "OK" : "FAILED") << std::endl;
    return correct ? 0 : -1;
}

} // namespace fk

int launch() {
    return fk::launch_impl();
}
