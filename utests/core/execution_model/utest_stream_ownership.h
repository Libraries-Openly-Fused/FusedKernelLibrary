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

// GPU streams own the CUDA stream they create: copies share it, assignment adopts the other stream, and the last copy
// destroys it (never before: a copy that outlives the original keeps it usable). A stream given by the user is never
// destroyed.
#define __ONLY_CU__

#include <tests/main.h>

#include <fused_kernel/core/execution_model/stream.h>

#include <algorithm>
#include <iostream>
#include <memory>

#if defined(__NVCC__)
namespace fk {

using GPUStream = Stream_<ParArch::GPU_NVIDIA>;

// Free device memory lost after creating, copying, assigning and destroying many streams. Every created stream holds
// device memory, so leaking them shows here. Other processes can use the GPU at the same time: the smallest of three
// rounds is reported, which they can only make smaller.
double megabytesLostToStreams() {
    double smallest = 1e30;
    for (int round = 0; round < 3; ++round) {
        size_t freeBefore = 0, freeAfter = 0, total = 0;
        gpuErrchk(cudaMemGetInfo(&freeBefore, &total));
        for (int i = 0; i < 1000; ++i) {
            GPUStream a;
            GPUStream b;
            GPUStream c = b;
            b = a;
            c = a;
        }
        gpuErrchk(cudaMemGetInfo(&freeAfter, &total));
        smallest = std::min(smallest, (static_cast<double>(freeBefore) - static_cast<double>(freeAfter)) / 1048576.0);
    }
    return smallest;
}

int launch_impl() {
    gpuErrchk(cudaFree(0));
    bool correct = true;

    // Copies and assignment share the CUDA stream
    GPUStream a;
    GPUStream b;
    const GPUStream copy = a;
    correct &= copy.getCUDAStream() == a.getCUDAStream() && b.getCUDAStream() != a.getCUDAStream();
    b = a;
    correct &= b.getCUDAStream() == a.getCUDAStream();

    // The stream stays usable while any copy is alive: a copy that outlives the original, and a copy that shares it
    // with an object that is assigned another stream
    std::unique_ptr<GPUStream> survivor;
    {
        GPUStream original;
        survivor = std::make_unique<GPUStream>(original);
    }
    correct &= cudaStreamQuery(survivor->getCUDAStream()) == cudaSuccess;
    correct &= cudaStreamSynchronize(survivor->getCUDAStream()) == cudaSuccess;
    {
        GPUStream shared;
        GPUStream sharer = shared;
        shared = a;
        correct &= cudaStreamQuery(sharer.getCUDAStream()) == cudaSuccess && sharer.getCUDAStream() != a.getCUDAStream();
    }

    // A stream given by the user is never destroyed by its wrappers
    cudaStream_t userStream;
    gpuErrchk(cudaStreamCreate(&userStream));
    {
        GPUStream wrapper(userStream);
        GPUStream wrapperCopy = wrapper;
        GPUStream other;
        other = wrapperCopy;
        correct &= other.getCUDAStream() == userStream;
    }
    correct &= cudaStreamQuery(userStream) == cudaSuccess;
    gpuErrchk(cudaStreamDestroy(userStream));

    // Owned streams are destroyed with their last copy, also when replaced by assignment (each leaked stream holds
    // about 160 KB of device memory: 2000 of them would be over 300 MB)
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
#else
int launch() {
    return 0;
}
#endif
