/* Copyright 2026 Oscar Amoros Huguet, Johnny Nunez

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

#include <fused_kernel/algorithms/attention/softmax.h>

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using namespace fk;

#if defined(__NVCC__) || CLANG_HOST_DEVICE

static int failures = 0;

static void runCase(const char* name, const int width, const int height,
                    const float lo, const float hi, const double tol,
                    const unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);

    Ptr2D<float> input(width, height, 0, MemType::Device);
    Ptr2D<float> output(width, height, 0, MemType::Device);

    std::vector<float> host((size_t)width * height);
    for (auto& v : host) v = dist(rng);
    gpuErrchk(cudaMemcpy2D(input.ptr().data, input.ptr().dims.pitch,
                           host.data(), width * sizeof(float),
                           width * sizeof(float), height, cudaMemcpyHostToDevice));

    Stream stream;
    executeSoftmax<float>(input, output, stream);
    stream.sync();

    std::vector<float> got((size_t)width * height);
    gpuErrchk(cudaMemcpy2D(got.data(), width * sizeof(float),
                           output.ptr().data, output.ptr().dims.pitch,
                           width * sizeof(float), height, cudaMemcpyDeviceToHost));

    double maxErr = 0.0;
    for (int y = 0; y < height; ++y) {
        double m = -1e300, l = 0.0, rowSum = 0.0;
        for (int x = 0; x < width; ++x) m = std::max(m, (double)host[(size_t)y*width+x]);
        for (int x = 0; x < width; ++x) l += std::exp((double)host[(size_t)y*width+x] - m);
        for (int x = 0; x < width; ++x) {
            const double ref = std::exp((double)host[(size_t)y*width+x] - m) / l;
            maxErr = std::max(maxErr, std::abs((double)got[(size_t)y*width+x] - ref));
            rowSum += got[(size_t)y*width+x];
        }
        if (std::abs(rowSum - 1.0) > 1e-2) {
            std::cout << "FAIL " << name << ": row " << y << " sums to " << rowSum << std::endl;
            ++failures;
            return;
        }
    }
    if (maxErr > tol) {
        std::cout << "FAIL " << name << ": maxErr=" << maxErr << std::endl;
        ++failures;
    } else {
        std::cout << "Running test " << name << ": Success!! (maxErr=" << maxErr << ")" << std::endl;
    }
}

int launch() {
    runCase("Softmax f32 7x3", 7, 3, -4.f, 4.f, 1e-6, 1);
    runCase("Softmax f32 256x16 (block-sized)", 256, 16, -8.f, 8.f, 1e-6, 2);
    runCase("Softmax f32 1000x8 (strided non-pow2)", 1000, 8, -8.f, 8.f, 1e-6, 3);
    runCase("Softmax f32 stability |x|<=500", 333, 5, -500.f, 500.f, 1e-6, 4);
    return failures == 0 ? 0 : -1;
}

#else
int launch() {
    std::cout << "Softmax DPP tests skipped (CUDA-only DPP)" << std::endl;
    return 0;
}
#endif
