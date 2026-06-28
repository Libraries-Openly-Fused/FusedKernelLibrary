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

#define __ONLY_CU__ // This file is only generated and compiled with nvcc, not with the host compiler
#include <tests/main.h>

#include <fused_kernel/algorithms/attention/softmax.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using namespace fk;

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
    const auto inIOp = PerThreadRead<ND::_2D, float>::build(input.ptr());
    const auto outIOp = PerThreadWrite<ND::_2D, float>::build(output.ptr());
    executeSoftmax<256>(inIOp, outIOp, stream);
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

// PROLOGUE: input enters through a Read IOp fused with a compute chain.
// softmax(2*x + 1) == softmax(2*x) (row-constant shift cancels), and
// softmax over 2*x differs from softmax over x -> both effects verified.
static void runPrologueCase(const char* name, const int width, const int height,
                            const double tol, const unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-4.f, 4.f);

    Ptr2D<float> input(width, height, 0, MemType::Device);
    Ptr2D<float> output(width, height, 0, MemType::Device);

    std::vector<float> host((size_t)width * height);
    for (auto& v : host) v = dist(rng);
    gpuErrchk(cudaMemcpy2D(input.ptr().data, input.ptr().dims.pitch,
                           host.data(), width * sizeof(float),
                           width * sizeof(float), height, cudaMemcpyHostToDevice));

    Stream stream;
    const auto inIOp = PerThreadRead<ND::_2D, float>::build(input.ptr())
                           .then(Mul<float>::build(2.f))
                           .then(Add<float>::build(1.f));
    const auto outIOp = PerThreadWrite<ND::_2D, float>::build(output.ptr());
    executeSoftmax<256>(inIOp, outIOp, stream);
    stream.sync();

    std::vector<float> got((size_t)width * height);
    gpuErrchk(cudaMemcpy2D(got.data(), width * sizeof(float),
                           output.ptr().data, output.ptr().dims.pitch,
                           width * sizeof(float), height, cudaMemcpyDeviceToHost));

    double maxErr = 0.0;
    for (int y = 0; y < height; ++y) {
        double m = -1e300, l = 0.0;
        for (int x = 0; x < width; ++x) m = std::max(m, 2.0 * host[(size_t)y*width+x] + 1.0);
        for (int x = 0; x < width; ++x) l += std::exp(2.0 * host[(size_t)y*width+x] + 1.0 - m);
        for (int x = 0; x < width; ++x) {
            const double ref = std::exp(2.0 * host[(size_t)y*width+x] + 1.0 - m) / l;
            maxErr = std::max(maxErr, std::abs((double)got[(size_t)y*width+x] - ref));
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
    runPrologueCase("Softmax prologue ReadIOp.then(Mul(2)).then(Add(1)) 100x6", 100, 6, 1e-6, 5);
    return failures == 0 ? 0 : -1;
}
