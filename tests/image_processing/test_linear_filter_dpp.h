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

#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/linear_filter.h>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace fk;

namespace {

using Details = LinearFilterDPPDetails<float, 16, 8, 7, 7>;
using MulIOp = decltype(Mul<float, float, float, UnaryType>::build());
using AddIOp = decltype(Add<float, float, float, UnaryType>::build());
using SubIOp = decltype(Sub<float, float, float, UnaryType>::build());

enum class Arithmetic { NORMAL, MUTATE_MUL_TO_ADD, MUTATE_ADD_TO_SUB };

float inputValue(const int x, const int y) {
    return static_cast<float>((x * 5 + y * 7) % 19 - 9) * 0.125f;
}

float coefficientValue(const int x, const int y) {
    return static_cast<float>((x * 3 + y * 2) % 11 - 5) * 0.0625f;
}

std::vector<float> makeInput(const int width, const int height) {
    std::vector<float> input(width * height);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            input[y * width + x] = inputValue(x, y);
    return input;
}

std::vector<float> makeKernel(const int width, const int height,
                              const bool box) {
    std::vector<float> kernel(width * height);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            kernel[y * width + x] = box
                ? 1.f / static_cast<float>(width * height)
                : coefficientValue(x, y);
    return kernel;
}

std::vector<float> oracle(const std::vector<float>& input,
                          const std::vector<float>& kernel,
                          const Details& details,
                          const bool fused,
                          const bool fusedKernel,
                          const Arithmetic arithmetic) {
    std::vector<float> output(details.width * details.height);
    for (int oy = 0; oy < details.height; ++oy) {
        for (int ox = 0; ox < details.width; ++ox) {
            float accumulator = 0.f;
            for (int ky = 0; ky < details.kernelHeight; ++ky) {
                for (int kx = 0; kx < details.kernelWidth; ++kx) {
                    int sx = ox + kx - details.anchorX;
                    int sy = oy + ky - details.anchorY;
                    sx = sx < 0 ? 0 : (sx >= details.width
                        ? details.width - 1 : sx);
                    sy = sy < 0 ? 0 : (sy >= details.height
                        ? details.height - 1 : sy);
                    float image = input[sy * details.width + sx];
                    float coeff = kernel[ky * details.kernelWidth + kx];
                    if (fused) image = image * 2.f + 1.f;
                    if (fusedKernel) coeff *= 0.5f;
                    const float product = arithmetic == Arithmetic::MUTATE_MUL_TO_ADD
                        ? image + coeff : image * coeff;
                    accumulator = arithmetic == Arithmetic::MUTATE_ADD_TO_SUB
                        ? accumulator - product : accumulator + product;
                }
            }
            output[oy * details.width + ox] =
                fused ? accumulator * 0.25f : accumulator;
        }
    }
    return output;
}

template <typename ImageRead, typename KernelRead,
          typename Multiply, typename Accumulate, typename Write>
bool runCpu(const Details& details, const ImageRead& image,
            const KernelRead& kernel, const Multiply& multiply,
            const Accumulate& accumulate, const Write& write) {
    LinearFilterDPP<ParArch::CPU, Details>::exec(
        details, make_tuple(image, kernel), multiply, accumulate, write);
    return true;
}

bool compare(const std::vector<float>& output,
             const std::vector<float>& expected,
             const char* label) {
    for (size_t i = 0; i < output.size(); ++i) {
        if (std::fabs(output[i] - expected[i]) > 2e-5f) {
            std::printf("%s index=%zu got=%g expected=%g\n",
                        label, i, output[i], expected[i]);
            return false;
        }
    }
    return true;
}

template <typename Multiply, typename Accumulate>
bool runCase(const int width, const int height,
             const int kernelWidth, const int kernelHeight,
             const int anchorX, const int anchorY,
             const bool box, const bool fused,
             const bool fusedKernel,
             const Arithmetic arithmetic,
             const Multiply& multiply,
             const Accumulate& accumulate,
             const char* label) {
    const Details details{width, height, kernelWidth, kernelHeight,
                          anchorX, anchorY};
    const auto input = makeInput(width, height);
    const auto kernel = makeKernel(kernelWidth, kernelHeight, box);
    const auto expected = oracle(input, kernel, details, fused,
                                 fusedKernel, arithmetic);
    std::vector<float> cpuOutput(width * height, -999.f);
    const RawPtr<ND::_2D, float> inputPtr{
        const_cast<float*>(input.data()),
        PtrDims<ND::_2D>(width, height, width * sizeof(float))};
    const RawPtr<ND::_2D, float> kernelPtr{
        const_cast<float*>(kernel.data()),
        PtrDims<ND::_2D>(kernelWidth, kernelHeight,
                         kernelWidth * sizeof(float))};
    const RawPtr<ND::_2D, float> outputPtr{
        cpuOutput.data(),
        PtrDims<ND::_2D>(width, height, width * sizeof(float))};
    const auto imageBase = PerThreadRead<ND::_2D, float>::build(inputPtr);
    const auto kernelBase = PerThreadRead<ND::_2D, float>::build(kernelPtr);
    const auto writeBase = PerThreadWrite<ND::_2D, float>::build(outputPtr);

    if (fused) {
        const auto imageRead = imageBase.then(Mul<float>::build(2.f))
                                        .then(Add<float>::build(1.f));
        const auto kernelRead =
            kernelBase.then(Mul<float>::build(0.5f));
        const auto write = Mul<float>::build(0.25f).then(writeBase);
        runCpu(details, imageRead, kernelRead, multiply, accumulate, write);
    } else if (fusedKernel) {
        const auto kernelRead = kernelBase.then(Mul<float>::build(0.5f));
        runCpu(details, imageBase, kernelRead, multiply, accumulate,
               writeBase);
    } else {
        runCpu(details, imageBase, kernelBase, multiply, accumulate,
               writeBase);
    }
    if (!compare(cpuOutput, expected, label)) return false;

#if defined(__NVCC__)
    Ptr2D<float> gpuInput(width, height);
    Ptr2D<float> gpuKernel(kernelWidth, kernelHeight);
    Ptr2D<float> gpuOutput(width, height);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            gpuInput.at(Point{x, y, 0}) = input[y * width + x];
    for (int y = 0; y < kernelHeight; ++y)
        for (int x = 0; x < kernelWidth; ++x)
            gpuKernel.at(Point{x, y, 0}) =
                kernel[y * kernelWidth + x];
    Stream stream;
    gpuInput.upload(stream);
    gpuKernel.upload(stream);
    const auto gpuImageBase =
        PerThreadRead<ND::_2D, float>::build(gpuInput);
    const auto gpuKernelBase =
        PerThreadRead<ND::_2D, float>::build(gpuKernel);
    const auto gpuWriteBase =
        PerThreadWrite<ND::_2D, float>::build(gpuOutput);
    bool launched = false;
    if (box && !fused && !fusedKernel &&
        arithmetic == Arithmetic::NORMAL) {
        launched = executeBoxFilter(details, gpuImageBase, gpuWriteBase,
                                    1.f, stream);
    } else if (fused) {
        const auto imageRead = gpuImageBase.then(Mul<float>::build(2.f))
                                           .then(Add<float>::build(1.f));
        const auto kernelRead =
            gpuKernelBase.then(Mul<float>::build(0.5f));
        const auto write = Mul<float>::build(0.25f).then(gpuWriteBase);
        launched = executeLinearFilter(
            details, make_tuple(imageRead, kernelRead),
            multiply, accumulate, write, stream);
    } else if (fusedKernel) {
        const auto kernelRead =
            gpuKernelBase.then(Mul<float>::build(0.5f));
        launched = executeLinearFilter(
            details, make_tuple(gpuImageBase, kernelRead),
            multiply, accumulate, gpuWriteBase, stream);
    } else {
        launched = executeLinearFilter(
            details, make_tuple(gpuImageBase, gpuKernelBase),
            multiply, accumulate, gpuWriteBase, stream);
    }
    if (!launched) return false;
    gpuOutput.download(stream);
    stream.sync();
    std::vector<float> result(width * height);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            result[y * width + x] = gpuOutput.at(Point{x, y, 0});
    if (!compare(result, expected, label)) return false;
#endif
    return true;
}

} // namespace

int launch() {
    const auto mul = Mul<float, float, float, UnaryType>::build();
    const auto add = Add<float, float, float, UnaryType>::build();
    const auto sub = Sub<float, float, float, UnaryType>::build();
    bool ok = true;
    ok = runCase(37, 19, 3, 3, 1, 1, true, false, false,
                 Arithmetic::NORMAL, mul, add, "box3") && ok;
    ok = runCase(35, 17, 5, 5, 2, 2, true, false, false,
                 Arithmetic::NORMAL, mul, add, "box5") && ok;
    ok = runCase(33, 21, 7, 3, 5, 1, false, true, true,
                 Arithmetic::NORMAL, mul, add, "anisotropic-fused") && ok;
    ok = runCase(19, 13, 3, 3, 0, 2, false, false, false,
                 Arithmetic::MUTATE_MUL_TO_ADD, add, add,
                 "mutate-mul") && ok;
    ok = runCase(21, 15, 3, 5, 1, 3, false, false, false,
                 Arithmetic::MUTATE_ADD_TO_SUB, mul, sub,
                 "mutate-add") && ok;

    const Details invalid{16, 16, 8, 3, 1, 1};
    if (Details::valid(invalid)) ok = false;
    if (ok) std::printf("LinearFilterDPP contracts: PASS\n");
    return ok ? 0 : -1;
}
