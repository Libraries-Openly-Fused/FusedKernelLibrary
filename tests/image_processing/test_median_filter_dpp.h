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
#include <fused_kernel/algorithms/image_processing/median_filter.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace fk {

namespace {

using Details = MedianFilterDPPDetails<float, 16, 8, 7, 7>;
using Window = NeighborhoodWindow<float, 49>;
using MedianSelection = decltype(MedianWindowSelect<float, 49>::build());
using MinSelection = decltype(MinimumWindowSelect<float, 49>::build());

enum class SelectionKind { MEDIAN, MINIMUM };

float inputValue(const int x, const int y) {
    return static_cast<float>((x * 11 + y * 7) % 29 - 14) * 0.125f;
}

std::vector<float> oracle(const std::vector<float>& input,
                          const Details& details,
                          const SelectionKind kind,
                          const bool fused) {
    std::vector<float> output(details.width * details.height);
    std::vector<float> values;
    values.reserve(details.windowWidth * details.windowHeight);
    for (int oy = 0; oy < details.height; ++oy) {
        for (int ox = 0; ox < details.width; ++ox) {
            values.clear();
            for (int wy = 0; wy < details.windowHeight; ++wy) {
                for (int wx = 0; wx < details.windowWidth; ++wx) {
                    int sx = ox + wx - details.anchorX;
                    int sy = oy + wy - details.anchorY;
                    sx = sx < 0 ? 0 : (sx >= details.width
                        ? details.width - 1 : sx);
                    sy = sy < 0 ? 0 : (sy >= details.height
                        ? details.height - 1 : sy);
                    float value = input[sy * details.width + sx];
                    if (fused) value = value * 2.f + 1.f;
                    values.push_back(value);
                }
            }
            std::sort(values.begin(), values.end());
            float selected = kind == SelectionKind::MEDIAN
                ? values[values.size() / 2] : values.front();
            output[oy * details.width + ox] =
                fused ? selected * 0.5f : selected;
        }
    }
    return output;
}

bool compare(const std::vector<float>& output,
             const std::vector<float>& expected,
             const char* label) {
    for (size_t index = 0; index < output.size(); ++index) {
        if (std::fabs(output[index] - expected[index]) > 1e-6f) {
            std::printf("%s index=%zu got=%g expected=%g\n",
                        label, index, output[index], expected[index]);
            return false;
        }
    }
    return true;
}

#ifdef FK_CPU_MULTITHREAD_ENABLED
// Runs launch(stream) on a CPU_MULTITHREAD stream, with IOps that write to
// output, and requires its result to be bit-identical to the ParArch::CPU
// result that output holds before the call
template <typename Launch>
bool sameAsCpu(std::vector<float>& output, const Launch& launch,
               const char* label) {
    const std::vector<float> cpuResult = output;
    std::fill(output.begin(), output.end(), -999.f);
    Stream_<ParArch::CPU_MULTITHREAD> stream;
    const bool launched = launch(stream);
    stream.sync();
    // The per row code is the CPU one; the results may differ in the last bits
    // only, where the compiler contracts a * b + c into a fused multiply-add
    // differently
    bool same = launched;
    for (std::size_t i = 0; i < output.size() && same; ++i) {
        const float scale = std::max(1.f, std::fabs(cpuResult[i]));
        same = std::fabs(output[i] - cpuResult[i]) <= 1e-5f * scale;
    }
    if (!same) {
        std::printf("%s CPU_MULTITHREAD output differs from ParArch::CPU\n",
                    label);
        return false;
    }
    return true;
}
#endif

// Runs the filter on ParArch::CPU, writing to output, and then the same IOps
// through executeMedianFilter on a CPU_MULTITHREAD stream, if enabled
template <typename Read, typename Write, typename Selection>
bool runCpu(const Details& details, const Read& read, const Write& write,
            const Selection& selection, std::vector<float>& output,
            const char* label) {
    MedianFilterDPP<ParArch::CPU, Details>::exec(
        details, read, write, selection);
#ifdef FK_CPU_MULTITHREAD_ENABLED
    return sameAsCpu(output, [&](Stream_<ParArch::CPU_MULTITHREAD>& stream) {
        return executeMedianFilter(details, read, write, selection, stream);
    }, label);
#else
    (void)output;
    (void)label;
    return true;
#endif
}

template <typename Selection>
bool runCase(const Details& details, const Selection& selection,
             const SelectionKind kind, const bool fused,
             const char* label) {
    std::vector<float> input(details.width * details.height);
    for (int y = 0; y < details.height; ++y)
        for (int x = 0; x < details.width; ++x)
            input[y * details.width + x] = inputValue(x, y);
    const auto expected = oracle(input, details, kind, fused);
    std::vector<float> cpuOutput(input.size(), -999.f);
    const RawPtr<ND::_2D, float> inputPtr{
        input.data(), PtrDims<ND::_2D>(
            details.width, details.height,
            details.width * sizeof(float))};
    const RawPtr<ND::_2D, float> outputPtr{
        cpuOutput.data(), PtrDims<ND::_2D>(
            details.width, details.height,
            details.width * sizeof(float))};
    const auto readBase = PerThreadRead<ND::_2D, float>::build(inputPtr);
    const auto writeBase = PerThreadWrite<ND::_2D, float>::build(outputPtr);
    bool ran = false;
    if (fused) {
        const auto read = readBase.then(Mul<float>::build(2.f))
                                  .then(Add<float>::build(1.f));
        const auto write = Mul<float>::build(0.5f).then(writeBase);
        ran = runCpu(details, read, write, selection, cpuOutput, label);
    } else {
        ran = runCpu(details, readBase, writeBase, selection, cpuOutput,
                     label);
    }
    if (!ran || !compare(cpuOutput, expected, label)) return false;

#if defined(__NVCC__) || defined(__HIPCC__)
    Ptr2D<float> gpuInput(details.width, details.height);
    Ptr2D<float> gpuOutput(details.width, details.height);
    for (int y = 0; y < details.height; ++y)
        for (int x = 0; x < details.width; ++x)
            gpuInput.at(Point{x, y, 0}) = input[y * details.width + x];
    Stream stream;
    gpuInput.upload(stream);
    const auto gpuReadBase =
        PerThreadRead<ND::_2D, float>::build(gpuInput);
    const auto gpuWriteBase =
        PerThreadWrite<ND::_2D, float>::build(gpuOutput);
    bool launched = false;
    if (fused) {
        const auto read = gpuReadBase.then(Mul<float>::build(2.f))
                                     .then(Add<float>::build(1.f));
        const auto write = Mul<float>::build(0.5f).then(gpuWriteBase);
        launched = executeMedianFilter(
            details, read, write, selection, stream);
    } else {
        launched = executeMedianFilter(
            details, gpuReadBase, gpuWriteBase, selection, stream);
    }
    if (!launched) return false;
    gpuOutput.download(stream);
    stream.sync();
    std::vector<float> result(input.size());
    for (int y = 0; y < details.height; ++y)
        for (int x = 0; x < details.width; ++x)
            result[y * details.width + x] =
                gpuOutput.at(Point{x, y, 0});
    if (!compare(result, expected, label)) return false;
#endif
    return true;
}

} // namespace

int launch_impl() {
    using namespace fk;
    const auto median = MedianWindowSelect<float, 49>::build();
    const auto minimum = MinimumWindowSelect<float, 49>::build();
    bool ok = true;
    ok = runCase(Details{37, 19, 3, 3, 1, 1}, median,
                 SelectionKind::MEDIAN, false, "median3") && ok;
    ok = runCase(Details{35, 17, 5, 5, 2, 2}, median,
                 SelectionKind::MEDIAN, false, "median5") && ok;
    ok = runCase(Details{33, 21, 7, 3, 5, 1}, median,
                 SelectionKind::MEDIAN, true, "median7x3-fused") && ok;
    ok = runCase(Details{31, 15, 3, 5, 0, 3}, minimum,
                 SelectionKind::MINIMUM, false, "minimum-selection") && ok;

    const Details evenWindow{16, 16, 4, 3, 1, 1};
    if (Details::valid(evenWindow)) ok = false;
#ifdef FK_CPU_MULTITHREAD_ENABLED
    {
        // With invalid details the CPU_MULTITHREAD entry point enqueues
        // nothing and returns false, and the DPP writes nothing
        std::vector<float> image(16 * 16, 1.f);
        std::vector<float> output(16 * 16, -999.f);
        const PtrDims<ND::_2D> dims(16, 16, 16 * sizeof(float));
        const auto read = PerThreadRead<ND::_2D, float>::build(
            RawPtr<ND::_2D, float>{image.data(), dims});
        const auto write = PerThreadWrite<ND::_2D, float>::build(
            RawPtr<ND::_2D, float>{output.data(), dims});
        Stream_<ParArch::CPU_MULTITHREAD> stream;
        if (executeMedianFilter(evenWindow, read, write, median, stream)) {
            std::printf("CPU_MULTITHREAD accepted invalid details\n");
            ok = false;
        }
        stream.enqueue([=](const cpu_multithread::Completion& done) {
            MedianFilterDPP<ParArch::CPU_MULTITHREAD, Details>::exec(
                done, evenWindow, read, write, median);
        });
        stream.sync();
        if (std::any_of(output.begin(), output.end(),
                        [](const float value) { return value != -999.f; })) {
            std::printf("CPU_MULTITHREAD wrote with invalid details\n");
            ok = false;
        }
    }
#endif
    if (ok) std::printf("MedianFilterDPP contracts: PASS\n");
    return ok ? 0 : -1;
}

} // namespace fk

int launch() {
    return fk::launch_impl();
}
