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
#include <fused_kernel/algorithms/basic_ops/logical.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/morphology.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <type_traits>
#include <vector>

using namespace fk;

namespace {

using Details = MorphologyDPPDetails<float, 16, 8, 7, 7>;
using MinReducer = Min<float, float, float, UnaryType>;
using MaxReducer = Max<float, float, float, UnaryType>;
using MinIOp = decltype(MinReducer::build());
using MaxIOp = decltype(MaxReducer::build());

static_assert(MorphologyDPP<ParArch::CPU, Details>::
              template validReducerCount<Tuple<MinIOp>>);
static_assert(!MorphologyDPP<ParArch::CPU, Details>::
              template validReducerCount<Tuple<>>);
static_assert(!MorphologyDPP<ParArch::CPU, Details>::
              template validReducerCount<
                  Tuple<MinIOp, MaxIOp, MinIOp, MaxIOp, MinIOp>>);
static_assert(Details::MAX_MASK_WIDTH == 7 && Details::MAX_MASK_HEIGHT == 7);

float inputValue(const int x, const int y) {
    return static_cast<float>((x * 17 + y * 29) % 101 - 50) * 0.125f;
}

std::vector<float> oracle(const std::vector<float>& original,
                          const int width, const int height,
                          const int maskW, const int maskH,
                          const int anchorX, const int anchorY,
                          const std::vector<bool>& erodePasses) {
    std::vector<float> current(original.size());
    for (size_t i = 0; i < original.size(); ++i)
        current[i] = original[i] * 2.f + 1.f;
    std::vector<float> next(original.size());
    for (const bool erode : erodePasses) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                bool first = true;
                float value = 0.f;
                for (int my = 0; my < maskH; ++my) {
                    const int sy = std::clamp(y + my - anchorY, 0, height - 1);
                    for (int mx = 0; mx < maskW; ++mx) {
                        const int sx = std::clamp(x + mx - anchorX, 0, width - 1);
                        const float sample = current[sy * width + sx];
                        if (first) {
                            value = sample;
                            first = false;
                        } else {
                            value = erode ? std::min(value, sample)
                                          : std::max(value, sample);
                        }
                    }
                }
                next[y * width + x] = value;
            }
        }
        current.swap(next);
    }
    for (float& value : current) value *= 0.5f;
    return current;
}

template <typename Reducers>
bool runCpuCase(const char* name, const Details& details,
                const Reducers& reducers,
                const std::vector<bool>& erodePasses) {
    std::vector<float> input(details.width * details.height);
    std::vector<float> output(input.size(), -777.f);
    for (int y = 0; y < details.height; ++y)
        for (int x = 0; x < details.width; ++x)
            input[y * details.width + x] = inputValue(x, y);

    const RawPtr<ND::_2D, float> inputPtr{
        input.data(), PtrDims<ND::_2D>(details.width, details.height,
                                      details.width * sizeof(float))};
    const RawPtr<ND::_2D, float> outputPtr{
        output.data(), PtrDims<ND::_2D>(details.width, details.height,
                                       details.width * sizeof(float))};
    const auto read = PerThreadRead<ND::_2D, float>::build(inputPtr)
        .then(Mul<float>::build(2.f))
        .then(Add<float>::build(1.f));
    const auto write = Mul<float>::build(0.5f)
        .then(PerThreadWrite<ND::_2D, float>::build(outputPtr));

    MorphologyDPP<ParArch::CPU, Details>::exec(
        details, read, write, reducers);
    const auto expected = oracle(
        input, details.width, details.height,
        details.maskW, details.maskH, details.anchorX, details.anchorY,
        erodePasses);
    for (size_t i = 0; i < output.size(); ++i) {
        if (std::fabs(output[i] - expected[i]) > 1e-6f) {
            std::printf("CPU %s index=%zu got=%g expected=%g\n",
                        name, i, output[i], expected[i]);
            return false;
        }
    }
    return true;
}

#if defined(__NVCC__)
template <typename Reducers>
bool runGpuCase(const char* name, const Details& details,
                const Reducers& reducers,
                const std::vector<bool>& erodePasses) {
    std::vector<float> hostInput(details.width * details.height);
    for (int y = 0; y < details.height; ++y)
        for (int x = 0; x < details.width; ++x)
            hostInput[y * details.width + x] = inputValue(x, y);

    Ptr2D<float> input(details.width, details.height);
    Ptr2D<float> output(details.width, details.height);
    for (int y = 0; y < details.height; ++y)
        for (int x = 0; x < details.width; ++x)
            input.at(Point{x, y, 0}) = hostInput[y * details.width + x];

    Stream stream;
    input.upload(stream);
    const auto read = PerThreadRead<ND::_2D, float>::build(input)
        .then(Mul<float>::build(2.f))
        .then(Add<float>::build(1.f));
    const auto write = Mul<float>::build(0.5f)
        .then(PerThreadWrite<ND::_2D, float>::build(output));
    bool launched = false;
    if constexpr (std::is_same_v<Reducers, Tuple<MinIOp>>) {
        launched = executeErode(details, read, write, stream);
    } else if constexpr (std::is_same_v<Reducers, Tuple<MaxIOp>>) {
        launched = executeDilate(details, read, write, stream);
    } else if constexpr (std::is_same_v<Reducers,
                                        Tuple<MinIOp, MaxIOp>>) {
        launched = executeOpen(details, read, write, stream);
    } else if constexpr (std::is_same_v<Reducers,
                                        Tuple<MaxIOp, MinIOp>>) {
        launched = executeClose(details, read, write, stream);
    } else {
        launched = executeMorphology(details, read, write, reducers, stream);
    }
    if (!launched) {
        std::printf("GPU %s launcher rejected valid details\n", name);
        return false;
    }
    output.download(stream);
    stream.sync();

    const auto expected = oracle(
        hostInput, details.width, details.height,
        details.maskW, details.maskH, details.anchorX, details.anchorY,
        erodePasses);
    for (int y = 0; y < details.height; ++y) {
        for (int x = 0; x < details.width; ++x) {
            const size_t i = static_cast<size_t>(y * details.width + x);
            const float got = output.at(Point{x, y, 0});
            if (std::fabs(got - expected[i]) > 1e-6f) {
                std::printf("GPU %s (%d,%d) got=%g expected=%g\n",
                            name, x, y, got, expected[i]);
                return false;
            }
        }
    }
    return true;
}
#endif

template <typename Reducers>
bool runCase(const char* name, const Details& details,
             const Reducers& reducers,
             const std::vector<bool>& erodePasses) {
    bool ok = runCpuCase(name, details, reducers, erodePasses);
#if defined(__NVCC__)
    ok = runGpuCase(name, details, reducers, erodePasses) && ok;
#endif
    return ok;
}

} // namespace

int launch() {
    const auto minOp = MinReducer::build();
    const auto maxOp = MaxReducer::build();
    bool ok = true;
    ok = runCase("erode-3x3", Details{37, 19, 3, 3, 1, 1},
                 make_tuple(minOp), {true}) && ok;
    ok = runCase("dilate-5x5", Details{35, 21, 5, 5, 2, 2},
                 make_tuple(maxOp), {false}) && ok;
    ok = runCase("erode-7x3", Details{39, 17, 7, 3, 3, 1},
                 make_tuple(minOp), {true}) && ok;
    ok = runCase("open", Details{31, 18, 3, 3, 1, 1},
                 make_tuple(minOp, maxOp), {true, false}) && ok;
    ok = runCase("close", Details{33, 15, 5, 3, 2, 1},
                 make_tuple(maxOp, minOp), {false, true}) && ok;
    ok = runCase("three-pass", Details{29, 17, 3, 5, 1, 2},
                 make_tuple(minOp, maxOp, minOp),
                 {true, false, true}) && ok;
    ok = runCase("four-pass", Details{27, 13, 3, 3, 1, 1},
                 make_tuple(maxOp, minOp, minOp, maxOp),
                 {false, true, true, false}) && ok;

    if (Details::valid(Details{10, 10, 9, 3, 4, 1}) ||
        Details::valid(Details{10, 10, 3, 3, 3, 1}) ||
        Details::valid(Details{0, 10, 3, 3, 1, 1})) {
        std::printf("invalid morphology details accepted\n");
        ok = false;
    }
    if (ok) std::printf("MorphologyDPP 1-4 pass contracts: PASS\n");
    return ok ? 0 : -1;
}
