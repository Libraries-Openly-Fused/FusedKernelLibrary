/* Copyright 2023-2025 Oscar Amoros Huguet

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

#include <fused_kernel/core/execution_model/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/resize.h>
#include <fused_kernel/core/data/array.h>
#include <array>

using namespace fk;

int launch() {
    constexpr size_t BATCH = 20;
    constexpr RawPtr<ND::_2D, float> data{ nullptr,{16,16,16} };
    constexpr std::array<RawPtr<ND::_2D, float>, BATCH> inputs = make_set_std_array<BATCH>(data);
    constexpr Size oneSize(8,8);
    constexpr std::array<Size, BATCH> resParams = make_set_std_array<BATCH>(oneSize);

    constexpr float defaultValue = 0;
    constexpr std::array<float, BATCH> defaultArray = make_set_std_array<BATCH>(defaultValue);

    constexpr auto readDFArray = PerThreadRead<ND::_2D, float>::build_batch(inputs);

    constexpr auto oneResizeread = Resize<InterpolationType::INTER_LINEAR>::build(readDFArray[0], resParams[0]);
    static_assert(!isBatchOperation<std::decay_t<decltype(oneResizeread)>>, "oneResize is BatchResize, and should not be");

    constexpr auto resizeDFArray = Resize<InterpolationType::INTER_LINEAR>::build(readDFArray, resParams);
    static_assert(decltype(resizeDFArray)::Operation::BATCH == BATCH, "resizeDFArray does not have the correct BATCH size");
    const auto resizeDFArray2 = Resize<InterpolationType::INTER_LINEAR, AspectRatio::PRESERVE_AR>::build(readDFArray, resParams, defaultArray);
    static_assert(decltype(resizeDFArray2)::Operation::BATCH == BATCH, "resizeDFArray2 does not have the correct BATCH size");

    return 0;
}
