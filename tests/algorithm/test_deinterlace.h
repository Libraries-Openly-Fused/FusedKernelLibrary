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

#include <tests/main.h>

#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/deinterlace.h>

int launch() {

    constexpr auto readIOp = fk::PerThreadRead<fk::ND::_2D, uchar3>::build(
        fk::RawPtr<fk::ND::_2D, uchar3>{ nullptr, { 128, 128, 128 * sizeof(uchar3) }});

    // Test BLEND deinterlacing
    constexpr auto deinterlaceBlendIOp = fk::Deinterlace<fk::DeinterlaceType::BLEND>::build(readIOp);

    static_assert(std::is_same_v<std::decay_t<decltype(deinterlaceBlendIOp)>,
        fk::ReadBack<fk::Deinterlace<fk::DeinterlaceType::BLEND, fk::Read<fk::PerThreadRead<fk::ND::_2D, uchar3>>>>>,
        "Unexpected type for deinterlaceBlendIOp");

    // Test INTER_LINEAR deinterlacing
    constexpr auto deinterlaceInterLinearIOp = fk::Deinterlace<fk::DeinterlaceType::INTER_LINEAR>::build(fk::DeinterlaceLinear::USE_EVEN, readIOp);

    static_assert(std::is_same_v<std::decay_t<decltype(deinterlaceInterLinearIOp)>,
        fk::ReadBack<fk::Deinterlace<fk::DeinterlaceType::INTER_LINEAR, fk::Read<fk::PerThreadRead<fk::ND::_2D, uchar3>>>>>,
        "Unexpected type for deinterlaceInterLinearIOp");

    // Test that both deinterlace types are different template instantiations
    static_assert(!std::is_same_v<decltype(deinterlaceBlendIOp), decltype(deinterlaceInterLinearIOp)>,
        "BLEND and INTER_LINEAR should be different types");

    // Test enum values
    static_assert(fk::DeinterlaceType::BLEND != fk::DeinterlaceType::INTER_LINEAR,
        "DeinterlaceType enum values should be different");

    return 0;
}