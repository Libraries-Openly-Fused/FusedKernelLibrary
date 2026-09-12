/* Copyright 2025-2026 Oscar Amoros Huguet

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

namespace fk {



int launch_impl() {
    constexpr auto readIOp = PerThreadRead<ND::_2D, uchar3>::build(
        RawPtr<ND::_2D, uchar3>{ nullptr, { 128, 128, 128 * sizeof(uchar3) }});

    // Test BLEND deinterlacing
    constexpr auto deinterlaceBlendIOp = Deinterlace<DeinterlaceType::BLEND>::build(readIOp);

    static_assert(std::is_same_v<std::decay_t<decltype(deinterlaceBlendIOp)>,
        ReadBack<Deinterlace<DeinterlaceType::BLEND, Read<PerThreadRead<ND::_2D, uchar3>>>>>,
        "Unexpected type for deinterlaceBlendIOp");

    // Test INTER_LINEAR deinterlacing
    constexpr auto deinterlaceInterLinearIOp = Deinterlace<DeinterlaceType::INTER_LINEAR>::build(DeinterlaceLinear::USE_EVEN, readIOp);

    static_assert(std::is_same_v<std::decay_t<decltype(deinterlaceInterLinearIOp)>,
        ReadBack<Deinterlace<DeinterlaceType::INTER_LINEAR, Read<PerThreadRead<ND::_2D, uchar3>>>>>,
        "Unexpected type for deinterlaceInterLinearIOp");

    // Test that both deinterlace types are different template instantiations
    static_assert(!std::is_same_v<decltype(deinterlaceBlendIOp), decltype(deinterlaceInterLinearIOp)>,
        "BLEND and INTER_LINEAR should be different types");

    // Test enum values
    static_assert(DeinterlaceType::BLEND != DeinterlaceType::INTER_LINEAR,
        "DeinterlaceType enum values should be different");

    return 0;
}

} // namespace fk

int launch() {
    return fk::launch_impl();
}
