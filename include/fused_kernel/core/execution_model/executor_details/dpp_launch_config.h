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

#ifndef FK_DPP_LAUNCH_CONFIG_H
#define FK_DPP_LAUNCH_CONFIG_H

#include <cstddef>

namespace fk {

/**
 * Launch metadata contract for generic GPU DPP execution.
 *
 * A DPP using the primary Executor must expose:
 *   static DPPLaunchConfig launchConfig(const DPPDetails&);
 * CPU execution ignores this metadata and calls DPP::exec directly.
 */
struct DPPLaunchConfig {
    unsigned int gridX{1};
    unsigned int gridY{1};
    unsigned int gridZ{1};
    unsigned int blockX{1};
    unsigned int blockY{1};
    unsigned int blockZ{1};
    std::size_t sharedMemoryBytes{0};
};

} // namespace fk

#endif // FK_DPP_LAUNCH_CONFIG_H
