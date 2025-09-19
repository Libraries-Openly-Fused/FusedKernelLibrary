/* Copyright 2025 Grup Mediapro S.L.U (Oscar Amoros Huguet)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_TEST_VBASE_H
#define FK_TEST_VBASE_H

#include <fused_kernel/core/utils/vector_utils.h>

template <typename InputTypeList, typename ExpectedTypeList, size_t... Idx>
constexpr bool validateVBaseFor(const std::index_sequence<Idx...>&) {
    return (std::is_same_v<fk::EquivalentType_t<fk::TypeAt_t<Idx, InputTypeList>, InputTypeList, ExpectedTypeList>, fk::VBase<fk::TypeAt_t<Idx, InputTypeList>>> && ...);
}

template <size_t First, size_t... Rest>
constexpr bool allEqual = ((First == Rest) && ...);

int launch() {

    static_assert(allEqual<fk::VOne::size, fk::VTwo::size, fk::VThree::size, fk::VFour::size, fk::BaseTypes::size>, "Those TypeLists must be all equal.");
    constexpr auto idxSeq = std::make_index_sequence<fk::BaseTypes::size>{};
    static_assert(validateVBaseFor<fk::BaseTypes, fk::BaseTypes>(idxSeq), "Error in VBase with fundamental types");
    static_assert(validateVBaseFor<fk::VOne, fk::BaseTypes>(idxSeq), "Error in VBase with cuda vector types of one channel");
    static_assert(validateVBaseFor<fk::VTwo, fk::BaseTypes>(idxSeq), "Error in VBase with cuda vector types of two channels");
    static_assert(validateVBaseFor<fk::VThree, fk::BaseTypes>(idxSeq), "Error in VBase with cuda vector types of three channels");
    static_assert(validateVBaseFor<fk::VFour, fk::BaseTypes>(idxSeq), "Error in VBase with cuda vector types of four channels");

    return 0;
}

#endif