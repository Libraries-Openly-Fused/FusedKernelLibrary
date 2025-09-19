/* Copyright 2023-2025 Oscar Amoros Huguet
   Copyright 2025 Grup Mediapro S.L.U (Oscar Amoros Huguet)

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

#include <fused_kernel/core/utils/utils.h>
#include <fused_kernel/algorithms/basic_ops/vector_ops.h>
#include <fused_kernel/algorithms/basic_ops/logical.h>
#include <fused_kernel/core/execution_model/thread_fusion.h>
#include "tests/nvtx.h"

template <typename OriginalType>
bool testThreadFusion() {
    constexpr OriginalType eightNumbers[8]{ static_cast<OriginalType>(10),
                                            static_cast<OriginalType>(2),
                                            static_cast<OriginalType>(3),
                                            static_cast<OriginalType>(4),
                                            static_cast<OriginalType>(10),
                                            static_cast<OriginalType>(2),
                                            static_cast<OriginalType>(3),
                                            static_cast<OriginalType>(4) };

    using BTInfo = fk::ThreadFusionInfo<OriginalType, OriginalType, true>;

    const typename BTInfo::BiggerReadType biggerType = ((typename BTInfo::BiggerReadType*) eightNumbers)[0];

    if constexpr (BTInfo::elems_per_thread == 1) {
        return eightNumbers[0] == biggerType;
    } else if constexpr (BTInfo::elems_per_thread == 2) {
        const OriginalType data0 = BTInfo::template get<0>(biggerType);
        const OriginalType data1 = BTInfo::template get<1>(biggerType);
        return (data0 == eightNumbers[0]) && (data1 == eightNumbers[1]);
    } else if constexpr (BTInfo::elems_per_thread == 4) {
        const OriginalType data0 = BTInfo::template get<0>(biggerType);
        const OriginalType data1 = BTInfo::template get<1>(biggerType);
        const OriginalType data2 = BTInfo::template get<2>(biggerType);
        const OriginalType data3 = BTInfo::template get<3>(biggerType);
        return data0 == eightNumbers[0] && data1 == eightNumbers[1] &&
            data2 == eightNumbers[2] && data3 == eightNumbers[3];
    } else if constexpr (BTInfo::elems_per_thread == 8) {
        const OriginalType data0 = BTInfo::template get<0>(biggerType);
        const OriginalType data1 = BTInfo::template get<1>(biggerType);
        const OriginalType data2 = BTInfo::template get<2>(biggerType);
        const OriginalType data3 = BTInfo::template get<3>(biggerType);
        const OriginalType data4 = BTInfo::template get<4>(biggerType);
        const OriginalType data5 = BTInfo::template get<5>(biggerType);
        const OriginalType data6 = BTInfo::template get<6>(biggerType);
        const OriginalType data7 = BTInfo::template get<7>(biggerType);
        return data0 == eightNumbers[0] && data1 == eightNumbers[1] &&
            data2 == eightNumbers[2] && data3 == eightNumbers[3] &&
            data4 == eightNumbers[4] && data5 == eightNumbers[5] &&
            data6 == eightNumbers[6] && data7 == eightNumbers[7];
    } else {
        return false;
    }
}

namespace fk {
    template <typename OriginalType>
    bool testThreadFusionAggregate() {
        constexpr OriginalType fourNumbers[4]{ fk::make_<OriginalType>(10),
                                               fk::make_<OriginalType>(2),
                                               fk::make_<OriginalType>(3),
                                               fk::make_<OriginalType>(4) };

        using BTInfo = fk::ThreadFusionInfo<OriginalType, OriginalType, true>;

        const typename BTInfo::BiggerReadType biggerType = ((typename BTInfo::BiggerReadType*) fourNumbers)[0];

        using Reduction = VectorReduce<VectorType_t<bool, (cn<OriginalType>)>, Equal<uchar>>;

        if constexpr (BTInfo::elems_per_thread == 1) {
            return Reduction::exec(biggerType == fourNumbers[0]);
        } else if constexpr (BTInfo::elems_per_thread == 2) {
            const OriginalType data0 = BTInfo::template get<0>(biggerType);
            const OriginalType data1 = BTInfo::template get<1>(biggerType);
            return Reduction::exec(data0 == fourNumbers[0]) &&
                Reduction::exec(data1 == fourNumbers[1]);
        } else if constexpr (BTInfo::elems_per_thread == 4) {
            const OriginalType data0 = BTInfo::template get<0>(biggerType);
            const OriginalType data1 = BTInfo::template get<1>(biggerType);
            const OriginalType data2 = BTInfo::template get<2>(biggerType);
            const OriginalType data3 = BTInfo::template get<3>(biggerType);
            return Reduction::exec(data0 == fourNumbers[0]) &&
                Reduction::exec(data1 == fourNumbers[1]) &&
                Reduction::exec(data2 == fourNumbers[2]) &&
                Reduction::exec(data3 == fourNumbers[3]);
        } else {
            return false;
        }
    }
}

int launch() {
    bool passed = true;
    {
        PUSH_RANGE_RAII p("testThreadFusion");
        passed &= testThreadFusion<uchar>();
        passed &= testThreadFusion<char>();
        passed &= testThreadFusion<ushort>();
        passed &= testThreadFusion<short>();
        passed &= testThreadFusion<uint>();
        passed &= testThreadFusion<int>();
        passed &= testThreadFusion<ulong>();
        passed &= testThreadFusion<long>();
        passed &= testThreadFusion<ulonglong>();
        passed &= testThreadFusion<longlong>();
        passed &= testThreadFusion<float>();
        passed &= testThreadFusion<double>();
    }

#define LAUNCH_AGGREGATE(type) \
    passed &= fk::testThreadFusionAggregate<type ## 2>(); \
    passed &= fk::testThreadFusionAggregate<type ## 3>(); \
    passed &= fk::testThreadFusionAggregate<type ## 4>();

    {
        PUSH_RANGE_RAII p("testThreadFusionAggregate");
        LAUNCH_AGGREGATE(char)
        LAUNCH_AGGREGATE(uchar)
        LAUNCH_AGGREGATE(short)
        LAUNCH_AGGREGATE(ushort)
        LAUNCH_AGGREGATE(int)
        LAUNCH_AGGREGATE(uint)
        LAUNCH_AGGREGATE(long)
        LAUNCH_AGGREGATE(ulong)
        LAUNCH_AGGREGATE(longlong)
        LAUNCH_AGGREGATE(ulonglong)
        LAUNCH_AGGREGATE(float)
        LAUNCH_AGGREGATE(double)
    }
#undef LAUNCH_AGGREGATE

    return passed ? 0 : -1;
}