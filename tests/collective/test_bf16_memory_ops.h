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

#define __ONLY_CU__
#include <tests/main.h>

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>

#include <cuda_bf16.h>
#include <cstring>
#include <type_traits>
#include <vector>

using namespace fk;

struct Bf16CopyDetails { int count; };

struct Bf16CopyDPP {
    static constexpr ParArch PAR_ARCH = ParArch::GPU_NVIDIA;

    FK_HOST_FUSE DPPLaunchConfig launchConfig(const Bf16CopyDetails& details) {
        const unsigned int count = static_cast<unsigned int>(details.count);
        return {(count + 31) / 32, 1, 1, 32, 1, 1, 0};
    }

    template <typename ReadIOp, typename WriteIOp>
    FK_DEVICE_STATIC void exec(const Bf16CopyDetails& details,
                               const ReadIOp& input,
                               const WriteIOp& output) {
        const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        if (index >= details.count) return;
        const auto value = ReadIOp::Operation::exec(Point{index, 0, 0}, input);
        WriteIOp::Operation::exec(Point{index, 0, 0}, value, output);
    }
};

template <typename T>
bool roundTrip(const int count) {
    Ptr1D<T> input(static_cast<uint>(count));
    Ptr1D<T> output(static_cast<uint>(count));
    for (int i = 0; i < count; ++i) {
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            input.at(Point{i, 0, 0}) = __float2bfloat16(0.125f * i - 1.f);
        } else {
            input.at(Point{i, 0, 0}) = __floats2bfloat162_rn(
                0.125f * i - 1.f, 0.25f * i + 0.5f);
        }
    }
    Stream stream;
    input.upload(stream);
    executeOperations<Bf16CopyDPP>(
        stream, Bf16CopyDetails{count},
        PerThreadRead<ND::_1D, T>::build(input),
        PerThreadWrite<ND::_1D, T>::build(output));
    output.download(stream);
    stream.sync();
    for (int i = 0; i < count; ++i) {
        const T expected = input.at(Point{i, 0, 0});
        const T actual = output.at(Point{i, 0, 0});
        if (std::memcmp(&expected, &actual, sizeof(T)) != 0) return false;
    }
    return true;
}

int launch() {
    return roundTrip<__nv_bfloat16>(35) &&
           roundTrip<__nv_bfloat162>(17) ? 0 : -1;
}
