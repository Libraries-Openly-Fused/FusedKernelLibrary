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

// A fused kernel of more than 1024 operations.
//
// fk::get_arg used to build a Tuple of references to all the arguments at every call. With the operations of a fused kernel as the arguments (each
// one a __grid_constant__ kernel parameter), nvcc removed that object up to 1024 operations and kept it as an 8 KB local array of pointers beyond:
// the kernel went from 17 registers to 255 and thousands of spills, with the right result. So this test checks, besides the result on the CPU and
// on the GPU, the resources of the generated kernel (registers and local memory, through cudaFuncGetAttributes), which is what regressed.
#include <tests/main.h>

#include <fused_kernel/algorithms/algorithms.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/core/execution_model/executors.h>
#include <fused_kernel/core/utils/parameter_pack_utils.h>

#include <iostream>
#include <utility>

// Everything is in a namespace: the test harness of main checks that no test header exposes fk::uint3 and friends in the global lookup
namespace get_arg_large_pack_test {
using namespace fk;

constexpr int NUM_OPS = 1100; // more than 1024, each with its own parameter (Add<int> of 1, 2, 3 ... NUM_OPS)
constexpr int WIDTH = 64, HEIGHT = 4;

// get_arg returns a reference to the argument itself, also for packs larger than 1024
template <size_t N, size_t I, size_t... Is>
bool getArgIsTheArgument(int (&values)[N], const std::index_sequence<Is...>&) {
    return &get_arg<I>(values[Is]...) == &values[I];
}

template <size_t N>
bool testGetArg() {
    int values[N];
    for (size_t i = 0; i < N; ++i) {
        values[i] = static_cast<int>(i);
    }
    const auto indices = std::make_index_sequence<N>{};
    bool ok = getArgIsTheArgument<N, 0>(values, indices);
    ok &= getArgIsTheArgument<N, 1>(values, indices);
    ok &= getArgIsTheArgument<N, 1023>(values, indices);
    ok &= getArgIsTheArgument<N, 1024>(values, indices);
    ok &= getArgIsTheArgument<N, N - 1>(values, indices);
    if (!ok) {
        std::cout << "get_arg did not return the argument on a pack of " << N << std::endl;
    }
    return ok;
}

template <typename ReadIOp, typename WriteIOp, size_t... Is>
void runChain(Stream& stream, const ReadIOp& readOp, const WriteIOp& writeOp, const std::index_sequence<Is...>&) {
    Executor<TransformDPP<>>::executeOperations(stream, readOp, Add<int>::build(static_cast<int>(Is) + 1)..., writeOp);
}

#if defined(__NVCC__)
// The attributes of the kernel that runChain launches
template <typename... IOps>
cudaFuncAttributes kernelAttributes(const IOps&... iOps) {
    constexpr ParArch PA = ParArch::GPU_NVIDIA;
    using Details = std::decay_t<decltype(TransformDPP<PA, TF::DISABLED>::build_details(iOps...))>;
    // The address of the specialization as a const void*: nvcc does not match a function pointer type with the const parameters of the pack
    const void* kernel = reinterpret_cast<const void*>(&launchTransformDPP_Kernel<PA, TF::DISABLED, true, Details, IOps...>);
    cudaFuncAttributes attributes{};
    gpuErrchk(cudaFuncGetAttributes(&attributes, kernel));
    return attributes;
}

template <typename ReadIOp, typename WriteIOp, size_t... Is>
cudaFuncAttributes chainKernelAttributes(const ReadIOp& readOp, const WriteIOp& writeOp, const std::index_sequence<Is...>&) {
    return kernelAttributes(readOp, Add<int>::build(static_cast<int>(Is) + 1)..., writeOp);
}
#endif

bool testChain() {
    Stream stream;
    Ptr2D<int> input(WIDTH, HEIGHT);
    Ptr2D<int> output(WIDTH, HEIGHT);
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            input.at(x, y) = x + 100 * y;
        }
    }
    input.upload(stream);

    const auto readOp = PerThreadRead<ND::_2D, int>::build(input);
    const auto writeOp = PerThreadWrite<ND::_2D, int>::build(output);
    const auto indices = std::make_index_sequence<NUM_OPS>{};
    runChain(stream, readOp, writeOp, indices);
    output.download(stream);
    stream.sync();

    constexpr int sumOfParameters = NUM_OPS * (NUM_OPS + 1) / 2;
    bool correct{ true };
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            if (output.at(x, y) != x + 100 * y + sumOfParameters) {
                std::cout << "Mismatch at (" << x << ", " << y << "): " << output.at(x, y) << " instead of " << x + 100 * y + sumOfParameters << std::endl;
                correct = false;
            }
        }
    }

#if defined(__NVCC__)
    const cudaFuncAttributes attributes = chainKernelAttributes(readOp, writeOp, indices);
    // 17 registers and no local memory with get_arg without the tuple; 255 registers and thousands of bytes of spills with it
    if (attributes.localSizeBytes != 0 || attributes.numRegs > 64) {
        std::cout << "The kernel of " << NUM_OPS << " operations uses " << attributes.numRegs << " registers and " << attributes.localSizeBytes
                  << " bytes of local memory (expected at most 64 and 0)" << std::endl;
        correct = false;
    }
#endif
    return correct;
}

int run() {
    bool ok = testGetArg<1100>();
    ok &= testGetArg<2100>();
    ok &= testChain();
    return ok ? 0 : -1;
}
} // namespace get_arg_large_pack_test

int launch() {
    return get_arg_large_pack_test::run();
}
