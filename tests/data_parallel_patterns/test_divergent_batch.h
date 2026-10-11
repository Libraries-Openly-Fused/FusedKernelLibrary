/* Copyright 2023 Mediaproduccion S.L.U. (Oscar Amoros Huguet)
   Copyright 2025-2026 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "tests/main.h"

#include <fused_kernel/core/execution_model/data_parallel_patterns.h>
#include <fused_kernel/core/core.h>
#include <fused_kernel/core/data/ptr_utils.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/fused_kernel.h>

#include <algorithm>
#include <array>
#include <cstring>

namespace fk {

// Host and device: the CPU and CPU_MULTITHREAD backends call it from host code
struct OneToOne {
    FK_HOST_DEVICE_FUSE uint at(const uint &zIdx) { return zIdx; }
};

bool testDivergentBatch() {
    constexpr uint WIDTH = 32;
    constexpr uint HEIGHT = 32;
    constexpr uint BATCH = 2;
    constexpr uint VAL_SUM = 3;

    Stream stream;

    std::vector<Ptr2D<uint>> inputAllocations;
    std::array<RawPtr<ND::_2D, uint>, BATCH> input;
    Tensor<uint> output;
    Tensor<uint> h_groundTruth;
    output.allocTensor(WIDTH, HEIGHT, BATCH);
    h_groundTruth.allocTensor(WIDTH, HEIGHT, BATCH, 1, MemType::Host);

    for (uint i = 0; i < BATCH; i++) {
        Ptr2D<uint> temp(WIDTH, HEIGHT);
        setTo(i, temp, stream);
        inputAllocations.push_back(temp);
        input[i] = temp;
    }

    for (int z = 0; z < BATCH; z++) {
        if (z == 0) {
            for (int y = 0; y < HEIGHT; y++) {
                for (int x = 0; x < WIDTH; x++) {
                    const Point p{x, y, z};
                    *PtrAccessor<ND::_3D>::point(p, h_groundTruth.ptr()) = VAL_SUM;
                }
            }
        } else {
            for (int y = 0; y < HEIGHT; y++) {
                for (int x = 0; x < HEIGHT; x++) {
                    const Point p{x, y, z};
                    *PtrAccessor<ND::_3D>::point(p, h_groundTruth.ptr()) = static_cast<uint>(z);
                }
            }
        }
    }

    auto opSeq1 = buildOperationSequence(Read<PerThreadRead<ND::_2D, uint>>{input[0]},
                                             Binary<Add<uint>>{VAL_SUM},
                                             Write<PerThreadWrite<ND::_3D, uint>>{output.ptr()});
    auto opSeq2 = buildOperationSequence(Read<PerThreadRead<ND::_2D, uint>>{input[1]},
                                             Write<PerThreadWrite<ND::_3D, uint>>{output.ptr()});

    executeOperations<DivergentBatchTransformDPP<defaultParArch, OneToOne>>(stream, opSeq1, opSeq2);

    output.download(stream);
    stream.sync();

    bool correct = true;
    for (int z = 0; z < BATCH; z++) {
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                const Point p{x, y, z};
                const uint gt = *PtrAccessor<ND::_3D>::point(p, h_groundTruth.ptr());
                const uint res = *PtrAccessor<ND::_3D>::point(p, output.ptrPinned());
                correct &= gt == res;
            }
        }
    }

    return correct;
}

#ifdef FK_CPU_MULTITHREAD_ENABLED
// Runs on host memory, on the CPU or CPU_MULTITHREAD backend, the sequences of testDivergentBatch (the input of plane
// 0 plus valSum, and the input of plane 1), which here may have different sizes
template <ParArch PA>
void runDivergentBatchOnHost(const std::array<Ptr2D<uint>, 2>& inputs, const uint valSum, const Tensor<uint>& output) {
    Stream_<PA> stream;
    const auto opSeq1 = buildOperationSequence(Read<PerThreadRead<ND::_2D, uint>>{inputs[0].ptr()},
                                               Binary<Add<uint>>{valSum},
                                               Write<PerThreadWrite<ND::_3D, uint>>{output.ptr()});
    const auto opSeq2 = buildOperationSequence(Read<PerThreadRead<ND::_2D, uint>>{inputs[1].ptr()},
                                               Write<PerThreadWrite<ND::_3D, uint>>{output.ptr()});
    executeOperations<DivergentBatchTransformDPP<PA, OneToOne>>(stream, opSeq1, opSeq2);
    stream.sync();
}

// The CPU_MULTITHREAD backend gives exactly the same output as the CPU one, and the expected one, also when the
// sequences have different sizes: the rows of the taller sequence that the other one does not have are empty chunks
// of the plane of the other one, and the elements of a plane its sequence does not cover keep their initial value
template <uint W0, uint H0, uint W1, uint H1>
bool testDivergentBatchMultithread() {
    constexpr uint BATCH = 2;
    constexpr uint VAL_SUM = 3;
    constexpr uint UNTOUCHED = 0xDEADBEEFu;
    constexpr uint WIDTH = W0 > W1 ? W0 : W1;
    constexpr uint HEIGHT = H0 > H1 ? H0 : H1;
    constexpr uint widths[BATCH] = {W0, W1};
    constexpr uint heights[BATCH] = {H0, H1};

    std::array<Ptr2D<uint>, BATCH> inputs{Ptr2D<uint>(W0, H0, 0, MemType::Host), Ptr2D<uint>(W1, H1, 0, MemType::Host)};
    for (uint i = 0; i < BATCH; i++) {
        for (uint y = 0; y < heights[i]; y++) {
            for (uint x = 0; x < widths[i]; x++) {
                inputs[i].at(static_cast<int>(x), static_cast<int>(y)) = i * 1000000u + y * 1000u + x;
            }
        }
    }
    Tensor<uint> cpuOutput(WIDTH, HEIGHT, BATCH, 1, MemType::Host);
    Tensor<uint> multithreadOutput(WIDTH, HEIGHT, BATCH, 1, MemType::Host);
    std::fill_n(cpuOutput.ptr().data, cpuOutput.getNumElements(), UNTOUCHED);
    std::fill_n(multithreadOutput.ptr().data, multithreadOutput.getNumElements(), UNTOUCHED);

    runDivergentBatchOnHost<ParArch::CPU>(inputs, VAL_SUM, cpuOutput);
    runDivergentBatchOnHost<ParArch::CPU_MULTITHREAD>(inputs, VAL_SUM, multithreadOutput);

    bool correct = std::memcmp(cpuOutput.ptr().data, multithreadOutput.ptr().data, cpuOutput.sizeInBytes()) == 0;
    for (uint z = 0; z < BATCH; z++) {
        for (uint y = 0; y < HEIGHT; y++) {
            for (uint x = 0; x < WIDTH; x++) {
                const bool covered = x < widths[z] && y < heights[z];
                const uint gt = !covered ? UNTOUCHED : z * 1000000u + y * 1000u + x + (z == 0 ? VAL_SUM : 0u);
                const Point p{static_cast<int>(x), static_cast<int>(y), static_cast<int>(z)};
                correct &= *PtrAccessor<ND::_3D>::point(p, multithreadOutput.ptr()) == gt;
            }
        }
    }
    return correct;
}
#endif // FK_CPU_MULTITHREAD_ENABLED

int launch_impl() {
    int returnValue = 0;
    if (testDivergentBatch()) {
        std::cout << "testDivergentBatch OK" << std::endl;
    } else {
        std::cout << "testDivergentBatch Failed!" << std::endl;
        throw std::runtime_error("Test failed!");
        returnValue = -1;
    }
#ifdef FK_CPU_MULTITHREAD_ENABLED
    bool multithreadCorrect = true;
    // The case of testDivergentBatch, and sequences of different sizes
    multithreadCorrect &= testDivergentBatchMultithread<32, 32, 32, 32>();
    multithreadCorrect &= testDivergentBatchMultithread<45, 17, 13, 70>();
    multithreadCorrect &= testDivergentBatchMultithread<640, 360, 1280, 720>();
    if (multithreadCorrect) {
        std::cout << "testDivergentBatchMultithread OK" << std::endl;
    } else {
        std::cout << "testDivergentBatchMultithread Failed!" << std::endl;
        returnValue = -1;
    }
#endif // FK_CPU_MULTITHREAD_ENABLED
    return returnValue;
}

} // namespace fk

int launch() {
    return fk::launch_impl();
}
