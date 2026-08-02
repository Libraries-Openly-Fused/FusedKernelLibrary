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

struct OneToOne {
    FK_DEVICE_FUSE uint at(const uint &zIdx) { return zIdx; }
};

bool testDivergentBatch() {
    constexpr uint WIDTH = 32;
    constexpr uint HEIGHT = 32;
    constexpr uint BATCH = 2;
    constexpr uint VAL_SUM = 3;

    fk::Stream stream;

    std::vector<fk::Ptr2D<uint>> inputAllocations;
    std::array<fk::RawPtr<fk::ND::_2D, uint>, BATCH> input;
    fk::Tensor<uint> output;
    fk::Tensor<uint> h_groundTruth;
    output.allocTensor(WIDTH, HEIGHT, BATCH);
    h_groundTruth.allocTensor(WIDTH, HEIGHT, BATCH, 1, fk::MemType::Host);

    for (uint i = 0; i < BATCH; i++) {
        fk::Ptr2D<uint> temp(WIDTH, HEIGHT);
        fk::setTo(i, temp, stream);
        inputAllocations.push_back(temp);
        input[i] = temp;
    }

    for (int z = 0; z < BATCH; z++) {
        if (z == 0) {
            for (int y = 0; y < HEIGHT; y++) {
                for (int x = 0; x < HEIGHT; x++) {
                    const fk::Point p{x, y, z};
                    *fk::PtrAccessor<fk::ND::_3D>::point(p, h_groundTruth.ptr()) = VAL_SUM;
                }
            }
        } else {
            for (int y = 0; y < HEIGHT; y++) {
                for (int x = 0; x < HEIGHT; x++) {
                    const fk::Point p{x, y, z};
                    *fk::PtrAccessor<fk::ND::_3D>::point(p, h_groundTruth.ptr()) = z;
                }
            }
        }
    }

    auto opSeq1 = fk::buildOperationSequence(fk::Read<fk::PerThreadRead<fk::ND::_2D, uint>>{input[0]},
                                             fk::Binary<fk::Add<uint>>{VAL_SUM},
                                             fk::Write<fk::PerThreadWrite<fk::ND::_3D, uint>>{output.ptr()});
    auto opSeq2 = fk::buildOperationSequence(fk::Read<fk::PerThreadRead<fk::ND::_2D, uint>>{input[1]},
                                             fk::Write<fk::PerThreadWrite<fk::ND::_3D, uint>>{output.ptr()});

    fk::executeOperations<fk::DivergentBatchTransformDPP<fk::defaultParArch, OneToOne>>(stream, opSeq1, opSeq2);

    output.download(stream);
    stream.sync();

    bool correct = true;
    for (int z = 0; z < BATCH; z++) {
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                const fk::Point p{x, y, z};
                const uint gt = *fk::PtrAccessor<fk::ND::_3D>::point(p, h_groundTruth.ptr());
                const uint res = *fk::PtrAccessor<fk::ND::_3D>::point(p, output.ptrPinned());
                correct &= gt == res;
            }
        }
    }

    return correct;
}

int launch() {
    int returnValue = 0;
    if (testDivergentBatch()) {
        std::cout << "testDivergentBatch OK" << std::endl;
    } else {
        std::cout << "testDivergentBatch Failed!" << std::endl;
        throw std::runtime_error("Test failed!");
        returnValue = -1;
    }
    return returnValue;
}