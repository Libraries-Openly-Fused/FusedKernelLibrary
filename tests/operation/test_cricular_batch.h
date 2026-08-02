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

#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/fused_kernel.h>

#include <iostream>

bool testCircularBatchRead() {
    constexpr uint WIDTH = 32;
    constexpr uint HEIGHT = 32;
    constexpr uint BATCH = 15;
    constexpr uint FIRST = 4;

    fk::Stream stream;
    fk::Stream fk_stream(stream);

    std::vector<fk::Ptr2D<uchar3>> inputAllocations;
    std::array<fk::RawPtr<fk::ND::_2D, uchar3>, BATCH> input;
    fk::Tensor<uchar3> output;

    for (int i = 0; i < BATCH; i++) {
        fk::Ptr2D<uchar3> temp(WIDTH, HEIGHT, 0);
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                const fk::Point p{x, y, 0};
                *fk::PtrAccessor<fk::ND::_2D>::point(p, temp.ptrPinned()) = fk::make_<uchar3>(i, i, i);
            }
        }
        temp.upload(stream);
        inputAllocations.push_back(temp);
        input[i] = temp;
    }
    output.allocTensor(WIDTH, HEIGHT, BATCH);

    fk::Read<fk::CircularBatchRead<fk::CircularDirection::Ascendent, fk::PerThreadRead<fk::ND::_2D, uchar3>, BATCH>>
        circularBatchRead;
    circularBatchRead.params.first = FIRST;
    for (int i = 0; i < BATCH; i++) {
        circularBatchRead.params.opData[i].params = input[i];
    }
    fk::Write<fk::PerThreadWrite<fk::ND::_3D, uchar3>> write3D{{output}};

    fk::executeOperations<fk::TransformDPP<>>(fk_stream, circularBatchRead, write3D);

    output.download(stream);
    stream.sync();

    bool correct = true;
    for (int z = 0; z < BATCH; z++) {
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                fk::Point p{x, y, z};
                uchar3 res = *fk::PtrAccessor<fk::ND::_3D>::point(p, output.ptrPinned());
                uchar newZ = (z + FIRST);
                uchar3 gt = newZ >= BATCH ? fk::make_set<uchar3>(newZ - BATCH) : fk::make_set<uchar3>(newZ);
                correct &= res.x == gt.x;
                correct &= res.y == gt.y;
                correct &= res.z == gt.z;
            }
        }
    }

    return correct;
}

int launch() {
    int returnValue = 0;
    if (testCircularBatchRead()) {
        std::cout << "testCircularBatchRead OK" << std::endl;
    } else {
        std::cout << "testCircularBatchRead Failed!" << std::endl;
        returnValue = -1;
    }
    return returnValue;
}