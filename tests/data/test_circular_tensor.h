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

#include "tests/main.h"

#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/saturate.h>
#include <fused_kernel/core/data/circular_tensor.h>
#include <fused_kernel/core/data/ptr_utils.h>
#include <fused_kernel/fused_kernel.h>

#include <iostream>

template <uint BATCH, uint WIDTH, uint HEIGHT, uint ITERS, typename IT, typename OT>
bool testCircularTensor() {
    using TensorOT = typename fk::VectorTraits<OT>::base;
    constexpr uint COLOR_PLANES = fk::cn<IT>;

    fk::CircularTensor<TensorOT, COLOR_PLANES, BATCH, fk::CircularTensorOrder::NewestFirst, fk::ColorPlanes::Standard>
        myTensor(WIDTH, HEIGHT);
    fk::Ptr2D<IT> input(WIDTH, HEIGHT);

    fk::Stream fk_stream;
    fk::setTo(10.0f, myTensor, fk_stream);

    for (int i = 0; i < ITERS; i++) {
        fk::setTo(fk::make_<IT>(i + 1, i + 1, i + 1), input, fk_stream);
        myTensor.update(fk_stream, fk::Read<fk::PerThreadRead<fk::ND::_2D, IT>>{input.ptr()},
                        fk::Unary<fk::SaturateCast<IT, OT>>{}, fk::Write<fk::TensorSplit<OT>>{myTensor.ptr()});
        fk_stream.sync();
    }

    myTensor.download(fk_stream);
    fk_stream.sync();

    bool correct = true;
    for (int z = 0; z < BATCH; z++) {
        const TensorOT value = (TensorOT)(ITERS - z);
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                const fk::Point p{x, y, z};
                const TensorOT res = *fk::PtrAccessor<fk::ND::_3D>::point(p, myTensor.ptrPinned());
                correct &= value == res;
            }
        }
    }

    return correct;
}

template <uint BATCH, uint WIDTH, uint HEIGHT, uint ITERS, typename IT, typename OT>
bool launchTest() {
    if (testCircularTensor<BATCH, WIDTH, HEIGHT, ITERS, IT, OT>()) {
        std::cout << "testCircularTensor<" << BATCH << ", " << WIDTH << ", " << HEIGHT << ", " << ITERS << ", " << typeid(IT).name() << ", " << typeid(OT).name() << "> OK" << std::endl;
        return true;
    } else {
        std::cout << "testCircularTensor<" << BATCH << ", " << WIDTH << ", " << HEIGHT << ", " << ITERS << ", " << typeid(IT).name() << ", " << typeid(OT).name() << "> Failed!"
                  << std::endl;
        return false;
    }
}

int launch() {
    bool correct = true;
    correct &= launchTest<2, 128, 128, 100, uchar3, float3>();
    correct &= launchTest<3, 128, 128, 100, uchar3, float3>();
    correct &= launchTest<4, 128, 128, 100, uchar3, float3>();
    correct &= launchTest<5, 128, 128, 100, uchar3, float3>();
    correct &= launchTest<6, 128, 128, 100, uchar3, float3>();
    correct &= launchTest<7, 128, 128, 100, uchar3, float3>();
    correct &= launchTest<8, 128, 128, 100, uchar3, float3>();
    correct &= launchTest<9, 128, 128, 100, uchar3, float3>();
    correct &= launchTest<10, 128, 128, 100, uchar3, float3>();
    correct &= launchTest<11, 128, 128, 100, uchar3, float3>();
    correct &= launchTest<12, 128, 128, 100, uchar3, float3>();
    correct &= launchTest<13, 128, 128, 100, uchar3, float3>();
    correct &= launchTest<14, 128, 128, 100, uchar3, float3>();
    correct &= launchTest<15, 128, 128, 100, uchar3, float3>();
    return correct ? 0 : -1;
}