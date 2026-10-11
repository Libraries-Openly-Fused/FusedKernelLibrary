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

#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

namespace fk {

template <uint BATCH, uint WIDTH, uint HEIGHT, uint ITERS, typename IT, typename OT>
bool testCircularTensor() {
    using TensorOT = typename VectorTraits<OT>::base;
    constexpr uint COLOR_PLANES = cn<IT>;

    CircularTensor<TensorOT, COLOR_PLANES, BATCH, CircularTensorOrder::NewestFirst, ColorPlanes::Standard>
        myTensor(WIDTH, HEIGHT);
    Ptr2D<IT> input(WIDTH, HEIGHT);

    Stream fk_stream;
    setTo(10.0f, myTensor, fk_stream);

    for (int i = 0; i < ITERS; i++) {
        setTo(make_<IT>(i + 1, i + 1, i + 1), input, fk_stream);
        myTensor.update(fk_stream, Read<PerThreadRead<ND::_2D, IT>>{input.ptr()},
                        Unary<SaturateCast<IT, OT>>{}, Write<TensorSplit<OT>>{myTensor.ptr()});
        fk_stream.sync();
    }

    myTensor.download(fk_stream);
    fk_stream.sync();

    bool correct = true;
    for (int z = 0; z < BATCH; z++) {
        const TensorOT value = (TensorOT)(ITERS - static_cast<uint>(z));
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                const Point p{x, y, z};
                const TensorOT res = *PtrAccessor<ND::_3D>::point(p, myTensor.ptrPinned());
                correct &= value == res;
            }
        }
    }

    return correct;
}

#ifdef FK_CPU_MULTITHREAD_ENABLED
template <uint BATCH, typename IT, typename OT>
using HostCircularTensor = CircularTensor<typename VectorTraits<OT>::base, cn<IT>, BATCH,
                                          CircularTensorOrder::NewestFirst, ColorPlanes::Standard>;

// The updates of testCircularTensor, on host memory and the CPU or CPU_MULTITHREAD backend. Every update has its own
// input, so all of them are enqueued without synchronizing in between: the stream runs each one after the previous
// one. The CircularTensor and the inputs are used until sync().
template <ParArch PA, uint BATCH, typename IT, typename OT>
void updateOnHost(HostCircularTensor<BATCH, IT, OT>& myTensor, const std::vector<Ptr2D<IT>>& inputs) {
    using TensorOT = typename VectorTraits<OT>::base;
    std::fill_n(myTensor.ptr().data, myTensor.getNumElements(), static_cast<TensorOT>(10));
    Stream_<PA> stream;
    for (const Ptr2D<IT>& input : inputs) {
        myTensor.update(stream, Read<PerThreadRead<ND::_2D, IT>>{input.ptr()}, Unary<SaturateCast<IT, OT>>{},
                        Write<TensorSplit<OT>>{myTensor.ptr()});
    }
    stream.sync();
}

// testCircularTensor on the CPU_MULTITHREAD backend: the result is exactly the one of the CPU backend, and the expected
// one
template <uint BATCH, uint WIDTH, uint HEIGHT, uint ITERS, typename IT, typename OT>
bool testCircularTensorMultithread() {
    using TensorOT = typename VectorTraits<OT>::base;
    constexpr uint COLOR_PLANES = cn<IT>;

    std::vector<Ptr2D<IT>> inputs;
    for (uint i = 0; i < ITERS; i++) {
        inputs.push_back(Ptr2D<IT>(WIDTH, HEIGHT, 0, MemType::Host));
        const IT value = make_set<IT>(static_cast<VBase<IT>>(i + 1));
        for (int y = 0; y < static_cast<int>(HEIGHT); y++) {
            for (int x = 0; x < static_cast<int>(WIDTH); x++) {
                inputs.back().at(x, y) = value;
            }
        }
    }

    HostCircularTensor<BATCH, IT, OT> cpuTensor(WIDTH, HEIGHT, MemType::Host);
    HostCircularTensor<BATCH, IT, OT> multithreadTensor(WIDTH, HEIGHT, MemType::Host);
    updateOnHost<ParArch::CPU, BATCH, IT, OT>(cpuTensor, inputs);
    updateOnHost<ParArch::CPU_MULTITHREAD, BATCH, IT, OT>(multithreadTensor, inputs);

    bool correct = std::memcmp(cpuTensor.ptr().data, multithreadTensor.ptr().data, cpuTensor.sizeInBytes()) == 0;
    // Every color plane of plane z holds the input of the z-th last update. The planes are contiguous, and each one
    // stores its color planes one after the other.
    const TensorOT* const data = multithreadTensor.ptr().data;
    for (uint z = 0; z < BATCH; z++) {
        const TensorOT value = static_cast<TensorOT>(ITERS - z);
        for (uint c = 0; c < COLOR_PLANES; c++) {
            for (uint y = 0; y < HEIGHT; y++) {
                for (uint x = 0; x < WIDTH; x++) {
                    correct &= data[((z * COLOR_PLANES + c) * HEIGHT + y) * WIDTH + x] == value;
                }
            }
        }
    }

    return correct;
}
#endif // FK_CPU_MULTITHREAD_ENABLED

template <uint BATCH, uint WIDTH, uint HEIGHT, uint ITERS, typename IT, typename OT>
bool launchTest() {
    bool correct = true;
    if (testCircularTensor<BATCH, WIDTH, HEIGHT, ITERS, IT, OT>()) {
        std::cout << "testCircularTensor<" << BATCH << ", " << WIDTH << ", " << HEIGHT << ", " << ITERS << ", " << typeid(IT).name() << ", " << typeid(OT).name() << "> OK" << std::endl;
    } else {
        std::cout << "testCircularTensor<" << BATCH << ", " << WIDTH << ", " << HEIGHT << ", " << ITERS << ", " << typeid(IT).name() << ", " << typeid(OT).name() << "> Failed!"
                  << std::endl;
        correct = false;
    }
#ifdef FK_CPU_MULTITHREAD_ENABLED
    if (testCircularTensorMultithread<BATCH, WIDTH, HEIGHT, ITERS, IT, OT>()) {
        std::cout << "testCircularTensorMultithread<" << BATCH << ", " << WIDTH << ", " << HEIGHT << ", " << ITERS << ", " << typeid(IT).name() << ", " << typeid(OT).name() << "> OK" << std::endl;
    } else {
        std::cout << "testCircularTensorMultithread<" << BATCH << ", " << WIDTH << ", " << HEIGHT << ", " << ITERS << ", " << typeid(IT).name() << ", " << typeid(OT).name() << "> Failed!"
                  << std::endl;
        correct = false;
    }
#endif // FK_CPU_MULTITHREAD_ENABLED
    return correct;
}

int launch_impl() {
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

} // namespace fk

int launch() {
    return fk::launch_impl();
}
