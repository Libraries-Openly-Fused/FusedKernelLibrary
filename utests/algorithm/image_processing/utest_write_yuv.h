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

#include <tests/main.h>

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/color_conversion.h>
#include <fused_kernel/algorithms/image_processing/image.h>

#include <iostream>
#include <random>

// WriteYUV for the 4:2:0 semi-planar formats: every pixel writes its luma, and the chroma of each 2x2 block comes from
// its top left pixel (even x and y). NV12 stores it as U, V and NV21 as V, U. NV21 used to write the chroma from the
// four pixels of the block: a data race on the GPU and on the multithreaded CPU backend, whose rows run concurrently.
namespace fk {

template <PixelFormat PF, typename DPP, typename StreamType>
bool testWriteYUV420(StreamType& stream, const Ptr2D<uchar3>& input, const int width, const int height,
                     const MemType memType, const char* backend) {
    Image<PF> image(static_cast<uint>(width), static_cast<uint>(height), memType);
    executeOperations<DPP>(stream, PerThreadRead<ND::_2D, uchar3>::build(input.ptr()), WriteYUV<PF>::build(image.ptr()));
    image.download(stream);
    stream.sync();

    const auto data = image.getData();
    size_t mismatches = 0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const uchar3 yuv = input.at(x, y);
            mismatches += data.at(x, y) != yuv.x;
            if (x % 2 == 0 && y % 2 == 0) {
                const uchar first = data.at(x, height + y / 2);
                const uchar second = data.at(x + 1, height + y / 2);
                const bool isNV21 = PF == PixelFormat::NV21;
                mismatches += first != (isNV21 ? yuv.z : yuv.y);
                mismatches += second != (isNV21 ? yuv.y : yuv.z);
            }
        }
    }
    std::cout << (PF == PixelFormat::NV21 ? "NV21" : "NV12") << " (" << backend << "): " << mismatches << " mismatches"
              << std::endl;
    return mismatches == 0;
}

Ptr2D<uchar3> randomYUV(const int width, const int height, const MemType memType) {
    Ptr2D<uchar3> input(static_cast<uint>(width), static_cast<uint>(height), 0, memType);
    std::mt19937 gen(3);
    std::uniform_int_distribution<int> byte(0, 255);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            input.at(x, y) = uchar3{static_cast<uchar>(byte(gen)), static_cast<uchar>(byte(gen)),
                                    static_cast<uchar>(byte(gen))};
        }
    }
    return input;
}

int launch_impl() {
    constexpr int W = 64, H = 32;
    Stream stream;
    Ptr2D<uchar3> input = randomYUV(W, H, defaultMemType);
    input.upload(stream);
    bool correct = testWriteYUV420<PixelFormat::NV12, TransformDPP<>>(stream, input, W, H, defaultMemType, "default");
    correct &= testWriteYUV420<PixelFormat::NV21, TransformDPP<>>(stream, input, W, H, defaultMemType, "default");
#ifdef FK_CPU_MULTITHREAD_ENABLED
    // The multithreaded CPU backend reads host memory
    using MTDPP = TransformDPP<ParArch::CPU_MULTITHREAD>;
    Stream_<ParArch::CPU_MULTITHREAD> mtStream;
    const Ptr2D<uchar3> hostInput = randomYUV(W, H, MemType::Host);
    correct &= testWriteYUV420<PixelFormat::NV12, MTDPP>(mtStream, hostInput, W, H, MemType::Host, "CPU_MULTITHREAD");
    correct &= testWriteYUV420<PixelFormat::NV21, MTDPP>(mtStream, hostInput, W, H, MemType::Host, "CPU_MULTITHREAD");
#endif
    return correct ? 0 : -1;
}

} // namespace fk

int launch() {
    return fk::launch_impl();
}
