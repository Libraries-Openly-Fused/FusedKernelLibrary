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

#include <tests/main.h>

#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/collective/copy.h>
#include <fused_kernel/algorithms/collective/tile.h>

#include <cstdio>
#include <vector>

using namespace fk;

namespace {

uint referenceSwizzle(const uint row, const uint col,
                      const uint elementBytes, const uint strideBytes,
                      const uint period) {
    const uint divisor = 64u / strideBytes > 1u
        ? 64u / strideBytes : 1u;
    const uint byteOffset = row * strideBytes + col * elementBytes;
    return (byteOffset ^ (((row % period) / divisor) << 4u)) /
           elementBytes;
}

template <typename Layout>
bool checkHostMapping(const uint elementBytes,
                      const uint strideBytes, const uint period) {
    std::vector<int> seen(Layout::size(), 0);
    for (uint row = 0; row < Layout::rows; ++row) {
        for (uint col = 0; col < Layout::cols; ++col) {
            const uint offset = Layout::offset(row, col);
            if (offset != referenceSwizzle(
                    row, col, elementBytes, strideBytes, period) ||
                offset >= Layout::size()) return false;
            ++seen[offset];
        }
    }
    for (const int count : seen)
        if (count != 1) return false;
    return true;
}

#if defined(__NVCC__)
template <typename Layout>
__global__ void mapOffsets(uint* offsets) {
    const uint index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < Layout::size())
        offsets[index] = Layout::offset(
            index / Layout::cols, index % Layout::cols);
}

template <typename Layout>
bool checkDeviceMapping(const uint elementBytes,
                        const uint strideBytes, const uint period) {
    uint* device = nullptr;
    cudaMalloc(&device, Layout::size() * sizeof(uint));
    mapOffsets<Layout><<<(Layout::size() + 127) / 128, 128>>>(device);
    std::vector<uint> offsets(Layout::size());
    cudaMemcpy(offsets.data(), device,
               Layout::size() * sizeof(uint), cudaMemcpyDeviceToHost);
    const cudaError_t error = cudaDeviceSynchronize();
    cudaFree(device);
    if (error != cudaSuccess) return false;
    for (uint index = 0; index < Layout::size(); ++index) {
        if (offsets[index] != referenceSwizzle(
                index / Layout::cols, index % Layout::cols,
                elementBytes, strideBytes, period)) return false;
    }
    return true;
}
#endif

#if defined(__NVCC__)
template <typename Layout, typename Details,
          typename ReadIOp, typename WriteIOp>
__global__ void roundTripKernel(const Details details,
                                const ReadIOp input,
                                const WriteIOp output);
#endif

template <typename Layout>
bool checkRoundTrip() {
    using Details = CopyTileDPPDetails<float, Layout, 128>;
    std::vector<float> input(Layout::size());
    std::vector<float> output(Layout::size(), -1.f);
    std::vector<float> storage(Layout::size(), 0.f);
    for (uint index = 0; index < Layout::size(); ++index)
        input[index] = static_cast<float>(index) * 0.25f - 3.f;
    const RawPtr<ND::_2D, float> inPtr{
        input.data(), PtrDims<ND::_2D>(
            Layout::cols, Layout::rows, Layout::cols * sizeof(float))};
    const RawPtr<ND::_2D, float> outPtr{
        output.data(), PtrDims<ND::_2D>(
            Layout::cols, Layout::rows, Layout::cols * sizeof(float))};
    const auto read = PerThreadRead<ND::_2D, float>::build(inPtr);
    const auto write = PerThreadWrite<ND::_2D, float>::build(outPtr);
    const Details details{0, 0,
                          static_cast<int>(Layout::cols),
                          static_cast<int>(Layout::rows), -99.f};
    Tile<float, Layout> tile(storage.data());
    CopyTileDPP<ParArch::CPU, Details>::load(details, read, tile);
    CopyTileDPP<ParArch::CPU, Details>::store(details, tile, write);
    if (output != input) return false;

#if defined(__NVCC__)
    Ptr2D<float> gpuInput(Layout::cols, Layout::rows);
    Ptr2D<float> gpuOutput(Layout::cols, Layout::rows);
    for (uint row = 0; row < Layout::rows; ++row)
        for (uint col = 0; col < Layout::cols; ++col)
            gpuInput.at(Point{static_cast<int>(col),
                              static_cast<int>(row), 0}) =
                input[row * Layout::cols + col];
    Stream stream;
    gpuInput.upload(stream);
    const auto gpuRead =
        PerThreadRead<ND::_2D, float>::build(gpuInput);
    const auto gpuWrite =
        PerThreadWrite<ND::_2D, float>::build(gpuOutput);
    roundTripKernel<Layout, Details><<<1, 128>>>(
        details, gpuRead, gpuWrite);
    gpuOutput.download(stream);
    stream.sync();
    for (uint row = 0; row < Layout::rows; ++row)
        for (uint col = 0; col < Layout::cols; ++col)
            if (gpuOutput.at(Point{static_cast<int>(col),
                                   static_cast<int>(row), 0}) !=
                input[row * Layout::cols + col]) return false;
#endif
    return true;
}

#if defined(__NVCC__)
template <typename Layout, typename Details,
          typename ReadIOp, typename WriteIOp>
__global__ void roundTripKernel(const Details details,
                                const ReadIOp input,
                                const WriteIOp output) {
    __shared__ float storage[Layout::size()];
    Tile<float, Layout> tile(storage);
    CopyTileDPP<ParArch::GPU_NVIDIA, Details>::load(
        details, input, tile);
    CopyTileDPP<ParArch::GPU_NVIDIA, Details>::store(
        details, tile, output);
}
#endif

template <typename Layout>
bool checkLayout(const uint elementBytes,
                 const uint strideBytes, const uint period) {
    bool ok = checkHostMapping<Layout>(elementBytes, strideBytes, period);
#if defined(__NVCC__)
    ok = checkDeviceMapping<Layout>(
        elementBytes, strideBytes, period) && ok;
#endif
    return ok;
}

} // namespace

int launch() {
    using E1 = ByteXorSwizzleLayout<16, 128, 1, 128, 8>;
    using E2 = ByteXorSwizzleLayout<16, 64, 2, 128, 8>;
    using E2Wide = ByteXorSwizzleLayout<16, 128, 2, 256, 8>;
    using E4 = ByteXorSwizzleLayout<16, 32, 4, 128, 8>;
    using E8 = ByteXorSwizzleLayout<16, 16, 8, 128, 8>;
    using E16 = ByteXorSwizzleLayout<16, 8, 16, 128, 8>;

    static_assert(!ByteXorSwizzleLayoutSupport<8, 12, 4, 48, 8>::value,
                  "non-power-of-two stride must be rejected");
    static_assert(!ByteXorSwizzleLayoutSupport<8, 32, 3, 96, 8>::value,
                  "unsupported element size must be rejected");
    static_assert(!ByteXorSwizzleLayoutSupport<8, 32, 4, 256, 8>::value,
                  "stride/shape mismatch must be rejected");
    static_assert(!ByteXorSwizzleLayoutSupport<8, 1, 16, 16, 8>::value,
                  "mask crossing a row must be rejected");
    static_assert(!ByteXorSwizzleLayoutSupport<8, 32, 4, 128, 3>::value,
                  "non-power-of-two period must be rejected");

    bool ok = checkLayout<E1>(1, 128, 8);
    ok = checkLayout<E2>(2, 128, 8) && ok;
    ok = checkLayout<E2Wide>(2, 256, 8) && ok;
    ok = checkLayout<E4>(4, 128, 8) && ok;
    ok = checkLayout<E8>(8, 128, 8) && ok;
    ok = checkLayout<E16>(16, 128, 8) && ok;
    ok = checkRoundTrip<E4>() && ok;
    if (ok) std::printf("ByteXorSwizzleLayout contracts: PASS\n");
    return ok ? 0 : -1;
}
