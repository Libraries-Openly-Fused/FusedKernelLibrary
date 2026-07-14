/* Copyright 2026 Oscar Amoros Huguet, Johnny Nunez

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

#include <fused_kernel/algorithms/collective/tile.h>
#include "tests/nvtx.h"

#include <vector>

using namespace fk;

/* Layout correctness is a host-side property (constexpr offset math), so the
 * core checks run identically on CPU and CUDA passes — the host/device parity
 * FKL wants. The critical invariant for a swizzle is BIJECTIVITY within the
 * buffer: distinct (row,col) must map to distinct offsets in [0, size), or a
 * tile silently corrupts data. */

template <typename Layout>
static bool layoutIsBijective() {
    const uint n = Layout::size();
    std::vector<int> seen(n, -1);
    for (uint r = 0; r < Layout::rows; ++r) {
        for (uint c = 0; c < Layout::cols; ++c) {
            const uint off = Layout::offset(r, c);
            if (off >= n) return false;                 // out of bounds
            if (seen[off] != -1) return false;          // collision -> not bijective
            seen[off] = (int)(r * Layout::cols + c);
        }
    }
    for (uint i = 0; i < n; ++i) if (seen[i] == -1) return false; // not surjective
    return true;
}

static bool rowMajorOffsetsAreContiguous() {
    using L = RowMajorLayout<4, 8>;
    for (uint r = 0; r < 4; ++r)
        for (uint c = 0; c < 8; ++c)
            if (L::offset(r, c) != r * 8 + c) return false;
    return true;
}

#if defined(__NVCC__)
/* Cooperative tile staging: the block loads a ROWS x COLS tile gmem->smem
 * through the layout (swizzle applied to the smem index), then stores it back
 * smem->gmem. A correct round-trip proves the DPP addresses the __shared__
 * via the layout consistently on both directions. No Tile object, no Ops:
 * just a __shared__ the DPP indexes with Layout::offset. */
template <typename T, typename Layout, int BLOCK_SIZE>
__global__ void copyTileRoundTrip(const T* gin, T* gout, int pitchElems) {
    __shared__ T smem[Layout::size()];
    using DPP = fk::CopyTileDPP<T, Layout, BLOCK_SIZE>;
    DPP::loadGlobalToShared(smem, gin, pitchElems);
    __syncthreads();
    DPP::storeSharedToGlobal(smem, gout, pitchElems);
}

template <typename Layout>
static bool deviceCopyCheck() {
    constexpr int BS = 128;
    const int rows = Layout::rows, cols = Layout::cols;
    std::vector<float> in(rows * cols), out(rows * cols, -1.f);
    for (int i = 0; i < rows * cols; ++i) in[i] = (float)(i + 1);
    float *gi, *go; cudaMalloc(&gi, in.size()*4); cudaMalloc(&go, out.size()*4);
    cudaMemcpy(gi, in.data(), in.size()*4, cudaMemcpyHostToDevice);
    copyTileRoundTrip<float, Layout, BS><<<1, BS>>>(gi, go, cols);
    cudaDeviceSynchronize();
    cudaMemcpy(out.data(), go, out.size()*4, cudaMemcpyDeviceToHost);
    cudaFree(gi); cudaFree(go);
    for (int i = 0; i < rows * cols; ++i) if (out[i] != in[i]) return false;
    return true;
}
#endif

int launch() {
    bool passed = true;
    {
        PUSH_RANGE_RAII p("tile_layout_host");
        passed &= rowMajorOffsetsAreContiguous();
        passed &= layoutIsBijective<RowMajorLayout<16, 64>>();
        passed &= layoutIsBijective<XorSwizzleLayout<16, 64, 8>>();
        passed &= layoutIsBijective<XorSwizzleLayout<32, 32, 8>>();
        passed &= layoutIsBijective<XorSwizzleLayout<8, 128, 4>>();
    }
#if defined(__NVCC__)
    {
        PUSH_RANGE_RAII p("tile_copy_dpp_device");
        passed &= deviceCopyCheck<RowMajorLayout<16, 64>>();
        passed &= deviceCopyCheck<XorSwizzleLayout<16, 64, 8>>();
        passed &= deviceCopyCheck<XorSwizzleLayout<32, 32, 8>>();
    }
#endif
    return passed ? 0 : -1;
}
