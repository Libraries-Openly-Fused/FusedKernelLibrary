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

#ifndef FK_COLLECTIVE_TILE_H
#define FK_COLLECTIVE_TILE_H

/* Tile LAYOUT POLICIES + a cooperative tile-staging DPP.
 *
 * There is deliberately NO Tile object. A "tile" is just a __shared__ buffer
 * a DPP owns and addresses through a compile-time LAYOUT POLICY that maps
 * (row, col) -> linear element offset. Swizzle becomes a property of the
 * layout instead of a swz() hand-inlined into one kernel (cf. swz() in
 * flash_attention_mma.h). The offset math is pure single-thread constexpr, so
 * it is a policy, not an Op; the cross-thread load/store of the shared buffer
 * is multi-thread work, so it lives in a DPP (CopyTileDPP) — per the FKL rule
 * that anything needing more than one thread is a DPP, not an Op.
 *
 * A Layout policy is:
 *     static constexpr uint offset(uint row, uint col);   // -> element index
 *     static constexpr uint rows, cols;
 *     static constexpr uint size();                       // elements to back
 * New layouts (column major, padded, banked, byte-swizzle) are just new
 * policies — nothing else changes.
 */

#include <fused_kernel/core/data/point.h>
#include <fused_kernel/core/utils/utils.h>

namespace fk {

/* ---- Layout policies: (row, col) -> linear element offset ---- */

// Plain row-major: offset = row * COLS + col.
template <uint ROWS, uint COLS>
struct RowMajorLayout {
    static constexpr uint rows = ROWS;
    static constexpr uint cols = COLS;
    FK_HOST_DEVICE_CNST static uint offset(const uint row, const uint col) {
        return row * COLS + col;
    }
    FK_HOST_DEVICE_CNST static uint size() { return ROWS * COLS; }
};

/* XOR swizzle (the classic smem bank-conflict-avoiding layout): within a
 * row-major buffer the column index is XORed with a function of the row so
 * consecutive rows land in different bank phases. PERIOD (power of two) is the
 * row period; the XOR uses (row % PERIOD). Bijective within a row, so reads
 * see exactly what writes wrote. Same FAMILY as the hand-inlined swz() in
 * flash_attention_mma.h, now a reusable layout policy. (#280 adds a
 * byte-granularity ByteXorSwizzleLayout bit-identical to swz().) */
template <uint ROWS, uint COLS, uint PERIOD = 8>
struct XorSwizzleLayout {
    static constexpr uint rows = ROWS;
    static constexpr uint cols = COLS;
    static_assert((COLS & (COLS - 1)) == 0, "XorSwizzleLayout requires power-of-two COLS");
    static_assert((PERIOD & (PERIOD - 1)) == 0, "PERIOD must be a power of two");
    FK_HOST_DEVICE_CNST static uint offset(const uint row, const uint col) {
        const uint phase = row & (PERIOD - 1);
        return row * COLS + (col ^ phase);
    }
    FK_HOST_DEVICE_CNST static uint size() { return ROWS * COLS; }
};

#if defined(__NVCC__)

/* CopyTileDPP: cooperatively stage a ROWS x COLS tile between global memory
 * (row-major, given pitch in elements) and a caller-owned __shared__ buffer
 * addressed through Layout. This is the multi-thread piece — the whole block
 * participates — so it is a DPP, not an Op. The kernel owns the __shared__
 * (it budgeted Layout::size() elements); the DPP just indexes it. */
template <typename T, typename Layout, int BLOCK_SIZE = 256>
struct CopyTileDPP {
private:
    using SelfType = CopyTileDPP<T, Layout, BLOCK_SIZE>;
public:
    FK_STATIC_STRUCT(CopyTileDPP, SelfType)
    static constexpr uint TILE_ELEMS = Layout::rows * Layout::cols;

    // global -> shared: load the tile, applying the layout to the smem index.
    static __device__ __forceinline__
    void loadGlobalToShared(T* smem, const T* gmem, const int gmemPitchElems) {
        for (uint i = threadIdx.x; i < TILE_ELEMS; i += BLOCK_SIZE) {
            const uint r = i / Layout::cols, c = i % Layout::cols;
            smem[Layout::offset(r, c)] = gmem[r * gmemPitchElems + c];
        }
    }

    // shared -> global: store the tile back, reading through the layout.
    static __device__ __forceinline__
    void storeSharedToGlobal(const T* smem, T* gmem, const int gmemPitchElems) {
        for (uint i = threadIdx.x; i < TILE_ELEMS; i += BLOCK_SIZE) {
            const uint r = i / Layout::cols, c = i % Layout::cols;
            gmem[r * gmemPitchElems + c] = smem[Layout::offset(r, c)];
        }
    }

    // element accessor for in-DPP-body use: which smem slot holds (row, col).
    static __device__ __forceinline__ T& at(T* smem, const uint row, const uint col) {
        return smem[Layout::offset(row, col)];
    }
};

#endif // defined(__NVCC__)

} // namespace fk

#endif // FK_COLLECTIVE_TILE_H
