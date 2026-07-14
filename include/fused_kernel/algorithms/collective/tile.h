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

#include <fused_kernel/core/utils/utils.h>

#include <type_traits>

namespace fk {

// Layout policies map logical tile coordinates to local element offsets.
template <uint ROWS, uint COLS>
struct RowMajorLayout {
    static_assert(ROWS > 0 && COLS > 0, "Tile dimensions must be positive");
    static constexpr uint rows = ROWS;
    static constexpr uint cols = COLS;

    FK_HOST_DEVICE_FUSE uint offset(const uint row, const uint col) {
        return row * COLS + col;
    }

    FK_HOST_DEVICE_FUSE uint size() { return ROWS * COLS; }
};

template <uint ROWS, uint COLS>
struct ColumnMajorLayout {
    static_assert(ROWS > 0 && COLS > 0, "Tile dimensions must be positive");
    static constexpr uint rows = ROWS;
    static constexpr uint cols = COLS;

    FK_HOST_DEVICE_FUSE uint offset(const uint row, const uint col) {
        return col * ROWS + row;
    }

    FK_HOST_DEVICE_FUSE uint size() { return ROWS * COLS; }
};

template <uint ROWS, uint COLS, uint ELEM_BYTES,
          uint STRIDE_BYTES, uint PERIOD = 8>
struct ByteXorSwizzleLayoutSupport {
private:
    FK_HOST_DEVICE_FUSE bool evaluate() {
        if (ROWS == 0 || COLS == 0 || ELEM_BYTES == 0 ||
            STRIDE_BYTES == 0 || PERIOD == 0) return false;
        if (16u % ELEM_BYTES != 0u) return false;
        if (STRIDE_BYTES != COLS * ELEM_BYTES) return false;
        if (STRIDE_BYTES < 16u || STRIDE_BYTES % 16u != 0u)
            return false;
        if ((STRIDE_BYTES & (STRIDE_BYTES - 1u)) != 0u)
            return false;
        if ((PERIOD & (PERIOD - 1u)) != 0u) return false;
        const uint divisor = 64u / STRIDE_BYTES > 1u
            ? 64u / STRIDE_BYTES : 1u;
        const uint maximumMask = ((PERIOD - 1u) / divisor) << 4u;
        return maximumMask < STRIDE_BYTES;
    }

public:
    static constexpr bool value = evaluate();
};

/**
 * Byte-address XOR layout bit-identical to attention's swz(byteOff):
 *   byteOff ^ (((row % PERIOD) / max(64 / STRIDE_BYTES, 1)) << 4)
 *
 * Preconditions are compile-time enforced: element size divides 16 bytes,
 * row stride equals COLS*ELEM_BYTES, is 16-byte aligned and power-of-two,
 * PERIOD is power-of-two, and the largest XOR mask stays inside one row.
 * These conditions keep every mapped byte on an element boundary and make the
 * mapping bijective inside the statically shaped tile.
 */
template <uint ROWS, uint COLS, uint ELEM_BYTES,
          uint STRIDE_BYTES, uint PERIOD = 8>
struct ByteXorSwizzleLayout {
    static_assert(ByteXorSwizzleLayoutSupport<
                      ROWS, COLS, ELEM_BYTES,
                      STRIDE_BYTES, PERIOD>::value,
                  "Unsupported ByteXorSwizzleLayout shape/alignment");
    static constexpr uint rows = ROWS;
    static constexpr uint cols = COLS;
    static constexpr uint elementBytes = ELEM_BYTES;
    static constexpr uint strideBytes = STRIDE_BYTES;
    static constexpr uint period = PERIOD;

    FK_HOST_DEVICE_FUSE uint offset(const uint row, const uint col) {
        const uint divisor = 64u / STRIDE_BYTES > 1u
            ? 64u / STRIDE_BYTES : 1u;
        const uint byteOffset = row * STRIDE_BYTES + col * ELEM_BYTES;
        const uint mask = ((row % PERIOD) / divisor) << 4u;
        return (byteOffset ^ mask) / ELEM_BYTES;
    }

    FK_HOST_DEVICE_FUSE uint size() { return ROWS * COLS; }
};

// A Tile is only a statically shaped view over DPP-owned local storage.
// It does not own memory and is neither an Operation nor a DPP.
template <typename T, typename Layout>
struct Tile {
    using ValueType = T;
    using LayoutType = Layout;
    static constexpr uint rows = Layout::rows;
    static constexpr uint cols = Layout::cols;
    static constexpr uint elements = Layout::size();

    T* storage;

    FK_HOST_DEVICE_CNST explicit Tile(T* localStorage)
        : storage(localStorage) {}

    FK_HOST_DEVICE_CNST T& at(const uint row, const uint col) const {
        return storage[Layout::offset(row, col)];
    }

    FK_HOST_DEVICE_CNST T* data() const { return storage; }
};

} // namespace fk

#endif // FK_COLLECTIVE_TILE_H

