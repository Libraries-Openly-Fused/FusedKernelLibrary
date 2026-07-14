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

