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

#ifndef FK_STATIC_MATRIX_H
#define FK_STATIC_MATRIX_H

#include <fused_kernel/core/utils/vector_utils.h>

namespace fk {
struct M3x3Float {
    const float3 x;
    const float3 y;
    const float3 z;
};
} // namespace fk

// Matrix * Scalar
FK_HOST_DEVICE_CNST fk::M3x3Float operator*(const fk::M3x3Float &matrix, float scalar) {
    return {matrix.x * scalar, matrix.y * scalar, matrix.z * scalar};
}

// Scalar * Matrix (Commutative)
FK_HOST_DEVICE_CNST fk::M3x3Float operator*(float scalar, const fk::M3x3Float &matrix) {
    return matrix * scalar; // Re-uses the implementation above
}

// Element-wise (Hadamard) multiplication. Note: this is NOT standard matrix multiplication.
FK_HOST_DEVICE_CNST fk::M3x3Float operator*(const fk::M3x3Float &matrix1, const fk::M3x3Float &matrix2) {
    return {matrix1.x * matrix2.x, matrix1.y * matrix2.y, matrix1.z * matrix2.z}; // Re-uses the implementation above
}

#endif // !FK_STATIC_MATRIX_H
