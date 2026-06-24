/* Copyright 2023-2026 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_BITWISE
#define FK_BITWISE

#include <fused_kernel/core/execution_model/operation_model/operation_model.h>
#include <fused_kernel/core/data/tuple.h>
#include <fused_kernel/core/utils/vector_utils.h>

namespace fk {
    // Bitwise AND. Binary form ANDs with a constant; Unary form ANDs two inputs
    // packed in a Tuple (e.g. two images read per-thread).
    template <typename I1, typename I2 = I1, typename O = I1, typename IT = BinaryType>
    struct BwAnd;

    template <typename I, typename P, typename O>
    struct BwAnd<I, P, O, BinaryType> {
    private:
        using SelfType = BwAnd<I, P, O, BinaryType>;
    public:
        FK_STATIC_STRUCT(BwAnd, SelfType)
        using Parent = BinaryOperation<I, P, O, BwAnd<I, P, O, BinaryType>>;
        DECLARE_BINARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType input, const ParamsType& params) {
            return input & params;
        }
    };

    template <typename I1, typename I2, typename O>
    struct BwAnd<I1, I2, O, UnaryType> {
    private:
        using SelfType = BwAnd<I1, I2, O, UnaryType>;
    public:
        FK_STATIC_STRUCT(BwAnd, SelfType)
        using Parent = UnaryOperation<Tuple<I1, I2>, O, BwAnd<I1, I2, O, UnaryType>>;
        DECLARE_UNARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType input) {
            return get<0>(input) & get<1>(input);
        }
    };

    // Bitwise OR.
    template <typename I1, typename I2 = I1, typename O = I1, typename IT = BinaryType>
    struct BwOr;

    template <typename I, typename P, typename O>
    struct BwOr<I, P, O, BinaryType> {
    private:
        using SelfType = BwOr<I, P, O, BinaryType>;
    public:
        FK_STATIC_STRUCT(BwOr, SelfType)
        using Parent = BinaryOperation<I, P, O, BwOr<I, P, O, BinaryType>>;
        DECLARE_BINARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType input, const ParamsType& params) {
            return input | params;
        }
    };

    template <typename I1, typename I2, typename O>
    struct BwOr<I1, I2, O, UnaryType> {
    private:
        using SelfType = BwOr<I1, I2, O, UnaryType>;
    public:
        FK_STATIC_STRUCT(BwOr, SelfType)
        using Parent = UnaryOperation<Tuple<I1, I2>, O, BwOr<I1, I2, O, UnaryType>>;
        DECLARE_UNARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType input) {
            return get<0>(input) | get<1>(input);
        }
    };

    // Bitwise XOR.
    template <typename I1, typename I2 = I1, typename O = I1, typename IT = BinaryType>
    struct BwXor;

    template <typename I, typename P, typename O>
    struct BwXor<I, P, O, BinaryType> {
    private:
        using SelfType = BwXor<I, P, O, BinaryType>;
    public:
        FK_STATIC_STRUCT(BwXor, SelfType)
        using Parent = BinaryOperation<I, P, O, BwXor<I, P, O, BinaryType>>;
        DECLARE_BINARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType input, const ParamsType& params) {
            return input ^ params;
        }
    };

    template <typename I1, typename I2, typename O>
    struct BwXor<I1, I2, O, UnaryType> {
    private:
        using SelfType = BwXor<I1, I2, O, UnaryType>;
    public:
        FK_STATIC_STRUCT(BwXor, SelfType)
        using Parent = UnaryOperation<Tuple<I1, I2>, O, BwXor<I1, I2, O, UnaryType>>;
        DECLARE_UNARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType input) {
            return get<0>(input) ^ get<1>(input);
        }
    };

    // Bitwise NOT (unary, single input).
    template <typename I, typename O = I>
    struct BwNot {
    private:
        using SelfType = BwNot<I, O>;
    public:
        FK_STATIC_STRUCT(BwNot, SelfType)
        using Parent = UnaryOperation<I, O, BwNot<I, O>>;
        DECLARE_UNARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType input) {
            return ~input;
        }
    };

    // Left shift by a compile-time-or-runtime constant amount.
    template <typename I, typename P = I, typename O = I>
    struct ShiftLeft {
    private:
        using SelfType = ShiftLeft<I, P, O>;
    public:
        FK_STATIC_STRUCT(ShiftLeft, SelfType)
        using Parent = BinaryOperation<I, P, O, ShiftLeft<I, P, O>>;
        DECLARE_BINARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType input, const ParamsType& params) {
            return input << params;
        }
    };

    // Right shift by a constant amount.
    template <typename I, typename P = I, typename O = I>
    struct ShiftRight {
    private:
        using SelfType = ShiftRight<I, P, O>;
    public:
        FK_STATIC_STRUCT(ShiftRight, SelfType)
        using Parent = BinaryOperation<I, P, O, ShiftRight<I, P, O>>;
        DECLARE_BINARY_PARENT
        FK_HOST_DEVICE_FUSE OutputType exec(const InputType input, const ParamsType& params) {
            return input >> params;
        }
    };
} // namespace fk

#endif
