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

#ifndef FK_COLLECTIVE_MMA_H
#define FK_COLLECTIVE_MMA_H

#include <fused_kernel/core/data/point.h>
#include <fused_kernel/core/execution_model/operation_model/operation_model.h>
#include <fused_kernel/core/utils/utils.h>

#include <cstdint>
#include <cstring>
#include <type_traits>

#if defined(__NVCC__)
#include <cuda_bf16.h>
#endif

namespace fk {

struct MmaBf16_16x8x16 {
    static constexpr int M = 16;
    static constexpr int N = 8;
    static constexpr int K = 16;
    static constexpr int A_REGS = 4;
    static constexpr int B_REGS = 2;
    static constexpr int D_REGS = 4;
};

template <typename Atom>
struct MmaDPPDetails {
    using AtomType = Atom;
    static constexpr int M = Atom::M;
    static constexpr int N = Atom::N;
    static constexpr int K = Atom::K;

    int aOriginX;
    int aOriginY;
    int bOriginX;
    int bOriginY;
    int dOriginX;
    int dOriginY;
};

template <ParArch PA, typename DPPDetails>
struct MmaWarpDPP;

template <typename DPPDetails>
struct MmaWarpDPP<ParArch::CPU, DPPDetails> {
private:
    using SelfType = MmaWarpDPP<ParArch::CPU, DPPDetails>;

public:
    FK_STATIC_STRUCT(MmaWarpDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::CPU;

    template <typename AReadIOp, typename BReadIOp, typename DWriteIOp>
    FK_HOST_STATIC void exec(const DPPDetails& details,
                             const AReadIOp& a,
                             const BReadIOp& b,
                             const DWriteIOp& d) {
        static_assert(isAnyCompleteReadType<AReadIOp> &&
                      isAnyCompleteReadType<BReadIOp>,
                      "MmaWarpDPP requires complete A/B Read IOps");
        static_assert(isAnyWriteType<DWriteIOp>,
                      "MmaWarpDPP requires a D Write IOp");
        for (int row = 0; row < DPPDetails::M; ++row) {
            for (int pair = 0; pair < DPPDetails::N / 2; ++pair) {
                float accum0 = 0.f;
                float accum1 = 0.f;
                for (int k = 0; k < DPPDetails::K; ++k) {
                    const float av = static_cast<float>(
                        AReadIOp::Operation::exec(
                            Point{details.aOriginX + k,
                                  details.aOriginY + row, 0}, a));
                    const float bv0 = static_cast<float>(
                        BReadIOp::Operation::exec(
                            Point{details.bOriginX + k,
                                  details.bOriginY + 2 * pair, 0}, b));
                    const float bv1 = static_cast<float>(
                        BReadIOp::Operation::exec(
                            Point{details.bOriginX + k,
                                  details.bOriginY + 2 * pair + 1, 0}, b));
                    accum0 += av * bv0;
                    accum1 += av * bv1;
                }
                DWriteIOp::Operation::exec(
                    Point{details.dOriginX + pair,
                          details.dOriginY + row, 0},
                    float2{accum0, accum1}, d);
            }
        }
    }
};

#if defined(__NVCC__)
template <typename DPPDetails>
struct MmaWarpDPP<ParArch::GPU_NVIDIA, DPPDetails> {
private:
    using SelfType = MmaWarpDPP<ParArch::GPU_NVIDIA, DPPDetails>;

    FK_DEVICE_STATIC uint32_t pack(const __nv_bfloat16 first,
                                   const __nv_bfloat16 second) {
        const __nv_bfloat16 pair[2]{first, second};
        uint32_t result;
        memcpy(&result, pair, sizeof(result));
        return result;
    }

    template <typename ReadIOp>
    FK_DEVICE_STATIC __nv_bfloat16 readBf16(
            const ReadIOp& input, const Point point) {
        const auto value = ReadIOp::Operation::exec(point, input);
        if constexpr (std::is_same_v<decltype(value), const uint16_t> ||
                      std::is_same_v<decltype(value), uint16_t>) {
            __nv_bfloat16 result;
            memcpy(&result, &value, sizeof(result));
            return result;
        } else {
            return __float2bfloat16(static_cast<float>(value));
        }
    }

    // Tensor-core MMA is the narrow exception that directly updates each
    // lane's register fragment. This helper is private to the warp DPP.
    FK_DEVICE_STATIC void mma(const uint32_t (&a)[4],
                              const uint32_t (&b)[2],
                              float (&d)[4]) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(b[0]), "r"(b[1]));
    }

public:
    FK_STATIC_STRUCT(MmaWarpDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::GPU_NVIDIA;

    template <typename AReadIOp, typename BReadIOp, typename DWriteIOp>
    FK_DEVICE_STATIC void exec(const DPPDetails& details,
                               const AReadIOp& aInput,
                               const BReadIOp& bInput,
                               const DWriteIOp& dOutput) {
        static_assert(isAnyCompleteReadType<AReadIOp> &&
                      isAnyCompleteReadType<BReadIOp>,
                      "MmaWarpDPP requires complete A/B Read IOps");
        static_assert(isAnyWriteType<DWriteIOp>,
                      "MmaWarpDPP requires a D Write IOp");
        static_assert(std::is_same_v<typename DPPDetails::AtomType,
                                     MmaBf16_16x8x16>,
                      "GPU specialization currently supports bf16 m16n8k16");
        if (threadIdx.x >= 32) return;

        const int lane = threadIdx.x;
        const int group = lane >> 2;
        const int threadInGroup = lane & 3;
        const int k0 = 2 * threadInGroup;
        const int k1 = k0 + 8;
        const int row0 = group;
        const int row1 = group + 8;

        uint32_t a[4];
        uint32_t b[2];
        a[0] = pack(
            readBf16(aInput, Point{details.aOriginX + k0,
                                    details.aOriginY + row0, 0}),
            readBf16(aInput, Point{details.aOriginX + k0 + 1,
                                    details.aOriginY + row0, 0}));
        a[1] = pack(
            readBf16(aInput, Point{details.aOriginX + k0,
                                    details.aOriginY + row1, 0}),
            readBf16(aInput, Point{details.aOriginX + k0 + 1,
                                    details.aOriginY + row1, 0}));
        a[2] = pack(
            readBf16(aInput, Point{details.aOriginX + k1,
                                    details.aOriginY + row0, 0}),
            readBf16(aInput, Point{details.aOriginX + k1 + 1,
                                    details.aOriginY + row0, 0}));
        a[3] = pack(
            readBf16(aInput, Point{details.aOriginX + k1,
                                    details.aOriginY + row1, 0}),
            readBf16(aInput, Point{details.aOriginX + k1 + 1,
                                    details.aOriginY + row1, 0}));
        b[0] = pack(
            readBf16(bInput, Point{details.bOriginX + k0,
                                    details.bOriginY + group, 0}),
            readBf16(bInput, Point{details.bOriginX + k0 + 1,
                                    details.bOriginY + group, 0}));
        b[1] = pack(
            readBf16(bInput, Point{details.bOriginX + k1,
                                    details.bOriginY + group, 0}),
            readBf16(bInput, Point{details.bOriginX + k1 + 1,
                                    details.bOriginY + group, 0}));

        float d[4]{0.f, 0.f, 0.f, 0.f};
        mma(a, b, d);

        // Each lane writes two adjacent output columns as float2. The four
        // lanes in a group cover a full 8-column row with coalesced stores.
        DWriteIOp::Operation::exec(
            Point{details.dOriginX + threadInGroup,
                  details.dOriginY + row0, 0},
            float2{d[0], d[1]}, dOutput);
        DWriteIOp::Operation::exec(
            Point{details.dOriginX + threadInGroup,
                  details.dOriginY + row1, 0},
            float2{d[2], d[3]}, dOutput);
    }
};
#endif // defined(__NVCC__)

} // namespace fk

#endif // FK_COLLECTIVE_MMA_H
