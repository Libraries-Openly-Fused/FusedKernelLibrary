/* Copyright 2023-2026 Oscar Amoros Huguet, Johnny Nunez

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_MORPHOLOGY
#define FK_MORPHOLOGY

#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/logical.h>
#include <fused_kernel/core/data/tuple.h>
#include <fused_kernel/core/execution_model/operation_model/operation_model.h>
#include <fused_kernel/core/execution_model/stream.h>
#include <fused_kernel/core/utils/utils.h>

#include <type_traits>
#include <utility>
#include <vector>

#if defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)
#include <fused_kernel/core/execution_model/cpu_thread_pool.h>
#include <array>
#include <cstddef>
#include <exception>
#include <memory>
#endif

namespace fk {

template <typename T, int TILE_W = 16, int TILE_H = 8,
          int MAX_MASK_W = 7, int MAX_MASK_H = 7>
struct MorphologyDPPDetails {
    using ValueType = T;
    static constexpr int TILE_WIDTH = TILE_W;
    static constexpr int TILE_HEIGHT = TILE_H;
    static constexpr int MAX_MASK_WIDTH = MAX_MASK_W;
    static constexpr int MAX_MASK_HEIGHT = MAX_MASK_H;
    static constexpr int MAX_PASSES = 4;
    // Elements of one shared memory stage of the GPU version running
    // PASSES passes: the tile plus the halo of every pass, for masks of up
    // to MAX_MASK_W x MAX_MASK_H
    template <int PASSES>
    static constexpr int STAGE_ELEMENTS =
        (TILE_W + PASSES * (MAX_MASK_W - 1)) *
        (TILE_H + PASSES * (MAX_MASK_H - 1));
    static constexpr int MAX_STAGE_WIDTH =
        TILE_W + MAX_PASSES * (MAX_MASK_W - 1);
    static constexpr int MAX_STAGE_HEIGHT =
        TILE_H + MAX_PASSES * (MAX_MASK_H - 1);
    static constexpr int MAX_STAGE_ELEMENTS = STAGE_ELEMENTS<MAX_PASSES>;

    static_assert(TILE_W > 0 && TILE_H > 0,
                  "Morphology tile dimensions must be positive");
    static_assert(MAX_MASK_W > 0 && MAX_MASK_H > 0,
                  "Morphology maximum mask dimensions must be positive");

    int width;
    int height;
    int maskW;
    int maskH;
    int anchorX;
    int anchorY;

    FK_HOST_DEVICE_FUSE bool valid(const MorphologyDPPDetails& details) {
        return details.width > 0 && details.height > 0 &&
               details.maskW > 0 && details.maskH > 0 &&
               details.maskW <= MAX_MASK_W &&
               details.maskH <= MAX_MASK_H &&
               details.anchorX >= 0 && details.anchorX < details.maskW &&
               details.anchorY >= 0 && details.anchorY < details.maskH;
    }
};

namespace morphology_detail {

template <int Index = 0, typename T, typename Reducers>
FK_HOST_DEVICE_STATIC T applyReducer(const int pass, const T a, const T b,
                                     const Reducers& reducers) {
    if constexpr (Index < std::decay_t<Reducers>::size) {
        if (pass == Index) {
            return static_cast<T>(make_tuple(a, b) | get<Index>(reducers));
        }
        return applyReducer<Index + 1>(pass, a, b, reducers);
    } else {
        return a;
    }
}

template <int Index = 0, typename Reducers>
FK_HOST_DEVICE_CNST bool reducersAreUnary() {
    if constexpr (Index == std::decay_t<Reducers>::size) {
        return true;
    } else {
        using Reducer = std::decay_t<decltype(
            get<Index>(std::declval<const Reducers&>()))>;
        return Reducer::template is<UnaryType> &&
               reducersAreUnary<Index + 1, Reducers>();
    }
}

FK_HOST_DEVICE_FUSE int clamp(const int value, const int low,
                              const int high) {
    return value < low ? low : (value > high ? high : value);
}

// The planes one execution processes: z of the input's active threads, that
// is the batch of a batched read (for instance Crop of an array of
// rectangles, then Resize), the planes of a 3D input, and 1 otherwise. Each
// plane z is a details.width x details.height image, read and written at
// Point{x, y, z}, with its own border.
template <typename InputReadIOp>
FK_HOST_FUSE int planesOf(const InputReadIOp& input) {
    return static_cast<int>(input.getActiveThreads().z);
}

// The box blur (executeBoxBlur) as a morphology: an Add reducer per pass, so
// that every pass sums its window, and a Mul by 1 / (maskW * maskH)^PASSES in
// front of the output, so that the sums of every pass become the mean of
// every pass. The passes in between keep the sums, which for floating point
// T is the mean times a constant, up to rounding.
template <typename T, int PASSES>
FK_HOST_FUSE auto boxBlurReducers() {
    static_assert(std::is_floating_point_v<VBase<T>>,
                  "The box blur sums in T: T must be a floating point type "
                  "or a vector of them");
    static_assert(PASSES >= 1 && PASSES <= 4,
                  "MorphologyDPP requires 1 through 4 reducers");
    const auto add = Add<T, T, T, UnaryType>::build();
    if constexpr (PASSES == 1) {
        return make_tuple(add);
    } else if constexpr (PASSES == 2) {
        return make_tuple(add, add);
    } else if constexpr (PASSES == 3) {
        return make_tuple(add, add, add);
    } else {
        return make_tuple(add, add, add, add);
    }
}

template <typename T, int PASSES, typename DPPDetails, typename OutputWriteIOp>
FK_HOST_FUSE auto boxBlurWrite(const DPPDetails& details,
                               const OutputWriteIOp& output) {
    double windows = 1.;
    for (int pass = 0; pass < PASSES; ++pass) {
        windows *= static_cast<double>(details.maskW) * details.maskH;
    }
    // In the precision of T's elements (float or double)
    const VBase<T> scale = static_cast<VBase<T>>(1. / windows);
    if constexpr (std::is_arithmetic_v<T>) {
        return Mul<T>::build(scale).then(output);
    } else {
        return Mul<T>::build(make_set<T>(scale)).then(output);
    }
}

} // namespace morphology_detail

template <ParArch PA, typename DPPDetails>
struct MorphologyDPP;

template <typename DPPDetails>
struct MorphologyDPP<ParArch::CPU, DPPDetails> {
private:
    using SelfType = MorphologyDPP<ParArch::CPU, DPPDetails>;
    using T = typename DPPDetails::ValueType;

    // Value of pixel (x, y) after pass `pass`, reduced from source, the
    // whole image (width * height) the previous phase produced
    template <typename Reducers>
    FK_HOST_STATIC T reduce_pixel(const int x, const int y, const int pass,
                                  const DPPDetails& details, const T* source,
                                  const Reducers& reducers) {
        bool first = true;
        T accumulator{};
        for (int my = 0; my < details.maskH; ++my) {
            const int sy = morphology_detail::clamp(
                y + my - details.anchorY, 0, details.height - 1);
            for (int mx = 0; mx < details.maskW; ++mx) {
                const int sx = morphology_detail::clamp(
                    x + mx - details.anchorX, 0, details.width - 1);
                const T value =
                    source[static_cast<size_t>(sy * details.width + sx)];
                if (first) {
                    accumulator = value;
                    first = false;
                } else {
                    accumulator = morphology_detail::applyReducer(
                        pass, accumulator, value, reducers);
                }
            }
        }
        return accumulator;
    }

public:
    FK_STATIC_STRUCT(MorphologyDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::CPU;

    template <typename Reducers>
    static constexpr bool validReducerCount =
        isTuple_v<Reducers> && std::decay_t<Reducers>::size >= 1 &&
        std::decay_t<Reducers>::size <= DPPDetails::MAX_PASSES;

    // The work runs in phases, each one over every row of the image and
    // each one reading only what the previous phases wrote: load_row
    // (input -> image), reduce_row for every pass but the last (image ->
    // image of the next pass) and reduce_write_row for the last pass
    // (image -> output). The images are whole width * height buffers, so a
    // row of a pass may read any row of the previous one. Since the whole
    // input of a plane is loaded before anything of it is written, the
    // output may be the input. Every plane of the input
    // (morphology_detail::planesOf) is processed in turn, each one with its
    // own images. ParArch::CPU_MULTITHREAD runs each phase as one chunk per
    // row of every plane.

    // Loads row y of plane `plane` of the input into row y of image
    template <typename InputReadIOp>
    FK_HOST_STATIC void load_row(const int y, const int plane,
                                 const DPPDetails& details,
                                 const InputReadIOp& input, T* image) {
        for (int x = 0; x < details.width; ++x)
            image[static_cast<size_t>(y * details.width + x)] =
                static_cast<T>(InputReadIOp::Operation::exec(
                    Point{x, y, plane}, input));
    }

    // Computes row y of pass `pass` from source into target
    template <typename Reducers>
    FK_HOST_STATIC void reduce_row(const int y, const int pass,
                                   const DPPDetails& details,
                                   const T* source, T* target,
                                   const Reducers& reducers) {
        for (int x = 0; x < details.width; ++x)
            target[static_cast<size_t>(y * details.width + x)] =
                reduce_pixel(x, y, pass, details, source, reducers);
    }

    // Computes row y of pass `pass` (the last one) from source, and writes
    // it with output as row y of plane `plane`
    template <typename OutputWriteIOp, typename Reducers>
    FK_HOST_STATIC void reduce_write_row(const int y, const int plane,
                                         const int pass,
                                         const DPPDetails& details,
                                         const T* source,
                                         const OutputWriteIOp& output,
                                         const Reducers& reducers) {
        for (int x = 0; x < details.width; ++x)
            OutputWriteIOp::Operation::exec(
                Point{x, y, plane},
                reduce_pixel(x, y, pass, details, source, reducers),
                output);
    }

    template <typename InputReadIOp, typename OutputWriteIOp,
              typename Reducers>
    FK_HOST_STATIC void exec(const DPPDetails& details,
                             const InputReadIOp& input,
                             const OutputWriteIOp& output,
                             const Reducers& reducers) {
        static_assert(validReducerCount<Reducers>,
                      "MorphologyDPP requires 1 through 4 reducers");
        static_assert(morphology_detail::reducersAreUnary<0, Reducers>(),
                      "Morphology reducers must be Unary IOps");
        static_assert(isAnyCompleteReadType<InputReadIOp>,
                      "Morphology input must be a complete Read IOp");
        static_assert(isAnyWriteType<OutputWriteIOp>,
                      "Morphology output must be a Write IOp");
        if (!DPPDetails::valid(details)) return;

        constexpr int PASSES =
            static_cast<int>(std::decay_t<Reducers>::size);
        const int elements = details.width * details.height;
        const int planes = morphology_detail::planesOf(input);
        std::vector<T> current(static_cast<size_t>(elements));
        std::vector<T> next(static_cast<size_t>(elements));
        for (int plane = 0; plane < planes; ++plane) {
            for (int y = 0; y < details.height; ++y)
                load_row(y, plane, details, input, current.data());

            for (int pass = 0; pass < PASSES - 1; ++pass) {
                for (int y = 0; y < details.height; ++y)
                    reduce_row(y, pass, details, current.data(), next.data(),
                               reducers);
                current.swap(next);
            }

            for (int y = 0; y < details.height; ++y)
                reduce_write_row(y, plane, PASSES - 1, details,
                                 current.data(), output, reducers);
        }
    }
};

#if defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)
// Runs the phases of MorphologyDPP<ParArch::CPU> (load, then one per pass,
// the last one writing the output) on cpuThreadPool(), each phase as one
// chunk per row of every plane, enqueued when the previous phase has
// finished. So the results are the same as with ParArch::CPU, and as there,
// the output may be the input. The buffers the IOps use must stay valid
// until done is called.
// The lambdas capture reducers as `reducers = reducers`, a copy that is not
// const: the pool and the stream move them, and moving a const fk::Tuple of
// one element selects Tuple's forwarding constructor instead of its copy
// constructor, which does not compile.
template <typename DPPDetails>
struct MorphologyDPP<ParArch::CPU_MULTITHREAD, DPPDetails> {
private:
    using SelfType = MorphologyDPP<ParArch::CPU_MULTITHREAD, DPPDetails>;
    using CPUDPP = MorphologyDPP<ParArch::CPU, DPPDetails>;
    using T = typename DPPDetails::ValueType;
    // The images the load phase writes ([0]) and the ones of the next pass
    // ([1]), plane after plane (width * height elements each); pass p reads
    // [p % 2] and writes [(p + 1) % 2]. Shared by the chunks of all the
    // phases of one execution.
    using Images = std::array<std::vector<T>, 2>;

    // Chunk c of a phase is row c % height of plane c / height; its image
    // starts at element (c / height) * width * height
    FK_HOST_FUSE int rowOf(const std::size_t chunk,
                           const DPPDetails& details) {
        return static_cast<int>(chunk %
                                static_cast<std::size_t>(details.height));
    }
    FK_HOST_FUSE int planeOf(const std::size_t chunk,
                             const DPPDetails& details) {
        return static_cast<int>(chunk /
                                static_cast<std::size_t>(details.height));
    }
    FK_HOST_FUSE std::size_t planeOffset(const int plane,
                                         const DPPDetails& details) {
        return static_cast<std::size_t>(plane) *
               static_cast<std::size_t>(details.width) *
               static_cast<std::size_t>(details.height);
    }

    template <typename Reducers>
    static constexpr int PASSES =
        static_cast<int>(std::decay_t<Reducers>::size);

    // Enqueues the chunks of pass `pass`, one per row of every plane. The
    // last pass writes the output and then calls done; the others start the
    // next pass when they finish.
    template <typename OutputWriteIOp, typename Reducers>
    FK_HOST_STATIC void enqueuePass(const int pass, const int planes,
                                    const cpu_multithread::Completion& done,
                                    const DPPDetails& details,
                                    const OutputWriteIOp& output,
                                    const Reducers& reducers,
                                    const std::shared_ptr<Images>& images) {
        const auto chunks = static_cast<std::size_t>(details.height) *
                            static_cast<std::size_t>(planes);
        const auto source = static_cast<std::size_t>(pass % 2);
        if (pass == PASSES<Reducers> - 1) {
            cpu_multithread::enqueueChunks(chunks,
                [pass, source, details, output, reducers = reducers,
                 images](const std::size_t chunk) {
                    const int plane = planeOf(chunk, details);
                    CPUDPP::reduce_write_row(
                        rowOf(chunk, details), plane, pass, details,
                        (*images)[source].data() +
                            planeOffset(plane, details),
                        output, reducers);
                }, done);
        } else {
            cpu_multithread::enqueueChunks(chunks,
                [pass, source, details, reducers = reducers,
                 images](const std::size_t chunk) {
                    const std::size_t offset =
                        planeOffset(planeOf(chunk, details), details);
                    CPUDPP::reduce_row(rowOf(chunk, details), pass, details,
                                       (*images)[source].data() + offset,
                                       (*images)[1 - source].data() + offset,
                                       reducers);
                },
                [pass, planes, done, details, output, reducers = reducers,
                 images](const std::exception_ptr error) noexcept {
                    startPass(pass + 1, planes, done, error, details, output,
                              reducers, images);
                });
        }
    }

    // Called when the previous phase has finished: enqueues pass `pass`,
    // or ends the execution if that phase, or enqueueing, failed
    template <typename OutputWriteIOp, typename Reducers>
    FK_HOST_STATIC void startPass(const int pass, const int planes,
                                  const cpu_multithread::Completion& done,
                                  const std::exception_ptr& error,
                                  const DPPDetails& details,
                                  const OutputWriteIOp& output,
                                  const Reducers& reducers,
                                  const std::shared_ptr<Images>& images)
        noexcept {
        if (error) {
            done(error);
            return;
        }
        try {
            enqueuePass(pass, planes, done, details, output, reducers,
                        images);
        } catch (...) {
            done(std::current_exception());
        }
    }

public:
    FK_STATIC_STRUCT(MorphologyDPP, SelfType)
    static constexpr ParArch PAR_ARCH = ParArch::CPU_MULTITHREAD;

    template <typename Reducers>
    static constexpr bool validReducerCount =
        CPUDPP::template validReducerCount<Reducers>;

    // Allocates the images, enqueues the load phase and returns; done is
    // called when the last pass has written the output (right away if the
    // details are not valid, doing nothing). If this throws (allocating, or
    // enqueueing the load phase), nothing was enqueued and done is not
    // called.
    template <typename InputReadIOp, typename OutputWriteIOp,
              typename Reducers>
    FK_HOST_STATIC void exec(const cpu_multithread::Completion& done,
                             const DPPDetails& details,
                             const InputReadIOp& input,
                             const OutputWriteIOp& output,
                             const Reducers& reducers) {
        static_assert(validReducerCount<Reducers>,
                      "MorphologyDPP requires 1 through 4 reducers");
        static_assert(morphology_detail::reducersAreUnary<0, Reducers>(),
                      "Morphology reducers must be Unary IOps");
        static_assert(isAnyCompleteReadType<InputReadIOp>,
                      "Morphology input must be a complete Read IOp");
        static_assert(isAnyWriteType<OutputWriteIOp>,
                      "Morphology output must be a Write IOp");
        if (!DPPDetails::valid(details)) {
            done();
            return;
        }

        const int planes = morphology_detail::planesOf(input);
        const auto chunks = static_cast<std::size_t>(details.height) *
                            static_cast<std::size_t>(planes);
        // Every plane: the offset of the plane after the last one
        const auto elements = planeOffset(planes, details);
        const auto images = std::make_shared<Images>();
        (*images)[0].resize(elements);
        if constexpr (PASSES<Reducers> > 1) {
            (*images)[1].resize(elements);
        }
        cpu_multithread::enqueueChunks(chunks,
            [details, input, images](const std::size_t chunk) {
                const int plane = planeOf(chunk, details);
                CPUDPP::load_row(rowOf(chunk, details), plane, details, input,
                                 (*images)[0].data() +
                                     planeOffset(plane, details));
            },
            [planes, done, details, output, reducers = reducers,
             images](const std::exception_ptr error) noexcept {
                startPass(0, planes, done, error, details, output, reducers,
                          images);
            });
    }
};
#endif // defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)

#if defined(__NVCC__) || defined(__HIPCC__)
// One block per TILE_WIDTH x TILE_HEIGHT tile of a plane, and one grid z
// plane per plane of the input (morphology_detail::planesOf). The block
// stages the tile and the halo of every pass in shared memory, sized for the
// number of passes (DPPDetails::STAGE_ELEMENTS), and runs the passes there.
template <ParArch PA, typename DPPDetails>
    requires(PA == ParArch::GPU_NVIDIA || PA == ParArch::GPU_AMD)
struct MorphologyDPP<PA, DPPDetails> {
private:
    using SelfType = MorphologyDPP<PA, DPPDetails>;
    using T = typename DPPDetails::ValueType;

    template <typename InputReadIOp>
    FK_DEVICE_FUSE T readClamped(const DPPDetails& details,
                                 const InputReadIOp& input,
                                 const int x, const int y, const int plane) {
        const int sx = morphology_detail::clamp(x, 0, details.width - 1);
        const int sy = morphology_detail::clamp(y, 0, details.height - 1);
        return static_cast<T>(InputReadIOp::Operation::exec(
            Point{sx, sy, plane}, input));
    }

public:
    FK_STATIC_STRUCT(MorphologyDPP, SelfType)
    static constexpr ParArch PAR_ARCH = PA;

    template <typename Reducers>
    static constexpr bool validReducerCount =
        isTuple_v<Reducers> && std::decay_t<Reducers>::size >= 1 &&
        std::decay_t<Reducers>::size <= DPPDetails::MAX_PASSES;

    template <typename InputReadIOp, typename OutputWriteIOp,
              typename Reducers>
    FK_HOST_DEVICE_STATIC void exec(const DPPDetails& details,
                                    const InputReadIOp& input,
                                    const OutputWriteIOp& output,
                                    const Reducers& reducers) {
        static_assert(validReducerCount<Reducers>,
                      "MorphologyDPP requires 1 through 4 reducers");
        static_assert(morphology_detail::reducersAreUnary<0, Reducers>(),
                      "Morphology reducers must be Unary IOps");
        static_assert(isAnyCompleteReadType<InputReadIOp>,
                      "Morphology input must be a complete Read IOp");
        static_assert(isAnyWriteType<OutputWriteIOp>,
                      "Morphology output must be a Write IOp");
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        constexpr int PASSES = std::decay_t<Reducers>::size;
        constexpr int STAGE_ELEMENTS =
            DPPDetails::template STAGE_ELEMENTS<PASSES>;
        // A single pass only reads stage0
        __shared__ T stage0[STAGE_ELEMENTS];
        __shared__ T stage1[PASSES > 1 ? STAGE_ELEMENTS : 1];

        const int tileX = blockIdx.x * DPPDetails::TILE_WIDTH;
        const int tileY = blockIdx.y * DPPDetails::TILE_HEIGHT;
        const int plane = static_cast<int>(blockIdx.z);
        const int tid = threadIdx.y * blockDim.x + threadIdx.x;
        const int threads = blockDim.x * blockDim.y;
        const int initialW = DPPDetails::TILE_WIDTH +
                             PASSES * (details.maskW - 1);
        const int initialH = DPPDetails::TILE_HEIGHT +
                             PASSES * (details.maskH - 1);
        const int initialX = tileX - PASSES * details.anchorX;
        const int initialY = tileY - PASSES * details.anchorY;

        for (int index = tid; index < initialW * initialH;
             index += threads) {
            const int lx = index % initialW;
            const int ly = index / initialW;
            stage0[index] = readClamped(
                details, input, initialX + lx, initialY + ly, plane);
        }
        __syncthreads();

        T* source = stage0;
        T* target = stage1;
        T finalValue{};
        #pragma unroll
        for (int pass = 0; pass < PASSES; ++pass) {
            const int remainingSource = PASSES - pass;
            const int remainingTarget = remainingSource - 1;
            const int sourceW = DPPDetails::TILE_WIDTH +
                                remainingSource * (details.maskW - 1);
            const int targetW = DPPDetails::TILE_WIDTH +
                                remainingTarget * (details.maskW - 1);
            const int targetH = DPPDetails::TILE_HEIGHT +
                                remainingTarget * (details.maskH - 1);
            const int sourceX = tileX -
                                remainingSource * details.anchorX;
            const int sourceY = tileY -
                                remainingSource * details.anchorY;
            const int targetX = tileX -
                                remainingTarget * details.anchorX;
            const int targetY = tileY -
                                remainingTarget * details.anchorY;
            const bool finalPass = pass == PASSES - 1;
            const int targetElements = targetW * targetH;

            for (int index = tid; index < targetElements;
                 index += threads) {
                const int lx = index % targetW;
                const int ly = index / targetW;
                const int centerX = morphology_detail::clamp(
                    targetX + lx, 0, details.width - 1);
                const int centerY = morphology_detail::clamp(
                    targetY + ly, 0, details.height - 1);
                bool first = true;
                T accumulator{};
                for (int my = 0; my < details.maskH; ++my) {
                    const int sampleY = morphology_detail::clamp(
                        centerY + my - details.anchorY,
                        0, details.height - 1);
                    const int sourceLY = sampleY - sourceY;
                    for (int mx = 0; mx < details.maskW; ++mx) {
                        const int sampleX = morphology_detail::clamp(
                            centerX + mx - details.anchorX,
                            0, details.width - 1);
                        const int sourceLX = sampleX - sourceX;
                        const T value = source[sourceLY * sourceW + sourceLX];
                        if (first) {
                            accumulator = value;
                            first = false;
                        } else {
                            accumulator = morphology_detail::applyReducer(
                                pass, accumulator, value, reducers);
                        }
                    }
                }
                if (finalPass)
                    finalValue = accumulator;
                else
                    target[index] = accumulator;
            }
            if (!finalPass) {
                __syncthreads();
                T* temporary = source;
                source = target;
                target = temporary;
            }
        }

        const int outputX = tileX + threadIdx.x;
        const int outputY = tileY + threadIdx.y;
        if (threadIdx.x < DPPDetails::TILE_WIDTH &&
            threadIdx.y < DPPDetails::TILE_HEIGHT &&
            outputX < details.width && outputY < details.height) {
            OutputWriteIOp::Operation::exec(
                Point{outputX, outputY, plane}, finalValue, output);
        }
#else
        (void)details;
        (void)input;
        (void)output;
        (void)reducers;
#endif
    }
};

template <ParArch PA, typename DPPDetails, typename InputReadIOp,
          typename OutputWriteIOp, typename Reducers>
__global__ void morphologyDPPKernel(const DPPDetails details,
                                    const InputReadIOp input,
                                    const OutputWriteIOp output,
                                    const Reducers reducers) {
    MorphologyDPP<PA, DPPDetails>::exec(
        details, input, output, reducers);
}

template <typename DPPDetails, typename InputReadIOp,
          typename OutputWriteIOp, typename Reducers, ParArch PA>
    requires(PA == ParArch::GPU_NVIDIA || PA == ParArch::GPU_AMD)
inline bool executeMorphology(const DPPDetails& details,
                              const InputReadIOp& input,
                              const OutputWriteIOp& output,
                              const Reducers& reducers,
                              Stream_<PA>& stream) {
    if (!DPPDetails::valid(details)) return false;
    static_assert(MorphologyDPP<PA, DPPDetails>::
                      template validReducerCount<Reducers>,
                  "MorphologyDPP requires 1 through 4 reducers");
    const dim3 block(DPPDetails::TILE_WIDTH, DPPDetails::TILE_HEIGHT, 1);
    const dim3 grid((details.width + DPPDetails::TILE_WIDTH - 1) /
                        DPPDetails::TILE_WIDTH,
                    (details.height + DPPDetails::TILE_HEIGHT - 1) /
                        DPPDetails::TILE_HEIGHT,
                    static_cast<unsigned int>(
                        morphology_detail::planesOf(input)));
#if defined(__HIPCC__)
    morphologyDPPKernel<PA><<<grid, block, 0, stream.getHIPStream()>>>(
        details, input, output, reducers);
    gpuErrchk(hipGetLastError());
#else
    morphologyDPPKernel<PA><<<grid, block, 0, stream.getCUDAStream()>>>(
        details, input, output, reducers);
    gpuErrchk(cudaGetLastError());
#endif
    return true;
}

template <typename DPPDetails, typename InputReadIOp,
          typename OutputWriteIOp, ParArch PA>
    requires(PA == ParArch::GPU_NVIDIA || PA == ParArch::GPU_AMD)
inline bool executeErode(const DPPDetails& details,
                         const InputReadIOp& input,
                         const OutputWriteIOp& output,
                         Stream_<PA>& stream) {
    using T = typename DPPDetails::ValueType;
    return executeMorphology(
        details, input, output,
        make_tuple(Min<T, T, T, UnaryType>::build()), stream);
}

template <typename DPPDetails, typename InputReadIOp,
          typename OutputWriteIOp, ParArch PA>
    requires(PA == ParArch::GPU_NVIDIA || PA == ParArch::GPU_AMD)
inline bool executeDilate(const DPPDetails& details,
                          const InputReadIOp& input,
                          const OutputWriteIOp& output,
                          Stream_<PA>& stream) {
    using T = typename DPPDetails::ValueType;
    return executeMorphology(
        details, input, output,
        make_tuple(Max<T, T, T, UnaryType>::build()), stream);
}

template <typename DPPDetails, typename InputReadIOp,
          typename OutputWriteIOp, ParArch PA>
    requires(PA == ParArch::GPU_NVIDIA || PA == ParArch::GPU_AMD)
inline bool executeOpen(const DPPDetails& details,
                        const InputReadIOp& input,
                        const OutputWriteIOp& output,
                        Stream_<PA>& stream) {
    using T = typename DPPDetails::ValueType;
    return executeMorphology(
        details, input, output,
        make_tuple(Min<T, T, T, UnaryType>::build(),
                   Max<T, T, T, UnaryType>::build()), stream);
}

template <typename DPPDetails, typename InputReadIOp,
          typename OutputWriteIOp, ParArch PA>
    requires(PA == ParArch::GPU_NVIDIA || PA == ParArch::GPU_AMD)
inline bool executeClose(const DPPDetails& details,
                         const InputReadIOp& input,
                         const OutputWriteIOp& output,
                         Stream_<PA>& stream) {
    using T = typename DPPDetails::ValueType;
    return executeMorphology(
        details, input, output,
        make_tuple(Max<T, T, T, UnaryType>::build(),
                   Min<T, T, T, UnaryType>::build()), stream);
}

// Box blur: PASSES passes (1 to 4) of the mean of the maskW x maskH window at
// (anchorX, anchorY), each one replicating the border of the image it reads,
// as the other morphology operations. T (details' ValueType) must be a
// floating point type or a vector of them. The output receives the mean of
// the last pass: the passes sum their windows (an Add reducer each) and a
// Mul by 1 / (maskW * maskH)^PASSES is fused in front of the output.
template <int PASSES = 1, typename DPPDetails, typename InputReadIOp,
          typename OutputWriteIOp, ParArch PA>
    requires(PA == ParArch::GPU_NVIDIA || PA == ParArch::GPU_AMD)
inline bool executeBoxBlur(const DPPDetails& details,
                           const InputReadIOp& input,
                           const OutputWriteIOp& output,
                           Stream_<PA>& stream) {
    using T = typename DPPDetails::ValueType;
    if (!DPPDetails::valid(details)) return false;
    return executeMorphology(
        details, input,
        morphology_detail::boxBlurWrite<T, PASSES>(details, output),
        morphology_detail::boxBlurReducers<T, PASSES>(), stream);
}
#endif // defined(__NVCC__) || defined(__HIPCC__)

#if defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)
// Enqueues the morphology on the stream and returns; it runs
// (MorphologyDPP<ParArch::CPU_MULTITHREAD>::exec) after the previous
// executions of the stream. Invalid details are rejected here, returning
// false without enqueueing anything. The buffers the IOps read and write
// must stay valid until the execution has finished (sync()). The output may
// be the input, as with ParArch::CPU.
template <typename DPPDetails, typename InputReadIOp,
          typename OutputWriteIOp, typename Reducers>
inline bool executeMorphology(const DPPDetails& details,
                              const InputReadIOp& input,
                              const OutputWriteIOp& output,
                              const Reducers& reducers,
                              Stream_<ParArch::CPU_MULTITHREAD>& stream) {
    using DPP = MorphologyDPP<ParArch::CPU_MULTITHREAD, DPPDetails>;
    if (!DPPDetails::valid(details)) return false;
    static_assert(DPP::template validReducerCount<Reducers>,
                  "MorphologyDPP requires 1 through 4 reducers");
    // reducers = reducers: see MorphologyDPP<ParArch::CPU_MULTITHREAD>
    stream.enqueue([details, input, output, reducers = reducers](
                       const cpu_multithread::Completion& done) {
        DPP::exec(done, details, input, output, reducers);
    });
    return true;
}

template <typename DPPDetails, typename InputReadIOp,
          typename OutputWriteIOp>
inline bool executeErode(const DPPDetails& details,
                         const InputReadIOp& input,
                         const OutputWriteIOp& output,
                         Stream_<ParArch::CPU_MULTITHREAD>& stream) {
    using T = typename DPPDetails::ValueType;
    return executeMorphology(
        details, input, output,
        make_tuple(Min<T, T, T, UnaryType>::build()), stream);
}

template <typename DPPDetails, typename InputReadIOp,
          typename OutputWriteIOp>
inline bool executeDilate(const DPPDetails& details,
                          const InputReadIOp& input,
                          const OutputWriteIOp& output,
                          Stream_<ParArch::CPU_MULTITHREAD>& stream) {
    using T = typename DPPDetails::ValueType;
    return executeMorphology(
        details, input, output,
        make_tuple(Max<T, T, T, UnaryType>::build()), stream);
}

template <typename DPPDetails, typename InputReadIOp,
          typename OutputWriteIOp>
inline bool executeOpen(const DPPDetails& details,
                        const InputReadIOp& input,
                        const OutputWriteIOp& output,
                        Stream_<ParArch::CPU_MULTITHREAD>& stream) {
    using T = typename DPPDetails::ValueType;
    return executeMorphology(
        details, input, output,
        make_tuple(Min<T, T, T, UnaryType>::build(),
                   Max<T, T, T, UnaryType>::build()), stream);
}

template <typename DPPDetails, typename InputReadIOp,
          typename OutputWriteIOp>
inline bool executeClose(const DPPDetails& details,
                         const InputReadIOp& input,
                         const OutputWriteIOp& output,
                         Stream_<ParArch::CPU_MULTITHREAD>& stream) {
    using T = typename DPPDetails::ValueType;
    return executeMorphology(
        details, input, output,
        make_tuple(Max<T, T, T, UnaryType>::build(),
                   Min<T, T, T, UnaryType>::build()), stream);
}

// Box blur, as the GPU executeBoxBlur
template <int PASSES = 1, typename DPPDetails, typename InputReadIOp,
          typename OutputWriteIOp>
inline bool executeBoxBlur(const DPPDetails& details,
                           const InputReadIOp& input,
                           const OutputWriteIOp& output,
                           Stream_<ParArch::CPU_MULTITHREAD>& stream) {
    using T = typename DPPDetails::ValueType;
    if (!DPPDetails::valid(details)) return false;
    return executeMorphology(
        details, input,
        morphology_detail::boxBlurWrite<T, PASSES>(details, output),
        morphology_detail::boxBlurReducers<T, PASSES>(), stream);
}
#endif // defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)

} // namespace fk

#endif // FK_MORPHOLOGY
