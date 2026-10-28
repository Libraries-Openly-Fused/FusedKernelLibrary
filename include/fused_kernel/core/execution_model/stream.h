/* Copyright 2025 Grup Mediapro S.L.U (Oscar Amoros Huguet)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_STREAM
#define FK_STREAM

#include <fused_kernel/core/execution_model/parallel_architectures.h>
#include <fused_kernel/core/data/ref_class.h>

#if defined(__NVCC__)
#include <fused_kernel/core/utils/utils.h>
#include <memory>
#include <type_traits>
#elif defined(__HIP__)
#include <hip/hip_runtime.h>
#endif

namespace fk {

    class BaseStream : public Ref {
    public:
        BaseStream() : Ref() {};
        BaseStream(const BaseStream& other) : Ref(other) {}
        virtual ~BaseStream() = default;

        BaseStream(BaseStream&&) = delete;
        BaseStream& operator=(BaseStream&&) = delete;

        BaseStream& operator=(const BaseStream& other) {
            if (this != &other) {
                Ref::operator=(other);
            }
            return *this;
        }

        virtual void sync() = 0;
    };

    template <enum ParArch PA>
    class Stream_;

#if defined(__NVCC__)
    template <>
    class Stream_<ParArch::GPU_NVIDIA> final : public BaseStream {
        // The cudaStream_t, shared by the copies of this Stream_. One created here is destroyed with the last copy:
        // cudaStreamDestroy returns at once and releases the stream when its pending work has finished (no
        // synchronization, which would invalidate CUDA graph captures running in other threads; errors can not be
        // reported from a destructor). One given by the user is never destroyed.
        std::shared_ptr<std::remove_pointer_t<cudaStream_t>> m_stream;

    public:
        Stream_() : BaseStream() {
            cudaStream_t stream;
            gpuErrchk(cudaStreamCreate(&stream));
            m_stream = std::shared_ptr<std::remove_pointer_t<cudaStream_t>>(stream, [](const cudaStream_t owned) {
                (void)cudaStreamDestroy(owned);
            });
        }
        Stream_(const Stream_<ParArch::GPU_NVIDIA>& other) = default;
        explicit Stream_(const cudaStream_t& stream)
            : BaseStream(), m_stream(stream, [](const cudaStream_t) {}) {}

        cudaStream_t operator()() const {
            return m_stream.get();
        }

        Stream_<ParArch::GPU_NVIDIA>& operator=(const Stream_<ParArch::GPU_NVIDIA>& other) = default;

        Stream_(Stream_<ParArch::GPU_NVIDIA>&&) = delete;
        Stream_<ParArch::GPU_NVIDIA>& operator=(Stream_<ParArch::GPU_NVIDIA>&&) = delete;

        ~Stream_() = default;

        operator cudaStream_t() const {
            return m_stream.get();
        }
        inline cudaStream_t getCUDAStream() const {
            return m_stream.get();
        }
        inline void sync() final {
            gpuErrchk(cudaStreamSynchronize(m_stream.get()));
        }
        constexpr inline enum ParArch getParArch() const {
            return ParArch::GPU_NVIDIA;
        };
        static constexpr inline enum ParArch parArch() {
            return ParArch::GPU_NVIDIA;
        }
    };
#endif

    template <>
    class Stream_<ParArch::CPU> final : public BaseStream {
    public:
        Stream_<ParArch::CPU>() : BaseStream() {}
        ~Stream_<ParArch::CPU>() = default;
        inline void sync() final {}
        constexpr inline enum ParArch getParArch() const {
            return ParArch::CPU;
        };
        static constexpr inline enum ParArch parArch() {
            return ParArch::CPU;
        }
    };

    using Stream = Stream_<defaultParArch>;
} // namespace fk

#endif
