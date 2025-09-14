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

#if defined(__NVCC__) || CLANG_HOST_DEVICE
#include <fused_kernel/core/utils/utils.h>
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

#if defined(__NVCC__) || CLANG_HOST_DEVICE
    template <>
    class Stream_<ParArch::GPU_NVIDIA> final : public BaseStream {
        cudaStream_t m_stream;
        bool m_isMine{ false };

        inline void initFromOther(const Stream_<ParArch::GPU_NVIDIA>& other) {
            m_stream = other.m_stream;
            m_isMine = other.m_isMine;
        }

    public:
        Stream_() : BaseStream() {
            gpuErrchk(cudaStreamCreate(&m_stream));
            m_isMine = true;
        }
        Stream_(const Stream_<ParArch::GPU_NVIDIA>& other) : BaseStream(other) {
            initFromOther(other);
        }
        explicit Stream_<ParArch::GPU_NVIDIA>(const cudaStream_t& stream) : m_stream(stream), BaseStream() {}

        cudaStream_t operator()() const {
            return m_stream;
        }

        Stream_<ParArch::GPU_NVIDIA>& operator=(const Stream_<ParArch::GPU_NVIDIA>& other) {
            if (this != &other) {
                BaseStream::operator=(other);
                initFromOther(other);
            }
            return *this;
        }

        Stream_(Stream_<ParArch::GPU_NVIDIA>&&) = delete;
        Stream_<ParArch::GPU_NVIDIA>& operator=(Stream_<ParArch::GPU_NVIDIA>&&) = delete;

        ~Stream_() {
            if ( this->getRefCount() == 0 && m_stream != 0 && m_isMine) {
                sync();
                gpuErrchk(cudaStreamDestroy(m_stream));
            }
        }

        operator cudaStream_t() const {
            return m_stream;
        }
        inline cudaStream_t getCUDAStream() const {
            return m_stream;
        }
        inline void sync() final {
            gpuErrchk(cudaStreamSynchronize(m_stream));
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
