/* Copyright 2023-2025 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_PTR_UTILS_H
#define FK_PTR_UTILS_H

#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/execution_model/executors.h>
#include <fused_kernel/core/execution_model/data_parallel_patterns.h>

namespace fk {
#if defined(__NVCC__)
    namespace set_to_internal {
        // True while the stream is being captured into a graph
        inline bool isCapturing(const Stream_<ParArch::GPU_NVIDIA>& stream) {
            cudaStreamCaptureStatus status{ cudaStreamCaptureStatusNone };
            gpuErrchk(cudaStreamIsCapturing(stream.getCUDAStream(), &status));
            return status != cudaStreamCaptureStatusNone;
        }
    } // namespace set_to_internal
#elif defined(FK_HIP_BACKEND)
    namespace set_to_internal {
        // True while the stream is being captured into a graph
        inline bool isCapturing(const Stream_<ParArch::GPU_AMD>& stream) {
            hipStreamCaptureStatus status{ hipStreamCaptureStatusNone };
            gpuErrchk(hipStreamIsCapturing(stream.getHIPStream(), &status));
            return status != hipStreamCaptureStatusNone;
        }
    } // namespace set_to_internal
#endif

    // With a GPU stream, Device, HostPinned and DeviceAndPinned memory (both copies) are filled by kernels on the
    // stream, in stream order: after the work already enqueued on it, and recorded by a graph capture. Pageable Host
    // memory is filled by the CPU once the work enqueued on the stream has finished, since the GPU can not write it
    // (unless the system gives GPUs access to pageable memory); while the stream is being captured into a graph that
    // can not be done, and setTo throws std::runtime_error, leaving the capture as it was. A CPU stream can not write
    // Device or DeviceAndPinned memory: setTo throws std::runtime_error.
    template <enum ParArch PA, enum ND D, typename T>
    inline void setTo(const T& value, Ptr<D, T>& outputPtr, Stream_<PA>& stream) {
        RawPtr<D, T> output = outputPtr.ptr();
#if defined(__NVCC__) || defined(FK_HIP_BACKEND)
        if constexpr (PA == ParArch::GPU_NVIDIA || PA == ParArch::GPU_AMD) {
            if (outputPtr.getMemType() == MemType::Host) {
                if (set_to_internal::isCapturing(stream)) {
                    throw std::runtime_error("setTo: Host memory can not be filled while the stream is being captured into a graph");
                }
                stream.sync();
                Stream_<ParArch::CPU> cpuStream;
                Executor<TransformDPP<ParArch::CPU>>::executeOperations(
                    cpuStream, ReadSet<T>::build(value, outputPtr.dims()), PerThreadWrite<D, T>::build(output));
            }
            else {
                Executor<TransformDPP<PA>>::executeOperations(stream, ReadSet<T>::build(value, outputPtr.dims()), PerThreadWrite<D, T>::build(output));
                if (outputPtr.getMemType() == MemType::DeviceAndPinned) {
                    Executor<TransformDPP<PA>>::executeOperations(stream, ReadSet<T>::build(value, outputPtr.dims()), PerThreadWrite<D, T>::build(outputPtr.ptrPinned()));
                }
            }
        }
        else {
            if (outputPtr.getMemType() == MemType::Device || outputPtr.getMemType() == MemType::DeviceAndPinned) {
                throw std::runtime_error("setTo: a CPU stream can not write Device or DeviceAndPinned memory");
            }
            Executor<TransformDPP<PA>>::executeOperations(stream, ReadSet<T>::build(value, outputPtr.dims()), PerThreadWrite<D, T>::build(output));
        }
#else
        Executor<TransformDPP<PA>>::executeOperations(stream, ReadSet<T>::build(value, outputPtr.dims()), PerThreadWrite<D, T>::build(outputPtr));
#endif
    }
} // namespace fk

#endif // FK_PTR_UTILS_H