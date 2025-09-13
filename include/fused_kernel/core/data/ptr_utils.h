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
    template <enum ParArch PA, enum ND D, typename T>
    inline void setTo(const T& value, Ptr<D, T>& outputPtr, Stream_<PA>& stream) {
        RawPtr<D, T> output = outputPtr.ptr();
#if defined(__NVCC__) || CLANG_HOST
        if constexpr (PA == ParArch::GPU_NVIDIA) {
            if (outputPtr.getMemType() == MemType::Device || outputPtr.getMemType() == MemType::DeviceAndPinned) {
                Executor<TransformDPP<ParArch::GPU_NVIDIA>>::executeOperations(stream, ReadSet<T>::build(value, outputPtr.dims()), PerThreadWrite<D, T>::build(output));
                if (outputPtr.getMemType() == MemType::DeviceAndPinned) {
                    Stream_<ParArch::CPU> cpuStream;
                    Executor<TransformDPP<ParArch::CPU>>::executeOperations(cpuStream, ReadSet<T>::build(value, outputPtr.dims()), PerThreadWrite<D, T>::build(outputPtr.ptrPinned()));
                }
            }
            else {
                Executor<TransformDPP<ParArch::GPU_NVIDIA>>::executeOperations(stream, ReadSet<T>::build(value, outputPtr.dims()), PerThreadWrite<D, T>::build(output));
            }
        }
        else {
            Executor<TransformDPP<PA>>::executeOperations(stream, ReadSet<T>::build(value, outputPtr.dims()), PerThreadWrite<D, T>::build(output));
        }
#else
        Executor<TransformDPP<PA>>::executeOperations(stream, ReadSet<T>::build(value, outputPtr.dims()), PerThreadWrite<D, T>::build(outputPtr));
#endif
    }
} // namespace fk

#endif // FK_PTR_UTILS_H