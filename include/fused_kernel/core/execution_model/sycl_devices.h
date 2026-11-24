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

#ifndef FK_SYCL_DEVICES_H
#define FK_SYCL_DEVICES_H

// The devices of the SYCL backend (ParArch::GPU_INTEL), the SYCL context of each one, and their memory. Available when
// FK_SYCL_ENABLED is defined: a translation unit compiled as SYCL (see compiler_macros.h).
//
// CUDA and HIP have a current device, and an implicit context per device: cudaMalloc and cudaStreamCreate use them.
// SYCL has neither. Its memory (USM) belongs to a sycl::context, and only the queues of that context can use it. So
// FKL has one context per device, where it allocates the memory of every Ptr of that device and creates the queue of
// every Stream_<ParArch::GPU_INTEL> of that device. A queue made by the user can use that memory if it is made with
// gpu_intel::context(deviceID).

#include <fused_kernel/core/utils/utils.h>

#if defined(FK_SYCL_ENABLED)

#include <cstddef>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

// The total size of the arguments of a kernel is limited: 2048 bytes on Intel GPUs, and 1024 bytes is the least a
// device can have (sycl::info::device::max_parameter_size). A kernel that exceeds the limit is only rejected when the
// device compiles it, at run time. The parameters of a DPP (its details and IOps) are the arguments of its kernel
// while they add up to this number of bytes at most; if they are larger (for instance, the IOps of a large batch)
// they are copied to device memory, and the kernel gets a pointer to them (see gpu_intel::launch).
#ifndef FK_SYCL_MAX_KERNEL_ARGUMENTS_BYTES
#define FK_SYCL_MAX_KERNEL_ARGUMENTS_BYTES 1024
#endif

namespace fk {
    namespace gpu_intel {

        // Alignment, in bytes, of the device and pinned memory FKL allocates, and of the rows of a 2D device buffer
        // allocated without a pitch (as cudaMallocPitch, the pitch is the size of a row rounded up to it)
        inline constexpr std::size_t MEMORY_ALIGNMENT = 64;

        namespace detail {
            // How good a device is for this backend. GPUs first, the ones of Intel before the others. The same GPU
            // is one device per SYCL backend: Level Zero is preferred to OpenCL.
            inline int deviceScore(const sycl::device& device) {
                int score = 0;
                if (device.is_gpu()) {
                    score += 1000;
                    if (device.get_info<sycl::info::device::vendor_id>() == 0x8086) {
                        score += 500;
                    }
                } else if (device.is_accelerator()) {
                    score += 100;
                }
                if (device.get_backend() == sycl::backend::ext_oneapi_level_zero) {
                    score += 50;
                }
                return score;
            }

            inline bool usable(const sycl::device& device) {
                return device.has(sycl::aspect::usm_device_allocations);
            }

            struct DeviceAndContext {
                sycl::device device;
                sycl::context context;
            };

            // The best device the SYCL runtime offers, and the devices of its platform that are as good as it: the
            // GPUs of the machine, each one once. The runtime only offers the devices that the environment variable
            // ONEAPI_DEVICE_SELECTOR allows, if it is set.
            inline std::vector<DeviceAndContext> findDevices() {
                const std::vector<sycl::device> all = sycl::device::get_devices();
                const sycl::device* best = nullptr;
                int bestScore = -1;
                for (const sycl::device& device : all) {
                    if (usable(device) && deviceScore(device) > bestScore) {
                        best = &device;
                        bestScore = deviceScore(device);
                    }
                }
                if (best == nullptr) {
                    throw std::runtime_error("ParArch::GPU_INTEL: the SYCL runtime has no device with USM device "
                                             "memory (check the drivers, and ONEAPI_DEVICE_SELECTOR if it is set)");
                }
                std::vector<DeviceAndContext> devices;
                for (const sycl::device& device : best->get_platform().get_devices()) {
                    if (usable(device) && deviceScore(device) == bestScore) {
                        devices.push_back({ device, sycl::context(device) });
                    }
                }
                return devices;
            }

            // The state of the backend: its devices, and the queues of the streams that are alive (see free)
            struct Backend {
                Backend() : devices(findDevices()) {}
                Backend(const Backend&) = delete;
                Backend& operator=(const Backend&) = delete;

                const std::vector<DeviceAndContext> devices;
                std::mutex queuesMutex;
                std::vector<sycl::queue> queues;
            };

            // The devices are found the first time the backend is used. A Ptr or a Stream_ uses it when it is
            // created, so it is destroyed after every Ptr and Stream_, also the ones with static storage duration.
            inline Backend& backend() {
                static Backend instance;
                return instance;
            }

            inline const DeviceAndContext& deviceAndContext(const int deviceID) {
                const std::vector<DeviceAndContext>& devices = backend().devices;
                if (deviceID < 0 || static_cast<std::size_t>(deviceID) >= devices.size()) {
                    throw std::out_of_range("ParArch::GPU_INTEL: there is no device " + std::to_string(deviceID) +
                                            " (the backend has " + std::to_string(devices.size()) + " device(s))");
                }
                return devices[static_cast<std::size_t>(deviceID)];
            }

            // A Stream_<ParArch::GPU_INTEL> registers its queue while it is alive
            inline void registerQueue(const sycl::queue& queue) {
                Backend& state = backend();
                const std::lock_guard<std::mutex> lock(state.queuesMutex);
                state.queues.push_back(queue);
            }

            inline void unregisterQueue(const sycl::queue& queue) noexcept {
                Backend& state = backend();
                const std::lock_guard<std::mutex> lock(state.queuesMutex);
                for (auto registered = state.queues.begin(); registered != state.queues.end(); ++registered) {
                    if (*registered == queue) {
                        state.queues.erase(registered);
                        return;
                    }
                }
            }

            // Waits for everything enqueued in the streams that are alive and use context. Their errors are not
            // reported here: they are left for the sync() of each stream.
            inline void waitForStreams(const sycl::context& context) noexcept {
                try {
                    std::vector<sycl::queue> queues;
                    {
                        Backend& state = backend();
                        const std::lock_guard<std::mutex> lock(state.queuesMutex);
                        queues = state.queues;
                    }
                    for (sycl::queue& queue : queues) {
                        if (queue.get_context() == context) {
                            queue.wait();
                        }
                    }
                } catch (...) {
                    // Called when memory is released, also from destructors: nothing to report an error to
                }
            }
        } // namespace detail

        // Device deviceID of the backend, the deviceID of Ptr and Stream_<ParArch::GPU_INTEL>. Device 0 is the best
        // device the SYCL runtime offers (an Intel GPU through Level Zero, if there is one), and the following ones
        // are the other devices of its platform that are as good. Throws std::out_of_range if there is no such device.
        inline const sycl::device& device(const int deviceID = 0) {
            return detail::deviceAndContext(deviceID).device;
        }

        // The context where FKL allocates the memory and creates the queues of device deviceID
        inline const sycl::context& context(const int deviceID = 0) {
            return detail::deviceAndContext(deviceID).context;
        }

        // Asynchronous errors of the queues FKL creates: Stream_::sync() (wait_and_throw) throws the first one
        inline void rethrowAsyncErrors(const sycl::exception_list errors) {
            for (const std::exception_ptr& error : errors) {
                std::rethrow_exception(error);
            }
        }

        // An in-order queue on device deviceID, in FKL's context: what a default constructed
        // Stream_<ParArch::GPU_INTEL> uses
        inline sycl::queue makeQueue(const int deviceID = 0) {
            return sycl::queue(context(deviceID), device(deviceID), &rethrowAsyncErrors,
                               sycl::property::queue::in_order());
        }

        // Device memory (MemType::Device) of device deviceID: USM device memory in FKL's context. An allocation of 0
        // bytes is a null pointer.
        inline void* allocDevice(const std::size_t bytes, const int deviceID = 0) {
            void* const data = sycl::aligned_alloc_device(MEMORY_ALIGNMENT, bytes, device(deviceID), context(deviceID));
            if (data == nullptr && bytes != 0) {
                throw std::runtime_error("ParArch::GPU_INTEL: could not allocate " + std::to_string(bytes) +
                                         " bytes of device memory");
            }
            return data;
        }

        // Pinned memory (MemType::HostPinned) for device deviceID: USM host memory in FKL's context, which the host
        // uses as any other memory and the device can access too
        inline void* allocHostPinned(const std::size_t bytes, const int deviceID = 0) {
            void* const data = sycl::aligned_alloc_host(MEMORY_ALIGNMENT, bytes, context(deviceID));
            if (data == nullptr && bytes != 0) {
                throw std::runtime_error("ParArch::GPU_INTEL: could not allocate " + std::to_string(bytes) +
                                         " bytes of pinned (USM host) memory");
            }
            return data;
        }

        // Releases memory of allocDevice or allocHostPinned. As cudaFree does, it first waits for the work enqueued in
        // the streams, which may still use the memory: SYCL does not allow releasing memory a kernel or a copy that
        // has not finished uses. It waits for the streams that are alive (a stream waits for its own work when its
        // last copy is destroyed), so memory can be released at any time.
        inline void free(void* const data, const int deviceID = 0) {
            if (data != nullptr) {
                const sycl::context& ctx = context(deviceID);
                detail::waitForStreams(ctx);
                sycl::free(data, ctx);
            }
        }

        // Enqueues the copy of height rows of widthBytes bytes, between buffers with different row strides
        inline sycl::event memcpy2D(sycl::queue& queue, void* const dst, const std::size_t dstPitch,
                                    const void* const src, const std::size_t srcPitch, const std::size_t widthBytes,
                                    const std::size_t height) {
#if defined(SYCL_EXT_ONEAPI_MEMCPY2D)
            return queue.ext_oneapi_memcpy2d(dst, dstPitch, src, srcPitch, widthBytes, height);
#else
            sycl::event last;
            for (std::size_t y = 0; y < height; ++y) {
                last = queue.memcpy(static_cast<char*>(dst) + y * dstPitch,
                                    static_cast<const char*>(src) + y * srcPitch, widthBytes);
            }
            return last;
#endif
        }
    } // namespace gpu_intel
} // namespace fk

#endif // defined(FK_SYCL_ENABLED)

#endif // FK_SYCL_DEVICES_H
