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

#if defined(__NVCC__) || defined(__HIPCC__)
#include <fused_kernel/core/utils/utils.h>
#include <memory>
#include <type_traits>
#endif
#if defined(FK_SYCL_ENABLED)
#include <fused_kernel/core/utils/utils.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#endif
#include <fused_kernel/core/execution_model/cpu_thread_pool.h>
#if defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)
#include <any>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>
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

#if defined(__HIPCC__)
    template <>
    class Stream_<ParArch::GPU_AMD> final : public BaseStream {
        // The hipStream_t, shared by the copies of this Stream_. One created here is destroyed with the last copy:
        // hipStreamDestroy releases the stream when its pending work has finished (errors can not be reported from a
        // destructor). One given by the user is never destroyed.
        std::shared_ptr<std::remove_pointer_t<hipStream_t>> m_stream;

    public:
        Stream_() : BaseStream() {
            hipStream_t stream;
            gpuErrchk(hipStreamCreate(&stream));
            m_stream = std::shared_ptr<std::remove_pointer_t<hipStream_t>>(stream, [](const hipStream_t owned) {
                (void)hipStreamDestroy(owned);
            });
        }
        Stream_(const Stream_<ParArch::GPU_AMD>& other) = default;
        explicit Stream_(const hipStream_t& stream)
            : BaseStream(), m_stream(stream, [](const hipStream_t) {}) {}

        hipStream_t operator()() const {
            return m_stream.get();
        }

        Stream_<ParArch::GPU_AMD>& operator=(const Stream_<ParArch::GPU_AMD>& other) = default;

        Stream_(Stream_<ParArch::GPU_AMD>&&) = delete;
        Stream_<ParArch::GPU_AMD>& operator=(Stream_<ParArch::GPU_AMD>&&) = delete;

        ~Stream_() = default;

        operator hipStream_t() const {
            return m_stream.get();
        }
        inline hipStream_t getHIPStream() const {
            return m_stream.get();
        }
        inline void sync() final {
            gpuErrchk(hipStreamSynchronize(m_stream.get()));
        }
        constexpr inline enum ParArch getParArch() const {
            return ParArch::GPU_AMD;
        };
        static constexpr inline enum ParArch parArch() {
            return ParArch::GPU_AMD;
        }
    };
#endif

#if defined(FK_SYCL_ENABLED)
    namespace gpu_intel {
        // FKL's primary context of each Intel GPU: what the CUDA runtime has for each device. In CUDA, cudaMalloc and
        // cudaStreamCreate use the primary context of the current device implicitly. SYCL has neither: its memory
        // (USM) belongs to a sycl::context, and only the queues of that context can use it. So FKL keeps one context
        // per device: every Ptr of the device allocates its memory in it, and every Stream_<ParArch::GPU_INTEL> of the
        // device has its queue in it, so all the streams of a device can use the memory of all its Ptr, as in CUDA.
        //
        // The devices are the Intel GPUs the SYCL runtime offers, and nothing else: not the CPU (the OpenCL CPU
        // device), not the GPUs of other vendors, as ParArch::GPU_NVIDIA only runs on NVIDIA GPUs. The runtime offers
        // a GPU once per SYCL backend that drives it (Level Zero and OpenCL), so the devices are those of one platform:
        // device 0 is the first Intel GPU the runtime offers through Level Zero (or, if none, the first one it
        // offers), and the next ones are the other Intel GPUs of its platform, in the order of the runtime. The runtime
        // only offers the devices that the environment variable ONEAPI_DEVICE_SELECTOR allows, if it is set.
        //
        // It also knows the queues of the streams that are alive, so that Ptr can release memory as cudaFree does:
        // after the work enqueued in the streams of its context has finished (waitForStreams).
        class PrimaryContexts {
          public:
            // The Intel GPUs are found, and their contexts created, the first time it is used (it throws
            // std::runtime_error if the runtime offers no Intel GPU). A Ptr or a Stream_ uses it when it is created,
            // so it is destroyed after every Ptr and Stream_, also the ones with static storage duration.
            static PrimaryContexts& get() {
                static PrimaryContexts instance;
                return instance;
            }

            PrimaryContexts(const PrimaryContexts&) = delete;
            PrimaryContexts& operator=(const PrimaryContexts&) = delete;

            const sycl::device& device(const int deviceID) const {
                return at(deviceID).device;
            }
            const sycl::context& context(const int deviceID) const {
                return at(deviceID).context;
            }
            bool isPrimary(const sycl::context& context) const {
                for (const DeviceAndContext& primary : devices) {
                    if (primary.context == context) {
                        return true;
                    }
                }
                return false;
            }

            // A Stream_<ParArch::GPU_INTEL> registers its queue while it is alive
            void registerQueue(const sycl::queue& queue) {
                const std::lock_guard<std::mutex> lock(queuesMutex);
                queues.push_back(queue);
            }
            void unregisterQueue(const sycl::queue& queue) noexcept {
                const std::lock_guard<std::mutex> lock(queuesMutex);
                for (auto registered = queues.begin(); registered != queues.end(); ++registered) {
                    if (*registered == queue) {
                        queues.erase(registered);
                        return;
                    }
                }
            }

            // Waits for everything enqueued in the streams that are alive and use context. Their errors are not
            // reported here: they are left for the sync() of each stream.
            void waitForStreams(const sycl::context& context) noexcept {
                try {
                    std::vector<sycl::queue> waited;
                    {
                        const std::lock_guard<std::mutex> lock(queuesMutex);
                        waited = queues;
                    }
                    for (sycl::queue& queue : waited) {
                        if (queue.get_context() == context) {
                            queue.wait();
                        }
                    }
                } catch (...) {
                    // Called when memory is released, also from destructors: nothing to report an error to
                }
            }

          private:
            // The PCI vendor ID of Intel, the vendor_id of its devices
            static constexpr std::uint32_t INTEL_VENDOR_ID = 0x8086;

            struct DeviceAndContext {
                sycl::device device;
                sycl::context context;
            };

            PrimaryContexts() : devices(findIntelGPUs()) {}

            static bool isIntelGPU(const sycl::device& device) {
                return device.is_gpu() && device.get_info<sycl::info::device::vendor_id>() == INTEL_VENDOR_ID &&
                       device.has(sycl::aspect::usm_device_allocations);
            }

            static std::vector<DeviceAndContext> findIntelGPUs() {
                const std::vector<sycl::device> gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
                const sycl::device* first = nullptr;
                for (const sycl::device& device : gpus) {
                    if (isIntelGPU(device) &&
                        (first == nullptr || (first->get_backend() != sycl::backend::ext_oneapi_level_zero &&
                                              device.get_backend() == sycl::backend::ext_oneapi_level_zero))) {
                        first = &device;
                    }
                }
                if (first == nullptr) {
                    throw std::runtime_error("ParArch::GPU_INTEL: the SYCL runtime offers no Intel GPU (check the "
                                             "Intel GPU driver, and ONEAPI_DEVICE_SELECTOR if it is set)");
                }
                std::vector<DeviceAndContext> found;
                for (const sycl::device& device : first->get_platform().get_devices(sycl::info::device_type::gpu)) {
                    if (isIntelGPU(device)) {
                        found.push_back({ device, sycl::context(device) });
                    }
                }
                return found;
            }

            const DeviceAndContext& at(const int deviceID) const {
                if (deviceID < 0 || static_cast<std::size_t>(deviceID) >= devices.size()) {
                    throw std::out_of_range("ParArch::GPU_INTEL: there is no device " + std::to_string(deviceID) +
                                            " (the backend has " + std::to_string(devices.size()) + " device(s))");
                }
                return devices[static_cast<std::size_t>(deviceID)];
            }

            const std::vector<DeviceAndContext> devices;
            std::mutex queuesMutex;
            std::vector<sycl::queue> queues;
        };

        // Device deviceID of the backend, the deviceID of Ptr and Stream_<ParArch::GPU_INTEL>: an Intel GPU (see
        // PrimaryContexts). Throws std::out_of_range if there is no such device.
        inline const sycl::device& device(const int deviceID = 0) {
            return PrimaryContexts::get().device(deviceID);
        }

        // FKL's primary context of device deviceID: where Ptr allocates, and where a queue has to be made to use that
        // memory (sycl::queue q(gpu_intel::context(id), gpu_intel::device(id), sycl::property::queue::in_order())).
        inline const sycl::context& context(const int deviceID = 0) {
            return PrimaryContexts::get().context(deviceID);
        }
    } // namespace gpu_intel

    // A SYCL in-order queue, which behaves like a CUDA stream: the kernels and the copies enqueued in it run in the
    // order they were enqueued, and sync() waits for all of them and throws the first error they had. The copies of a
    // Stream_ share the queue, and the last copy to be destroyed waits for the work enqueued in it (errors nobody
    // synchronized with are lost).
    //
    // Its queue is in the primary context of its device (gpu_intel::PrimaryContexts), where the memory of every Ptr of
    // the device is, so the streams of a device can use all of it, as CUDA streams can. A default constructed Stream_
    // has its own queue on device 0 (or on the device given). A Stream_ can also use a queue of the user, which has to
    // be in order (sycl::property::queue::in_order) and in a primary context: made with gpu_intel::context(deviceID).
    template <>
    class Stream_<ParArch::GPU_INTEL> final : public BaseStream {
        // Alignment of the blocks of kernel parameters, in host and in device memory
        static constexpr std::size_t PARAMS_ALIGNMENT = 64;

        // Asynchronous errors of the queues of the streams: sync() (wait_and_throw) throws the first one
        static void rethrowAsyncErrors(const sycl::exception_list errors) {
            for (const std::exception_ptr& error : errors) {
                std::rethrow_exception(error);
            }
        }

        // An in-order queue on device deviceID, in its primary context
        static sycl::queue makeQueue(const int deviceID) {
            return sycl::queue(gpu_intel::context(deviceID), gpu_intel::device(deviceID), &rethrowAsyncErrors,
                               sycl::property::queue::in_order());
        }

        // The parameters of a kernel that do not fit in its arguments (FK_SYCL_MAX_KERNEL_ARGUMENTS_BYTES, see
        // launchDPP_Kernel) are in a block of device memory, which the queue copies from a block of host memory. Both
        // are released when the kernel that uses them has finished.
        struct KernelParams {
            sycl::event lastUse; // The copy to the device, and then the kernel that uses the parameters
            void* device{ nullptr };
            void* host{ nullptr };
        };

        struct State {
            // While the stream is alive, releasing the memory of a Ptr waits for the work enqueued in its queue
            // (PrimaryContexts::waitForStreams)
            explicit State(const sycl::queue& queue_) : queue(queue_) {
                gpu_intel::PrimaryContexts::get().registerQueue(queue);
            }
            State(const State&) = delete;
            State& operator=(const State&) = delete;

            // The last copy of the stream waits for the work enqueued in it, which uses memory that could be released
            // once the stream is not there to wait for
            ~State() {
                try {
                    queue.wait();
                } catch (...) {
                    // Errors nobody synchronized with are lost with the stream
                }
                for (const KernelParams& params : pending) {
                    release(params);
                }
                gpu_intel::PrimaryContexts::get().unregisterQueue(queue);
            }

            void release(const KernelParams& params) noexcept {
                try {
                    if (params.device != nullptr) {
                        sycl::free(params.device, queue.get_context());
                    }
                } catch (...) {
                    // Nothing to report an error to: this is also called from destructors
                }
                ::operator delete(params.host, std::align_val_t(PARAMS_ALIGNMENT));
            }

            // Called with the mutex held. The kernels of an in-order queue finish in order.
            void releaseFinished() {
                while (!pending.empty() &&
                       pending.front().lastUse.get_info<sycl::info::event::command_execution_status>() ==
                           sycl::info::event_command_status::complete) {
                    release(pending.front());
                    pending.pop_front();
                }
            }

            sycl::queue queue;
            std::mutex mutex;
            std::deque<KernelParams> pending;
        };

        std::shared_ptr<State> m_state;

    public:
        Stream_() : BaseStream(), m_state(std::make_shared<State>(makeQueue(0))) {}
        explicit Stream_(const int deviceID) : BaseStream(), m_state(std::make_shared<State>(makeQueue(deviceID))) {}
        Stream_(const Stream_<ParArch::GPU_INTEL>& other) = default;
        // The queue must be in order and in a primary context: a queue the user made with
        // sycl::queue q(gpu_intel::context(id), gpu_intel::device(id), sycl::property::queue::in_order())
        explicit Stream_(const sycl::queue& queue) : BaseStream() {
            if (!queue.is_in_order()) {
                throw std::invalid_argument("Stream_<ParArch::GPU_INTEL> needs an in-order queue "
                                            "(sycl::property::queue::in_order): what is enqueued in a stream runs in "
                                            "the order it was enqueued");
            }
            if (!gpu_intel::PrimaryContexts::get().isPrimary(queue.get_context())) {
                throw std::invalid_argument("Stream_<ParArch::GPU_INTEL> needs a queue in FKL's context of its device "
                                            "(gpu_intel::context(deviceID)): the memory of Ptr is in that context, and "
                                            "a queue of another context can not use it");
            }
            m_state = std::make_shared<State>(queue);
        }

        Stream_<ParArch::GPU_INTEL>& operator=(const Stream_<ParArch::GPU_INTEL>& other) = default;

        Stream_(Stream_<ParArch::GPU_INTEL>&&) = delete;
        Stream_<ParArch::GPU_INTEL>& operator=(Stream_<ParArch::GPU_INTEL>&&) = delete;

        ~Stream_() = default;

        sycl::queue& operator()() const {
            return m_state->queue;
        }
        operator sycl::queue&() const {
            return m_state->queue;
        }
        inline sycl::queue& getSYCLQueue() const {
            return m_state->queue;
        }
        inline void sync() final {
            State& state = *m_state;
            state.queue.wait_and_throw();
            const std::lock_guard<std::mutex> lock(state.mutex);
            state.releaseFinished();
        }
        constexpr inline enum ParArch getParArch() const {
            return ParArch::GPU_INTEL;
        };
        static constexpr inline enum ParArch parArch() {
            return ParArch::GPU_INTEL;
        }

        // Copies params to device memory, with a copy enqueued in the stream, and returns where it will be: for the
        // kernels that get their parameters through a pointer. One kernel, enqueued in this stream after this call,
        // can use that memory: its event must be given to kernelParamsUsedBy, and the memory is released when it has
        // finished (it is released after the copy if no kernel is given).
        template <typename Params>
        inline const Params* kernelParams(const Params& params) {
            static_assert(std::is_trivially_copyable_v<Params>,
                          "The parameters of a kernel must be trivially copyable: they are copied to the device");
            static_assert(alignof(Params) <= PARAMS_ALIGNMENT, "Unsupported alignment of kernel parameters");
            State& state = *m_state;
            const std::lock_guard<std::mutex> lock(state.mutex);
            state.releaseFinished();
            KernelParams block;
            block.host = ::operator new(sizeof(Params), std::align_val_t(PARAMS_ALIGNMENT));
            std::memcpy(block.host, &params, sizeof(Params));
            // The block is in the list before its copy is enqueued: once enqueued, it must not be released early
            bool listed = false;
            try {
                block.device = sycl::aligned_alloc_device(PARAMS_ALIGNMENT, sizeof(Params), state.queue);
                if (block.device == nullptr) {
                    throw std::runtime_error("ParArch::GPU_INTEL: could not allocate device memory for the "
                                             "parameters of a kernel");
                }
                state.pending.push_back(block);
                listed = true;
                state.pending.back().lastUse = state.queue.memcpy(block.device, block.host, sizeof(Params));
            } catch (...) {
                if (listed) {
                    state.pending.pop_back();
                }
                state.release(block);
                throw;
            }
            return static_cast<const Params*>(block.device);
        }
        inline void kernelParamsUsedBy(const void* const deviceParams, const sycl::event& kernel) {
            State& state = *m_state;
            const std::lock_guard<std::mutex> lock(state.mutex);
            for (auto params = state.pending.rbegin(); params != state.pending.rend(); ++params) {
                if (params->device == deviceParams) {
                    params->lastUse = kernel;
                    return;
                }
            }
        }
    };
#endif // defined(FK_SYCL_ENABLED)

    template <>
    class Stream_<ParArch::CPU> final : public BaseStream {
    public:
        Stream_() : BaseStream() {}
        ~Stream_() = default;
        inline void sync() final {}
        constexpr inline enum ParArch getParArch() const {
            return ParArch::CPU;
        };
        static constexpr inline enum ParArch parArch() {
            return ParArch::CPU;
        }
    };

#if defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)
    namespace cpu_multithread {
        namespace detail {
            // Owning copies of the buffers (Ptr, Tensor...) the execution being enqueued on this thread uses, which
            // it keeps alive until it has finished
            inline std::vector<std::any>& pendingKeepAlive() {
                static thread_local std::vector<std::any> objects;
                return objects;
            }

            // One execution enqueued on a stream. start() is called when the previous execution of the stream has
            // finished: it starts the work of a DPP, which calls the Completion when that work has finished.
            class Execution {
            public:
                Execution() : m_keepAlive(std::move(pendingKeepAlive())) {
                    pendingKeepAlive().clear();
                }
                virtual ~Execution() = default;
                virtual void start(const Completion& done) = 0;

                // Position of the execution in its stream (1 for the first one), set when it is enqueued
                std::uint64_t number{0};

            private:
                std::vector<std::any> m_keepAlive;
            };

            template <typename StartFunction>
            class FunctionExecution final : public Execution {
            public:
                explicit FunctionExecution(StartFunction&& start) : m_start(std::move(start)) {}
                void start(const Completion& done) final {
                    m_start(done);
                }

            private:
                StartFunction m_start;
            };

            // The executions of one stream: the front one is running, the rest wait for it
            class ExecutionQueue final : public CompletionTarget {
            public:
                // Like a CUDA launch queue, a stream holds a limited number of pending executions: past it, enqueue
                // waits until one of them finishes
                static constexpr std::uint64_t MAX_PENDING = 1024;

                ExecutionQueue() = default;
                ExecutionQueue(const ExecutionQueue&) = delete;
                ExecutionQueue& operator=(const ExecutionQueue&) = delete;

                ~ExecutionQueue() {
                    try {
                        sync();
                    } catch (...) {
                        // Errors of executions nobody synchronized with are lost with the stream
                    }
                }

                void enqueue(std::unique_ptr<Execution> execution) {
                    Execution* start = nullptr;
                    {
                        std::unique_lock<std::mutex> lock(m_mutex);
                        // Wait for room and queue under the same lock, so that threads enqueueing at the same time
                        // can not exceed MAX_PENDING
                        waitForRoom(lock);
                        // Counted only once it is queued: push_back may throw, and then nothing changes
                        execution->number = m_enqueued + 1;
                        m_executions.push_back(std::move(execution));
                        ++m_enqueued;
                        if (m_executions.size() == 1) {
                            start = m_executions.front().get();
                        }
                    }
                    if (start != nullptr) {
                        if (const std::exception_ptr error = startOrDrop(*start, true)) {
                            std::rethrow_exception(error);
                        }
                    }
                }

                // Called through the Completion of the running execution: destroys it and starts the next one
                void executionFinished(const std::exception_ptr error) noexcept final {
                    std::unique_ptr<Execution> done;
                    Execution* next = nullptr;
                    {
                        std::lock_guard<std::mutex> lock(m_mutex);
                        done = std::move(m_executions.front());
                        m_executions.pop_front();
                        ++m_finished;
                        if (error) {
                            m_errors.emplace_back(done->number, error);
                        }
                        m_finishedAtomic.store(m_finished, std::memory_order_release);
                        // Notify while holding the lock: the queue may be destroyed as soon as a waiter sees that its
                        // executions have finished
                        m_condition.notify_all();
                        if (!m_executions.empty()) {
                            next = m_executions.front().get();
                        }
                    }
                    if (next != nullptr) {
                        // On failure, startOrDrop records the error for sync(); the queue may be destroyed as soon
                        // as it returns, so nothing here uses it afterwards
                        (void)startOrDrop(*next, false);
                    }
                }

                // Waits for the executions enqueued before the call. Rethrows the first exception, not reported yet,
                // thrown by the work of one of those executions (or by starting it); errors of executions enqueued
                // after the call are left for a later sync().
                void sync() {
                    std::uint64_t target;
                    {
                        std::lock_guard<std::mutex> lock(m_mutex);
                        target = m_enqueued;
                    }
                    if (m_finishedAtomic.load(std::memory_order_acquire) < target) {
                        thread_pool::ThreadPool& pool = cpuThreadPool();
                        if (pool.isWorkerThread()) {
                            // Blocking a worker could leave no worker to run the executions: help with them instead
                            while (m_finishedAtomic.load(std::memory_order_acquire) < target) {
                                if (!pool.runOneChunk()) {
                                    std::this_thread::yield();
                                }
                            }
                        } else {
                            // Spin for a short time before sleeping, like the workers of the pool
                            const auto spinEnd = std::chrono::steady_clock::now() + std::chrono::microseconds(100);
                            while (m_finishedAtomic.load(std::memory_order_acquire) < target &&
                                   std::chrono::steady_clock::now() < spinEnd) {
                                std::this_thread::yield();
                            }
                        }
                    }
                    // Always take the lock, so that executionFinished() is not using this queue any more when
                    // sync() returns
                    std::unique_lock<std::mutex> lock(m_mutex);
                    m_condition.wait(lock, [this, target] { return m_finished >= target; });
                    // m_errors is ordered by execution number: report the first error up to target, and forget the
                    // others up to target
                    if (!m_errors.empty() && m_errors.front().first <= target) {
                        const std::exception_ptr error = m_errors.front().second;
                        while (!m_errors.empty() && m_errors.front().first <= target) {
                            m_errors.pop_front();
                        }
                        std::rethrow_exception(error);
                    }
                }

            private:
                // Called and returns with the lock held: waits until fewer than MAX_PENDING executions are pending.
                // On a worker of the pool it runs queued chunks meanwhile (without the lock) instead of blocking.
                void waitForRoom(std::unique_lock<std::mutex>& lock) {
                    const auto full = [this] { return m_enqueued - m_finished >= MAX_PENDING; };
                    if (!full()) {
                        return;
                    }
                    thread_pool::ThreadPool& pool = cpuThreadPool();
                    if (pool.isWorkerThread()) {
                        while (full()) {
                            lock.unlock();
                            if (!pool.runOneChunk()) {
                                std::this_thread::yield();
                            }
                            lock.lock();
                        }
                        return;
                    }
                    m_condition.wait(lock, [&] { return !full(); });
                }

                // Starts the execution. If that fails (no memory, or the pool has been shut down at program exit),
                // the executions of this stream are dropped, so that nobody waits for them forever, and the error is
                // returned. It is also recorded for sync(), under the same lock that publishes the drop, as the error
                // of the execution that failed to start; when that execution was just enqueued (fromEnqueue), the
                // enqueuing thread gets the exception, and it is recorded only if executions enqueued after it were
                // dropped too, as the error of the first of them. Nothing uses the queue after that lock: a thread
                // in sync() may destroy it right away.
                std::exception_ptr startOrDrop(Execution& execution, const bool fromEnqueue) noexcept {
                    try {
                        execution.start(Completion(*this));
                        return nullptr;
                    } catch (...) {
                        const std::exception_ptr error = std::current_exception();
                        std::deque<std::unique_ptr<Execution>> dropped;
                        {
                            std::lock_guard<std::mutex> lock(m_mutex);
                            dropped.swap(m_executions);
                            m_finished = m_enqueued;
                            if (!fromEnqueue) {
                                m_errors.emplace_back(execution.number, error);
                            } else if (dropped.size() > 1) {
                                m_errors.emplace_back(dropped[1]->number, error);
                            }
                            m_finishedAtomic.store(m_finished, std::memory_order_release);
                            m_condition.notify_all();
                        }
                        return error;
                    }
                }

                std::mutex m_mutex;
                std::condition_variable m_condition;
                std::deque<std::unique_ptr<Execution>> m_executions;
                std::uint64_t m_enqueued{0};
                std::uint64_t m_finished{0};
                std::atomic<std::uint64_t> m_finishedAtomic{0};
                // Errors not reported yet, with the number of their execution, in increasing order
                std::deque<std::pair<std::uint64_t, std::exception_ptr>> m_errors;
            };
        } // namespace detail

        // Makes the next execution enqueued from this thread keep owning copies of these objects (Ptr, Tensor,
        // arrays of them...) until it has finished, so that the buffers it uses stay alive
        template <typename... Objects>
        inline void keepAlive(const Objects&... objects) {
            auto& pending = detail::pendingKeepAlive();
            (pending.emplace_back(objects), ...);
        }
    } // namespace cpu_multithread

    // Behaves like a CUDA stream: enqueue() returns immediately, the executions of a stream run one after the other in
    // the order they were enqueued, and sync() waits for the executions enqueued before it (and rethrows the first
    // exception their work threw). Copies share the queue, and the last copy to be destroyed waits for all its
    // executions. The stream does not run any work itself: it starts each execution, whose DPP enqueues its work in
    // cpuThreadPool().
    template <>
    class Stream_<ParArch::CPU_MULTITHREAD> final : public BaseStream {
    public:
        Stream_() : BaseStream(), m_queue(std::make_shared<cpu_multithread::detail::ExecutionQueue>()) {}
        Stream_(const Stream_<ParArch::CPU_MULTITHREAD>& other) = default;
        Stream_<ParArch::CPU_MULTITHREAD>& operator=(const Stream_<ParArch::CPU_MULTITHREAD>& other) = default;
        ~Stream_() = default;

        inline void sync() final {
            m_queue->sync();
        }

        // Enqueues an execution: start(const cpu_multithread::Completion& done) is called, on some thread, when the
        // previous executions of the stream have finished. It must start the work (typically DPP::exec(done, ...))
        // and return; the work calls done when it has finished. start is stored by value.
        template <typename StartFunction>
        inline void enqueue(StartFunction&& start) {
            using Execution = cpu_multithread::detail::FunctionExecution<std::decay_t<StartFunction>>;
            m_queue->enqueue(std::make_unique<Execution>(std::forward<StartFunction>(start)));
        }

        constexpr inline enum ParArch getParArch() const {
            return ParArch::CPU_MULTITHREAD;
        }
        static constexpr inline enum ParArch parArch() {
            return ParArch::CPU_MULTITHREAD;
        }

    private:
        std::shared_ptr<cpu_multithread::detail::ExecutionQueue> m_queue;
    };
#endif // defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)

    using Stream = Stream_<defaultParArch>;
} // namespace fk

#endif
