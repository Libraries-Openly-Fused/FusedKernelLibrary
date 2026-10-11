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

#if defined(__HIPCC__)
    template <>
    class Stream_<ParArch::GPU_AMD> final : public BaseStream {
        hipStream_t m_stream;
        bool m_isMine{ false };

        inline void initFromOther(const Stream_<ParArch::GPU_AMD>& other) {
            m_stream = other.m_stream;
            m_isMine = other.m_isMine;
        }

    public:
        Stream_() : BaseStream() {
            gpuErrchk(hipStreamCreate(&m_stream));
            m_isMine = true;
        }
        Stream_(const Stream_<ParArch::GPU_AMD>& other) : BaseStream(other) {
            initFromOther(other);
        }
        explicit Stream_<ParArch::GPU_AMD>(const hipStream_t& stream) : m_stream(stream), BaseStream() {}

        hipStream_t operator()() const {
            return m_stream;
        }

        Stream_<ParArch::GPU_AMD>& operator=(const Stream_<ParArch::GPU_AMD>& other) {
            if (this != &other) {
                BaseStream::operator=(other);
                initFromOther(other);
            }
            return *this;
        }

        Stream_(Stream_<ParArch::GPU_AMD>&&) = delete;
        Stream_<ParArch::GPU_AMD>& operator=(Stream_<ParArch::GPU_AMD>&&) = delete;

        ~Stream_() {
            if (this->getRefCount() == 0 && m_stream != 0 && m_isMine) {
                sync();
                gpuErrchk(hipStreamDestroy(m_stream));
            }
        }

        operator hipStream_t() const {
            return m_stream;
        }
        inline hipStream_t getHIPStream() const {
            return m_stream;
        }
        inline void sync() final {
            gpuErrchk(hipStreamSynchronize(m_stream));
        }
        constexpr inline enum ParArch getParArch() const {
            return ParArch::GPU_AMD;
        };
        static constexpr inline enum ParArch parArch() {
            return ParArch::GPU_AMD;
        }
    };
#endif

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
                    waitForRoom();
                    Execution* start = nullptr;
                    {
                        std::lock_guard<std::mutex> lock(m_mutex);
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
                        if (error && !m_error) {
                            m_error = error;
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

                // Waits for the executions enqueued before the call. Rethrows the first exception thrown by the
                // work of an execution since the previous sync().
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
                    if (m_error) {
                        const std::exception_ptr error = m_error;
                        m_error = nullptr;
                        std::rethrow_exception(error);
                    }
                }

            private:
                // Waits until fewer than MAX_PENDING executions are pending, running queued chunks meanwhile on a
                // worker of the pool. Several threads may enqueue at the same time, so the limit can be exceeded by a
                // few executions.
                void waitForRoom() {
                    const auto full = [this] {
                        return m_enqueued - m_finishedAtomic.load(std::memory_order_acquire) >= MAX_PENDING;
                    };
                    {
                        std::lock_guard<std::mutex> lock(m_mutex);
                        if (!full()) {
                            return;
                        }
                    }
                    thread_pool::ThreadPool& pool = cpuThreadPool();
                    if (pool.isWorkerThread()) {
                        while (true) {
                            {
                                std::lock_guard<std::mutex> lock(m_mutex);
                                if (!full()) {
                                    return;
                                }
                            }
                            if (!pool.runOneChunk()) {
                                std::this_thread::yield();
                            }
                        }
                    }
                    std::unique_lock<std::mutex> lock(m_mutex);
                    m_condition.wait(lock, [&] { return !full(); });
                }

                // Starts the execution. If that fails (no memory, or the pool has been shut down at program exit),
                // the executions of this stream are dropped, so that nobody waits for them forever, and the error is
                // returned. It is also recorded for sync(), under the same lock that publishes the drop, unless it is
                // reported to the enqueuing thread only (fromEnqueue) and no other thread's execution was dropped.
                // Nothing uses the queue after that lock: a thread in sync() may destroy it right away.
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
                            if (!m_error && (!fromEnqueue || dropped.size() > 1)) {
                                m_error = error;
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
                std::exception_ptr m_error;
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
