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

#ifndef FK_CPU_MULTITHREAD_H
#define FK_CPU_MULTITHREAD_H

// Multithreaded CPU backend, ParArch::CPU_MULTITHREAD. Needs the thread pool of the thread-pool submodule
// (https://github.com/Libraries-Openly-Fused/thread-pool) in the include path, which the FKL CMake target adds when
// ENABLE_CPU_MULTITHREAD is ON.
//
// - cpuThreadPool() is the pool shared by every CPU_MULTITHREAD stream, with one worker per hardware thread.
// - Stream_<ParArch::CPU_MULTITHREAD> behaves like a CUDA stream: executeOperations enqueues the execution of a DPP
//   and returns, the executions of one stream run one after the other in the order they were enqueued, and sync()
//   waits for the executions enqueued before it. Executions of different streams can run concurrently in the shared
//   pool. Copies of a stream share its queue, and the last copy to be destroyed waits for all its executions. Like a
//   CUDA launch queue, a stream holds a limited number of pending executions (MAX_PENDING): past it, executeOperations
//   waits until one of them finishes.
// - TransformDPP<ParArch::CPU_MULTITHREAD> runs the per thread code of TransformDPP<ParArch::CPU>. Its work is split
//   in chunks of one row of one plane of the output, which the workers of the pool consume.
//
// Memory: as with the host buffers of cudaMemcpyAsync, the buffers an execution reads or writes must stay valid until
// it has finished (sync()). The executeOperations overloads that take Ptr or Tensor objects keep a reference to them in
// the enqueued execution, so those buffers stay alive by themselves; buffers given to build() as RawPtr (ptr.ptr()) or
// as an Image do not.
//
// sync() can be called from a worker of the pool (for instance from a function given to cpuThreadPool().submit()):
// the worker then runs queued chunks while it waits, instead of blocking.

#include <fused_kernel/core/execution_model/executors.h>

#include <ThreadPool.h>

#include <any>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace fk {

    inline thread_pool::ThreadPool& cpuThreadPool() {
        static thread_pool::ThreadPool pool;
        // If a worker can not be created, init() throws and the initialization is tried again in the next call
        static const bool started = (pool.init(), true);
        (void)started;
        return pool;
    }

    namespace cpu_multithread_detail {
        class LaunchQueue;

        // Owning copies of the Ptr and Tensor arguments of executeOperations, which the execution being built on this
        // thread keeps alive until it has finished
        inline std::vector<std::any>& pendingKeepAlive() {
            static thread_local std::vector<std::any> objects;
            return objects;
        }

        // One enqueued execution of a DPP: a bulk task of the pool that, when all its chunks are done, lets its
        // stream start the next execution
        class Launch : public thread_pool::ThreadPool::BulkTask {
        public:
            explicit Launch(LaunchQueue& queue) : m_queue(queue), m_keepAlive(std::move(pendingKeepAlive())) {
                pendingKeepAlive().clear();
            }
            virtual std::size_t chunks() const = 0;
            void complete() final;

        private:
            LaunchQueue& m_queue;
            std::vector<std::any> m_keepAlive;
        };

        // The executions of one stream. The front one is running in the pool, the rest wait for it.
        class LaunchQueue {
        public:
            static constexpr std::uint64_t MAX_PENDING = 1024;

            LaunchQueue() = default;
            LaunchQueue(const LaunchQueue&) = delete;
            LaunchQueue& operator=(const LaunchQueue&) = delete;

            ~LaunchQueue() {
                sync();
            }

            void enqueue(std::unique_ptr<Launch> launch) {
                // Starts the workers, if they are not running yet, before anything is queued: this may throw
                thread_pool::ThreadPool& pool = cpuThreadPool();
                waitForRoom(pool);
                Launch* start = nullptr;
                {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    m_launches.push_back(std::move(launch));
                    ++m_enqueued;
                    if (m_launches.size() == 1) {
                        start = m_launches.front().get();
                    }
                }
                if (start != nullptr) {
                    if (const std::exception_ptr error = startOrDrop(pool, *start)) {
                        std::rethrow_exception(error);
                    }
                }
            }

            // Called by the running launch when its last chunk has finished. Destroys it and starts the next one.
            void finished() {
                std::unique_ptr<Launch> done;
                Launch* next = nullptr;
                {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    done = std::move(m_launches.front());
                    m_launches.pop_front();
                    ++m_finished;
                    m_finishedAtomic.store(m_finished, std::memory_order_release);
                    // Notify while holding the lock: the queue may be destroyed as soon as a waiter sees that its
                    // executions have finished
                    m_condition.notify_all();
                    if (!m_launches.empty()) {
                        next = m_launches.front().get();
                    }
                }
                if (next != nullptr) {
                    // It can only fail once the pool has been shut down, and then startOrDrop drops the executions
                    (void)startOrDrop(cpuThreadPool(), *next);
                }
            }

            // Waits for the executions enqueued before the call
            void sync() {
                std::uint64_t target;
                {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    target = m_enqueued;
                    if (m_finished >= target) {
                        return;
                    }
                }
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
                // Always take the lock, so that finished() is not using this queue any more when sync() returns
                std::unique_lock<std::mutex> lock(m_mutex);
                m_condition.wait(lock, [this, target] { return m_finished >= target; });
            }

        private:
            // Waits until fewer than MAX_PENDING executions are pending, running queued chunks meanwhile on a worker.
            // Several threads may enqueue at the same time, so the limit can be exceeded by a few executions.
            void waitForRoom(thread_pool::ThreadPool& pool) {
                const auto full = [this] { return m_enqueued - m_finishedAtomic.load(std::memory_order_acquire) >=
                                                  MAX_PENDING; };
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

            // Enqueues the launch in the pool. If that fails (the pool has been shut down, at program exit), the
            // executions of this stream can never run: they are dropped, so that nobody waits for them forever.
            std::exception_ptr startOrDrop(thread_pool::ThreadPool& pool, Launch& launch) noexcept {
                try {
                    pool.enqueue(launch, launch.chunks());
                    return nullptr;
                } catch (...) {
                    std::deque<std::unique_ptr<Launch>> dropped;
                    {
                        std::lock_guard<std::mutex> lock(m_mutex);
                        dropped.swap(m_launches);
                        m_finished = m_enqueued;
                        m_finishedAtomic.store(m_finished, std::memory_order_release);
                        m_condition.notify_all();
                    }
                    return std::current_exception();
                }
            }

            std::mutex m_mutex;
            std::condition_variable m_condition;
            std::deque<std::unique_ptr<Launch>> m_launches;
            std::uint64_t m_enqueued{0};
            std::uint64_t m_finished{0};
            std::atomic<std::uint64_t> m_finishedAtomic{0};
        };

        inline void Launch::complete() {
            // finished() destroys this launch, so nothing can be done after it
            m_queue.finished();
        }

        template <typename DPP, typename Details, typename... IOps>
        class TransformLaunch final : public Launch {
        public:
            TransformLaunch(LaunchQueue& queue, const Details& details, const IOps&... iOps)
                : Launch(queue), m_details(details), m_iOps(make_tuple(iOps...)),
                  m_chunks(DPP::numChunks(details, iOps...)) {}

            std::size_t chunks() const final { return m_chunks; }

            void run(const std::size_t chunk) final {
                apply([this, chunk](const auto&... iOps) { DPP::exec_chunk(chunk, m_details, iOps...); }, m_iOps);
            }

        private:
            Details m_details;
            Tuple<IOps...> m_iOps;
            std::size_t m_chunks;
        };
    } // namespace cpu_multithread_detail

    template <>
    class Stream_<ParArch::CPU_MULTITHREAD> final : public BaseStream {
    public:
        Stream_() : BaseStream(), m_queue(std::make_shared<cpu_multithread_detail::LaunchQueue>()) {}
        Stream_(const Stream_<ParArch::CPU_MULTITHREAD>& other) = default;
        Stream_<ParArch::CPU_MULTITHREAD>& operator=(const Stream_<ParArch::CPU_MULTITHREAD>& other) = default;
        // The queue is destroyed with the last copy of the stream, and waits for its executions
        ~Stream_() = default;

        inline void sync() final { m_queue->sync(); }

        inline void enqueue(std::unique_ptr<cpu_multithread_detail::Launch> launch) {
            m_queue->enqueue(std::move(launch));
        }

        inline cpu_multithread_detail::LaunchQueue& launchQueue() { return *m_queue; }

        constexpr inline enum ParArch getParArch() const {
            return ParArch::CPU_MULTITHREAD;
        }
        static constexpr inline enum ParArch parArch() {
            return ParArch::CPU_MULTITHREAD;
        }

    private:
        std::shared_ptr<cpu_multithread_detail::LaunchQueue> m_queue;
    };

    template <enum TF TFEN, typename DPPDetails, bool THREAD_DIVISIBLE>
    struct TransformDPP<ParArch::CPU_MULTITHREAD, TFEN, DPPDetails, THREAD_DIVISIBLE,
                        std::enable_if_t<!std::is_same_v<DPPDetails, void>, void>> {
    private:
        using CPUDPP = TransformDPP<ParArch::CPU, TFEN, DPPDetails, THREAD_DIVISIBLE>;
        using Details = DPPDetails;
    public:
        static constexpr ParArch PAR_ARCH = ParArch::CPU_MULTITHREAD;
        template <typename FirstIOp>
        FK_HOST_FUSE ActiveThreads getActiveThreads(const Details& details, const FirstIOp& iOp) {
            return CPUDPP::getActiveThreads(details, iOp);
        }

        // One chunk per row of every plane of the output
        template <typename... IOps>
        FK_HOST_FUSE std::size_t numChunks(const Details& details, const IOps&... iOps) {
            const ActiveThreads activeThreads = getActiveThreads(details, get_arg<0>(iOps...));
            return static_cast<std::size_t>(activeThreads.y) * activeThreads.z;
        }

        template <typename... IOps>
        FK_HOST_FUSE void exec_chunk(const std::size_t chunk, const Details& details, const IOps&... iOps) {
            const ActiveThreads activeThreads = getActiveThreads(details, get_arg<0>(iOps...));
            const int y = static_cast<int>(chunk % activeThreads.y);
            const int z = static_cast<int>(chunk / activeThreads.y);
            for (int x = 0; x < static_cast<int>(activeThreads.x); ++x) {
                CPUDPP::exec_thread(Point{x, y, z}, details, iOps...);
            }
        }
    };

    template <enum TF TFEN>
    struct Executor<TransformDPP<ParArch::CPU_MULTITHREAD, TFEN, void>> {
    private:
        using Child = Executor<TransformDPP<ParArch::CPU_MULTITHREAD, TFEN>>;
        using Parent = BaseExecutor<Child>;
        using MTStream = Stream_<ParArch::CPU_MULTITHREAD>;
        friend struct BaseExecutor<Child>;

        // The execution being built keeps these objects (Ptr, Tensor, arrays of them) alive until it has finished
        template <typename... Objects>
        static inline void keepAlive(const Objects&... objects) {
            auto& pending = cpu_multithread_detail::pendingKeepAlive();
            (pending.emplace_back(objects), ...);
        }

        template <typename... IOps>
        FK_HOST_FUSE void executeOperations_helper(MTStream& stream, const IOps&... iOps) {
            constexpr ParArch PA = ParArch::CPU_MULTITHREAD;
            const auto tDetails = TransformDPP<PA, TFEN>::build_details(iOps...);
            using TDPPDetails = std::decay_t<decltype(tDetails)>;
            const auto enqueue = [&](auto threadDivisible) {
                using DPP = TransformDPP<PA, TFEN, TDPPDetails, decltype(threadDivisible)::value>;
                using Launch = cpu_multithread_detail::TransformLaunch<DPP, TDPPDetails, std::decay_t<IOps>...>;
                stream.enqueue(std::make_unique<Launch>(stream.launchQueue(), tDetails, iOps...));
            };
            if constexpr (TDPPDetails::TFI::ENABLED) {
                if (!tDetails.threadDivisible) {
                    enqueue(std::false_type{});
                } else {
                    enqueue(std::true_type{});
                }
            } else {
                enqueue(std::true_type{});
            }
        }
    public:
        FK_STATIC_STRUCT(Executor, Child)
        FK_HOST_FUSE ParArch parArch() {
            return ParArch::CPU_MULTITHREAD;
        }

        // The entry points of DECLARE_EXECUTOR_PARENT_IMPL, keeping their Ptr and Tensor arguments alive
        template <typename... IOps>
        static inline void executeOperations(MTStream& stream, const IOps&... iOps) {
            Parent::executeOperations(stream, iOps...);
        }
        template <ND D, typename I, typename... IOps>
        static inline void executeOperations(const Ptr<D, I>& input, MTStream& stream, const IOps&... iOps) {
            keepAlive(input);
            Parent::executeOperations(input, stream, iOps...);
        }
        template <ND D, typename I, typename O, typename... IOps>
        static inline void executeOperations(const Ptr<D, I>& input, const Ptr<D, O>& output, MTStream& stream,
                                             const IOps&... iOps) {
            keepAlive(input, output);
            Parent::executeOperations(input, output, stream, iOps...);
        }
        template <typename I, size_t BATCH, typename... IOps>
        static inline void executeOperations(const std::array<Ptr2D<I>, BATCH>& input, const int& activeBatch,
                                             const I& defaultValue, MTStream& stream, const IOps&... iOps) {
            keepAlive(input);
            Parent::executeOperations(input, activeBatch, defaultValue, stream, iOps...);
        }
        template <typename I, size_t BATCH, typename... IOps>
        static inline void executeOperations(const std::array<Ptr2D<I>, BATCH>& input, MTStream& stream,
                                             const IOps&... iOps) {
            keepAlive(input);
            Parent::executeOperations(input, stream, iOps...);
        }
        template <typename I, typename O, size_t Batch, typename... IOps>
        static inline void executeOperations(const std::array<Ptr2D<I>, Batch>& input, const int& activeBatch,
                                             const I& defaultValue, const Tensor<O>& output, MTStream& stream,
                                             const IOps&... iOps) {
            keepAlive(input, output);
            Parent::executeOperations(input, activeBatch, defaultValue, output, stream, iOps...);
        }
        template <typename I, typename O, size_t Batch, typename... IOps>
        static inline void executeOperations(const std::array<Ptr2D<I>, Batch>& input, const Tensor<O>& output,
                                             MTStream& stream, const IOps&... iOps) {
            keepAlive(input, output);
            Parent::executeOperations(input, output, stream, iOps...);
        }
    };

} // namespace fk

#endif // FK_CPU_MULTITHREAD_H
