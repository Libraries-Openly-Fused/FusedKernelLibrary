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
//   waits for all of them. Executions of different streams can run concurrently in the shared pool.
// - TransformDPP<ParArch::CPU_MULTITHREAD> runs the per thread code of TransformDPP<ParArch::CPU>. Its work is split
//   in chunks of one row of one plane of the output, which the workers of the pool consume.

#include <fused_kernel/core/execution_model/executors.h>

#include <ThreadPool.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>

namespace fk {

    inline ThreadPool& cpuThreadPool() {
        static ThreadPool pool;
        static const bool started = (pool.init(), true);
        (void)started;
        return pool;
    }

    namespace cpu_multithread_detail {
        class LaunchQueue;

        // One enqueued execution of a DPP: a bulk task of the pool that, when all its chunks are done, lets its
        // stream start the next execution
        class Launch : public ThreadPool::BulkTask {
        public:
            explicit Launch(LaunchQueue& queue) : m_queue(queue) {}
            virtual std::size_t chunks() const = 0;
            void complete() final;

        private:
            LaunchQueue& m_queue;
        };

        // The executions of one stream. The front one is running in the pool, the rest wait for it.
        class LaunchQueue {
        public:
            void enqueue(std::unique_ptr<Launch> launch) {
                Launch* start = nullptr;
                {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    m_launches.push_back(std::move(launch));
                    if (m_launches.size() == 1) {
                        m_idle.store(false, std::memory_order_relaxed);
                        start = m_launches.front().get();
                    }
                }
                if (start != nullptr) {
                    cpuThreadPool().enqueue(*start, start->chunks());
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
                    if (m_launches.empty()) {
                        // Notify while holding the lock: the stream may be destroyed as soon as sync() returns
                        m_idle.store(true, std::memory_order_release);
                        m_condition.notify_all();
                    } else {
                        next = m_launches.front().get();
                    }
                }
                if (next != nullptr) {
                    cpuThreadPool().enqueue(*next, next->chunks());
                }
            }

            void sync() {
                // Spin for a short time before sleeping, like the workers of the pool
                const auto spinEnd = std::chrono::steady_clock::now() + std::chrono::microseconds(100);
                while (!m_idle.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < spinEnd) {
                    std::this_thread::yield();
                }
                // Always take the lock, so that finished() is not using this queue any more when sync() returns
                std::unique_lock<std::mutex> lock(m_mutex);
                m_condition.wait(lock, [this] { return m_launches.empty(); });
            }

        private:
            std::mutex m_mutex;
            std::condition_variable m_condition;
            std::deque<std::unique_ptr<Launch>> m_launches;
            std::atomic<bool> m_idle{true};
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
        Stream_(const Stream_<ParArch::CPU_MULTITHREAD>& other) : BaseStream(other), m_queue(other.m_queue) {}
        Stream_<ParArch::CPU_MULTITHREAD>& operator=(const Stream_<ParArch::CPU_MULTITHREAD>& other) {
            if (this != &other) {
                if (m_queue.use_count() == 1) {
                    m_queue->sync();
                }
                BaseStream::operator=(other);
                m_queue = other.m_queue;
            }
            return *this;
        }
        ~Stream_() {
            if (m_queue.use_count() == 1) {
                m_queue->sync();
            }
        }

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
        template <typename... IOps>
        FK_HOST_FUSE void executeOperations_helper(Stream_<ParArch::CPU_MULTITHREAD>& stream, const IOps&... iOps) {
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
        DECLARE_EXECUTOR_PARENT_IMPL
    };

} // namespace fk

#endif // FK_CPU_MULTITHREAD_H
