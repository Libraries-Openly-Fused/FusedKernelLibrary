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

#ifndef FK_CPU_THREAD_POOL_H
#define FK_CPU_THREAD_POOL_H

// The thread pool of the multithreaded CPU backend (ParArch::CPU_MULTITHREAD), and what its DPPs use to run their work
// on it. Available when FK_CPU_MULTITHREAD_ENABLED is defined (CMake option ENABLE_CPU_MULTITHREAD, which needs the
// thread-pool submodule).
//
// The responsibilities are split as with a GPU: a CPU_MULTITHREAD stream only orders executions (each one starts when
// the previous one has finished), and does not know about the pool. When an execution starts, its DPP splits the work
// in chunks and enqueues them in the pool with enqueueChunks(); the DPP does not know about streams. The stream gives
// the DPP a Completion, which the DPP calls once all its work has finished, and the stream then starts the next
// execution.

#include <fused_kernel/core/execution_model/parallel_architectures.h>

#if defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)

#include <ThreadPool.h>

#include <cstddef>
#include <exception>
#include <memory>
#include <utility>

namespace fk {

    // The pool shared by every CPU_MULTITHREAD execution, with one worker per hardware thread
    inline thread_pool::ThreadPool& cpuThreadPool() {
        static thread_pool::ThreadPool pool;
        // If a worker can not be created, init() throws and the initialization is tried again in the next call
        static const bool started = (pool.init(), true);
        (void)started;
        return pool;
    }

    namespace cpu_multithread {

        // Whoever started an execution, and is told when it finishes (a stream)
        class CompletionTarget {
        public:
            // error: the first exception the work of the execution threw, or null
            virtual void executionFinished(std::exception_ptr error) noexcept = 0;
        protected:
            ~CompletionTarget() = default;
        };

        // Given to a DPP when its execution starts. The DPP calls it exactly once, when all its work has finished,
        // with the first exception that work threw, if any.
        class Completion {
        public:
            explicit Completion(CompletionTarget& target) : m_target(&target) {}
            void operator()(const std::exception_ptr error = nullptr) const noexcept {
                m_target->executionFinished(error);
            }

        private:
            CompletionTarget* m_target;
        };

        namespace detail {
            template <typename ChunkFunction, typename OnComplete>
            class ChunkTask final : public thread_pool::ThreadPool::BulkTask {
            public:
                ChunkTask(ChunkFunction&& function, OnComplete&& onComplete)
                    : m_function(std::move(function)), m_onComplete(std::move(onComplete)) {}

                void run(const std::size_t chunk) override {
                    m_function(chunk);
                }

                void complete() override {
                    // The task is destroyed first: onComplete may end the execution, after which nothing the task
                    // holds may be used any more
                    OnComplete onComplete = std::move(m_onComplete);
                    const std::exception_ptr error = exception();
                    delete this;
                    onComplete(error);
                }

            private:
                ChunkFunction m_function;
                OnComplete m_onComplete;
            };
        } // namespace detail

        // Runs function(0) ... function(numChunks - 1) concurrently on the workers of cpuThreadPool(), then
        // onComplete(error) on the worker that finished last, where error is the first exception a chunk threw, or
        // null (the other chunks still run). Returns immediately. function and onComplete are stored by value, so
        // they must capture by value everything they use. onComplete must not throw; it can enqueue more chunks (for
        // work in several phases, each one after the previous) or call the Completion of the execution.
        template <typename ChunkFunction, typename OnComplete>
        inline void enqueueChunks(const std::size_t numChunks, ChunkFunction function, OnComplete onComplete) {
            using Task = detail::ChunkTask<ChunkFunction, OnComplete>;
            auto task = std::make_unique<Task>(std::move(function), std::move(onComplete));
            cpuThreadPool().enqueue(*task, numChunks);
            task.release(); // deletes itself in complete()
        }

    } // namespace cpu_multithread
} // namespace fk

#endif // defined(FK_CPU_MULTITHREAD_ENABLED) && !defined(NVRTC_COMPILER)

#endif // FK_CPU_THREAD_POOL_H
