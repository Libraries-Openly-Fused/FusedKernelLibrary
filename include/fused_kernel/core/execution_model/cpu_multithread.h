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

// Multithreaded CPU backend, ParArch::CPU_MULTITHREAD. Its pieces live next to the other backends, and are available
// when FK_CPU_MULTITHREAD_ENABLED is defined (CMake option ENABLE_CPU_MULTITHREAD, ON when the thread-pool submodule is
// checked out); this header includes them all:
//
// - cpu_thread_pool.h: cpuThreadPool(), the thread pool shared by all CPU_MULTITHREAD work, and
//   cpu_multithread::enqueueChunks(), with which a DPP runs its work on it as chunks (for instance one per output row).
// - stream.h: Stream_<ParArch::CPU_MULTITHREAD>, which behaves like a CUDA stream: executeOperations enqueues an
//   execution and returns, the executions of a stream run one after the other in order, and sync() waits for the ones
//   enqueued before it. The stream only orders executions; it starts each one, and the DPP of the execution enqueues
//   the work in the pool and calls the cpu_multithread::Completion it was given when the work has finished.
// - data_parallel_patterns.h, executors.h and the DPPs of algorithms/: the CPU_MULTITHREAD DPPs run the per row code
//   of their ParArch::CPU version, so both give the same results (float results up to the last bits, where the
//   compiler contracts a * b + c into a fused multiply-add differently). The attention DPPs are GPU only.
//
// Memory: as with the host buffers of cudaMemcpyAsync, the buffers an execution reads or writes must stay valid until
// it has finished (sync()). The executeOperations overloads that take Ptr or Tensor objects keep a reference to them in
// the execution, so those buffers stay alive by themselves; buffers given to build() as RawPtr or as an Image do not.
//
// sync() can be called from a worker of the pool (for instance from a function given to cpuThreadPool().submit()):
// the worker then runs queued chunks while it waits, instead of blocking. It rethrows the first exception the work of
// an execution threw since the previous sync().

#if !defined(FK_CPU_MULTITHREAD_ENABLED)
#error "ParArch::CPU_MULTITHREAD needs FK_CPU_MULTITHREAD_ENABLED (CMake option ENABLE_CPU_MULTITHREAD, with the thread-pool submodule: git submodule update --init)"
#endif

#include <fused_kernel/core/execution_model/cpu_thread_pool.h>
#include <fused_kernel/core/execution_model/stream.h>
#include <fused_kernel/core/execution_model/data_parallel_patterns.h>
#include <fused_kernel/core/execution_model/executors.h>

#endif // FK_CPU_MULTITHREAD_H
