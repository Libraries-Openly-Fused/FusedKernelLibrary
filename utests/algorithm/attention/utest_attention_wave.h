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

/* attention_wave.h: attnShflXor exchanges values inside each 32-lane wave
 * (never across waves) and attnRequireWave32 lets a wave32 kernel run. Two
 * waves per block: each lane holds its thread index, a butterfly sum must give
 * the sum of its own wave, and a single exchange must return lane ^ mask. */

#include <tests/main.h>

#include <fused_kernel/algorithms/attention/attention_wave.h>
#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/execution_model/stream.h>

#include <iostream>

#if defined(__NVCC__) || defined(__HIPCC__)

namespace fk {

constexpr int WAVE_TEST_THREADS = 2 * ATTN_WAVE_SIZE;
constexpr int WAVE_TEST_MASK = 5;

__global__ void attnWaveTestKernel(const RawPtr<ND::_1D, float> sums, const RawPtr<ND::_1D, int> exchanged) {
    attnRequireWave32();
    const int tid = static_cast<int>(threadIdx.x);
    float sum = static_cast<float>(tid);
    for (int offset = ATTN_WAVE_SIZE / 2; offset > 0; offset >>= 1) {
        sum += attnShflXor(sum, offset);
    }
    sums.data[tid] = sum;
    exchanged.data[tid] = attnShflXor(tid, WAVE_TEST_MASK);
}

int launch_impl() {
    Ptr1D<float> sums(WAVE_TEST_THREADS);
    Ptr1D<int> exchanged(WAVE_TEST_THREADS);
    Stream stream;
#if defined(__HIPCC__)
    attnWaveTestKernel<<<1, WAVE_TEST_THREADS, 0, stream.getHIPStream()>>>(sums.ptr(), exchanged.ptr());
    gpuErrchk(hipGetLastError());
#else
    attnWaveTestKernel<<<1, WAVE_TEST_THREADS, 0, stream.getCUDAStream()>>>(sums.ptr(), exchanged.ptr());
    gpuErrchk(cudaGetLastError());
#endif
    sums.download(stream);
    exchanged.download(stream);
    stream.sync();

    int failures = 0;
    for (int tid = 0; tid < WAVE_TEST_THREADS; ++tid) {
        const int first = tid - tid % ATTN_WAVE_SIZE;
        const int expectedIntSum = ATTN_WAVE_SIZE * first + ATTN_WAVE_SIZE * (ATTN_WAVE_SIZE - 1) / 2;
        const float expectedSum = static_cast<float>(expectedIntSum);
        const int expectedExchange = tid ^ WAVE_TEST_MASK;
        const float gotSum = sums.ptrPinned().data[tid];
        const int gotExchange = exchanged.ptrPinned().data[tid];
        if (gotSum != expectedSum || gotExchange != expectedExchange) {
            std::cout << "FAIL attnShflXor lane " << tid << ": sum " << gotSum << " (expected " << expectedSum
                      << "), exchange " << gotExchange << " (expected " << expectedExchange << ")" << std::endl;
            ++failures;
        }
    }
    if (failures == 0) {
        std::cout << "Running test attnShflXor butterfly sum and exchange, 2 waves: Success!!" << std::endl;
        return 0;
    }
    return -1;
}

} // namespace fk

int launch() {
    return fk::launch_impl();
}

#else

int launch() {
    std::cout << "attention_wave.h has no CPU code to test" << std::endl;
    return 0;
}

#endif // defined(__NVCC__) || defined(__HIPCC__)
