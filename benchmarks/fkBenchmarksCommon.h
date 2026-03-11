/* Copyright 2024 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_BENCHMARKS_COMMON_H
#define FK_BENCHMARKS_COMMON_H

#include <array>
#include <chrono>
#include <unordered_map>
#include <sstream>
#include <fstream>
#include <iostream>

#include <fused_kernel/core/utils/vlimits.h>
#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/algorithms/basic_ops/logical.h>

#include <fused_kernel/core/execution_model/parallel_architectures.h>

constexpr int ITERS = 100;
std::unordered_map<std::string, std::stringstream> benchmarkResultsText;
std::unordered_map<std::string, std::ofstream> currentFile;
// Select the path where to write the benchmark files
const std::string path{ "" };

template <size_t START_VALUE, size_t INCREMENT, std::size_t... Is>
constexpr std::array<size_t, sizeof...(Is)> generate_sequence(std::index_sequence<Is...>) {
    return std::array<size_t, sizeof...(Is)>{(START_VALUE + (INCREMENT * Is))...};
}

template <size_t START_VALUE, size_t INCREMENT, size_t NUM_ELEMS>
constexpr std::array<size_t, NUM_ELEMS> arrayIndexSecuence = generate_sequence<START_VALUE, INCREMENT>(std::make_index_sequence<NUM_ELEMS>{});

template <typename T>
inline bool compareAndCheck(const fk::Ptr2D<T>& firstResult, const fk::Ptr2D<T>& secondResult) {
    const bool sameDims = firstResult.dims().width == secondResult.dims().width && firstResult.dims().height == secondResult.dims().height;
    if (!sameDims) {
        std::cout << "Dimensions do not match: " << firstResult.dims().width << "x" << firstResult.dims().height << " vs " << secondResult.dims().width << "x" << secondResult.dims().height << std::endl;
        return false;
    }
    for (int y = 0; y < static_cast<int>(firstResult.dims().height); ++y) {
        for (int x = 0; x < static_cast<int>(firstResult.dims().width); ++x) {
            if (!fk::Equal<T>::exec(fk::make_tuple(firstResult.at(fk::Point{ x, y, 0 }), secondResult.at(fk::Point{ x, y, 0 })))) {
                std::cout << "Mismatch at (" << x << ", " << y << ") " << std::endl;
                return false;
            }
        }
    }
    return true;
}

struct BenchmarkResultsNumbersOne {
    float fkElapsedTimeMax{ fk::minValue<float> };;
    float fkElapsedTimeMin{ fk::maxValue<float> };
    float fkElapsedTimeAcum{ 0.f };
};

class TimeMarkerInterfaceOne {
    virtual void start() = 0;
    virtual void stop(BenchmarkResultsNumbersOne& resF, const int& idx) = 0;
    virtual std::array<float, ITERS> getElapsedTime() const = 0;
};

template <enum fk::ParArch PA = fk::defaultParArch>
class TimeMarkerOne;

template <>
class TimeMarkerOne<fk::ParArch::CPU> final : public TimeMarkerInterfaceOne {
    std::array<float, ITERS> m_elapsedTime;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start, m_stop;
public:
    TimeMarkerOne(fk::Stream stream) {
        m_elapsedTime.fill(0.f);
    }
    ~TimeMarkerOne() = default;
    void start() final {
        m_start = std::chrono::high_resolution_clock::now();
    }
    void stop(BenchmarkResultsNumbersOne& resF, const int& idx) final {
        m_stop = std::chrono::high_resolution_clock::now();
        m_elapsedTime[idx] = std::chrono::duration<float, std::milli>(m_stop - m_start).count();
        resF.fkElapsedTimeMax = resF.fkElapsedTimeMax < m_elapsedTime[idx] ? m_elapsedTime[idx] : resF.fkElapsedTimeMax;
        resF.fkElapsedTimeMin = resF.fkElapsedTimeMin > m_elapsedTime[idx] ? m_elapsedTime[idx] : resF.fkElapsedTimeMin;
        resF.fkElapsedTimeAcum += m_elapsedTime[idx];
    };
    std::array<float, ITERS> getElapsedTime() const final {
        return m_elapsedTime;
    }
};

#if defined(__CUDACC__) || defined(__HIP__)
template <>
class TimeMarkerOne<fk::ParArch::GPU_NVIDIA> final : public TimeMarkerInterfaceOne {
    cudaEvent_t m_start, m_stop;
    cudaStream_t m_stream;
    std::array<float, ITERS> m_elapsedTime;
public:
    TimeMarkerOne(fk::Stream stream) : m_stream(stream) {
        gpuErrchk(cudaEventCreate(&m_start));
        gpuErrchk(cudaEventCreate(&m_stop));
        m_elapsedTime.fill(0.f);
    }
    ~TimeMarkerOne() {
        gpuErrchk(cudaEventDestroy(m_start));
        gpuErrchk(cudaEventDestroy(m_stop));
    }
    void start() final {
        gpuErrchk(cudaEventRecord(m_start, m_stream));
    }
    void stop(BenchmarkResultsNumbersOne& resF, const int& idx) final {
        gpuErrchk(cudaEventRecord(m_stop, m_stream));
        gpuErrchk(cudaEventSynchronize(m_stop));
        gpuErrchk(cudaEventElapsedTime(&m_elapsedTime[idx], m_start, m_stop));
        resF.fkElapsedTimeMax = resF.fkElapsedTimeMax < m_elapsedTime[idx] ? m_elapsedTime[idx] : resF.fkElapsedTimeMax;
        resF.fkElapsedTimeMin = resF.fkElapsedTimeMin > m_elapsedTime[idx] ? m_elapsedTime[idx] : resF.fkElapsedTimeMin;
        resF.fkElapsedTimeAcum += m_elapsedTime[idx];
    }
    std::array<float, ITERS> getElapsedTime() const final {
        return m_elapsedTime;
    }
};
#endif // defined(__CUDACC__) || defined(__HIP__)

#if defined(NVRTC_ENABLED)
template <>
class TimeMarkerOne<fk::ParArch::GPU_NVIDIA_JIT> final : public TimeMarkerInterfaceOne {
    cudaEvent_t m_start, m_stop;
    cudaStream_t m_stream;
    std::array<float, ITERS> m_elapsedTime;
public:
    TimeMarkerOne(fk::Stream stream) : m_stream(stream) {
        gpuErrchk(cudaEventCreate(&m_start));
        gpuErrchk(cudaEventCreate(&m_stop));
        m_elapsedTime.fill(0.f);
    }
    ~TimeMarkerOne() {
        gpuErrchk(cudaEventDestroy(m_start));
        gpuErrchk(cudaEventDestroy(m_stop));
    }
    void start() final {
        gpuErrchk(cudaEventRecord(m_start, m_stream));
    }
    void stop(BenchmarkResultsNumbersOne& resF, const int& idx) final {
        gpuErrchk(cudaEventRecord(m_stop, m_stream));
        gpuErrchk(cudaEventSynchronize(m_stop));
        gpuErrchk(cudaEventElapsedTime(&m_elapsedTime[idx], m_start, m_stop));
        resF.fkElapsedTimeMax = resF.fkElapsedTimeMax < m_elapsedTime[idx] ? m_elapsedTime[idx] : resF.fkElapsedTimeMax;
        resF.fkElapsedTimeMin = resF.fkElapsedTimeMin > m_elapsedTime[idx] ? m_elapsedTime[idx] : resF.fkElapsedTimeMin;
        resF.fkElapsedTimeAcum += m_elapsedTime[idx];
    }
    std::array<float, ITERS> getElapsedTime() const final {
        return m_elapsedTime;
    }
};
#endif // defined(NVRTC_ENABLED)

struct BenchmarkResultsNumbersTwo {
    float firstElapsedTimeMax{ fk::minValue<float> };
    float firstElapsedTimeMin{ fk::maxValue<float> };
    float firstElapsedTimeAcum{ 0.f };
    float secondElapsedTimeMax{ fk::minValue<float> };
    float secondElapsedTimeMin{ fk::maxValue<float> };
    float secondElapsedTimeAcum{ 0.f };
};

class TimeMarkerInterfaceTwo {
    virtual void startFirst() = 0;
    virtual void stopFirstStartSecond(BenchmarkResultsNumbersTwo& resF, const int& idx) = 0;
    virtual void stopSecond(BenchmarkResultsNumbersTwo& resF, const int& idx) = 0;
    virtual std::array<float, ITERS> getFirstElapsedTime() const = 0;
    virtual std::array<float, ITERS> getSecondElapsedTime() const = 0;
};

template <enum fk::ParArch PA = fk::defaultParArch>
class TimeMarkerTwo;

template <>
class TimeMarkerTwo<fk::ParArch::CPU> final : public TimeMarkerInterfaceTwo {
    std::array<float, ITERS> m_firstElapsedTime;
    std::array<float, ITERS> m_secondElapsedTime;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start, m_stop;
public:
    TimeMarkerTwo(fk::Stream stream) {
        m_firstElapsedTime.fill(0.f);
        m_secondElapsedTime.fill(0.f);
    }

    ~TimeMarkerTwo() = default;

    void startFirst() final {
        m_start = std::chrono::high_resolution_clock::now();
    };

    void stopFirstStartSecond(BenchmarkResultsNumbersTwo& resF, const int& idx) final {
        m_stop = std::chrono::high_resolution_clock::now();
        m_firstElapsedTime[idx] = std::chrono::duration<float, std::milli>(m_stop - m_start).count();
        resF.firstElapsedTimeMax = resF.firstElapsedTimeMax < m_firstElapsedTime[idx] ? m_firstElapsedTime[idx] : resF.firstElapsedTimeMax;
        resF.firstElapsedTimeMin = resF.firstElapsedTimeMin > m_firstElapsedTime[idx] ? m_firstElapsedTime[idx] : resF.firstElapsedTimeMin;
        resF.firstElapsedTimeAcum += m_firstElapsedTime[idx];
        m_start = std::chrono::high_resolution_clock::now();
    }

    void stopSecond(BenchmarkResultsNumbersTwo& resF, const int& idx) final {
        m_stop = std::chrono::high_resolution_clock::now();
        m_secondElapsedTime[idx] = std::chrono::duration<float, std::milli>(m_stop - m_start).count();
        resF.secondElapsedTimeMax = resF.secondElapsedTimeMax < m_secondElapsedTime[idx] ? m_secondElapsedTime[idx] : resF.secondElapsedTimeMax;
        resF.secondElapsedTimeMin = resF.secondElapsedTimeMin > m_secondElapsedTime[idx] ? m_secondElapsedTime[idx] : resF.secondElapsedTimeMin;
        resF.secondElapsedTimeAcum += m_secondElapsedTime[idx];
    };
    std::array<float, ITERS> getFirstElapsedTime() const final {
        return m_firstElapsedTime;
    };
    std::array<float, ITERS> getSecondElapsedTime() const final {
        return m_secondElapsedTime;
    };
};

#if defined(__CUDACC__) || defined(__HIP__)
template <>
class TimeMarkerTwo<fk::ParArch::GPU_NVIDIA> final : public TimeMarkerInterfaceTwo {
    cudaEvent_t m_start, m_stop;
    cudaStream_t m_stream;
    std::array<float, ITERS> m_firstElapsedTime;
    std::array<float, ITERS> m_secondElapsedTime;
public:
    TimeMarkerTwo(fk::Stream stream) : m_stream(stream) {
        gpuErrchk(cudaEventCreate(&m_start));
        gpuErrchk(cudaEventCreate(&m_stop));
        m_firstElapsedTime.fill(0.f);
        m_secondElapsedTime.fill(0.f);
    }

    ~TimeMarkerTwo() {
        gpuErrchk(cudaEventDestroy(m_start));
        gpuErrchk(cudaEventDestroy(m_stop));
    }

    void startFirst() final {
        gpuErrchk(cudaEventRecord(m_start, m_stream));
    };
    void stopFirstStartSecond(BenchmarkResultsNumbersTwo& resF, const int& idx) final {
        gpuErrchk(cudaEventRecord(m_stop, m_stream));
        gpuErrchk(cudaEventSynchronize(m_stop));
        gpuErrchk(cudaEventElapsedTime(&m_firstElapsedTime[idx], m_start, m_stop));
        resF.firstElapsedTimeMax = resF.firstElapsedTimeMax < m_firstElapsedTime[idx] ? m_firstElapsedTime[idx] : resF.firstElapsedTimeMax;
        resF.firstElapsedTimeMin = resF.firstElapsedTimeMin > m_firstElapsedTime[idx] ? m_firstElapsedTime[idx] : resF.firstElapsedTimeMin;
        resF.firstElapsedTimeAcum += m_firstElapsedTime[idx];
        gpuErrchk(cudaEventRecord(m_start, m_stream));
    }
    void stopSecond(BenchmarkResultsNumbersTwo& resF, const int& idx) final {
        gpuErrchk(cudaEventRecord(m_stop, m_stream));
        gpuErrchk(cudaEventSynchronize(m_stop));
        gpuErrchk(cudaEventElapsedTime(&m_secondElapsedTime[idx], m_start, m_stop));
        resF.secondElapsedTimeMax = resF.secondElapsedTimeMax < m_secondElapsedTime[idx] ? m_secondElapsedTime[idx] : resF.secondElapsedTimeMax;
        resF.secondElapsedTimeMin = resF.secondElapsedTimeMin > m_secondElapsedTime[idx] ? m_secondElapsedTime[idx] : resF.secondElapsedTimeMin;
        resF.secondElapsedTimeAcum += m_secondElapsedTime[idx];
    };
    std::array<float, ITERS> getFirstElapsedTime() const final {
        return m_firstElapsedTime;
    };
    std::array<float, ITERS> getSecondElapsedTime() const final {
        return m_secondElapsedTime;
    };
};
#endif

#if defined(NVRTC_ENABLED)
template <>
class TimeMarkerTwo<fk::ParArch::GPU_NVIDIA_JIT> final : public TimeMarkerInterfaceTwo {
    cudaEvent_t m_start, m_stop;
    cudaStream_t m_stream;
    std::array<float, ITERS> m_firstElapsedTime;
    std::array<float, ITERS> m_secondElapsedTime;
public:
    TimeMarkerTwo(fk::Stream stream) : m_stream(stream) {
        gpuErrchk(cudaEventCreate(&m_start));
        gpuErrchk(cudaEventCreate(&m_stop));
        m_firstElapsedTime.fill(0.f);
        m_secondElapsedTime.fill(0.f);
    }

    ~TimeMarkerTwo() {
        gpuErrchk(cudaEventDestroy(m_start));
        gpuErrchk(cudaEventDestroy(m_stop));
    }

    void startFirst() final {
        gpuErrchk(cudaEventRecord(m_start, m_stream));
    };
    void stopFirstStartSecond(BenchmarkResultsNumbersTwo& resF, const int& idx) final {
        gpuErrchk(cudaEventRecord(m_stop, m_stream));
        gpuErrchk(cudaEventSynchronize(m_stop));
        gpuErrchk(cudaEventElapsedTime(&m_firstElapsedTime[idx], m_start, m_stop));
        resF.firstElapsedTimeMax = resF.firstElapsedTimeMax < m_firstElapsedTime[idx] ? m_firstElapsedTime[idx] : resF.firstElapsedTimeMax;
        resF.firstElapsedTimeMin = resF.firstElapsedTimeMin > m_firstElapsedTime[idx] ? m_firstElapsedTime[idx] : resF.firstElapsedTimeMin;
        resF.firstElapsedTimeAcum += m_firstElapsedTime[idx];
        gpuErrchk(cudaEventRecord(m_start, m_stream));
    }
    void stopSecond(BenchmarkResultsNumbersTwo& resF, const int& idx) final {
        gpuErrchk(cudaEventRecord(m_stop, m_stream));
        gpuErrchk(cudaEventSynchronize(m_stop));
        gpuErrchk(cudaEventElapsedTime(&m_secondElapsedTime[idx], m_start, m_stop));
        resF.secondElapsedTimeMax = resF.secondElapsedTimeMax < m_secondElapsedTime[idx] ? m_secondElapsedTime[idx] : resF.secondElapsedTimeMax;
        resF.secondElapsedTimeMin = resF.secondElapsedTimeMin > m_secondElapsedTime[idx] ? m_secondElapsedTime[idx] : resF.secondElapsedTimeMin;
        resF.secondElapsedTimeAcum += m_secondElapsedTime[idx];
    };
    std::array<float, ITERS> getFirstElapsedTime() const final {
        return m_firstElapsedTime;
    };
    std::array<float, ITERS> getSecondElapsedTime() const final {
        return m_secondElapsedTime;
    };
};
#endif // NVRTC_ENABLED

#endif // FK_BENCHMARKS_COMMON_H