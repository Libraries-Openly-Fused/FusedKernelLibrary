/* Copyright 2026 Oscar Amoros Huguet
   Copyright 2026 Johnny Nunez

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_UTESTS_ATTENTION_TEST_COMMON_H
#define FK_UTESTS_ATTENTION_TEST_COMMON_H

/* Shared by the attention utests (the "_common" in the name keeps test
 * discovery from turning it into a test): the double-precision CPU oracle,
 * pass/fail reporting, and Ptr1D buffers moved through their pinned mirror.
 * Include it after the attention header under test. */

#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/execution_model/stream.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace fk {

// double-precision CPU oracle: O = softmax(scale * Q K^T [causal]) V
inline void cpuAttention(const std::vector<double>& q, const std::vector<double>& k,
                         const std::vector<double>& v, std::vector<double>& o,
                         const int bh, const int seqQ, const int seqK, const int d,
                         const double scale, const bool causal) {
    o.assign((size_t)bh * seqQ * d, 0.0);
    std::vector<double> s(seqK);
    for (int b = 0; b < bh; ++b) {
        const double* Q = q.data() + (size_t)b * seqQ * d;
        const double* K = k.data() + (size_t)b * seqK * d;
        const double* V = v.data() + (size_t)b * seqK * d;
        double* O = o.data() + (size_t)b * seqQ * d;
        for (int i = 0; i < seqQ; ++i) {
            const int kEnd = causal ? std::min(seqK, i + 1) : seqK;
            double m = -1e300;
            for (int j = 0; j < kEnd; ++j) {
                double dot = 0.0;
                for (int c = 0; c < d; ++c) dot += Q[(size_t)i*d+c] * K[(size_t)j*d+c];
                s[j] = dot * scale;
                m = std::max(m, s[j]);
            }
            double l = 0.0;
            for (int j = 0; j < kEnd; ++j) { s[j] = std::exp(s[j] - m); l += s[j]; }
            for (int j = 0; j < kEnd; ++j) {
                const double pj = s[j] / l;
                for (int c = 0; c < d; ++c) O[(size_t)i*d+c] += pj * V[(size_t)j*d+c];
            }
        }
    }
}

inline int failures = 0;

inline void report(const char* name, const double maxErr, const double tol) {
    if (maxErr > tol) {
        std::cout << "FAIL " << name << ": maxErr=" << maxErr << " tol=" << tol << std::endl;
        ++failures;
    } else {
        std::cout << "Running test " << name << ": Success!! (maxErr=" << maxErr << ")" << std::endl;
    }
}

template <typename T>
inline double maxAbsErr(const std::vector<T>& got, const std::vector<double>& ref) {
    double maxErr = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        maxErr = std::max(maxErr, std::abs(static_cast<double>(got[i]) - ref[i]));
    }
    return maxErr;
}

// Device buffer holding a copy of host: written to the pinned mirror, then
// uploaded on the stream.
template <typename T>
inline Ptr1D<T> uploadBuffer(const std::vector<T>& host, Stream& stream) {
    Ptr1D<T> buffer(static_cast<uint>(host.size()));
    std::copy(host.begin(), host.end(), buffer.ptrPinned().data);
    buffer.upload(stream);
    return buffer;
}

// Device buffer of n elements, zeroed (through the pinned mirror).
template <typename T>
inline Ptr1D<T> zeroBuffer(const size_t n, Stream& stream) {
    return uploadBuffer(std::vector<T>(n, T{}), stream);
}

// Waits for the stream and returns the contents of the device buffer.
template <typename T>
inline std::vector<T> downloadBuffer(Ptr1D<T>& buffer, Stream& stream) {
    buffer.download(stream);
    stream.sync();
    const T* host = buffer.ptrPinned().data;
    return std::vector<T>(host, host + buffer.dims().width);
}

} // namespace fk

#endif // FK_UTESTS_ATTENTION_TEST_COMMON_H
