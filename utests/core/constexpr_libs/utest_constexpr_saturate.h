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

#ifndef FK_TEST_CONSTEXPR_SATURATE_H
#define FK_TEST_CONSTEXPR_SATURATE_H

#include <fused_kernel/core/constexpr_libs/constexpr_saturate.h>
#include <limits>
#include <iostream>

// Compile-time tests for cxp::saturate_float
constexpr bool test_saturate_float_ct() {
	// Values inside [0, 1] are returned unchanged
	static_assert(cxp::saturate_float::f(0.f) == 0.f, "saturate_float(0.f) should be 0.f");
	static_assert(cxp::saturate_float::f(1.f) == 1.f, "saturate_float(1.f) should be 1.f");
	static_assert(cxp::saturate_float::f(0.5f) == 0.5f, "saturate_float(0.5f) should be 0.5f");
	static_assert(cxp::saturate_float::f(0.25f) == 0.25f, "saturate_float(0.25f) should be 0.25f");

	// Values below 0 are clamped to 0
	static_assert(cxp::saturate_float::f(-0.5f) == 0.f, "saturate_float(-0.5f) should be 0.f");
	static_assert(cxp::saturate_float::f(-1.f) == 0.f, "saturate_float(-1.f) should be 0.f");
	static_assert(cxp::saturate_float::f(-1000.f) == 0.f, "saturate_float(-1000.f) should be 0.f");
	static_assert(cxp::saturate_float::f(std::numeric_limits<float>::lowest()) == 0.f,
				  "saturate_float(lowest) should be 0.f");
	static_assert(cxp::saturate_float::f(-std::numeric_limits<float>::infinity()) == 0.f,
				  "saturate_float(-inf) should be 0.f");

	// Values above 1 are clamped to 1
	static_assert(cxp::saturate_float::f(1.5f) == 1.f, "saturate_float(1.5f) should be 1.f");
	static_assert(cxp::saturate_float::f(1000.f) == 1.f, "saturate_float(1000.f) should be 1.f");
	static_assert(cxp::saturate_float::f(std::numeric_limits<float>::max()) == 1.f,
				  "saturate_float(max) should be 1.f");
	static_assert(cxp::saturate_float::f(std::numeric_limits<float>::infinity()) == 1.f,
				  "saturate_float(+inf) should be 1.f");

	// Values just outside and just inside the bounds
	constexpr float epsilon = std::numeric_limits<float>::epsilon();
	static_assert(cxp::saturate_float::f(1.f - epsilon) == 1.f - epsilon,
				  "saturate_float(1 - epsilon) should be unchanged");
	static_assert(cxp::saturate_float::f(1.f + epsilon) == 1.f, "saturate_float(1 + epsilon) should be 1.f");
	static_assert(cxp::saturate_float::f(std::numeric_limits<float>::denorm_min()) ==
					  std::numeric_limits<float>::denorm_min(),
				  "saturate_float(denorm_min) should be unchanged");
	static_assert(cxp::saturate_float::f(-std::numeric_limits<float>::denorm_min()) == 0.f,
				  "saturate_float(-denorm_min) should be 0.f");

	// The result is always a float
	static_assert(std::is_same_v<decltype(cxp::saturate_float::f(0.5f)), float>,
				  "saturate_float should return float");

	return true;
}

// Runtime tests for cxp::saturate_float
inline bool test_saturate_float_rt() {
	bool allCorrect{true};

	const auto check = [&allCorrect](const float input, const float expected, const char* const message) {
		const float result = cxp::saturate_float::f(input);
		if (!(result == expected)) {
			std::cout << "Failed: " << message << " (got " << result << ", expected " << expected << ")" << std::endl;
			allCorrect = false;
		}
	};

	check(0.f, 0.f, "cxp::saturate_float::f(0.f) should be 0.f");
	check(1.f, 1.f, "cxp::saturate_float::f(1.f) should be 1.f");
	check(0.5f, 0.5f, "cxp::saturate_float::f(0.5f) should be 0.5f");
	check(0.75f, 0.75f, "cxp::saturate_float::f(0.75f) should be 0.75f");
	check(-0.f, 0.f, "cxp::saturate_float::f(-0.f) should be 0.f");
	check(-0.5f, 0.f, "cxp::saturate_float::f(-0.5f) should be 0.f");
	check(-1000.f, 0.f, "cxp::saturate_float::f(-1000.f) should be 0.f");
	check(1.5f, 1.f, "cxp::saturate_float::f(1.5f) should be 1.f");
	check(1000.f, 1.f, "cxp::saturate_float::f(1000.f) should be 1.f");
	check(std::numeric_limits<float>::lowest(), 0.f, "cxp::saturate_float::f(lowest) should be 0.f");
	check(std::numeric_limits<float>::max(), 1.f, "cxp::saturate_float::f(max) should be 1.f");
	check(-std::numeric_limits<float>::infinity(), 0.f, "cxp::saturate_float::f(-inf) should be 0.f");
	check(std::numeric_limits<float>::infinity(), 1.f, "cxp::saturate_float::f(+inf) should be 1.f");

	const float epsilon = std::numeric_limits<float>::epsilon();
	check(1.f - epsilon, 1.f - epsilon, "cxp::saturate_float::f(1 - epsilon) should be unchanged");
	check(1.f + epsilon, 1.f, "cxp::saturate_float::f(1 + epsilon) should be 1.f");
	check(std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::denorm_min(),
		  "cxp::saturate_float::f(denorm_min) should be unchanged");
	check(-std::numeric_limits<float>::denorm_min(), 0.f, "cxp::saturate_float::f(-denorm_min) should be 0.f");

	return allCorrect;
}

int launch() {
	static_assert(test_saturate_float_ct(), "saturate_float compile-time tests failed");

	if (!test_saturate_float_rt()) {
		return -1;
	}

	std::cout << "All tests passed!" << std::endl;
	return 0;
}

#endif // FK_TEST_CONSTEXPR_SATURATE_H
