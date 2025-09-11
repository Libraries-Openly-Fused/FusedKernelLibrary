/* Copyright 2025 Oscar Amoros Huguet
   Copyright 2025 Albert Andaluz
   Copyright 2025 Grup Mediapro S.L.U

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef FK_OPERATION_TEST_UTILS_H
#define FK_OPERATION_TEST_UTILS_H

#include <map>
#include <functional>
#include <iostream>

#include <fused_kernel/core/utils/cuda_vector_utils.h>
#include <fused_kernel/core/utils/type_to_string.h>
#include <fused_kernel/core/execution_model/executors.h>

std::map<std::string, std::function<bool()>> testCases;

#define START_ADDING_TESTS void addTests() {
#define STOP_ADDING_TESTS }
#define RUN_ALL_TESTS \
addTests(); \
bool correct{true}; \
for (const auto& [testName, testFunc] : testCases) { \
    if (!testFunc()) { \
        correct = false; \
    } \
} \
testCases.clear(); \
return correct ? 0 : -1;

template <typename T>
constexpr inline bool equalValues(const T & val1, const T & val2) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::abs(val1 - val2) < static_cast<T>(0.0001);
    } else {
        return val1 == val2;
    }
}

template <typename T>
constexpr inline bool equalInstances(const T& instance1, const T& instance2) {
    if constexpr (fk::validCUDAVec<T>) {
        const auto i1 = fk::toArray(instance1);
        const auto i2 = fk::toArray(instance2);
        constexpr size_t N = static_cast<size_t>(fk::cn<T>);
        const fk::Array<bool, N> equalArray = fk::transformArray(fk::makeIndexArray<N>(),
            [&] (const size_t& idx) constexpr {
                return equalValues(i1[idx], i2[idx]);
            }
        );
        return fk::allValuesAre(true, equalArray);
    } else if constexpr (std::is_fundamental_v<T>) {
        return equalValues(instance1, instance2);
    } else {
        // Assuming the type has an equality operator defined
        return instance1 == instance2;
    }
}

template <typename T>
inline bool comparePtrs1D(const fk::Ptr<fk::ND::_1D, T>&ptr1, const fk::Ptr<fk::ND::_1D, T>&ptr2) {
    const fk::PtrDims<fk::ND::_1D> dims1 = ptr1.dims();
    const fk::PtrDims<fk::ND::_1D> dims2 = ptr2.dims();
    if (dims1.width != dims2.width) {
        return false;
    }
    for (uint i = 0; i < dims1.width; ++i) {
        if (!equalInstances(ptr1.at(i), ptr2.at(i))) {
            return false;
        }
    }
    return true;
}

template <fk::ND D, typename T>
void printPtr(const fk::Ptr<D, T>& ptr) {
    if constexpr (D == fk::ND::_1D) {
        const auto dims = ptr.dims();
        std::cout << "{ ";
        for (uint i = 0; i < dims.width; ++i) {
            if constexpr (fk::validCUDAVec<T>) {
                const auto val = ptr.at(i);
                const auto arr = fk::toArray(val);
                std::cout << "(";
                for (size_t j = 0; j < fk::cn<T>; ++j) {
                    if (j > 0) std::cout << ", ";
                    if constexpr (sizeof(fk::VBase<T>) < 4) {
                        std::cout << static_cast<int>(arr[j]);
                    } else {
                        std::cout << arr[j];
                    }
                }
                std::cout << ")";
            } else {
                const auto val = ptr.at(i);
                if constexpr (sizeof(T) < 4) {
                    std::cout << static_cast<int>(val);
                } else {
                    std::cout << val;
                }
            }
            if (i < dims.width - 1) std::cout << ", ";
        }
        std::cout << " }" << std::endl;

    } else if constexpr (D == fk::ND::_2D) {
        const auto dims = ptr.dims();
        std::cout << "{" << std::endl;
        for (uint y = 0; y < dims.height; ++y) {
            std::cout << "  ";
            for (uint x = 0; x < dims.width; ++x) {
                if constexpr (fk::validCUDAVec<T>) {
                    const auto val = ptr.at(x, y);
                    const auto arr = fk::toArray(val);
                    std::cout << "(";
                    for (size_t j = 0; j < fk::cn<T>; ++j) {
                        if (j > 0) std::cout << ", ";
                        if constexpr (sizeof(fk::VBase<T>) < 4) {
                            std::cout << static_cast<int>(arr[j]);
                        } else {
                            std::cout << arr[j];
                        }
                    }
                    std::cout << ")";
                } else {
                    const auto val = ptr.at(x, y);
                    if constexpr (sizeof(T) < 4) {
                        std::cout << static_cast<int>(val);
                    } else {
                        std::cout << val;
                    }
                }
                if (x < dims.width - 1) std::cout << ", ";
            }
            std::cout << std::endl;
        }
        std::cout << "}" << std::endl;

    } else if constexpr (D == fk::ND::_3D) {
        const auto dims = ptr.dims();
        for (uint z = 0; z < dims.planes; ++z) {
            std::cout << "Plane " << z << ":" << std::endl;
            std::cout << "{" << std::endl;
            for (uint y = 0; y < dims.height; ++y) {
                std::cout << "  ";
                for (uint x = 0; x < dims.width; ++x) {
                    if constexpr (fk::validCUDAVec<T>) {
                        const auto val = ptr.at(x, y, z);
                        const auto arr = fk::toArray(val);
                        std::cout << "(";
                        for (size_t j = 0; j < fk::cn<T>; ++j) {
                            if (j > 0) std::cout << ", ";
                            if constexpr (sizeof(fk::VBase<T>) < 4) {
                                std::cout << static_cast<int>(arr[j]);
                            } else {
                                std::cout << arr[j];
                            }
                        }
                        std::cout << ")";
                    } else {
                        const auto val = ptr.at(x, y, z);
                        if constexpr (sizeof(T) < 4) {
                            std::cout << static_cast<int>(val);
                        } else {
                            std::cout << val;
                        }
                    }
                    if (x < dims.width - 1) std::cout << ", ";
                }
                std::cout << std::endl;
            }
            std::cout << "}" << std::endl;
            if (z < dims.planes - 1) std::cout << std::endl;
        }
    }
}

template <typename T>
inline bool comparePtrs2D(const fk::Ptr<fk::ND::_2D, T>& ptr1, const fk::Ptr<fk::ND::_2D, T>& ptr2) {
    const fk::PtrDims<fk::ND::_2D> dims1 = ptr1.dims();
    const fk::PtrDims<fk::ND::_2D> dims2 = ptr2.dims();
    if (dims1.width != dims2.width || dims1.height != dims2.height) {
        return false;
    }
    for (uint y = 0; y < dims1.height; ++y) {
        for (uint x = 0; x < dims1.width; ++x) {
            if (!equalInstances(ptr1.at(x, y), ptr2.at(x, y))) {
                printPtr(ptr1);
                printPtr(ptr2);
                return false;
            }
        }
    }
    return true;
}

template <typename T>
inline bool comparePtrs3D(const fk::Ptr<fk::ND::_3D, T>& ptr1, const fk::Ptr<fk::ND::_3D, T>& ptr2) {
    const fk::PtrDims<fk::ND::_3D> dims1 = ptr1.dims();
    const fk::PtrDims<fk::ND::_3D> dims2 = ptr2.dims();
    if (dims1.width != dims2.width || dims1.height != dims2.height ||
        dims1.planes != dims2.planes || dims1.color_planes != dims2.color_planes) {
        return false;
    }
    for (uint z = 0; z < dims1.planes; ++z) {
        for (uint y = 0; y < dims1.height; ++y) {
            for (uint x = 0; x < dims1.width; ++x) {
                if (!equalInstances(ptr1.at(x, y, z), ptr2.at(x, y, z))) {
                    return false;
                }
            }
        }
    }
    return true;
}

template <fk::ND D, typename T>
inline bool comparePtrs(const fk::Ptr<D, T>& ptr1, const fk::Ptr<D, T>&ptr2) {
    if constexpr (D == fk::ND::_1D) {
        return comparePtrs1D(ptr1, ptr2);
    } else if constexpr (D == fk::ND::_2D) {
        return comparePtrs2D(ptr1, ptr2);
    } else if constexpr (D == fk::ND::_3D) {
        return comparePtrs3D(ptr1, ptr2);
    }
}


template <typename Operation, typename = void>
struct TestCaseBuilder;

namespace test_case_builder::detail {
    template <typename Operation, size_t N>
    fk::Ptr1D<typename Operation::OutputType>
    launchUnary(const std::string& testName,
                const std::array<typename Operation::InputType, N>& inputElems) {
        fk::Stream stream;
        using I = typename Operation::InputType;
        using O = typename Operation::OutputType;
        fk::Ptr1D<I> inputPtr(N);
        fk::Ptr1D<O> outputPtr(N);
        for (size_t i = 0; i < N; ++i) {
            inputPtr.at(fk::Point(i)) = inputElems[i];
        }
        std::cout << "Running test for " << "\033[1;33m" <<testName << "\033[1;33m" << ": ";
        inputPtr.upload(stream);
        fk::Executor<fk::TransformDPP<>>::executeOperations(inputPtr, outputPtr, stream,
                                                            Operation::build());
        outputPtr.download(stream);
        stream.sync();
        return outputPtr;
    }
    template <typename Operation, fk::ND D, size_t N, typename BuildParams>
    auto launchRead(const std::string& testName, fk::Stream stream,
                    const std::array<BuildParams, N>& inputElems) {
        using OutputType = typename Operation::OutputType;

        auto readOps = fk::transformArray(inputElems,
            [](const BuildParams& input) {
                return Operation::build(input);
            }
        );
     
        std::array<fk::Ptr<D, OutputType>, N> outputElems =
            fk::transformArray(inputElems, [](const BuildParams& input) {
                const auto iROp = Operation::build(input);
                using ROp = typename std::decay_t<decltype(iROp)>::Operation;
                const fk::Point point(0, 0, 0);
                const uint num_elems_x = ROp::num_elems_x(point, iROp);
                if constexpr (D == fk::ND::_1D) {
                    return fk::Ptr<D, OutputType>(num_elems_x);
                } else if constexpr (D == fk::ND::_2D) {
                    const uint num_elems_y = ROp::num_elems_y(point, iROp);
                    return fk::Ptr<D, OutputType>(num_elems_x, num_elems_y);
                } else {
                    const uint num_elems_y = ROp::num_elems_y(point, iROp);
                    const uint num_elems_z = ROp::num_elems_z(point, iROp);
                    return fk::Ptr<D, OutputType>(num_elems_x, num_elems_y, num_elems_z);
                }
            }
        );

        auto writeOps = fk::transformArray(outputElems,
            [](const fk::Ptr<D, OutputType>& output) {
                return fk::PerThreadWrite<D, OutputType>::build(output);
            }
        );

        std::cout << "Running test for " << "\033[1;33m" << testName << "\033[1;33m" << ": ";
        
        for (int i = 0; i < N; ++i) {
            const auto activeThreads = std::decay_t<decltype(readOps[i])>::Operation::getActiveThreads(readOps[i]);
            fk::Executor<fk::TransformDPP<>>::executeOperations(stream, readOps[i], writeOps[i]);
        }
        for (auto&& output : outputElems) {
            output.download(stream);
        }
        stream.sync();
        
        return outputElems;
    }
}

template <typename Operation>
struct TestCaseBuilder<Operation, std::enable_if_t<fk::IsUnaryType<Operation>::value &&
                                    std::is_fundamental_v<typename Operation::InputType> &&
                                    std::is_fundamental_v<typename Operation::OutputType>, void>> {
    template <size_t N>
    static inline void addTest(std::map<std::string, std::function<bool()>>& testCases,
                               const std::array<typename Operation::InputType, N>& inputElems,
                               const std::array<typename Operation::OutputType, N>& expectedElems) {
        const std::string testName = fk::typeToString<Operation>();
        testCases[testName] = [testName, inputElems, expectedElems]() {
            const auto outputPtr = test_case_builder::detail::launchUnary<Operation>(testName, inputElems);
            bool result{ true };
            for (size_t i = 0; i < N; ++i) {
                const auto generated = outputPtr.at(fk::Point(i));
                static_assert(std::is_same_v<std::decay_t<decltype(generated)>, std::decay_t<decltype(expectedElems[i])>>, "Output and Expected types are not the same");
                const auto resultV = generated == expectedElems[i];
                if (!resultV) {
                    std::cout << "\033[32m" << "FAIL!!" << "\033[0m" << std::endl;
                    if constexpr (sizeof(typename Operation::OutputType) < 4) {
                            std::cout << "\033[31m Mismatch at test element index " << i << ": Expected value "
                                  << static_cast<int>(expectedElems[i]) << ", got " << static_cast<int>(generated) << "\033[0m" << std::endl;
                    } else {
                            std::cout << "\033[31m Mismatch at test element index " << i << ": Expected value "
                                << expectedElems[i] << ", got " << generated << "\033[0m" << std::endl;
                    }
                }
                result &= resultV;
            }
            if (result)
                std::cout << "\033[32m" << "Success!!" << "\033[0m" <<std::endl;
            return result;
            };
    }
};

template <typename Operation>
struct TestCaseBuilder<Operation, std::enable_if_t<fk::IsUnaryType<Operation>::value &&
                                    (fk::validCUDAVec<typename Operation::InputType> ||
                                     fk::validCUDAVec<typename Operation::OutputType>), void>> {
    template <size_t N>
    static inline void addTest(std::map<std::string, std::function<bool()>>& testCases,
                               const std::array<typename Operation::InputType, N>& inputElems,
                               const std::array<typename Operation::OutputType, N>& expectedElems) {
        const std::string testName = fk::typeToString<Operation>();
        testCases[testName] = [testName, inputElems, expectedElems]() -> bool {
            const auto outputPtr = test_case_builder::detail::launchUnary<Operation>(testName, inputElems);
            bool result{ true };
            for (size_t i = 0; i < N; ++i) {
                const auto generated = outputPtr.at(fk::Point(i));
                static_assert(std::is_same_v<std::decay_t<decltype(generated)>, std::decay_t<decltype(expectedElems[i])>>, "Output and Expected types are not the same");
                const auto resultV = generated == expectedElems[i];
                const auto arrayGenerated = fk::toArray(generated);
                const auto arrayResult = fk::toArray(resultV);
                for (size_t j = 0; j < fk::cn<decltype(resultV)>; ++j) {
                    if (!arrayResult.at[j]) {
                      
                        std::cout << "\033[31m" << "FAIL!!" << "\033[31m" <<std::endl;
                        if constexpr (sizeof(fk::VBase<typename Operation::OutputType>) < 4) {

                            std::cout << "\033[31m" << "Mismatch at test element index " << i << " for vector index "
                                      << j << ": Expected value "
                                      << static_cast<int>(fk::toArray(expectedElems[i]).at[j]) << ", got "
                                      << static_cast<int>(arrayGenerated.at[j]) << "\033[0m" << std::endl;
                        } else {
                            std::cout << "\033[31m" << "Mismatch at test element index " << i << " for vector index "
                                      << j << ": Expected value "
                                      << fk::toArray(expectedElems[i]).at[j] << ", got "
                                      << arrayGenerated.at[j] << "\033[0m" << std::endl;
                        }
                    }
                    result &= arrayResult[j];
                }
            }
            if (result) {
                std::cout << "\033[32m" << "Success!!" << "\033[0m" << std::endl;
            }
            return result;
        };
    }
};

template <typename Operation>
struct TestCaseBuilder<Operation, std::enable_if_t<fk::IsReadType<Operation>::value || fk::IsReadBackType<Operation>::value, void>> {
    template <fk::ND D, size_t N, typename BuildParams>
    static inline void addTest(std::map<std::string, std::function<bool()>>& testCases,
                               fk::Stream& stream,
                               const std::array<BuildParams, N>& inputElems,
                               const std::array<fk::Ptr<D, typename Operation::OutputType>, N>& expectedElems) {
        const std::string testName = fk::typeToString<Operation>();
        testCases[testName] = [testName, stream, inputElems, expectedElems]() -> bool {
            const auto outArray = test_case_builder::detail::launchRead<Operation, D>(testName, stream, inputElems);
            bool result{ true };
            for (size_t i = 0; i < N; ++i) {
                const auto& outputPtr = outArray[i];
                const auto& expectedPtr = expectedElems[i];

                const bool correct = comparePtrs<D>(outputPtr, expectedPtr);
                if (!correct) {
                    result = false;
                    std::cout << "\033[31m" << "FAIL!!" << "\033[0m" << std::endl;
                    std::cout << "\033[31m" << "Mismatch at test element index " << i << ": Expected output does not match generated output." << "\033[0m" << std::endl;
                }
            }
            if (result) {
                std::cout << "\033[32m" << "Success!!" << "\033[0m" << std::endl;
            }

            return result;
            };
    }
    template <fk::ND D, typename BuildParams>
    static inline void addTest(std::map<std::string, std::function<bool()>>&testCases,
                                fk::Stream & stream,
                                const BuildParams & inputElem,
                                const fk::Ptr<D, typename Operation::OutputType>& expectedElem) {
        const std::array<BuildParams, 1> inputElems{ inputElem };
        const std::array<fk::Ptr<D, typename Operation::OutputType>, 1> expectedElems{ expectedElem };
        addTest<D, 1>(testCases, stream, inputElems, expectedElems);
    }
};

#endif // FK_OPERATION_TEST_UTILS_H
