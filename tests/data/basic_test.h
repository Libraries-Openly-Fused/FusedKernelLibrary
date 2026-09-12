/* Copyright 2023-2026 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include <tests/main.h>

#include <iostream>

#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/data/ptr_utils.h>
#include <fused_kernel/core/execution_model/operation_model/operation_model.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/image_processing/saturate.h>
#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/core/utils/template_operations.h>
#include <fused_kernel/core/execution_model/stream.h>
#include <fused_kernel/algorithms/basic_ops/vector_ops.h>

namespace fk {

template <typename T>
bool testPtr_2D() {
    constexpr size_t width = 1920;
    constexpr size_t height = 1080;
    constexpr size_t width_crop = 300;
    constexpr size_t height_crop = 200;

    Point startPoint = {100, 200, 0};

    Stream stream;

    Ptr2D<T> input(width, height);
    setTo(make_set<T>(2), input, stream);
    Ptr2D<T> cropedInput = input.crop(startPoint, PtrDims<ND::_2D>(width_crop, height_crop));
    Ptr2D<T> output(width_crop, height_crop);
    Ptr2D<T> outputBig(width, height);

    Read<PerThreadRead<ND::_2D, T>> readCrop{{cropedInput}};
    Read<PerThreadRead<ND::_2D, T>> readFull{{input}};

    WriteInstantiableOperation<PerThreadWrite<ND::_2D, T>> opFinal_2D = { {output} };
    WriteInstantiableOperation<PerThreadWrite<ND::_2D, T>> opFinal_2DBig = { {outputBig} };

    for (int i=0; i<100; i++) {
        executeOperations<TransformDPP<>>(stream, readCrop, opFinal_2D);
        executeOperations<TransformDPP<>>(stream, readFull, opFinal_2DBig);
    }

    output.download(stream);
    outputBig.download(stream);

    stream.sync();

    for (int y = 0; y < output.dims().height; ++y) {
        for (int x = 0; x < output.dims().width; ++x) {
            const auto result = output.at({ x, y }) != make_set<T>(2);
            if (cxp::vector_and::f(result)) {
                if constexpr (cn<T> == 1 && !std::is_aggregate_v<T>) {
                    std::cout << "Error in output at (" << x << ", " << y << "): " << static_cast<int>(output.at({ x, y })) << std::endl;
                } else {
                    std::cout << "Error in output at (" << x << ", " << y << "): ";
                    for (size_t i = 0; i < cn<T>; ++i) {
                        std::cout << static_cast<int>(toArray(output.at({ x, y })).at[i]) << " ";
                    }
                    std::cout << std::endl;
                }
                return false;
            }
        }
    }

    return true;
}



int launch_impl() {
    bool test2Dpassed = true;

    test2Dpassed &= testPtr_2D<uchar>();
    test2Dpassed &= testPtr_2D<uchar3>();
    test2Dpassed &= testPtr_2D<float>();
    test2Dpassed &= testPtr_2D<float3>();

    Stream stream;

    Ptr2D<uchar> input(64,64);
    Ptr2D<uint> output(64,64);

    Read<PerThreadRead<ND::_2D, uchar>> read{ {input} };
    Unary<SaturateCast<uchar, uint>> cast = {};
    Write<PerThreadWrite<ND::_2D, uint>> write { {output} };

    auto fusedDF = fuse(read, cast, Binary<Mul<uint>>{4u});
    constexpr bool correct = std::is_same_v<std::decay_t<decltype(fusedDF.params)>,
                       OperationTuple_<void, Read<PerThreadRead<ND::_2D, uchar>>,
                                              Unary<SaturateCast<uchar, uint>>, Binary<Mul<uint>>>>;
    static_assert(correct, "Unexpected type for fusedDF.params");
    constexpr bool correct2 =
        std::is_same_v<std::decay_t<decltype(get_opt<0>(fusedDF.params))>, Read<PerThreadRead<ND::_2D, uchar>>>;
    static_assert(correct2, "Unexpected type for get<0>(fusedDF.params)");
    //fusedDF.params.next.instance.params; // Should not compile
    auto params2 = get_opt<2>(fusedDF.params).params;
    static_assert(std::is_same_v<std::decay_t<decltype(params2)>, uint>, "Unexpected type for params");

    executeOperations<TransformDPP<>>(stream, fusedDF, write);
    stream.sync();

    OperationTuple<Read<PerThreadRead<ND::_2D, uchar>>, Unary<SaturateCast<uchar, uint>>, Write<PerThreadWrite<ND::_2D, uint>>> myTup{};

    get_opt<2>(myTup);
    constexpr bool test1 = std::is_same_v<TypeAt_t<0, typename decltype(myTup)::Operations>, Read<PerThreadRead<ND::_2D, uchar>>>;
    constexpr bool test2 =
        std::is_same_v<TypeAt_t<1, typename decltype(myTup)::Operations>, Unary<SaturateCast<uchar, uint>>>;
    constexpr bool test3 =
        std::is_same_v<TypeAt_t<2, typename decltype(myTup)::Operations>, Write<PerThreadWrite<ND::_2D, uint>>>;

    if (test2Dpassed && and_v<test1, test2, test3>) {
        std::cout << "gpu_transform executed!!" << std::endl;
        return 0;
    } else {
        std::cout << "gpu_transform failed!!" << std::endl;
        if (!test2Dpassed) {
            std::cout << "Specifically testPtr_2D failed!!" << std::endl;
        }
        if (!test1) {
            std::cout << "Specifically test1 failed!!" << std::endl;
        }
        if (!test2) {
            std::cout << "Specifically test2 failed!!" << std::endl;
        }
        if (!test3) {
            std::cout << "Specifically test3 failed!!" << std::endl;
        }
        return -1;
    }
}

} // namespace fk

int launch() {
    return fk::launch_impl();
}
