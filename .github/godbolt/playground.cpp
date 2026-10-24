#include <fused_kernel/core/execution_model/memory_operations.h>
#include <fused_kernel/core/utils/utils.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/image_processing/crop.h>
#include <fused_kernel/algorithms/image_processing/resize.h>
#include <fused_kernel/fused_kernel.h>

using namespace fk;

int main() {
    // Define input and output data
    Ptr2D<uchar4> input(1920, 1080);
    std::array<Rect, 5> crops{ Rect(0, 0, 120, 40),
                               Rect(100, 200, 60, 40),
                               Rect(400, 20, 30, 50),
                               Rect(1000, 800, 30, 30),
                               Rect(40, 40, 40, 40)}; 
    Tensor<uchar4> output(64, 64, 5);
    Stream stream;

    // Define and execute operations over the data
    executeOperations<TransformDPP<>>(input, stream,
                                      Crop<>::build(crops),
                                      Resize<InterpolationType::INTER_LINEAR>::build(Size(64,64)),
                                      Mul<float4>::build(make_set<float4>(1.f/255.f)),
                                      Mul<float4>::build(make_set<float4>(0.33f)),
                                      Add<float4>::build(make_set<float4>(0.5f)),
                                      Mul<float4>::build(make_set<float4>(255.f)),
                                      SaturateCast<float4, uchar4>::build(),
                                      TensorWrite<uchar4>::build(output));

    stream.sync();

    return 0;
}
