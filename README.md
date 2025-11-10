# Fused Kernel Library (FKL) Main branch
  
The Fused Kernel Library is a C++20 implementation of a methodology that allows to define a set of code elements that need to be executed inside a kernel, in the same order that they are expressed. The library currently has CPU and CUDA backends, but other GPU language implemenations (such as HIP) are possible.

It automatically implements Vertical and Horizontal fusion, and also implements two new Fusion techniques, Backwards Vertical Fusion (akin to OpenCV Filters, but with an standard generic API), and Divergent Horizontal Fusion.

This is the Main branch of the repository, where the FKL API can be modified, the minimum C++ version can be increased (currently it is C++20), and in general, we can break retro-compatibility with the goal of enabling more maintainable code, and being able to fuse more types of algorithms.

If you are looking for a more stable branch, where new features can be added, but the FKL API is frozen, to ensure your user code will continue to work with newer versions, check the LTS-C++17 branch.

## Reference paper and other publications
This repository provides the official implementation of a kernel fusion methodology for GPU libraries, providing the mechanisms to perform automatic Vertical Fusion, Horizontal Fusion, Backwards Vertical Fusion and Divergent Horizontal Fusion.

A poster was presented at NVIDIA GTC 2025: https://www.nvidia.com/gtc/posters/?search=P73324#/session/1728599648492001N7Sn

A preprint journal paper is available at arxiv, pending approval at an IEEE journal: https://arxiv.org/abs/2508.07071v2

A continuation poster was presented and awarded by NVIDIA members at PUMPS + AI summer school 2025, at Barcelona Supercomputing Center: [LinkedIn post](https://www.linkedin.com/posts/oscar-amoros-huguet_newgpuautomatickernelfusionspeciesposter-activity-7352086240935972867-znFD?utm_source=share&utm_medium=member_desktop&rcm=ACoAAAd9NREBlLso8JKbOumKpptnMzrUH9tmAgg)

## Sample code
You can try FKL v0.1.9 at Compiler Explorer: https://godbolt.org

Here is an example code, so you don't have to start from scratch: https://godbolt.org/z/6ncb3YMd8

Let's see an example where we crop 5 images from a source image, and then apply some changes to those images, before storing them in a Tensor.

You can view and run a similar code in this [FKL Playground](https://colab.research.google.com/drive/1WZd8FcWEKWAuxnJEOTfr0mrWVBtz8bzl?usp=sharing) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WZd8FcWEKWAuxnJEOTfr0mrWVBtz8bzl?usp=sharing)

```C++
#include <fused_kernel/core/execution_model/memory_operations.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/image_processing/crop.h>
#include <fused_kernel/algorithms/image_processing/color_conversion.h>
#include <fused_kernel/algorithms/image_processing/resize.h>
#include <fused_kernel/fused_kernel.h>

using namespace fk;

// Create the fkl CUDA stream
Stream stream;

// Get the input image
const Ptr2D<uchar3> inputImage = getGPUSourceImage(stream);

// Define the crops on the source image
constexpr std::array<Rect, BATCH> crops{
    Rect(300, 125, 60, 40),
    Rect(400, 125, 60, 40),
    Rect(530, 270, 130, 140),
    Rect(560, 115, 100, 35),
    Rect(572, 196, 40, 15)
};

// We want a Tensor of contiguous memory for all crops as output
Tensor<uchar3> output(outputSize.width, outputSize.height, BATCH);

// CREATING AND EXECUTING YOUR FUSED CUDA KERNEL
// Create a fused operation that reads the input image,
// crops it, resizes it, and applies arithmetic operations.
// At compile time, the types are used to define the kernel code.
// At runtime, the kernel is executed with the provided parameters.
executeOperations<TransformDPP<>>(stream,
                                  PerThreadRead<ND::_2D, uchar3>::build(inputImage.ptr()),
                                  Crop<>::build(crops),
                                  Resize<InterpolationType::INTER_LINEAR, AspectRatio::PRESERVE_AR>::build(outputSize, backgroundColor),
                                  Mul<float3>::build(make_<float3>(2.f, 2.f, 2.f)),
                                  Sub<float3>::build(make_set<float3>(128.f)),
                                  SaturateCast<float3, uchar3>::build(),
                                  TensorWrite<uchar3>::build(output.ptr()));

stream.sync();

```
Let's see a bit more in detail what is going on in the code.

First of all, take into account that there is no CUDA kernel launch until we call the function executeOperations. Until then, we accumulate and combine information to build the final kernel.

```C++
PerThreadRead<ND::_2D, uchar3>::build(inputImage.ptr()),
```
In this line we are specifying that we want to read a 4K (2D) image, where we will have one CUDA thread per each pixel.

The call to `build(...)` will return a `Read<PerThreadRead<ND::_2D, uchar3>>` instance, that contains the code as an static member and the parameters stored in the instance.

```C++
Crop<>::build(crops),
```
In the second line, we are changing the threads being used. Since crops it's an `std::array` of size 5, we now know that we need a 3D set of threadBlocks, where each plane will generate one of the crops, and where width and heigth will be different on each plane. The number of threads on each plane will be the maximum width and the maximum height of all crops, and only the useful threads (width and height of the current plane) will actually read, using the Operation PerThreadRead that we defined previously.

The `build()` method will return an instance of `Read<BatchRead<BATCH, Crop<>>>` that only knows about the crop sizes, but nothing about the source image. Inside the executeOperations function, we will fuse the PreThreadRead and Crop operations into a `Read<BatchRead<BATCH, Crop<Read<PerThreadRead<ND::_2D, uchar3>>>>>` fused Operation.

```C++
Resize<InterpolationType::INTER_LINEAR, AspectRatio::PRESERVE_AR>::build(outputSize, backgroundColor),
```
In the third line, we are changing the threads again. This time we are setting a grid made of 60x60x5 active threads (outputSize 5 or BATCH times). We are going to resize each crop from it's original size to 60x60, while preserving the orginal aspect ratio of the crop. The crop will be centered in the 60x60 image, filling the width or the height and having vertical or horizontal bands where all the pixels have the backgroundColor.

The `build()` method will return an instance of `ReadBack<Resize<InterpolationType::INTER_LINEAR, AspectRatio::PRESERVE_AR>>` that only knows about the target size and the aspect ratio, but nothig about the source image. Inside the executeOperations function, we will fuse the `Read<BatchRead<BATCH, Crop<Read<PerThreadRead<ND::_2D, uchar3>>>>>` with the `ReadBack<Resize<InterpolationType::INTER_LINEAR, AspectRatio::PRESERVE_AR>>` into a `Read<BatchRead<BATCH, Resize<InterpolationType::INTER_LINEAR, AspectRatio::PRESERVE_AR, ReadBack<Crop<Read<PerThreadRead<ND::_2D, uchar3>>>>>>>`.

Each thread will work as if the source image had the size informed by the corresponding Crop operation.
Each thread will ask the Crop operation for the pixels it needs to interpolate the output pixel.

```C++
Mul<float3>::build(mulValue),
Sub<float3>::build(subValue),
Div<float3>::build(divValue),
ColorConversion<COLOR_RGB2BGR, float3, float3>::build(),
```

The following 4 lines will add element wise continuation operations to be applied to the output of the fused operation "PerThreadRead + Crop + Resize" (`Read<BatchRead<BATCH, Resize<InterpolationType::INTER_LINEAR, AspectRatio::PRESERVE_AR, ReadBack<Crop<Read<PerThreadRead<ND::_2D, uchar3>>>>>>>`).

```C++
TensorWrite<float3>::build(output));
```

The Operation TensorWrite will write the 60x60x5 pixels into a contiguous memory region, without padding on the x axis. The CUDA kernel will receive as parameters the fused operation and the continuation operations, including TensorWrite, as a parameter pack. The kernel is a variadic template kernel function.

DNNs generated with Pytorch usually expect the 3 pixel channels to be split into separated planes, but despite we include the split Operation, this is not the most efficient memory layout for the GPUs, and we wanted to show the creation of most efficient Tensor shape. 

You can view and run a similar code in this [FKL Playground](https://colab.research.google.com/drive/1WZd8FcWEKWAuxnJEOTfr0mrWVBtz8bzl?usp=sharing) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WZd8FcWEKWAuxnJEOTfr0mrWVBtz8bzl?usp=sharing)

## Fusion and inclusion

The way the FusedKernel library (FKL) is implemented, allows not only to use the already implemented Operations and data types like Ptr2D or Tensor, but also the fusion can be performend using any code that conforms to the FusedKernel interface (the InstantiableOperation structs and the OperationTypes). The operations in FKL can use any data type that the user wants to use, basic types, structs, tuples (we implemented fk::Tuple to be used on GPU code, along with fk::apply and other utilites).

This was done in purpose to make it easier to join efforts with other libraries that already exist and are also OpenSource and want to take advantage of the FusedKernelLibrary methodology.

Additionally, this flexibility helps companies with closed-source frameworks to apply the FKL methodology and even make use of the library, combining it with their own code and data types.
### Horizontal Fusion

This fusion technique is widely known and used. It is based on the idea of processing several data planes in parallel, with the same CUDA kernel. For that, we use the blockIdx.z, to distinguish between thread planes and data planes.

This is usually very beneficial when each plane is very small, and the resulting 2D Grid is not taking advantage of the GPU memory bandwidth.

We also support what we call Divergent Horizontal Fusion. This variant allows to execute different kernels that can be executed in parallel. Each "kernel" can use one or more z planes of the grid, so each kernel can do Horizontal Fusion. This technique allows to exploit the possibility of using diferent components in the SM's in parallel, improving the overall performance.

### Vertical Fusion (Generic Vertical Fusion)

Vertical Fusion is usually limited to having a kernel that is configurable up to a certain level, or there is a list of pre-compiled fused kernels to choose from. In our case, we are abstrating away the thread behavior from the actual functionality, and allowing to fuse almost every kernel possible, without having to rewrite neither the thread handling, nor the functionality. You only have to combine code in the different ways that the code can be combined. We call this Generic Vertical Fusion.

For Memory Bound kernels, vertical fusion is bringing most of the performance improvements possible, since adding more functions to the kernel will not increase the execution time, up to a limit where the kernel becomes Compute Bound.

Not only that, but thanks to the way the code is written, the nvcc compiler will treat the consecutive operations as if you where writting the code in one line, adding all sorts of optimizations. This can be seen by compiling the code in Release mode, or in Debug mode. The performance difference is abismal.

### Backwards Vertical Fusion (read and compute, only what you need)

This is an optimization that can already be used with the current code, but will be refined and further increase the use cases when adding more Operations.

The idea, is aplicable for situations where you have a big plane, from which you will only use a subset of the data. If you need to transform that plane into something different before before operating on the subset, you can use Backwards Vertical Fusion in order to have a single kernel, that will read only what it needs, and apply to it all the operations needed.

For example, let's assume that you receive an image in YUV420_NV12 format, and you need to crop a region of this image, then convert the pixels to RGB, then resize the crop, normalize it to floating point values from 0 to 1, and store the resulting image in RGB planar format. Usually, this would lead to many kernels, one after the other. The first kernel that converts to RGB, will convert the full image, and write the result to memory. Instead, with the Fused Kernel library, it is possible to create a Fused Kernel in a few lines, that will only read the YUV data for the pixels required by the interpolation process, in the resize of the crop. All the steps will be performed using GPU registers, until the last step where we will finally write into GPU ram memory.

This is way faster to program than the conventional way of programming CUDA.

### Divergent Horizontal Fusion

This novel type of Horizontal Fusion, allows to Horizontally Fuse kernels that are completelly different, and read the same or different data, but write the results in different memory regions.

This has been tested before, by creating special compilers that generate the assembly code, and the performance benefits have been already reported. The novelty in our approach is that we do not require a different compiler. We do this by leveraging the C++17 capabilities found in nvcc.

## Closed source friendly

A company that has it's own CUDA kernels, and wants to start fusing them along with operations present in this library, can do so by shaping their kernels into a conformant FusedKernel Operation, that can be passed as a template parameter of one of the FKL InstantiableOperation structs.

With this strategy, they don't need to share any of their code. They just need to make their kernels fusionable.
