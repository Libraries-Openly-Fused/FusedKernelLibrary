# Fused Kernel Library (FKL)
  
The Fused Kernel Library is a C++17  implementation of a methodology that allows to define a set of  code elements that need to be executed inside a kernel, in the same order that they are expressed. The library currently has CPU and CUDA backends, but other GPU language implemenations (such as HIP)  are possible.

It automatically implements Vertical and Horizontal fusion, and also implements two new Fusion techniques, Backwards Vertical Fusion (akin to OpenCV Filters, but with an standard generic API), and Divergent Horizontal Fusion.

## Reference paper and other publications
This repository provides the official implementation of a kernel fusion methodology for GPU libraries, providing the mechanisms to perform automatic Vertical Fusion, Horizontal Fusion, Backwards Vertical Fusion and Divergent Horizontal Fusion.

A poster was presented at NVIDIA GTC 2025: https://www.nvidia.com/gtc/posters/?search=P73324#/session/1728599648492001N7Sn

A preprint journal paper is available at arxiv, pending approval at an IEEE journal: https://arxiv.org/abs/2508.07071v2

A continuation poster was presented and awarded by NVIDIA members at PUMPS + AI summer school 2025, at Barcelona Supercomputing Center: [LinkedIn post](https://www.linkedin.com/posts/oscar-amoros-huguet_newgpuautomatickernelfusionspeciesposter-activity-7352086240935972867-znFD?utm_source=share&utm_medium=member_desktop&rcm=ACoAAAd9NREBlLso8JKbOumKpptnMzrUH9tmAgg)

## Sample code
Let's see an example where we crop 5 images from a source image, and then apply some changes to those images, before storing them in a Tensor.

You can view and run a similar code in this [FKL Playground](https://colab.research.google.com/drive/1WZd8FcWEKWAuxnJEOTfr0mrWVBtz8bzl?usp=sharing) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WZd8FcWEKWAuxnJEOTfr0mrWVBtz8bzl?usp=sharing)

```C++
#include <fused_kernel/core/execution_model/memory_operations.cuh>
#include <fused_kernel/algorithms/basic_ops/arithmetic.cuh>
#include <fused_kernel/algorithms/image_processing/crop.cuh>
#include <fused_kernel/algorithms/image_processing/color_conversion.cuh>
#include <fused_kernel/algorithms/image_processing/resize.cuh>
#include <fused_kernel/fused_kernel.cuh>

using namespace fk;

Stream stream;

// We set all outputs to the same size
const Size outputSize(60, 60);
// We perform 5 crops on the image
constexpr int BATCH = 5;

// We have a 4K source image
Ptr2D<uchar3> inputImage(3840, 2160);

// We want a Tensor of contiguous memory for all images
Tensor<float3> output(outputSize.width, outputSize.height, BATCH);

// Crops can be of different sizes
std::array<Rect, BATCH> crops{
    Rect(0, 0, 34, 25),
    Rect(10, 10, 70, 15),
    Rect(20, 20, 60, 59),
    Rect(30, 30, 20, 23),
    Rect(40, 40, 12, 11)
};

initImageValues(inputImage);
const float3 backgroundColor{ 0.f, 0.f, 0.f };

const float3 mulValue = make_set<float3>(1.4f);
const float3 subValue = make_set<float3>(0.5f);
const float3 divValue = make_set<float3>(255.f);

// Create a fused operation that reads the input image,
// crops it, resizes it, and applies arithmetic operations
const auto mySender = PerThreadRead<_2D, uchar3>::build(inputImage)
    .then(Crop<void>::build(crops))
    .then(Resize<INTER_LINEAR, PRESERVE_AR>::build(outputSize, backgroundColor))
    .then(Mul<float3>::build(mulValue))
    .then(Sub<float3>::build(subValue))
    .then(Div<float3>::build(divValue))
    .then(ColorConversion<COLOR_RGB2BGR, float3, float3>::build());

// Define the last operation that will write the results to the output pointer
const auto myReceiver = TensorWrite<float3>::build(output);

// Execute the operations in a single kernel
// At compile time, the types are used to define the kernel code
// At runtime, the kernel is executed with the provided parameters
executeOperations(stream, mySender, myReceiver);

// Use the Tensor for inference
stream.sync()

```
Let's see a bit more in detail what is going on in the code.

First of all, take into account that there is no CUDA kernel launch until we call the function executeOperations. Until then, we accumulate and combine information to build the final kernel.

```C++
const auto mySender = PerThreadRead<_2D, uchar3>::build(inputImage)
```
In this line we are specifying that we want to read a 4K (2D) image, where we will have one CUDA thread per each pixel.

The variable mySender will contain an object representing an Instantiable Operation that combines all the operations specified with the then() method.

```C++
.then(Crop<void>::build(crops))
```
In the second line, we are changing the threads being used. Since crops it's an std::array of size 5, we now know that we need a 3D set of threadBlocks, where each plane will generate one of the crops, and where width and heigth will be different on each plane. The number of threads on each plane will be the maximum width and the maximum height of all crops, and only the useful threads (width and height of the current plane) will actually read, using the Operation PerThreadRead that we defined previously.

The then() method, will return an instance of an Instantiable Operation that performs what we explained with the combination of PerThreadRead + Crop.

```C++
.then(Resize<INTER_LINEAR, PRESERVE_AR>::build(outputSize, backgroundColor))
```
In the third line, we are changing the threads again. This time we are setting a grid made of 60x60x5 active threads (outputSize 5 times). We are going to resize each crop from it's original size to 60x60, while preserving the orginal aspect ratio of the crop. The crop will be centered in the 60x60 image, filling the width or the height and having vertical or horizontal bands where all the pixels have the backgroundColor.

Each thread will work as if the source image had the size informed by the corresponding Crop operation.
Each thread will ask the Crop operation for the pixels it needs to interpolate the output pixel.

The then() method, knows that the Crop operation is a batch operation, so it will assume that we need to apply the non batch resize operation to each one of the cropped images.
The then() method, will return an instance of an Instantiable Operation that performs what we explained with the combination of PerThreadRead + Crop + Resize.

```C++
.then(Mul<float3>::build(mulValue))
.then(Sub<float3>::build(subValue))
.then(Div<float3>::build(divValue))
.then(ColorConversion<COLOR_RGB2BGR, float3, float3>::build());
```

The following 4 lines will add element wise continuation operations to be applied to the output of the combined operation "PerThreadRead + Crop + Resize".
The result will be an Instantiable Operation that combines "PerThreadRead + Crop + Resize + Mul + Sub + Div + ColorConversion".

```C++
const auto myReceiver = TensorWrite<float3>::build(output);
```

The variable myReceiver will contain the last operation of the kernel, that writes the results from the previous Operations to the GPU DRAM.

The Operation TensorWrite will write the 60x60x5 pixels into a contiguous memory region, without padding on the x axis. DNNs generated with Pytorch usually expect the 3 pixel channels to be split into separated planes, but despite we include the split Operation, this is not the most efficient memory layout for the GPUs, and we wanted to show the creation of most efficient Tensor shape. 

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

## MCP Integration for AI Agents

The FusedKernelLibrary now includes Model Context Protocol (MCP) server integration, enabling AI agents to interact with and test the library through a standardized interface.

### AI Agent Capabilities

Through the MCP integration, AI agents can:
- **Discover** library capabilities and documentation
- **Build** the library with configurable options (CPU/CUDA backends)
- **Execute** comprehensive test suites
- **Generate** fusion kernel code examples
- **Validate** code implementations against the library API
- **Analyze** system requirements and CUDA support

### Quick Start with MCP

```bash
# Navigate to MCP integration directory
cd mcp

# Install dependencies
npm install

# Start MCP server
npm start

# Run AI agent tests
npm test
```

The MCP server provides tools for building, testing, and validating FusedKernelLibrary code, making it ideal for AI-assisted development and automated code generation workflows.

For detailed MCP integration documentation, see [mcp/README.md](mcp/README.md).

## Closed source friendly

A company that has it's own CUDA kernels, and wants to start fusing them along with operations present in this library, can do so by shaping their kernels into a conformant FusedKernel Operation, that can be passed as a template parameter of one of the FKL InstantiableOperation structs.

With this strategy, they don't need to share any of their code. They just need to make their kernels fusionable.
