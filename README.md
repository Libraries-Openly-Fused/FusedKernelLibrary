# Fused Kernel Library (FKL) LTS-C++17 branch
  
The Fused Kernel Library is a C++17  implementation of a methodology that allows to define a set of  code elements that need to be executed inside a kernel, in the same order that they are expressed. The library currently has CPU, CUDA and HIP (AMD ROCm) backends, and other GPU language implementations are possible.

It automatically implements Vertical and Horizontal fusion, and also implements two new Fusion techniques, Backwards Vertical Fusion (akin to OpenCV Filters, but with an standard generic API), and Divergent Horizontal Fusion.

This is the LTS-C++17 branch of the FKL, which means that the C++ version will be kept to C++17, and the FKL API will be frozen to it's current state, so that old code still works on newer LTS-C++17 versions. The goal of this branch is to have an stable FKL API to facilitate adding new Operations, DataParallelPatterns, Executors and architectures support, without having to refactor any user code or any pre-existing Operation, DataParallelPattern or Executor.

## Reference paper and other publications
This repository provides the official implementation of a kernel fusion methodology for GPU libraries, providing the mechanisms to perform automatic Vertical Fusion, Horizontal Fusion, Backwards Vertical Fusion and Divergent Horizontal Fusion.

A poster was presented at NVIDIA GTC 2025: https://www.nvidia.com/gtc/posters/?search=P73324#/session/1728599648492001N7Sn

A preprint journal paper is available at arxiv, pending approval at an IEEE journal: https://arxiv.org/abs/2508.07071v2

A continuation poster was presented and awarded by NVIDIA members at PUMPS + AI summer school 2025, at Barcelona Supercomputing Center: [LinkedIn post](https://www.linkedin.com/posts/oscar-amoros-huguet_newgpuautomatickernelfusionspeciesposter-activity-7352086240935972867-znFD?utm_source=share&utm_medium=member_desktop&rcm=ACoAAAd9NREBlLso8JKbOumKpptnMzrUH9tmAgg)

## Sample code
Let's see an example where we crop 5 images from a source image, and then apply some changes to those images, before storing them in a Tensor.

You can view and run a similar code in this [FKL Playground](https://colab.research.google.com/drive/1WZd8FcWEKWAuxnJEOTfr0mrWVBtz8bzl?usp=sharing) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WZd8FcWEKWAuxnJEOTfr0mrWVBtz8bzl?usp=sharing)

```C++
// NOTE: FKL version Beta-0.1.12-LTS
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
const Ptr2D<fk::uchar3> inputImage = getGPUSourceImage(stream);

// Define the crops on the source image
constexpr std::array<Rect, BATCH> crops{
    Rect(300, 125, 60, 40),
    Rect(400, 125, 60, 40),
    Rect(530, 270, 130, 140),
    Rect(560, 115, 100, 35),
    Rect(572, 196, 40, 15)
};

// We want a Tensor of contiguous memory for all crops as output
Tensor<fk::uchar3> output(outputSize.width, outputSize.height, BATCH);

// CREATING AND EXECUTING YOUR FUSED CUDA KERNEL
// Create a fused operation that reads the input image,
// crops it, resizes it, and applies arithmetic operations.
// At compile time, the types are used to define the kernel code.
// At runtime, the kernel is executed with the provided parameters.
executeOperations<TransformDPP<>>(stream,
                                  PerThreadRead<ND::_2D, fk::uchar3>::build(inputImage.ptr()),
                                  Crop<>::build(crops),
                                  Resize<InterpolationType::INTER_LINEAR, AspectRatio::PRESERVE_AR>::build(outputSize, backgroundColor),
                                  Mul<fk::float3>::build(make_<fk::float3>(2.f, 2.f, 2.f)),
                                  Sub<fk::float3>::build(make_set<fk::float3>(128.f)),
                                  SaturateCast<fk::float3, fk::uchar3>::build(),
                                  TensorWrite<fk::uchar3>::build(output.ptr()));

stream.sync();

```
Let's see a bit more in detail what is going on in the code.

First of all, take into account that there is no CUDA kernel launch until we call the function executeOperations. Until then, we accumulate and combine information to build the final kernel.

```C++
PerThreadRead<ND::_2D, fk::uchar3>::build(inputImage.ptr()),
```
In this line we are specifying that we want to read a 4K (2D) image, where we will have one CUDA thread per each pixel.

The call to `build(...)` will return a Read<PerThreadRead<ND::_2D, uchar3>> instance, that contains the code as an static member and the parameters stored in the instance.

```C++
Crop<>::build(crops),
```
In the second line, we are changing the threads being used. Since crops it's an std::array of size 5, we now know that we need a 3D set of threadBlocks, where each plane will generate one of the crops, and where width and heigth will be different on each plane. The number of threads on each plane will be the maximum width and the maximum height of all crops, and only the useful threads (width and height of the current plane) will actually read, using the Operation PerThreadRead that we defined previously.

The `build()` method will return an instance of Read<BatchRead<BATCH, Crop<>>> that only knows about the crop sizes, but nothing about the source image. Inside the executeOperations function, we will fuse the PreThreadRead and Crop operations into a Read<BatchRead<BATCH, Crop<Read<PerThreadRead<ND::_2D, uchar3>>>>> fused Operation.

```C++
Resize<InterpolationType::INTER_LINEAR, AspectRatio::PRESERVE_AR>::build(outputSize, backgroundColor),
```
In the third line, we are changing the threads again. This time we are setting a grid made of 60x60x5 active threads (outputSize 5 or BATCH times). We are going to resize each crop from it's original size to 60x60, while preserving the orginal aspect ratio of the crop. The crop will be centered in the 60x60 image, filling the width or the height and having vertical or horizontal bands where all the pixels have the backgroundColor.

The `build()` method will return an instance of ReadBack<Resize<InterpolationType::INTER_LINEAR, AspectRatio::PRESERVE_AR>> that only knows about the target size and the aspect ratio, but nothig about the source image. Inside the executeOperations function, we will fuse the Read<BatchRead<BATCH, Crop<Read<PerThreadRead<ND::_2D, uchar3>>>>> with the ReadBack<Resize<InterpolationType::INTER_LINEAR, AspectRatio::PRESERVE_AR>> into a Read<BatchRead<BATCH, Resize<InterpolationType::INTER_LINEAR, AspectRatio::PRESERVE_AR, ReadBack<Crop<Read<PerThreadRead<ND::_2D, uchar3>>>>>>>.

Each thread will work as if the source image had the size informed by the corresponding Crop operation.
Each thread will ask the Crop operation for the pixels it needs to interpolate the output pixel.

```C++
Mul<fk::float3>::build(mulValue),
Sub<fk::float3>::build(subValue),
Div<fk::float3>::build(divValue),
ColorConversion<COLOR_RGB2BGR, fk::float3, fk::float3>::build(),
```

The following 4 lines will add element wise continuation operations to be applied to the output of the fused operation "PerThreadRead + Crop + Resize" (Read<BatchRead<BATCH, Resize<InterpolationType::INTER_LINEAR, AspectRatio::PRESERVE_AR, ReadBack<Crop<Read<PerThreadRead<ND::_2D, uchar3>>>>>>>).

```C++
TensorWrite<fk::float3>::build(output));
```

The Operation TensorWrite will write the 60x60x5 pixels into a contiguous memory region, without padding on the x axis. The CUDA kernel will receive as parameters the fused operation and the continuation operations, including TensorWrite, as a parameter pack. The kernel is a variadic template kernel function.

DNNs generated with Pytorch usually expect the 3 pixel channels to be split into separated planes, but despite we include the split Operation, this is not the most efficient memory layout for the GPUs, and we wanted to show the creation of most efficient Tensor shape. 

You can view and run a similar code in this [FKL Playground](https://colab.research.google.com/drive/1WZd8FcWEKWAuxnJEOTfr0mrWVBtz8bzl?usp=sharing) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WZd8FcWEKWAuxnJEOTfr0mrWVBtz8bzl?usp=sharing)

`executeOperations<TransformDPP<...>>` checks the chain at compile time (`chain_check.h`), as the executor gets it (the `IncompleteReadBack` operations already fused with what precedes them, an operation fused with `then()` or `fuse()` counting as one): an output type that does not convert to the next input type, a complete Read after the first operation, or batch operations of different `BATCH` stop the build with a `static_assert` that names the operations involved.

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

## Reductions: ReduceDPP

`TransformDPP` gives every thread one element to transform. `ReduceDPP` (`fused_kernel/core/execution_model/reduce_dpp.h`, which the code that uses it includes) is the DPP for the opposite case: from 1 to N reductions of the same data, reading every element once.

```C++
#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/core/execution_model/reduce_dpp.h>
#include <fused_kernel/core/data/ptr_utils.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/cast.h>
#include <fused_kernel/algorithms/basic_ops/logical.h>

using namespace fk;

Stream stream;
const Ptr2D<fk::uchar3> image = getGPUSourceImage(stream);
Ptr1D<fk::float3> results(3); // one element per reduction

// The mean, the maximum and the minimum of every channel of the image, in one kernel
const float pixels = static_cast<float>(image.dims().width * image.dims().height);
const auto read = PerThreadRead<ND::_2D, fk::uchar3>::build(image);
const auto reductions = make_tuple(
    make_tuple(Cast<fk::uchar3, fk::float3>::build(),                           // FirstIOp
               Add<fk::float3, fk::float3, fk::float3, UnaryType>::build(),     // AccumulationIOp
               Div<fk::float3>::build(make_set<fk::float3>(pixels))),           // FinalIOp
    make_tuple(Cast<fk::uchar3, fk::uchar3>::build(),
               Max<fk::uchar3, fk::uchar3, fk::uchar3, UnaryType>::build(),
               Cast<fk::uchar3, fk::float3>::build()),
    make_tuple(Cast<fk::uchar3, fk::uchar3>::build(),
               Min<fk::uchar3, fk::uchar3, fk::uchar3, UnaryType>::build(),
               Cast<fk::uchar3, fk::float3>::build()));
const auto write = PerThreadWrite<ND::_1D, fk::float3>::build(results);

// Once: the workspace, device memory where the thread blocks of a plane leave their values. It starts at zero.
const size_t bytes = ReduceDPP<>::workspaceBytes(read, reductions); // 0 when no workspace is needed
Ptr1D<uchar> workspace(static_cast<uint>(bytes));
if (bytes > 0) {
    setTo(uchar{0}, workspace, stream);
}

// Every time
executeOperations<ReduceDPP<>>(stream, ReduceDPP<>::build_details(workspace), read, reductions, write);

stream.sync();
```

- The read is any complete read: a Read, a ReadBack (`read.then(Crop<>::build(rect))`), a batch of them, or a read with operations fused after it. A 1D source is a plane of height 1.
- Each reduction is a `Tuple<FirstIOp, AccumulationIOp, FinalIOp>` of compute IOps. `FirstIOp` is applied to every element read and gives a value of the accumulator type, which is the `OutputType` of the `AccumulationIOp`. `AccumulationIOp` takes a `Tuple` of two accumulated values and gives one (`Add<A, A, A, UnaryType>`, `Max<A, A, A, UnaryType>`...): it must be associative and commutative, because the elements are not accumulated in the order of the data, and it needs no neutral element. `FinalIOp` is applied once, to the accumulated value. The accumulators of the reductions can have different types; `Cast<T, T>` is the operation that does nothing, for a scalar or a vector `T`.
- The write receives the value of reduction `i` at `Point{i, 0, z}`: a `float3` matrix with three reductions becomes three `float3` values. Every plane `z` of the read (the images of a batch, the planes of a 3D read) is reduced on its own, at its own size, so a batch of `B` images with `N` reductions writes `B` planes of `N` values (`TensorWrite` on a `Tensor<T>(N, 1, B)`, or a batch of writes). A write without `N` elements in x, without rows or with other planes than the read, and a read without elements, throw `std::invalid_argument`; a plane without elements, among planes that have them, is not written. The values of a plane are written when all its elements have been read, so the output of a plane can be the memory that the same plane reads; it must not overlap what another plane reads.
- The DPP does not operate on the values, it only passes them from one IOp to the next, so the elements and the accumulators can be scalars, vectors or any other trivially copyable type (a `Tuple<float, uint>` of a sum and a count, for instance, with the operations that know how to add and divide it: `Cast`, and the reads and writes of memory of the library, only work with scalars and vectors). An accumulator also has to be default constructible, copy constructible and copy assignable, which the executors check, and it is meant to be small: on a GPU, one with arrays of some KB (a histogram) makes the device reserve GBs of memory for the stacks of its threads (`ReduceDPPPolicy::BLOCK_THREADS` has the numbers). `FirstIOp` has to give exactly the accumulator type: a conversion is a `Cast` in the `FirstIOp`.

On a GPU it is one kernel: each thread accumulates elements in registers, the threads of a block accumulate in shared memory, and when a plane has more than one block, the blocks store their values in device memory and the last one to arrive accumulates them. That device memory is the workspace of the example, which the caller owns: the GPU executors allocate no memory, and an execution is one kernel launch, asynchronous, that can be captured in a CUDA graph as any other (the workspace has to live as long as the graph).

- `workspaceBytes` is 0 when one block reduces every plane (planes of up to 4096 elements with the default policy), and on the CPU backends, which need no workspace: the code of the example is the same for every backend. A read that needs none can also be executed without details, `executeOperations<ReduceDPP<>>(stream, read, reductions, write)`. An execution that needs a workspace and has none, or a smaller one, throws `std::invalid_argument`, which says the bytes that it needs.
- The workspace has to be zero the first time it is used: its first bytes are counters, which every execution leaves at zero again. After that any execution of `ReduceDPP`, of any read and reductions that it is large enough for, can use it, as long as two that use it do not run at the same time: the executions of one stream can share one.
- A read has 65535 planes at most on a GPU (the planes are the z of the grid), and the grid covers the largest plane. Each plane has its part of the blocks of the read, counted from the multiprocessors of the device as if every plane had the elements of the largest one (and at least one block per 16384 elements of it), so in a read of many planes of very different sizes the large ones are reduced by fewer blocks than alone, and every small one costs the blocks of the large one: such planes can be better reduced in one execution per size.
- `build_details(workspace, dim3(threads, 1, 1), dim3(blocks, 1, planes))` gives the launch configuration instead of the policy: blocks of 1 to 1024 threads, and the same blocks for every plane, without clusters (any other shape throws `std::invalid_argument`). A grid of more than one block per plane needs a workspace of `workspaceBytes(read, reductions, grid)` bytes. The accumulated values of a block are in its dynamic shared memory, which can go above 48 KB where the device allows it.

On the NVIDIA GPUs with thread block clusters (compute capability 9.0 and newer) the blocks of a cluster read each other's shared memory, so a cluster reduces a plane without device memory: an execution without a workspace reduces there planes of up to 8 blocks of 4096 elements (32768), each with one cluster. With a workspace no cluster is launched, on any GPU, because clusters measured no faster on an RTX 5050: one cluster of 8 blocks took the same time as the same 8 blocks synchronized through the workspace (10.3 us both, for 16K elements, in a build for that architecture alone; 4% to 5% longer in a build that also has an architecture without clusters, where the launch of a cluster asks the driver which image runs), a plane reduced alone gets more blocks than 8 with the workspace, which are faster (the cluster took 4% to 15% longer in that build, for 8K to 32K elements of a uchar matrix and of a resize in float and in double), and the blocks of larger planes in clusters of 8 took from 1.3 to 3.4 times longer than out of clusters (49K to 33M elements), because the device runs few clusters at a time. `ReduceDPPPolicy::CLUSTERS` is that choice (`ReduceDPPClusters::AUTO`; `ALWAYS` uses clusters for every plane of more than one block, with or without a workspace, and `NEVER` for none).

Both versions are in the same kernel under the same `ParArch::GPU_NVIDIA`, and the one of each compute capability is chosen at compile time: `__CUDA_ARCH__` selects the code of each architecture of the build when its image of the kernel is compiled, and `__CUDA_ARCH_LIST__` tells the host code which architectures those are, so a build without architectures with clusters has no host code for them. Which of the images of a build a device runs is only known at run time: in a build with architectures of both kinds, the executions that would launch clusters ask the driver for it, and in every build with architectures with clusters they ask it for the largest cluster that the device accepts. With the default policy, an execution with a workspace only asks the driver for the multiprocessors of the device, once per device: its grid depends on them, and so does `workspaceBytes`, which is the workspace that the current device needs.

`ParArch::CPU` and `ParArch::CPU_MULTITHREAD` split every plane in chunks of consecutive elements, the same ones in both, and give the same results: bit for bit with integers and with plain additions of floats, and up to the last bits where a multiplication of the `FirstIOp` feeds the accumulation, because the compiler can contract `a * b + c` differently in each of them. `ParArch::GPU_AMD` is the GPU version without clusters. The second template parameter of `ReduceDPP` (by default `ReduceDPPPolicy`, and `ReduceDPPPolicyAMD` for `ParArch::GPU_AMD`) has the sizes of the blocks, of the clusters and of the chunks, and what was measured to choose them: the blocks are 4 per multiprocessor of the device for all the planes of a read together, with at least one per 16384 elements of the largest plane and at most one per 512 (`BLOCKS_PER_MULTIPROCESSOR`, `LARGE_BLOCK_ELEMENTS` and `BLOCK_ELEMENTS`), measured on an RTX 4090 and an RTX 5050: within 4.0% and 3.8% of the best grid measured for each of 31 cases, against 12.2% and 5.1% for the previous rule (geometric means). A policy for one device, or a launch configuration, can do better. `ReduceDPPPolicyAMD` is `ReduceDPPPolicy` with blocks of 128 threads in place of 256, measured on an RX 9060: within 4.5% of the best policy measured for each case, against 10.6% with 256 threads (geometric means), and 16% to 17% faster for 16M elements and more; and with 12 blocks per multiprocessor in place of 4 (HIP counts the WGPs: 14 on the RX 9060), within 2.8% of the best grid of the same 31 cases there, against 6.6% with 4 and 4.8% for the previous rule; a policy derived from `ReduceDPPPolicy` keeps its 256 threads on `GPU_AMD`. A reduction whose accumulation is not exact, like a sum of floats, gives results that depend on that split, and so on the backend. On a GPU the blocks of a plane also depend on the other planes of the read (how many they are, and the largest one), so such a result can differ in its last bits between an image reduced alone and the same image in a batch; the threads of a block depend on the bytes of all the accumulators of the execution (fewer than `BLOCK_THREADS` when they are more than 32768 / `BLOCK_THREADS` bytes: 128 with `ReduceDPPPolicy`, 256 with `ReduceDPPPolicyAMD`), so it can also differ when reductions are added to an execution, and, where there are clusters, between an execution with a workspace and one without; the CPU backends split every plane by its own size only.

## Backends and building

Each test and utest is built once per enabled backend: `_cpp` (CPU, `ParArch::CPU`), `_cu` (CUDA with nvcc, `ParArch::GPU_NVIDIA`) and `_hip` (HIP with amdclang++, `ParArch::GPU_AMD`), except for the test and utest sources marked `__ONLY_CU__`, `__ONLY_CPU__` or `__ONLY_GPU__` (CUDA and HIP). CUDA and HIP are optional: `ENABLE_CUDA` and `ENABLE_HIP` are ON by default and are disabled when nvcc or amdclang++ is not found. Every backend compiles as C++17.

A HIP (ROCm) build, with the default C++ compiler for the CPU targets and the GPU architecture detected by CMake (`-DCMAKE_HIP_ARCHITECTURES=gfx1200` to choose it, or `-DHIP_ARCH=...` on the first configure of a build directory):

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_HIP_COMPILER=/opt/rocm/bin/amdclang++
cmake --build build
cd build && ctest
```

The benchmarks of the paper are in the separate repository [fkl-benchmarks](https://github.com/Libraries-Openly-Fused/fkl-benchmarks), which pins this library as a submodule.

Under HIP, the global names `float3`, `uchar3`... are HIP's own vector types, which are not FKL vector types. Use the FKL vector types as `fk::float3`, `fk::uchar3`... They exist in every backend: with nvcc they are the CUDA vector types and on the CPU the same types as the global names, so code written with the `fk::` names compiles for all of them.

### Multithreaded CPU backend

`ParArch::CPU_MULTITHREAD` executes `TransformDPP` on a thread pool, one chunk per output row, with the same per thread code as `ParArch::CPU`, so both backends give the same output: `utest_cpu_multithread_bit_identity` checks that the bytes are the same on integer, float, thread fusion, batch and NV12 chains. For float chains that is a property of the build, not a guarantee of the language: a compiler may contract `a * b + c` into a fused multiply-add differently where the shared code is inlined. The pool is the `thread-pool` git submodule: clone with `git clone --recursive`, or run `git submodule update --init` in an existing clone. `ENABLE_CPU_MULTITHREAD` is ON when the submodule is checked out (and `-DENABLE_CPU_MULTITHREAD=ON` fails if it is not), and defines `FK_CPU_MULTITHREAD_ENABLED`.

`Stream_<ParArch::CPU_MULTITHREAD>` behaves like a CUDA stream: `executeOperations` enqueues the execution and returns, the executions of a stream run one after the other in the order they were enqueued, and `sync()` waits for the ones enqueued before it and rethrows the first exception their work threw.

```cpp
#include <fused_kernel/core/execution_model/cpu_multithread.h>

fk::Stream_<fk::ParArch::CPU_MULTITHREAD> stream;
fk::executeOperations<fk::TransformDPP<fk::ParArch::CPU_MULTITHREAD>>(input, output, stream, /* operations */);
stream.sync();
```

As with the host buffers of `cudaMemcpyAsync`, the buffers of an execution must stay valid until `sync()`: the overloads that take `Ptr` or `Tensor` objects keep them alive by themselves, buffers given to `build()` as `RawPtr` are not. `setTo`, `upload` and `download` accept the stream and need `Host` or `HostPinned` memory (they throw for `Device` and `DeviceAndPinned`, the default memory type of CUDA and HIP builds). `DivergentBatchTransformDPP` has no `CPU_MULTITHREAD` (nor `CPU`) executor.

Each test and utest executable is built for a single backend. Do not link translation units that include FKL compiled for different backends (for instance a .cpp compiled for the CPU and a .hip or .cu file) into one program: FKL is header only, and its templates and inline functions would get a different definition in each backend under the same name, of which the linker keeps only one.

## Memory layout

Thread fusion reads and writes several elements per thread with vector types, so `Ptr2D` rows are padded to keep them aligned:
- Device `Ptr2D` memory is allocated with `cudaMallocPitch` (`hipMallocPitch` with HIP).
- Host and pinned `Ptr2D` memory allocated by FKL (`MemType::Host`, `MemType::HostPinned` and the pinned side of `MemType::DeviceAndPinned`) is 64 byte aligned, and its pitch is rounded up to a multiple of 64 bytes.

A `Ptr2D` is therefore not contiguous in general: index rows with `dims().pitch` (or use `at()` and `PtrAccessor`), and note that `sizeInBytes()` includes the row padding. To get a contiguous host buffer, pass the pitch explicitly when constructing the `Ptr2D`. `Tensor` and other 1D and 3D pointers are not padded.

When a pointer or a pitch is not aligned to the thread fusion vector types, for example a crop or memory provided by the user, the executors use the non thread fused path instead.

## Closed source friendly

A company that has it's own CUDA kernels, and wants to start fusing them along with operations present in this library, can do so by shaping their kernels into a conformant FusedKernel Operation, that can be passed as a template parameter of one of the FKL InstantiableOperation structs.

With this strategy, they don't need to share any of their code. They just need to make their kernels fusionable.
