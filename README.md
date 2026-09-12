<div align="center">
  <h1>🚀 Fused Kernel Library (FKL)</h1>
  <p><strong>Redefining Data Parallel portability, performance, and programmability using standard C++20.</strong></p>
  
  [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
  [![C++](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B20)
  [![CUDA](https://img.shields.io/badge/CUDA-Supported-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
  [![ROCm](https://img.shields.io/badge/ROCm-10.0.0_Supported-ED1C24.svg)](https://rocm.amd.com/)
</div>

---

## 💡 What is FKL?

**Built for speed.** The number one purpose of the Fused Kernel Library is to deliver uncompromising, programmer-friendly fast code.

The **Fused Kernel Library (FKL)** is a C++20 header-only framework designed to perform automatic GPU kernel fusion. Instead of relying on custom external compilers or domain-specific languages, FKL leverages modern C++ capabilities to fuse sequential and parallel operations into a single, highly optimized kernel at compile time. 

Currently supporting **CPU**, **CUDA**, and **ROCm** backends (with an architecture designed to easily adopt other GPU languages), FKL transforms memory-bound operations into compute-bound powerhouses by keeping data in registers and eliminating redundant global VRAM reads and writes.

## ✨ Key Features

- **🔥 Generic Vertical Fusion:** Combine operations sequentially without rewriting thread handling. The compiler treats consecutive operations as if written inline, unlocking massive performance optimizations.
- **⏪ Backwards Vertical Fusion:** *Read and compute only what you need.* Apply complex pipelines (e.g., YUV decode $\rightarrow$ crop $\rightarrow$ resize $\rightarrow$ normalize) in a single pass. Data stays in GPU registers until the final write.
- **⚡ Horizontal Fusion:** Process multiple data planes in parallel within the same GPU kernel using `blockIdx.z`, maximizing memory bandwidth for small data payloads.
- **🔀 Divergent Horizontal Fusion:** A novel approach that executes completely different kernels in parallel across the same grid. This allows different Streaming Multiprocessor (SM) components to be saturated simultaneously.
- **🤝 Closed-Source Friendly:** Integrate FKL into proprietary codebases easily. Wrap your custom CUDA or HIP kernels in FKL's `InstantiableOperation` interface to fuse them without exposing your internal source code.
- **🧼 Pure Imperative Ergonomics:** Despite relying on compile-time metaprogramming under the hood, consuming FKL requires **zero template gymnastics**. Pipelines are declared as clean, flat, sequential arguments (`Operation::build(params)`).
- **🔬 Non-Invasive Inline Probing:** Debug intermediate register data effortlessly. Drop a `MidWrite` at any point in the pipeline to capture in-flight values to memory for CPU inspection without breaking fusion.
- **🖥️ Trivial CPU Debugging:** Because operations are hardware-agnostic, you can seamlessly switch to the CPU backend to step through your pipeline logic using standard C++ debuggers (like GDB or Visual Studio) before deploying to the GPU.

## 🧬 Vector Types and HIP/CUDA Interoperability

FKL owns its vector types and operators in the `fk` namespace on every backend. 

⚠️ **Crucial Namespace Guidelines:**
* **Use `fk::float3`, `fk::uchar4`, etc.** outside that namespace, even with `using namespace fk;`. HIP and CUDA declare different types with the same names in the global namespace.
* **Do NOT put `using namespace fk;` at global scope** in CUDA translation units or headers. `nvcc` appends host-registration code that uses unqualified CUDA types (like `uint3`), making names ambiguous. Keep using-directives inside functions or qualify FKL names explicitly.
* **Native HIP/CUDA vector types are NOT FKL vector types.** Convert values explicitly by component at APIs requiring a native vector (e.g., `::float3 native{value.x, value.y, value.z}`). Do not reinterpret vector pointers, as layouts differ (e.g., `fk::double3` has size/alignment 32/16, while AMD Clang's `::double3` is 24/8).

Internal FKL expressions do not need backend-specific casts. **FKL implements almost all standard C++ operators for its vector types**, strictly following scalar C++ type promotion conventions per component. Furthermore, it seamlessly supports **mixing scalar and vector types in binary operators** where it logically makes sense (e.g., multiplying a vector by a scalar).

## 💻 Show Me The Code

Traditional GPU programming requires launching multiple sequential kernels to process an image. With FKL, you define the operations and fuse them lazily. 

Here is an example that extracts 5 crops from an image, resizes them, applies arithmetic changes, and writes them contiguously to a Tensor—**all in one kernel launch.**

```cpp
#include <fused_kernel/core/execution_model/memory_operations.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/image_processing/crop.h>
#include <fused_kernel/algorithms/image_processing/color_conversion.h>
#include <fused_kernel/algorithms/image_processing/resize.h>
#include <fused_kernel/fused_kernel.h>

void preprocess() {
    using namespace fk;

    // 1. Create the fkl CUDA stream
    Stream stream;
    
    // 2. Get the input image
    const Ptr2D<fk::uchar3> inputImage = getGPUSourceImage(stream);
    
    // 3. Define the crops on the source image
    constexpr std::array<Rect, BATCH> crops{
        Rect(300, 125, 60, 40),
        Rect(400, 125, 60, 40),
        Rect(530, 270, 130, 140),
        Rect(560, 115, 100, 35),
        Rect(572, 196, 40, 15)
    };
    
    // 4. We want a Tensor of contiguous memory for all crops as output
    Tensor<fk::uchar3> output(outputSize.width, outputSize.height, BATCH);

    // 5. CREATING AND EXECUTING YOUR FUSED CUDA/HIP KERNEL
    // No execution happens until `executeOperations` is called.
    // Types determine the compile-time kernel code; parameters are passed at runtime.
    executeOperations<TransformDPP<>>(
        stream,
        PerThreadRead<ND::_2D, fk::uchar3>::build(inputImage.ptr()),
        Crop<>::build(crops),
        Resize<InterpolationType::INTER_LINEAR, AspectRatio::PRESERVE_AR>::build(outputSize, backgroundColor),
        Mul<fk::float3>::build(make_<fk::float3>(2.f, 2.f, 2.f)),
        Sub<fk::float3>::build(make_set<fk::float3>(128.f)),
        SaturateCast<fk::float3, fk::uchar3>::build(),
        TensorWrite<fk::uchar3>::build(output.ptr())
    );

    stream.sync();
}
```

### 🔍 Under the Hood: How it Works
1. **`PerThreadRead` & `Crop`**: Defines a 3D threadblock configuration based on the `BATCH` size. FKL automatically calculates that only the useful threads for each crop size will perform reads.
2. **`Resize`**: Re-evaluates the thread grid to `60x60x5`. Each thread asks the `Crop` operation for the specific pixels it needs to interpolate the output.
3. **Continuation Operations (`Mul`, `Sub`, `ColorConversion`)**: Applied element-wise entirely in GPU registers.
4. **`TensorWrite`**: Finally writes the optimized, contiguous data to global memory. 

*Result: A highly efficient, variadic template kernel that compiles down to a single optimized footprint.*

## 🧘 Developer Experience: Common Myths vs. Reality

| Myth | Reality in FKL |
|---|---|
| **"It's a template metaprogramming library, so the user API must be complex."** | **False.** All complexity is hidden behind static `.build()` factories. Chaining operations is as simple as: `executeOperations(stream, Op1::build(...), Op2::build(...));`. You write standard imperative-looking C++. |
| **"Fused kernels are a black box and impossible to debug."** | **False.** You can tap intermediate register values at any point by dropping a `MidWrite` or custom probe operation directly into the operation list without breaking the pipeline. |
| **"GPU kernels are impossible to step through line-by-line."** | **False.** Since operations work across all backends, you can simply compile with the CPU backend to step through your pipeline using standard debuggers (GDB, MSVC) to verify logic before deploying to the GPU. |
| **"Template errors will be unreadable."** | **True and False.** While C++ template errors are notoriously verbose, modern AI tools can instantly translate them and explain exactly where the issue lies. Furthermore, because FKL constructs arguments via modular `.build()` calls, type mismatches are often caught early and localized to your specific operation rather than buried deep inside internal instantiation trees. |

---

## 🚀 Try It Live (Zero Setup)

You can explore how FKL generates highly optimized assembly without installing anything:
- 🛠️ **Compiler Explorer (Godbolt):** [Try FKL v0.1.13-LTS](https://godbolt.org/z/WWjGfj1hY)
- 📓 **Google Colab:** [Run the FKL Playground](#)

---

## 📚 Research & Publications

FKL is built on rigorously tested academic methodology. If you use FKL in your research, please refer to our publications:

- 📄 **Preprint Journal Paper (arXiv):** [Methodology for GPU Kernel Fusion](https://arxiv.org/abs/2508.07071v2) *(Pending IEEE approval)*
- 🖼️ **NVIDIA GTC 2025:** [Poster Presentation](https://www.nvidia.com/gtc/posters/?search=P73324#/session/1728599648492001N7Sn)
- 🏆 **PUMPS + AI 2025:** Award-winning continuation poster at the Barcelona Supercomputing Center ([LinkedIn Post](#)).

---

## 🛠️ Testing & Support

FKL is an Apache 2.0 OpenSource project maintained by contributors in their spare time (and supported by Grup Mediapro S.L.U. during work hours). 

While we provide no formal guarantees or free support, we actively test against:

**CUDA Builds (x86_64 and arm64)**
- **Ubuntu 24.04:** `g++ 13` + CUDA 13.4
- **Windows 11:** Visual Studio 2026 (14.44 toolset) + CUDA 13.0
- **Windows:** Visual Studio 2026 (14.51 toolset) + CUDA 13.4

**ROCm Builds (x86_64 only)**
- **Ubuntu 24.04:** ROCm 10.0.0
- **Windows 11 25H2:** ROCm 10.0.0

**CPU Backend**
- **Linux:** `g++ 13` or `clang++-23`
- **Windows:** Visual Studio 2026 (14.51 toolset)

> **Note on Versioning:** The `main` branch is where active API development happens (minimum C++20, subject to breaking changes). For production stability, please use the `LTS-C++17` branch *(Note: ROCm is not supported on the LTS branch)*.
