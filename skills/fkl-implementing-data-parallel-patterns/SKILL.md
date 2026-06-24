---
name: fkl-implementing-data-parallel-patterns
description: Implement a new Data Parallel Pattern (DPP) for the Fused Kernel Library. Use when adding an algorithm to FKL that requires thread coordination, or when debugging a DPP.
---

# Implementing FKL Data Parallel Patterns

## Anatomy of a DPP

A DPP is composed of three parts:

- Executor implementation: as in include/fused_kernel/core/execution_model/executors.h
- For GPU implementations, a kernel function that gets the DPP parameters and internally calls the DPP (for CPU, nothing).
- The DPP implementation.

Every DPP must always have a single thread CPU implementation using the ParArch::CPU and then it can have implementations
for other ParArchs, as in include/fused_kernel/core/execution_model/parallel_architectures.h

Every DPP is a STATELESS struct: static `exec()`. The exec function will get as parameters, DPP details which are external parameters
not directly related to the data being processed, and IOps (Instantiable Operations) which will contain the implementation
of the instructions to be applied over the data being processed, along with dimensions information.

A DPP must not contain the definition of code that modifies the data (except for the case of tensor core code where it's impossible to apply that abstraction).
In order to operate on the data the DPP must use IOps that must have been passed to the DPP as exec function parameters.

Those IOps will be used by each thread to access the device input data, to operate on the data and to write the results back to device memory.
The implementation of the DPP exec function is responsible for deciding which threads will execute which IOps in which order,
as well as applying any thread synchronization or using any shared memory.

Each one of the IOps can contain a single Operation, or a FusedOperation (several consecutive Operations), or a fk::Tuple of IOps,
depending on the structure the DPP requires.

Each DPP defines the number of input ReadOperations and or ReadBackOperations it requires.

Each DPP defines the number of compute Operations (non Read/ReadBack and non Write) it requires.

Each DPP defines the number of output WriteOperations it requires.

Never pass a raw pointer (T*) to device (global) memory as an input or output parameter to exec function.
Device pointers must be passed as part of a Read/ReadBack or Write IOps.

Example TransformDPP which is a special case, because it does not require any specific structure in the IOps passed as parameters.
The IOps in this DPP will be executed one after the other by all the threads that have to participate.
```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

template <typename DPPDetails>
struct TransformDPP<ParArch::GPU_NVIDIA, DPPDetails> {
private:
	using Parent = TransformDPPBase<DPPDetails>;
	using SelfType = TransformDPP<ParArch::GPU_NVIDIA, DPPDetails>;
	using Details = DPPDetails;
public:
    FK_STATIC_STRUCT(TransformDPP, SelfType)  // deletes ctors: pure static
	static constexpr ParArch PAR_ARCH = ParArch::GPU_NVIDIA;
	template <typename FirstIOp>
	FK_HOST_DEVICE_FUSE ActiveThreads getActiveThreads(const Details& details,
														const FirstIOp& iOp) {
		return Parent::getActiveThreads(details, iOp);
	}

	template <typename... IOps>
	FK_DEVICE_FUSE void exec(const Details& details, const IOps&... iOps) {
		const cg::thread_block g = cg::this_thread_block();

		const int x = (g.dim_threads().x * g.group_index().x) + g.thread_index().x;
		const int y = (g.dim_threads().y * g.group_index().y) + g.thread_index().y;
		const int z = g.group_index().z; // So far we only consider the option of using the z dimension to specify n (x*y) thread planes
		const Point thread{ x, y, z };

		const ActiveThreads activeThreads = getActiveThreads(details, get_arg<0>(iOps...));

		if (x < activeThreads.x && y < activeThreads.y) {
			Parent::execute_thread(thread, activeThreads, iOps...);
		}
	}
};
```

A more generic DPP example
```cpp
template <typename DPPDetails, typename ReadIOp, typename ComputeIOp, typename WriteIOp>
struct MyDPP {
private:
    using SelfType = MyDPP<DPPDetails, ReadIOp, ComputeIOp, WriteIOp>;
public:
    FK_STATIC_STRUCT(MyDPP, SelfType)  // deletes ctors: pure static

    FK_HOST_DEVICE_FUSE void exec(const DPPDetails& details,
										const ReadIOp& input,
										const ComputeIOp& compute,
                                        const WriteIOp& output) {
										
		// ReadIOp can be a single Read/ReadBack IOp or a fk::Tuple of IOps.
		// For instance a MMA DPP must have two IOps as input, therefore fk::Tuple<MatrixAIOp, MatrixBIOp>
		// ComputeIOp can be a single IOp or an fk::Tuple of IOps.
		// For instance a ReduceDPP can apply a first operation to the data the first time it reads it, and then a different one
		// when working with already accumulated data. Therefore, for instance fk::Tuple<Pow, Max>. If we want to perform,
		// more than one reduction in parallel, reading the data once, we can pass a nested fk::Tuple: fk::Tuple<fk::Tuple<Pow, Max>, fk::Tuple<Add, Add>>
        
		// In here goes the code that computes the thread index for each thread, decides which threads have to operate on the IOps,
		// synchronizes the threads if necessary, uses shared memory if necessary, iterates etc...
    }
};
```

## Runtime values vs compile-time types (the golden rule)

Anything users may change per call (factors, rects, matrices, sizes) goes
in ParamsType. Anything that changes the generated code (dtype, channel
count, batch size, interpolation mode) is a template parameter. Getting
this wrong either recompiles on every value change or silently bakes
stale values into kernels.

## Vector types

Use the helpers instead of hand-rolled per-channel code:
- `VBase<T>` scalar base; `cn<T>` channel count; `VectorType_t<T, N>`.
- `make_<float3>(x, y, z)` construction; binary operators are already
  overloaded channel-wise for CUDA vector types (vector_utils.h).
- Write exec() once with `if constexpr (cn<I> == ...)` branches only when
  semantics differ per arity (see Equal, TensorSplit).

## Forwarding references in helpers

Never call `fuse_back<ExplicitArgs...>(...)`: explicit template arguments
turn `IOps&&...` into rvalue refs that cannot bind to const lvalues stored
in tuples (issue #245). Let deduction happen, e.g. through a lambda:

```cpp
apply([](const auto&... iOps) { return BackFuser::fuse_back(iOps...); }, tup);
```

## Testing a new DPP

1. Add a utest header under `utests/<area>/`.
2. Register it in the test list before `STOP_ADDING_TESTS`.
3. Build and run: see the fkl-build-and-test skill.

## Checklist before opening a PR

- [ ] FK_STATIC_STRUCT
- [ ] CPU exec() is FK_HOST_FUSE; GPU exec() is FK_DEVICE_FUSE
- [ ] utest instantiating every public alias/build path
- [ ] compiles warning-clean on nvcc AND clang (CUDA 12.x and 13.x)
