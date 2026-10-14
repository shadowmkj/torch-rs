# Mini-Torch: Rust Machine Learning Library Blueprint & Implementation Plan

A comprehensive architectural blueprint, mathematical and computational theory guide, crate catalog, and phased implementation plan for building an educational, high-performance machine learning library from scratch in Rust supporting CPU, Apple Silicon (Metal), and NVIDIA (CUDA) backends.

---

## 1. Project Overview & Architecture Blueprint

### 1.1 Architectural Philosophy
Building a tensor and deep learning library in Rust requires reconciling Rust's strict compile-time borrow checker with the dynamic, cyclic, and shared-memory characteristics of computational graphs:
- **Separation of Metadata and Storage**: A `Tensor` should be an immutable, lightweight handle pointing to shared buffer storage. Zero-copy operations (transposition, reshaping, slicing) alter index arithmetic (shape, strides, offsets) without allocating or copying underlying memory.
- **Dynamic Interior Mutability for Gradient Propagation**: Reverse-mode automatic differentiation accumulates gradients along multiple paths into leaf tensors. Thread-safe interior mutability (`Arc<RwLock<...>>`) allows in-place gradient accumulation without requiring exclusive mutable references to the user-facing tensor handle.
- **Decoupled Backend Kernel Dispatch**: Compute kernels are separated from high-level tensor abstractions. The tensor inspects its target `Device` and dispatches execution to CPU, Metal, or CUDA executors.
- **Cross-Platform Dual-GPU Strategy**: Native Metal support on macOS (via MSL compute pipelines and Apple Metal Performance Shaders) combined with feature-gated CUDA support (via CUDA Driver/cuBLAS) ensures native high-performance execution on the host development machine (Apple Silicon MacBook) while maintaining production compatibility with NVIDIA hardware.

### 1.2 System Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│                        High-Level Framework API                        │
│             nn::Module, nn::Linear, optim::SGD, optim::Adam            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                             Tensor Engine                              │
│       Handle (Arc<TensorImpl>), Shape, Strides, Offset, Device         │
└─────────────────┬───────────────────────────────────┬──────────────────┘
                  │                                   │
┌─────────────────▼─────────────────┐   ┌─────────────▼──────────────────┐
│      Strided Layout & Math        │   │        Autograd Engine         │
│  - N-D to 1-D Index Projection    │   │  - Computation DAG Builder     │
│  - Zero-Copy Transpose & Permute  │   │  - Vector-Jacobian Ops (VJPs)  │
│  - Broadcast Stride Synthesis     │   │  - Topological DFS Traversal   │
│  - Contiguity Compaction Checker  │   │  - In-Place Grad Accumulator   │
└─────────────────┬─────────────────┘   └─────────────┬──────────────────┘
                  │                                   │
┌─────────────────▼───────────────────────────────────▼──────────────────┐
│                        Device Dispatch Layer                           │
│           Routes operations based on (self.device, rhs.device)         │
└─────────┬─────────────────────────┬──────────────────────────┬─────────┘
          │                         │                          │
┌─────────▼─────────┐     ┌─────────▼─────────┐      ┌─────────▼─────────┐
│    CPU Backend    │     │   Metal Backend   │      │   CUDA Backend    │
│  - Contiguous RAM │     │  - Metal Buffers  │      │  - CudaSlice VRAM │
│  - Rayon Threads  │     │  - MSL Shaders    │      │  - PTX / NVRTC    │
│  - BLAS / Matmul  │     │  - MPS Matrix Ops │      │  - cuBLAS sgemm   │
└───────────────────┘     └───────────────────┘      └───────────────────┘
```

---

## 2. Recommended Crates & Tooling Catalog

### 2.1 Core Utilities & Concurrency
- **`thiserror`**:
  - *Role*: Ergonomic, zero-cost error definition.
  - *Usage*: Defines structured domain errors (`TensorError::ShapeMismatch`, `TensorError::DeviceMismatch`, `TensorError::NonContiguous`, `TensorError::MetalCompilationFailed`, `TensorError::CudaError`).
- **Standard Library Primitives (`std::sync::{Arc, RwLock}`)**:
  - *Role*: Shared ownership and safe interior mutability.
  - *Usage*: `Arc` shares physical buffer allocations and computation graph nodes across multiple view tensors. `RwLock` provides concurrent read-access during forward passes and mutable write-access during gradient accumulation.
- **`rand` & `rand_distr`**:
  - *Role*: High-quality pseudo-random number generation and statistical distributions.
  - *Usage*: Initializing weights via Uniform, Gaussian/Normal, Xavier/Glorot, and Kaiming/He distributions.

### 2.2 CPU Acceleration
- **`rayon`**:
  - *Role*: Data-parallelism and work-stealing thread pool.
  - *Usage*: Parallelizing CPU element-wise operations (addition, multiplication, activations) and reduction operations across chunks of strided or contiguous arrays.
- **`matrixmultiply`**:
  - *Role*: Pure Rust, cache-oblivious, high-performance General Matrix Multiply (GEMM) for f32/f64.
  - *Usage*: High-speed CPU matrix multiplication competitive with native BLAS without requiring external Fortran/C system toolchains.

### 2.3 Metal Backend (macOS Apple Silicon)
- **`metal`**:
  - *Role*: Safe idiomatic Rust bindings to Apple's Metal framework (`MTLDevice`, `MTLCommandQueue`, `MTLBuffer`, `MTLComputePipelineState`).
  - *Usage*: Allocating shared unified memory buffers, loading and compiling Metal Shading Language (MSL) source strings at runtime, encoding compute commands, and dispatching threadgrids.
- **`objc`**:
  - *Role*: Objective-C runtime interaction.
  - *Usage*: Interfacing with Apple's `MetalPerformanceShaders` (MPS) framework for hardware-accelerated matrix multiplication (`MPSMatrixMultiplication`), which leverages the Apple Silicon AMX (Apple Matrix Coprocessor) hardware units.

### 2.4 CUDA Backend (NVIDIA Acceleration)
- **`cudarc`**:
  - *Role*: Safe, modern Rust wrapper around the NVIDIA CUDA Driver API, runtime kernel compilation (NVRTC), and cuBLAS.
  - *Usage*: Allocating GPU VRAM (`CudaSlice<f32>`), managing streams, compiling raw CUDA C kernels to PTX dynamically, and executing high-throughput cuBLAS `sgemm` routines.
  - *Gating*: Configured as an optional dependency gated behind `[features] cuda = ["dep:cudarc"]`.

### 2.5 Cargo Configuration Strategy
- **Platform-Targeted Dependencies**: Metal and Objective-C dependencies are scoped strictly to `cfg(target_os = "macos")` so non-Mac platforms can build without error.
- **Feature-Gated GPU Backends**: CUDA is kept behind an opt-in feature flag so developers on Apple Silicon can compile and run tests instantly without installing the CUDA toolkit.

---

## 3. Theoretical Foundations

### 3.1 Strided N-Dimensional Tensor Memory Layout

#### 3.1.1 Mapping Multi-Dimensional Coordinates to 1D Memory
Computer physical memory is a flat, contiguous 1D array of bytes. A tensor represents an $n$-dimensional mathematical object defined by:
- **Shape Vector**: $S = (d_0, d_1, \dots, d_{n-1})$, where $d_i$ represents the size of dimension $i$.
- **Stride Vector**: $T = (s_0, s_1, \dots, s_{n-1})$, where $s_i$ indicates the number of scalar elements in storage that must be skipped to advance by 1 in dimension $i$.
- **Storage Offset**: An integer offset $\delta \ge 0$ denoting the starting index in the underlying storage buffer.

For any coordinate tuple $(i_0, i_1, \dots, i_{n-1})$ where $0 \le i_k < d_k$, the exact physical index in the 1D storage array is given by the linear affine transformation:
$$\text{LinearIndex}(i_0, i_1, \dots, i_{n-1}) = \delta + \sum_{k=0}^{n-1} i_k \cdot s_k$$

#### 3.1.2 Row-Major (C-Contiguous) Canonical Strides
In standard row-major format, the last dimension changes fastest in memory:
$$s_{n-1} = 1$$
$$s_k = s_{k+1} \cdot d_{k+1} = \prod_{j=k+1}^{n-1} d_j \quad \text{for } k \in [0, n-2]$$
The total number of logical elements is:
$$N = \prod_{k=0}^{n-1} d_k$$

#### 3.1.3 Zero-Copy Views
By manipulating only shape, strides, and offsets, tensor operations can be performed in $O(1)$ time with zero memory allocation:
- **Transposition / Dimension Permutation**: Swapping two axes $p$ and $q$ simply swaps the corresponding entries in the shape vector ($d_p \leftrightarrow d_q$) and the stride vector ($s_p \leftrightarrow s_q$). The underlying storage remains completely unchanged.
- **Narrowing / Slicing**: Slicing along dimension $k$ from index $a$ to $b$ sets the new shape $d_k' = b - a$ and advances the storage offset:
  $$\delta' = \delta + a \cdot s_k$$
- **Broadcasting Arithmetic**: When an operation involves tensors of differing shapes (e.g., adding a vector of shape $[C]$ to a matrix of shape $[B, C]$):
  1. Shapes are right-aligned. Leading dimensions of size 1 are prepended to the smaller tensor.
  2. For every dimension where the shape is 1 while the other tensor has size $D > 1$, the dimension is expanded to size $D$ and its stride is set to **$0$**.
  3. A stride of $0$ means incrementing the coordinate produces an offset of $i \cdot 0 = 0$, causing all reads along that axis to access the exact same memory location repeatedly without duplicating data.

#### 3.1.4 Contiguity and Physical Compaction
A tensor is contiguous if and only if its elements are laid out sequentially in memory without gaps or inversions:
$$s_k = \prod_{j=k+1}^{n-1} d_j$$
Non-contiguous tensors occur after transpositions, non-uniform slices, or broadcasting. Hardware acceleration libraries (cuBLAS, MPS, custom compute shaders) require contiguous memory blocks. A `contiguous()` compaction step allocates a new buffer of length $N$ and iterates through all logical coordinates in canonical order, copying values from the strided source into the contiguous target.

---

### 3.2 Reverse-Mode Automatic Differentiation (Autograd)

#### 3.2.1 Reverse-Mode vs. Forward-Mode
For a scalar loss function $\mathcal{L} : \mathbb{R}^n \to \mathbb{R}$ computed over millions of model parameters, reverse-mode automatic differentiation computes the full gradient vector $\nabla_{\theta} \mathcal{L}$ in a single backward pass with a computational cost proportional to the forward pass. Forward-mode would require $n$ distinct passes (one per parameter), making it intractable for neural networks.

#### 3.2.2 Vector-Jacobian Products (VJPs)
Rather than instantiating the massive Jacobian matrix $J \in \mathbb{R}^{m \times n}$ of an intermediate transformation $y = f(x)$, reverse-mode AD computes the vector-Jacobian product:
$$v^T J = \left(\frac{\partial \mathcal{L}}{\partial y}\right)^T \frac{\partial y}{\partial x}$$
where $v = \frac{\partial \mathcal{L}}{\partial y}$ is the upstream gradient passed backwards from the loss.

Mathematical VJP rules for primary primitive operations:
- **Element-wise Addition**: $z = x + y$
  $$\frac{\partial \mathcal{L}}{\partial x} = v, \quad \frac{\partial \mathcal{L}}{\partial y} = v$$
  *(If broadcasting occurred in the forward pass, gradients are summed along the expanded axes).*
- **Element-wise Multiplication**: $z = x \odot y$
  $$\frac{\partial \mathcal{L}}{\partial x} = v \odot y, \quad \frac{\partial \mathcal{L}}{\partial y} = v \odot x$$
- **Matrix Multiplication**: $Z = X Y$, where $X \in \mathbb{R}^{M \times K}, Y \in \mathbb{R}^{K \times N}, Z \in \mathbb{R}^{M \times N}$
  $$\frac{\partial \mathcal{L}}{\partial X} = v Y^T, \quad \frac{\partial \mathcal{L}}{\partial Y} = X^T v$$
- **Rectified Linear Unit (ReLU)**: $z = \max(0, x)$
  $$\frac{\partial \mathcal{L}}{\partial x} = v \odot \mathbb{I}(x > 0)$$
- **Sum Reduction**: $z = \sum x$
  $$\frac{\partial \mathcal{L}}{\partial x} = \text{broadcast}(v, \text{shape}(x))$$
- **Mean Reduction**: $z = \frac{1}{N} \sum x$
  $$\frac{\partial \mathcal{L}}{\partial x} = \frac{1}{N} \text{broadcast}(v, \text{shape}(x))$$

#### 3.2.3 Dynamic Computation Graph & Topological Ordering
Every tensor operation whose inputs require gradients dynamically registers a backward operation node (`BackwardOp`). The backward operation stores references to its direct input tensors and captures the necessary forward activations.

To compute gradients during `loss.backward()`:
1. **Topological Sort**: Traverses the DAG starting at the loss node using Depth-First Search (DFS) or Kahn's algorithm. Because a tensor can be used in multiple downstream operations, its gradient cannot be finalized until all downstream consumers have contributed their gradients.
2. **Seed Gradient**: The scalar loss node is initialized with an upstream gradient of $1.0$ (matching its shape).
3. **Reverse Sweep**: Operations are executed in reverse topological order. Each operation calls its backward function using its accumulated output gradient, producing gradients for each input tensor.
4. **Gradient Accumulation**: Input tensors accumulate these gradients into their interior `grad` buffer (`grad += incoming_grad`).

#### 3.2.4 Avoiding Memory Leaks & Reference Cycles in Rust
If a backward operator struct holds a strong `Arc` reference to its own output tensor (or if the graph creates cycles), the reference count will never drop to zero, causing silent, catastrophic VRAM and RAM memory leaks on every iteration.
- *Strict Invariant*: A `BackwardOp` must **only** store references to its parent input nodes, never to its output node.
- *Graph Detachment*: Graph traversal structures and temporary gradient caches must be dropped immediately upon completion of the backward pass.

---

### 3.3 Hardware Acceleration & GPU Architecture

#### 3.3.1 Apple Silicon Unified Memory & Metal Architecture
- **Unified Memory Architecture (UMA)**: The M-series SoC shares a single high-bandwidth physical memory pool between the CPU and the Metal GPU. When buffers are created with `MTLResourceOptions::StorageModeShared`, both the CPU and GPU access the identical physical RAM addresses. Zero data copy over PCIe is required.
- **Execution Hierarchy**:
  - Grid of Threads $\to$ Threadgroups $\to$ SIMD-groups (32 threads running in lockstep).
  - Compute pipelines compile MSL source code into GPU pipelines via runtime driver compilation.
  - Threadgroup memory acts as fast on-chip scratchpad SRAM for tiled matrix operations.
- **Metal Performance Shaders (MPS)**: Apple provides pre-tuned, highly optimized matrix algebra kernels in the MPS framework. Calling `MPSMatrixMultiplication` directly utilizes the specialized Apple Matrix Coprocessor (AMX) instructions, yielding performance orders of magnitude faster than naive custom shaders.

#### 3.3.2 NVIDIA CUDA Architecture
- **Discrete Memory Model**: Host system RAM and device VRAM are physically separated across the PCIe / NVLink bus. Data must be explicitly staged and transferred via Host-to-Device (H2D) and Device-to-Host (D2H) copy routines.
- **Execution Hierarchy**:
  - Grid $\to$ Thread Blocks $\to$ Warps (32 threads).
  - Shared memory within thread blocks enables matrix tiling.
- **cuBLAS & NVRTC**: Standard GEMM operations are offloaded to cuBLAS (`cublasSgemm`), while custom element-wise operations and activation functions can be compiled from C strings at runtime using NVRTC.

---

### 3.4 Neural Network Abstractions & Optimization Theory

#### 3.4.1 Modular Hierarchy
- **`Parameter`**: A semantic wrapper around a `Tensor` indicating that its values represent learnable weights and must have `requires_grad = true`.
- **`Module`**: A composable interface standardizing the forward computation pass and recursive collection of learnable parameters.
- **Common Layers**:
  - *Linear (Fully Connected)*: Computes $y = x W^T + b$.
  - *Activation Functions*: Non-linear mapping (ReLU, Sigmoid, GELU, Tanh).
  - *Loss Functions*: Measures divergence between model predictions and ground-truth targets (e.g., Mean Squared Error for regression, Softmax Cross-Entropy for multi-class classification).

#### 3.4.2 Numerical Stability in Loss Formulations
Directly computing $\text{CrossEntropy}(p, y) = -\sum y_i \log(\text{Softmax}(x_i))$ suffers from catastrophic floating-point underflow and overflow due to exponential explosion in $\exp(x_i)$ and logarithmic collapse in $\log(0)$.
- **Log-Sum-Exp Trick**: The Softmax denominator is stabilized by subtracting the maximum logit $m = \max_k(x_k)$:
  $$\log\left(\sum_k e^{x_k}\right) = m + \log\left(\sum_k e^{x_k - m}\right)$$
- Always compute cross-entropy directly from raw unnormalized logits using the fused Log-Sum-Exp formulation.

#### 3.4.3 Optimization Algorithms
- **Stochastic Gradient Descent (SGD) with Momentum**:
  $$v_t = \beta v_{t-1} + g_t$$
  $$\theta_t = \theta_{t-1} - \alpha v_t$$
  where $\theta$ is the parameter vector, $g_t = \nabla_\theta \mathcal{L}$ is the gradient, $\alpha$ is the learning rate, and $\beta$ is the momentum factor.
- **Adam (Adaptive Moment Estimation)**:
  Maintains running exponential averages of both the gradients ($m_t$) and the uncentered variance ($v_t$):
  $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
  $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
  Bias-correction accounts for initialization at zero:
  $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
  Parameter update:
  $$\theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
- **In-Place Mutation & No-Grad Context**: Parameter updates mutate underlying physical storage buffers in-place without registering new operations into the autograd computation graph. Gradients must be explicitly reset to zero (`zero_grad()`) at each training iteration.

---

## 4. Detailed Implementation Steps & Phased Roadmap

### Phase 1: Environment Setup, Configuration & Foundation

#### Step 1.1: Cargo Workspace & Dependency Configuration
- Set up `Cargo.toml` with edition 2024.
- Configure conditional dependencies:
  - Add `thiserror`, `rand`, `rand_distr`, and `rayon`.
  - Add `matrixmultiply` for high-performance CPU GEMM.
  - Add target-specific dependencies under `cfg(target_os = "macos")` for `metal` and `objc`.
  - Add optional `cudarc` gated under the `cuda` cargo feature flag.
- Define feature flags: `default = []`, `cuda = ["dep:cudarc"]`.

#### Step 1.2: Core Error Architecture
- Implement a comprehensive `TensorError` enum using `thiserror`.
- Define error variants covering:
  - Incompatible shapes for binary operations and matrix multiplication.
  - Dimension out of bounds during slicing, transposing, or reducing.
  - Cross-device operation mismatch (attempting to add a CPU tensor to a Metal tensor).
  - Attempting operations requiring contiguous memory on non-contiguous tensors.
  - Driver and backend initialization failures (Metal device absent, CUDA driver error).

#### Step 1.3: Device & Storage Domain Modeling
- Define the `Device` enum:
  - `Device::Cpu`
  - `Device::Metal`
  - `Device::Cuda(usize)`
- Define the physical `Storage` enum:
  - `Storage::Cpu(Vec<f32>)`
  - `Storage::Metal(metal::Buffer)` (conditionally compiled on macOS)
  - `Storage::Cuda(cudarc::driver::CudaSlice<f32>)` (conditionally compiled under `cuda` feature)
- Implement basic storage helper methods: element count, byte size, and device identification.

---

### Phase 2: Strided Layout Engine & CPU Tensor Core

#### Step 2.1: Strided Indexing Mathematics
- Implement standard row-major stride generation from any arbitrary dimension shape vector.
- Implement dimension-swapping math for transpositions and multi-axis permutations.
- Implement the canonical contiguity validation rule: comparing current strides against the expected standard contiguous strides.
- Implement multi-dimensional to 1D index resolution supporting non-zero offsets and arbitrary strides.

#### Step 2.2: Broadcasting Logic
- Implement shape alignment and compatibility verification according to NumPy/PyTorch broadcasting conventions:
  - Right-align shape vectors.
  - Pad leading dimensions with size 1.
  - Validate that each dimension either matches or has a size of 1.
- Implement stride synthesis for broadcasted shapes: setting the stride of any expanded dimension (size changed from 1 to $D$) to zero.

#### Step 2.3: Tensor Memory Structs
- Construct `TensorImpl`:
  - Storage buffer wrapped in shared access primitives (`Arc<RwLock<Storage>>`).
  - Shape vector (`Vec<usize>`).
  - Stride vector (`Vec<usize>`).
  - Offset scalar (`usize`).
  - Target device tag (`Device`).
  - Gradient tracking flag (`requires_grad: bool`).
  - Gradient accumulation slot (`RwLock<Option<Tensor>>`).
  - Backward operation pointer (`Option<Arc<dyn BackwardOp + Send + Sync>>`).
- Construct `Tensor`:
  - Newtype pattern wrapping `Arc<TensorImpl>`.
  - Implements `Clone` (shallow handle copy, zero data duplication).

#### Step 2.4: Tensor Creation & Zero-Copy View Methods
- Implement constructors:
  - `zeros(shape, device)`
  - `ones(shape, device)`
  - `from_slice(slice, shape, device)`
  - `randn(shape, mean, std, device)`
- Implement zero-copy transformations:
  - `transpose(dim0, dim1)`: Swaps shape and stride entries; returns a new `Tensor` pointing to the exact same storage.
  - `reshape(new_shape)`: If contiguous, computes new strides directly; if non-contiguous, triggers compaction first.
  - `slice(dim, start, end)`: Adjusts offset and dimension size.
  - `broadcast_to(target_shape)`: Returns a new tensor handle with zero-strides on expanded dimensions.

#### Step 2.5: Physical Memory Compaction (`contiguous`)
- Implement the `contiguous()` method:
  - Checks if the tensor already satisfies canonical contiguity (if so, returns `self`).
  - If non-contiguous, allocates a new contiguous storage buffer.
  - Traverses the logical coordinate space of the strided source, reading values from non-contiguous indices and writing them sequentially into the new buffer.

---

### Phase 3: Primitive CPU Compute Kernels

#### Step 3.1: Element-wise Unary & Binary Arithmetic
- Implement CPU kernels for:
  - Addition, Subtraction, Multiplication, Division.
  - Invocations over both contiguous and broadcasted strided buffers.
  - Parallel acceleration using `rayon` chunk iterators for large buffers.
- Implement operator overloading traits (`std::ops::{Add, Sub, Mul, Div}`) for `&Tensor` to allow natural arithmetic expressions without consuming tensor ownership.

#### Step 3.2: Activations & Mathematical Functions
- Implement CPU element-wise activations:
  - `relu`: Computes $\max(0, x)$.
  - `sigmoid`: Computes $\frac{1}{1 + e^{-x}}$.
  - `tanh`: Computes hyperbolic tangent.
  - `exp`, `ln`, `powf`.

#### Step 3.3: Reduction Operations
- Implement multi-dimensional reduction routines:
  - `sum(dim, keepdim)`
  - `mean(dim, keepdim)`
  - `max(dim, keepdim)`
- Support full reduction to scalar tensors when no dimension is specified.

#### Step 3.4: CPU Matrix Multiplication (GEMM)
- Implement 2D matrix multiplication (`matmul`):
  - Validate 2D dimension constraints ($[M, K] \times [K, N] \to [M, N]$).
  - Ensure inputs are contiguous (invoking `.contiguous()` automatically if transposed).
  - Integrate `matrixmultiply` crate to execute tiled, multi-threaded CPU GEMM.
- Implement batched matrix multiplication support for 3D/4D tensors ($[B, M, K] \times [B, K, N] \to [B, M, N]$).

---

### Phase 4: Reverse-Mode Autograd Engine

#### Step 4.1: The `BackwardOp` Trait Definition
- Define the `BackwardOp` trait:
  - Must be thread-safe (`Send + Sync`).
  - Exposes `fn backward(&self, upstream_grad: &Tensor) -> Vec<(Tensor, Tensor)>`.
  - Returns pairs of `(Input Tensor Handle, Calculated Input Gradient)`.

#### Step 4.2: Vector-Jacobian Product Operations
- Implement backward structs for all core operations:
  - `AddBackward`: Routes upstream gradient to both inputs; handles reduction over broadcasted axes.
  - `SubBackward`: Routes positive gradient to LHS and negated gradient to RHS.
  - `MulBackward`: Multiplies upstream gradient by opposite input tensors ($v \odot y$ and $v \odot x$).
  - `MatMulBackward`: Computes matrix gradients via transposed products ($v Y^T$ and $X^T v$).
  - `ReluBackward`: Masks upstream gradient where forward activation was $\le 0$.
  - `SumBackward`: Broadcasts upstream scalar gradient back to original input shape.
- Verify that backward structs hold references **only to input tensors**, strictly avoiding references to output tensors.

#### Step 4.3: Computation Graph Registration
- Hook each forward operation into autograd:
  - Check if any input has `requires_grad == true`.
  - If true, construct the appropriate `BackwardOp` capturing input clones.
  - Attach the `BackwardOp` to the newly created output tensor and set its `requires_grad = true`.

#### Step 4.4: Topological Sorting & Reverse Traversal
- Implement depth-first search (DFS) post-order traversal to sort the graph from the target loss node.
- In `Tensor::backward()`:
  - Enforce that the initiating tensor is a scalar (shape `[]` or `[1]`).
  - Collect topological order into a linear sequence.
  - Seed the loss gradient with `1.0`.
  - Iterate in reverse topological order.
  - For each node with an attached `grad_fn`, evaluate the backward pass.
  - Accumulate returned input gradients into each target tensor's `grad` field using thread-safe write locks.

#### Step 4.5: Gradient Verification via Finite Differences
- Implement a test utility executing numerical gradient checking:
  $$\frac{\partial \mathcal{L}}{\partial x_i} \approx \frac{\mathcal{L}(x_i + \epsilon) - \mathcal{L}(x_i - \epsilon)}{2\epsilon}$$
- Compare analytical autograd gradients against finite difference approximations with a tolerance of $\epsilon = 10^{-4}$.

---

### Phase 5: Metal GPU Backend (macOS Apple Silicon)

#### Step 5.1: Metal Context & Resource Management
- Conditionally compile under `#[cfg(target_os = "macos")]`.
- Create a centralized, lazily initialized `MetalContext`:
  - Queries `MTLDevice::system_default()`.
  - Instantiates the primary `MTLCommandQueue`.
  - Manages a pipeline state cache for compiled MSL kernels.
- Implement shared-memory buffer allocation using `MTLResourceOptions::StorageModeShared`.

#### Step 5.2: Metal Shading Language (MSL) Compute Shaders
- Author compute kernels in MSL:
  - Element-wise arithmetic: `kernel void add_arrays(...)`, `sub_arrays`, `mul_arrays`, `div_arrays`.
  - Activation functions: `relu_array`, `sigmoid_array`, `tanh_array`.
  - Strided copy and compaction kernels for GPU-side `.contiguous()` operations.
- Implement threadgroup and threadgrid sizing calculations matching Metal hardware limits (e.g., maximum threads per threadgroup).

#### Step 5.3: Compute Pipeline Encoding & Execution
- Implement kernel dispatch workflow:
  - Acquire a command buffer from the command queue.
  - Create a compute command encoder.
  - Bind the cached compute pipeline state.
  - Bind input and output `metal::Buffer` references to buffer slots.
  - Dispatch threadgroups.
  - Commit the command buffer and synchronize execution via `wait_until_completed()`.

#### Step 5.4: Apple Metal Performance Shaders (MPS) Integration
- Use the `objc` crate to bind to Apple's `MetalPerformanceShaders` dynamic library.
- Interface with `MPSMatrixMultiplication`:
  - Create `MPSMatrix` descriptors representing contiguous row-major matrices.
  - Encode matrix multiplication commands directly into the active Metal command buffer.
  - Leverage Apple Silicon hardware AMX co-processors for high-performance GPU GEMM.

#### Step 5.5: Host-Device Transfers
- Implement `to_device(Device)` transfers:
  - `Cpu -> Metal`: Allocate shared Metal buffer and copy host memory into it.
  - `Metal -> Cpu`: Read contents from the shared buffer back into host `Vec<f32>`.

---

### Phase 6: CUDA GPU Backend (NVIDIA Acceleration)

#### Step 6.1: Feature Gating & Context Management
- Enclose all CUDA-specific modules behind `#[cfg(feature = "cuda")]`.
- Use `cudarc` to initialize the CUDA driver context and stream on `Device::Cuda(device_id)`.
- Manage device memory allocations using `CudaSlice<f32>`.

#### Step 6.2: cuBLAS Matrix Multiplication
- Initialize cuBLAS context via `cudarc::cublas`.
- Implement matrix multiplication mapping to `cublasSgemm`:
  - Translate row-major layout to column-major cuBLAS expectations (using $C^T = B^T A^T$ identity).
  - Ensure input buffers are contiguous prior to invoking the BLAS call.

#### Step 6.3: Custom CUDA Kernels via NVRTC
- Write CUDA C kernel source code strings for element-wise operations and activations.
- Use NVRTC (NVIDIA Runtime Compilation) via `cudarc::nvrtc` to compile kernels to PTX at runtime.
- Cache loaded CUDA modules and functions in a central device manager.

#### Step 6.4: CUDA Memory Transfers
- Implement device transfers:
  - `Cpu -> Cuda`: Copy host slice to device VRAM via asynchronous stream copies.
  - `Cuda -> Cpu`: Synchronize stream and copy VRAM buffer back to host `Vec<f32>`.

---

### Phase 7: Neural Network Module API

#### Step 7.1: Learnable Parameters
- Implement the `Parameter` struct:
  - Wraps a `Tensor` with `requires_grad = true`.
  - Provides dereferencing to the underlying tensor for seamless use in arithmetic expressions.

#### Step 7.2: The `Module` Trait
- Define the standard `Module` trait:
  - `fn forward(&self, input: &Tensor) -> Tensor`
  - `fn parameters(&self) -> Vec<Tensor>`
  - `fn zero_grad(&self)`: Iterates through parameters and clears accumulated gradients.

#### Step 7.3: Common Neural Network Layers
- **`Linear` Layer**:
  - Contains `weight: Parameter` (shape $[C_{\text{out}}, C_{\text{in}}]$) and optional `bias: Parameter` (shape $[C_{\text{out}}]$).
  - Weight initialization using Kaiming Uniform:
    $$\mathcal{U}\left(-\frac{1}{\sqrt{C_{\text{in}}}}, \frac{1}{\sqrt{C_{\text{in}}}}\right)$$
  - Forward implementation: $y = x W^T + b$.
- **Activation Layers & Modules**:
  - `ReLU`, `Sigmoid`, `Tanh`, `GELU` wrapped as modular building blocks.
- **`Sequential` Container**:
  - Stores a sequence of boxed `dyn Module` objects.
  - Chaining forward passes where output of layer $i$ becomes input to layer $i+1$.

#### Step 7.4: Loss Functions
- **Mean Squared Error (`mse_loss`)**:
  - Computes $\frac{1}{N} \sum (y_{\text{pred}} - y_{\text{true}})^2$.
- **Cross-Entropy Loss (`cross_entropy_loss`)**:
  - Inputs: raw logits and class target indices.
  - Implements the numerically stable Log-Sum-Exp formulation to prevent numerical underflow/overflow.

---

### Phase 8: Optimization Engine & Training Loop

#### Step 8.1: Optimizer Interface
- Define an `Optimizer` trait with methods:
  - `step(&mut self)`: Updates parameter weights using accumulated gradients.
  - `zero_grad(&mut self)`: Resets gradient buffers to `None`.

#### Step 8.2: SGD with Momentum
- Implement `SGD`:
  - Configurable learning rate $\alpha$ and momentum $\beta$.
  - State tracking: velocity buffers for each parameter.
  - In-place mutation of the parameter's underlying storage buffer.

#### Step 8.3: Adam Optimizer
- Implement `Adam`:
  - Parameters: learning rate $\alpha$, $\beta_1$, $\beta_2$, $\epsilon$, and weight decay.
  - State tracking: first moment vector $m$ and second moment vector $v$ for each parameter, along with time step counter $t$.
  - Executes bias-corrected momentum updates directly into storage buffers.

#### Step 8.4: Training Workflow Validation
- Build an integration test training a multi-layer perceptron (MLP) on a non-linear synthetic problem (e.g., XOR classification or sine wave regression).
- Verify loss strictly decreases across epochs on CPU, Metal, and CUDA backends.

---

## 5. Summary Table: Implementation Phasing & Milestones

| Phase | Milestone Name | Key Theory & Invariants | Target Crates / Tech | Deliverable Checkpoint |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1** | **Setup & Architecture** | Platform isolation, unified error handling, type definitions | `thiserror`, `Cargo.toml` conditional compilation | Compiling workspace with `Device` & `Storage` enums |
| **Phase 2** | **Strided Tensor Core** | N-D to 1-D projection, canonical strides, zero-copy views, compaction | `std::sync::{Arc, RwLock}`, `rand` | Zero-copy transpose, reshape, broadcast, and `.contiguous()` |
| **Phase 3** | **CPU Kernels & GEMM** | Cache-friendly BLAS loops, multi-threaded SIMD, reductions | `rayon`, `matrixmultiply` | Fast CPU add, mul, relu, matmul, and sum reduction |
| **Phase 4** | **Reverse-Mode Autograd** | Dynamic DAG, VJPs, DFS topological sort, cycle avoidance | Interior mutability via `RwLock` | Automated backward pass verified against finite differences |
| **Phase 5** | **Metal GPU Backend** | Unified memory, MSL compute pipelines, Apple AMX co-processors | `metal`, `objc` (Apple MPS) | Metal-accelerated forward/backward tensor arithmetic |
| **Phase 6** | **CUDA GPU Backend** | Discrete VRAM, thread warps, cuBLAS GEMM, runtime NVRTC | `cudarc` (optional feature) | CUDA-accelerated tensor arithmetic on NVIDIA GPUs |
| **Phase 7** | **Neural Network API** | Parameter tracking, weight initialization, modular containers | Layer composition | `nn::Linear`, `nn::Sequential`, activations, cross-entropy |
| **Phase 8** | **Optimizers & End-to-End** | In-place parameter updates, momentum, bias correction | `optim::SGD`, `optim::Adam` | Trained MLP on non-linear dataset across all backends |
