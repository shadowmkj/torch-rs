# torch-rs 🦀🔥

> *"How bad could it be to build a Neural Network library from scratch?"*

**torch-rs** is an educational, high-performance machine learning library written from scratch in Rust. It explores tensor computation, strided memory layouts, automatic differentiation (autograd), and multi-hardware backends (CPU, Apple Silicon Metal, and NVIDIA CUDA) from first principles.

---

## ⚡ Features Implemented So Far

- **Compile-Time Rank Verification**: Built with modern Rust const generics (`Tensor<const D: usize>`), ensuring rank mismatches are caught by the compiler before code ever runs.
- **Strided Layout Engine**: Foundation for zero-copy views (`transpose`, `slice`, and `broadcast_to`) using stride arithmetic over shared memory (`Arc<Storage>`).
- **Stack-Allocated Metadata**: Shapes and strides are stored as fixed-size stack arrays `[usize; D]`, eliminating dynamic heap allocations for tensor handles.
- **Actionable Diagnostics**: Structured domain errors via `thiserror` that report exact shapes, axes, and element counts.

> Yeah.. This is nowhere near a neural network library (yet!)

---

## 🚀 Quick Start

Add `torch-rs` to your `Cargo.toml`:

```toml
[dependencies]
torch-rs = { path = "." }
```

All it can do now is create tensors, and index them.

### Creating Tensors & Indexing

```rust
use torch_rs::tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a 2D tensor (2 rows, 3 columns) from raw data
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::from_vec(data, [2, 3])?;

    // Create tensors filled with zeros or ones
    let zeros = Tensor::zeros([2, 3])?;
    let ones = Tensor::ones([2, 3])?;

    // Multi-dimensional coordinate to flat storage indexing
    let flat_idx = tensor.linear_index([1, 2])?;
    assert_eq!(flat_idx, 5);

    Ok(())
}
```

---

## 🛠️ Development & Testing

Run all unit tests:
```bash
cargo test
```

Generate and view documentation:
```bash
cargo doc --no-deps --open
```

Run the example runner:
```bash
cargo run --example base
```

---

## 🗺️ Project Roadmap

> [!NOTE]
> The roadmap could change — I haven't made a rigid plan yet, so I'm figuring things out as I go!

- [x] **Phase 1: Foundations**
  - [x] Memory layout & strided indexing mathematics
  - [x] Const-generic rank architecture (`Tensor<const D: usize>`)
  - [x] Structured domain errors (`TensorError`)
  - [x] Core constructors: `from_vec`, `from_slice`, `zeros`, `ones`
- [ ] **Phase 2: Zero-Copy Views & Operations**
  - [ ] Zero-copy transformations (`transpose`, `slice`, `reshape`)
  - [ ] Broadcasting engine with zero-stride synthesis
  - [ ] Physical memory compaction (`is_contiguous`, `contiguous`)
- [ ] **Phase 3: Automatic Differentiation**
  - [ ] Dynamic computation graph (DAG) builder
  - [ ] Reverse-mode automatic differentiation (`loss.backward()`)
  - [ ] Vector-Jacobian products (VJPs) for arithmetic and matrix operations
- [ ] **Phase 4: Neural Network Modules & Optimizers**
  - [ ] High-level layer abstractions (`nn::Module`, `nn::Linear`)
  - [ ] Activations & loss functions (ReLU, GELU, CrossEntropy)
  - [ ] Optimizers (SGD with momentum, Adam)

---

## 📜 License

Licensed under the MIT License.
