//! # `torch-rs`
//!
//! An educational deep learning library built from scratch in Rust.
//!
//! `torch-rs` provides PyTorch-like tensor abstractions with compile-time rank verification,
//! strided multi-dimensional memory layout, zero-copy views, and hardware-accelerated storage backends.
//!
//! ## Core Architecture
//!
//! - **Compile-Time Rank Verification**: Tensors are parameterized with const generics [`Tensor<const D: usize>`](tensor::Tensor),
//!   preventing coordinate and rank mismatches at compile time without heap allocations for shape or stride vectors.
//! - **Strided Memory Layout**: Zero-copy operations like [`transpose`](tensor::Tensor::transpose) manipulate
//!   shape, strides, and offsets without copying the underlying buffer.
//! - **Decoupled Physical Storage**: Tensors hold a lightweight handle (`Arc<Storage>`) to shared buffers across
//!   backends ([`Storage::Cpu`](storage::Storage::Cpu) and future Metal / CUDA targets).
//! - **PyTorch-Style Formatting**: Tensors implement [`std::fmt::Display`] with aligned indentation, sign-aware padding,
//!   and bracketed multi-dimensional slicing.
//!
//! ## Quickstart
//!
//! ```rust
//! use torch_rs::tensor::Tensor;
//!
//! // Create a 2x3 tensor filled with zeros
//! let t = Tensor::zeros([2, 3]).unwrap();
//!
//! // Create a 2D tensor from a vector of numbers
//! let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]).unwrap();
//!
//! // Transpose axes without copying underlying storage
//! let transposed = x.transpose(0, 1).unwrap();
//!
//! // Format tensor with PyTorch-style bracketed output
//! let output = format!("{transposed}");
//! assert!(output.starts_with("tensor("));
//! ```

pub mod storage;
pub mod tensor;
