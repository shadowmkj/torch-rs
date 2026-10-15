//! Error types and failure modes for tensor operations.
//!
//! This module defines [`TensorError`], the central error enum used across all
//! tensor initializations, memory layout calculations, view transformations, and indexing operations.

use thiserror::Error;

/// Errors that can occur during tensor creation, transformation, or indexing.
#[non_exhaustive]
#[derive(Error, Debug, PartialEq)]
pub enum TensorError {
    #[error(
        "cannot create tensor with shape {shape:?} ({expected} elements) from buffer of {actual} elements"
    )]
    ElementCountMismatch {
        shape: Vec<usize>,
        expected: usize,
        actual: usize,
    },

    #[error("index {index} is out of bounds for axis {axis} with size {size}")]
    IndexOutOfBounds {
        axis: usize,
        index: usize,
        size: usize,
    },

    #[error("coordinate count ({provided}) does not match tensor rank ({expected})")]
    RankMismatch { expected: usize, provided: usize },

    #[error("zero-sized dimension is unsupported: shape {shape:?} contains 0")]
    ZeroSizedDimension { shape: Vec<usize> },

    #[error("shape mismatch in {op}: lhs {lhs:?}, rhs {rhs:?}")]
    ShapeMismatch {
        op: &'static str,
        lhs: Vec<usize>,
        rhs: Vec<usize>,
    },

    #[error("device mismatch: operation between {lhs:?} and {rhs:?}")]
    DeviceMismatch { lhs: String, rhs: String },
    #[error(transparent)]
    NormalError(#[from] rand_distr::NormalError),
    #[error(transparent)]
    UniformError(#[from] rand_distr::uniform::Error),
}
