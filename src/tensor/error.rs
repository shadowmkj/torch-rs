use thiserror::Error;

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
}
