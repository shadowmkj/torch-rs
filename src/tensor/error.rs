use thiserror::Error;

#[derive(Error, Debug)]
pub enum TensorError {
    #[error("invalid shape")]
    InvalidShape,
    #[error("index out of bounds got: {index}, maximum: {limit}")]
    IndexOutOfBounds { limit: usize, index: usize },
}
