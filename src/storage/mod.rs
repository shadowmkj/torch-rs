//! Physical memory storage abstractions and device buffers.
//!
//! This module defines the [`Storage`] enum, representing raw numerical data
//! allocated across various hardware backends.
//!
//! Tensors share storage buffers using reference-counted pointers (`Arc<Storage>`),
//! enabling zero-copy views (such as slices and transposed layouts) without
//! re-allocating or duplicating element buffers.

/// Physical memory buffer storing raw numerical elements across backends.
#[derive(Debug, Clone, PartialEq)]
pub enum Storage {
    /// Contiguous heap-allocated vector in host memory.
    Cpu(Vec<f32>),
    //TODO: add more storage (Metal, CUDA, wgpu) (20261014-114506)
}

impl PartialEq<Vec<f32>> for Storage {
    fn eq(&self, other: &Vec<f32>) -> bool {
        match self {
            Storage::Cpu(items) => items == other,
        }
    }
}
