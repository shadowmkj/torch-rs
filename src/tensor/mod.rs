//! Multi-dimensional strided tensor abstraction and operations.
//!
//! A [`Tensor<const D: usize>`] represents an `N`-dimensional array of 32-bit floating-point
//! values with a strided memory layout.
//!
//! ## Mathematical Layout
//!
//! Tensors map multi-dimensional coordinates `(i_0, i_1, ..., i_{D-1})` to a flat 1D
//! storage offset using:
//!
//! ```text
//! linear_index(i_0, ..., i_{D-1}) = offset + sum_{k=0}^{D-1} (i_k * strides[k])
//! ```
//!
//! ## Key Capabilities
//!
//! - **Constant & Vector Initializers**: [`Tensor::zeros`], [`Tensor::ones`], [`Tensor::from_vec`], [`Tensor::from_slice`].
//! - **Random Distributions**: Uniform distributions ([`Tensor::rand`], [`Tensor::uniform`]) and
//!   Gaussian distributions ([`Tensor::randn`], [`Tensor::normal`]), with seeded deterministic options.
//! - **Zero-Copy Views**: [`Tensor::transpose`] creates a new view into the same underlying storage
//!   by swapping shape and stride dimensions in `O(1)` time without copying data.

pub mod error;
pub mod iterator;

use rand_distr::{Distribution, Normal, Uniform};

use crate::storage::Storage;
use crate::tensor::error::TensorError;
use std::fmt;
use std::sync::Arc;

/// Hardware execution device for tensor storage and compute kernels.
#[derive(Debug, Copy, Clone, Default)]
pub enum Device {
    /// Host CPU device using standard memory allocation.
    #[default]
    Cpu,
}

/// An `N`-dimensional array of numerical values with strided memory layout and shared storage.
///
/// The rank `D` is verified at compile-time via const generics, while dimensions
/// and strides are stored in fixed-size stack arrays without heap allocation.
///
/// # Errors
///
/// Returns [`TensorError::ZeroSizedDimension`] if any dimension is `0`.
/// Zero-volume Tensors (e.g., `[2, 0, 3]`) are currently unsupported
#[derive(Debug, Clone)]
pub struct Tensor<const D: usize> {
    storage: Arc<Storage>,
    shape: [usize; D],
    strides: [usize; D],
    offset: usize,
    device: Device,
}

impl<const D: usize> Tensor<D> {
    /// Returns a reference to the tensor's shape.
    #[inline]
    pub fn shape(&self) -> &[usize; D] {
        &self.shape
    }

    /// Returns the number of elements in the tensor.
    #[inline]
    pub fn numel(&self) -> usize {
        self.shape.iter().product::<usize>()
    }

    /// Returns a reference to the tensor's strides.
    #[inline]
    pub fn strides(&self) -> &[usize; D] {
        &self.strides
    }

    /// Returns the storage offset of the tensor.
    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Return the `f32` value at the coordinates.
    pub fn get(&self, coords: [usize; D]) -> Result<f32, TensorError> {
        let idx = self.linear_index(coords)?;
        match &*self.storage {
            Storage::Cpu(data) => Ok(data[idx]),
        }
    }

    /// Creates a new tensor of the given shape filled with zeros (`0.0`).
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::ZeroSizedDimension`] if any dimension in `shape` is `0`.
    pub fn zeros(shape: [usize; D]) -> Result<Self, TensorError> {
        let strides = Self::compute_strides(&shape)?;

        // let device = device.unwrap_or_default(); //uncomment this
        // For now just use CPU as default
        let device = Device::default(); //TODO: Must change this back when designing the public API
        let num_elements: usize = shape.iter().product();
        let storage = match device {
            Device::Cpu => Storage::Cpu(vec![0.0; num_elements]),
            //TODO: Update once more storage options are implemented (20261014-114506)
        };

        Ok(Self {
            storage: Arc::new(storage),
            shape,
            offset: 0,
            strides,
            device,
        })
    }

    /// Creates a new tensor of the given shape filled with ones (`1.0`).
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::ZeroSizedDimension`] if any dimension in `shape` is `0`.
    pub fn ones(shape: [usize; D]) -> Result<Self, TensorError> {
        let strides = Self::compute_strides(&shape)?;

        // let device = device.unwrap_or_default(); //uncomment this
        // For now just use CPU as default
        let device = Device::default(); //TODO: Must change this back when designing the public API
        let num_elements: usize = shape.iter().product();
        let storage = match device {
            Device::Cpu => Storage::Cpu(vec![1.0; num_elements]),
            //TODO: Update once more storage options are implemented (20261014-114506)
        };

        Ok(Self {
            storage: Arc::new(storage),
            shape,
            offset: 0,
            strides,
            device,
        })
    }

    /// Creates a new tensor from an owned vector of `f32` elements with the specified shape.
    ///
    /// # Errors
    ///
    /// - Returns [`TensorError::ZeroSizedDimension`] if any dimension in `shape` is `0`.
    /// - Returns [`TensorError::ElementCountMismatch`] if the number of elements in `data`
    ///   does not match the total product of the dimensions.
    pub fn from_vec(data: Vec<f32>, shape: [usize; D]) -> Result<Self, TensorError> {
        let strides = Self::compute_strides(&shape)?;
        // let device = device.unwrap_or_default(); //uncomment this
        // For now just use CPU as default
        let device = Device::default(); //TODO: Must change this back when designing the public API

        // Shape must match the input data
        let num_elements: usize = shape.iter().product();
        if data.len() != num_elements {
            return Err(TensorError::ElementCountMismatch {
                shape: shape.to_vec(),
                expected: num_elements,
                actual: data.len(),
            });
        }

        let storage = match device {
            Device::Cpu => Storage::Cpu(data),
            //TODO: Update once more storage options are implemented (20261014-114506)
        };

        Ok(Self {
            storage: Arc::new(storage),
            shape,
            offset: 0,
            strides,
            device,
        })
    }

    /// Creates a new tensor by copying elements from a slice with the specified shape.
    ///
    /// # Errors
    ///
    /// - Returns [`TensorError::ZeroSizedDimension`] if any dimension in `shape` is `0`.
    /// - Returns [`TensorError::ElementCountMismatch`] if the slice length does not match
    ///   the total product of the dimensions.
    pub fn from_slice(slice: &[f32], shape: [usize; D]) -> Result<Self, TensorError> {
        let strides = Self::compute_strides(&shape)?;

        // let device = device.unwrap_or_default(); //uncomment this
        // For now just use CPU as default
        let device = Device::default(); //TODO: Must change this back when designing the public API

        // Shape must match the input data
        let num_elements: usize = shape.iter().product();
        if slice.len() != num_elements {
            return Err(TensorError::ElementCountMismatch {
                shape: shape.to_vec(),
                expected: num_elements,
                actual: slice.len(),
            });
        }

        let storage = match device {
            Device::Cpu => Storage::Cpu(Vec::from(slice)),
            //TODO: Update once more storage options are implemented (20261014-114506)
        };

        Ok(Self {
            storage: Arc::new(storage),
            shape,
            offset: 0,
            strides,
            device,
        })
    }

    /// Creates a new tensor of the given shape filled with uniform random values in `[0.0, 1.0)`.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::ZeroSizedDimension`] if any dimension in `shape` is `0`.
    pub fn rand(shape: [usize; D]) -> Result<Self, TensorError> {
        Self::uniform(0.0, 1.0, shape)
    }

    /// Creates a new tensor filled with uniform random values in `[0.0, 1.0)` using a fixed seed for determinism.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::ZeroSizedDimension`] if any dimension in `shape` is `0`.
    pub fn rand_seeded(shape: [usize; D], seed: u64) -> Result<Self, TensorError> {
        Self::uniform_seeded(0.0, 1.0, shape, seed)
    }

    /// Creates a new tensor with uniform random values in the range `[low, high)`.
    ///
    /// # Errors
    ///
    /// - Returns [`TensorError::ZeroSizedDimension`] if any dimension in `shape` is `0`.
    /// - Returns [`TensorError::UniformError`] if `low >= high` or parameters are not finite.
    pub fn uniform(low: f32, high: f32, shape: [usize; D]) -> Result<Self, TensorError> {
        Self::uniform_with_rng(low, high, shape, rand::rng())
    }

    /// Creates a new tensor with uniform random values in `[low, high)` using a fixed seed for determinism.
    ///
    /// # Errors
    ///
    /// - Returns [`TensorError::ZeroSizedDimension`] if any dimension in `shape` is `0`.
    /// - Returns [`TensorError::UniformError`] if `low >= high` or parameters are not finite.
    pub fn uniform_seeded(
        low: f32,
        high: f32,
        shape: [usize; D],
        seed: u64,
    ) -> Result<Self, TensorError> {
        use rand::SeedableRng;
        let rng = rand::rngs::StdRng::seed_from_u64(seed);
        Self::uniform_with_rng(low, high, shape, rng)
    }

    /// Creates a new tensor with uniform random values in `[low, high)` using a custom random number generator.
    ///
    /// # Errors
    ///
    /// - Returns [`TensorError::ZeroSizedDimension`] if any dimension in `shape` is `0`.
    /// - Returns [`TensorError::UniformError`] if `low >= high` or parameters are not finite.
    pub fn uniform_with_rng<R: rand::Rng>(
        low: f32,
        high: f32,
        shape: [usize; D],
        mut rng: R,
    ) -> Result<Self, TensorError> {
        let strides = Self::compute_strides(&shape)?;

        let uniform = Uniform::new(low, high)?;
        let num_elements: usize = shape.iter().product();
        let data: Vec<f32> = uniform.sample_iter(&mut rng).take(num_elements).collect();

        // let device = device.unwrap_or_default(); //uncomment this
        // For now just use CPU as default
        let device = Device::default(); //TODO: Must change this back when designing the public API
        let storage = match device {
            Device::Cpu => Storage::Cpu(data),
            //TODO: Update once more storage options are implemented (20261014-114506)
        };

        Ok(Self {
            storage: Arc::new(storage),
            shape,
            offset: 0,
            strides,
            device,
        })
    }

    /// Creates a new tensor of the given shape with a standard normal distribution.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::ZeroSizedDimension`] if any dimension in `shape` is `0`.
    pub fn randn(shape: [usize; D]) -> Result<Self, TensorError> {
        Self::normal(0.0, 1.0, shape)
    }

    /// Creates a new tensor with a standard normal distribution using a fixed seed for determinism.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::ZeroSizedDimension`] if any dimension in `shape` is `0`.
    pub fn randn_seeded(shape: [usize; D], seed: u64) -> Result<Self, TensorError> {
        Self::normal_seeded(0.0, 1.0, shape, seed)
    }

    /// Creates a new tensor of the given shape with a normal distribution.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::ZeroSizedDimension`] if any dimension in `shape` is `0`.
    pub fn normal(mean: f32, std_dev: f32, shape: [usize; D]) -> Result<Self, TensorError> {
        Self::normal_with_rng(mean, std_dev, shape, rand::rng())
    }

    /// Creates a new tensor with a normal distribution using a fixed seed for determinism.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::ZeroSizedDimension`] if any dimension in `shape` is `0`.
    pub fn normal_seeded(
        mean: f32,
        std_dev: f32,
        shape: [usize; D],
        seed: u64,
    ) -> Result<Self, TensorError> {
        use rand::SeedableRng;
        let rng = rand::rngs::StdRng::seed_from_u64(seed);
        Self::normal_with_rng(mean, std_dev, shape, rng)
    }

    /// Creates a new tensor with a normal distribution using a custom random number generator.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::ZeroSizedDimension`] if any dimension in `shape` is `0`.
    pub fn normal_with_rng<R: rand::Rng>(
        mean: f32,
        std_dev: f32,
        shape: [usize; D],
        mut rng: R,
    ) -> Result<Self, TensorError> {
        let strides = Self::compute_strides(&shape)?;

        let normal = Normal::new(mean, std_dev)?;
        let num_elements: usize = shape.iter().product();
        let data: Vec<f32> = normal.sample_iter(&mut rng).take(num_elements).collect();

        // let device = device.unwrap_or_default(); //uncomment this
        // For now just use CPU as default
        let device = Device::default(); //TODO: Must change this back when designing the public API
        let storage = match device {
            Device::Cpu => Storage::Cpu(data),
            //TODO: Update once more storage options are implemented (20261014-114506)
        };

        Ok(Self {
            storage: Arc::new(storage),
            shape,
            offset: 0,
            strides,
            device,
        })
    }

    /// Returns a new tensor that is a narrowed view of `self` along `dim`
    /// from index `start` (inclusive) to `end` (exclusive).
    ///
    /// This is an O(1) zero-copy operation that adjusts the storage offset and dimension size.
    ///
    /// # Errors
    ///
    /// - Returns [`TensorError::IndexOutOfBounds`] if `dim >= D` or `end > self.shape[dim]`.
    /// - Returns [`TensorError::InvalidSliceRange`] if `start > end`.
    /// - Returns [`TensorError::ZeroSizedDimension`] if `start == end` (while zero-sized shapes are
    ///   unsupported).
    pub fn slice(&self, dim: usize, start: usize, end: usize) -> Result<Self, TensorError> {
        if dim >= D {
            return Err(TensorError::IndexOutOfBounds {
                axis: dim,
                index: dim,
                size: D,
            });
        }

        if start > end {
            return Err(TensorError::InvalidSliceRange {
                axis: dim,
                start,
                end,
                size: self.shape[dim],
            });
        }

        if end > self.shape[dim] {
            return Err(TensorError::IndexOutOfBounds {
                axis: dim,
                index: end,
                size: self.shape[dim],
            });
        }

        let new_dim_size = end - start;
        if new_dim_size == 0 {
            let mut new_shape = self.shape.to_vec();
            new_shape[dim] = 0;
            return Err(TensorError::ZeroSizedDimension { shape: new_shape });
        }

        let mut shape = self.shape;
        shape[dim] = new_dim_size;

        let offset = self.offset + start * self.strides[dim];

        Ok(Self {
            storage: Arc::clone(&self.storage),
            shape,
            strides: self.strides,
            offset,
            device: self.device,
        })
    }

    /// Returns a new contiguous tensor with elements copied into a fresh, compact storage layout.
    ///
    /// If the tensor is already compact (contiguous row-major layout with zero offset spanning
    /// the full storage buffer), this creates an identical compact representation. Otherwise,
    /// elements are gathered according to current strides and offset into a newly allocated buffer.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError`] if allocating storage or computing strides fails.
    pub fn compaction(&self) -> Result<Tensor<D>, TensorError> {
        if self.is_compact() {
            return Ok(self.clone());
        }

        let new_data: Vec<f32> = self.iter().collect();
        Self::from_vec(new_data, self.shape)
    }

    /// Return `true` if the tensor elements are stored contiguosly in standard row-major order.
    pub fn is_contiguous(&self) -> bool {
        // Strides must match the expected strides for new a new tensor
        match Self::compute_strides(&self.shape) {
            Ok(expected_strides) => expected_strides == self.strides,
            Err(_) => false,
        }
    }

    /// Returns `true` if the tensor is contiguous in memory and starts at offset 0,
    /// spanning the storage from its beginning without any offset or slicing gaps.
    pub fn is_compact(&self) -> bool {
        if self.offset != 0 {
            return false;
        }

        let size: usize = self.shape().iter().product();

        let storage_size = match &*self.storage {
            Storage::Cpu(items) => items.len(),
        };

        if size != storage_size {
            return false;
        }

        // Strides must match the expected strides for new a new tensor
        match Self::compute_strides(&self.shape) {
            Ok(expected_strides) => expected_strides == self.strides,
            Err(_) => false,
        }
    }

    pub fn reshape<const R: usize>(&self, new_shape: [usize; R]) -> Result<Tensor<R>, TensorError> {
        // The count of elements must match
        let current_shape_element_count: usize = self.shape.iter().product();
        let new_shape_element_count: usize = new_shape.iter().product();
        if current_shape_element_count != new_shape_element_count {
            return Err(TensorError::ElementCountMismatch {
                shape: new_shape.to_vec(),
                expected: current_shape_element_count,
                actual: new_shape_element_count,
            });
        }

        let base = if self.is_compact() {
            self.clone()
        } else {
            self.compaction()?
        };

        let new_strides = Tensor::<R>::compute_strides(&new_shape)?;
        Ok(Tensor {
            storage: Arc::clone(&base.storage),
            shape: new_shape,
            strides: new_strides,
            offset: 0,
            device: self.device,
        })
    }

    /// Computes the 1D flat storage offset corresponding to the given multi-dimensional coordinate indices.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::IndexOutOfBounds`] if any coordinate is greater than or equal to
    /// the dimension size along that axis.
    pub fn linear_index(&self, indices: [usize; D]) -> Result<usize, TensorError> {
        let mut flat_index = self.offset;
        // index is the actual value, i is the axis
        for (i, &index) in indices.iter().enumerate().take(D) {
            if index >= self.shape[i] {
                return Err(TensorError::IndexOutOfBounds {
                    axis: i,
                    index,
                    size: self.shape[i],
                });
            }
            flat_index += index * self.strides[i];
        }

        Ok(flat_index)
    }

    /// Returns a tensor that is a transposed version of `self`, with dimensions `dim0` and `dim1` swapped.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::IndexOutOfBounds`] if `dim0` or `dim1` is greater than or equal to the number of dimensions `D`.
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Self, TensorError> {
        if dim0 >= D || dim1 >= D {
            return Err(TensorError::IndexOutOfBounds {
                axis: dim0.max(dim1),
                index: dim0.max(dim1),
                size: D,
            });
        }

        let mut shape = self.shape;
        let mut strides = self.strides;

        shape.swap(dim0, dim1);
        strides.swap(dim0, dim1);

        Ok(Self {
            storage: Arc::clone(&self.storage),
            shape,
            strides,
            offset: self.offset,
            device: self.device,
        })
    }

    /// Computes canonical row-major (C-contiguous) strides for the given shape.
    fn compute_strides(shape: &[usize; D]) -> Result<[usize; D], TensorError> {
        //WARN: No support for 0-sized dimensions. (shapes with 0 in it)
        // eg: creating a tensor with shape [2,0,3] will result in runtime error, 0 must NOT be in
        // shape.
        // This is to make development of Tensor methods easier and reduce edge cases involving 0

        //TODO: remove this check after supporting 0-sized dimensions (20261015-130759)
        if shape.contains(&0) {
            return Err(TensorError::ZeroSizedDimension {
                shape: shape.to_vec(),
            });
        }

        let mut strides = [1; D];
        if D > 1 {
            for i in (0..D - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            } //NOTE: refer ./tasks/20261014-021101/strides.png
        }

        Ok(strides)
    }

    /// Recursively formats tensor elements across dimensions with PyTorch-style aligned indentation.
    fn format_dim(
        &self,
        f: &mut fmt::Formatter<'_>,
        dim: usize,
        coords: &mut [usize; D],
        indent: usize,
    ) -> fmt::Result {
        write!(f, "[")?;
        if dim == D - 1 {
            for i in 0..self.shape[dim] {
                coords[dim] = i;
                if i > 0 {
                    write!(f, ", ")?;
                }
                let idx = self.linear_index(*coords).map_err(|_| fmt::Error)?;
                let val = match &*self.storage {
                    Storage::Cpu(data) => data[idx],
                };
                if val.is_sign_negative() {
                    write!(f, "{:.4}", val)?;
                } else {
                    write!(f, " {:.4}", val)?;
                }
            }
        } else {
            for i in 0..self.shape[dim] {
                coords[dim] = i;
                if i > 0 {
                    if dim == D - 2 {
                        writeln!(f, ",")?;
                    } else {
                        writeln!(f, ",\n")?;
                    }
                    write!(f, "{:width$}", "", width = indent + 1)?;
                }
                self.format_dim(f, dim + 1, coords, indent + 1)?;
            }
        }
        write!(f, "]")
    }
}

impl Tensor<2> {
    /// Returns the transpose of a 2D tensor, swapping dimensions 0 and 1.
    ///
    /// This is a convenience shortcut for [`transpose(0, 1)`](Tensor::transpose).
    ///
    /// # Errors
    ///
    /// Returns [`TensorError`] if transposing fails.
    pub fn t(&self) -> Result<Tensor<2>, TensorError> {
        self.transpose(0, 1)
    }
}

impl<const D: usize> fmt::Display for Tensor<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "tensor(")?;
        if D == 0 {
            let idx = self.linear_index([0; D]).map_err(|_| fmt::Error)?;
            let val = match &*self.storage {
                Storage::Cpu(data) => data[idx],
            };
            write!(f, "{:.4}", val)?;
        } else {
            let mut coords = [0; D];
            self.format_dim(f, 0, &mut coords, 7)?;
        }
        write!(f, ")")
    }
}

#[test]
fn test_basic_tensor() {
    let tensor = Tensor::zeros([2, 3]).unwrap();
    assert_eq!(tensor.shape, [2, 3]);
    assert_eq!(tensor.offset, 0);
    assert_eq!(tensor.strides, [3, 1]);
}

#[test]
fn test_tensor_zeros() {
    let tensor = Tensor::zeros([2, 3]).unwrap();
    assert_eq!(*tensor.storage, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    assert_eq!(tensor.shape, [2, 3]);
    assert_eq!(tensor.offset, 0);
    assert_eq!(tensor.strides, [3, 1]);
}

#[test]
fn test_tensor_ones() {
    let tensor = Tensor::ones([2, 3]).unwrap();
    assert_eq!(*tensor.storage, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    assert_eq!(tensor.shape, [2, 3]);
    assert_eq!(tensor.offset, 0);
    assert_eq!(tensor.strides, [3, 1]);
}

#[test]
fn test_unit_tensor() {
    let tensor = Tensor::from_vec(vec![1.0], []).unwrap();
    assert_eq!(tensor.shape, []);
    assert_eq!(tensor.offset, 0);
    assert_eq!(tensor.strides, []);
}

#[test]
fn test_tensor_from_slice() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::from_slice(&data, [2, 2]).unwrap();
    assert_eq!(*tensor.storage, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(tensor.offset, 0);
    assert_eq!(tensor.strides, [2, 1]);
}

#[test]
fn test_compute_strides_dimensions() -> Result<(), TensorError> {
    assert_eq!(Tensor::compute_strides(&[])?, []);
    assert_eq!(Tensor::compute_strides(&[5])?, [1]);
    assert_eq!(Tensor::compute_strides(&[2, 3])?, [3, 1]);
    assert_eq!(Tensor::compute_strides(&[2, 3, 4])?, [12, 4, 1]);
    assert_eq!(Tensor::compute_strides(&[1, 5])?, [5, 1]);

    Ok(())
}

#[test]
fn test_zero_volume_tensor() {
    let t = Tensor::zeros([2, 0, 3]);
    assert!(matches!(t, Err(TensorError::ZeroSizedDimension { .. })));
}

#[test]
fn test_tensor_from_vec() -> Result<(), TensorError> {
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::from_vec(data, [2, 3])?;
    assert_eq!(tensor.strides, [3, 1]);
    assert_eq!(*tensor.storage, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    Ok(())
}

#[test]
fn test_tensor_with_wrong_dimensions_1() -> Result<(), TensorError> {
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let e = Tensor::from_vec(data, [1, 1]).err().unwrap();
    assert_eq!(
        e,
        TensorError::ElementCountMismatch {
            shape: vec![1, 1],
            expected: 1,
            actual: 6
        }
    );
    Ok(())
}

#[test]
fn test_tensor_with_wrong_dimensions_2() -> Result<(), TensorError> {
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let e = Tensor::from_vec(data, [3, 3]).err().unwrap();
    assert_eq!(
        e,
        TensorError::ElementCountMismatch {
            shape: vec![3, 3],
            expected: 9,
            actual: 6
        }
    );
    Ok(())
}

#[test]
fn tensor_linear_indexing() -> Result<(), TensorError> {
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::from_vec(data, [2, 3])?;
    assert_eq!(tensor.linear_index([1, 2])?, 5);
    assert_eq!(tensor.linear_index([0, 1])?, 1);
    assert_eq!(tensor.linear_index([0, 0])?, 0);
    assert_eq!(tensor.linear_index([0, 2])?, 2);
    let e = tensor.linear_index([0, 3]).err().unwrap();
    assert_eq!(
        e,
        TensorError::IndexOutOfBounds {
            axis: 1,
            index: 3,
            size: 3
        }
    );
    Ok(())
}

#[test]
fn test_storage_sharing_on_view() -> Result<(), TensorError> {
    let t1 = Tensor::zeros([2, 3])?;
    let t2 = t1.clone(); // Shallow handle clone
    assert!(Arc::ptr_eq(&t1.storage, &t2.storage));

    Ok(())
}

#[test]
fn test_tensor_normal() {
    let tensor = Tensor::<1>::normal(0.0, 1.0, [10_000]).unwrap();
    let data = match &*tensor.storage {
        Storage::Cpu(vec) => vec,
    };

    let mean = data.iter().sum::<f32>() / data.len() as f32;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
    let std_dev = variance.sqrt();

    // Check that mean is roughly 0.0 and std_dev is roughly 1.0
    assert!((mean - 0.0).abs() < 0.05, "Mean was {mean}");
    assert!((std_dev - 1.0).abs() < 0.05, "Std dev was {std_dev}");
}

#[test]
fn test_tensor_randn() {
    let tensor = Tensor::<3>::randn([5, 30, 100]).unwrap();
    let data = match &*tensor.storage {
        Storage::Cpu(vec) => vec,
    };

    let mean = data.iter().sum::<f32>() / data.len() as f32;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
    let std_dev = variance.sqrt();

    // Check that mean is roughly 0.0 and std_dev is roughly 1.0
    assert!((mean - 0.0).abs() < 0.05, "Mean was {mean}");
    assert!((std_dev - 1.0).abs() < 0.05, "Std dev was {std_dev}");
}

#[test]
fn test_tensor_seeded_deterministic() {
    let t1 = Tensor::<2>::randn_seeded([3, 4], 42).unwrap();
    let t2 = Tensor::<2>::randn_seeded([3, 4], 42).unwrap();
    let t3 = Tensor::<2>::randn_seeded([3, 4], 999).unwrap();

    // Identical seeds must produce bit-for-bit identical storage values
    assert_eq!(*t1.storage, *t2.storage);

    // Different seeds must produce different values
    assert_ne!(*t1.storage, *t3.storage);
}

#[test]
fn test_tensor_transpose() {
    let t = Tensor::<2>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).unwrap();
    let transposed = t.transpose(0, 1).unwrap();

    assert_eq!(transposed.shape, [3, 2]);
    assert_eq!(transposed.strides, [1, 3]);

    // Out of bounds transpose error test
    assert!(t.transpose(0, 2).is_err());
    assert!(t.transpose(2, 0).is_err());
}

#[test]
fn test_tensor_transpose_higher_dimensions() {
    let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    let t = Tensor::<3>::from_vec(data, [2, 3, 4]).unwrap();

    // Default strides for shape [2, 3, 4] should be [12, 4, 1]
    let transposed = t.transpose(0, 2).unwrap();
    assert_eq!(transposed.shape, [4, 3, 2]);
    assert_eq!(transposed.strides, [1, 4, 12]);

    let transposed_inner = t.transpose(1, 2).unwrap();
    assert_eq!(transposed_inner.shape, [2, 4, 3]);
    assert_eq!(transposed_inner.strides, [12, 1, 4]);

    // Out of bounds transpose error test
    assert!(t.transpose(0, 3).is_err());
    assert!(t.transpose(3, 1).is_err());
}

#[test]
fn test_tensor_rand() {
    let t = Tensor::<2>::rand([10, 20]).unwrap();
    let data = match &*t.storage {
        Storage::Cpu(vec) => vec,
    };

    // Every value must be within [0.0, 1.0)
    for &val in data {
        assert!(val >= 0.0 && val < 1.0, "Value {val} out of bounds");
    }

    // Mean of uniform [0, 1) should be roughly 0.5
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    assert!((mean - 0.5).abs() < 0.1, "Mean was {mean}");
}

#[test]
fn test_tensor_uniform() {
    let t = Tensor::<2>::uniform(-2.0, 5.0, [10, 20]).unwrap();
    let data = match &*t.storage {
        Storage::Cpu(vec) => vec,
    };

    // Every value must be within [-2.0, 5.0)
    for &val in data {
        assert!(val >= -2.0 && val < 5.0, "Value {val} out of bounds");
    }

    // Invalid bounds (low >= high) should return an error
    assert!(Tensor::<2>::uniform(5.0, 2.0, [2, 2]).is_err());
}

#[test]
fn test_tensor_rand_seeded() {
    let t1 = Tensor::<2>::rand_seeded([4, 4], 12345).unwrap();
    let t2 = Tensor::<2>::rand_seeded([4, 4], 12345).unwrap();
    let t3 = Tensor::<2>::rand_seeded([4, 4], 54321).unwrap();

    // Same seed yields identical values
    assert_eq!(*t1.storage, *t2.storage);

    // Different seeds yield different values
    assert_ne!(*t1.storage, *t3.storage);
}

#[test]
fn test_tensor_display_scalar() {
    let t = Tensor::from_vec(vec![3.14159], []).unwrap();
    assert_eq!(format!("{t}"), "tensor(3.1416)");
}

#[test]
fn test_tensor_display_1d() {
    let t = Tensor::from_vec(vec![1.0, 2.5, -3.25], [3]).unwrap();
    assert_eq!(format!("{t}"), "tensor([ 1.0000,  2.5000, -3.2500])");
}

#[test]
fn test_tensor_display_2d() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]).unwrap();
    let expected = "\
tensor([[ 1.0000,  2.0000],
        [ 3.0000,  4.0000]])";
    assert_eq!(format!("{t}"), expected);
}

#[test]
fn test_tensor_display_2d_transposed() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).unwrap();
    let transposed = t.transpose(0, 1).unwrap();
    let expected = "\
tensor([[ 1.0000,  4.0000],
        [ 2.0000,  5.0000],
        [ 3.0000,  6.0000]])";
    assert_eq!(format!("{transposed}"), expected);
}

#[test]
fn test_tensor_display_3d() {
    let data: Vec<f32> = (1..=8).map(|x| x as f32).collect();
    let t = Tensor::from_vec(data, [2, 2, 2]).unwrap();
    let expected = "\
tensor([[[ 1.0000,  2.0000],
         [ 3.0000,  4.0000]],

        [[ 5.0000,  6.0000],
         [ 7.0000,  8.0000]]])";
    assert_eq!(format!("{t}"), expected);
}

#[test]
fn test_is_contiguous() {
    // Newly created tensor is contiguous
    let t = Tensor::zeros([2, 3]).unwrap();
    assert!(t.is_contiguous());

    // Transposed tensor is NOT contiguous (strides become [1, 3] instead of [3, 1])
    let transposed = t.transpose(0, 1).unwrap();
    assert!(!transposed.is_contiguous());

    // Slicing along dimension 0 remains contiguous
    // (a contiguous block from offset to end)
    let slice_dim0 = t.slice(0, 0, 1).unwrap();
    assert!(slice_dim0.is_contiguous());

    // 4. Slicing along inner dimensions leaves memory gaps -> NOT contiguous
    let slice_dim1 = t.slice(1, 0, 2).unwrap();
    assert!(!slice_dim1.is_contiguous());
}
#[test]
fn test_slice_1d() -> Result<(), TensorError> {
    // Original: [10.0, 20.0, 30.0, 40.0, 50.0]
    let t = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0], [5])?;

    // Slice [1..4] -> [20.0, 30.0, 40.0]
    let s = t.slice(0, 1, 4)?;

    assert_eq!(s.shape(), &[3]);
    assert_eq!(s.offset(), 1);
    assert_eq!(s.strides(), &[1]);
    assert_eq!(format!("{s}"), "tensor([ 20.0000,  30.0000,  40.0000])");

    Ok(())
}

#[test]
fn test_slice_2d_dim0() -> Result<(), TensorError> {
    // 3x2 matrix:
    // [[ 1.0, 2.0 ],
    //  [ 3.0, 4.0 ],
    //  [ 5.0, 6.0 ]]
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = Tensor::from_vec(data, [3, 2])?;

    // Slice rows 1..3 -> [[3.0, 4.0], [5.0, 6.0]]
    let s = t.slice(0, 1, 3)?;

    assert_eq!(s.shape(), &[2, 2]);
    assert_eq!(s.offset(), 2); // row 1 starts at index 1 * stride[0] = 1 * 2 = 2
    assert_eq!(s.strides(), &[2, 1]); // strides remain unchanged

    let expected = "\
tensor([[ 3.0000,  4.0000],
        [ 5.0000,  6.0000]])";
    assert_eq!(format!("{s}"), expected);

    Ok(())
}

#[test]
fn test_slice_2d_dim1() -> Result<(), TensorError> {
    // 2x3 matrix:
    // [[ 1.0, 2.0, 3.0 ],
    //  [ 4.0, 5.0, 6.0 ]]
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = Tensor::from_vec(data, [2, 3])?;

    // Slice columns 1..3 -> columns 1 and 2 of each row
    // [[ 2.0, 3.0 ],
    //  [ 5.0, 6.0 ]]
    let s = t.slice(1, 1, 3)?;

    assert_eq!(s.shape(), &[2, 2]);
    assert_eq!(s.offset(), 1); // column 1 starts at index 1 * stride[1] = 1 * 1 = 1
    assert_eq!(s.strides(), &[3, 1]); // strides remain unchanged

    let expected = "\
tensor([[ 2.0000,  3.0000],
        [ 5.0000,  6.0000]])";
    assert_eq!(format!("{s}"), expected);

    Ok(())
}

#[test]
fn test_slice_chained() -> Result<(), TensorError> {
    // 3x3 matrix:
    // [[ 1.0, 2.0, 3.0 ],
    //  [ 4.0, 5.0, 6.0 ],
    //  [ 7.0, 8.0, 9.0 ]]
    let data: Vec<f32> = (1..=9).map(|x| x as f32).collect();
    let t = Tensor::from_vec(data, [3, 3])?;

    // Slice rows 1..3, then columns 1..3 -> bottom-right 2x2:
    // [[ 5.0, 6.0 ],
    //  [ 8.0, 9.0 ]]
    let s = t.slice(0, 1, 3)?.slice(1, 1, 3)?;

    assert_eq!(s.shape(), &[2, 2]);
    assert_eq!(s.offset(), 4); // row 1 (1*3=3) + col 1 (1*1=1) = offset 4

    let expected = "\
tensor([[ 5.0000,  6.0000],
        [ 8.0000,  9.0000]])";
    assert_eq!(format!("{s}"), expected);

    Ok(())
}

#[test]
fn test_slice_zero_copy_storage_sharing() -> Result<(), TensorError> {
    let t = Tensor::zeros([4, 4])?;
    let s = t.slice(0, 1, 3)?;

    // Both views point to the exact same physical Storage buffer
    assert!(Arc::ptr_eq(&t.storage, &s.storage));
    Ok(())
}

#[test]
fn test_slice_error_dim_out_of_bounds() -> Result<(), TensorError> {
    let t = Tensor::zeros([2, 3])?;

    // dim 2 is out of bounds for a 2D tensor (valid dims: 0, 1)
    let res = t.slice(2, 0, 1);
    assert_eq!(
        res.unwrap_err(),
        TensorError::IndexOutOfBounds {
            axis: 2,
            index: 2,
            size: 2,
        }
    );
    Ok(())
}

#[test]
fn test_slice_error_start_greater_than_end() -> Result<(), TensorError> {
    let t = Tensor::zeros([5])?;

    // start (3) > end (1) is an invalid range
    let res = t.slice(0, 3, 1);
    assert_eq!(
        res.unwrap_err(),
        TensorError::InvalidSliceRange {
            axis: 0,
            start: 3,
            end: 1,
            size: 5,
        }
    );
    Ok(())
}

#[test]
fn test_slice_error_end_out_of_bounds() -> Result<(), TensorError> {
    let t = Tensor::zeros([4])?;

    // end (5) exceeds dimension size (4)
    let res = t.slice(0, 1, 5);
    assert_eq!(
        res.unwrap_err(),
        TensorError::IndexOutOfBounds {
            axis: 0,
            index: 5,
            size: 4,
        }
    );
    Ok(())
}

#[test]
fn test_slice_error_zero_sized() -> Result<(), TensorError> {
    let t = Tensor::zeros([3, 3])?;

    // start == end results in length 0 (which torch-rs currently rejects)
    let res = t.slice(0, 1, 1);
    assert_eq!(
        res.unwrap_err(),
        TensorError::ZeroSizedDimension { shape: vec![0, 3] }
    );
    Ok(())
}

#[test]
fn test_numel() -> Result<(), TensorError> {
    let t0 = Tensor::from_vec(vec![42.0], [])?;
    assert_eq!(t0.numel(), 1);

    let t1 = Tensor::zeros([5])?;
    assert_eq!(t1.numel(), 5);

    let t2 = Tensor::zeros([2, 3])?;
    assert_eq!(t2.numel(), 6);

    let t3 = Tensor::zeros([2, 3, 4])?;
    assert_eq!(t3.numel(), 24);

    let s = t2.slice(0, 1, 2)?;
    assert_eq!(s.numel(), 3);

    Ok(())
}

#[test]
fn test_get() -> Result<(), TensorError> {
    let data = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
    let t = Tensor::from_vec(data, [2, 3])?;

    assert_eq!(t.get([0, 0])?, 10.0);
    assert_eq!(t.get([0, 1])?, 20.0);
    assert_eq!(t.get([0, 2])?, 30.0);
    assert_eq!(t.get([1, 0])?, 40.0);
    assert_eq!(t.get([1, 1])?, 50.0);
    assert_eq!(t.get([1, 2])?, 60.0);

    // Out of bounds check
    assert!(t.get([2, 0]).is_err());
    assert!(t.get([0, 3]).is_err());

    // Scalar get
    let scalar = Tensor::from_vec(vec![3.14], [])?;
    assert_eq!(scalar.get([])?, 3.14);

    Ok(())
}

#[test]
fn test_iterator_contiguous() -> Result<(), TensorError> {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = Tensor::from_vec(data.clone(), [2, 3])?;

    // Check size_hint and exact size
    let iter = t.iter();
    assert_eq!(iter.size_hint(), (6, Some(6)));

    // Elements collected in row-major order
    let collected: Vec<f32> = t.iter().collect();
    assert_eq!(collected, data);

    // IntoIterator test for &Tensor
    let mut values = Vec::new();
    for val in &t {
        values.push(val);
    }
    assert_eq!(values, data);

    Ok(())
}

#[test]
fn test_iterator_non_contiguous() -> Result<(), TensorError> {
    // 2x3 matrix:
    // [[ 1.0, 2.0, 3.0 ],
    //  [ 4.0, 5.0, 6.0 ]]
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = Tensor::from_vec(data, [2, 3])?;

    // Transposed to 3x2:
    // [[ 1.0, 4.0 ],
    //  [ 2.0, 5.0 ],
    //  [ 3.0, 6.0 ]]
    let transposed = t.transpose(0, 1)?;
    let collected: Vec<f32> = transposed.iter().collect();
    assert_eq!(collected, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

    // Slice columns 1..3 of 2x3 -> 2x2:
    // [[ 2.0, 3.0 ],
    //  [ 5.0, 6.0 ]]
    let sliced = t.slice(1, 1, 3)?;
    let collected_slice: Vec<f32> = sliced.iter().collect();
    assert_eq!(collected_slice, vec![2.0, 3.0, 5.0, 6.0]);

    Ok(())
}

#[test]
fn test_is_compact() -> Result<(), TensorError> {
    // Freshly created tensors are compact
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])?;
    assert!(t.is_compact());

    // Transpose changes strides -> not compact
    let transposed = t.transpose(0, 1)?;
    assert!(!transposed.is_compact());

    // Slice along dim 0: offset != 0 or trailing unused buffer -> not compact
    let row_slice = t.slice(0, 1, 2)?;
    assert!(!row_slice.is_compact());

    // Slice along dim 1: gaps in memory -> not compact
    let col_slice = t.slice(1, 0, 2)?;
    assert!(!col_slice.is_compact());

    Ok(())
}

#[test]
fn test_compaction() -> Result<(), TensorError> {
    // 1. Compaction on already compact tensor returns identical storage (shared Arc)
    let compact_t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2])?;
    let compacted_same = compact_t.compaction()?;
    assert!(compacted_same.is_compact());
    assert!(Arc::ptr_eq(&compact_t.storage, &compacted_same.storage));

    // 2. Compaction on non-contiguous transposed tensor creates new compact storage
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])?;
    let transposed = t.transpose(0, 1)?; // shape [3, 2]
    assert!(!transposed.is_compact());

    let compacted = transposed.compaction()?;
    assert!(compacted.is_compact());
    assert_eq!(compacted.shape(), &[3, 2]);
    assert_eq!(compacted.strides(), &[2, 1]);
    assert_eq!(compacted.offset(), 0);
    assert!(!Arc::ptr_eq(&transposed.storage, &compacted.storage));

    let expected = "\
tensor([[ 1.0000,  4.0000],
        [ 2.0000,  5.0000],
        [ 3.0000,  6.0000]])";
    assert_eq!(format!("{compacted}"), expected);

    // 3. Compaction on sliced tensor
    let sliced = t.slice(1, 1, 3)?; // 2x2 with offset 1 and strides [3, 1]
    assert!(!sliced.is_compact());

    let compacted_slice = sliced.compaction()?;
    assert!(compacted_slice.is_compact());
    assert_eq!(compacted_slice.shape(), &[2, 2]);
    assert_eq!(compacted_slice.strides(), &[2, 1]);
    assert_eq!(compacted_slice.offset(), 0);

    let expected_slice = "\
tensor([[ 2.0000,  3.0000],
        [ 5.0000,  6.0000]])";
    assert_eq!(format!("{compacted_slice}"), expected_slice);

    Ok(())
}
