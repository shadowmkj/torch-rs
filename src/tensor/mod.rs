pub mod error;

use crate::tensor::error::TensorError;
use std::sync::Arc;

#[derive(Debug, Copy, Clone, Default)]
pub enum Device {
    #[default]
    Cpu,
}

#[derive(Debug, Clone)]
pub enum Storage {
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

    pub fn linear_index(&self, indices: [usize; D]) -> Result<usize, TensorError> {
        let mut flat_index = self.offset;
        for i in 0..D {
            if indices[i] >= self.shape[i] {
                return Err(TensorError::IndexOutOfBounds {
                    axis: i,
                    index: indices[i],
                    size: self.shape[i],
                });
            }
            flat_index += indices[i] * self.strides[i];
        }

        Ok(flat_index)
    }

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
            shape: shape,
            offset: 0,
            strides,
            device,
        })
    }

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
