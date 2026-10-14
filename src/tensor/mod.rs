pub mod error;

use crate::{linear_index, tensor::error::TensorError};
use std::{ops::Index, sync::Arc};

#[derive(Debug, Copy, Clone, Default)]
pub enum Device {
    #[default]
    Cpu,
}

#[derive(Debug, Clone)]
pub enum Storage {
    Cpu(Vec<f32>),
    //TODO: add more storage (Metal, CUDA, wgpu)
}

impl PartialEq<Vec<f32>> for Storage {
    fn eq(&self, other: &Vec<f32>) -> bool {
        match self {
            Storage::Cpu(items) => items == other,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Tensor {
    storage: Arc<Storage>,
    shape: Vec<usize>,
    offset: usize,
    strides: Vec<usize>,
    device: Device,
}

impl Tensor {
    fn compute_strides(shape: &[usize]) -> Result<Vec<usize>, TensorError> {
        if shape.contains(&0) {
            return Err(TensorError::InvalidShape);
        }

        if shape.is_empty() {
            return Ok(vec![]);
        }

        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        } //NOTE: refer ./tasks/20261014-021101/strides.png

        Ok(strides)
    }

    pub fn linear_index(&self, indices: &[usize]) -> Result<usize, TensorError> {
        if indices.len() != self.shape.len() {
            return Err(TensorError::InvalidShape);
        }

        let mut flat_index = self.offset;
        for (idx, (&i, &s)) in indices.iter().zip(&self.strides).enumerate() {
            if i >= self.shape[idx] {
                return Err(TensorError::IndexOutOfBounds {
                    limit: self.shape[idx] - 1,
                    index: i,
                });
            }
            flat_index += i * s;
        }

        Ok(flat_index)
    }

    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Result<Self, TensorError> {
        let strides = Self::compute_strides(&shape)?;
        // let device = device.unwrap_or_default(); //uncomment this
        // For now just use CPU as default
        let device = Device::default(); //TODO: Must change this back when designing the public API
        let num_elements: usize = shape.iter().product();
        let storage = match device {
            Device::Cpu => Storage::Cpu(data),
            //TODO: Update once more storage options are implemented
        };

        Ok(Self {
            storage: Arc::new(storage),
            shape,
            offset: 0,
            strides,
            device,
        })
    }

    pub fn zeros(shape: Vec<usize>) -> Result<Self, TensorError> {
        let strides = Self::compute_strides(&shape)?;

        // let device = device.unwrap_or_default(); //uncomment this
        // For now just use CPU as default
        let device = Device::default(); //TODO: Must change this back when designing the public API
        let num_elements: usize = shape.iter().product();
        let storage = match device {
            Device::Cpu => Storage::Cpu(vec![0.0; num_elements]),
            //TODO: Update once more storage options are implemented
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
fn basic_tensor() {
    let tensor = Tensor::zeros(vec![2, 3]).unwrap();
    assert_eq!(tensor.shape, vec![2, 3]);
    assert_eq!(tensor.offset, 0);
    assert_eq!(tensor.strides, vec![3, 1]);
}

#[test]
fn test_compute_strides_dimensions() -> Result<(), TensorError> {
    assert_eq!(Tensor::compute_strides(&[])?, Vec::<usize>::new());
    assert_eq!(Tensor::compute_strides(&[5])?, vec![1]);
    assert_eq!(Tensor::compute_strides(&[2, 3])?, vec![3, 1]);
    assert_eq!(Tensor::compute_strides(&[2, 3, 4])?, vec![12, 4, 1]);

    Ok(())
}

#[test]
fn test_zero_volume_tensor() {
    let t = Tensor::zeros(vec![2, 0, 3]);
    assert!(matches!(t, Err(TensorError::InvalidShape)));
}

#[test]
fn test_tensor_from_vec() -> Result<(), TensorError> {
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::from_vec(data, vec![2, 3])?;
    assert_eq!(tensor.strides, vec![3, 1]);
    assert_eq!(*tensor.storage, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    Ok(())
}

#[test]
fn tensor_linear_indexing() -> Result<(), TensorError> {
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::from_vec(data, vec![2, 3])?;
    assert_eq!(tensor.linear_index(&[1, 2])?, 5);
    assert_eq!(linear_index!(tensor, 1, 2)?, 5);
    assert_eq!(linear_index!(tensor, 1, 4)?, 5);
    Ok(())
}

#[test]
fn test_storage_sharing_on_view() -> Result<(), TensorError> {
    let t1 = Tensor::zeros(vec![2, 3])?;
    let t2 = t1.clone(); // Shallow handle clone
    assert!(Arc::ptr_eq(&t1.storage, &t2.storage));

    Ok(())
}
