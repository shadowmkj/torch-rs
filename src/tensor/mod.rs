use std::sync::Arc;

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

#[derive(Debug, Clone)]
pub struct Tensor {
    storage: Arc<Storage>,
    shape: Vec<usize>,
    offset: usize,
    strides: Vec<usize>,
    device: Device,
}

impl Tensor {
    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        if shape.contains(&0) {
            panic!("Error: invalid shape") //TODO: Implement Error
        }

        if shape.is_empty() {
            return vec![];
        }

        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        } //NOTE: refer ./tasks/20261014-021101/strides.png

        strides
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let strides = Self::compute_strides(&shape);

        // let device = device.unwrap_or_default(); //uncomment this
        // For now just use CPU as default
        let device = Device::default(); //TODO: Must change this back when designing the public API
        let num_elements: usize = shape.iter().product();
        let storage = match device {
            Device::Cpu => Storage::Cpu(vec![0.0; num_elements]),
            //TODO: Update once more storage options are implemented
        };

        Self {
            storage: Arc::new(storage),
            shape,
            offset: 0,
            strides,
            device,
        }
    }
}

#[test]
fn basic_tensor() {
    let tensor = Tensor::zeros(vec![2, 3]);
    assert_eq!(tensor.shape, vec![2, 3]);
    assert_eq!(tensor.offset, 0);
    assert_eq!(tensor.strides, vec![3, 1]);
}

#[test]
fn test_compute_strides_dimensions() {
    assert_eq!(Tensor::compute_strides(&[]), Vec::<usize>::new());
    assert_eq!(Tensor::compute_strides(&[5]), vec![1]);
    assert_eq!(Tensor::compute_strides(&[2, 3]), vec![3, 1]);
    assert_eq!(Tensor::compute_strides(&[2, 3, 4]), vec![12, 4, 1]);
}

#[test]
fn test_zero_volume_tensor() {
    //NOTE: Must fail for now
    //since no support for 0 dimension shapes
    let t = Tensor::zeros(vec![2, 0, 3]);
    assert_eq!(t.shape, vec![2, 0, 3]);
    assert_eq!(t.strides, vec![0, 3, 1]);
}

#[test]
fn test_storage_sharing_on_view() {
    let t1 = Tensor::zeros(vec![2, 3]);
    let t2 = t1.clone(); // Shallow handle clone
    assert!(Arc::ptr_eq(&t1.storage, &t2.storage));
}
