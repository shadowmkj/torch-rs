use mini_torch::{
    linear_index,
    tensor::{Tensor, error::TensorError},
};

fn main() -> Result<(), TensorError> {
    let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
    let _ = match linear_index!(tensor, 1, 3,) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("ERROR: {e}");
            panic!()
        }
    };

    Ok(())
}
