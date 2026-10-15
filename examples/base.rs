use torch_rs::tensor::Tensor;

fn main() -> anyhow::Result<()> {
    let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2])?;
    println!("value: {:?}", tensor);
    Ok(())
}
