use torch_rs::tensor::Tensor;

fn main() -> anyhow::Result<()> {
    let tensor = Tensor::randn([2, 2, 2]).unwrap();
    let transposed = tensor.transpose(0, 1)?;
    println!("Original:\n{tensor}\n");
    println!("Transposed:\n{transposed}");

    println!("Original:\n{:?}\n", tensor);
    println!("Transposed:\n{:?}", tensor);
    Ok(())
}
