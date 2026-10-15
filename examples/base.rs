use torch_rs::tensor::Tensor;

fn main() -> anyhow::Result<()> {
    let tensor = Tensor::<3>::randn([10, 30, 100]).unwrap();

    println!("{:?}", tensor);
    Ok(())
}
