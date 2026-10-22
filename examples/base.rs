use torch_rs::tensor::Tensor;

fn main() -> anyhow::Result<()> {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2])?;

    let random = Tensor::normal(100f32, 20f32, [2, 2, 12, 4, 8])?;

    let reshaped = random.reshape([32, 12, 4])?;
    let mean = random.iter().sum::<f32>() / random.numel() as f32;

    // println!("base = {}", random);
    // println!("reshaped = {}", reshaped);
    // println!("linear = {}", linear);
    println!("mean = {}", mean);
    for i in &random {
        println!("{}", i);
    }

    Ok(())
}
