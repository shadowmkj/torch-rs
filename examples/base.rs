use mini_torch::tensor::Tensor;

fn main() {
    let tensor = Tensor::zeros(vec![2, 3]);
    dbg!(tensor);
}
