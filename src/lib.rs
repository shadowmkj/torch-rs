pub mod tensor;

#[macro_export]
macro_rules! linear_index {
    ($tensor:expr, $($idx:expr),+$(,)?) => {
        $tensor.linear_index(&[$($idx),+])
    };
}
