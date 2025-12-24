
use std::vec;

use raxis::NDArray;


fn test_array() {
    let data: Vec<f32> = vec![2.0, 2., 1.3, 4.];
    let shape : Vec<usize> = vec![2,2];
    let arr1 = NDArray::new(data, shape);

    let data: Vec<f32> = vec![1., 1.,  1., 1.];
    let shape : Vec<usize> = vec![2,2];
    let arr2 = NDArray::new(data, shape);

    let arr3 = &arr1 + &arr2;

    println!("{:?}", arr1);
    println!("{:?}", arr2);
    println!("=================================");
    println!("{:?}", arr3);
}


fn main() {
    test_array();
}