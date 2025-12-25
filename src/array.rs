
use std::ops::Add;
use std::sync::Arc;

#[derive(Debug)]
pub struct NDArray<Dtype> {
    data: Arc<Vec<Dtype>>,
    strides: Vec<usize>,
    shape: Vec<usize>
}

// fn broadcast<Dtype>(array: NDArray<Dtype>) -> NDArray<Dtype> {
    
// }

impl<Dtype> NDArray<Dtype> {

    pub fn new(data: Vec<Dtype>, shape: Vec<usize>) -> Self {
        let strides = Self::compute_strides(&shape);
        NDArray { data: Arc::new(data), strides, shape }
    }

    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }
    
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn data(&self) -> &Arc<Vec<Dtype>> {
        &self.data
    }

    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub fn strides(&self) -> &Vec<usize> {
        &self.strides
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().fold(1, |acc, dim| acc * dim)
    }

}

impl<Dtype> Add for &NDArray<Dtype>
where
    Dtype: Add<Output = Dtype> + Copy,
{
    type Output = NDArray<Dtype>;
    
    fn add(self, rhs: &NDArray<Dtype>) -> Self::Output {
        assert_eq!(self.shape, rhs.shape);

        let result = self.data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| *a + *b)
            .collect();

        let data = Arc::new(result);
        
        NDArray {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }
}