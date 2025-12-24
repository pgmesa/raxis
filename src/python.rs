
use pyo3::prelude::*;
use crate::array::NDArray;

#[pyclass(name = "NDArray")]
pub struct PyNDArray {
    inner: NDArray<f64>,
}

#[pymethods]
impl PyNDArray {
    #[new]
    fn new(data: Vec<f64>, shape: Vec<usize>) -> Self {
        PyNDArray {
            inner: NDArray::new(data, shape),
        }
    }

    fn __add__(&self, other: &PyNDArray) -> Self {
        PyNDArray {
            inner: &self.inner + &other.inner
        }
    }
    
    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

#[pymodule]
#[pyo3(name = "_rust")]
fn raxis(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNDArray>()?;
    Ok(())
}