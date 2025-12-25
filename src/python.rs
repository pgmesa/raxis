
use pyo3::prelude::*;
use numpy::PyArrayMethods;

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

    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        use numpy::PyArray;
        
        let data = (*self.inner.data()).clone();
        let shape: Vec<usize> = self.inner.shape().clone();
        
        // Create numpy array with specific shape
        let arr = PyArray::from_vec(py, data.to_vec()).reshape(shape)?;
        
        Ok(arr.into())
    }

    fn __add__(&self, other: &PyNDArray) -> Self {
        PyNDArray {
            inner: &self.inner + &other.inner
        }
    }
    
    fn __repr__<'py>(&self, py: Python<'py>) -> PyResult<String> {
        let numpy_array = self.to_numpy(py)?;
        
        // Call NumPy's __repr__ method
        let repr = numpy_array.call_method0(py, "__str__")?;
        let repr_str: String = repr.extract(py)?;
        
        Ok(repr_str)
    }
}

#[pymodule]
#[pyo3(name = "_rust")]
fn raxis(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNDArray>()?;
    Ok(())
}