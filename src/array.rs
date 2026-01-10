
use std::ops::Add;
use std::sync::Arc;
use std::cell::UnsafeCell;

#[derive(Debug)]
struct Storage<Dtype> {
    data: UnsafeCell<Vec<Dtype>>
}

// unsafe impl<T: Send> Send for Storage<T> {}
// unsafe impl<T: Sync> Sync for Storage<T> {}

#[derive(Debug)]
pub struct NDArray<Dtype> {
    storage: Arc<Storage<Dtype>>,
    shape: Vec<isize>,
    strides: Vec<isize>,
    offset: isize
}


pub fn full<Dtype>(shape: &[isize], value: Dtype) -> NDArray<Dtype>
where
    Dtype: Clone,  // Need Clone to duplicate value
{
    let n = shape.iter().product::<isize>() as usize;
    let vec = vec![value; n];
    
    NDArray::new(vec, shape.to_vec())
}

impl<Dtype> NDArray<Dtype> {

    pub fn new(data: Vec<Dtype>, shape: Vec<isize>) -> Self {
        let strides = compute_strides(&shape);
        let storage = Arc::new(Storage { data: UnsafeCell::new(data) });

        NDArray { storage, shape, strides, offset: 0 }
    }
    
    pub fn len(&self) -> isize {
        self.shape.first().copied().unwrap()
    }

    // pub fn storage(&self) -> Arc<Storage<Dtype>> {
    //     self.storage.clone()
    // }

    pub fn shape(&self) -> &Vec<isize> {
        &self.shape
    }

    pub fn strides(&self) -> &Vec<isize> {
        &self.strides
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn numel(&self) -> isize {
        self.shape.iter().fold(1, |acc, dim| acc * dim)
    }

    // pub fn slice(&self, s: &Slice) -> NDArray<Dtype> {
    //     assert!(self.shape.len() == 1);
    //     assert!(s.start < s.end);
    //     assert!(s.end <= self.shape[0]);
    //     assert!(s.step > 0);

    //     let new_len = (s.end - s.start + s.step - 1) / s.step;

    //     NDArray {
    //         storage: self.storage.clone(),
    //         offset: self.offset + s.start * self.strides[0] as usize,
    //         shape: vec![new_len],
    //         strides: vec![self.strides[0] * s.step as isize],
    //     }
    // }

    // pub fn slice(&self, slices: &[Slice]) -> NDArray<Dtype> {
    //     assert!(slices.len() <= self.ndim());
        
    //     // assert!(s.start < s.end);
    //     // assert!(s.end <= self.shape[0]);
    //     // assert!(s.step > 0);

    //     let view_shape = Vec::new();
    //     let view_strides = Vec::new();
    //     let view_offsets = Vec::new();



    //     let new_len = (s.end - s.start + s.step - 1) / s.step;

    //     NDArray {
    //         storage: self.storage.clone(),

    //         shape: vec![new_len],
    //         strides: vec![self.strides[0] * s.step as isize],
    //         offset: self.offset + s.start * self.strides[0] as usize,
    //     }
    // }

    // pub fn get(&self, index: usize) -> Dtype
    // where
    //     Dtype: Copy,
    // {
    //     assert!(self.shape.len() == 1);
    //     assert!(index < self.shape[0]);

    //     unsafe {
    //         let data = &*self.storage.data.get();
    //         let idx = self.offset + index * self.strides[0] as usize;
    //         data[idx]
    //     }
    // }

    // pub fn set(&self, index: usize, value: Dtype) {
    //     assert!(self.shape.len() == 1);
    //     assert!(index < self.shape[0]);

    //     unsafe {
    //         let data = &mut *self.storage.data.get();
    //         let idx = self.offset + index * self.strides[0] as usize;
    //         data[idx] = value;
    //     }
    // }

}


// pub struct Slice {
//     pub start: usize,
//     pub end: usize,
//     pub step: usize
// }

// impl Slice {

//     pub fn start(&self) -> usize {
//         self.start
//     }

//     pub fn end(&self) -> usize {
//         self.end
//     }

//     pub fn step(&self) -> usize {
//         self.step
//     }
// }


// pub fn broadcast<Dtype>(array: &NDArray<Dtype>, shape: &[usize]) -> NDArray<Dtype> {
//     let target_ndim = shape.len();
//     let array_ndim = array.ndim();
//     assert!(target_ndim >= array.ndim());

//     let mut new_strides = vec![0; target_ndim];
//     let offset = target_ndim - array_ndim;

//     for i in (0..target_ndim).rev() {
//         // Dimension doesn't exist
//         if i < offset { continue; }
        
//         let arr_idx = i - offset;
//         let current_dim = array.shape[arr_idx];
        
//         // Broadcast dimension
//         if current_dim == 1 { continue; }

//         assert_eq!(current_dim, shape[i], "Incompatible shapes for broadcasting");
//         new_strides[i] = array.strides[arr_idx];
//     }

//     NDArray { 
//         storage: array.storage.clone(),
//         shape: shape.to_vec(),
//         strides: new_strides, 
//         offset: array.offset
//     }
// }


impl<Dtype> Add for &NDArray<Dtype>
where
    Dtype: Add<Output = Dtype> + Copy,
{
    type Output = NDArray<Dtype>;
    
    fn add(self, rhs: &NDArray<Dtype>) -> Self::Output {
        let mut result_shape = broadcast_shapes(&self.shape, &rhs.shape);
        let mut strides_a = broadcast_strides(&self.shape, &self.strides, &result_shape);
        let mut strides_b = broadcast_strides(&rhs.shape, &rhs.strides, &result_shape);
        let mut strides_out = compute_strides(&result_shape);

        println!("shape_out={:?} sa={:?} sb={:?} so={:?}", &result_shape, &strides_a, &strides_b, &strides_out);
        
        // Collapse dimensions
        collapse_dims(&mut result_shape, &mut strides_a, &mut strides_b, &mut strides_out);

        println!("shape_out={:?} sa={:?} sb={:?} so={:?}", &result_shape, &strides_a, &strides_b, &strides_out);

        let plan = LoopPlan {
            shape: result_shape.clone(), strides_a, strides_b, strides_out, inner_dim: (result_shape.len() - 1)
        };
        
        let out_container;
        unsafe {
            let a_data: *const f64 = (&(*self.storage.data.get())).as_ptr() as *const f64;
            let b_data: *const f64 = (&(*rhs.storage.data.get())).as_ptr() as *const f64;
            out_container = execute_add(plan, a_data, b_data);
        }

        // OLD
        // let n = result_shape.iter().product::<isize>();
        // let mut result_data = Vec::with_capacity(n as usize);
        
        // unsafe {
        //     let a_data: *const f64 = (&(*self.storage.data.get())).as_ptr() as *const f64;
        //     let b_data: *const f64 = (&(*rhs.storage.data.get())).as_ptr() as *const f64;
        //     let out_data: *mut f64 = result_data.as_mut_ptr();
            
        //     for flat_idx in 0..n {
        //         // Convert flat index to multi-dimensional indices
        //         let mut idx = flat_idx;
        //         let mut a_offset = self.offset;
        //         let mut b_offset = rhs.offset;
                
        //         for dim in (0..result_shape.len()).rev() {
        //             let coord = idx % result_shape[dim];
        //             idx /= result_shape[dim];
                    
        //             a_offset += coord * strides_a[dim];
        //             b_offset += coord * strides_b[dim];
        //         }
                
        //         *out_data.add(flat_idx as usize) = *a_data.add(a_offset as usize) + *b_data.add(b_offset as usize);
        //     }
            
        //     result_data.set_len(n as usize);
        // }

        let n = result_shape.iter().product::<isize>() as usize;
        let fake_container = Vec::with_capacity(n);
        
        NDArray::new(fake_container, result_shape.to_vec())
        
    }
}


// ========================================================================

pub fn broadcast_shapes(shape1: &[isize], shape2: &[isize]) -> Vec<isize> {
    let ndim1 = shape1.len();
    let ndim2 = shape2.len();
    let target_ndim = ndim1.max(ndim2);
    
    let mut result_shape = Vec::with_capacity(target_ndim);
    
    for i in 0..target_ndim {
        // Access dimensions from right to left
        let dim1 = if i < ndim1 {
            shape1[ndim1 - 1 - i]
        } else {
            1
        };
        
        let dim2 = if i < ndim2 {
            shape2[ndim2 - 1 - i]
        } else {
            1
        };
        
        // Broadcasting rules
        let result_dim = if dim1 == dim2 {
            dim1
        } else if dim1 == 1 {
            dim2
        } else if dim2 == 1 {
            dim1
        } else {
            panic!(
                "Shapes {:?} and {:?} are not broadcastable at dimension {}",
                shape1, shape2, i
            );
        };
        
        result_shape.push(result_dim);
    }
    
    // Reverse since we built it right-to-left
    result_shape.reverse();
    result_shape
}


fn compute_strides(shape: &[isize]) -> Vec<isize> {
    let mut strides: Vec<isize> = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}


fn broadcast_strides(
    shape: &[isize],
    strides: &[isize],
    result_shape: &[isize],
) -> Vec<isize> {
    let ndim = result_shape.len();
    let mut out = vec![0; ndim];

    let offset = ndim - shape.len();

    for i in 0..ndim {
        if i < offset {
            out[i] = 0;
        } else if shape[i - offset] == 1 {
            out[i] = 0;
        } else {
            out[i] = strides[i - offset];
        }
    }
    out
}


fn can_collapse_stride(s_prev: isize, s_cur: isize, dim: isize) -> bool {
    s_prev == s_cur * dim || (s_prev == 0 && s_cur == 0)
}

fn collapse_dims(
    shape: &mut Vec<isize>,
    sa: &mut Vec<isize>,
    sb: &mut Vec<isize>,
    so: &mut Vec<isize>,
) {
    let mut i = shape.len() - 1;

    while i > 0 {
        let can_collapse =
            can_collapse_stride(sa[i-1], sa[i], shape[i]) &&
            can_collapse_stride(sb[i-1], sb[i], shape[i]) &&
            can_collapse_stride(so[i-1], so[i], shape[i]);

        if can_collapse {
            shape[i - 1] *= shape[i];
            
            // replace outer stride with inner stride
            sa[i - 1] = sa[i];
            sb[i - 1] = sb[i];
            so[i - 1] = so[i];

            // remove inner dimension
            shape.remove(i);
            sa.remove(i);
            sb.remove(i);
            so.remove(i);
        }
        i -= 1;
    }
}


struct LoopPlan {
    shape: Vec<isize>,               // iteration shape
    strides_a: Vec<isize>,            // element strides
    strides_b: Vec<isize>,
    strides_out: Vec<isize>,
    inner_dim: usize,                 // last dim is inner loop
}


unsafe fn execute_add(plan: LoopPlan, a: *const f64, b: *const f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(plan.shape.iter().product::<isize>() as usize);
    let out_ptr: *mut f64 = out.as_mut_ptr();

    println!("{}", plan.inner_dim);
    

    unsafe {
        recursive_loop(
            0,
            plan.inner_dim,
            plan.shape.as_ptr(),
            plan.strides_a.as_ptr(),
            plan.strides_b.as_ptr(),
            plan.strides_out.as_ptr(),
            a,
            b,
            out_ptr,
        );

        out.set_len(plan.shape.iter().product::<isize>() as usize);
    }

    out
}

unsafe fn recursive_loop(
    dim: usize,
    inner: usize,
    shape: *const isize,
    sa: *const isize,
    sb: *const isize,
    so: *const isize,
    mut a: *const f64,
    mut b: *const f64,
    mut out: *mut f64,
) {
    unsafe {
        let n = *shape.add(dim);

        if dim == inner {
            for _ in 0..n {
                *out = *a + *b;
                a = a.offset(*sa.add(dim));
                b = b.offset(*sb.add(dim));
                out = out.offset(*so.add(dim));
            }
        } else {
            for _ in 0..n {
                recursive_loop(
                    dim + 1,
                    inner,
                    shape,
                    sa,
                    sb,
                    so,
                    a,
                    b,
                    out,
                );

                a = a.offset(*sa.add(dim));
                b = b.offset(*sb.add(dim));
                out = out.offset(*so.add(dim));
            }
        }
    }
}
