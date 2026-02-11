#pragma once


namespace raxis {

// template <typename scalar_t>
// inline void fill_inplace(Tensor& self, Scalar value_scalar) {
//     scalar_t value = value_scalar.to<scalar_t>();    }
//     scalar_t* dptr = static_cast<scalar_t*>(self.data_ptr());
//     *dptr = value;
// }

// Tensor scalar_fill() {
//     dtype_distpatch({
//         fill_inplace<scalar_t>(self, value)
//     })
// }

// Tensor scalar_tensor_static() {
//     result = empty_cpu();
//     scalar_fill()
// }

// inline Tensor scalar_to_tensor() {
//     if cpu:
//         scalar_arrat_static()  // OR
//     scalar_tensor()
// }

// inline Tensor wrapped_scalar_tensor() {
//     scalar_to_tensor();
//     // mark tensor as wrapped scalar value
// }

}


