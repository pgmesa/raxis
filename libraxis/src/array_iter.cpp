
#include <vector>


// struct LoopPlan {
//     std::vector<size_t> shape
//     IntArray strides_a
//     IntArray strides_b
//     IntArray strides_out
//     size_t inner_dim
// }



// unsafe fn recursive_loop(
//     dim: usize,
//     inner: usize,
//     shape: *const isize,
//     sa: *const isize,
//     sb: *const isize,
//     so: *const isize,
//     mut a: *const f64,
//     mut b: *const f64,
//     mut out: *mut f64,
// ) {
//     unsafe {
//         let n = *shape.add(dim);

//         if dim == inner {
//             for _ in 0..n {
//                 *out = *a + *b;
//                 a = a.offset(*sa.add(dim));
//                 b = b.offset(*sb.add(dim));
//                 out = out.offset(*so.add(dim));
//             }
//         } else {
//             for _ in 0..n {
//                 recursive_loop(
//                     dim + 1,
//                     inner,
//                     shape,
//                     sa,
//                     sb,
//                     so,
//                     a,
//                     b,
//                     out,
//                 );

//                 a = a.offset(*sa.add(dim));
//                 b = b.offset(*sb.add(dim));
//                 out = out.offset(*so.add(dim));
//             }
//         }
//     }