#pragma once

#include <raxis/tensor/tensor_iterator.h>


namespace raxis::kernels::cpu {

void add_kernel(TensorIterator& iter);
void mul_kernel(TensorIterator& iter);
void sub_kernel(TensorIterator& iter);
void div_kernel(TensorIterator& iter);

}
