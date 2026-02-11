#pragma once

#include <raxis/tensor/tensor_iterator.h>


namespace raxis::kernels::cpu {

void direct_copy_kernel(TensorIterator& iter);
void copy_kernel(TensorIterator& iter);

}