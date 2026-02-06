#pragma once

#include <raxis/core/array_iterator.hpp>


namespace raxis::kernels::cpu {

void add_kernel(ArrayIterator& iter);
void mul_kernel(ArrayIterator& iter);
void sub_kernel(ArrayIterator& iter);
void div_kernel(ArrayIterator& iter);

}
