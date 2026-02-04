#pragma once

#include <raxis/core/array_iterator.hpp>


namespace raxis::kernels::cpu {

void add_kernel(const ArrayIterator& iter);
void mul_kernel(const ArrayIterator& iter);
void sub_kernel(const ArrayIterator& iter);
void div_kernel(const ArrayIterator& iter);

}
