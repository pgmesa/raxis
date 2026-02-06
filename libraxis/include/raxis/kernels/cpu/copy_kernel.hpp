#pragma once

#include <raxis/core/array_iterator.hpp>


namespace raxis::kernels::cpu {

void direct_copy_kernel(ArrayIterator& iter);
void copy_kernel(ArrayIterator& iter);

}