#pragma once

#include <raxis/api.hpp>
#include <raxis/array.hpp>


namespace raxis {

RAXIS_API NDArray add(const NDArray& a, const NDArray& b);
RAXIS_API NDArray sub(const NDArray& a, const NDArray& b);
RAXIS_API NDArray mul(const NDArray& a, const NDArray& b);
RAXIS_API NDArray div(const NDArray& a, const NDArray& b);

}