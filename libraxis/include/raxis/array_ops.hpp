#pragma once

#include <raxis/api.hpp>
#include <raxis/array.hpp>


namespace raxis {

RAXIS_API NDArray& copy(const NDArray& a, NDArray& out);
RAXIS_API NDArray clone(const NDArray& a);
RAXIS_API NDArray to(const NDArray& a, Dtype dtype);

RAXIS_API NDArray& add(const NDArray& a, const NDArray& b, NDArray& out);
RAXIS_API NDArray& sub(const NDArray& a, const NDArray& b, NDArray& out);
RAXIS_API NDArray& mul(const NDArray& a, const NDArray& b, NDArray& out);
RAXIS_API NDArray& div(const NDArray& a, const NDArray& b, NDArray& out);

RAXIS_API NDArray add(const NDArray& a, const NDArray& b);
RAXIS_API NDArray sub(const NDArray& a, const NDArray& b);
RAXIS_API NDArray mul(const NDArray& a, const NDArray& b);
RAXIS_API NDArray div(const NDArray& a, const NDArray& b);

// In-place
RAXIS_API NDArray& add_(NDArray& a, const NDArray& b);
RAXIS_API NDArray& sub_(NDArray& a, const NDArray& b);
RAXIS_API NDArray& mul_(NDArray& a, const NDArray& b);
RAXIS_API NDArray& div_(NDArray& a, const NDArray& b);

// Reversed
RAXIS_API NDArray rsub(const NDArray& a, const NDArray& b);
RAXIS_API NDArray rdiv(const NDArray& a, const NDArray& b);

RAXIS_API NDArray& rsub_(NDArray& a, const NDArray& b);
RAXIS_API NDArray& rdiv_(NDArray& a, const NDArray& b);

}