#include <raxis/tensor/tensor_options.h>

#include <ostream>


namespace raxis {

std::ostream& operator<<(std::ostream& stream, const TensorOptions& options) {
    stream << "TensorOptions("
           << "dtype=" << options.dtype()
           << (options.has_dtype() ? "" : " (default)")
           << ", device=" << options.device()
           << (options.has_device() ? "" : " (default)")
           << ", requires_grad=" << std::boolalpha << options.requires_grad()
           << (options.has_requires_grad() ? "" : " (default)")
           << ")";
    return stream;
}

} // namespace raxis