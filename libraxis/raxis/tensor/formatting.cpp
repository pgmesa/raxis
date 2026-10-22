
#include <raxis/tensor/formatting.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <iterator>
#include <ostream>
#include <string>
#include <vector>

#include <raxis/core/dtype.h>

#include <raxis/tensor/tensor.h>


namespace raxis {

namespace {

enum class FormatType {
    Default,     // '{:g}', used for integer values
    Scientific,  // '{:.4e}'
    Fixed        // '{:.4f}'
};

struct PrintFormat {
    double scale;
    int width;
    FormatType type;
};

// One format for all the values: integers when every finite value is
// integral, otherwise 4 decimals, scaled by a power of 10 when the values are
// all large or all small, or scientific when their magnitudes span more than
// 4 orders. expMin and expMax count the digits of the integer part of the
// smallest and largest finite magnitudes.
// From: aten/src/ATen/core/Formatting.cpp (__printFormat)
PrintFormat print_format(const double* data, int64_t size) {
    if (size == 0) {
        return {1.0, 0, FormatType::Default};
    }

    bool int_mode = true;
    for (int64_t i = 0; i < size; ++i) {
        if (std::isfinite(data[i]) && data[i] != std::ceil(data[i])) {
            int_mode = false;
            break;
        }
    }

    int64_t offset = 0;
    while (offset < size && !std::isfinite(data[offset])) {
        ++offset;
    }

    double exp_min = 1;
    double exp_max = 1;
    if (offset != size) {
        exp_min = std::fabs(data[offset]);
        exp_max = std::fabs(data[offset]);
        for (int64_t i = offset; i < size; ++i) {
            const double z = std::fabs(data[i]);
            if (std::isfinite(z)) {
                exp_min = std::min(exp_min, z);
                exp_max = std::max(exp_max, z);
            }
        }
        exp_min = exp_min != 0 ? std::floor(std::log10(exp_min)) + 1 : 1;
        exp_max = exp_max != 0 ? std::floor(std::log10(exp_max)) + 1 : 1;
    }

    if (int_mode) {
        if (exp_max > 9) {
            return {1.0, 11, FormatType::Scientific};
        }
        return {1.0, static_cast<int>(exp_max) + 1, FormatType::Default};
    }
    if (exp_max - exp_min > 4) {
        const int width = std::fabs(exp_max) > 99 || std::fabs(exp_min) > 99 ? 12 : 11;
        return {1.0, width, FormatType::Scientific};
    }
    if (exp_max > 5 || exp_max < 0) {
        return {std::pow(10, exp_max - 1), 7, FormatType::Fixed};
    }
    return {1.0, exp_max == 0 ? 7 : static_cast<int>(exp_max) + 6, FormatType::Fixed};
}

void print_value(std::ostream& stream, double value, const PrintFormat& format) {
    auto out = std::ostreambuf_iterator<char>(stream);
    const double scaled = value / format.scale;
    switch (format.type) {
        case FormatType::Default:
            std::format_to(out, "{:>{}g}", scaled, format.width);
            break;
        case FormatType::Scientific:
            std::format_to(out, "{:>{}.4e}", scaled, format.width);
            break;
        case FormatType::Fixed:
            std::format_to(out, "{:>{}.4f}", scaled, format.width);
            break;
    }
}

// A row-major rows x cols matrix, in blocks of as many columns as fit in
// linesize. The quirks (no line break after "Columns a to b", one extra
// space on the rows after the first when scaled) are libtorch's.
// From: aten/src/ATen/core/Formatting.cpp (__printMatrix)
void print_matrix(std::ostream& stream, const double* data, int64_t rows, int64_t cols, int64_t linesize, int64_t indent) {
    const PrintFormat format = print_format(data, rows * cols);
    const std::string pad(static_cast<size_t>(indent), ' ');
    const int64_t columns_per_line = std::max<int64_t>(1, (linesize - indent) / (format.width + 1));

    for (int64_t first = 0; first < cols; first += columns_per_line) {
        const int64_t last = std::min(first + columns_per_line, cols) - 1;
        if (columns_per_line < cols) {
            if (first != 0) {
                stream << '\n';
            }
            stream << "Columns " << first + 1 << " to " << last + 1 << pad;
        }
        if (format.scale != 1) {
            stream << std::format("{}", format.scale) << " *\n" << pad;
        }
        for (int64_t r = 0; r < rows; ++r) {
            for (int64_t c = first; c <= last; ++c) {
                print_value(stream, data[r * cols + c], format);
                if (c != last) {
                    stream << ' ';
                    continue;
                }
                stream << '\n';
                if (r != rows - 1) {
                    stream << pad << (format.scale != 1 ? " " : "");
                }
            }
        }
    }
}

// One "(i,j,.,.) = " matrix per index of the leading dimensions, 1-based.
// The first dimension varies fastest (column-major order, from Lua Torch).
// From: aten/src/ATen/core/Formatting.cpp (__printTensor)
void print_nd(std::ostream& stream, const double* data, IntArrayRef shape, int64_t linesize) {
    const int64_t ndim = static_cast<int64_t>(shape.size());
    const int64_t rows = shape[ndim - 2];
    const int64_t cols = shape[ndim - 1];
    std::vector<int64_t> counter(static_cast<size_t>(ndim - 2), 0);

    for (bool first = true;; first = false) {
        if (!first) {
            stream << '\n';
        }
        int64_t matrix = 0;
        stream << '(';
        for (int64_t i = 0; i < ndim - 2; ++i) {
            matrix = matrix * shape[i] + counter[i];
            stream << counter[i] + 1 << ',';
        }
        stream << ".,.) = \n";
        print_matrix(stream, data + matrix * rows * cols, rows, cols, linesize, 1);

        int64_t i = 0;
        for (; i < ndim - 2; ++i) {
            if (++counter[i] < shape[i]) {
                break;
            }
            counter[i] = 0;
        }
        if (i == ndim - 2) {
            break;
        }
    }
}

// libtorch's type names, e.g. CPUFloatType
std::string type_string(const Tensor& tensor) {
    const Dtype dtype = tensor.dtype();
    const char* scalar =
        dtype == dtypes::Bool    ? "Bool" :
        dtype == dtypes::Uint8   ? "Byte" :
        dtype == dtypes::Int32   ? "Int" :
        dtype == dtypes::Int64   ? "Long" :
        dtype == dtypes::Float32 ? "Float" :
        dtype == dtypes::Float64 ? "Double" : "Undefined";
    return std::format("{}{}Type", tensor.is_cuda() ? "CUDA" : "CPU", scalar);
}

} // namespace

std::ostream& print(std::ostream& stream, const Tensor& tensor_, int64_t linesize) {
    if (!tensor_.defined()) {
        return stream << "[ Tensor (undefined) ]";
    }

    const Tensor tensor = tensor_.to(dtypes::Float64).contiguous();
    const double* data = tensor.data_ptr<double>();
    const IntArrayRef shape = tensor.shape();
    const int64_t numel = tensor.numel();

    if (tensor.ndim() == 0) {
        stream << std::format("{}", data[0]) << "\n[ " << type_string(tensor_) << "{}";
    } else if (tensor.ndim() == 1) {
        if (numel > 0) {
            const PrintFormat format = print_format(data, numel);
            if (format.scale != 1) {
                stream << std::format("{}", format.scale) << " *\n";
            }
            for (int64_t i = 0; i < numel; ++i) {
                print_value(stream, data[i], format);
                stream << '\n';
            }
        }
        stream << "[ " << type_string(tensor_) << '{' << shape[0] << '}';
    } else {
        if (numel > 0 && tensor.ndim() == 2) {
            print_matrix(stream, data, shape[0], shape[1], linesize, 0);
        } else if (numel > 0) {
            print_nd(stream, data, shape, linesize);
        }
        stream << "[ " << type_string(tensor_) << '{';
        for (size_t i = 0; i < shape.size(); ++i) {
            stream << (i == 0 ? "" : ",") << shape[i];
        }
        stream << '}';
    }
    return stream << " ]";
}

std::ostream& operator<<(std::ostream& stream, const Tensor& tensor) {
    return print(stream, tensor);
}

} // namespace raxis
