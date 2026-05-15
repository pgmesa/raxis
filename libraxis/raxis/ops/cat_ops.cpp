
#include <raxis/ops/cat_ops.h>

#include <raxis/ops/dispatch_stub.h>

#include <raxis/util/exception.h>

#include <raxis/tensor/tensor.h>
#include <raxis/tensor/factories.h>
#include <raxis/tensor/tensor_iterator.h>
#include <raxis/tensor/tensor_options.h>


namespace raxis {

namespace {
 
bool cat_should_skip_tensor(const Tensor& t) {
    return t.ndim() == 1 && t.shape()[0] == 0;
}
 
Dtype compute_cat_output_dtype(ArrayRef<Tensor> tensors) {
    Dtype result = tensors[0].dtype();
    for (size_t i = 1; i < tensors.size(); ++i) {
        result = promote_types(result, tensors[i].dtype());
    }
    return result;
}
 
} // namespace
 
Tensor cat(ArrayRef<Tensor> tensors, int64_t dim) {
    RAXIS_CHECK(!tensors.empty(), "cat(): expected a non-empty list of tensors");
 
    int64_t valid = -1;
    for (size_t i = 0; i < tensors.size(); ++i) {
        if (!cat_should_skip_tensor(tensors[i])) {
            valid = static_cast<int64_t>(i);
            break;
        }
    }
 
    if (valid < 0) {
        return empty({0}, tensors[0].options());
    }
 
    const Tensor& ref = tensors[valid];
    int ndim = ref.ndim();
    RAXIS_CHECK(ndim > 0, "cat(): zero-dimensional tensors cannot be concatenated");
    dim = wrap_dim(dim, ndim);
 
    Dtype out_dtype = compute_cat_output_dtype(tensors);
    Device out_device = ref.device();
 
    DimVector out_shape(ref.shape());
    int64_t cat_dim_size = 0;
    bool all_contiguous = true;
    bool all_same_dtype = true;
 
    for (size_t i = 0; i < tensors.size(); ++i) {
        const Tensor& t = tensors[i];
        if (cat_should_skip_tensor(t)) {
            continue;
        }
 
        RAXIS_CHECK(t.device() == out_device,
            "cat(): expected all tensors on the same device, "
            "but found ", out_device, " and ", t.device(),
            " (tensor ", i, ")");
 
        RAXIS_CHECK(t.ndim() == ndim,
            "cat(): tensor ", i, " has ", t.ndim(),
            " dimensions, expected ", ndim);
 
        for (int d = 0; d < ndim; ++d) {
            if (d == dim) continue;
            RAXIS_CHECK(t.shape()[d] == ref.shape()[d],
                "cat(): sizes of tensors must match except in dimension ", dim,
                ". Expected size ", ref.shape()[d], " but got size ",
                t.shape()[d], " for tensor ", i, " at dimension ", d);
        }
 
        cat_dim_size += t.shape()[dim];
        all_contiguous = all_contiguous && t.is_contiguous();
        all_same_dtype = all_same_dtype && (t.dtype() == out_dtype);
    }
    out_shape[dim] = cat_dim_size;
 
    Tensor result = empty(out_shape, TensorOptions().dtype(out_dtype).device(out_device));
 
    if (result.numel() == 0) {
        return result;
    }
 
    if (all_contiguous && all_same_dtype && out_device.is_cpu()) {
        if (dim == 0) {
            char* dst = static_cast<char*>(result.mutable_data_ptr());
            for (const Tensor& t : tensors) {
                if (cat_should_skip_tensor(t)) {
                    continue;
                }
                int64_t nbytes = t.nbytes();
                if (nbytes > 0) {
                    std::memcpy(dst, t.data_ptr(), nbytes);
                    dst += nbytes;
                }
            }
            return result;
        }
        cat_serial_stub(DeviceType::CPU, result, tensors, dim);
        return result;
    }
 
    int64_t offset = 0;
    for (const Tensor& t : tensors) {
        if (cat_should_skip_tensor(t)) {
            continue;
        }
        int64_t slice_size = t.shape()[dim];
        Tensor result_slice = result.narrow(dim, offset, slice_size);
        result_slice.copy_(t);
        offset += slice_size;
    }
 
    return result;
}
 
Tensor stack(ArrayRef<Tensor> tensors, int64_t dim) {
    RAXIS_CHECK(!tensors.empty(), "stack(): expected a non-empty list of tensors");
 
    const Tensor& ref = tensors[0];
    Device out_device = ref.device();
 
    for (size_t i = 1; i < tensors.size(); ++i) {
        RAXIS_CHECK(tensors[i].shape() == ref.shape(),
            "stack(): each tensor must be the same size. "
            "Tensor ", i, " has shape ", tensors[i].shape(),
            " but tensor 0 has shape ", ref.shape());
 
        RAXIS_CHECK(tensors[i].device() == out_device,
            "stack(): expected all tensors on the same device, "
            "but found ", out_device, " and ", tensors[i].device(),
            " (tensor ", i, ")");
    }
 
    dim = wrap_dim_allow_upper(dim, ref.ndim());
 
    bool all_contiguous = true;
    bool all_same_dtype = true;
    Dtype first_dtype = ref.dtype();
    for (const Tensor& t : tensors) {
        all_contiguous = all_contiguous && t.is_contiguous();
        all_same_dtype = all_same_dtype && (t.dtype() == first_dtype);
    }
 
    if (all_contiguous && all_same_dtype && dim < ref.ndim() && out_device.is_cpu()) {
        DimVector result_shape(ref.shape());
        result_shape.insert(
            result_shape.begin() + dim, static_cast<int64_t>(tensors.size()));
        Tensor result = empty(result_shape,
            TensorOptions().dtype(first_dtype).device(out_device));
        stack_serial_stub(DeviceType::CPU, result, tensors, dim);
        return result;
    }
 
    std::vector<Tensor> unsqueezed;
    unsqueezed.reserve(tensors.size());
    for (const Tensor& t : tensors) {
        unsqueezed.push_back(t.unsqueeze(dim));
    }
 
    return cat(unsqueezed, dim);
}
 
std::vector<Tensor> unbind(const Tensor& self, int64_t dim) {
    RAXIS_CHECK(self.ndim() > 0, "unbind() cannot be applied to a 0-d tensor");
    dim = wrap_dim(dim, self.ndim());
 
    int64_t size = self.shape()[dim];
    std::vector<Tensor> result(size);
    for (int64_t i = 0; i < size; ++i) {
        result[i] = self.select(dim, i);
    }
    return result;
}

DEFINE_DISPATCH(cat_serial_stub);
DEFINE_DISPATCH(stack_serial_stub);

}