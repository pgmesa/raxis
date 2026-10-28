
#include <raxis/ops/cat_ops.h>

#include <cstdint>
#include <cstring>

#include <raxis/ops/copy.h>
#include <raxis/ops/dispatch_stub.h>

#include <raxis/util/exception.h>
#include <raxis/util/parallel.h>

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
 
// From: aten/src/ATen/native/TensorShape.cpp (cat_out_cpu)
// Writes the inputs one after the other along dim of result.
//
// The serial kernels walk the output once, copying a row of each input in
// turn, but on a single thread. Above GRAIN_SIZE elements PyTorch instead
// copies input by input with copy_stub, which splits each copy among the
// threads (the inputs are copied one at a time, each one in parallel).
void cat_out_impl(
    const Tensor& result, ArrayRef<Tensor> tensors, int64_t dim, int64_t valid,
    bool all_contiguous, bool all_same_dtype, bool all_same_sizes_and_stride
) {
    if (result.numel() == 0) {
        return;
    }

    // From: aten/src/ATen/native/cuda/Shape.cu (cat_out_cuda: parallel_cat
    // when all the inputs are contiguous and share the result's dtype)
    if (!result.is_cpu() && all_contiguous && all_same_dtype && result.is_contiguous()
            && result.numel() <= INT32_MAX) {
        cat_batched_stub(result.device().type(), result, tensors, dim);
        return;
    }

    // The serial kernels read on the host; a GPU copies input by input below
    bool use_serial_kernel = result.is_cpu() && (result.numel() < detail::GRAIN_SIZE || get_num_threads() == 1);
    if (use_serial_kernel && all_contiguous && all_same_dtype && result.is_contiguous()) {
        if (dim == 0) {
            // Each input is one contiguous block of the output
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
        } else {
            cat_serial_stub(result.device().type(), result, tensors, dim);
        }
        return;
    }

    int64_t offset = 0;
    if (all_same_sizes_and_stride && all_same_dtype && result.is_contiguous()) {
        // Same layout for every (output slice, input) pair: build the
        // iterator once and only move its data pointers
        const Tensor& source_slice = tensors[valid];
        int64_t slice_dim_size = source_slice.shape()[dim];
        Tensor result_slice = result.narrow(dim, 0, slice_dim_size);
        char* result_data = static_cast<char*>(result_slice.mutable_data_ptr());
        int64_t result_stride_bytes = result.strides()[dim] * result.dtype().itemsize();

        auto iter = TensorIteratorConfig()
            .check_mem_overlap(false)
            .add_output(result_slice)
            .add_input(source_slice)
            .build();

        for (const Tensor& t : tensors) {
            if (cat_should_skip_tensor(t)) {
                continue;
            }
            iter.unsafe_replace_operand(0, result_data + offset * result_stride_bytes);
            iter.unsafe_replace_operand(1, const_cast<void*>(t.data_ptr()));
            copy_stub(iter.device_type(), iter, /*non_blocking=*/false);
            offset += slice_dim_size;
        }
        return;
    }

    for (const Tensor& t : tensors) {
        if (cat_should_skip_tensor(t)) {
            continue;
        }
        int64_t slice_size = t.shape()[dim];
        Tensor result_slice = result.narrow(dim, offset, slice_size);
        result_slice.copy_(t);
        offset += slice_size;
    }
}

} // namespace

namespace ops {


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
    bool all_same_sizes_and_stride = true;

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
        all_same_sizes_and_stride = all_same_sizes_and_stride
            && t.shape() == ref.shape() && t.strides() == ref.strides();
    }
    out_shape[dim] = cat_dim_size;
 
    Tensor result = empty(out_shape, TensorOptions().dtype(out_dtype).device(out_device));
    cat_out_impl(result, tensors, dim, valid, all_contiguous, all_same_dtype, all_same_sizes_and_stride);
    return result;
}

// From: aten/src/ATen/native/TensorShape.cpp (stack_out, _stack_out_cpu)
Tensor stack(ArrayRef<Tensor> tensors, int64_t dim) {
    RAXIS_CHECK(!tensors.empty(), "stack(): expected a non-empty list of tensors");

    const Tensor& ref = tensors[0];
    Device out_device = ref.device();
    Dtype out_dtype = compute_cat_output_dtype(tensors);

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
    if (dim == ref.ndim()) {
        std::vector<Tensor> unsqueezed;
        unsqueezed.reserve(tensors.size());
        for (const Tensor& t : tensors) {
            unsqueezed.push_back(t.unsqueeze(dim));
        }
        return ops::cat(unsqueezed, dim);
    }

    DimVector result_shape(ref.shape());
    result_shape.insert(result_shape.begin() + dim, static_cast<int64_t>(tensors.size()));
    Tensor result = empty(result_shape, TensorOptions().dtype(out_dtype).device(out_device));
    if (result.numel() == 0) {
        return result;
    }

    // From: aten/src/ATen/native/cpu/SerialStackImpl.h (can_use_native_serial_stack_impl)
    bool all_contiguous = true;
    bool all_same_dtype = true;
    bool all_same_strides = true;
    for (const Tensor& t : tensors) {
        all_contiguous = all_contiguous && t.is_contiguous();
        all_same_dtype = all_same_dtype && t.dtype() == out_dtype;
        all_same_strides = all_same_strides && t.strides() == ref.strides();
    }
    int64_t numel_in_stack = ref.numel() * static_cast<int64_t>(tensors.size());
    if (result.is_cpu() && all_contiguous && all_same_dtype
            && (numel_in_stack < detail::GRAIN_SIZE || get_num_threads() == 1)) {
        stack_serial_stub(result.device().type(), result, tensors, dim);
        return result;
    }

    // Stacking n tensors along dim is a cat along dim into a view of the
    // result where dim is n times longer: (n, a, b) seen as (n * a, b)
    DimVector cat_shape(ref.shape());
    cat_shape[dim] *= static_cast<int64_t>(tensors.size());
    cat_out_impl(result.view(cat_shape), tensors, dim, /*valid=*/0,
                 all_contiguous, all_same_dtype, all_same_dtype && all_same_strides);
    return result;
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

}

DEFINE_DISPATCH(cat_serial_stub);
DEFINE_DISPATCH(stack_serial_stub);
DEFINE_DISPATCH(cat_batched_stub);
REGISTER_NO_CPU_DISPATCH(cat_batched_stub)

}