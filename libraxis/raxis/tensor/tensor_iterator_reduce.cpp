
#include <raxis/tensor/tensor_iterator.h>

#include <algorithm>
#include <tuple>
#include <vector>

#include <raxis/util/exception.h>
#include <raxis/util/parallel.h>
#include <raxis/util/range.h>

#include <raxis/tensor/tensor.h>
#include <raxis/tensor/factories.h>


namespace raxis {

using loop2d_t = TensorIterator::loop2d_t;

static bool use_two_pass_reduction(const TensorIterator& iter);
static void two_pass_reduction(const TensorIterator& iter, loop2d_t loop);
static void parallel_dim_reduction(const TensorIterator& iter, loop2d_t loop);

static int find_split_dim(const TensorIterator& iter);
static std::tuple<int64_t, int64_t> round_columns(
    const TensorIterator& iter, int dim, int multiple, int64_t begin, int64_t end
);


void TensorIterator::parallel_reduce(loop2d_t loop) const {
    RAXIS_CHECK(
        num_operands() == 2,
        "parallel_reduce only supports one input and one output");

    int64_t numel = this->numel();
    if (numel < detail::GRAIN_SIZE || get_num_threads() == 1 || in_parallel_region()) {
        serial_for_each(loop, {0, numel});
    } else if (use_two_pass_reduction(*this)) {
        two_pass_reduction(*this, loop);
    } else {
        parallel_dim_reduction(*this, loop);
    }
}


static bool use_two_pass_reduction(const TensorIterator& iter) {
    return iter.get_output(0).numel() == 1;
}

static void two_pass_reduction(const TensorIterator& iter, loop2d_t loop) {
    const int max_threads = get_num_threads();

    const Tensor& dst = iter.get_output(0);
    Tensor unsqueezed = dst.unsqueeze(0);
    DimVector buffer_shape(unsqueezed.shape());
    buffer_shape[0] = max_threads;
    Tensor buffer = empty(buffer_shape, dst.options());
    buffer.copy_(unsqueezed);

    int64_t buffer_stride = buffer.strides()[0] *
        static_cast<int64_t>(buffer.dtype().itemsize());
    Tensor buffer_0 = buffer[0];

    auto first_reduce = TensorIterator::reduce_op(buffer_0, iter.get_input(0));

    auto shape = first_reduce.shape();
    auto strides = first_reduce.get_strides();

    parallel_for(
        0, iter.numel(), detail::GRAIN_SIZE,
        [&](int64_t begin, int64_t end) {
            const int thread_num = get_thread_num();

            auto base_ptrs = first_reduce.get_base_ptrs();
            base_ptrs[0] += buffer_stride * thread_num;

            ::raxis::detail::serial_for_each(
                shape, strides, base_ptrs.data(),
                base_ptrs.size(), loop, {begin, end});
        });

    auto final_reduce = TensorIterator::reduce_op(unsqueezed, buffer);
    final_reduce.for_each(loop);
}


static int find_split_dim(const TensorIterator& iter) {
    int num_threads = get_num_threads();
    IntArrayRef shape = iter.shape();

    int best_dim = iter.ndim() - 1;
    for (int dim = best_dim; dim >= 0 && !iter.is_dim_reduced(dim); dim--) {
        if (shape[dim] >= num_threads) {
            return dim;
        } else if (shape[dim] > shape[best_dim]) {
            best_dim = dim;
        }
    }

    RAXIS_INTERNAL_ASSERT(!iter.is_dim_reduced(best_dim));
    return best_dim;
}

static std::tuple<int64_t, int64_t> round_columns(
    const TensorIterator& iter,
    int dim,
    int multiple,
    int64_t begin,
    int64_t end
) {
    begin = begin - (begin % multiple);
    if (end != iter.shape()[dim]) {
        end = end - (end % multiple);
    }
    return std::make_tuple(begin, end);
}

static void parallel_dim_reduction(const TensorIterator& iter, loop2d_t loop) {
    RAXIS_INTERNAL_ASSERT(iter.ndim() >= 1);
    int dim = find_split_dim(iter);
    int64_t cols = iter.shape()[dim];
    int64_t element_size = iter.element_size(1);

    bool should_round_columns =
        iter.get_operand(1).stride_bytes[dim] == element_size;

    parallel_for(0, cols, 1, [&](int64_t begin, int64_t end) {
        if (should_round_columns) {
            int64_t cols_per_128_bytes = 128 / element_size;
            std::tie(begin, end) =
                round_columns(iter, dim, static_cast<int>(cols_per_128_bytes), begin, end);
        }
        if (begin == end) {
            return;
        }
        auto sub_iter = TensorIterator(iter);
        sub_iter.narrow(dim, begin, end - begin);
        sub_iter.for_each(loop);
    });
}

} // namespace raxis