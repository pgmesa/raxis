#pragma once

#include <array>

#include <raxis/autograd/graph.h>
#include <raxis/core/dtype.h>
#include <raxis/tensor/tensor.h>


namespace raxis::autograd {

// From: torch/csrc/autograd/functions/tensor.h
// The geometry of a tensor within its storage: enough to look at the same
// elements of another buffer laid out the same way
struct TensorGeometry {
    explicit TensorGeometry(const Tensor& t)
        : shape(t.shape()), strides(t.strides()), storage_offset(t.storage_offset()) {}

    DimVector shape;
    DimVector strides;
    int64_t storage_offset;
};

// self.copy_(src): self loses its old values (zero gradient), src gets the
// gradient, converted back to its dtype and reduced to its shape
class RAXIS_API CopyBackwards : public Node {
public:
    CopyBackwards(const Tensor& self, const Tensor& src);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    std::array<bool, 2> needs_grad_;
    Dtype src_dtype_;
    DimVector src_shape_;
};

// From: torch/csrc/autograd/functions/tensor.h (CopySlices)
//
// An in-place op on a view writes a slice of its base. CopySlices becomes
// the grad_fn of the base: from the gradient of the base it takes the
// view's slice, runs the backward of the op (fn) on it and writes the
// result back into the slice. The rest of the base was not touched by the
// op and keeps its gradient.
//
// Its first edge goes to the old history of the base; the others are fn's
// edges to the other inputs of the op.
class RAXIS_API CopySlices : public Node {
public:
    CopySlices(const Tensor& base, const Tensor& view, NodePtr fn);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;
    void release_variables() override;

private:
    TensorGeometry base_;
    TensorGeometry view_;
    NodePtr fn_;
};

// From: tools/autograd/derivatives.yaml (as_strided)
//
// The grad_fn of a view rebuilt after an in-place op elsewhere on its
// memory (see refresh_view_grad_fn): the view's gradient goes into its
// slice of a zero gradient for the base. A single node replaces the chain
// of view ops (select, transpose, ...) that made the view.
class RAXIS_API AsStridedBackward0 : public Node {
public:
    AsStridedBackward0(const Tensor& base, const Tensor& view);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    TensorGeometry base_;
    TensorGeometry view_;
};

} // namespace raxis::autograd
