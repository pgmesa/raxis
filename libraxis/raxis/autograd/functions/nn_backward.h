#pragma once

#include <array>
#include <optional>

#include <raxis/core/scalar.h>
#include <raxis/autograd/graph.h>
#include <raxis/autograd/saved_variable.h>
#include <raxis/ops/dropout.h>
#include <raxis/ops/loss.h>
#include <raxis/ops/normalization.h>
#include <raxis/ops/pooling.h>
#include <raxis/tensor/tensor.h>
#include <raxis/util/small_vector.h>


namespace raxis::autograd {

class RAXIS_API ReluBackward0 : public Node {
public:
    explicit ReluBackward0(const Tensor& result);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    SavedVariable result_{this, "result"};
};


class RAXIS_API SoftmaxBackward0 : public Node {
public:
    SoftmaxBackward0(const Tensor& result, int64_t dim);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    SavedVariable result_{this, "result"};
    int64_t dim_;
};


class RAXIS_API LogSoftmaxBackward0 : public Node {
public:
    LogSoftmaxBackward0(const Tensor& result, int64_t dim);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    SavedVariable result_{this, "result"};
    int64_t dim_;
};


class RAXIS_API NllLossBackward0 : public Node {
public:
    NllLossBackward0(
        const Tensor& self, const Tensor& target, const Tensor& weight,
        Reduction reduction, int64_t ignore_index, double total_weight);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    SavedVariable self_{this, "self"};
    SavedVariable target_{this, "target"};
    SavedVariable weight_{this, "weight"};
    Reduction reduction_;
    int64_t ignore_index_;
    double total_weight_;
};



// Saves input and weight unconditionally, as PyTorch: grad_weight needs the
// input and grad_input needs the weight, which is alive anyway as a parameter.
class RAXIS_API ConvolutionBackward0 : public Node {
public:
    ConvolutionBackward0(
        const Tensor& input, const Tensor& weight, const Tensor& bias,
        IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    SavedVariable input_{this, "input"};
    SavedVariable weight_{this, "weight"};
    DimVector stride_;
    DimVector padding_;
    DimVector dilation_;
    int64_t groups_;
    bool input_requires_grad_;
    bool weight_requires_grad_;
    bool bias_requires_grad_;
};


// Saves the input shape, not the input (PyTorch saves self): the scatter only
// needs the indices of the maxima
class RAXIS_API MaxPool2DWithIndicesBackward0 : public Node {
public:
    MaxPool2DWithIndicesBackward0(const Tensor& self, const Tensor& indices);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    DimVector self_shape_;
    SavedVariable result1_{this, "result1"};
};


class RAXIS_API AvgPool2DBackward0 : public Node {
public:
    AvgPool2DBackward0(
        const Tensor& self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding,
        bool ceil_mode, bool count_include_pad, std::optional<int64_t> divisor_override);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    DimVector self_shape_;
    DimVector kernel_size_;
    DimVector stride_;
    DimVector padding_;
    bool ceil_mode_;
    bool count_include_pad_;
    std::optional<int64_t> divisor_override_;
};


class RAXIS_API NativeBatchNormBackward0 : public Node {
public:
    NativeBatchNormBackward0(
        const Tensor& input, const Tensor& weight, const Tensor& bias,
        const Tensor& running_mean, const Tensor& running_var,
        const Tensor& save_mean, const Tensor& save_invstd, bool training, double eps);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    SavedVariable input_{this, "input"};
    SavedVariable weight_{this, "weight"};
    SavedVariable running_mean_{this, "running_mean"};
    SavedVariable running_var_{this, "running_var"};
    SavedVariable result1_{this, "result1"};  // save_mean
    SavedVariable result2_{this, "result2"};  // save_invstd
    bool training_;
    double eps_;
    bool input_requires_grad_;
    bool weight_requires_grad_;
    bool bias_requires_grad_;
};


class RAXIS_API NativeLayerNormBackward0 : public Node {
public:
    NativeLayerNormBackward0(
        const Tensor& input, IntArrayRef normalized_shape, const Tensor& weight, const Tensor& bias,
        const Tensor& mean, const Tensor& rstd);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    SavedVariable input_{this, "input"};
    DimVector normalized_shape_;
    SavedVariable weight_{this, "weight"};
    SavedVariable bias_{this, "bias"};
    SavedVariable result1_{this, "result1"};  // mean
    SavedVariable result2_{this, "result2"};  // rstd
    bool input_requires_grad_;
    bool weight_requires_grad_;
    bool bias_requires_grad_;
};


// The mask is a bool tensor (1 byte per element); the scale 1 / (1 - p) is
// applied again in the backward instead of being stored in the mask
class RAXIS_API NativeDropoutBackward0 : public Node {
public:
    NativeDropoutBackward0(const Tensor& mask, double scale);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    SavedVariable result1_{this, "result1"};
    double scale_;
};


// ============================================================
// Double backward: the backward ops above as differentiable ops
// ============================================================
//
// The backward of ReLU, softmax, ... runs a fused kernel (threshold_backward,
// softmax_backward_data, ...). With create_graph that kernel is recorded
// like any op, so it needs its own backward node (its entry in PyTorch's
// derivatives.yaml). Kernels without one get NotImplementedBackward: the
// error is raised if the second backward reaches them.

class RAXIS_API ThresholdBackwardBackward0 : public Node {
public:
    ThresholdBackwardBackward0(const Tensor& grad_output, const Tensor& self, Scalar threshold);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    SavedVariable self_{this, "self"};
    Scalar threshold_;
    std::array<bool, 2> needs_grad_;
};


class RAXIS_API SoftmaxBackwardDataBackward0 : public Node {
public:
    SoftmaxBackwardDataBackward0(const Tensor& grad_output, const Tensor& output, int64_t dim);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    SavedVariable grad_output_{this, "grad_output"};
    SavedVariable output_{this, "output"};
    int64_t dim_;
    std::array<bool, 2> needs_grad_;
};


class RAXIS_API LogSoftmaxBackwardDataBackward0 : public Node {
public:
    LogSoftmaxBackwardDataBackward0(const Tensor& grad_output, const Tensor& output, int64_t dim);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    SavedVariable grad_output_{this, "grad_output"};
    SavedVariable output_{this, "output"};
    int64_t dim_;
    std::array<bool, 2> needs_grad_;
};


class RAXIS_API NativeDropoutBackwardBackward0 : public Node {
public:
    NativeDropoutBackwardBackward0(const Tensor& mask, double scale);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    SavedVariable mask_{this, "mask"};
    double scale_;
};


class RAXIS_API NllLossBackwardBackward0 : public Node {
public:
    NllLossBackwardBackward0(const Tensor& self, const Tensor& target, const Tensor& weight,
                             Reduction reduction, int64_t ignore_index);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    SavedVariable target_{this, "target"};
    SavedVariable weight_{this, "weight"};
    Reduction reduction_;
    int64_t ignore_index_;
    bool self_requires_grad_;
};


class RAXIS_API MaxPool2DWithIndicesBackwardBackward0 : public Node {
public:
    explicit MaxPool2DWithIndicesBackwardBackward0(const Tensor& indices);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    SavedVariable indices_{this, "indices"};
};


class RAXIS_API AvgPool2DBackwardBackward0 : public Node {
public:
    AvgPool2DBackwardBackward0(IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding,
                               bool ceil_mode, bool count_include_pad,
                               std::optional<int64_t> divisor_override);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    SmallVector<int64_t, 2> kernel_size_;
    SmallVector<int64_t, 2> stride_;
    SmallVector<int64_t, 2> padding_;
    bool ceil_mode_;
    bool count_include_pad_;
    std::optional<int64_t> divisor_override_;
};

} // namespace raxis::autograd
