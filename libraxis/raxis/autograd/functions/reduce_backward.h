#pragma once

#include <raxis/autograd/graph.h>
#include <raxis/autograd/saved_variable.h>
#include <raxis/tensor/tensor.h>


namespace raxis::autograd {

class RAXIS_API SumBackward0 : public Node {
public:
    explicit SumBackward0(const Tensor& self);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    DimVector self_shape_;
};

class RAXIS_API SumBackward1 : public Node {
public:
    SumBackward1(const Tensor& self, IntArrayRef dim, bool keepdim);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    DimVector self_shape_;
    DimVector dim_;
    bool keepdim_;
};


class RAXIS_API MeanBackward0 : public Node {
public:
    explicit MeanBackward0(const Tensor& self);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    DimVector self_shape_;
    int64_t numel_;
};

class RAXIS_API MeanBackward1 : public Node {
public:
    MeanBackward1(const Tensor& self, IntArrayRef dim, bool keepdim);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    DimVector self_shape_;
    DimVector dim_;
    bool keepdim_;
    int64_t numel_reduced_;
};


class RAXIS_API ProdBackward0 : public Node {
public:
    ProdBackward0(const Tensor& self, const Tensor& result);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    Tensor self_;
    SavedVariable result_;
};

class RAXIS_API ProdBackward1 : public Node {
public:
    ProdBackward1(const Tensor& self, IntArrayRef dim, bool keepdim,
                  const Tensor& result);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    Tensor self_;
    SavedVariable result_;
    DimVector dim_;
    bool keepdim_;
};

class RAXIS_API AmaxBackward0 : public Node {
public:
    AmaxBackward0(const Tensor& self, IntArrayRef dim, bool keepdim,
                  const Tensor& result);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    Tensor self_;
    SavedVariable result_;
    DimVector dim_;
    bool keepdim_;
};

class RAXIS_API AminBackward0 : public Node {
public:
    AminBackward0(const Tensor& self, IntArrayRef dim, bool keepdim,
                  const Tensor& result);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    Tensor self_;
    SavedVariable result_;
    DimVector dim_;
    bool keepdim_;
};

// max(dim) / min(dim): the gradient goes to the selected element
class RAXIS_API MaxBackward0 : public Node {
public:
    MaxBackward0(const Tensor& self, int64_t dim, const Tensor& indices, bool keepdim);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    DimVector self_shape_;
    int64_t dim_;
    Tensor indices_;
    bool keepdim_;
};

class RAXIS_API MinBackward0 : public MaxBackward0 {
public:
    using MaxBackward0::MaxBackward0;
    std::string name() const override;
};

class RAXIS_API MaxBackward1 : public Node {
public:
    MaxBackward1(const Tensor& self, const Tensor& result);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    Tensor self_;
    SavedVariable result_;
};

class RAXIS_API MinBackward1 : public Node {
public:
    MinBackward1(const Tensor& self, const Tensor& result);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    Tensor self_;
    SavedVariable result_;
};


class RAXIS_API NansumBackward0 : public Node {
public:
    explicit NansumBackward0(const Tensor& self);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    Tensor self_;
    DimVector self_shape_;
};

class RAXIS_API NansumBackward1 : public Node {
public:
    NansumBackward1(const Tensor& self, IntArrayRef dim, bool keepdim);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    Tensor self_;
    DimVector self_shape_;
    DimVector dim_;
    bool keepdim_;
};


class RAXIS_API NanmeanBackward0 : public Node {
public:
    explicit NanmeanBackward0(const Tensor& self);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    Tensor self_;
    DimVector self_shape_;
};

class RAXIS_API NanmeanBackward1 : public Node {
public:
    NanmeanBackward1(const Tensor& self, IntArrayRef dim, bool keepdim);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    Tensor self_;
    DimVector self_shape_;
    DimVector dim_;
    bool keepdim_;
};

} // namespace raxis::autograd