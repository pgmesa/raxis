#pragma once

#include <raxis/autograd/graph.h>
#include <raxis/tensor/tensor.h>


namespace raxis::autograd {

class RAXIS_API DotBackward0 : public Node {
public:
    DotBackward0(const Tensor& self, const Tensor& other);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    bool self_requires_grad_;
    bool other_requires_grad_;
    Tensor self_;
    Tensor other_;
};

class RAXIS_API MvBackward0 : public Node {
public:
    MvBackward0(const Tensor& mat, const Tensor& vec);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    bool mat_requires_grad_;
    bool vec_requires_grad_;
    Tensor mat_;
    Tensor vec_;
};

class RAXIS_API MmBackward0 : public Node {
public:
    MmBackward0(const Tensor& mat1, const Tensor& mat2);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    bool mat1_requires_grad_;
    bool mat2_requires_grad_;
    Tensor mat1_;
    Tensor mat2_;
};

class RAXIS_API BmmBackward0 : public Node {
public:
    BmmBackward0(const Tensor& self, const Tensor& mat2);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    bool self_requires_grad_;
    bool mat2_requires_grad_;
    Tensor self_;
    Tensor mat2_;
};

class RAXIS_API AddmmBackward0 : public Node {
public:
    AddmmBackward0(const Tensor& self, const Tensor& mat1, const Tensor& mat2,
                   Scalar beta, Scalar alpha);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    bool self_requires_grad_;
    bool mat1_requires_grad_;
    bool mat2_requires_grad_;
    DimVector self_shape_;
    Tensor mat1_;
    Tensor mat2_;
    Scalar beta_;
    Scalar alpha_;
};

class RAXIS_API AddmvBackward0 : public Node {
public:
    AddmvBackward0(const Tensor& self, const Tensor& mat, const Tensor& vec,
                   Scalar beta, Scalar alpha);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    bool self_requires_grad_;
    bool mat_requires_grad_;
    bool vec_requires_grad_;
    DimVector self_shape_;
    Tensor mat_;
    Tensor vec_;
    Scalar beta_;
    Scalar alpha_;
};

class RAXIS_API AddbmmBackward0 : public Node {
public:
    AddbmmBackward0(const Tensor& self, const Tensor& batch1, const Tensor& batch2,
                    Scalar beta, Scalar alpha);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;

private:
    bool self_requires_grad_;
    bool batch1_requires_grad_;
    bool batch2_requires_grad_;
    DimVector self_shape_;
    Tensor batch1_;
    Tensor batch2_;
    Scalar beta_;
    Scalar alpha_;
    int64_t batch_size_;
};

}