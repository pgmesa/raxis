#pragma once

#include <raxis/autograd/graph.h>
#include <raxis/tensor/tensor.h>


namespace raxis::autograd {

class RAXIS_API NegBackward0 : public Node {
public:
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;
};

class RAXIS_API AbsBackward0 : public Node {
public:
    explicit AbsBackward0(const Tensor& self);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;
private:
    Tensor self_;
};

class RAXIS_API ExpBackward0 : public Node {
public:
    explicit ExpBackward0(const Tensor& result);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;
private:
    Tensor result_;
};

class RAXIS_API Expm1Backward0 : public Node {
public:
    explicit Expm1Backward0(const Tensor& result);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;
private:
    Tensor result_;
};

class RAXIS_API LogBackward0 : public Node {
public:
    explicit LogBackward0(const Tensor& self);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;
private:
    Tensor self_;
};

class RAXIS_API Log2Backward0 : public Node {
public:
    explicit Log2Backward0(const Tensor& self);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;
private:
    Tensor self_;
};

class RAXIS_API Log10Backward0 : public Node {
public:
    explicit Log10Backward0(const Tensor& self);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;
private:
    Tensor self_;
};

class RAXIS_API Log1PBackward0 : public Node {
public:
    explicit Log1PBackward0(const Tensor& self);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;
private:
    Tensor self_;
};

class RAXIS_API SqrtBackward0 : public Node {
public:
    explicit SqrtBackward0(const Tensor& result);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;
private:
    Tensor result_;
};

class RAXIS_API RsqrtBackward0 : public Node {
public:
    explicit RsqrtBackward0(const Tensor& result);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;
private:
    Tensor result_;
};

class RAXIS_API ReciprocalBackward0 : public Node {
public:
    explicit ReciprocalBackward0(const Tensor& result);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;
private:
    Tensor result_;
};

class RAXIS_API SigmoidBackward0 : public Node {
public:
    explicit SigmoidBackward0(const Tensor& result);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;
private:
    Tensor result_;
};

class RAXIS_API SinBackward0 : public Node {
public:
    explicit SinBackward0(const Tensor& self);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;
private:
    Tensor self_;
};

class RAXIS_API CosBackward0 : public Node {
public:
    explicit CosBackward0(const Tensor& self);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;
private:
    Tensor self_;
};

class RAXIS_API TanBackward0 : public Node {
public:
    explicit TanBackward0(const Tensor& result);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;
private:
    Tensor result_;
};

class RAXIS_API AsinBackward0 : public Node {
public:
    explicit AsinBackward0(const Tensor& self);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;
private:
    Tensor self_;
};

class RAXIS_API AcosBackward0 : public Node {
public:
    explicit AcosBackward0(const Tensor& self);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;
private:
    Tensor self_;
};

class RAXIS_API AtanBackward0 : public Node {
public:
    explicit AtanBackward0(const Tensor& self);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;
private:
    Tensor self_;
};

class RAXIS_API SinhBackward0 : public Node {
public:
    explicit SinhBackward0(const Tensor& self);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;
private:
    Tensor self_;
};

class RAXIS_API CoshBackward0 : public Node {
public:
    explicit CoshBackward0(const Tensor& self);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;
private:
    Tensor self_;
};

class RAXIS_API TanhBackward0 : public Node {
public:
    explicit TanhBackward0(const Tensor& result);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;
private:
    Tensor result_;
};

class RAXIS_API ErfBackward0 : public Node {
public:
    explicit ErfBackward0(const Tensor& self);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;
private:
    Tensor self_;
};

class RAXIS_API ErfcBackward0 : public Node {
public:
    explicit ErfcBackward0(const Tensor& self);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;
private:
    Tensor self_;
};

} // namespace raxis::autograd