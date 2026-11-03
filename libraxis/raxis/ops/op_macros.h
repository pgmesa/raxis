#pragma once


#define RAXIS_DECLARE_UNARY_OP(op_name)                                        \
    namespace ops { RAXIS_API Tensor op_name(const Tensor& self); }            \
    RAXIS_API Tensor op_name(const Tensor& self);

#define RAXIS_DECLARE_UNARY_OP_INPLACE(op_name)                                \
    namespace ops { RAXIS_API Tensor& op_name##_(Tensor& self); }              \
    RAXIS_API Tensor& op_name##_(Tensor& self);

// out= variants: ops::op_name_out writes into out (allocating it when it is
// undefined); raxis::op_name_out also checks it for autograd and moves its
// version
#define RAXIS_DECLARE_UNARY_OP_OUT(op_name)                                    \
    namespace ops { RAXIS_API Tensor& op_name##_out(const Tensor& self, Tensor& out); } \
    RAXIS_API Tensor& op_name##_out(const Tensor& self, Tensor& out);

#define RAXIS_DECLARE_UNARY_ARITHMETIC(op_name)                                \
    RAXIS_DECLARE_UNARY_OP(op_name)                                            \
    RAXIS_DECLARE_UNARY_OP_INPLACE(op_name)                                    \
    RAXIS_DECLARE_UNARY_OP_OUT(op_name)


#define RAXIS_DECLARE_BINARY_OP(op_name)                                       \
    namespace ops { RAXIS_API Tensor op_name(const Tensor& a, const Tensor& b); } \
    RAXIS_API Tensor op_name(const Tensor& a, const Tensor& b);

#define RAXIS_DECLARE_BINARY_OP_SCALAR(op_name)                                \
    namespace ops { RAXIS_API Tensor op_name(const Tensor& a, Scalar s); }     \
    RAXIS_API Tensor op_name(const Tensor& a, Scalar s);

#define RAXIS_DECLARE_BINARY_OP_INPLACE_TENSOR(op_name)                        \
    namespace ops { RAXIS_API Tensor& op_name##_(Tensor& a, const Tensor& b); } \
    RAXIS_API Tensor& op_name##_(Tensor& a, const Tensor& b);

#define RAXIS_DECLARE_BINARY_OP_INPLACE_SCALAR(op_name)                        \
    namespace ops { RAXIS_API Tensor& op_name##_(Tensor& a, Scalar s); }       \
    RAXIS_API Tensor& op_name##_(Tensor& a, Scalar s);

#define RAXIS_DECLARE_BINARY_OP_OUT(op_name)                                   \
    namespace ops { RAXIS_API Tensor& op_name##_out(const Tensor& a, const Tensor& b, Tensor& out); } \
    RAXIS_API Tensor& op_name##_out(const Tensor& a, const Tensor& b, Tensor& out);

#define RAXIS_DECLARE_BINARY_OP_SCALAR_OUT(op_name)                            \
    namespace ops { RAXIS_API Tensor& op_name##_out(const Tensor& a, Scalar s, Tensor& out); } \
    RAXIS_API Tensor& op_name##_out(const Tensor& a, Scalar s, Tensor& out);

#define RAXIS_DECLARE_BINARY_ARITHMETIC(op_name)                               \
    RAXIS_DECLARE_BINARY_OP(op_name)                                           \
    RAXIS_DECLARE_BINARY_OP_SCALAR(op_name)                                    \
    RAXIS_DECLARE_BINARY_OP_INPLACE_TENSOR(op_name)                            \
    RAXIS_DECLARE_BINARY_OP_INPLACE_SCALAR(op_name)                            \
    RAXIS_DECLARE_BINARY_OP_OUT(op_name)                                       \
    RAXIS_DECLARE_BINARY_OP_SCALAR_OUT(op_name)

#define RAXIS_DECLARE_BINARY_REVERSED(op_name)                                 \
    RAXIS_DECLARE_BINARY_OP(r##op_name)                                        \
    RAXIS_DECLARE_BINARY_OP_SCALAR(r##op_name)                                 \
    RAXIS_DECLARE_BINARY_OP_INPLACE_TENSOR(r##op_name)                         \
    RAXIS_DECLARE_BINARY_OP_INPLACE_SCALAR(r##op_name)
