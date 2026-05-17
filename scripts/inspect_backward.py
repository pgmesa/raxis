
import torch
import raxis


def test_backward(name, torch_fn, raxis_fn, *args):
    """Compare grad_fn names between torch and raxis."""

    def clone_args(lib):
        tensors = []
        for a in args:
            t = a.clone().detach()

            if lib is raxis:
                t = raxis.from_numpy(t.numpy())

            t.requires_grad_(True)
            tensors.append(t)
        return tensors

    # --- Torch ---
    try:
        torch_args = clone_args(torch)
        torch_out = torch_fn(*torch_args)
        torch_grad_fn = (
            torch_out.grad_fn.name()
            if torch_out.grad_fn is not None
            else None
        )
    except Exception as e:
        torch_grad_fn = f"FORWARD FAILED: {e}"

    # --- Raxis ---
    try:
        raxis_args = clone_args(raxis)
        raxis_out = raxis_fn(*raxis_args)

        raxis_grad_fn = (
            raxis_out.grad_fn.name()
            if raxis_out.grad_fn is not None
            else None
        )
    except Exception as e:
        raxis_grad_fn = f"FORWARD FAILED: {e}"

    equal = torch_grad_fn == raxis_grad_fn

    print(
        f"{name:<30s} "
        f"torch={str(torch_grad_fn):<30s} "
        f"raxis={str(raxis_grad_fn):<30s} "
        f"{'OK' if equal else 'DIFF'}"
    )


# ============================================================
# DATA
# ============================================================

mat_a = torch.randn(4, 8)
mat_b = torch.randn(8, 6)
mat_sq = torch.randn(4, 4)

vec = torch.randn(8)
vec4 = torch.randn(4)

batch_a = torch.randn(3, 4, 8)
batch_b = torch.randn(3, 8, 6)

scalar = torch.tensor(2.0)

# ============================================================
# LINEAR ALGEBRA
# ============================================================

print("=" * 120)
print("LINEAR ALGEBRA")
print("=" * 120)

test_backward("mm", torch.mm, raxis.mm, mat_a, mat_b)
test_backward("mv", torch.mv, raxis.mv, mat_a, vec)
test_backward("dot", torch.dot, raxis.dot, vec, vec.clone())
test_backward("bmm", torch.bmm, raxis.bmm, batch_a, batch_b)

test_backward(
    "matmul (mm)",
    torch.matmul,
    raxis.matmul,
    mat_a,
    mat_b,
)

test_backward(
    "matmul (mv)",
    torch.matmul,
    raxis.matmul,
    mat_a,
    vec,
)

test_backward(
    "matmul (dot)",
    torch.matmul,
    raxis.matmul,
    vec,
    vec.clone(),
)

test_backward(
    "matmul (bmm)",
    torch.matmul,
    raxis.matmul,
    batch_a,
    batch_b,
)

self_addmm = torch.randn(4, 6)
test_backward(
    "addmm",
    lambda s, a, b: torch.addmm(s, a, b),
    lambda s, a, b: raxis.addmm(s, a, b),
    self_addmm,
    mat_a,
    mat_b,
)

test_backward(
    "addmv",
    lambda s, m, v: torch.addmv(s, m, v),
    lambda s, m, v: raxis.addmv(s, m, v),
    vec4,
    mat_sq,
    vec4.clone(),
)

test_backward(
    "matmul (1Dx2D)",
    torch.matmul,
    raxis.matmul,
    vec,
    mat_a.transpose(0, 1),
)

self_addbmm = torch.randn(4, 6)
test_backward(
    "addbmm",
    lambda s, a, b: torch.addbmm(s, a, b),
    lambda s, a, b: raxis.addbmm(s, a, b),
    self_addbmm,
    batch_a,
    batch_b,
)

# ============================================================
# BINARY ARITHMETIC
# ============================================================

print("=" * 120)
print("BINARY ARITHMETIC")
print("=" * 120)

a = torch.randn(4, 4)
b = torch.randn(4, 4)

test_backward("add", torch.add, raxis.add, a, b)
test_backward("sub", torch.sub, raxis.sub, a, b)
test_backward("mul", torch.mul, raxis.mul, a, b)
test_backward("div", torch.div, raxis.div, a, b)
test_backward("maximum", torch.maximum, raxis.maximum, a, b)
test_backward("minimum", torch.minimum, raxis.minimum, a, b)

# ============================================================
# POW
# ============================================================

print("=" * 120)
print("POW")
print("=" * 120)

pos = torch.randn(4, 4).abs() + 0.1

test_backward("pow(t,t)", torch.pow, raxis.pow, pos, pos.clone())

test_backward(
    "pow(t,2.0)",
    lambda t: torch.pow(t, 2.0),
    lambda t: raxis.pow(t, 2.0),
    pos,
)

test_backward(
    "pow(2.0,t)",
    lambda t: torch.pow(2.0, t),
    lambda t: raxis.pow(2.0, t),
    pos,
)

# ============================================================
# REDUCTIONS
# ============================================================

print("=" * 120)
print("REDUCTIONS")
print("=" * 120)

test_backward("sum(all)", lambda t: t.sum(), lambda t: t.sum(), a)

test_backward(
    "sum(dim=0)",
    lambda t: t.sum(dim=0),
    lambda t: t.sum(dim=0),
    a,
)

test_backward("nansum(all)", lambda t: t.nansum(), lambda t: t.nansum(), a)

test_backward(
    "nansum(dim=0)",
    lambda t: t.nansum(dim=0),
    lambda t: t.nansum(dim=0),
    a,
)

test_backward(
    "mean(all)",
    lambda t: t.mean(),
    lambda t: t.mean(),
    a,
)

test_backward(
    "mean(dim=0)",
    lambda t: t.mean(dim=0),
    lambda t: t.mean(dim=0),
    a,
)

test_backward("nanmean(all)", lambda t: t.nanmean(), lambda t: t.nanmean(), a)

test_backward(
    "nanmean(dim=0)",
    lambda t: t.nanmean(dim=0),
    lambda t: t.nanmean(dim=0),
    a,
)

test_backward(
    "prod(all)",
    lambda t: t.prod(),
    lambda t: t.prod(),
    a,
)

test_backward(
    "max(all)",
    lambda t: t.max(),
    lambda t: t.max(),
    a,
)

test_backward(
    "min(all)",
    lambda t: t.min(),
    lambda t: t.min(),
    a,
)

test_backward(
    "amax(all)",
    lambda t: t.amax(),
    lambda t: t.amax(),
    a,
)

test_backward(
    "amax(dim=0)",
    lambda t: t.amax(dim=0),
    lambda t: t.amax(dim=0),
    a,
)


test_backward(
    "amin(all)",
    lambda t: t.amin(),
    lambda t: t.amin(),
    a,
)

test_backward(
    "amin(dim=0)",
    lambda t: t.amin(dim=0),
    lambda t: t.amin(dim=0),
    a,
)

test_backward(
    "prod(dim=0)",
    lambda t: t.prod(dim=0),
    lambda t: t.prod(dim=0),
    a,
)

test_backward(
    "sum(dim=0,keepdim)",
    lambda t: t.sum(dim=0, keepdim=True),
    lambda t: t.sum(dim=0, keepdim=True),
    a,
)

test_backward(
    "mean(dim=0,keepdim)",
    lambda t: t.mean(dim=0, keepdim=True),
    lambda t: t.mean(dim=0, keepdim=True),
    a,
)

# ============================================================
# SHAPE OPS
# ============================================================

print("=" * 120)
print("SHAPE OPS")
print("=" * 120)

test_backward(
    "reshape",
    lambda t: t.reshape((-1,)),
    lambda t: t.reshape((-1,)),
    a,
)

test_backward(
    "transpose",
    lambda t: t.transpose(0, 1),
    lambda t: t.transpose(0, 1),
    a,
)

test_backward(
    "permute",
    lambda t: t.permute((1, 0)),
    lambda t: t.permute((1, 0)),
    a,
)

test_backward(
    "unsqueeze",
    lambda t: t.unsqueeze(0),
    lambda t: t.unsqueeze(0),
    a,
)

test_backward(
    "squeeze",
    lambda t: t.unsqueeze(0).squeeze(0),
    lambda t: t.unsqueeze(0).squeeze(0),
    a,
)

test_backward(
    "select",
    lambda t: t[0, 0],
    lambda t: t[0, 0],
    a,
)

test_backward(
    "slice",
    lambda t: t[1:3],
    lambda t: t[1:3],
    a,
)

test_backward(
    "broadcast_to",
    lambda t: t[0:1].broadcast_to(4, 4),
    lambda t: t[0:1].broadcast_to([4, 4]),
    a,
)

test_backward(
    "view",
    lambda t: t.view(-1),
    lambda t: t.view((-1,)),
    a,
)

# ============================================================
# UNARY OPS
# ============================================================

print("=" * 120)
print("UNARY OPS")
print("=" * 120)

for name in [
    "neg", "abs", "exp", "expm1", "log", "log2", "log10", "log1p",
    "sqrt", "rsqrt", "reciprocal", "sigmoid",
    "sin", "cos", "tan", "asin", "acos", "atan",
    "sinh", "cosh", "tanh", "erf", "erfc",
    "ceil", "floor", "round",
]:
    if name in ("log", "log2", "log10", "log1p", "sqrt", "rsqrt", "reciprocal"):
        data = pos
    elif name in ("asin", "acos"):
        data = torch.randn(4, 4).clamp(-0.9, 0.9)
    elif name in ("sinh", "cosh", "tan", "expm1"):
        data = torch.randn(4, 4).clamp(-2.0, 2.0)
    else:
        data = a

    test_backward(
        name,
        lambda t, n=name: getattr(t, n)(),
        lambda t, n=name: getattr(t, n)(),
        data,
    )
    
# ============================================================
# Cat / Stack / Unbind
# ============================================================
print("=" * 120)
print("CAT / STACK")
print("=" * 120)

test_backward(
    "cat",
    lambda a, b: torch.cat([a, b], dim=0),
    lambda a, b: raxis.cat([a, b], dim=0),
    a, b,
)

test_backward(
    "stack",
    lambda a, b: torch.stack([a, b], dim=0),
    lambda a, b: raxis.stack([a, b], dim=0),
    a, b,
)

def test_unbind_backward(name, dim):
    data = torch.randn(3, 4)

    t = data.clone().detach().requires_grad_(True)
    t_parts = torch.unbind(t, dim=dim)
    t_out = t_parts[0] + t_parts[1] + t_parts[2] if dim == 0 else sum(torch.unbind(t, dim=dim))
    t_grad_fn = t_parts[0].grad_fn.name() if t_parts[0].grad_fn else None

    r_data = raxis.from_numpy(data.numpy())
    r = r_data.requires_grad_(True)
    r_parts = raxis.unbind(r, dim=dim)
    r_grad_fn = r_parts[0].grad_fn.name() if r_parts[0].grad_fn else None

    equal = t_grad_fn == r_grad_fn
    print(
        f"{name:<30s} "
        f"torch={str(t_grad_fn):<30s} "
        f"raxis={str(r_grad_fn):<30s} "
        f"{'OK' if equal else 'DIFF'}"
    )

test_unbind_backward("unbind(dim=0)", 0)
test_unbind_backward("unbind(dim=1)", 1)

# ============================================================
# Scalar Ops
# ============================================================
print("=" * 120)
print("SCALAR OPS (reversed)")
print("=" * 120)

test_backward(
    "rsub (scalar - tensor)",
    lambda t: 2.0 - t,
    lambda t: 2.0 - t,
    a,
)

test_backward(
    "rdiv (scalar / tensor)",
    lambda t: 2.0 / t,
    lambda t: 2.0 / t,
    pos,
)

test_backward(
    "add scalar",
    lambda t: t + 2.0,
    lambda t: t + 2.0,
    a,
)

test_backward(
    "mul scalar",
    lambda t: t * 2.0,
    lambda t: t * 2.0,
    a,
)