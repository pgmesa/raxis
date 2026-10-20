
import math

import pytest
import raxis
import torch
import numpy as np

from raxis import nn
from test_utils import check_match, randf_centered


def copy_params(r_model, t_model):
    """Give the torch model the same parameter values as the raxis model."""
    r_params = dict(r_model.named_parameters())
    with torch.no_grad():
        for name, t_param in t_model.named_parameters():
            t_param.copy_(torch.from_numpy(np.array(r_params[name].detach())))


def mlp(lib, in_features=6, hidden=8, out_features=3, final=None):
    lib.manual_seed(0)
    layers =[lib.nn.Linear(in_features, hidden), lib.nn.ReLU(),
              lib.nn.Linear(hidden, out_features)]
    if final is not None:
        layers.append(final(lib))
    return lib.nn.Sequential(*layers)


# ===========================================================================
# MODULE
# ===========================================================================

class TestModule:

    def test_registers_parameters_and_submodules(self):
        model = mlp(raxis)
        names = [name for name, _ in model.named_parameters()]
        assert names == ["0.weight", "0.bias", "2.weight", "2.bias"]
        assert all(isinstance(p, nn.Parameter) for p in model.parameters())
        assert all(p.requires_grad and p.is_leaf for p in model.parameters())

    def test_custom_module(self):
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.body = nn.Linear(4, 4)
                self.head = nn.Linear(4, 1, bias=False)
                self.scale = 2.0  # plain attribute, not registered

            def forward(self, x):
                return self.head(self.body(x)) * self.scale

        net = Net()
        assert [n for n, _ in net.named_parameters()] == ["body.weight", "body.bias", "head.weight"]
        assert net(raxis.rand((5, 4))).shape == (5, 1)

    def test_reassigning_a_parameter_name_unregisters_it(self):
        layer = nn.Linear(2, 2)
        layer.bias = None
        assert [n for n, _ in layer.named_parameters()] == ["weight"]

    def test_inplace_update_keeps_parameter(self):
        layer = nn.Linear(3, 3)
        with raxis.no_grad():
            layer.weight *= 0.1
        assert isinstance(layer.weight, nn.Parameter)
        assert len(list(layer.parameters())) == 2

    def test_train_eval_propagate(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.Sequential(nn.Dropout(0.5)))
        model.eval()
        assert all(not m.training for m in model.modules())
        model.train()
        assert all(m.training for m in model.modules())

    def test_dropout_is_identity_in_eval(self):
        model = nn.Sequential(nn.Dropout(0.9)).eval()
        x = raxis.rand((10, 10))
        check_match(model(x), torch.from_numpy(np.array(x)))

    def test_apply_visits_every_module(self):
        visited = []
        model = mlp(raxis)
        model.apply(lambda m: visited.append(type(m).__name__))
        assert visited == ["Linear", "ReLU", "Linear", "Sequential"]

    def test_zero_grad(self):
        model = mlp(raxis)
        model(raxis.rand((4, 6))).sum().backward()
        assert all(p.grad is not None for p in model.parameters())
        model.zero_grad()
        assert all(p.grad is None for p in model.parameters())

    def test_repr_matches_torch(self):
        r = repr(mlp(raxis, final=lambda lib: lib.nn.LogSoftmax(dim=1)))
        t = repr(mlp(torch, final=lambda lib: lib.nn.LogSoftmax(dim=1)))
        assert r == t

    def test_to_cuda_raises(self):
        with pytest.raises(NotImplementedError):
            mlp(raxis).to("cuda")


# ===========================================================================
# HOOKS
# ===========================================================================

class TestHooks:

    @pytest.mark.parametrize("lib", [raxis, torch], ids=["raxis", "torch"])
    def test_call_order_and_arguments(self, lib):
        model = mlp(lib)
        calls = []
        for name, module in model.named_modules():
            module.register_forward_pre_hook(
                lambda m, args, name=name: calls.append(("pre", name, len(args))))
            module.register_forward_hook(
                lambda m, args, out, name=name: calls.append(("post", name, tuple(out.shape))))
        model(lib.ones((5, 6)))
        assert calls == [
            ("pre", "", 1),
            ("pre", "0", 1), ("post", "0", (5, 8)),
            ("pre", "1", 1), ("post", "1", (5, 8)),
            ("pre", "2", 1), ("post", "2", (5, 3)),
            ("post", "", (5, 3)),
        ]

    @pytest.mark.parametrize("lib", [raxis, torch], ids=["raxis", "torch"])
    def test_hooks_replace_input_and_output(self, lib):
        layer = lib.nn.ReLU()
        layer.register_forward_pre_hook(lambda m, args: -args[0])  # single value
        layer.register_forward_hook(lambda m, args, out: out * 2.0)
        layer.register_forward_hook(lambda m, args, out: None)  # keeps the output
        out = layer(lib.tensor([-1.0, 2.0]))
        assert out.tolist() == [2.0, 0.0]

    @pytest.mark.parametrize("lib", [raxis, torch], ids=["raxis", "torch"])
    def test_remove(self, lib):
        layer = lib.nn.ReLU()
        calls = []
        handle = layer.register_forward_hook(lambda m, args, out: calls.append(1))
        layer(lib.ones(2))
        handle.remove()
        handle.remove()  # idempotent
        layer(lib.ones(2))
        with layer.register_forward_pre_hook(lambda m, args: calls.append(2)):
            layer(lib.ones(2))
        layer(lib.ones(2))
        assert calls == [1, 2]


# ===========================================================================
# SUMMARY
# ===========================================================================

class Block(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.fc = nn.Linear(features, features)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.fc(x))


class TestSummary:

    def test_shapes_and_params_match_torch(self):
        r_model = mlp(raxis, final=lambda lib: lib.nn.LogSoftmax(dim=1))
        t_model = mlp(torch, final=lambda lib: lib.nn.LogSoftmax(dim=1))
        t_shapes = []
        for _, m in t_model.named_modules():
            m.register_forward_hook(lambda m, args, out: t_shapes.append(tuple(out.shape)))
        t_model(torch.ones(5, 6))

        result = nn.summary(r_model, raxis.ones((5, 6)))
        # Rows are in call order, outputs in return order: compare per module
        assert sorted(layer.output_shape for layer in result.layers) == sorted(t_shapes)
        assert [(l.name, l.type_name, l.depth) for l in result.layers] == [
            ("", "Sequential", 0), ("0", "Linear", 1), ("1", "ReLU", 1),
            ("2", "Linear", 1), ("3", "LogSoftmax", 1)]
        assert [l.num_params for l in result.layers] == [0, 6 * 8 + 8, 0, 8 * 3 + 3, 0]
        assert result.total_params == sum(p.numel() for p in t_model.parameters())
        assert result.trainable_params == result.total_params

    def test_shared_module_counted_once(self):
        block = Block(4)
        model = nn.Sequential(block, block, nn.Linear(4, 2))
        result = nn.summary(model, raxis.ones((3, 4)))
        assert [(l.name, l.depth) for l in result.layers] == [
            ("", 0), ("0", 1), ("0.fc", 2), ("0.act", 2),
            ("0", 1), ("0.fc", 2), ("0.act", 2), ("2", 1)]
        assert [l.num_params for l in result.layers] == [0, 0, 20, 0, 0, None, 0, 10]
        assert result.total_params == 30
        text = str(result)
        assert "(recursive)" in text
        assert "Total params: 30" in text

    def test_frozen_params(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
        model[0].requires_grad_(False)
        result = nn.summary(model, raxis.ones((1, 4)))
        assert (result.total_params, result.trainable_params) == (30, 10)
        assert "Non-trainable params: 20" in str(result)

    def test_several_inputs_and_outputs(self):
        class TwoHeads(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(4, 2)
                self.b = nn.Linear(3, 5)

            def forward(self, x, y):
                return self.a(x), self.b(y)

        result = nn.summary(TwoHeads(), (raxis.ones((7, 4)), raxis.ones((7, 3))))
        assert result.layers[0].output_shape == [(7, 2), (7, 5)]
        assert "[(7, 2), (7, 5)]" in str(result)

    def test_no_graph_and_hooks_removed(self):
        model = mlp(raxis)
        nn.summary(model, raxis.ones((2, 6)))
        assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
        assert all(p.grad is None for p in model.parameters())

    def test_hooks_removed_on_error(self):
        model = mlp(raxis)
        with pytest.raises(RuntimeError):
            nn.summary(model, raxis.ones((2, 5)))  # wrong in_features
        assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


# ===========================================================================
# INIT
# ===========================================================================

class TestInit:

    def test_linear_default_bounds(self):
        layer = nn.Linear(100, 50)
        bound = 1 / math.sqrt(100)
        assert np.abs(np.array(layer.weight.detach())).max() <= bound
        assert np.abs(np.array(layer.bias.detach())).max() <= bound

    def test_xavier_uniform_bounds(self):
        w = nn.init.xavier_uniform_(raxis.empty((200, 300)))
        a = math.sqrt(6.0 / 500)
        values = np.array(w)
        assert np.abs(values).max() <= a
        assert values.std() == pytest.approx(a / math.sqrt(3), rel=0.05)

    @pytest.mark.parametrize("fn, expected_std", [
        (lambda w: nn.init.kaiming_normal_(w, nonlinearity="relu"), math.sqrt(2.0 / 300)),
        (lambda w: nn.init.xavier_normal_(w), math.sqrt(2.0 / 500)),
    ], ids=["kaiming_normal", "xavier_normal"])
    def test_normal_std(self, fn, expected_std):
        values = np.array(fn(raxis.empty((200, 300))))
        assert values.std() == pytest.approx(expected_std, rel=0.05)
        assert values.mean() == pytest.approx(0.0, abs=expected_std * 0.05)

    def test_init_on_parameter_keeps_it_a_leaf(self):
        layer = nn.Linear(4, 4)
        nn.init.zeros_(layer.bias)
        assert layer.bias.is_leaf and layer.bias.grad_fn is None
        assert (np.array(layer.bias.detach()) == 0).all()


# ===========================================================================
# TRAINING: same model, same weights, same data -> same trajectory as torch
# ===========================================================================

def train_both(r_model, t_model, r_opt, t_opt, r_loss_fn, t_loss_fn,
               x, y, y_dtype, steps=20):
    r_x, t_x = raxis.tensor(x), torch.from_numpy(x)
    r_y = raxis.tensor(y, dtype=y_dtype[0])
    t_y = torch.from_numpy(y).to(y_dtype[1])
    r_losses, t_losses = [], []
    for _ in range(steps):
        for model, opt, loss_fn, xs, ys, losses in (
            (r_model, r_opt, r_loss_fn, r_x, r_y, r_losses),
            (t_model, t_opt, t_loss_fn, t_x, t_y, t_losses),
        ):
            opt.zero_grad()
            loss = loss_fn(model(xs), ys)
            loss.backward()
            opt.step()
            losses.append(loss.item())
    return np.array(r_losses), np.array(t_losses)


OPTIMIZERS = [
    ("sgd", lambda lib, p: lib.optim.SGD(p, lr=0.1)),
    ("sgd_momentum", lambda lib, p: lib.optim.SGD(p, lr=0.01, momentum=0.9)),
    ("sgd_nesterov_wd", lambda lib, p: lib.optim.SGD(p, lr=0.01, momentum=0.9,
                                                     nesterov=True, weight_decay=1e-2)),
    ("adam", lambda lib, p: lib.optim.Adam(p, lr=1e-2)),
    ("adam_wd", lambda lib, p: lib.optim.Adam(p, lr=1e-2, weight_decay=1e-2)),
    ("adamw", lambda lib, p: lib.optim.AdamW(p, lr=1e-2)),
]


class TestTraining:

    @pytest.mark.parametrize("name, make_opt", OPTIMIZERS, ids=[n for n, _ in OPTIMIZERS])
    def test_regression_mse(self, name, make_opt):
        r_model, t_model = mlp(raxis, out_features=1), mlp(torch, out_features=1)
        copy_params(r_model, t_model)
        x = randf_centered(16, 6, seed=1)
        y = randf_centered(16, 1, seed=2)
        r_losses, t_losses = train_both(
            r_model, t_model,
            make_opt(raxis, r_model.parameters()), make_opt(torch, t_model.parameters()),
            nn.MSELoss(), torch.nn.MSELoss(), x, y, (raxis.float32, torch.float32))
        np.testing.assert_allclose(r_losses, t_losses, rtol=1e-4, atol=1e-5)
        assert r_losses[-1] < r_losses[0]

    @pytest.mark.parametrize("name, make_opt", OPTIMIZERS[:2], ids=[n for n, _ in OPTIMIZERS[:2]])
    def test_classification_log_softmax_nll(self, name, make_opt):
        final = lambda lib: lib.nn.LogSoftmax(dim=1)
        r_model, t_model = mlp(raxis, final=final), mlp(torch, final=final)
        copy_params(r_model, t_model)
        x = randf_centered(32, 6, seed=3)
        y = np.random.RandomState(4).randint(0, 3, size=32).astype(np.int64)
        r_losses, t_losses = train_both(
            r_model, t_model,
            make_opt(raxis, r_model.parameters()), make_opt(torch, t_model.parameters()),
            nn.NLLLoss(), torch.nn.NLLLoss(), x, y, (raxis.long, torch.long))
        np.testing.assert_allclose(r_losses, t_losses, rtol=1e-4, atol=1e-5)
        assert r_losses[-1] < r_losses[0]

    def test_classification_cross_entropy(self):
        r_model, t_model = mlp(raxis), mlp(torch)
        copy_params(r_model, t_model)
        x = randf_centered(32, 6, seed=5)
        y = np.random.RandomState(6).randint(0, 3, size=32).astype(np.int64)
        r_losses, t_losses = train_both(
            r_model, t_model,
            raxis.optim.SGD(r_model.parameters(), lr=0.1),
            torch.optim.SGD(t_model.parameters(), lr=0.1),
            nn.CrossEntropyLoss(), torch.nn.CrossEntropyLoss(), x, y, (raxis.long, torch.long))
        np.testing.assert_allclose(r_losses, t_losses, rtol=1e-4, atol=1e-5)

    def test_optimizer_rejects_empty_params(self):
        with pytest.raises(ValueError):
            raxis.optim.SGD([], lr=0.1)


# ===========================================================================
# PARAM GROUPS
# ===========================================================================

def two_groups(lib, model, lr=0.1):
    return [{"params": model[0].parameters()},
            {"params": model[2].parameters(), "lr": lr / 10, "momentum": 0.5}]


class TestParamGroups:

    @pytest.mark.parametrize("name, make_opt", OPTIMIZERS, ids=[n for n, _ in OPTIMIZERS])
    def test_groups_match_torch(self, name, make_opt):
        r_model, t_model = mlp(raxis, out_features=1), mlp(torch, out_features=1)
        copy_params(r_model, t_model)
        groups = lambda lib, m: [{"params": m[0].parameters()},
                                 {"params": m[2].parameters(), "lr": 1e-3, "weight_decay": 0.1}]
        r_opt, t_opt = make_opt(raxis, groups(raxis, r_model)), make_opt(torch, groups(torch, t_model))
        assert r_opt.param_groups[1]["lr"] == t_opt.param_groups[1]["lr"] == 1e-3
        x, y = randf_centered(16, 6, seed=1), randf_centered(16, 1, seed=2)
        r_losses, t_losses = train_both(r_model, t_model, r_opt, t_opt, nn.MSELoss(), torch.nn.MSELoss(),
                                        x, y, (raxis.float32, torch.float32), steps=10)
        np.testing.assert_allclose(r_losses, t_losses, rtol=1e-4, atol=1e-5)

    def test_changing_lr(self):
        r_model, t_model = mlp(raxis, out_features=1), mlp(torch, out_features=1)
        copy_params(r_model, t_model)
        r_opt = raxis.optim.SGD(two_groups(raxis, r_model), lr=0.1, momentum=0.9)
        t_opt = torch.optim.SGD(two_groups(torch, t_model), lr=0.1, momentum=0.9)
        x, y = randf_centered(16, 6, seed=1), randf_centered(16, 1, seed=2)
        args = (nn.MSELoss(), torch.nn.MSELoss(), x, y, (raxis.float32, torch.float32))
        train_both(r_model, t_model, r_opt, t_opt, *args, steps=3)
        for opt in (r_opt, t_opt):
            for group in opt.param_groups:
                group["lr"] *= 0.1
        r_losses, t_losses = train_both(r_model, t_model, r_opt, t_opt, *args, steps=3)
        np.testing.assert_allclose(r_losses, t_losses, rtol=1e-4, atol=1e-5)

    def test_parameter_in_two_groups(self):
        layer = nn.Linear(2, 2)
        with pytest.raises(ValueError, match="more than one parameter group"):
            raxis.optim.SGD([{"params": layer.parameters()}, {"params": [layer.weight]}], lr=0.1)


# ===========================================================================
# ACTIVATION MODULES
# ===========================================================================

ACTIVATIONS = [
    ("ReLU", lambda lib: lib.nn.ReLU()),
    ("LeakyReLU", lambda lib: lib.nn.LeakyReLU(0.2)),
    ("SELU", lambda lib: lib.nn.SELU()),
    ("Sigmoid", lambda lib: lib.nn.Sigmoid()),
    ("Tanh", lambda lib: lib.nn.Tanh()),
    ("Softmax", lambda lib: lib.nn.Softmax(dim=1)),
    ("LogSoftmax", lambda lib: lib.nn.LogSoftmax(dim=1)),
]


@pytest.mark.parametrize("name, make", ACTIVATIONS, ids=[n for n, _ in ACTIVATIONS])
def test_activation_modules(name, make):
    data = randf_centered(5, 7)
    r = raxis.tensor(data, requires_grad=True)
    t = torch.tensor(data, requires_grad=True)
    r_out, t_out = make(raxis)(r), make(torch)(t)
    check_match(r_out, t_out, context=name)
    grad = randf_centered(5, 7, seed=9)
    r_out.backward(raxis.tensor(grad))
    t_out.backward(torch.from_numpy(grad))
    check_match(r.grad, t.grad, atol=1e-5, context=f"{name} grad")
