
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

    def test_tied_weights_are_yielded_once(self):
        """A weight shared by two modules (nanoGPT ties the token embedding
        to the output projection) must reach the optimizer once."""
        names = []
        for lib in (raxis, torch):
            model = lib.nn.Sequential(lib.nn.Linear(4, 4), lib.nn.Linear(4, 4))
            model[1].weight = model[0].weight
            names.append([n for n, _ in model.named_parameters()])
            assert len(list(model.parameters())) == 3
            assert len(list(model.named_parameters(remove_duplicate=False))) == 4
            assert len(model.state_dict()) == 4
        assert names[0] == names[1]

    def test_shared_module_is_yielded_once(self):
        for lib in (raxis, torch):
            layer = lib.nn.Linear(2, 2)
            model = lib.nn.Sequential(layer, lib.nn.ReLU(), layer)
            assert [n for n, _ in model.named_modules()] == ["", "0", "1"]
            assert len(list(model.children())) == 2
            assert len(list(model.parameters())) == 2

    def test_parameters_without_recurse(self):
        model = nn.Sequential(nn.Linear(2, 2))
        model.scale = nn.Parameter(raxis.ones(1))
        assert [n for n, _ in model.named_parameters(recurse=False)] == ["scale"]


# ===========================================================================
# CONTAINERS
# ===========================================================================

class TestModuleList:

    def test_registers_and_indexes(self):
        r, t = (lib.nn.ModuleList([lib.nn.Linear(2, 3), lib.nn.Linear(3, 1)]) for lib in (raxis, torch))
        assert [n for n, _ in r.named_parameters()] == [n for n, _ in t.named_parameters()]
        assert len(r) == 2 and r[-1] is r[1] and isinstance(r[:1], nn.ModuleList) and len(r[:1]) == 1
        with pytest.raises(IndexError, match="index 2 is out of range"):
            r[2]

    def test_mutation_keeps_names_contiguous(self):
        layers = [nn.Linear(1, 1) for _ in range(4)]
        ml = nn.ModuleList(layers[:2])
        ml.append(layers[2])
        ml += [layers[3]]
        ml.insert(1, nn.ReLU())
        assert list(ml._modules) == ["0", "1", "2", "3", "4"]
        assert isinstance(ml[1], nn.ReLU)
        del ml[1]
        assert list(ml) == layers and list(ml._modules) == ["0", "1", "2", "3"]
        assert ml.pop(0) is layers[0] and ml[0] is layers[1] and ml._modules["0"] is layers[1]

    def test_repr_compresses_repeats_as_torch(self):
        reprs = []
        for lib in (raxis, torch):
            ml = lib.nn.ModuleList([lib.nn.Linear(2, 2) for _ in range(3)] + [lib.nn.ReLU()])
            reprs.append(repr(lib.nn.Sequential(ml)))
        assert reprs[0] == reprs[1]


class TestModuleDict:

    def test_dict_interface(self):
        md = nn.ModuleDict(dict(fc=nn.Linear(2, 2), act=nn.ReLU()))
        assert list(md) == ["fc", "act"] and "fc" in md and len(md) == 2
        assert md.fc is md["fc"]
        md.update([("out", nn.Linear(2, 1))])
        assert list(md.keys()) == ["fc", "act", "out"]
        assert [n for n, _ in md.named_parameters()] == ["fc.weight", "fc.bias", "out.weight", "out.bias"]
        assert isinstance(md.pop("act"), nn.ReLU) and "act" not in md and not hasattr(md, "act")
        md.clear()
        assert len(md) == 0 and list(md.parameters()) == []

    def test_repr_matches_torch(self):
        r, t = (repr(lib.nn.ModuleDict(dict(fc=lib.nn.Linear(2, 2), h=lib.nn.ModuleList([lib.nn.ReLU()]))))
                for lib in (raxis, torch))
        assert r == t


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


def _grad_values(grads):
    return [None if g is None else np.asarray(g.tolist()).round(4).tolist() for g in grads]


def _fixed_mlp(lib):
    """mlp with the same parameter values in both libraries (their random
    initializations differ for the same seed)."""
    model = mlp(lib)
    rng = np.random.default_rng(0)
    with lib.no_grad():
        for p in model.parameters():
            values = rng.standard_normal(tuple(p.shape)).astype(np.float32)
            p.copy_(lib.tensor(values.tolist()))
    return model


class TestBackwardHooks:

    @staticmethod
    def full_and_pre_hooks(lib):
        model = _fixed_mlp(lib)
        calls = []
        for name, module in model.named_modules():
            module.register_full_backward_pre_hook(
                lambda m, grad_output, name=name: calls.append(("pre", name, _grad_values(grad_output))))
            module.register_full_backward_hook(
                lambda m, grad_input, grad_output, name=name: calls.append(("full", name, _grad_values(grad_input))))
        x = lib.ones((2, 6), requires_grad=True)
        model(x).sum().backward()
        return calls, _grad_values([x.grad])

    @staticmethod
    def hooks_replace_gradients(lib):
        layer = _fixed_mlp(lib)[0]
        layer.register_full_backward_pre_hook(lambda m, grad_output: (grad_output[0] * 2,))
        layer.register_full_backward_hook(lambda m, grad_input, grad_output: (grad_input[0] + 1,))
        x = lib.ones((2, 6), requires_grad=True)
        layer(x).sum().backward()
        return _grad_values([x.grad, layer.weight.grad])

    @staticmethod
    def no_input_requires_grad(lib):
        import warnings
        layer = _fixed_mlp(lib)[0]
        calls = []
        layer.register_full_backward_hook(lambda m, grad_input, grad_output: calls.append(
            (grad_input, _grad_values(grad_output))))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # torch warns that the hook sees no input gradients
            layer(lib.ones((2, 6))).sum().backward()
        return calls

    CASES = [full_and_pre_hooks, hooks_replace_gradients, no_input_requires_grad]

    @pytest.mark.parametrize("case", CASES, ids=[c.__name__ for c in CASES])
    def test_matches_torch(self, case):
        assert case(raxis) == case(torch)

    def test_remove_and_prepend(self):
        layer = nn.ReLU()
        calls = []
        handle = layer.register_full_backward_hook(lambda m, gi, go: calls.append("removed"))
        layer.register_full_backward_hook(lambda m, gi, go: calls.append("last"))
        layer.register_full_backward_hook(lambda m, gi, go: calls.append("first"), prepend=True)
        handle.remove()
        layer(raxis.ones(2, requires_grad=True)).sum().backward()
        assert calls == ["first", "last"]


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

    def test_classification_weighted_cross_entropy(self):
        r_model, t_model = mlp(raxis), mlp(torch)
        copy_params(r_model, t_model)
        x = randf_centered(32, 6, seed=5)
        y = np.random.RandomState(6).randint(0, 3, size=32).astype(np.int64)
        weight = np.array([0.2, 1.0, 3.0], dtype=np.float32)
        r_loss_fn = nn.CrossEntropyLoss(weight=raxis.from_numpy(weight))
        t_loss_fn = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(weight))
        # The weight is a buffer: it goes into the state_dict
        assert list(r_loss_fn.state_dict()) == list(t_loss_fn.state_dict()) == ["weight"]
        r_losses, t_losses = train_both(
            r_model, t_model,
            raxis.optim.SGD(r_model.parameters(), lr=0.1),
            torch.optim.SGD(t_model.parameters(), lr=0.1),
            r_loss_fn, t_loss_fn, x, y, (raxis.long, torch.long))
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
# STATE DICT
# ===========================================================================

def with_buffers(lib):
    base = nn.Module if lib is raxis else torch.nn.Module

    class Net(base):
        def __init__(self):
            super().__init__()
            self.fc = lib.nn.Linear(3, 2)
            self.register_buffer("running_mean", lib.zeros(2))
            self.register_buffer("cache", lib.ones(2), persistent=False)
            self.inner = lib.nn.Sequential(lib.nn.Linear(2, 2))

        def forward(self, x):
            return self.inner(self.fc(x) - self.running_mean)

    return Net()


def to_raxis(state_dict):
    return {k: raxis.tensor(v.detach().numpy()) for k, v in state_dict.items()}


class TestModuleStateDict:

    def test_keys_match_torch(self):
        assert list(mlp(raxis).state_dict()) == list(mlp(torch).state_dict())
        assert list(with_buffers(raxis).state_dict()) == list(with_buffers(torch).state_dict())

    def test_values_are_detached_aliases(self):
        model = mlp(raxis)
        sd = model.state_dict()
        assert not sd["0.weight"].requires_grad
        with raxis.no_grad():
            model[0].weight.fill_(2.0)
        assert sd["0.weight"].tolist()[0][0] == 2.0
        assert model.state_dict(keep_vars=True)["0.weight"] is model[0].weight

    def test_buffers(self):
        model = with_buffers(raxis)
        assert [n for n, _ in model.named_buffers()] == ["running_mean", "cache"]
        model.running_mean = raxis.ones(2)
        assert model.state_dict()["running_mean"].tolist() == [1.0, 1.0]
        assert "cache" not in model.state_dict()

    def test_load_from_torch(self):
        r_model, t_model = with_buffers(raxis), with_buffers(torch)
        with torch.no_grad():
            t_model.running_mean.fill_(0.5)
        result = r_model.load_state_dict(to_raxis(t_model.state_dict()))
        assert repr(result) == "<All keys matched successfully>"
        x = randf_centered(4, 3)
        check_match(r_model(raxis.tensor(x)), t_model(torch.from_numpy(x)))

    def test_load_round_trip(self):
        a, b = mlp(raxis), mlp(raxis)
        with raxis.no_grad():
            for p in b.parameters():
                p.fill_(0.0)
        b.load_state_dict(a.state_dict())
        x = raxis.tensor(randf_centered(4, 6))
        assert (a(x) - b(x)).abs().max().item() == 0.0
        assert all(p.requires_grad for p in b.parameters())

    def test_strict_errors_match_torch(self):
        r_sd = mlp(raxis).state_dict()
        t_sd = mlp(torch).state_dict()
        for sd in (r_sd, t_sd):
            sd["extra"] = sd.pop("2.bias")
        r_msg = t_msg = None
        try:
            mlp(raxis).load_state_dict(r_sd)
        except RuntimeError as e:
            r_msg = str(e)
        try:
            mlp(torch).load_state_dict(t_sd)
        except RuntimeError as e:
            t_msg = str(e)
        assert r_msg == t_msg

    def test_not_strict_returns_keys(self):
        r_sd, t_sd = mlp(raxis).state_dict(), mlp(torch).state_dict()
        for sd in (r_sd, t_sd):
            sd["extra"] = sd.pop("2.bias")
        r_keys = mlp(raxis).load_state_dict(r_sd, strict=False)
        t_keys = mlp(torch).load_state_dict(t_sd, strict=False)
        assert (r_keys.missing_keys, r_keys.unexpected_keys) == (t_keys.missing_keys, t_keys.unexpected_keys)

    def test_size_mismatch_raises_even_if_not_strict(self):
        sd = mlp(raxis, hidden=5).state_dict()
        with pytest.raises(RuntimeError, match=r"size mismatch for 0.weight: copying a param with shape \(5, 6\)"):
            mlp(raxis).load_state_dict(sd, strict=False)


class TestOptimizerStateDict:

    @pytest.mark.parametrize("name, make_opt", OPTIMIZERS, ids=[n for n, _ in OPTIMIZERS])
    def test_layout_matches_torch(self, name, make_opt):
        r_model, t_model = mlp(raxis, out_features=1), mlp(torch, out_features=1)
        copy_params(r_model, t_model)
        r_opt, t_opt = make_opt(raxis, r_model.parameters()), make_opt(torch, t_model.parameters())
        x, y = randf_centered(16, 6, seed=1), randf_centered(16, 1, seed=2)
        train_both(r_model, t_model, r_opt, t_opt, nn.MSELoss(), torch.nn.MSELoss(),
                   x, y, (raxis.float32, torch.float32), steps=2)
        r_sd, t_sd = r_opt.state_dict(), t_opt.state_dict()
        assert r_sd["state"].keys() == t_sd["state"].keys()
        for idx, r_state in r_sd["state"].items():
            assert r_state.keys() == t_sd["state"][idx].keys()
            for key, value in r_state.items():
                if key != "step":
                    check_match(value, t_sd["state"][idx][key], atol=1e-6)
        [r_group], [t_group] = r_sd["param_groups"], t_sd["param_groups"]
        assert r_group["params"] == t_group["params"]
        assert all(r_group[k] == t_group[k] for k in r_group)

    @pytest.mark.parametrize("name, make_opt", OPTIMIZERS, ids=[n for n, _ in OPTIMIZERS])
    def test_resume_training(self, name, make_opt):
        # 3 steps, checkpoint, 3 more steps in new objects == 6 steps in a row
        x, y = randf_centered(16, 6, seed=1), randf_centered(16, 1, seed=2)
        xs, ys = raxis.tensor(x), raxis.tensor(y)

        def train(model, opt, steps):
            losses = []
            for _ in range(steps):
                opt.zero_grad()
                loss = nn.functional.mse_loss(model(xs), ys)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            return losses

        model = mlp(raxis, out_features=1)
        opt = make_opt(raxis, model.parameters())
        expected = train(model, opt, 6)

        model = mlp(raxis, out_features=1)
        opt = make_opt(raxis, model.parameters())
        first = train(model, opt, 3)
        model_sd, opt_sd = model.state_dict(), opt.state_dict()

        resumed_model = mlp(raxis, out_features=1)
        resumed_model.load_state_dict(model_sd)
        resumed_opt = make_opt(raxis, resumed_model.parameters())
        resumed_opt.load_state_dict(opt_sd)
        np.testing.assert_allclose(first + train(resumed_model, resumed_opt, 3), expected, rtol=1e-6)

    def test_loaded_state_is_a_copy(self):
        model = mlp(raxis)
        opt = raxis.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        model(raxis.ones((2, 6))).sum().backward()
        opt.step()
        other = raxis.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        other.load_state_dict(opt.state_dict())
        buf = opt.state[model[0].weight]["momentum_buffer"]
        assert other.state[model[0].weight]["momentum_buffer"]._cdata != buf._cdata

    def test_loads_hyperparameters_keeps_params(self):
        model = mlp(raxis)
        opt = raxis.optim.SGD(two_groups(raxis, model), lr=0.1)
        sd = opt.state_dict()
        sd["param_groups"][1]["lr"] = 0.5
        new_opt = raxis.optim.SGD(two_groups(raxis, model), lr=0.1)
        new_opt.load_state_dict(sd)
        assert new_opt.param_groups[1]["lr"] == 0.5
        assert new_opt.param_groups[1]["params"][0] is model[2].weight

    def test_group_size_mismatch(self):
        model = mlp(raxis)
        sd = raxis.optim.SGD(model[0].parameters(), lr=0.1).state_dict()
        with pytest.raises(ValueError):
            raxis.optim.SGD(model.parameters(), lr=0.1).load_state_dict(sd)


# ===========================================================================
# ACTIVATION MODULES
# ===========================================================================

ACTIVATIONS = [
    ("ReLU", lambda lib: lib.nn.ReLU()),
    ("LeakyReLU", lambda lib: lib.nn.LeakyReLU(0.2)),
    ("SELU", lambda lib: lib.nn.SELU()),
    ("GELU", lambda lib: lib.nn.GELU()),
    ("GELU tanh", lambda lib: lib.nn.GELU(approximate="tanh")),
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


# ===========================================================================
# CONVOLUTION AND POOLING MODULES
# ===========================================================================

CONV_MODULES = [
    ("conv2d", lambda lib: lib.nn.Conv2d(3, 8, 3, padding=1), (2, 3, 7, 7)),
    ("conv2d_groups", lambda lib: lib.nn.Conv2d(4, 6, (3, 2), stride=2, groups=2, bias=False), (2, 4, 9, 8)),
    ("conv2d_same", lambda lib: lib.nn.Conv2d(2, 4, 3, padding="same", dilation=2), (1, 2, 8, 8)),
    ("conv1d", lambda lib: lib.nn.Conv1d(3, 5, 4, stride=2, padding="valid"), (2, 3, 11)),
    ("conv2d_unbatched", lambda lib: lib.nn.Conv2d(3, 4, 3), (3, 6, 6)),
    # Even kernel: the right side gets one more pad than the left
    ("conv2d_same_even", lambda lib: lib.nn.Conv2d(2, 3, (2, 4), padding="same"), (1, 2, 7, 6)),
    ("conv1d_same_even", lambda lib: lib.nn.Conv1d(2, 3, 4, padding="same", dilation=3), (2, 2, 9)),
    ("conv2d_reflect", lambda lib: lib.nn.Conv2d(3, 4, 3, padding=(1, 2), padding_mode="reflect"), (2, 3, 6, 6)),
    ("conv2d_replicate_same", lambda lib: lib.nn.Conv2d(2, 3, 4, padding="same", padding_mode="replicate"),
     (1, 2, 5, 5)),
    ("conv1d_circular", lambda lib: lib.nn.Conv1d(2, 3, 3, padding=2, padding_mode="circular"), (2, 2, 7)),
]


@pytest.mark.parametrize("name, make, x_shape", CONV_MODULES, ids=[n for n, _, _ in CONV_MODULES])
def test_conv_modules(name, make, x_shape):
    r_conv, t_conv = make(raxis), make(torch)
    assert repr(r_conv) == repr(t_conv)
    copy_params(r_conv, t_conv)
    data = randf_centered(*x_shape)
    r = raxis.tensor(data, requires_grad=True)
    t = torch.tensor(data, requires_grad=True)
    r_out, t_out = r_conv(r), t_conv(t)
    check_match(r_out, t_out, atol=1e-5, context=name)
    grad = randf_centered(*t_out.shape, seed=9)
    r_out.backward(raxis.tensor(grad))
    t_out.backward(torch.from_numpy(grad))
    check_match(r.grad, t.grad, atol=1e-5, context=f"{name} input grad")
    for (r_name, r_param), (t_name, t_param) in zip(r_conv.named_parameters(), t_conv.named_parameters()):
        assert r_name == t_name
        check_match(r_param.grad, t_param.grad, atol=1e-4, context=f"{name} {r_name} grad")


def test_conv_init():
    raxis.manual_seed(0)
    conv = nn.Conv2d(8, 16, 3)
    assert tuple(conv.weight.shape) == (16, 8, 3, 3)
    assert list(conv.state_dict()) == ["weight", "bias"]
    # kaiming_uniform_ with a = sqrt(5) is U(-1/sqrt(fan_in), 1/sqrt(fan_in))
    bound = 1 / math.sqrt(8 * 3 * 3)
    # The max of n draws of |U(-b, b)| exceeds q * b with probability 1 - q^n:
    # 1152 weights reach 0.9 b almost surely, the 16 biases reach 0.5 b with
    # probability 1 - 2^-16
    for param, q in ((conv.weight, 0.9), (conv.bias, 0.5)):
        values = np.array(param.detach())
        assert np.abs(values).max() <= bound
        assert np.abs(values).max() > q * bound


def test_conv_module_errors():
    with pytest.raises(ValueError, match="in_channels must be divisible by groups"):
        nn.Conv2d(3, 6, 3, groups=2)
    with pytest.raises(ValueError, match="Invalid padding string"):
        nn.Conv2d(3, 6, 3, padding="full")
    with pytest.raises(ValueError, match="not supported for strided convolutions"):
        nn.Conv2d(3, 6, 3, stride=2, padding="same")
    with pytest.raises(ValueError, match="padding_mode must be one of"):
        nn.Conv2d(3, 6, 3, padding_mode="mirror")


POOL_MODULES = [
    ("max", lambda lib: lib.nn.MaxPool2d(2)),
    ("max_ceil", lambda lib: lib.nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=True)),
    ("max_dilation", lambda lib: lib.nn.MaxPool2d(2, stride=1, dilation=2)),
    ("avg", lambda lib: lib.nn.AvgPool2d(2)),
    ("avg_no_pad_count", lambda lib: lib.nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)),
]


@pytest.mark.parametrize("name, make", POOL_MODULES, ids=[n for n, _ in POOL_MODULES])
def test_pool_modules(name, make):
    r_pool, t_pool = make(raxis), make(torch)
    assert repr(r_pool) == repr(t_pool)
    data = randf_centered(2, 3, 9, 9)
    r = raxis.tensor(data, requires_grad=True)
    t = torch.tensor(data, requires_grad=True)
    r_out, t_out = r_pool(r), t_pool(t)
    check_match(r_out, t_out, context=name)
    grad = randf_centered(*t_out.shape, seed=9)
    r_out.backward(raxis.tensor(grad))
    t_out.backward(torch.from_numpy(grad))
    check_match(r.grad, t.grad, atol=1e-6, context=f"{name} grad")


def test_max_pool_return_indices():
    data = randf_centered(2, 3, 8, 8)
    r_out, r_idx = nn.MaxPool2d(2, return_indices=True)(raxis.tensor(data))
    t_out, t_idx = torch.nn.MaxPool2d(2, return_indices=True)(torch.tensor(data))
    check_match(r_out, t_out)
    check_match(r_idx, t_idx, atol=0)


def cnn(lib):
    lib.manual_seed(0)
    return lib.nn.Sequential(
        lib.nn.Conv2d(1, 4, 3, padding=1), lib.nn.ReLU(), lib.nn.MaxPool2d(2),
        lib.nn.Conv2d(4, 8, 3), lib.nn.ReLU(), lib.nn.AvgPool2d(2),
        lib.nn.Flatten(), lib.nn.Linear(8, 3))


def test_cnn_training_matches_torch():
    r_model, t_model = cnn(raxis), cnn(torch)
    copy_params(r_model, t_model)
    x = randf_centered(16, 1, 10, 10, seed=1)
    y = np.random.RandomState(2).randint(0, 3, size=16).astype(np.int64)
    r_losses, t_losses = train_both(
        r_model, t_model,
        raxis.optim.SGD(r_model.parameters(), lr=0.05, momentum=0.9),
        torch.optim.SGD(t_model.parameters(), lr=0.05, momentum=0.9),
        nn.CrossEntropyLoss(), torch.nn.CrossEntropyLoss(), x, y, (raxis.long, torch.long))
    np.testing.assert_allclose(r_losses, t_losses, rtol=1e-4, atol=1e-5)
    assert r_losses[-1] < r_losses[0]


# ===========================================================================
# BATCH NORM MODULES
# ===========================================================================

BATCH_NORM_MODULES = [
    ("bn2d", lambda lib: lib.nn.BatchNorm2d(4), (8, 4, 5, 5)),
    ("bn1d_2d_input", lambda lib: lib.nn.BatchNorm1d(6), (32, 6)),
    ("bn1d_3d_input", lambda lib: lib.nn.BatchNorm1d(3, momentum=0.3), (4, 3, 7)),
    ("cumulative_average", lambda lib: lib.nn.BatchNorm2d(3, momentum=None), (4, 3, 4, 4)),
    ("no_affine", lambda lib: lib.nn.BatchNorm2d(3, affine=False), (4, 3, 4, 4)),
    ("no_running_stats", lambda lib: lib.nn.BatchNorm1d(5, track_running_stats=False), (16, 5)),
]


@pytest.mark.parametrize("name, make, x_shape", BATCH_NORM_MODULES, ids=[n for n, _, _ in BATCH_NORM_MODULES])
def test_batch_norm_modules(name, make, x_shape):
    r_bn, t_bn = make(raxis), make(torch)
    assert repr(r_bn) == repr(t_bn)
    assert list(r_bn.state_dict()) == list(t_bn.state_dict())
    # A few training steps update the running statistics, then eval uses them
    for step in range(3):
        data = randf_centered(*x_shape, seed=step) * 2 + 1
        r = raxis.tensor(data, requires_grad=True)
        t = torch.tensor(data, requires_grad=True)
        r_out, t_out = r_bn(r), t_bn(t)
        check_match(r_out, t_out, atol=1e-5, context=f"{name} train step {step}")
        grad = randf_centered(*x_shape, seed=10 + step)
        r_out.backward(raxis.tensor(grad))
        t_out.backward(torch.from_numpy(grad))
        check_match(r.grad, t.grad, atol=1e-5, context=f"{name} input grad")
    for (key, r_value), t_value in zip(r_bn.state_dict().items(), t_bn.state_dict().values()):
        check_match(r_value, t_value, atol=1e-6, context=f"{name} {key}")
    r_bn.eval()
    t_bn.eval()
    data = randf_centered(*x_shape, seed=99)
    check_match(r_bn(raxis.tensor(data)), t_bn(torch.tensor(data)), atol=1e-5, context=f"{name} eval")


def test_batch_norm_module_errors():
    with pytest.raises(ValueError, match="expected 4D input"):
        nn.BatchNorm2d(3)(raxis.randn(4, 3))
    with pytest.raises(ValueError, match="expected 2D or 3D input"):
        nn.BatchNorm1d(3)(raxis.randn(4, 3, 2, 2))
    with pytest.raises(ValueError, match="Expected more than 1 value per channel"):
        nn.BatchNorm1d(3)(raxis.randn(1, 3))
    # One sample is fine in eval: the running statistics are used
    bn = nn.BatchNorm1d(3)
    bn.eval()
    assert tuple(bn(raxis.randn(1, 3)).shape) == (1, 3)


# ===========================================================================
# LAYER NORM MODULE
# ===========================================================================

LAYER_NORM_MODULES = [
    ("last_dim", lambda lib: lib.nn.LayerNorm(8), (4, 6, 8)),
    ("two_dims", lambda lib: lib.nn.LayerNorm((5, 6), eps=1e-3), (3, 5, 6)),
    ("no_bias", lambda lib: lib.nn.LayerNorm(7, bias=False), (5, 7)),
    ("no_affine", lambda lib: lib.nn.LayerNorm(7, elementwise_affine=False), (5, 7)),
]


@pytest.mark.parametrize("name, make, x_shape", LAYER_NORM_MODULES, ids=[n for n, _, _ in LAYER_NORM_MODULES])
def test_layer_norm_modules(name, make, x_shape):
    r_ln, t_ln = make(raxis), make(torch)
    assert repr(r_ln) == repr(t_ln)
    assert list(r_ln.state_dict()) == list(t_ln.state_dict())
    # Non-trivial parameters, so their gradients are checked through them
    for (_, r_param), (_, t_param) in zip(r_ln.named_parameters(), t_ln.named_parameters()):
        values = randf_centered(*t_param.shape, seed=3) + 1
        with torch.no_grad():
            t_param.copy_(torch.from_numpy(values))
        with raxis.no_grad():
            r_param.copy_(raxis.tensor(values))
    data = randf_centered(*x_shape) * 3 + 1
    r = raxis.tensor(data, requires_grad=True)
    t = torch.tensor(data, requires_grad=True)
    r_out, t_out = r_ln(r), t_ln(t)
    check_match(r_out, t_out, atol=1e-5, context=name)
    grad = randf_centered(*x_shape, seed=9)
    r_out.backward(raxis.tensor(grad))
    t_out.backward(torch.from_numpy(grad))
    check_match(r.grad, t.grad, atol=1e-5, context=f"{name} input grad")
    for (r_name, r_param), (_, t_param) in zip(r_ln.named_parameters(), t_ln.named_parameters()):
        check_match(r_param.grad, t_param.grad, atol=1e-4, context=f"{name} {r_name} grad")


def cnn_bn(lib):
    lib.manual_seed(0)
    return lib.nn.Sequential(
        lib.nn.Conv2d(1, 4, 3, padding="same", bias=False), lib.nn.BatchNorm2d(4),
        lib.nn.ReLU(), lib.nn.MaxPool2d(2),
        lib.nn.Flatten(), lib.nn.Linear(4 * 5 * 5, 8), lib.nn.BatchNorm1d(8), lib.nn.ReLU(),
        lib.nn.Linear(8, 3))


def test_cnn_batch_norm_training_matches_torch():
    # SGD, not Adam: the bias of the Linear before BatchNorm1d has a zero
    # gradient (BatchNorm subtracts the mean), so its computed gradient is
    # rounding noise that Adam normalizes into steps of size lr, different in
    # each library. It does not change the loss, but it moves running_mean.
    r_model, t_model = cnn_bn(raxis), cnn_bn(torch)
    copy_params(r_model, t_model)
    x = randf_centered(16, 1, 10, 10, seed=1)
    y = np.random.RandomState(2).randint(0, 3, size=16).astype(np.int64)
    r_losses, t_losses = train_both(
        r_model, t_model,
        raxis.optim.SGD(r_model.parameters(), lr=0.05, momentum=0.9),
        torch.optim.SGD(t_model.parameters(), lr=0.05, momentum=0.9),
        nn.CrossEntropyLoss(), torch.nn.CrossEntropyLoss(), x, y, (raxis.long, torch.long))
    np.testing.assert_allclose(r_losses, t_losses, rtol=1e-4, atol=1e-5)
    assert r_losses[-1] < r_losses[0]
    r_model.eval()
    t_model.eval()
    check_match(r_model(raxis.tensor(x)), t_model(torch.from_numpy(x)), atol=1e-4, context="eval")


# ===========================================================================
# GRADIENT CLIPPING
# ===========================================================================

def grads_pair(seed=1):
    """The same gradients on raxis and torch parameters; the last one has none."""
    pairs = []
    for lib in (raxis, torch):
        params = [lib.nn.Parameter(lib.zeros(5, 3)), lib.nn.Parameter(lib.zeros(7)), lib.nn.Parameter(lib.zeros(2))]
        rng = np.random.RandomState(seed)
        for p in params[:2]:
            p.grad = lib.tensor(rng.randn(*p.shape).astype(np.float32))
        pairs.append(params)
    return pairs


class TestClipGrad:

    @pytest.mark.parametrize("norm_type", [2.0, 1, 3.5, math.inf, 0])
    @pytest.mark.parametrize("max_norm", [0.1, 100.0])
    def test_clip_grad_norm_matches_torch(self, norm_type, max_norm):
        r_params, t_params = grads_pair()
        r_norm = nn.utils.clip_grad_norm_(r_params, max_norm, norm_type)
        t_norm = torch.nn.utils.clip_grad_norm_(t_params, max_norm, norm_type)
        check_match(r_norm, t_norm, atol=1e-5)
        for r, t in zip(r_params[:2], t_params[:2]):
            check_match(r.grad, t.grad, atol=1e-6)
        assert r_params[2].grad is None

    def test_clip_grad_value_matches_torch(self):
        r_params, t_params = grads_pair()
        nn.utils.clip_grad_value_(r_params, 0.5)
        torch.nn.utils.clip_grad_value_(t_params, 0.5)
        for r, t in zip(r_params[:2], t_params[:2]):
            check_match(r.grad, t.grad)

    def test_no_gradients(self):
        assert nn.utils.clip_grad_norm_([nn.Parameter(raxis.zeros(2))], 1.0).item() == 0.0

    def test_non_finite_norm(self):
        p = nn.Parameter(raxis.zeros(2))
        p.grad = raxis.tensor([1.0, math.inf])
        with pytest.raises(RuntimeError, match="is non-finite, so it cannot be clipped"):
            nn.utils.clip_grad_norm_(p, 1.0, error_if_nonfinite=True)
