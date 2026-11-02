import pytest
import raxis
import torch

from test_utils import check_match, device, make_pair, randf  # noqa: F401 (fixture)


CONVERSIONS = ["float", "double", "int", "long", "bool", "byte"]


@pytest.mark.parametrize("name", CONVERSIONS)
def test_dtype_conversions(name, device):
    r, t = make_pair(randf(3, 4, lo=-3, hi=3), device)
    r_out, t_out = getattr(r, name)(), getattr(t, name)()
    check_match(r_out, t_out)
    assert str(r_out.dtype).split(".")[-1] == str(t_out.dtype).split(".")[-1]


def test_conversion_to_the_same_dtype_returns_self():
    # torch returns the same object; raxis a new handle to the same TensorImpl
    r = raxis.zeros(3)
    assert r.float()._cdata == r._cdata


def test_type_and_type_as(device):
    r, t = make_pair(randf(2, 3), device)
    assert r.type() == t.type().replace("torch", "raxis")
    assert r.long().type() == t.long().type().replace("torch", "raxis")
    check_match(r.type(raxis.int64), t.type(torch.int64))
    r_int, t_int = r.long(), t.long()
    check_match(r.type_as(r_int), t.type_as(t_int))


@pytest.mark.parametrize("name", ["new_zeros", "new_ones", "new_empty"])
def test_new_factories(name, device):
    r, t = make_pair(randf(2, 3).astype("float64"), device)
    for args in [((4, 5),), (4, 5), (3,)]:
        r_out, t_out = getattr(r, name)(*args), getattr(t, name)(*args)
        assert r_out.shape == tuple(t_out.shape)
        assert r_out.dtype == raxis.float64 and r_out.device.type == t_out.device.type
        if name != "new_empty":
            check_match(r_out, t_out)
    r_out = getattr(r, name)((2,), dtype=raxis.int32, requires_grad=False)
    assert r_out.dtype == raxis.int32


def test_new_full_and_new_tensor(device):
    r, t = make_pair(randf(2, 3), device)
    check_match(r.new_full((2, 2), 7), t.new_full((2, 2), 7))
    check_match(r.new_full((2,), 1.5, dtype=raxis.float64), t.new_full((2,), 1.5, dtype=torch.float64))
    check_match(r.new_tensor([[1, 2], [3, 4]]), t.new_tensor([[1, 2], [3, 4]]))
    r_out = r.new_tensor([1, 2], dtype=raxis.int64, requires_grad=False)
    assert r_out.dtype == raxis.int64
    assert r.new_zeros(2, requires_grad=True).requires_grad


def test_size_information():
    r, t = make_pair(randf(2, 3, 4))
    for r_x, t_x in [(r, t), (r.double(), t.double()), (r.bool(), t.bool()), (r[:, 1], t[:, 1])]:
        assert r_x.element_size() == t_x.element_size()
        assert r_x.itemsize == t_x.itemsize
        assert r_x.nbytes == t_x.nbytes
        assert r_x.nelement() == t_x.nelement()
        assert r_x.ndimension() == t_x.ndimension()
        assert r_x.is_floating_point() == t_x.is_floating_point()
