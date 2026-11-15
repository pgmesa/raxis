import pytest
import raxis
import torch

from test_utils import check_match, device, make_pair, randf  # noqa: F401 (fixture)


CONVERSIONS = ["float", "double", "int", "long", "bool", "byte"]


@pytest.mark.parametrize("name", CONVERSIONS)
def test_dtype_conversions(name, device):
    # byte() of a negative float is where platforms disagree: PyTorch's rule
    # wraps through int64, but its macOS arm64 wheel gives 0. Non-negative
    # values for byte here; test_byte_of_negative_floats_wraps pins the rule
    lo = 0 if name == "byte" else -3
    r, t = make_pair(randf(3, 4, lo=lo, hi=3), device)
    r_out, t_out = getattr(r, name)(), getattr(t, name)()
    check_match(r_out, t_out)
    assert str(r_out.dtype).split(".")[-1] == str(t_out.dtype).split(".")[-1]


def test_byte_of_negative_floats_wraps(device):
    # From: c10/util/TypeCast.h: float -> uint8 goes through int64, so the
    # result wraps the same on every CPU and on the GPU (a direct cast of a
    # negative float to an unsigned type is undefined behaviour in C++)
    r = raxis.tensor([-1.5, -2.5, 300.7, 3.2], device=device)
    assert r.byte().tolist() == [255, 254, 44, 3]


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


class TestResize:

    def test_shrink_and_grow_keep_the_data(self, device):
        # clone(): a CPU make_pair shares a NumPy array, which cannot grow
        r, t = (x.clone() for x in make_pair(randf(6), device))
        for x in (r, t):
            address = x.data_ptr()
            x.resize_(2, 2)
            x.resize_(0)
            x.resize_(3)       # every size fits in the storage: the same memory
            assert x.data_ptr() == address
            x.resize_(4, 4)    # grows: the 6 old values are copied over
        # Compared at the end: t.numpy() makes torch's storage not resizable
        check_match(r.flatten()[:6], t.flatten()[:6])

    def test_grown_storage_is_shared_by_its_views(self, device):
        r = raxis.arange(4.0, device=device)
        v = r[:2]
        r.resize_(1000)
        r[1] = 7.0
        assert v.tolist() == [0.0, 7.0]

    def test_returns_self(self):
        x = raxis.empty(3)
        assert x.resize_(4) is x and x.resize_((2, 2)) is x and x.shape == (2, 2)

    def test_refuses_tensors_that_require_grad(self):
        for lib in (raxis, torch):
            with pytest.raises(RuntimeError, match="cannot resize variables that require grad"):
                lib.ones(3, requires_grad=True).resize_(4)

    def test_numpy_view_freezes_the_storage(self):
        # Growing would free the memory the array reads, as in torch
        for lib in (raxis, torch):
            x = lib.zeros(4)
            x.numpy()
            with pytest.raises(RuntimeError, match="Trying to resize storage that is not resizable"):
                x.resize_(1000)

    def test_refuses_memory_it_does_not_own(self):
        import numpy as np
        for lib in (raxis, torch):
            x = lib.from_numpy(np.zeros(2, dtype=np.float32))
            x.resize_(1)       # shrinking needs no memory
            with pytest.raises(RuntimeError, match="Trying to resize storage that is not resizable"):
                x.resize_(1000)

