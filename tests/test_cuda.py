"""The CUDA backend: transfers, the caching allocator and the elementwise
kernels it has so far, compared with PyTorch on CUDA."""
import gc

import numpy as np
import pytest
import raxis
import torch


pytestmark = pytest.mark.skipif(not raxis.cuda.is_available(), reason="no CUDA device")

DTYPES = ["float32", "float64", "int32", "int64", "uint8", "bool"]


def make(shape, dtype="float32", seed=0):
    rng = np.random.RandomState(seed)
    data = rng.randn(*shape) * 4
    if dtype == "bool":
        return data > 0
    if dtype == "uint8":
        return np.abs(data).astype(np.uint8)
    return data.astype(dtype)


def check(r, t, atol=1e-6, rtol=1e-6):
    assert r.device.type == "cuda"
    assert r.dtype.name == str(t.dtype).replace("torch.", "")
    assert tuple(r.shape) == tuple(t.shape)
    np.testing.assert_allclose(r.cpu().numpy(), t.cpu().numpy(), atol=atol, rtol=rtol)


class TestTransfers:

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_roundtrip(self, dtype):
        data = make((5, 7), dtype)
        x = raxis.from_numpy(data)
        y = x.cuda()
        assert y.is_cuda and y.device == raxis.device("cuda:0")
        np.testing.assert_array_equal(y.cpu().numpy(), data)

    def test_to_device_strings(self):
        x = raxis.from_numpy(make((3,)))
        y = x.to("cuda")
        assert y.is_cuda
        # Same tensor (same TensorImpl) when nothing changes
        assert y.to("cuda")._cdata == y._cdata and y.cuda()._cdata == y._cdata
        assert y.to("cpu").device.type == "cpu"
        assert x.cpu()._cdata == x._cdata

    def test_to_device_and_dtype(self):
        data = make((4, 4), "float64")
        y = raxis.from_numpy(data).to("cuda", raxis.float32)
        assert y.dtype == raxis.float32 and y.is_cuda
        np.testing.assert_array_equal(y.cpu().numpy(), data.astype(np.float32))

    @pytest.mark.parametrize("view", ["transpose", "slice", "expand", "offset"])
    def test_non_contiguous(self, view):
        data = make((6, 8))
        x = raxis.from_numpy(data)
        views = {
            "transpose": (lambda a: a.transpose(0, 1)),
            "slice": (lambda a: a[1:5, ::2]),
            "expand": (lambda a: a[:, :1].expand((6, 8))),
            "offset": (lambda a: a.reshape((48,))[3:]),
        }
        f = views[view]
        # To the GPU from a strided CPU view, back from a strided GPU view
        np.testing.assert_array_equal(f(x).cuda().cpu().numpy(), f(x).numpy())
        np.testing.assert_array_equal(f(x.cuda()).cpu().numpy(), f(x).numpy())

    def test_copy_with_cast(self):
        src = raxis.from_numpy(make((3, 5), "float64"))
        dst = raxis.zeros((3, 5), dtype=raxis.int32, device="cuda")
        dst.copy_(src)
        np.testing.assert_array_equal(dst.cpu().numpy(), make((3, 5), "float64").astype(np.int32))
        back = raxis.zeros((3, 5), dtype=raxis.float64)
        back.copy_(dst)
        np.testing.assert_array_equal(back.numpy(), dst.cpu().numpy().astype(np.float64))

    def test_device_to_device(self):
        y = raxis.from_numpy(make((4, 6))).cuda()
        z = y.transpose(0, 1).clone()
        np.testing.assert_array_equal(z.cpu().numpy(), make((4, 6)).T)
        w = y.to(raxis.float64)
        assert w.is_cuda and w.dtype == raxis.float64

    def test_item_tolist_repr(self):
        y = raxis.from_numpy(np.array([[1.5, 2.0], [3.0, 4.0]], dtype=np.float32)).cuda()
        assert y[1, 0].item() == 3.0
        assert y.tolist() == [[1.5, 2.0], [3.0, 4.0]]
        assert "device='cuda:0'" in repr(y)

    def test_numpy_raises(self):
        y = raxis.zeros((2,), device="cuda")
        with pytest.raises(TypeError, match="Tensor.cpu"):
            y.numpy()

    def test_to_backward(self):
        x = raxis.from_numpy(make((3, 4))).requires_grad_()
        y = x.cuda() * 3.0
        y.backward(raxis.ones((3, 4), device="cuda"))
        assert x.grad.device.type == "cpu"
        np.testing.assert_array_equal(x.grad.numpy(), np.full((3, 4), 3.0, dtype=np.float32))


class TestFactories:

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_full(self, dtype):
        value = True if dtype == "bool" else 3
        r = raxis.full((5, 3), value, dtype=getattr(raxis, dtype), device="cuda")
        t = torch.full((5, 3), value, dtype=getattr(torch, dtype), device="cuda")
        check(r, t)

    def test_zeros_ones(self):
        check(raxis.zeros((7, 9), device="cuda"), torch.zeros((7, 9), device="cuda"))
        check(raxis.ones((1000,), device="cuda"), torch.ones((1000,), device="cuda"))

    def test_fill_strided(self):
        y = raxis.zeros((4, 6), device="cuda")
        y[:, 1::2].fill_(5.0)
        t = torch.zeros((4, 6), device="cuda")
        t[:, 1::2].fill_(5.0)
        check(y, t)


BINARY_OPS = ["add", "sub", "mul", "div"]


class TestBinary:

    @pytest.mark.parametrize("op", BINARY_OPS)
    # Sizes around the vectorized kernel's blocks (128 threads x 8 elements)
    @pytest.mark.parametrize("n", [1, 7, 1023, 1024, 1025, 4099, 100_003])
    def test_contiguous(self, op, n):
        a, b = make((n,), seed=1), make((n,), seed=2) + 10
        r = getattr(raxis.from_numpy(a).cuda(), op)(raxis.from_numpy(b).cuda())
        t = getattr(torch.from_numpy(a).cuda(), op)(torch.from_numpy(b).cuda())
        check(r, t)

    @pytest.mark.parametrize("op", BINARY_OPS)
    def test_misaligned(self, op):
        # Starting at element 1: 4-byte aligned, not 16, so no float4 loads
        a, b = make((4097,), seed=1), make((4097,), seed=2) + 10
        r = getattr(raxis.from_numpy(a).cuda()[1:], op)(raxis.from_numpy(b).cuda()[:-1])
        t = getattr(torch.from_numpy(a).cuda()[1:], op)(torch.from_numpy(b).cuda()[:-1])
        check(r, t)

    @pytest.mark.parametrize("op", BINARY_OPS)
    @pytest.mark.parametrize("shapes", [((4, 1, 5), (3, 1)), ((1,), (6, 7)), ((2, 3, 4), (4,))])
    def test_broadcast(self, op, shapes):
        a, b = make(shapes[0], seed=1), make(shapes[1], seed=2) + 10
        r = getattr(raxis.from_numpy(a).cuda(), op)(raxis.from_numpy(b).cuda())
        t = getattr(torch.from_numpy(a).cuda(), op)(torch.from_numpy(b).cuda())
        check(r, t)

    @pytest.mark.parametrize("op", BINARY_OPS)
    def test_non_contiguous(self, op):
        a, b = make((33, 65), seed=1), make((65, 33), seed=2) + 10
        r = getattr(raxis.from_numpy(a).cuda(), op)(raxis.from_numpy(b).cuda().transpose(0, 1))
        t = getattr(torch.from_numpy(a).cuda(), op)(torch.from_numpy(b).cuda().transpose(0, 1))
        check(r, t)

    @pytest.mark.parametrize("op", BINARY_OPS)
    @pytest.mark.parametrize("scalar", [2, 2.5, -3])
    def test_python_scalar(self, op, scalar):
        a = make((100,), seed=1)
        r = getattr(raxis.from_numpy(a).cuda(), op)(scalar)
        t = getattr(torch.from_numpy(a).cuda(), op)(scalar)
        check(r, t)

    @pytest.mark.parametrize("op", ["__radd__", "__rsub__", "__rmul__", "__rtruediv__"])
    def test_reversed_scalar(self, op):
        a = make((100,), seed=1) + 10
        check(getattr(raxis.from_numpy(a).cuda(), op)(3.0), getattr(torch.from_numpy(a).cuda(), op)(3.0))

    def test_cpu_scalar_tensor(self):
        # A 0-dim CPU tensor is allowed next to a CUDA one, as in torch
        a = make((50,))
        r = raxis.from_numpy(a).cuda() * raxis.tensor(4.0)
        t = torch.from_numpy(a).cuda() * torch.tensor(4.0)
        check(r, t)

    @pytest.mark.parametrize("dtypes", [("int32", "float32"), ("int64", "float64"), ("uint8", "int32"), ("float32", "float64")])
    def test_type_promotion(self, dtypes):
        a, b = make((8, 9), dtypes[0], seed=1), make((8, 9), dtypes[1], seed=2)
        for op in ["add", "mul"]:
            r = getattr(raxis.from_numpy(a).cuda(), op)(raxis.from_numpy(b).cuda())
            t = getattr(torch.from_numpy(a).cuda(), op)(torch.from_numpy(b).cuda())
            check(r, t)

    @pytest.mark.parametrize("dtype", ["int32", "int64", "uint8"])
    def test_integer(self, dtype):
        a, b = make((300,), dtype, seed=1), make((300,), dtype, seed=2)
        for op in ["add", "sub", "mul", "div"]:
            if op == "sub" and dtype == "uint8":
                continue
            r = getattr(raxis.from_numpy(a).cuda(), op)(raxis.from_numpy(b).cuda())
            t = getattr(torch.from_numpy(a).cuda(), op)(torch.from_numpy(b).cuda())
            check(r, t)

    def test_bool(self):
        a, b = make((64,), "bool", seed=1), make((64,), "bool", seed=2)
        for op in ["add", "mul"]:
            r = getattr(raxis.from_numpy(a).cuda(), op)(raxis.from_numpy(b).cuda())
            t = getattr(torch.from_numpy(a).cuda(), op)(torch.from_numpy(b).cuda())
            check(r, t)

    def test_inplace_and_out_dtype(self):
        y = raxis.from_numpy(make((10,))).cuda()
        t = torch.from_numpy(make((10,))).cuda()
        y.add_(raxis.from_numpy(make((10,), seed=3)).cuda())
        t.add_(torch.from_numpy(make((10,), seed=3)).cuda())
        check(y, t)

    def test_device_mismatch(self):
        with pytest.raises(RuntimeError, match="same device"):
            raxis.ones((3,), device="cuda") + raxis.ones((3,))

    def test_missing_kernel(self):
        with pytest.raises(NotImplementedError, match="CUDA"):
            raxis.ones((3,), device="cuda").exp()

    def test_32bit_split(self):
        # 2.2e9 one-byte elements: offsets past INT32_MAX, so gpu_kernel
        # splits the iterator before launching
        n = 2_200_000_000
        try:
            y = raxis.zeros((n,), dtype=raxis.uint8, device="cuda")
        except RuntimeError as e:
            pytest.skip(f"not enough GPU memory: {e}")
        y.fill_(7)
        y.add_(1)
        assert y[:3].tolist() == [8, 8, 8]
        assert y[n // 2 - 1:n // 2 + 1].tolist() == [8, 8]
        assert y[-3:].tolist() == [8, 8, 8]
        del y
        gc.collect()
        raxis.cuda.empty_cache()


class TestCachingAllocator:

    def test_memory_allocated(self):
        gc.collect()
        base = raxis.cuda.memory_allocated()
        y = raxis.empty((1000,), device="cuda")      # 4000 bytes -> 4096 (512-byte blocks)
        assert raxis.cuda.memory_allocated() == base + 4096
        z = raxis.empty((10, 1000, 1000), device="cuda")  # 40 MB, its own segment
        assert raxis.cuda.memory_allocated() == base + 4096 + 40_000_000
        del y, z
        assert raxis.cuda.memory_allocated() == base

    def test_blocks_are_reused(self):
        gc.collect()
        x = raxis.ones((256, 256), device="cuda")
        for _ in range(3):
            x = x + 1.0
        reserved = raxis.cuda.memory_reserved()
        for _ in range(100):
            x = x * 1.0 + 1.0
        # Freed blocks go back to the pool: no new cudaMalloc
        assert raxis.cuda.memory_reserved() == reserved
        assert x[0, 0].item() == 104.0

    def test_peak_and_empty_cache(self):
        gc.collect()
        raxis.cuda.empty_cache()
        raxis.cuda.reset_peak_memory_stats()
        base = raxis.cuda.memory_allocated()
        y = raxis.empty((25, 1000, 1000), device="cuda")  # 100 MB
        del y
        assert raxis.cuda.max_memory_allocated() >= base + 100_000_000
        assert raxis.cuda.memory_allocated() == base
        reserved = raxis.cuda.memory_reserved()
        raxis.cuda.empty_cache()
        assert raxis.cuda.memory_reserved() <= reserved - 100_000_000

    def test_out_of_memory(self):
        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            raxis.empty((1 << 40,), dtype=raxis.uint8, device="cuda")
        # The allocator still works afterwards
        assert raxis.ones((4,), device="cuda").tolist() == [1.0] * 4

    def test_split_blocks_merge(self):
        gc.collect()
        raxis.cuda.empty_cache()
        before = raxis.cuda.memory_reserved()
        # Small tensors split 2 MiB segments into blocks. Freed in any order,
        # the blocks merge back into whole segments, which empty_cache returns
        ts = [raxis.empty((1000,), device="cuda") for _ in range(600)]
        assert raxis.cuda.memory_reserved() > before
        for i in list(range(0, 600, 2)) + list(range(1, 600, 2)):
            ts[i] = None
        raxis.cuda.empty_cache()
        assert raxis.cuda.memory_reserved() == before


class TestDevice:

    def test_info(self):
        assert raxis.cuda.device_count() >= 1
        assert raxis.cuda.current_device() == 0

    def test_synchronize(self):
        y = raxis.ones((1000, 1000), device="cuda") * 2.0
        raxis.cuda.synchronize()
        assert y[999, 999].item() == 2.0
