"""The CUDA backend, compared with PyTorch on CUDA: transfers, the caching
allocator, the kernels and paths the device-parametrized suites may not
reach, TF32, checkpoints of GPU tensors, and training the MNIST MLP, a CNN
and a small GPT on the GPU."""
import gc
import io
import subprocess
import sys

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
            raxis.ones((3,), device="cuda").cumsum(0)

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


class TestCat:

    @pytest.mark.parametrize("dim", [0, 1])
    @pytest.mark.parametrize("n", [3, 300_000])
    def test_cat(self, dim, n):
        # n = 3 takes the serial path on the CPU, which reads on the host
        a, b = make((n, 4), seed=0), make((n, 4), seed=1)
        r = raxis.cat([raxis.from_numpy(a).cuda(), raxis.from_numpy(b).cuda()], dim)
        t = torch.cat([torch.from_numpy(a).cuda(), torch.from_numpy(b).cuda()], dim)
        check(r, t)

    def test_cat_mixed_layouts_and_dtypes(self):
        a, b = make((4, 3), seed=0), make((3, 4), "float64", seed=1)
        r = raxis.cat([raxis.from_numpy(a).cuda(), raxis.from_numpy(b).cuda().transpose(0, 1)])
        t = torch.cat([torch.from_numpy(a).cuda(), torch.from_numpy(b).cuda().transpose(0, 1)])
        check(r, t)

    @pytest.mark.parametrize("dim", [0, 1, 2])
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_cat_many_inputs(self, dim, dtype):
        # More inputs than one batched launch holds (64), of different
        # lengths along dim, some empty, and the legacy 1-D empty tensor
        xs = []
        for i in range(150):
            shape = [3, 4, 5]
            shape[dim] = i % 4
            xs.append(make(tuple(shape), dtype, seed=i))
        r = raxis.cat([raxis.from_numpy(x).cuda() for x in xs] + [raxis.zeros(0, dtype=dtype).cuda()], dim)
        t = torch.cat([torch.from_numpy(x).cuda() for x in xs] + [torch.zeros(0, dtype=getattr(torch, dtype)).cuda()], dim)
        check(r, t, atol=0, rtol=0)

    def test_stack_of_scalars(self):
        # clip_grad_norm_ stacks one 0-d norm per parameter: views of one vector
        r_vec, t_vec = raxis.arange(100.0).cuda(), torch.arange(100.0).cuda()
        r = raxis.stack([r_vec[i] for i in range(99, -1, -1)])
        t = torch.stack([t_vec[i] for i in range(99, -1, -1)])
        check(r, t, atol=0, rtol=0)

    @pytest.mark.parametrize("dim", [0, 1, 2])
    def test_stack(self, dim):
        xs = [make((5, 6), seed=i) for i in range(3)]
        r = raxis.stack([raxis.from_numpy(x).cuda() for x in xs], dim)
        t = torch.stack([torch.from_numpy(x).cuda() for x in xs], dim)
        check(r, t)


class TestHostOnlyOps:
    """Ops whose only implementation reads the data on the host must raise
    on CUDA tensors instead of dereferencing device pointers (a crash)."""

    @pytest.mark.parametrize("op", [
        lambda x: raxis.multinomial(x.abs(), 2, replacement=True),
        lambda x: x.var(),
        lambda x: x.index_add_(0, raxis.tensor([0, 1]).cuda(), x[:2]),
    ])
    def test_raises(self, op):
        x = raxis.from_numpy(make((4, 4))).cuda()
        with pytest.raises(NotImplementedError, match="CUDA|cuda"):
            op(x)

    def test_stub_name_in_message(self):
        with pytest.raises(NotImplementedError, match="cumsum_stub"):
            raxis.ones((3,), device="cuda").cumsum(0)


class TestKernels:
    """GPU paths the shared suites do not reach."""

    def test_conv_chunks(self):
        # Images whose columns exceed the 128 MB budget are unfolded in
        # several chunks; the weight gradient accumulates over them. Checked
        # against float64: each weight gradient sums 24500 float32 products,
        # and torch's cuDNN float32 result is ~100x further from it than
        # raxis's
        x = make((5, 64, 70, 70))
        w = make((8, 64, 3, 3), seed=1)
        out = []
        for lib, dtype in ((raxis, "float32"), (torch, "float64")):
            xd = lib.tensor(x.astype(dtype), device="cuda", requires_grad=True)
            wd = lib.tensor(w.astype(dtype), device="cuda", requires_grad=True)
            y = lib.nn.functional.conv2d(xd, wd, padding=1)
            y.backward(lib.ones_like(y))
            out.append([np.array(t.detach().cpu(), dtype=np.float64) for t in (y, xd.grad, wd.grad)])
        for r, ref in zip(*out):
            np.testing.assert_allclose(r, ref, rtol=0, atol=1e-6 * np.abs(ref).max())

    def test_index_select(self):
        x = make((5, 7, 3))
        idx = np.array([6, 0, 2, 2])
        for dim in range(3):
            i = idx[idx < x.shape[dim]]
            r = raxis.tensor(x, device="cuda").index_select(dim, raxis.tensor(i, device="cuda"))
            t = torch.tensor(x, device="cuda").index_select(dim, torch.tensor(i, device="cuda"))
            check(r, t, atol=0)

    @pytest.mark.parametrize("op", ["tril", "triu"])
    @pytest.mark.parametrize("k", [-2, 0, 3])
    def test_triangular(self, op, k):
        x = make((2, 5, 6))
        r = getattr(raxis.tensor(x, device="cuda"), op)(k)
        t = getattr(torch.tensor(x, device="cuda"), op)(k)
        check(r, t, atol=0)
        # A transposed matrix is read through its strides
        r = getattr(raxis.tensor(x[0], device="cuda").transpose(0, 1), op)(k)
        t = getattr(torch.tensor(x[0], device="cuda").transpose(0, 1), op)(k)
        check(r, t, atol=0)

    def test_masked_fill_where_clamp(self):
        x = make((4, 9))
        x[1, 2] = np.nan
        r, t = raxis.tensor(x, device="cuda"), torch.tensor(x, device="cuda")
        check(r.masked_fill(r > 0, -1.0), t.masked_fill(t > 0, -1.0), atol=0)
        check(raxis.where(r > 0, r, r * 2), torch.where(t > 0, t, t * 2), atol=0)
        check(r.clamp(-1, 1), t.clamp(-1, 1), atol=0)
        check(r.clamp(min=0.5), t.clamp(min=0.5), atol=0)
        lo, hi = make((4, 9), seed=1) - 2, make((4, 9), seed=2) + 2
        check(r.clamp(raxis.tensor(lo, device="cuda"), raxis.tensor(hi, device="cuda")),
              t.clamp(torch.tensor(lo, device="cuda"), torch.tensor(hi, device="cuda")), atol=0)

    @pytest.mark.parametrize("args", [(10,), (0, 1, 0.1), (-3, 7, 2)])
    def test_arange(self, args):
        # Same values as torch on each device: CUDA computes in float, the CPU in double
        check(raxis.arange(*args, device="cuda"), torch.arange(*args, device="cuda"), atol=0, rtol=0)


class TestTF32:

    def test_defaults(self):
        # As torch: matmuls in full float32, convolutions in TF32. A fresh
        # process, since conftest.py turns the convolution flag off
        code = ("import raxis; b = raxis.backends; "
                "print(b.cuda.matmul.allow_tf32, b.cudnn.allow_tf32, raxis.get_float32_matmul_precision())")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        assert out.stdout.split() == ["False", "True", "highest"]

    def test_flags(self):
        matmul = raxis.backends.cuda.matmul
        try:
            matmul.allow_tf32 = True
            assert matmul.allow_tf32 and raxis.get_float32_matmul_precision() == "high"
            raxis.set_float32_matmul_precision("highest")
            assert not matmul.allow_tf32
            with pytest.raises(AttributeError):
                matmul.allow_tf33 = True
            with pytest.raises(RuntimeError, match="Invalid precision"):
                raxis.set_float32_matmul_precision("low")
        finally:
            matmul.allow_tf32 = False

    def test_matmul_and_conv_use_tf32(self):
        a, b = make((256, 256)), make((256, 256), seed=1)
        x, w = make((4, 32, 16, 16)), make((32, 32, 3, 3), seed=1)
        ra, rb = raxis.tensor(a, device="cuda"), raxis.tensor(b, device="cuda")
        rx, rw = raxis.tensor(x, device="cuda"), raxis.tensor(w, device="cuda")
        exact_mm = (ra @ rb).cpu().numpy()
        exact_conv = raxis.nn.functional.conv2d(rx, rw).cpu().numpy()
        try:
            raxis.backends.cuda.matmul.allow_tf32 = True
            raxis.backends.cudnn.allow_tf32 = True
            tf32_mm = (ra @ rb).cpu().numpy()
            tf32_conv = raxis.nn.functional.conv2d(rx, rw).cpu().numpy()
            # float64 never uses TF32
            exact64 = (ra.to(raxis.float64) @ rb.to(raxis.float64)).cpu().numpy()
        finally:
            raxis.backends.cuda.matmul.allow_tf32 = False
            raxis.backends.cudnn.allow_tf32 = False
        for exact, tf32 in [(exact_mm, tf32_mm), (exact_conv, tf32_conv)]:
            assert not np.array_equal(exact, tf32)
            # 10-bit mantissa inputs: about 1e-3 relative error per product
            np.testing.assert_allclose(tf32, exact, rtol=0, atol=2e-3 * np.abs(exact).max())
        np.testing.assert_allclose(exact64, a.astype(np.float64) @ b, rtol=1e-12)


class TestSerialization:

    def test_roundtrip(self):
        x = raxis.tensor(make((3, 4)), device="cuda")
        state = {"x": x, "view": x[1:], "cpu": raxis.ones((2,))}
        buf = io.BytesIO()
        raxis.save(state, buf)
        buf.seek(0)
        loaded = raxis.load(buf)
        # Back on the device each one was saved from, views still sharing
        assert loaded["x"].is_cuda and loaded["view"].is_cuda and not loaded["cpu"].is_cuda
        np.testing.assert_array_equal(loaded["x"].cpu().numpy(), x.cpu().numpy())
        loaded["x"][1, 0] = 5.0
        assert loaded["view"][0, 0].item() == 5.0
        for map_location in ["cpu", raxis.device("cpu"), {"cuda:0": "cpu"}]:
            buf.seek(0)
            assert not raxis.load(buf, map_location=map_location)["x"].is_cuda
        buf.seek(0)
        assert raxis.load(buf, map_location="cuda")["cpu"].is_cuda

    def test_torch_reads_and_writes(self):
        data = make((5,))
        buf = io.BytesIO()
        raxis.save({"x": raxis.tensor(data, device="cuda")}, buf)
        buf.seek(0)
        t = torch.load(buf, weights_only=True)["x"]
        assert t.is_cuda
        np.testing.assert_array_equal(t.cpu().numpy(), data)
        buf = io.BytesIO()
        torch.save({"x": torch.tensor(data, device="cuda")}, buf)
        buf.seek(0)
        r = raxis.load(buf)["x"]
        assert r.is_cuda
        np.testing.assert_array_equal(r.cpu().numpy(), data)

    def test_cuda_storage_has_no_buffer(self):
        # numpy would read GPU memory from the host (pybind11 turns the
        # error into a BufferError)
        storage = raxis.ones((3,), device="cuda").untyped_storage()
        with pytest.raises(BufferError):
            memoryview(storage)


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


class TestReduce:
    """The paths of reduce.cuh that the shared suites may not reach."""

    def test_sum_across_blocks(self):
        # Few outputs, long reductions: several blocks per output and the
        # last one to finish combines the staged partial sums
        data = make((4_000_000,), "float64")
        r = raxis.from_numpy(data).cuda().sum()
        np.testing.assert_allclose(r.item(), data.sum(), rtol=1e-12)
        data = make((4, 1_000_000), "float64")
        r = raxis.from_numpy(data).cuda().sum(1)
        np.testing.assert_allclose(r.cpu().numpy(), data.sum(1), rtol=1e-12)

    @pytest.mark.parametrize("shape, dim", [((3000, 5), 0), ((7, 3000), 1), ((64, 33, 17), (0, 2))])
    def test_layouts(self, shape, dim):
        # Reducing the contiguous dim (threads of a warp share an output)
        # and an outer one (threads take neighbouring outputs)
        data = make(shape, "float64")
        r = raxis.from_numpy(data).cuda().sum(dim)
        np.testing.assert_allclose(r.cpu().numpy(), data.sum(dim), rtol=1e-12)

    def test_argmax_ties_and_nan(self):
        data = np.zeros((2, 3000), dtype=np.float32)
        data[0, [5, 1700, 2999]] = 3.0
        data[1, [10, 2000]] = np.nan
        r = raxis.from_numpy(data).cuda()
        t = torch.from_numpy(data).cuda()
        # The first of equal maxima, and NaN as the maximum
        assert r.argmax(1).tolist() == t.argmax(1).tolist() == [5, 10]
        assert np.isnan(r.amax(1).cpu().numpy()[1])

    def test_all_any(self):
        x = raxis.from_numpy(make((5, 300), "bool")).cuda()
        t = torch.from_numpy(make((5, 300), "bool")).cuda()
        assert x.all(1).tolist() == t.all(1).tolist()
        assert x.any(0).tolist() == t.any(0).tolist()


class TestSoftmax:

    @pytest.mark.parametrize("op", ["softmax", "log_softmax"])
    def test_long_rows(self, op):
        # Over 1024 elements a row no longer fits a warp: one block per row
        data = make((3, 5000), "float32")
        r = getattr(raxis, op)(raxis.from_numpy(data).cuda(), 1)
        t = getattr(torch, op)(torch.from_numpy(data).cuda(), 1)
        check(r, t, atol=1e-6, rtol=1e-5)


class TestDropout:

    @pytest.mark.parametrize("n", [1, 63, 64, 1000, 100_003])
    def test_same_mask_as_cpu(self, n):
        # The GPU kernel draws each element from the same Philox block and
        # word as the CPU kernel, so a seed gives the same mask on both
        x = raxis.ones((n,))
        raxis.manual_seed(7)
        cpu = raxis.nn.functional.dropout(x, 0.3)
        raxis.manual_seed(7)
        gpu = raxis.nn.functional.dropout(x.cuda(), 0.3)
        np.testing.assert_array_equal(gpu.cpu().numpy(), cpu.numpy())


class TestRandom:

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    @pytest.mark.parametrize("n", [1, 15, 64, 1000, 100_003])
    def test_uniform_same_as_cpu(self, n, dtype):
        # Same Philox layout as the CPU kernels: bitwise equal for a seed
        dt = getattr(raxis, dtype)
        out = []
        for device in ("cpu", "cuda"):
            raxis.manual_seed(5)
            out.append(raxis.empty(n, dtype=dt, device=device).uniform_(-2, 3).cpu().numpy())
        np.testing.assert_array_equal(out[1], out[0])

    @pytest.mark.parametrize("dtype", ["int64", "int32", "uint8", "bool"])
    def test_randint_same_as_cpu(self, dtype):
        high = 2 if dtype == "bool" else 100
        out = []
        for device in ("cpu", "cuda"):
            raxis.manual_seed(5)
            out.append(raxis.randint(0, high, (1001,), dtype=getattr(raxis, dtype), device=device).cpu().numpy())
        np.testing.assert_array_equal(out[1], out[0])

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_normal_close_to_cpu(self, dtype):
        # Same uniforms; CUDA's log / sin / cos differ from Sleef's in the last bits
        out = []
        for device in ("cpu", "cuda"):
            raxis.manual_seed(5)
            out.append(raxis.empty(10_001, dtype=getattr(raxis, dtype), device=device).normal_(1, 2).cpu().numpy())
        np.testing.assert_allclose(out[1], out[0], rtol=0, atol=1e-4)

    def test_multinomial(self):
        p = np.tile(np.array([[0.1, 0.2, 0.7], [0.5, 0.0, 0.5]], dtype=np.float32), (2000, 1))
        samples = raxis.multinomial(raxis.tensor(p, device="cuda"), 1).cpu().numpy().ravel()
        rows = np.arange(4000) % 2
        freq0 = np.bincount(samples[rows == 0], minlength=3) / 2000
        freq1 = np.bincount(samples[rows == 1], minlength=3) / 2000
        np.testing.assert_allclose(freq0, [0.1, 0.2, 0.7], atol=0.04)
        assert freq1[1] == 0
        # Without replacement every row is a permutation prefix
        draws = raxis.multinomial(raxis.ones(50, 10, device="cuda"), 10).cpu().numpy()
        assert all(sorted(row) == list(range(10)) for row in draws)


def attention_both(B, H, L, S, E, mask=None, sliced=False, **kwargs):
    """Output and q, k, v gradients of SDPA in both libraries. sliced builds
    q, k, v as nanoGPT does: [B, L, H, E] slices of a wider projection,
    viewed as [B, H, L, E]."""
    rng = np.random.RandomState(0)
    q, k, v = (rng.randn(B, H, n, E).astype(np.float32) for n in (L, S, S))
    g = rng.randn(B, H, L, E).astype(np.float32)
    out = []
    for lib in (raxis, torch):
        def leaf(a):
            if sliced:
                wide = np.concatenate([a.transpose(0, 2, 1, 3)] * 2, axis=-1)
                return lib.tensor(wide, device="cuda")[..., :E].transpose(1, 2).requires_grad_(True)
            return lib.tensor(a, device="cuda", requires_grad=True)
        qt, kt, vt = leaf(q), leaf(k), leaf(v)
        m = None if mask is None else lib.tensor(mask, device="cuda")
        y = lib.nn.functional.scaled_dot_product_attention(qt, kt, vt, attn_mask=m, **kwargs)
        y.backward(lib.tensor(g, device="cuda"))
        out.append((y.grad_fn.name(), [t.detach().cpu().numpy() for t in (y, qt.grad, kt.grad, vt.grad)]))
    return out


class TestAttention:
    """The fused GPU kernel (ops/cuda/attention_kernel.cu) against torch's
    memory-efficient attention, which float32 takes on the GPU in both."""

    @pytest.mark.parametrize("B, H, L, S, E", [
        (2, 3, 64, 64, 64),      # one tile
        (2, 3, 100, 100, 64),    # partial tiles
        (1, 2, 37, 129, 32),     # three key tiles, short query tile
        (1, 2, 130, 70, 16),     # more queries than keys
        (1, 1, 5, 3, 8),
    ])
    @pytest.mark.parametrize("is_causal", [False, True])
    def test_matches_torch(self, B, H, L, S, E, is_causal):
        (r_name, r), (t_name, t) = attention_both(B, H, L, S, E, is_causal=is_causal)
        assert r_name == t_name == "ScaledDotProductEfficientAttentionBackward0"
        for r_val, t_val in zip(r, t):
            np.testing.assert_allclose(r_val, t_val, rtol=0, atol=2e-5 * np.abs(t_val).max())

    def test_nanogpt_layout(self):
        (_, r), (_, t) = attention_both(2, 2, 77, 77, 64, sliced=True, is_causal=True)
        for r_val, t_val in zip(r, t):
            np.testing.assert_allclose(r_val, t_val, rtol=0, atol=2e-5 * np.abs(t_val).max())

    @pytest.mark.parametrize("mask_shape, dtype", [
        ((50, 90), "bool"), ((2, 1, 50, 90), "float32"), ((1, 2, 1, 90), "float32"), ((2, 2, 50, 90), "bool"),
    ])
    def test_masks(self, mask_shape, dtype):
        rng = np.random.RandomState(1)
        mask = rng.randn(*mask_shape) > -1 if dtype == "bool" else rng.randn(*mask_shape).astype(np.float32)
        (_, r), (_, t) = attention_both(2, 2, 50, 90, 64, mask=mask)
        for r_val, t_val in zip(r, t):
            np.testing.assert_allclose(r_val, t_val, rtol=0, atol=2e-5 * np.abs(t_val).max())

    def test_fully_masked_row_is_zero(self):
        mask = np.ones((3, 5), dtype=bool)
        mask[1] = False
        (_, r), _ = attention_both(1, 1, 3, 5, 8, mask=mask)
        assert np.all(r[0][0, 0, 1] == 0) and np.all(np.isfinite(r[1]))

    def test_large_head_dim_takes_the_math_path(self):
        (r_name, _), _ = attention_both(1, 1, 8, 8, 72)
        assert r_name != "ScaledDotProductEfficientAttentionBackward0"

    def test_dropout(self):
        # The backward redraws the forward's mask from the Philox counters:
        # dV = A^T dO with A the dropped attention matrix, recovered column by
        # column through value = one-hot rows (the output is linear in value)
        rng = np.random.RandomState(2)
        L, S, E = 4, 70, 8
        q, k, v = (raxis.tensor(rng.randn(1, 1, n, E).astype(np.float32), device="cuda", requires_grad=True)
                   for n in (L, S, S))
        w = rng.randn(L, E).astype(np.float32)
        F = raxis.nn.functional
        raxis.manual_seed(11)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.5)
        (y * raxis.tensor(w, device="cuda")).sum().backward()
        attn = np.zeros((L, S), dtype=np.float32)
        for j in range(S):
            probe = np.zeros((1, 1, S, E), dtype=np.float32)
            probe[0, 0, j, 0] = 1.0
            raxis.manual_seed(11)
            with raxis.no_grad():
                yj = F.scaled_dot_product_attention(q, k, raxis.tensor(probe, device="cuda"), dropout_p=0.5)
            attn[:, j] = yj.cpu().numpy()[0, 0, :, 0]
        assert 0.3 < np.mean(attn == 0) < 0.7
        np.testing.assert_allclose(v.grad.cpu().numpy()[0, 0], attn.T @ w, atol=1e-5)

    def test_graph_lifetime_and_double_backward(self):
        # The node saves the output as an output (no reference cycle), and
        # has no double backward (as torch's)
        x = raxis.rand((1, 2, 3, 4), device="cuda", requires_grad=True)
        y = raxis.nn.functional.scaled_dot_product_attention(x, x, x)
        assert y.grad_fn.name() == "ScaledDotProductEfficientAttentionBackward0"
        assert y._use_count() == 1
        g, = raxis.autograd.grad(y.pow(3).sum(), x, create_graph=True)
        with pytest.raises(RuntimeError, match="'_scaled_dot_product_efficient_attention_backward' is not"):
            raxis.autograd.grad(g.sum(), x)

    def test_backward_is_deterministic(self):
        x = [raxis.randn(2, 3, 200, 64, device="cuda", requires_grad=True) for _ in range(3)]
        grads = []
        for _ in range(2):
            for t in x:
                t.grad = None
            raxis.manual_seed(3)
            raxis.nn.functional.scaled_dot_product_attention(*x, dropout_p=0.1, is_causal=True).sum().backward()
            grads.append([t.grad.cpu().numpy() for t in x])
        for a, b in zip(*grads):
            np.testing.assert_array_equal(a, b)


class TestFusedAdam:

    @pytest.mark.parametrize("name, kwargs", [
        ("Adam", dict(weight_decay=0.1)), ("AdamW", dict(weight_decay=0.1)), ("AdamW", dict(maximize=True)),
    ])
    def test_bitwise_torch(self, name, kwargs):
        # 40 tensors: more than one launch holds (36), and odd sizes for the
        # scalar path; the 21M-element tensor spans more blocks than a launch
        # holds (320 chunks of 65536), so it continues in the next launch
        rng = np.random.RandomState(0)
        shapes = [(1536, 384), (384,), (65, 384), (3, 5), (70001,), (1,), (0,)] * 5 + [(321 * 65536 + 7,)] + [(7,)] * 4
        data = [rng.randn(*s).astype(np.float32) for s in shapes]
        grads = [[rng.randn(*s).astype(np.float32) for s in shapes] for _ in range(2)]
        out = []
        for lib in (raxis, torch):
            ps = [lib.nn.Parameter(lib.tensor(d, device="cuda")) for d in data]
            opt = getattr(lib.optim, name)(ps, lr=1e-2, fused=True, **kwargs)
            for g in grads:
                for p, gi in zip(ps, g):
                    p.grad = lib.tensor(gi, device="cuda")
                opt.step()
            out.append([p.detach().cpu().numpy() for p in ps])
        for r, t in zip(*out):
            np.testing.assert_array_equal(r, t)


class TestForeach:
    """_foreach_norm / _foreach_mul_: the fused kernels for ord 0, 1, 2 and inf
    follow PyTorch's order of additions, so they are bitwise torch's."""

    # More tensors than one launch holds (80), odd sizes for the unaligned
    # path, and one tensor of more chunks than a launch holds (320)
    SHAPES = [(1536, 384), (384,), (65, 384), (3, 5), (70001,), (1,), (0,)] * 12 + [(321 * 65536 + 7,), (7,)]

    def lists(self, dtype, with_empty=True):
        rng = np.random.RandomState(0)
        data = [rng.randn(*s).astype(dtype) for s in self.SHAPES if with_empty or np.prod(s) > 0]
        return ([raxis.tensor(d, device="cuda") for d in data], [torch.tensor(d, device="cuda") for d in data])

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    @pytest.mark.parametrize("ord", [0, 1, 2, float("inf"), 3])
    def test_norm(self, dtype, ord):
        r_list, t_list = self.lists(dtype, with_empty=ord != float("inf"))
        r_list[4][3] = float("nan")
        t_list[4][3] = float("nan")
        r = np.array([n.cpu().numpy() for n in raxis._C._foreach_norm(r_list, ord)])
        t = np.array([n.cpu().numpy() for n in torch._foreach_norm(t_list, ord)])
        if ord == 3:   # composed per tensor, as torch's slow path
            np.testing.assert_allclose(r, t, rtol=1e-6 if dtype == "float32" else 1e-13)
        else:
            np.testing.assert_array_equal(r, t)

    def test_norm_infinity_of_empty_raises(self):
        with pytest.raises(RuntimeError, match="infinity norm on an empty tensor"):
            raxis._C._foreach_norm([raxis.zeros(0, device="cuda")], float("inf"))

    def test_mul_(self):
        r_list, t_list = self.lists("float32")
        r_versions = [t._version for t in r_list]
        raxis._C._foreach_mul_(r_list, raxis.tensor(0.25, device="cuda"))
        torch._foreach_mul_(t_list, torch.tensor(0.25, device="cuda"))
        for r, t, v in zip(r_list, t_list, r_versions):
            np.testing.assert_array_equal(r.cpu().numpy(), t.cpu().numpy())
            assert r._version == v + 1

    def test_mixed_devices_and_dtypes(self):
        # clip_grad_norm_ groups by (device, dtype); the slow path takes the CPU
        rng = np.random.RandomState(1)
        data = [rng.randn(30, 7), rng.randn(5).astype(np.float32), rng.randn(4, 4).astype(np.float32)]
        devices = ["cuda", "cpu", "cuda"]
        for foreach in (None, False):
            norms = []
            for lib in (raxis, torch):
                ps = [lib.nn.Parameter(lib.tensor(d, device=dev)) for d, dev in zip(data, devices)]
                for p in ps:
                    p.grad = p.detach() * 3
                norm = lib.nn.utils.clip_grad_norm_(ps, 1.0, foreach=foreach)
                norms.append((norm.item(), [p.grad.cpu().numpy() for p in ps]))
            (r_norm, r_grads), (t_norm, t_grads) = norms
            assert r_norm == pytest.approx(t_norm, rel=1e-6)
            for r, t in zip(r_grads, t_grads):
                np.testing.assert_allclose(r, t, rtol=1e-6)


def mlp():
    nn = raxis.nn
    return nn.Sequential(
        nn.Flatten(), nn.Linear(784, 200), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(200, 100), nn.ReLU(), nn.Linear(100, 10), nn.LogSoftmax(dim=1))


class TestModule:

    def test_to_keeps_parameters(self):
        # The same Parameter objects take the GPU data (torch's param.data =
        # ...), so an optimizer created before .to() updates the moved model
        raxis.manual_seed(0)
        model = mlp()
        params = list(model.parameters())
        optimizer = raxis.optim.SGD(model.parameters(), lr=0.1)
        assert model.to("cuda") is model
        assert all(p is q for p, q in zip(params, model.parameters()))
        assert all(p.is_cuda for p in model.parameters())
        before = params[0].cpu().numpy()
        x = raxis.ones((4, 28, 28), device="cuda")
        model(x).sum().backward()
        assert all(p.grad.is_cuda for p in params)
        optimizer.step()
        assert not np.array_equal(params[0].cpu().numpy(), before)

    def test_buffers_move_with_the_model(self):
        bn = raxis.nn.BatchNorm2d(3).cuda()
        assert bn.running_mean.is_cuda and bn.num_batches_tracked.is_cuda
        assert bn.running_mean is bn._buffers["running_mean"]
        bn(raxis.ones((2, 3, 4, 4), device="cuda"))
        assert bn.num_batches_tracked.item() == 1

    def test_grads_move_with_the_model(self):
        model = raxis.nn.Linear(3, 2)
        model(raxis.ones((1, 3))).sum().backward()
        model.cuda()
        assert model.weight.grad.is_cuda
        model.cpu()
        assert not model.weight.is_cuda and not model.weight.grad.is_cuda


class TestTraining:

    def train(self, device, steps):
        raxis.manual_seed(0)
        model = mlp().to(device)
        optimizer = raxis.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        loss_fn = raxis.nn.NLLLoss()
        rng = np.random.default_rng(0)
        losses, allocated = [], []
        for _ in range(steps):
            x = raxis.tensor(rng.random((64, 28, 28), dtype=np.float32), device=device)
            y = raxis.tensor(rng.integers(0, 10, 64), device=device)
            loss = loss_fn(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if device == "cuda":
                allocated.append(raxis.cuda.memory_allocated())
        return losses, allocated

    def test_memory_stays_flat(self):
        # synapx leaked GPU memory every step; here, after the first steps
        # create the momentum buffers, every step frees what it allocates
        _, allocated = self.train("cuda", 30)
        assert len(set(allocated[5:])) == 1, allocated

    def test_matches_cpu(self):
        # Same seed, same dropout masks: only the matmul rounding differs
        cpu, _ = self.train("cpu", 10)
        gpu, _ = self.train("cuda", 10)
        np.testing.assert_allclose(gpu, cpu, rtol=1e-4)


def cnn():
    nn = raxis.nn
    return nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.AvgPool2d(2),
        nn.Flatten(), nn.Dropout(0.25), nn.Linear(16 * 7 * 7, 10))


class TinyGPT(raxis.nn.Module):
    """The layers of nanoGPT's model: embeddings, layer norm, causal
    attention, GELU MLP and a tied output head."""

    def __init__(self, vocab=17, block=12, dim=16, heads=2):
        super().__init__()
        nn = raxis.nn
        self.heads = heads
        self.wte = nn.Embedding(vocab, dim)
        self.wpe = nn.Embedding(block, dim)
        self.ln1, self.ln2, self.ln_f = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.wte.weight

    def forward(self, idx):
        B, T = idx.shape
        pos = raxis.arange(0, T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        q, k, v = self.attn(self.ln1(x)).split(x.shape[-1], dim=2)
        q, k, v = (t.view(B, T, self.heads, -1).transpose(1, 2) for t in (q, k, v))
        y = raxis.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).contiguous().view(B, T, -1))
        x = x + self.mlp(self.ln2(x))
        return self.head(self.ln_f(x))


class TestTrainingModels:

    def train(self, make_model, make_batch, loss_fn, device, steps, optimizer_cls):
        raxis.manual_seed(0)
        model = make_model().to(device)
        optimizer = optimizer_cls(model.parameters())
        rng = np.random.default_rng(0)
        losses, allocated = [], []
        for _ in range(steps):
            x, y = (raxis.tensor(a, device=device) for a in make_batch(rng))
            loss = loss_fn(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            raxis.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
            if device == "cuda":
                allocated.append(raxis.cuda.memory_allocated())
        return losses, allocated

    def cnn_run(self, device, steps):
        return self.train(
            cnn, lambda rng: (rng.random((32, 1, 28, 28), dtype=np.float32), rng.integers(0, 10, 32)),
            raxis.nn.functional.cross_entropy, device, steps,
            lambda params: raxis.optim.SGD(params, lr=0.05, momentum=0.9))

    def gpt_run(self, device, steps):
        def batch(rng):
            tokens = rng.integers(0, 17, (4, 13))
            return tokens[:, :-1], tokens[:, 1:]
        return self.train(
            TinyGPT, batch, lambda logits, y: raxis.nn.functional.cross_entropy(logits.view(-1, 17), y.reshape(-1)),
            device, steps, lambda params: raxis.optim.AdamW(params, lr=1e-2))

    @pytest.mark.parametrize("model", ["cnn", "gpt"])
    def test_matches_cpu(self, model):
        run = getattr(self, f"{model}_run")
        cpu, _ = run("cpu", 8)
        gpu, _ = run("cuda", 8)
        np.testing.assert_allclose(gpu, cpu, rtol=1e-4)

    @pytest.mark.parametrize("model", ["cnn", "gpt"])
    def test_memory_stays_flat(self, model):
        _, allocated = getattr(self, f"{model}_run")("cuda", 20)
        assert len(set(allocated[5:])) == 1, allocated
