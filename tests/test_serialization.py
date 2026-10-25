import io
import pickle
import zipfile
from collections import OrderedDict

import numpy as np
import pytest
import raxis
import torch


def saved_by_raxis(obj):
    buf = io.BytesIO()
    raxis.save(obj, buf)
    return buf.getvalue()


def saved_by_torch(obj):
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def views_of_one_storage(lib):
    """A base tensor and views with offsets and transposed strides: all of
    them must come back sharing one storage."""
    base = lib.tensor(np.arange(12, dtype=np.float32).reshape(4, 3))
    return OrderedDict(base=base, t=base.transpose(0, 1), rows=base[1:], col=base[:, 2])


class TestTorchLoadsRaxis:

    def test_views_dtypes_and_containers(self):
        obj = {
            "views": views_of_one_storage(raxis),
            "scalar": raxis.tensor(2.5),
            "int64": raxis.arange(3),
            "bool": raxis.tensor([True, False]),
            "float64": raxis.tensor(np.ones((2, 2))),
            "meta": {"iter": 7, "name": "gpt", "bytes": b"\x00\xff", "shape": (1, 2), "lr": 1e-3},
        }
        loaded = torch.load(io.BytesIO(saved_by_raxis(obj)), weights_only=True)
        for name, tensor in obj["views"].items():
            t = loaded["views"][name]
            assert t.shape == tensor.shape and t.stride() == tensor.strides
            assert t.storage_offset() == tensor.storage_offset()
            np.testing.assert_array_equal(t.numpy(), tensor.numpy())
        assert len({t.untyped_storage().data_ptr() for t in loaded["views"].values()}) == 1
        assert loaded["scalar"].item() == 2.5
        assert loaded["int64"].dtype == torch.int64 and loaded["bool"].dtype == torch.bool
        assert loaded["float64"].dtype == torch.float64
        assert loaded["meta"] == obj["meta"]

    def test_requires_grad_and_parameter(self):
        obj = {"leaf": raxis.randn(3, requires_grad=True), "param": raxis.nn.Parameter(raxis.randn(2))}
        loaded = torch.load(io.BytesIO(saved_by_raxis(obj)), weights_only=True)
        assert loaded["leaf"].requires_grad
        assert isinstance(loaded["param"], torch.nn.Parameter)

    def test_records_aligned_for_mmap(self, tmp_path):
        path = tmp_path / "ckpt.pt"
        raxis.save({"w": raxis.randn(5, 7), "b": raxis.arange(3)}, path)
        with zipfile.ZipFile(path) as zf:
            # The archive's root directory is named after the file, as torch does
            assert all(name.startswith("ckpt/") for name in zf.namelist())
            raw = path.read_bytes()
            for info in zf.infolist():
                name_len, extra_len = np.frombuffer(raw[info.header_offset + 26:info.header_offset + 30], np.uint16)
                assert (info.header_offset + 30 + name_len + extra_len) % 64 == 0, info.filename
        loaded = torch.load(path, mmap=True, weights_only=True)
        assert loaded["w"].shape == (5, 7)


class TestRaxisLoadsTorch:

    def test_views_stay_shared(self):
        loaded = raxis.load(io.BytesIO(saved_by_torch(views_of_one_storage(torch))))
        expected = views_of_one_storage(raxis)
        for name, tensor in expected.items():
            assert loaded[name].strides == tensor.strides
            np.testing.assert_array_equal(loaded[name].numpy(), tensor.numpy())
        loaded["base"][0, 2] = -1.0
        assert loaded["t"][2, 0].item() == -1.0 and loaded["col"][0].item() == -1.0

    def test_module_state_dict(self):
        torch.manual_seed(0)
        t_model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.LayerNorm(8), torch.nn.Linear(8, 2))
        r_model = raxis.nn.Sequential(raxis.nn.Linear(4, 8), raxis.nn.LayerNorm(8), raxis.nn.Linear(8, 2))
        r_model.load_state_dict(raxis.load(io.BytesIO(saved_by_torch(t_model.state_dict()))))
        x = np.random.RandomState(0).randn(3, 4).astype(np.float32)
        np.testing.assert_allclose(r_model(raxis.tensor(x)).detach().numpy(),
                                   t_model(torch.tensor(x)).detach().numpy(), atol=1e-6)

    def test_unsupported_dtype(self):
        with pytest.raises(RuntimeError, match="float16 tensors"):
            raxis.load(io.BytesIO(saved_by_torch({"h": torch.ones(2, dtype=torch.float16)})))

    def test_only_allowlisted_globals(self):
        data = saved_by_torch({"array": np.zeros(2)})
        with pytest.raises(pickle.UnpicklingError, match="unsupported global"):
            raxis.load(io.BytesIO(data))
        # Trusted files may hold other objects
        assert raxis.load(io.BytesIO(data), weights_only=False)["array"].shape == (2,)


class TestRoundTrip:

    def test_tied_weights_and_optimizer(self):
        emb = raxis.nn.Embedding(5, 3)
        head = raxis.nn.Linear(3, 5, bias=False)
        head.weight = emb.weight
        model = raxis.nn.ModuleList([emb, head])
        opt = raxis.optim.AdamW(model.parameters(), lr=1e-2)
        head(emb(raxis.tensor([1, 2]))).sum().backward()
        opt.step()
        ckpt = raxis.load(io.BytesIO(saved_by_raxis({"model": model.state_dict(keep_vars=True),
                                                     "optimizer": opt.state_dict(), "iter": 1})))
        # Saved once, loaded as one storage
        assert ckpt["model"]["0.weight"].untyped_storage()._cdata == ckpt["model"]["1.weight"].untyped_storage()._cdata
        assert isinstance(ckpt["model"]["0.weight"], raxis.nn.Parameter)
        state = ckpt["optimizer"]["state"][0]
        np.testing.assert_array_equal(state["exp_avg"].numpy(), opt.state_dict()["state"][0]["exp_avg"].numpy())
        assert ckpt["optimizer"]["param_groups"][0]["lr"] == 1e-2

    def test_errors(self):
        with pytest.raises(RuntimeError, match="not a zip checkpoint"):
            raxis.load(io.BytesIO(b"not a checkpoint"))
        with pytest.raises(ValueError, match="only has CPU tensors"):
            raxis.load(io.BytesIO(saved_by_raxis({})), map_location="cuda")
