"""raxis.save and raxis.load, in PyTorch's checkpoint format.

A file written by torch.save (since 1.6) is an uncompressed zip:

    archive/data.pkl            the pickled object (dicts, numbers, tensors)
    archive/data/0, data/1 ...  the raw bytes of each storage
    archive/byteorder, version, .format_version, .storage_alignment

Tensors are not inside the pickle as data. Each one pickles as a call
torch._utils._rebuild_tensor_v2(storage, storage_offset, size, stride,
requires_grad, backward_hooks), and its storage as a persistent id
("storage", torch.FloatStorage, key, location, numel) that points to the
record data/<key>. Storages are written once, so views and tied weights stay
shared after loading.

raxis writes and reads exactly that, without importing torch: torch.load
reads raxis checkpoints (also with weights_only=True) and raxis.load reads
torch's. Loading is restricted to an allowlist of globals, as
torch.load(weights_only=True): a checkpoint cannot run arbitrary code.

Each storage records its device ("cpu", "cuda:0"); its bytes are always
written from a CPU copy. load puts every storage back on its device, or
where map_location says, as torch.load.

Not supported: the legacy (non-zip) format from before torch 1.6, dtypes
raxis does not have (float16, bfloat16, ...), whole pickled Modules.
"""

import io
import os
import pickle
import struct
import sys
import zipfile
from collections import OrderedDict
from typing import Any, BinaryIO

import raxis
from raxis._C import Tensor, UntypedStorage, _tensor_from_storage

__all__ = ["save", "load"]

# From: torch/serialization.py (DEFAULT_PROTOCOL, _get_storage_alignment)
DEFAULT_PROTOCOL = 2
STORAGE_ALIGNMENT = 64

# The typed storage class torch names for each dtype in the pickle
_STORAGE_NAME_BY_DTYPE = {
    "float32": "FloatStorage", "float64": "DoubleStorage", "int64": "LongStorage",
    "int32": "IntStorage", "uint8": "ByteStorage", "bool": "BoolStorage",
}
_DTYPE_BY_STORAGE_NAME = {name: getattr(raxis, dtype) for dtype, name in _STORAGE_NAME_BY_DTYPE.items()}
# Storage classes of torch dtypes raxis does not have, for the error message
_UNSUPPORTED_STORAGE_NAMES = {
    "HalfStorage": "float16", "BFloat16Storage": "bfloat16", "CharStorage": "int8",
    "ShortStorage": "int16", "ComplexFloatStorage": "complex64", "ComplexDoubleStorage": "complex128",
}


# -- Saving --

class _TorchGlobal:
    """Stands for torch.<module>.<name> in the pickle: written as a GLOBAL
    opcode, so torch.load finds its own function or class there."""

    def __init__(self, module: str, name: str):
        self.module = module
        self.name = name

    def __call__(self, *args):
        raise RuntimeError(f"{self.module}.{self.name} is a placeholder for torch.save's pickle")


_REBUILD_TENSOR_V2 = _TorchGlobal("torch._utils", "_rebuild_tensor_v2")
_REBUILD_PARAMETER = _TorchGlobal("torch._utils", "_rebuild_parameter")
_STORAGE_GLOBALS = {dtype: _TorchGlobal("torch", name) for dtype, name in _STORAGE_NAME_BY_DTYPE.items()}


class _StorageRef:
    """A storage as a tensor's pickle references it: becomes the persistent
    id ("storage", torch.<Type>Storage, key, location, numel)."""

    def __init__(self, key: str, dtype_name: str, location: str, numel: int):
        self.key = key
        self.dtype_name = dtype_name
        self.location = location
        self.numel = numel


# torch's location strings: "cpu", "cuda:0"
def _location(device) -> str:
    return "cpu" if device.type == "cpu" else f"{device.type}:{device.index}"


def _storage_bytes(storage: UntypedStorage):
    """The bytes of a storage on any device, as a buffer on the CPU."""
    if storage.device.type == "cpu":
        return memoryview(storage)
    as_bytes = _tensor_from_storage(storage, raxis.uint8, 0, (storage.nbytes(),), (1,))
    return memoryview(as_bytes.cpu().untyped_storage())


def _storage_to(storage: UntypedStorage, device) -> UntypedStorage:
    if device.type == "cpu":
        return storage
    as_bytes = _tensor_from_storage(storage, raxis.uint8, 0, (storage.nbytes(),), (1,))
    return as_bytes.to(device).untyped_storage()


# From: torch/serialization.py (_save, persistent_id) and torch/_tensor.py (_reduce_ex_internal)
#
# The C pickler checks that every global it writes can be imported, and
# torch may not be installed: the Python pickler lets _TorchGlobal write its
# GLOBAL opcode directly. data.pkl only holds the structure (the bytes go to
# the storage records), so its speed does not matter.
class _TorchPickler(pickle._Pickler):

    dispatch = pickle._Pickler.dispatch.copy()

    def __init__(self, file: BinaryIO, protocol: int):
        super().__init__(file, protocol)
        self.storages: dict[str, UntypedStorage] = {}
        self._keys: dict[int, str] = {}
        self._dtypes: dict[str, str] = {}

    def save_torch_global(self, obj: _TorchGlobal) -> None:
        if self.proto >= 4:
            self.save(obj.module)
            self.save(obj.name)
            self.write(pickle.STACK_GLOBAL)
        else:
            self.write(pickle.GLOBAL + obj.module.encode() + b"\n" + obj.name.encode() + b"\n")
        self.memoize(obj)

    dispatch[_TorchGlobal] = save_torch_global

    def persistent_id(self, obj: Any):
        if type(obj) is not _StorageRef:
            return None
        return ("storage", _STORAGE_GLOBALS[obj.dtype_name], obj.key, obj.location, obj.numel)

    def reducer_override(self, obj: Any):
        if not isinstance(obj, Tensor):
            return NotImplemented
        if isinstance(obj, raxis.nn.Parameter):
            return _REBUILD_PARAMETER, (obj.detach(), obj.requires_grad, OrderedDict())
        dtype_name = obj.dtype.name
        if dtype_name not in _STORAGE_NAME_BY_DTYPE:
            raise pickle.PicklingError(f"raxis.save: cannot save a tensor of dtype {dtype_name}")
        # One record per storage: views and tied weights keep sharing it
        storage = obj.untyped_storage()
        key = self._keys.setdefault(storage._cdata, str(len(self._keys)))
        if self._dtypes.setdefault(key, dtype_name) != dtype_name:
            raise RuntimeError("Cannot save multiple tensors or storages that view the same data as different types")
        self.storages[key] = storage
        ref = _StorageRef(key, dtype_name, _location(storage.device), storage.nbytes() // obj.dtype.itemsize)
        args = (ref, obj.storage_offset(), tuple(obj.shape), tuple(obj.strides), obj.requires_grad, OrderedDict())
        return _REBUILD_TENSOR_V2, args


def _write_record(zf: zipfile.ZipFile, archive_start: int, name: str, data) -> None:
    """Stored (uncompressed) record whose data starts at a multiple of 64
    bytes from the start of the archive, so it can be memory mapped. As
    torch's PyTorchStreamWriter, the local header gets an extra field "FB"
    with the padding."""
    # From: caffe2/serialize/inline_container.cc (getOffset, getPadding)
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    size = memoryview(data).nbytes
    cursor = zf.fp.tell() - archive_start
    # local header (30) + name + the 4 bytes of the "FB" field header, and
    # the zip64 sizes zipfile adds after the extra field for large records
    start = cursor + 30 + len(name.encode()) + 4 + (20 if size > zipfile.ZIP64_LIMIT else 0)
    padding = -start % STORAGE_ALIGNMENT
    info.extra = b"FB" + struct.pack("<H", padding) + b"Z" * padding
    zf.writestr(info, data)


def save(obj: Any, f, pickle_protocol: int = DEFAULT_PROTOCOL) -> None:
    """Saves obj (tensors, state dicts, nested containers of numbers and
    strings) to the path or binary file f, in torch.save's format."""
    # From: torch/serialization.py (_save)
    data_buf = io.BytesIO()
    pickler = _TorchPickler(data_buf, pickle_protocol)
    pickler.dump(obj)

    if isinstance(f, (str, os.PathLike)):
        # torch names the archive's root directory after the file
        stem = os.path.splitext(os.path.basename(os.fspath(f)))[0]
        archive_name = stem if stem.isascii() and stem else "archive"
        with open(f, "wb") as file:
            _write_archive(file, archive_name, data_buf.getvalue(), pickler.storages)
    else:
        _write_archive(f, "archive", data_buf.getvalue(), pickler.storages)


def _write_archive(file: BinaryIO, archive_name: str, data_pkl: bytes, storages: dict) -> None:
    archive_start = file.tell()
    with zipfile.ZipFile(file, "w", zipfile.ZIP_STORED) as zf:
        def record(name, data):
            _write_record(zf, archive_start, f"{archive_name}/{name}", data)

        record("data.pkl", data_pkl)
        record(".format_version", b"1")
        record(".storage_alignment", str(STORAGE_ALIGNMENT).encode())
        record("byteorder", sys.byteorder.encode())
        for key, storage in storages.items():
            record(f"data/{key}", _storage_bytes(storage))
        record("version", b"3\n")


# -- Loading --

class _StorageType:
    """What find_class returns for torch.<Type>Storage: the dtype of the
    storage records that reference it."""

    def __init__(self, name: str):
        self.name = name
        self.dtype = _DTYPE_BY_STORAGE_NAME.get(name)
        if name == "UntypedStorage":
            self.dtype = raxis.uint8


class _TypedStorage:
    def __init__(self, untyped: UntypedStorage, dtype):
        self.untyped = untyped
        self.dtype = dtype


# From: torch/_utils.py (_rebuild_tensor_v2, _rebuild_parameter)
def _rebuild_tensor_v2(storage: _TypedStorage, storage_offset, size, stride, requires_grad, backward_hooks,
                       metadata=None):
    tensor = _tensor_from_storage(storage.untyped, storage.dtype, storage_offset, tuple(size), tuple(stride))
    if requires_grad:
        tensor.requires_grad_(True)
    return tensor


def _rebuild_parameter(data, requires_grad, backward_hooks):
    return raxis.nn.Parameter(data, requires_grad)


def _rebuild_device(type_or_str, index=None):
    return raxis.device(type_or_str if index is None else f"{type_or_str}:{index}")


def _encode(obj, encoding="latin1"):
    # Protocol 2 pickles bytes as _codecs.encode(str, "latin1")
    return obj.encode(encoding)


# From: torch/_weights_only_unpickler.py (_get_allowed_globals): the globals
# a state dict needs, nothing that can run code
_SAFE_GLOBALS = {
    "collections.OrderedDict": OrderedDict,
    "torch._utils._rebuild_tensor_v2": _rebuild_tensor_v2,
    "torch._utils._rebuild_parameter": _rebuild_parameter,
    "torch.device": _rebuild_device,
    "_codecs.encode": _encode,
    "builtins.bytearray": bytearray,
    "builtins.set": set,
    "builtins.frozenset": frozenset,
    "builtins.complex": complex,
}


class _TorchUnpickler(pickle.Unpickler):

    def __init__(self, file: BinaryIO, zf: zipfile.ZipFile, prefix: str, weights_only: bool, map_location):
        super().__init__(file)
        self.zf = zf
        self.prefix = prefix
        self.weights_only = weights_only
        self.map_location = map_location
        self.storages: dict[str, UntypedStorage] = {}

    def find_class(self, module: str, name: str):
        if module == "torch" and name.endswith("Storage"):
            return _StorageType(name)
        full_name = f"{module}.{name}"
        if full_name in _SAFE_GLOBALS:
            return _SAFE_GLOBALS[full_name]
        # torch dtypes (torch.float32 pickles as the global torch.float32)
        if module == "torch" and isinstance(getattr(raxis, name, None), raxis.dtype):
            return getattr(raxis, name)
        if not self.weights_only and not module.startswith("torch"):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"raxis.load: unsupported global {full_name} in the checkpoint (only tensors, state dicts "
            f"and plain Python containers are loaded{'' if not self.weights_only else '; weights_only=True'})")

    # From: torch/serialization.py (_load, persistent_load)
    def persistent_load(self, pid):
        typename, storage_type, key, location, numel = pid
        if typename != "storage":
            raise pickle.UnpicklingError(f"raxis.load: unknown persistent id type {typename!r}")
        if storage_type.dtype is None:
            dtype = _UNSUPPORTED_STORAGE_NAMES.get(storage_type.name, storage_type.name)
            raise RuntimeError(f"raxis.load: the checkpoint has {dtype} tensors, a dtype raxis does not support")
        storage = self.storages.get(key)
        if storage is None:
            data = self.zf.read(f"{self.prefix}data/{key}")
            expected = numel * storage_type.dtype.itemsize
            if len(data) != expected:
                raise RuntimeError(f"raxis.load: storage {key} has {len(data)} bytes, expected {expected}")
            storage = UntypedStorage(len(data))
            memoryview(storage)[:] = data
            storage = _storage_to(storage, _map_location(location, self.map_location))
            self.storages[key] = storage
        return _TypedStorage(storage, storage_type.dtype)


# From: torch/serialization.py (_get_restore_location)
def _map_location(location: str, map_location):
    if map_location is None:
        target = location
    elif isinstance(map_location, dict):
        target = map_location.get(location, location)
    else:
        target = map_location
    device = raxis.device(target) if isinstance(target, str) else target
    if device.type == "cuda" and not raxis.cuda.is_available():
        raise RuntimeError(f"raxis.load: the checkpoint has tensors on {location} but CUDA is not available; "
                           "load them with map_location='cpu'")
    return device


def load(f, map_location=None, weights_only: bool = True) -> Any:
    """Loads an object saved by raxis.save or torch.save from the path or
    binary file f. Tensors go back to the device they were saved from, or
    to map_location: a device (or its name) for all of them, or a dict
    {saved location: device name}. weights_only=False also allows non-torch
    globals, which can run arbitrary code: only for trusted files."""
    start = f.tell() if hasattr(f, "tell") else None
    if not zipfile.is_zipfile(f):
        raise RuntimeError("raxis.load: not a zip checkpoint (the legacy format from before torch 1.6 "
                           "is not supported)")
    if start is not None:
        f.seek(start)
    with zipfile.ZipFile(f) as zf:
        names = zf.namelist()
        if not names or "/" not in names[0]:
            raise RuntimeError("raxis.load: the archive has no root directory")
        prefix = names[0].split("/", 1)[0] + "/"
        if f"{prefix}byteorder" in names and zf.read(f"{prefix}byteorder").decode() != sys.byteorder:
            raise RuntimeError("raxis.load: the checkpoint was saved with another byte order")
        with zf.open(f"{prefix}data.pkl") as data_pkl:
            return _TorchUnpickler(data_pkl, zf, prefix, weights_only, map_location).load()
