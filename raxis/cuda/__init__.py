"""CUDA devices and the memory of the caching allocator, as torch.cuda.

Available when raxis was built with its CUDA backend (libraxis_cuda) and the
machine has an NVIDIA driver and GPU.
"""
from functools import lru_cache
from typing import NamedTuple

from raxis import _C


__all__ = [
    "is_available", "device_count", "current_device", "set_device", "get_device_capability",
    "get_device_properties", "synchronize",
    "memory_allocated", "max_memory_allocated", "memory_reserved", "max_memory_reserved",
    "reset_peak_memory_stats", "empty_cache",
]


def is_available() -> bool:
    return _C._cuda_is_available()


def device_count() -> int:
    return _C._cuda_device_count()


def current_device() -> int:
    return _C._cuda_current_device()


def set_device(device) -> None:
    _C._cuda_set_device(_device_index(device))


class _CudaDeviceProperties(NamedTuple):
    """A subset of torch's _CudaDeviceProperties."""
    name: str
    major: int
    minor: int
    total_memory: int
    multi_processor_count: int
    shared_memory_per_block_optin: int


@lru_cache
def _device_properties(index: int) -> _CudaDeviceProperties:
    return _CudaDeviceProperties(*_C._cuda_get_device_properties(index))


def get_device_properties(device=None) -> _CudaDeviceProperties:
    """Name, compute capability, memory and limits of the device (read once, then cached)."""
    return _device_properties(_device_index(device))


def get_device_capability(device=None) -> tuple[int, int]:
    """(major, minor) compute capability of the device: (8, 0) for an A100, (12, 0) for an RTX 50xx."""
    return tuple(_C._cuda_get_device_capability(_device_index(device)))


def synchronize(device=None) -> None:
    """Waits until every kernel queued on the device has finished.

    Ops on CUDA tensors return as soon as their kernel is queued, so time
    them with a synchronize() before reading the clock.
    """
    _C._cuda_synchronize(_device_index(device))


def memory_allocated(device=None) -> int:
    """Bytes held by tensors on the device (rounded up to the allocator's 512-byte blocks)."""
    return _C._cuda_memory_stats(_device_index(device))["allocated_bytes"]


def max_memory_allocated(device=None) -> int:
    return _C._cuda_memory_stats(_device_index(device))["max_allocated_bytes"]


def memory_reserved(device=None) -> int:
    """Bytes the caching allocator got from CUDA: allocated plus cached for reuse."""
    return _C._cuda_memory_stats(_device_index(device))["reserved_bytes"]


def max_memory_reserved(device=None) -> int:
    return _C._cuda_memory_stats(_device_index(device))["max_reserved_bytes"]


def reset_peak_memory_stats(device=None) -> None:
    _C._cuda_reset_peak_memory_stats(_device_index(device))


def empty_cache() -> None:
    """Returns the cached memory that holds no tensor to CUDA (nvidia-smi then shows less)."""
    _C._cuda_empty_cache()


def _device_index(device) -> int:
    if device is None:
        return current_device()
    if isinstance(device, int):
        return device
    if isinstance(device, str):
        device = _C.device(device)
    if device.type != "cuda":
        raise ValueError(f"Expected a cuda device, but got: {device}")
    return device.index if device.index >= 0 else current_device()
