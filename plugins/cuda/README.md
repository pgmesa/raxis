# raxis-cuda{cuda_major}

The CUDA {cuda_major} backend of [raxis](https://github.com/pgmesa/raxis), a
readable re-implementation of the core of PyTorch (ATen + autograd) in C++20.

```bash
pip install raxis-cuda{cuda_major}
```

installs raxis together with its GPU kernels, caching allocator and cuBLAS
bindings; `import raxis` loads them, and tensors move to the GPU with
`.cuda()` / `.to("cuda")` as in PyTorch. It needs an NVIDIA driver (the CUDA
runtime is linked statically, cuBLAS comes from NVIDIA's wheel).

raxis-cuda12 runs on drivers r525 and newer and on GPUs from Volta (sm_70)
on; raxis-cuda13 needs a driver r580 or newer and a GPU from Turing (sm_75)
on. Install only one of them.
