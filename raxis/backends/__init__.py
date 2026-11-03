"""Switches that change how raxis computes, under torch.backends' names so
that torch code setting them runs unchanged.

    raxis.backends.cpu.get_cpu_capability() the SIMD instruction set of the CPU kernels
    raxis.backends.cuda.matmul.allow_tf32   float32 matmuls on TF32 tensor cores (off)
    raxis.backends.cudnn.allow_tf32         float32 convolutions on TF32 tensor cores (on)
"""
from raxis.backends import cpu, cuda, cudnn
