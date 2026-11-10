"""
The two CIFAR-10 models, written once for both backends: CIFAR10_BACKEND=torch
runs the same code on PyTorch (train.py and infer.py set it from --backend).

- ResNet9: David Page's ResNet for CIFAR-10 (myrtle.ai, "How to train your
  ResNet", 2018), trained from scratch. Convolutions and batch norm: the part
  of a network where PyTorch calls cuDNN and Raxis runs its own kernels.
- ViT: ViT-B/16 (Dosovitskiy et al., 2020), with timm's parameter names so
  that its pretrained weights load as they are, fine-tuned on CIFAR-10.
  Matrix multiplications, layer norm and attention.
"""

import math
import os

import numpy as np

if os.environ.get("CIFAR10_BACKEND", "raxis") == "torch":
    import torch
    import torch.nn as nn
    from torch.nn import functional as F
else:
    import raxis as torch
    import raxis.nn as nn
    from raxis.nn import functional as F


def conv_bn(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU())


class Residual(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.res1 = conv_bn(c, c)
        self.res2 = conv_bn(c, c)

    def forward(self, x):
        return x + self.res2(self.res1(x))


class ResNet9(nn.Module):
    """6.6M parameters; 94% on CIFAR-10 in 24 epochs with Page's recipe."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            conv_bn(3, 64),
            conv_bn(64, 128), nn.MaxPool2d(2), Residual(128),
            conv_bn(128, 256), nn.MaxPool2d(2),
            conv_bn(256, 512), nn.MaxPool2d(2), Residual(512),
            nn.MaxPool2d(4), nn.Flatten(),
            nn.Linear(512, num_classes, bias=False),
        )

    def forward(self, x):
        return self.net(x) * 0.125


def bilinear_matrix(n_in, n_out):
    """(n_out, n_in) matrix that resizes one axis bilinearly, as F.interpolate
    with align_corners=False: M @ image @ M.T resizes both axes."""
    m = np.zeros((n_out, n_in), dtype=np.float32)
    for i in range(n_out):
        src = max((i + 0.5) * n_in / n_out - 0.5, 0.0)
        i0 = min(int(math.floor(src)), n_in - 1)
        i1 = min(i0 + 1, n_in - 1)
        w = src - i0
        m[i, i0] += 1.0 - w
        m[i, i1] += w
    return m


class PatchEmbed(nn.Module):
    def __init__(self, patch, dim):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, patch, stride=patch)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)    # (B, patches, dim)


class Attention(nn.Module):
    def __init__(self, dim, n_head):
        super().__init__()
        self.n_head = n_head
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, C // self.n_head).permute(2, 0, 3, 1, 4).unbind(0)
        y = F.scaled_dot_product_attention(q, k, v)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, n_head):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, n_head)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, 4 * dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ViT(nn.Module):
    """ViT-B/16 at 224 x 224 (86M parameters). CIFAR-10 images are resized
    from 32 x 32 inside the model, with a bilinear matrix on each axis."""

    def __init__(self, num_classes=10, image=224, patch=16, dim=768, depth=12, n_head=12, input_size=32):
        super().__init__()
        self.patch_embed = PatchEmbed(patch, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, (image // patch) ** 2 + 1, dim))
        self.blocks = nn.Sequential(*[Block(dim, n_head) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, num_classes)
        self.register_buffer("resize", torch.from_numpy(bilinear_matrix(input_size, image)))

    def forward(self, x):
        if x.shape[-1] != self.resize.shape[0]:
            x = self.resize @ x @ self.resize.t()                # (B, 3, 32, 32) -> (B, 3, 224, 224)
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1) + self.pos_embed
        x = self.norm(self.blocks(x))
        return self.head(x[:, 0])


MODELS = {"resnet9": ResNet9, "vit": ViT}
