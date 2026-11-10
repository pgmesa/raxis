"""
CIFAR-10: download, load, augment and batch, with numpy, for either backend.

The images stay uint8 (N, 32, 32, 3) on the CPU. A batch is augmented there
(random crop from a 4-pixel reflect padding, random horizontal flip), copied
to the device as uint8, a quarter of the bytes of float32, and turned into
normalized, contiguous float32 (N, 3, 32, 32) on the device.
"""

import os
import pickle
import tarfile
import urllib.request

import numpy as np

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
here = os.path.dirname(os.path.abspath(__file__))


def load_cifar10():
    """Returns (x_train, y_train, x_test, y_test): uint8 (N, 32, 32, 3) and int64 (N,)."""
    data_dir = os.path.join(here, "data")
    cache = os.path.join(data_dir, "cifar10.npz")
    if not os.path.exists(cache):
        os.makedirs(data_dir, exist_ok=True)
        archive = os.path.join(data_dir, "cifar-10-python.tar.gz")
        if not os.path.exists(archive):
            print(f"downloading {URL}")
            urllib.request.urlretrieve(URL, archive + ".part")
            os.replace(archive + ".part", archive)
        parts = {}
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                name = os.path.basename(member.name)
                if name.startswith(("data_batch_", "test_batch")):
                    batch = pickle.load(tar.extractfile(member), encoding="bytes")
                    parts[name] = (batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1),
                                   np.array(batch[b"labels"], dtype=np.int64))
        train = [parts[f"data_batch_{i}"] for i in range(1, 6)]
        np.savez(cache, x_train=np.concatenate([x for x, _ in train]), y_train=np.concatenate([y for _, y in train]),
                 x_test=parts["test_batch"][0], y_test=parts["test_batch"][1])
    d = np.load(cache)
    return d["x_train"], d["y_train"], d["x_test"], d["y_test"]


def augment(x, rng):
    """Random 32 x 32 crop of the image padded by 4 (reflect), and a random flip."""
    n = len(x)
    padded = np.pad(x, ((0, 0), (4, 4), (4, 4), (0, 0)), mode="reflect")
    dy, dx = rng.integers(0, 9, size=n), rng.integers(0, 9, size=n)
    rows = dy[:, None] + np.arange(32)[None, :]           # (n, 32)
    cols = dx[:, None] + np.arange(32)[None, :]
    flip = rng.random(n) < 0.5
    cols[flip] = cols[flip][:, ::-1]
    return padded[np.arange(n)[:, None, None], rows[:, :, None], cols[:, None, :]]


class Batches:
    """Iterates over (images, labels) on the device, shuffled and augmented when training."""

    def __init__(self, lib, x, y, batch_size, device, mean, std, train, seed=0):
        self.lib, self.x, self.y = lib, x, y
        self.batch_size, self.device, self.train = batch_size, device, train
        self.rng = np.random.default_rng(seed)
        self.mean = lib.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = lib.tensor(std, device=device).view(1, 3, 1, 1)

    def __len__(self):
        n = len(self.x)
        return n // self.batch_size if self.train else -(-n // self.batch_size)

    def to_device(self, array):
        t = self.lib.from_numpy(np.ascontiguousarray(array))
        if self.device.startswith("cuda"):
            # From pinned memory the copy is asynchronous, so the CPU prepares
            # the next batch while the GPU works on this one
            return t.pin_memory().to(self.device, non_blocking=True)
        return t

    def __iter__(self):
        n = len(self.x)
        order = self.rng.permutation(n) if self.train else np.arange(n)
        for i in range(len(self)):
            idx = order[i * self.batch_size:(i + 1) * self.batch_size]
            images = augment(self.x[idx], self.rng) if self.train else self.x[idx]
            x = self.to_device(images).permute(0, 3, 1, 2).float().div(255.0)
            # contiguous(): the permuted batch has channels-last strides, and
            # PyTorch would carry that layout through the whole network (its
            # float32 cuDNN convolutions then run 4x slower here, 185 vs 47 ms
            # per ResNet9 step); Raxis convolves both layouts the same way
            yield ((x - self.mean) / self.std).contiguous(), self.to_device(self.y[idx])
