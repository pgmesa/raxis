"""
Downloads timm's ViT-B/16 (vit_base_patch16_224.augreg2_in21k_ft_in1k:
pretrained on ImageNet-21k, fine-tuned on ImageNet-1k) from Hugging Face and
writes its weights, without the 1000-class head, to data/vit_b16.pt in
torch's format, for train.py --model vit.

The safetensors file is read with numpy: an 8-byte header length, a JSON
header with each tensor's dtype, shape and byte range, and the raw data.

    python prepare_vit.py
"""

import json
import os
import struct
import urllib.request

import numpy as np

import raxis

URL = "https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k/resolve/main/model.safetensors"
here = os.path.dirname(os.path.abspath(__file__))


def read_safetensors(path):
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
        data = f.read()
    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        assert info["dtype"] == "F32", (name, info["dtype"])
        begin, end = info["data_offsets"]
        tensors[name] = np.frombuffer(data, dtype="<f4", count=(end - begin) // 4, offset=begin).reshape(info["shape"])
    return tensors


if __name__ == "__main__":
    out = os.path.join(here, "data", "vit_b16.pt")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists")
    path = os.path.join(here, "data", "vit_b16.safetensors")
    if not os.path.exists(path):
        print(f"downloading {URL}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(URL, path + ".part")
        os.replace(path + ".part", path)
    weights = {k: v for k, v in read_safetensors(path).items() if not k.startswith("head.")}
    print(f"{len(weights)} tensors, {sum(v.size for v in weights.values()):,} parameters without the head")
    raxis.save({k: raxis.from_numpy(v.copy()) for k, v in weights.items()}, out)
    print(f"wrote {out}")
