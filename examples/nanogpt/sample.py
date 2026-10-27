"""
Samples text from a model trained with train.py, as karpathy/nanoGPT's
sample.py (init_from='resume'), with raxis or with PyTorch.

The checkpoint is in torch's format whichever backend wrote it, so a model
trained with one backend can be sampled with the other:

    python sample.py --backend raxis                       # out-raxis/ckpt.pt
    python sample.py --backend raxis --out_dir out-torch   # trained with torch
    python sample.py --start "ROMEO:" --num_samples 3 --max_new_tokens 300
    python sample.py --out_dir out-raxis-full --device cuda
"""

import argparse
import os
import pickle
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--backend", choices=["raxis", "torch"], default="raxis")
parser.add_argument("--out_dir", help="checkpoint directory, relative to this file (default: out-<backend>)")
parser.add_argument("--start", default="\n", help="prompt the samples continue")
parser.add_argument("--num_samples", type=int, default=10)
parser.add_argument("--max_new_tokens", type=int, default=500)
parser.add_argument("--temperature", type=float, default=0.8,
                    help="1.0 = no change, < 1.0 = less random, > 1.0 = more random")
parser.add_argument("--top_k", type=int, default=200, help="keep only the top_k most likely tokens")
parser.add_argument("--seed", type=int, default=1337)
parser.add_argument("--device", default="cpu", help="cpu or cuda")
args = parser.parse_args()

here = os.path.dirname(os.path.abspath(__file__))
os.environ["NANOGPT_BACKEND"] = args.backend
if args.backend == "torch":
    import torch as lib
else:
    import raxis as lib
sys.path.insert(0, here)
from model import GPT, GPTConfig

lib.manual_seed(args.seed)

ckpt_path = os.path.join(here, args.out_dir or f"out-{args.backend}", "ckpt.pt")
if not os.path.exists(ckpt_path):
    sys.exit(f"no checkpoint at {ckpt_path}: run train.py first")
checkpoint = lib.load(ckpt_path, map_location="cpu")
model = GPT(GPTConfig(**checkpoint["model_args"]))
state_dict = checkpoint["model"]
# torch.compile prefixes the parameter names
unwanted_prefix = "_orig_mod."
for k in list(state_dict):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(args.device)
print(f"loaded {ckpt_path} (iter {checkpoint['iter_num']}, val loss {checkpoint['best_val_loss']:.4f})")

meta_path = os.path.join(here, "data", checkpoint["config"]["dataset"], "meta.pkl")
with open(meta_path, "rb") as f:
    meta = pickle.load(f)
stoi, itos = meta["stoi"], meta["itos"]


def encode(s):
    return [stoi[c] for c in s]


def decode(ids):
    return "".join(itos[i] for i in ids)


x = lib.tensor(encode(args.start), dtype=lib.long, device=args.device)[None, ...]
start = time.perf_counter()
with lib.no_grad():
    for _ in range(args.num_samples):
        y = model.generate(x, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
        print(decode(y[0].tolist()))
        print("---------------")
elapsed = time.perf_counter() - start
print(f"{args.num_samples * args.max_new_tokens / elapsed:.0f} tokens/s")
