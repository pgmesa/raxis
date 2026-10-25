"""
Trains the shakespeare_char GPT of karpathy/nanoGPT on the CPU, with raxis or
with PyTorch, to compare loss curves and time per iteration.

The training loop is nanoGPT's train.py without what only matters on GPUs
(DDP, torch.compile, autocast, GradScaler) or for logging (wandb). The
defaults are the CPU run of the nanoGPT README:

    python train.py config/train_shakespeare_char.py --device=cpu --compile=False
        --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12
        --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000
        --lr_decay_iters=2000 --dropout=0.0

Usage (after python prepare.py):

    python train.py --backend raxis
    python train.py --backend torch
    python train.py --backend raxis --full    # the full shakespeare_char config

As nanoGPT, the model is saved to <out_dir>/ckpt.pt whenever the val loss
improves (out-raxis or out-torch by default), for sample.py. Both backends
write torch's checkpoint format, so either one can sample from the other's.
"""

import argparse
import json
import math
import os
import pickle
import sys
import time

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--backend", choices=["raxis", "torch"], default="raxis")
parser.add_argument("--full", action="store_true",
                    help="6 layers, n_embd 384, block 256, batch 64, dropout 0.2, 5000 iters")
parser.add_argument("--max_iters", type=int)
parser.add_argument("--eval_interval", type=int, default=250)
parser.add_argument("--eval_iters", type=int)
parser.add_argument("--log_interval", type=int, default=10)
parser.add_argument("--threads", type=int, help="number of threads (default: the library's)")
parser.add_argument("--seed", type=int, default=1337)
parser.add_argument("--out", help="write the losses and timings to this json file")
parser.add_argument("--out_dir", help="checkpoint directory, relative to this file (default: out-<backend>)")
args = parser.parse_args()

# The model module picks the library from this variable at import time
os.environ["NANOGPT_BACKEND"] = args.backend
if args.backend == "torch":
    import torch as lib
else:
    import raxis as lib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPT, GPTConfig

# config/train_shakespeare_char.py, with the README's CPU overrides unless --full
config = dict(
    batch_size=12, block_size=64, n_layer=4, n_head=4, n_embd=128, dropout=0.0, bias=False,
    learning_rate=1e-3, max_iters=2000, lr_decay_iters=2000, min_lr=1e-4, warmup_iters=100,
    weight_decay=1e-1, beta1=0.9, beta2=0.99, grad_clip=1.0, eval_iters=20,
)
if args.full:
    config.update(batch_size=64, block_size=256, n_layer=6, n_head=6, n_embd=384, dropout=0.2,
                  max_iters=5000, lr_decay_iters=5000, eval_iters=200)
if args.max_iters is not None:
    config.update(max_iters=args.max_iters)
if args.eval_iters is not None:
    config.update(eval_iters=args.eval_iters)
globals().update(config)

if args.threads:
    lib.set_num_threads(args.threads)
lib.manual_seed(args.seed)
print(f"backend {args.backend}, {lib.get_num_threads()} threads")
print(f"tokens per iteration will be: {batch_size * block_size:,}")

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "shakespeare_char")
if not os.path.exists(os.path.join(data_dir, "train.bin")):
    sys.exit("run python prepare.py first")
with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
    vocab_size = pickle.load(f)["vocab_size"]

# The batches are drawn with numpy so that both backends see the same data
batch_rng = np.random.default_rng(args.seed)


def get_batch(split):
    data = np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16, mode="r")
    ix = batch_rng.integers(len(data) - block_size, size=batch_size)
    x = np.stack([data[i:i + block_size] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1:i + 1 + block_size] for i in ix]).astype(np.int64)
    return lib.from_numpy(x), lib.from_numpy(y)


model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=vocab_size, dropout=dropout)
model = GPT(GPTConfig(**model_args))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), "cpu")


@lib.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = []
        for _ in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


# Linear warmup, then cosine decay down to min_lr
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out_dir or f"out-{args.backend}")
os.makedirs(out_dir, exist_ok=True)
best_val_loss = 1e9

history = {"backend": args.backend, "config": config, "threads": lib.get_num_threads(),
           "train_loss": [], "eval": [], "step_ms": []}
X, Y = get_batch("train")
start = time.perf_counter()
for iter_num in range(max_iters + 1):
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    if iter_num % args.eval_interval == 0 or iter_num == max_iters:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        history["eval"].append({"iter": iter_num, **losses})
        if iter_num > 0 and losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "model_args": model_args, "iter_num": iter_num, "best_val_loss": best_val_loss,
                          "config": {**config, "dataset": "shakespeare_char"}}
            print(f"saving checkpoint to {out_dir}")
            lib.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
    if iter_num == max_iters:
        break

    t0 = time.perf_counter()
    _, loss = model(X, Y)
    X, Y = get_batch("train")
    loss.backward()
    lib.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    lossf = loss.item()
    dt = time.perf_counter() - t0

    history["train_loss"].append(lossf)
    history["step_ms"].append(dt * 1000)
    if iter_num % args.log_interval == 0:
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms")

total = time.perf_counter() - start
steady = sorted(history["step_ms"][10:]) or history["step_ms"]
history["total_s"] = total
history["median_step_ms"] = steady[len(steady) // 2]
print(f"total {total:.1f}s, median step {history['median_step_ms']:.2f}ms")
if args.out:
    with open(args.out, "w") as f:
        json.dump(history, f)
