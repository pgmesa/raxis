"""
Trains the shakespeare_char GPT of karpathy/nanoGPT with raxis or with
PyTorch, on the CPU or a GPU, to compare loss curves and time per iteration.

The training loop is nanoGPT's train.py without DDP, torch.compile,
autocast / GradScaler (raxis has no float16 / bfloat16) and wandb; on a GPU
it sets TF32 for matmuls and convolutions as nanoGPT does. The defaults are
the CPU run of the nanoGPT README:

    python train.py config/train_shakespeare_char.py --device=cpu --compile=False
        --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12
        --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000
        --lr_decay_iters=2000 --dropout=0.0

Usage (after python prepare.py):

    python train.py --backend raxis
    python train.py --backend torch
    python train.py --backend raxis --full    # the full shakespeare_char config
    python train.py --backend raxis --full --device cuda

Fine-tuning GPT-2 (124M) on Shakespeare, as nanoGPT's
config/finetune_shakespeare.py with gpt2 instead of gpt2-xl (after python
prepare_gpt2.py; 20 iterations of 32 sequences of 1024 tokens, here as 4
micro-batches of 8):

    python train.py --backend raxis --finetune --device cuda

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
parser.add_argument("--device", default="cpu", help="cpu or cuda")
parser.add_argument("--full", action="store_true",
                    help="6 layers, n_embd 384, block 256, batch 64, dropout 0.2, 5000 iters")
parser.add_argument("--finetune", action="store_true",
                    help="fine-tune GPT-2 (out-gpt2, from prepare_gpt2.py) on BPE Shakespeare")
parser.add_argument("--max-iters", type=int)
parser.add_argument("--eval-interval", type=int)
parser.add_argument("--eval-iters", type=int)
parser.add_argument("--log-interval", type=int, default=10)
parser.add_argument("--threads", type=int, help="number of threads (default: the library's)")
parser.add_argument("--seed", type=int, default=1337)
parser.add_argument("--out", help="write the losses and timings to this json file")
parser.add_argument("--out-dir", help="checkpoint directory, relative to this file (default: out-<backend>)")
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
    gradient_accumulation_steps=1, decay_lr=True, eval_interval=250, dataset="shakespeare_char",
)
if args.full:
    config.update(batch_size=64, block_size=256, n_layer=6, n_head=6, n_embd=384, dropout=0.2,
                  max_iters=5000, lr_decay_iters=5000, eval_iters=200)
if args.finetune:
    # config/finetune_shakespeare.py (batch 1 x 32 there; 8 x 4 here, the same
    # 32 sequences per iteration) on train.py's defaults (beta2 0.95); the
    # architecture comes from the GPT-2 checkpoint
    config.update(batch_size=8, gradient_accumulation_steps=4, block_size=1024, learning_rate=3e-5,
                  decay_lr=False, max_iters=20, eval_interval=5, eval_iters=5, beta2=0.95,
                  dataset="shakespeare")
if args.max_iters is not None:
    config.update(max_iters=args.max_iters)
if args.eval_iters is not None:
    config.update(eval_iters=args.eval_iters)
if args.eval_interval is not None:
    config.update(eval_interval=args.eval_interval)
globals().update(config)

if args.threads:
    lib.set_num_threads(args.threads)
lib.manual_seed(args.seed)
device = args.device
if device.startswith("cuda"):
    # As nanoGPT: float32 matmuls and convolutions on TF32 tensor cores
    lib.backends.cuda.matmul.allow_tf32 = True
    lib.backends.cudnn.allow_tf32 = True
print(f"backend {args.backend}, device {device}, {lib.get_num_threads()} threads")
print(f"tokens per iteration will be: {gradient_accumulation_steps * batch_size * block_size:,}")

here = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(here, "data", dataset)
if not os.path.exists(os.path.join(data_dir, "train.bin")):
    sys.exit("run python prepare_gpt2.py first" if args.finetune else "run python prepare.py first")
with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
    meta = pickle.load(f)
vocab_size = meta["vocab_size"]

# The batches are drawn with numpy so that both backends see the same data
batch_rng = np.random.default_rng(args.seed)


def get_batch(split):
    data = np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16, mode="r")
    ix = batch_rng.integers(len(data) - block_size, size=batch_size)
    x = np.stack([data[i:i + block_size] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1:i + 1 + block_size] for i in ix]).astype(np.int64)
    x, y = lib.from_numpy(x), lib.from_numpy(y)
    if device.startswith("cuda"):
        # As nanoGPT: from pinned memory the copies are asynchronous, so the
        # next batch moves while the GPU still runs the current step
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


if args.finetune:
    pretrained = lib.load(os.path.join(here, "out-gpt2", "ckpt.pt"), map_location="cpu")
    model_args = pretrained["model_args"]
    model = GPT(GPTConfig(**model_args))
    model.load_state_dict(pretrained["model"])
    del pretrained
else:
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                      bias=bias, vocab_size=vocab_size, dropout=dropout)
    model = GPT(GPTConfig(**model_args))
model.to(device)
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device.split(":")[0])


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
    if not decay_lr:
        return learning_rate
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


out_dir = os.path.join(here, args.out_dir or f"out-{args.backend}" + ("-gpt2" if args.finetune else ""))
os.makedirs(out_dir, exist_ok=True)
best_val_loss = 1e9

history = {"backend": args.backend, "device": device, "config": config, "threads": lib.get_num_threads(),
           "train_loss": [], "eval": [], "step_ms": []}
X, Y = get_batch("train")
start = time.perf_counter()
for iter_num in range(max_iters + 1):
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    if iter_num % eval_interval == 0 or iter_num == max_iters:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        history["eval"].append({"iter": iter_num, **losses})
        if iter_num > 0 and losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "model_args": model_args, "iter_num": iter_num, "best_val_loss": best_val_loss,
                          "config": {**config, "tokenizer": meta.get("tokenizer")}}
            print(f"saving checkpoint to {out_dir}")
            lib.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
    if iter_num == max_iters:
        break

    t0 = time.perf_counter()
    # Gradient accumulation: the micro-batches add their gradients, each
    # loss scaled so that the sum is the mean over all of them
    for micro_step in range(gradient_accumulation_steps):
        _, loss = model(X, Y)
        loss = loss / gradient_accumulation_steps
        X, Y = get_batch("train")
        loss.backward()
    lib.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    lossf = loss.item() * gradient_accumulation_steps
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
