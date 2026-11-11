"""
Trains a model on CIFAR-10 with raxis or with PyTorch, the same code and data
for both, to compare accuracy and time.

    python train.py --model resnet9 --backend raxis --device cuda
    python train.py --model resnet9 --backend torch --device cuda
    python prepare_vit.py                     # once, for the pretrained ViT
    python train.py --model vit --backend raxis --device cuda

- resnet9: from scratch with David Page's recipe: 24 epochs, batch 512, SGD
  with Nesterov momentum, the learning rate rising linearly to 0.4 over 5
  epochs and back to 0, weight decay 5e-4, label smoothing 0.2. ~94%.
- vit: ViT-B/16 pretrained on ImageNet (prepare_vit.py), fine-tuned for 3
  epochs, batch 64, SGD with momentum, the learning rate warming up to 0.01
  and decaying with a cosine to 0, gradient clipping at 1. ~98%.

Both backends run in float32 with TF32 matmuls and convolutions, as nanoGPT
sets them; PyTorch also gets cudnn.benchmark, so it runs its fastest cuDNN
convolutions. The model is saved to out-<backend>-<model>/ckpt.pt (in torch's
format: either backend loads it) for infer.py.
"""

import argparse
import json
import math
import os
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["resnet9", "vit"], default="resnet9")
parser.add_argument("--backend", choices=["raxis", "torch"], default="raxis")
parser.add_argument("--device", default="cuda", help="cuda or cpu")
parser.add_argument("--epochs", type=int, help="default: the model's recipe")
parser.add_argument("--batch-size", type=int, help="default: the model's recipe")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", help="write the per-epoch losses, accuracies and times to this json file")
parser.add_argument("--out-dir", help="checkpoint directory, relative to this file (default: out-<backend>-<model>)")
args = parser.parse_args()

here = os.path.dirname(os.path.abspath(__file__))
os.environ["CIFAR10_BACKEND"] = args.backend
if args.backend == "torch":
    import torch as lib
else:
    import raxis as lib
sys.path.insert(0, here)
from data import CLASSES, Batches, load_cifar10  # noqa: E402
from models import MODELS  # noqa: E402

F = lib.nn.functional

RECIPES = {
    "resnet9": dict(epochs=24, batch_size=512, lr=0.4, warmup_epochs=5, decay="linear", momentum=0.9,
                    nesterov=True, weight_decay=5e-4, label_smoothing=0.2, grad_clip=None,
                    mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
    "vit": dict(epochs=3, batch_size=64, lr=0.01, warmup_epochs=0.3, decay="cosine", momentum=0.9,
                nesterov=False, weight_decay=0.0, label_smoothing=0.0, grad_clip=1.0,
                mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
}
recipe = dict(RECIPES[args.model])
if args.epochs:
    recipe["epochs"] = args.epochs
if args.batch_size:
    recipe["batch_size"] = args.batch_size

lib.manual_seed(args.seed)
device = args.device
if device.startswith("cuda"):
    lib.backends.cuda.matmul.allow_tf32 = True
    lib.backends.cudnn.allow_tf32 = True
    if args.backend == "torch":
        lib.backends.cudnn.benchmark = True
print(f"{args.model}, backend {args.backend}, device {device}, {recipe}")

x_train, y_train, x_test, y_test = load_cifar10()
train_batches = Batches(lib, x_train, y_train, recipe["batch_size"], device, recipe["mean"], recipe["std"],
                        train=True, seed=args.seed)
test_batches = Batches(lib, x_test, y_test, 500, device, recipe["mean"], recipe["std"], train=False)

model = MODELS[args.model]()
if args.model == "vit":
    pretrained = os.path.join(here, "data", "vit_b16.pt")
    if not os.path.exists(pretrained):
        sys.exit("run python prepare_vit.py first")
    state = model.state_dict()
    state.update(lib.load(pretrained, map_location="cpu"))
    model.load_state_dict(state)
    with lib.no_grad():
        # A new head for 10 classes, starting at zero as in the ViT paper's fine-tuning
        model.head.weight.zero_()
        model.head.bias.zero_()
model.to(device)
print(f"{sum(p.numel() for p in model.parameters()):,} parameters")

optimizer = lib.optim.SGD(model.parameters(), lr=recipe["lr"], momentum=recipe["momentum"],
                          nesterov=recipe["nesterov"], weight_decay=recipe["weight_decay"])


def lr_at(epoch):
    """Learning rate at a fractional epoch: linear warmup, then linear or cosine decay to 0."""
    warmup, total, peak = recipe["warmup_epochs"], recipe["epochs"], recipe["lr"]
    if epoch < warmup:
        return peak * epoch / warmup
    progress = (epoch - warmup) / (total - warmup)
    return peak * (1 - progress) if recipe["decay"] == "linear" else peak * 0.5 * (1 + math.cos(math.pi * progress))


@lib.no_grad()
def evaluate():
    model.eval()
    loss_sum = lib.zeros((), device=device)
    correct = lib.zeros((), dtype=lib.long, device=device)
    for x, y in test_batches:
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum")
        correct += (logits.argmax(1) == y).sum()
    model.train()
    return loss_sum.item() / len(x_test), correct.item() / len(x_test)


history = {"model": args.model, "backend": args.backend, "device": device, "recipe": recipe, "epochs": []}
steps_per_epoch = len(train_batches)
start = time.perf_counter()
for epoch in range(recipe["epochs"]):
    t0 = time.perf_counter()
    loss_sum = lib.zeros((), device=device)
    correct = lib.zeros((), dtype=lib.long, device=device)
    for step, (x, y) in enumerate(train_batches):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(epoch + step / steps_per_epoch)
        logits = model(x)
        loss = F.cross_entropy(logits, y, label_smoothing=recipe["label_smoothing"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if recipe["grad_clip"]:
            lib.nn.utils.clip_grad_norm_(model.parameters(), recipe["grad_clip"])
        optimizer.step()
        # Kept on the device: a .item() per step would wait for the GPU every time
        loss_sum += loss.detach() * len(y)
        correct += (logits.argmax(1) == y).sum()
    n = steps_per_epoch * recipe["batch_size"]
    train_loss, train_acc = loss_sum.item() / n, correct.item() / n
    epoch_s = time.perf_counter() - t0
    test_loss, test_acc = evaluate()
    history["epochs"].append(dict(epoch=epoch + 1, train_loss=train_loss, train_acc=train_acc,
                                  test_loss=test_loss, test_acc=test_acc, seconds=epoch_s))
    print(f"epoch {epoch + 1:2d}: train loss {train_loss:.4f} acc {train_acc:.4f}, "
          f"test loss {test_loss:.4f} acc {test_acc:.4f}, {epoch_s:.1f}s")

history["total_s"] = time.perf_counter() - start
history["test_acc"] = history["epochs"][-1]["test_acc"]
print(f"total {history['total_s']:.0f}s, test accuracy {history['test_acc']:.4f}")

out_dir = os.path.join(here, args.out_dir or f"out-{args.backend}-{args.model}")
os.makedirs(out_dir, exist_ok=True)
state = {k: v for k, v in model.state_dict().items() if k != "resize"}
lib.save({"model": state, "model_name": args.model, "classes": CLASSES, "mean": recipe["mean"],
          "std": recipe["std"], "test_acc": history["test_acc"], "recipe": recipe}, os.path.join(out_dir, "ckpt.pt"))
print(f"saved {out_dir}")
if args.out:
    with open(args.out, "w") as f:
        json.dump(history, f, indent=1)
