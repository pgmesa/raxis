"""
Classifies images with a model trained by train.py, with raxis or with
PyTorch (the checkpoint is in torch's format, so either one loads it).

    python infer.py out-raxis-resnet9                      # test accuracy, per class
    python infer.py out-raxis-vit --show 16 --grid preds.png
    python infer.py out-raxis-vit cat.jpg truck.png        # your own images
    python infer.py out-torch-resnet9 --backend raxis      # trained with torch

Your own images are resized to 32 x 32 first, the size of CIFAR-10's.
"""

import argparse
import os
import sys

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", help="checkpoint directory (out-<backend>-<model>) or .pt file")
parser.add_argument("images", nargs="*", help="image files to classify (default: the CIFAR-10 test set)")
parser.add_argument("--backend", choices=["raxis", "torch"], default="raxis")
parser.add_argument("--device", default="cuda", help="cuda or cpu")
parser.add_argument("--show", type=int, default=0, help="print the predictions for this many random test images")
parser.add_argument("--grid", help="save those images with their predictions to this PNG (needs matplotlib)")
args = parser.parse_args()

here = os.path.dirname(os.path.abspath(__file__))
os.environ["CIFAR10_BACKEND"] = args.backend
if args.backend == "torch":
    import torch as lib
else:
    import raxis as lib
sys.path.insert(0, here)
from data import Batches, load_cifar10  # noqa: E402
from models import MODELS  # noqa: E402

path = args.checkpoint if args.checkpoint.endswith(".pt") else os.path.join(args.checkpoint, "ckpt.pt")
if not os.path.isabs(path) and not os.path.exists(path):
    path = os.path.join(here, path)
ckpt = lib.load(path, map_location="cpu")
classes = ckpt["classes"]
model = MODELS[ckpt["model_name"]]()
state = model.state_dict()
state.update(ckpt["model"])
model.load_state_dict(state)
model.to(args.device).eval()
print(f"{ckpt['model_name']} from {path}, test accuracy when trained {ckpt['test_acc']:.4f}")


@lib.no_grad()
def predict(images):
    """uint8 (N, 32, 32, 3) -> class probabilities (N, classes) as numpy."""
    batches = Batches(lib, images, np.zeros(len(images), dtype=np.int64), 250, args.device,
                      ckpt["mean"], ckpt["std"], train=False)
    probs = [lib.softmax(model(x), 1).cpu().numpy() for x, _ in batches]
    return np.concatenate(probs)


def report(images, probs, names):
    for name, p in zip(names, probs):
        top = np.argsort(p)[::-1][:3]
        print(f"{name:>24s}: " + ", ".join(f"{classes[i]} {p[i]:.1%}" for i in top))


if args.images:
    from PIL import Image
    images = np.stack([np.asarray(Image.open(f).convert("RGB").resize((32, 32), Image.BILINEAR))
                       for f in args.images])
    report(images, predict(images), [os.path.basename(f) for f in args.images])
    sys.exit()

_, _, x_test, y_test = load_cifar10()
probs = predict(x_test)
pred = probs.argmax(1)
print(f"test accuracy {np.mean(pred == y_test):.4f}")
for c, name in enumerate(classes):
    print(f"{name:>12s} {np.mean(pred[y_test == c] == c):.3f}")

if args.show:
    idx = np.random.default_rng(0).choice(len(x_test), args.show, replace=False)
    report(x_test[idx], probs[idx], [f"#{i} ({classes[y_test[i]]})" for i in idx])
    if args.grid:
        import matplotlib.pyplot as plt
        cols = min(8, args.show)
        rows = -(-args.show // cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.8))
        for ax, i in zip(np.ravel(axes), idx):
            ax.imshow(x_test[i])
            ok = pred[i] == y_test[i]
            ax.set_title(classes[pred[i]], color="green" if ok else "red", fontsize=9)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(args.grid, dpi=120)
        print(f"saved {args.grid}")
