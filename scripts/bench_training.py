"""
Training step time of the MNIST example models: raxis vs PyTorch.

One step is forward, loss, zero_grad, backward and optimizer step on a
random batch, with the models and optimizers of examples/mnist_mlp.ipynb
and examples/mnist_cnn.ipynb.

Usage:
    python bench_training.py                   # both models
    python bench_training.py -k cnn            # filter by model name
    python bench_training.py --steps 100       # more steps per sample
"""

import argparse
import time

import numpy as np
import torch
import raxis


def mlp(lib):
    nn = lib.nn
    return nn.Sequential(
        nn.Flatten(), nn.Linear(784, 200), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(200, 100), nn.ReLU(), nn.Linear(100, 10), nn.LogSoftmax(dim=1))


def cnn(lib):
    nn = lib.nn

    def block(c_in, c_out):
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding="same", bias=False), nn.BatchNorm2d(c_out),
            nn.ReLU(), nn.Dropout(0.3), nn.MaxPool2d(2))

    return nn.Sequential(block(1, 16), block(16, 32), nn.Flatten(), nn.Linear(32 * 7 * 7, 10))


# name, model, input shape (without batch), batch size, optimizer, loss
MODELS = [
    ("mlp", mlp, (28, 28), 64,
     lambda lib, params: lib.optim.SGD(params, lr=0.01, momentum=0.9),
     lambda lib: lib.nn.NLLLoss()),
    ("cnn", cnn, (1, 28, 28), 128,
     lambda lib, params: lib.optim.Adam(params, lr=0.01),
     lambda lib: lib.nn.CrossEntropyLoss()),
]


def time_steps(lib, make_model, input_shape, batch_size, make_opt, make_loss, steps, warmup, repeats):
    lib.manual_seed(0)
    model = make_model(lib)
    model.train()
    optimizer = make_opt(lib, model.parameters())
    loss_fn = make_loss(lib)
    rng = np.random.default_rng(0)
    x = lib.tensor(rng.random((batch_size, *input_shape), dtype=np.float32))
    y = lib.tensor(rng.integers(0, 10, size=batch_size), dtype=lib.long)

    def step():
        loss = loss_fn(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for _ in range(warmup):
        step()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(steps):
            step()
        samples.append((time.perf_counter() - t0) / steps)
    return min(samples) * 1e3


def main():
    parser = argparse.ArgumentParser(description="Training step benchmarks")
    parser.add_argument("-k", "--filter", default=None, help="model name substring")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    print(f"Raxis: {raxis.__version__} | PyTorch: {torch.__version__}")
    print(f"Threads: raxis {raxis.get_num_threads()} | torch {torch.get_num_threads()}\n")
    print(f"  {'model':6} {'batch':>6} {'raxis (ms)':>11} {'torch (ms)':>11} {'ratio':>7}")
    for name, make_model, input_shape, batch_size, make_opt, make_loss in MODELS:
        if args.filter and args.filter not in name:
            continue
        times = [
            time_steps(lib, make_model, input_shape, batch_size, make_opt, make_loss,
                       args.steps, args.warmup, args.repeats)
            for lib in (raxis, torch)
        ]
        print(f"  {name:6} {batch_size:>6} {times[0]:>11.3f} {times[1]:>11.3f} {times[0] / times[1]:>6.2f}x")


if __name__ == "__main__":
    main()
