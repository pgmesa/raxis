# CIFAR-10: a ResNet from scratch and a fine-tuned ViT

Two image classifiers on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
(60,000 32 x 32 images in 10 classes), each written once and run with Raxis
or with PyTorch: `--backend raxis` or `--backend torch`.

| Model | What it shows | Raxis | PyTorch |
|---|---|--:|--:|
| ResNet-9, from scratch, 24 epochs | Convolutions and batch norm: PyTorch runs cuDNN, Raxis im2col + cuBLAS | 94.1%, 214 s | 93.9%, 120 s |
| ViT-B/16 pretrained on ImageNet, fine-tuned 3 epochs | Matrix multiplications, layer norm, attention | 98.85%, 649 s | 98.85%, 591 s |

Test accuracy at the end of training and the whole run with its evaluations,
on an RTX 5070 Ti (CUDA 13.0, driver 610.88), float32 with TF32 matmuls and
convolutions in both libraries, PyTorch 2.14.1 with `cudnn.benchmark`. Per
epoch: ResNet-9 8.3 s with Raxis and 4.6 s with PyTorch; ViT 204 s and 186 s.
Random numbers differ between the libraries, so the accuracies do too.

## Run it

From the repo's environment (see the main README), or with `pip install
raxis[cuda13] numpy pillow` (plus `torch` for `--backend torch`):

```bash
python train.py --model resnet9 --backend raxis     # downloads CIFAR-10 (163 MB) the first time
python train.py --model resnet9 --backend torch

python prepare_vit.py                               # timm's ViT-B/16 weights (346 MB)
python train.py --model vit --backend raxis
```

`--device cpu` runs on the CPU (slowly), and `--epochs` / `--batch_size`
change the recipe. Each run saves its model to `out-<backend>-<model>/ckpt.pt`,
in PyTorch's checkpoint format, so either library loads it.

## Use the trained model

```bash
python infer.py out-raxis-vit                       # test accuracy, and per class
python infer.py out-raxis-vit --show 16 --grid preds.png
python infer.py out-raxis-resnet9 cat.jpg truck.png # your own images, resized to 32 x 32
python infer.py out-torch-vit --backend raxis       # a model trained with PyTorch, run by Raxis
```

## The files

| File | What it holds |
|---|---|
| `data.py` | Download, augmentation (random crop from a 4-pixel reflect padding, horizontal flip) and batches on the device |
| `models.py` | `ResNet9` and `ViT`, with `nn` from the backend that `CIFAR10_BACKEND` names |
| `prepare_vit.py` | Downloads timm's `vit_base_patch16_224.augreg2_in21k_ft_in1k` and reads its safetensors file with numpy |
| `train.py` | The training loop and both recipes |
| `infer.py` | Test accuracy, predictions for random test images, or for your own files |

**ResNet-9** is David Page's network from
[How to train your ResNet](https://myrtle.ai/learn/how-to-train-your-resnet/)
(myrtle.ai, 2018) with his recipe: batch 512, SGD with Nesterov momentum, the
learning rate rising linearly to 0.4 over 5 epochs and back to 0, weight decay
5e-4, label smoothing 0.2.

**ViT-B/16** (86M parameters) has timm's parameter names, so the pretrained
weights load as they are; only the 10-class head is new, starting at zero.
CIFAR's 32 x 32 images are resized to 224 x 224 inside the model by a
bilinear matrix on each axis (`M @ image @ M.T`). Fine-tuning: batch 64, SGD
with momentum, warmup to 0.01 then a cosine to 0, gradient clipping at 1.

One thing found while measuring: the batches arrive as (N, 32, 32, 3) and are
permuted to (N, 3, 32, 32), which leaves them with channels-last strides.
PyTorch keeps that layout through the whole network, and on this GPU its
float32 cuDNN convolutions in channels-last took 185 ms per ResNet-9 step
instead of 47. `data.py` makes the batch contiguous, as torchvision's
`ToTensor` does; Raxis takes the same time with either layout.
