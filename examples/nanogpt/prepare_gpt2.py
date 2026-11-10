"""
Downloads OpenAI's GPT-2 (124M) from Hugging Face and prepares tiny
shakespeare with its BPE tokenizer, for fine-tuning as karpathy/nanoGPT's
config/finetune_shakespeare.py does (with gpt2 instead of gpt2-xl).

Writes, next to this file:

    out-gpt2/ckpt.pt            GPT-2's weights as a nanoGPT checkpoint
    data/shakespeare/train.bin  token ids (uint16), and val.bin
    data/shakespeare/meta.pkl   {"vocab_size": 50257, "tokenizer": "gpt2"}

The weights come in a safetensors file, read here with numpy: an 8-byte
header length, a JSON header with each tensor's dtype, shape and byte range,
and the raw data. The checkpoint is written with raxis in torch's format, so
both backends load it.

    python prepare_gpt2.py
    python sample.py --out_dir out-gpt2 --start "Hello, I'm a language model,"
"""

import json
import os
import pickle
import struct
import urllib.request

import numpy as np
import tiktoken

import raxis

WEIGHTS_URL = "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors"
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
here = os.path.dirname(os.path.abspath(__file__))


def download(url, path):
    if not os.path.exists(path):
        print(f"downloading {url}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)
    return path


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


def gpt2_state_dict(hf):
    # From: karpathy/nanoGPT model.py (GPT.from_pretrained). Hugging Face's
    # GPT-2 uses Conv1D modules, whose weights are the transpose of a Linear's
    transposed = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")
    state = {}
    for name, value in hf.items():
        if name.endswith((".attn.bias", ".attn.masked_bias")):
            continue  # the causal mask buffer, not a parameter
        key = "transformer." + name.removeprefix("transformer.")
        state[key] = np.ascontiguousarray(value.T if name.endswith(transposed) else value)
    state["lm_head.weight"] = state["transformer.wte.weight"]  # tied with the token embedding
    return state


def prepare_weights():
    ckpt_path = os.path.join(here, "out-gpt2", "ckpt.pt")
    if os.path.exists(ckpt_path):
        print(f"{ckpt_path} exists")
        return
    path = download(WEIGHTS_URL, os.path.join(here, "data", "gpt2", "model.safetensors"))
    state = gpt2_state_dict(read_safetensors(path))
    print(f"GPT-2: {len(state)} tensors, {sum(v.size for k, v in state.items() if k != 'lm_head.weight'):,} parameters")
    model_args = dict(n_layer=12, n_head=12, n_embd=768, block_size=1024, bias=True, vocab_size=50257,
                      dropout=0.0)
    checkpoint = {"model": {k: raxis.from_numpy(v.copy()) for k, v in state.items()}, "model_args": model_args,
                  "iter_num": 0, "best_val_loss": float("nan"),
                  "config": {"dataset": "openwebtext", "tokenizer": "gpt2"}}
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    raxis.save(checkpoint, ckpt_path)
    print(f"wrote {ckpt_path}")


def prepare_shakespeare():
    # From: karpathy/nanoGPT data/shakespeare/prepare.py
    data_dir = os.path.join(here, "data", "shakespeare")
    if os.path.exists(os.path.join(data_dir, "meta.pkl")):
        print(f"{data_dir} exists")
        return
    with open(download(DATA_URL, os.path.join(data_dir, "input.txt")), encoding="utf-8") as f:
        data = f.read()
    n = len(data)
    enc = tiktoken.get_encoding("gpt2")
    train_ids = np.array(enc.encode_ordinary(data[:int(n * 0.9)]), dtype=np.uint16)
    val_ids = np.array(enc.encode_ordinary(data[int(n * 0.9):]), dtype=np.uint16)
    print(f"train has {len(train_ids):,} tokens, val has {len(val_ids):,} tokens")
    train_ids.tofile(os.path.join(data_dir, "train.bin"))
    val_ids.tofile(os.path.join(data_dir, "val.bin"))
    with open(os.path.join(data_dir, "meta.pkl"), "wb") as f:
        pickle.dump({"vocab_size": enc.n_vocab, "tokenizer": "gpt2"}, f)


if __name__ == "__main__":
    prepare_weights()
    prepare_shakespeare()
