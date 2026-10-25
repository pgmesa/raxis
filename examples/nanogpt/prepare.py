"""
Prepares tiny shakespeare for character-level language modeling, as
karpathy/nanoGPT's data/shakespeare_char/prepare.py: every character is a
token. Writes train.bin and val.bin (uint16 token ids) and meta.pkl (the
vocabulary) to data/shakespeare_char next to this file.
"""

import os
import pickle
import urllib.request

import numpy as np

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "shakespeare_char")
os.makedirs(data_dir, exist_ok=True)

input_path = os.path.join(data_dir, "input.txt")
if not os.path.exists(input_path):
    urllib.request.urlretrieve(DATA_URL, input_path)
with open(input_path, "r", encoding="utf-8") as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

chars = sorted(set(data))
vocab_size = len(chars)
print(f"vocab size: {vocab_size:,}")
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

n = len(data)
train_ids = np.array([stoi[c] for c in data[:int(n * 0.9)]], dtype=np.uint16)
val_ids = np.array([stoi[c] for c in data[int(n * 0.9):]], dtype=np.uint16)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")
train_ids.tofile(os.path.join(data_dir, "train.bin"))
val_ids.tofile(os.path.join(data_dir, "val.bin"))
with open(os.path.join(data_dir, "meta.pkl"), "wb") as f:
    pickle.dump({"vocab_size": vocab_size, "itos": itos, "stoi": stoi}, f)
