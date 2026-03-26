"""
Runtime helpers for the autoresearch training agent.
Imported by train.py — do not modify.

Provides: ensure_cache, load_data, evaluate_bpb, and shared constants.
"""
import json
import math
import os

import numpy as np
import torch
import wandb

# ---------------------------------------------------------------------------
# Constants (fixed — do not modify)
# ---------------------------------------------------------------------------
CACHE_DIR   = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
TIME_BUDGET = 120    # training seconds
MAX_SEQ_LEN = 256    # context length used for training and evaluation

ARTIFACT_NAME = "autoresearch-data"

_VOCAB_FILE = os.path.join(CACHE_DIR, "vocab.json")
_TRAIN_FILE = os.path.join(CACHE_DIR, "train.npy")
_VAL_FILE   = os.path.join(CACHE_DIR, "val.npy")


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def ensure_cache() -> None:
    """Download the dataset artifact from W&B if the local cache is missing."""
    if all(os.path.exists(p) for p in [_VOCAB_FILE, _TRAIN_FILE, _VAL_FILE]):
        return
    entity  = os.environ.get("WANDB_ENTITY")
    project = os.environ.get("WANDB_PROJECT", "autoresearch")
    if not entity:
        raise RuntimeError("WANDB_ENTITY not set — cannot download dataset artifact")
    print(f"Cache missing. Downloading {ARTIFACT_NAME} from W&B ({entity}/{project})...")
    api      = wandb.Api()
    artifact = api.artifact(f"{entity}/{project}/{ARTIFACT_NAME}:latest")
    artifact.download(root=CACHE_DIR)
    print(f"Dataset ready at {CACHE_DIR}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    """Return (train_data, val_data, vocab_size) as numpy uint8 arrays."""
    with open(_VOCAB_FILE) as f:
        vocab = json.load(f)
    train = np.load(_TRAIN_FILE)
    val   = np.load(_VAL_FILE)
    return train, val, vocab["vocab_size"]


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_bpb(model, val_data: np.ndarray, device: torch.device) -> float:
    """
    Bits per byte on the validation set.

    Iterates over val_data in non-overlapping windows of MAX_SEQ_LEN,
    averages cross-entropy (nats/token), converts to bits/byte.
    Since TinyShakespeare is ASCII (1 byte per char), bits/char == bits/byte.
    """
    model.eval()
    total_nats = 0.0
    n_windows  = 0
    for i in range(0, len(val_data) - MAX_SEQ_LEN - 1, MAX_SEQ_LEN):
        x = torch.from_numpy(
            val_data[i : i + MAX_SEQ_LEN].astype(np.int64)
        ).unsqueeze(0).to(device)
        y = torch.from_numpy(
            val_data[i + 1 : i + MAX_SEQ_LEN + 1].astype(np.int64)
        ).unsqueeze(0).to(device)
        total_nats += model(x, y).item()
        n_windows  += 1
    model.train()
    nats_per_token = total_nats / max(n_windows, 1)
    return nats_per_token / math.log(2)
