"""
One-time data preparation for autoresearch experiments.
Downloads TinyShakespeare, builds char vocab, creates train/val splits.

Usage:
    uv run prepare.py            # download and prepare
    uv run prepare.py --upload   # also upload to W&B as a dataset artifact
"""
import argparse
import json
import os

import numpy as np
import requests
import wandb

# ---------------------------------------------------------------------------
# Constants (keep in sync with agent/prepare.py)
# ---------------------------------------------------------------------------
CACHE_DIR     = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
ARTIFACT_NAME = "autoresearch-data"
DATA_URL      = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

_VOCAB_FILE = os.path.join(CACHE_DIR, "vocab.json")
_TRAIN_FILE = os.path.join(CACHE_DIR, "train.npy")
_VAL_FILE   = os.path.join(CACHE_DIR, "val.npy")


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------

def prepare() -> None:
    if all(os.path.exists(p) for p in [_VOCAB_FILE, _TRAIN_FILE, _VAL_FILE]):
        print("Data already prepared — nothing to do.")
        return

    os.makedirs(CACHE_DIR, exist_ok=True)

    # Download
    print("Downloading TinyShakespeare...")
    resp = requests.get(DATA_URL, timeout=30)
    resp.raise_for_status()
    text = resp.text
    print(f"  {len(text):,} characters")

    # Build char vocab
    chars    = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    vocab    = {
        "vocab_size": len(chars),
        "char2idx":   char2idx,
        "idx2char":   {str(i): c for c, i in char2idx.items()},
    }
    with open(_VOCAB_FILE, "w") as f:
        json.dump(vocab, f)
    print(f"  Vocab size: {len(chars)}")

    # Encode and split (last 10% = val)
    data               = np.array([char2idx[c] for c in text], dtype=np.uint8)
    n_val              = max(len(data) // 10, 1)
    train_data, val_data = data[:-n_val], data[-n_val:]
    np.save(_TRAIN_FILE, train_data)
    np.save(_VAL_FILE,   val_data)
    print(f"  Train: {len(train_data):,} chars  Val: {len(val_data):,} chars")
    print(f"Done. Cache at {CACHE_DIR}")


# ---------------------------------------------------------------------------
# W&B artifact helpers (also used by agent/prepare.py for download)
# ---------------------------------------------------------------------------

def upload_artifact() -> None:
    entity  = os.environ.get("WANDB_ENTITY")
    project = os.environ.get("WANDB_PROJECT", "autoresearch")
    if not entity:
        raise RuntimeError("WANDB_ENTITY not set — cannot upload artifact")
    print(f"Uploading {CACHE_DIR} to W&B ({entity}/{project}/{ARTIFACT_NAME})...")
    run      = wandb.init(entity=entity, project=project, job_type="data-prep")
    artifact = wandb.Artifact(
        ARTIFACT_NAME,
        type="dataset",
        description="TinyShakespeare character-level data (vocab.json + train.npy + val.npy)",
    )
    artifact.add_dir(CACHE_DIR)
    run.log_artifact(artifact)
    run.finish()
    print(f"Uploaded: {entity}/{project}/{ARTIFACT_NAME}:latest")


def ensure_cache() -> None:
    """Download the artifact if local cache is missing (used by sandboxes)."""
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
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare TinyShakespeare data for autoresearch")
    parser.add_argument("--upload", action="store_true", help="Upload prepared cache to W&B after preparation")
    args = parser.parse_args()

    prepare()

    if args.upload:
        print()
        upload_artifact()
