"""
Autoresearch: character-level Transformer on TinyShakespeare.
Runs on CPU or GPU. Usage: python train.py

Edit the constants below, or let the sweep override them via wandb.config.
"""
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from prepare import (
    CACHE_DIR, TIME_BUDGET, MAX_SEQ_LEN,
    ensure_cache, load_data, evaluate_bpb,
)

# ---------------------------------------------------------------------------
# Hyperparameters (sweep can override any of these via wandb.config)
# ---------------------------------------------------------------------------
N_LAYER      = 1
N_EMBD       = 192
N_HEAD       = 2
FFN_MULT     = 1   # FFN hidden dim = FFN_MULT * n_embd
DROPOUT      = 0.0
BATCH_SIZE   = 32
LR           = 5e-3
WEIGHT_DECAY = 0.01
MIN_LR_RATIO = 0.0  # cosine decays to 0 (full decay)
WARMUP_SECS  = 10.0  # linear warmup duration in seconds
ADAM_BETA2   = 0.999  # AdamW beta2 — shorter memory (0.95-0.99) adapts faster
USE_BF16     = True   # bfloat16 autocast on CPU (AMD EPYC supports native BF16)
USE_COMPILE  = False  # torch.compile — fuses ops, may speed up CPU forward pass
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def get_batch(data: np.ndarray, batch_size: int, seq_len: int):
    ix = np.random.randint(0, len(data) - seq_len, size=(batch_size,))
    x = np.stack([data[i : i + seq_len] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1 : i + seq_len + 1] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float, ffn_mult: int = 4):
        super().__init__()
        assert n_embd % n_head == 0
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)
        self.qkv  = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        ffn_dim   = ffn_mult * n_embd
        self.mlp  = nn.Sequential(
            nn.Linear(n_embd, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, n_embd),
            nn.Dropout(dropout),
        ) if ffn_dim > 0 else None
        self.n_head   = n_head
        self.head_dim = n_embd // n_head

    def forward(self, x):
        B, T, C = x.shape
        normed = self.ln1(x)
        q, k, v = self.qkv(normed).split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.proj(y)
        if self.mlp is not None:
            x = x + self.mlp(self.ln2(x))
        return x


class CharTransformer(nn.Module):
    def __init__(self, vocab_size: int, n_layer: int, n_embd: int, n_head: int,
                 seq_len: int, dropout: float, ffn_mult: int = 4):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(seq_len, n_embd)
        self.drop    = nn.Dropout(dropout)
        self.blocks  = nn.ModuleList([Block(n_embd, n_head, dropout, ffn_mult) for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(n_embd)
        self.head    = nn.Linear(n_embd, vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, targets=None):
        B, T = x.shape
        tok = self.tok_emb(x)
        pos = self.pos_emb(torch.arange(T, device=x.device))
        h   = self.drop(tok + pos)
        for block in self.blocks:
            h = block(h)
        logits = self.head(self.ln_f(h))
        if targets is None:
            return logits
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


# ---------------------------------------------------------------------------
# Learning rate schedule: linear warmup then cosine decay over TIME_BUDGET
# ---------------------------------------------------------------------------

def get_lr(elapsed: float, lr: float, min_lr_ratio: float, warmup_secs: float = 5.0) -> float:
    if elapsed < warmup_secs:
        return lr * elapsed / warmup_secs
    progress = min((elapsed - warmup_secs) / (TIME_BUDGET - warmup_secs), 1.0)
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine_factor)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    import os
    num_threads = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 4))
    torch.set_num_threads(num_threads)

    ensure_cache()

    wandb.init()
    cfg = wandb.config

    n_layer    = cfg.get("n_layer",    N_LAYER)
    n_embd     = cfg.get("n_embd",     N_EMBD)
    n_head     = cfg.get("n_head",     N_HEAD)
    ffn_mult     = cfg.get("ffn_mult",     FFN_MULT)
    dropout      = cfg.get("dropout",      DROPOUT)
    batch_size   = cfg.get("batch_size",   BATCH_SIZE)
    lr           = cfg.get("lr",           LR)
    weight_decay = cfg.get("weight_decay", WEIGHT_DECAY)
    min_lr_ratio = cfg.get("min_lr_ratio", MIN_LR_RATIO)
    warmup_secs  = cfg.get("warmup_secs",  WARMUP_SECS)
    adam_beta2   = cfg.get("adam_beta2",   ADAM_BETA2)
    use_bf16     = cfg.get("use_bf16",     USE_BF16)
    use_compile  = cfg.get("use_compile",  USE_COMPILE)

    train_data, val_data, vocab_size = load_data()

    model = CharTransformer(vocab_size, n_layer, n_embd, n_head, MAX_SEQ_LEN, dropout, ffn_mult).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device} | params: {num_params:,} | vocab: {vocab_size}")
    if use_compile:
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                                  betas=(0.9, adam_beta2))

    t_start     = time.time()
    step        = 0
    smooth_loss = None

    while True:
        elapsed = time.time() - t_start
        if elapsed >= TIME_BUDGET:
            break

        # Update learning rate
        current_lr = get_lr(elapsed, lr, min_lr_ratio, warmup_secs)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        x, y = get_batch(train_data, batch_size, MAX_SEQ_LEN)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=use_bf16):
            loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        lv = loss.item()
        smooth_loss = lv if smooth_loss is None else 0.9 * smooth_loss + 0.1 * lv
        step += 1

        if step % 20 == 0:
            wandb.log({
                "train/loss":        lv,
                "train/smooth_loss": smooth_loss,
                "train/lr":          current_lr,
            }, step=step)

    elapsed = time.time() - t_start
    val_bpb = evaluate_bpb(model, val_data, device)

    print("---")
    print(f"val_bpb:          {val_bpb:.6f}")
    print(f"training_seconds: {elapsed:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.2f}")
    print(f"n_layer:          {n_layer}")
    print(f"n_embd:           {n_embd}")

    wandb.log({
        "final/val_bpb":          val_bpb,
        "final/training_seconds": elapsed,
        "final/num_steps":        step,
        "final/num_params_M":     num_params / 1e6,
        "final/n_layer":          n_layer,
        "final/n_embd":           n_embd,
    })
    wandb.finish()


if __name__ == "__main__":
    main()
