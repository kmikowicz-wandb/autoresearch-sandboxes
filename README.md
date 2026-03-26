# autoresearch

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The idea: give an AI agent a small but real training setup and let it experiment autonomously overnight. It modifies the code, launches a parallel sweep across sandbox agents, checks if the results improved, keeps or discards, and repeats. You wake up to a W&B dashboard of experiments and (hopefully) a better model. The core idea is that you're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the `program.md` Markdown file that provides context to the AI agent and sets up your autonomous research loop. A bit more context on this project is here in this [tweet](https://x.com/karpathy/status/2029701092347630069) and [this tweet](https://x.com/karpathy/status/2031135152349524125).

## How it works

The repo has a few files that matter:

- **`agent/train.py`** — the single file the agent edits. Contains the character-level Transformer model, optimizer, and training loop. Everything is fair game: architecture, hyperparameters, optimizer, learning rate schedule, etc. **This file is edited and iterated on by the agent**.
- **`agent/prepare.py`** — fixed constants, data loading, and the evaluation function. Not modified.
- **`prepare.py`** — one-time data prep (downloads TinyShakespeare, builds char vocab, uploads to W&B). Run once by the human.
- **`sweep_harness.py`** — configures and launches a W&B sandbox sweep. The agent edits `SWEEP_CONFIG`, `NUM_AGENTS`, and `RESOURCES` before each run.
- **`query_sweep.py`** — queries W&B for results; prints the best runs across the project sorted by val_bpb.
- **`program.md`** — instructions for the agent. **This file is edited and iterated on by the human**.

Training runs for a **fixed 2-minute time budget** (wall clock). The metric is **val_bpb** (validation bits per byte) — lower is better. A random model scores ~6.0; a well-tuned small Transformer reaches ~1.4–1.6.

Each sweep launches multiple sandbox agents in parallel on CoreWeave, each training independently with a different hyperparameter configuration sampled from the sweep. The agent picks the best config and bakes it into `agent/train.py` before moving on.

## Quick start

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/), a [Weights & Biases](https://wandb.ai) account, and access to CoreWeave sandboxes (for parallel sweeps).

```bash
# 1. Install uv (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Log in to Weights & Biases (one-time — stores credentials in ~/.netrc)
uv run wandb login

# 4. Download data and upload to W&B (one-time, ~30 seconds)
uv run prepare.py --upload

# 5. Launch a sweep
uv run sweep_harness.py
```

Once the above commands work, your setup is ready for autonomous research mode.

## Running the agent

Spin up Claude Code (or any agent) in this repo and prompt:

```
Have a look at program.md and let's kick off a new experiment!
```

The `program.md` file is the lightweight "OS" for the research loop.

## Project structure

```
agent/
  train.py        — model, optimizer, training loop (agent modifies this)
  prepare.py      — constants, data loading, evaluation (do not modify)
  pyproject.toml  — sandbox dependencies

prepare.py        — one-time data prep, W&B artifact upload (run by human)
sweep_harness.py  — sweep config and launch (agent modifies SWEEP_CONFIG / NUM_AGENTS)
query_sweep.py    — query W&B for best results
program.md        — agent instructions (human iterates on this)
pyproject.toml    — local/orchestration dependencies
```

## Design choices

- **Single file to modify.** The agent only touches `agent/train.py`. This keeps scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 2 minutes. This makes all experiments directly comparable regardless of what the agent changes (model size, batch size, architecture, etc.).
- **CPU-capable by default.** The training problem (character-level Transformer on TinyShakespeare) runs on CPU or GPU. Sandboxes default to CPU (`SandboxResources(cpus=4, memory=8)`), making sweeps cheap. Switch to a GPU instance if the model grows large enough to benefit.
- **Parallel sweeps.** Each call to `sweep_harness.py` launches multiple sandbox agents simultaneously via W&B Sweeps on CoreWeave. Exploring N configurations takes the same wall-clock time as exploring 1.
- **W&B as the research record.** Runs, sweeps, and a living Report are all in W&B. No local result files to manage.

## Notable forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)
- [andyluo7/autoresearch](https://github.com/andyluo7/autoresearch) (AMD)
- [indianspeedster/autoresearch](https://github.com/indianspeedster/autoresearch) (AMD ROCm)

## License

MIT
