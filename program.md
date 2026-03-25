# autoresearch

This is an experiment to have the LLM do its own research.

## Skills available

You have access to the **wandb** and **sandbox-sweeps** skills. Invoke them when you need API details, patterns for querying runs, or sweep configuration reference.

## Setup

To set up a new experiment, work with the user to:

1. **Verify W&B login**: run `uv run python -c "import wandb; print(wandb.Api().viewer()['entity'])"`. If it prints an entity name, credentials are present (stored in `~/.netrc` by `wandb login`). If it throws, ask the user to run `uv run wandb login` and try again. Confirm the entity and agree on a project name (default: `autoresearch`); set `WANDB_ENTITY` and `WANDB_PROJECT` in the session if not already in the environment.

2. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar25`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
3. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
4. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `agent/prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `agent/train.py` — the model file you modify. Model architecture, optimizer, training loop.
   - `sweep_harness.py` — the harness file you modify to configure each sweep.
5. **Verify data artifact exists**: Check that `autoresearch-data:latest` exists in the agreed W&B project. If not, tell the human to run `uv run prepare.py --upload` from the repo root to prepare and upload it.
6. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment is a **W&B sandbox sweep**: multiple hyperparameter configurations run in parallel across CoreWeave H100s, each for the fixed 5-minute time budget. You control what to explore by editing `SWEEP_CONFIG` and `NUM_AGENTS` in `sweep_harness.py`, then running:

```bash
uv run sweep_harness.py > sweep.log 2>&1
```

After all agents finish, `sweep_harness.py` writes `last_sweep_result.json` with the best run's config and `val_bpb`. Use the W&B API to query richer detail (all runs, loss curves, comparisons across sweeps).

**What you CAN do:**
- Modify `agent/train.py` — this is the only model file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.
- Modify `sweep_harness.py` — edit `SWEEP_CONFIG` and the `AGENTS` list before each sweep. Each entry in `AGENTS` is a `SandboxResources` that defines one sandbox; the length of the list controls how many run in parallel. Mix accelerator types freely within the list for a heterogeneous fleet (e.g. H100s for large-model candidates alongside A100s for cheaper small-model runs).

**What you CANNOT do:**
- Modify `agent/prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win.

**The first sweep**: Your very first sweep should always establish the baseline — run with an empty `parameters: {}` and `NUM_AGENTS = 1` to confirm the harness works and record the unmodified val_bpb.

## Output format

When a training run completes it prints a summary and logs everything to W&B under the `final/` prefix. The printed summary looks like:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

After `sweep_harness.py` completes, the best result is in `last_sweep_result.json` and the full sweep is queryable via the W&B API. Prefer the API over log files for comparing runs across sweeps.

## Logging results

When a sweep is done, log the best result to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars) — use the commit that was active when the sweep ran
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this sweep explored

Example:

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	sweep depth=[6,8,10,12] — best was depth=10
c3d4e5f	1.005000	44.0	discard	sweep GeLU activation variants
d4e5f6g	0.000000	0.0	crash	sweep wide model (OOM across all agents)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar25`).

LOOP FOREVER:

1. **Review current state**

   ```bash
   git log --oneline -10
   cat results.tsv
   cat last_sweep_result.json   # if it exists
   ```

   For richer analysis, query the W&B API (use the **wandb** skill for patterns):

   ```python
   import wandb, os, pandas as pd
   api = wandb.Api()
   path = f"{os.environ['WANDB_ENTITY']}/{os.environ.get('WANDB_PROJECT', 'autoresearch')}"
   runs = api.runs(path, order="-created_at")
   for r in runs[:20]:
       print(r.id, r.summary_metrics.get("final/val_bpb", "?"), dict(r.config))
   ```

2. **Form a hypothesis and configure the sweep**

   Decide what to vary: a hyperparameter range, an architectural change, or both. Edit `SWEEP_CONFIG` and the `AGENTS` list in `sweep_harness.py`. Use `bayes` method for efficient multi-parameter search, `grid` for small exhaustive searches.

   Scale the fleet by adding or removing entries in `AGENTS`: 1–2 for a quick sanity check, 4–8 for normal exploration, more for a broad sweep. Choose the accelerator per entry to match what that agent will run — A40/A100 for cheaper small-model configs, H100 for larger models or when MFU matters. For a heterogeneous fleet simply mix `SandboxResources` entries in the list; each sandbox is created independently. Consult the **sandbox-sweeps** skill for resource config syntax.

   If your hypothesis involves a structural change to the model or optimizer, edit `agent/train.py` directly first.

3. **Commit if agent/train.py changed**

   ```bash
   git add agent/train.py
   git commit -m "experiment: <brief description of hypothesis>"
   ```

   Always commit before running so the W&B run is linked to the correct code state. Do not commit `sweep_harness.py` changes — they are ephemeral sweep configuration.

4. **Run the sweep**

   ```bash
   uv run sweep_harness.py > sweep.log 2>&1
   ```

   This blocks until all agents finish. Each agent runs for the 5-minute time budget. With 4 agents this takes ~5–10 minutes total.

   If the sweep fails, check `sweep.log`:

   ```bash
   tail -n 100 sweep.log
   ```

   Fix the error and re-run. If a training crash (OOM, bug), decide: if easily fixed, fix and re-run; if fundamentally broken, log as `crash` and move on.

5. **Read results**

   ```bash
   cat last_sweep_result.json
   ```

   For the full picture across all runs in the sweep, use the W&B API:

   ```python
   import wandb, json, pandas as pd
   with open("last_sweep_result.json") as f:
       result = json.load(f)
   api   = wandb.Api()
   sweep = api.sweep(result["sweep_id"])
   rows  = [{"id": r.id,
             "val_bpb": r.summary_metrics.get("final/val_bpb", float("inf")),
             "peak_vram_mb": r.summary_metrics.get("final/peak_vram_mb", 0),
             **dict(r.config)} for r in sweep.runs]
   df = pd.DataFrame(rows).sort_values("val_bpb")
   print(df.to_string(index=False))
   ```

6. **Decide: keep or discard**

   - **Keep** if val_bpb improved (lower) vs the current best in `results.tsv`, or if val_bpb is equal and the code is simpler.
   - **Discard** if val_bpb is higher (worse), or VRAM blew up without a meaningful gain.

7. **If keeping: bake the best config into agent/train.py defaults**

   Take the winning config from `last_sweep_result.json` and update the corresponding constants at the top of `agent/train.py`. Then commit:

   ```bash
   git add agent/train.py
   git commit -m "keep: <description>  val_bpb=<new_best>"
   ```

8. **If discarding: revert agent/train.py**

   ```bash
   git checkout agent/train.py
   ```

   If you made a commit in step 3 for a structural change you're now reverting:

   ```bash
   git reset HEAD~1
   git checkout agent/train.py
   ```

9. **Record in results.tsv**

   Log the best result from the sweep (keep or discard). Do NOT commit `results.tsv` — leave it untracked.

10. **Repeat**

**Timeout**: If a sweep exceeds 20 minutes wall clock, kill it (`Ctrl-C`), treat it as a failure, and move on.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human may be asleep and expects you to continue working *indefinitely* until manually stopped. If you run out of ideas, think harder — read the README for paper references, query the W&B API to find patterns across all runs, try combining previous near-misses, or make more radical architectural changes.

Each sweep batch with 4 agents explores 4+ configs in the time a single serial run would explore 1. Over an 8-hour session you can cover 200+ configurations.
