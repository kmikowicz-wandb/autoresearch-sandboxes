# autoresearch

This is an experiment to have the LLM do its own research.

## Skills available

You have access to the **wandb** and **sandbox-sweeps** skills. Invoke them when you need API details, patterns for querying runs, or sweep configuration reference.

## Setup

To set up a new experiment, work with the user to:

1. **Verify W&B login**: run `uv run python -c "import wandb; print(wandb.Api().viewer()['entity'])"`. If it prints an entity name, credentials are present. If it throws, ask the user to run `uv run wandb login` and try again. Confirm the entity and agree on a project name (default: `autoresearch`); set `WANDB_ENTITY` and `WANDB_PROJECT` in the session if not already in the environment.

2. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar25`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
3. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
4. **Read the in-scope files**: Read these files for full context:
   - `README.md` — repository context.
   - `agent/prepare.py` — fixed constants, data loading, evaluation. Do not modify.
   - `agent/train.py` — the model file you modify. Architecture, optimizer, training loop.
   - `sweep_harness.py` — the harness file you modify to configure each sweep.
5. **Verify data artifact exists**: Check that `autoresearch-data:latest` exists in the agreed W&B project. If not, tell the human to run `uv run prepare.py --upload` from the repo root.
6. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## The task

Train a **character-level Transformer** on TinyShakespeare (~1.1 MB of text, 65-char vocab). The model trains for a fixed 2-minute budget. Minimize `val_bpb` — bits per byte (= bits per character) on the held-out validation set. Lower is better; a random model scores ~6.0, a good model reaches ~1.4–1.6.

The dataset is tiny and CPU-capable, but GPU will be faster. Sandboxes run on CoreWeave.

## Experimentation

Each experiment is a **W&B sandbox sweep**: multiple hyperparameter configurations run in parallel, each for the fixed 2-minute time budget. You control what to explore by editing `SWEEP_CONFIG` and the `AGENTS` list in `sweep_harness.py`, then running:

```bash
uv run sweep_harness.py > sweep.log 2>&1
```

After all agents finish, `sweep_harness.py` writes `last_sweep_result.json`. Use the W&B API to query richer detail.

**What you CAN do:**
- Modify `agent/train.py` — this is the only model file you edit. Everything is fair game: architecture, optimizer, hyperparameters, learning rate schedule, etc.
- Modify `sweep_harness.py` — edit `SWEEP_CONFIG`, `NUM_AGENTS`, and `RESOURCES` before each sweep. `NUM_AGENTS` controls how many sandboxes run in parallel. `RESOURCES` is a single `SandboxResources` applied to all agents. CPU-only sandboxes (`SandboxResources(cpus=4, memory=8)`) are fine and cheap for this problem.

**What you CANNOT do:**
- Modify `agent/prepare.py`. It is read-only. It defines `TIME_BUDGET`, `MAX_SEQ_LEN`, `evaluate_bpb`, and `ensure_cache`.
- Modify the `evaluate_bpb` function. It is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Everything in `agent/train.py` is fair game: depth, width, optimizer, learning rate schedule, positional encoding, attention variant, etc. The only constraints are that the code runs and finishes within the 2-minute budget.

**Simplicity criterion**: All else being equal, simpler is better. Removing something and getting equal or better results is a win.

**The first sweep**: Always establish the baseline first — run with `parameters: {}` and one agent to confirm the harness works and record the unmodified val_bpb.

## Output format

When a training run completes it logs everything to W&B under the `final/` prefix and prints a summary:

```
---
val_bpb:          1.512345
training_seconds: 120.1
num_steps:        4832
num_params_M:     0.37
n_layer:          4
n_embd:           128
```

After `sweep_harness.py` completes, the best result is in `last_sweep_result.json`.

## Logging results

When a sweep is done, log the best result to `results.tsv` (tab-separated, NOT comma-separated):

```
commit	val_bpb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.512345) — use 0.000000 for crashes
3. status: `keep`, `discard`, or `crash`
4. short text description of what this sweep explored

Example:

```
commit	val_bpb	status	description
a1b2c3d	1.512345	keep	baseline
b2c3d4e	1.490000	keep	sweep n_layer=[2,4,6,8] — best was 6
c3d4e5f	1.530000	discard	wider FFN (4x→8x) — no improvement
d4e5f6g	0.000000	crash	n_embd=512 OOM on CPU sandbox
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
   api  = wandb.Api()
   path = f"{os.environ['WANDB_ENTITY']}/{os.environ.get('WANDB_PROJECT', 'autoresearch')}"
   runs = api.runs(path, order="-created_at")
   for r in runs[:20]:
       print(r.id, r.summary_metrics.get("final/val_bpb", "?"), dict(r.config))
   ```

2. **Form a hypothesis and configure the sweep**

   Decide what to vary. Edit `SWEEP_CONFIG` and `AGENTS` in `sweep_harness.py`. Use `bayes` for efficient multi-parameter search, `grid` for small exhaustive searches.

   Scale the fleet by increasing `NUM_AGENTS`. CPU sandboxes (`SandboxResources(cpus=4, memory=8)`) are cheap; use them for most runs. Switch `RESOURCES` to a GPU instance for larger-model candidates if needed. Consult the **sandbox-sweeps** skill for resource config syntax.

   If your hypothesis involves a structural change, edit `agent/train.py` first.

3. **Commit if agent/train.py changed**

   ```bash
   git add agent/train.py
   git commit -m "experiment: <brief description of hypothesis>"
   ```

   Always commit before running. Do not commit `sweep_harness.py` changes.

4. **Run the sweep**

   ```bash
   uv run sweep_harness.py > sweep.log 2>&1
   ```

   This blocks until all agents finish. Each agent runs for 2 minutes. With 4 agents this takes ~3–5 minutes total.

   If the sweep fails, check `sweep.log`:

   ```bash
   tail -n 100 sweep.log
   ```

5. **Read results**

   ```bash
   cat last_sweep_result.json
   ```

   For the full picture across all runs in the sweep:

   ```python
   import wandb, json, pandas as pd
   with open("last_sweep_result.json") as f:
       result = json.load(f)
   api   = wandb.Api()
   sweep = api.sweep(result["sweep_id"])
   rows  = [{"id": r.id,
             "val_bpb": r.summary_metrics.get("final/val_bpb", float("inf")),
             **dict(r.config)} for r in sweep.runs]
   df = pd.DataFrame(rows).sort_values("val_bpb")
   print(df.to_string(index=False))
   ```

6. **Decide: keep or discard**

   - **Keep** if val_bpb improved (lower) vs the current best, or if equal and simpler.
   - **Discard** if val_bpb is higher (worse).

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

   If you made a commit in step 3:

   ```bash
   git reset HEAD~1
   git checkout agent/train.py
   ```

9. **Record in results.tsv**

   Log the best result (keep or discard). Do NOT commit `results.tsv` — leave it untracked.

10. **Repeat**

**Timeout**: If a sweep exceeds 15 minutes wall clock, kill it (`Ctrl-C`), treat as failure, and move on.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human may be asleep and expects you to continue working *indefinitely* until manually stopped. If you run out of ideas, think harder — query W&B for patterns across all runs, try combining previous near-misses, or make more radical architectural changes.

Each sweep with 4 agents explores 4+ configs in the time a single serial run would explore 1. Over a 2-hour session you can cover 100+ configurations.
