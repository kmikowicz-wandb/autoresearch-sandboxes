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
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation. The W&B dashboard (see below) is created after the baseline sweep completes.

## The task

Train a **character-level Transformer** on TinyShakespeare (~1.1 MB of text, 65-char vocab). The model trains for a fixed 2-minute budget. Minimize `val_bpb` — bits per byte (= bits per character) on the held-out validation set. Lower is better; a random model scores ~6.0, a good model reaches ~1.4–1.6.

The dataset is tiny and CPU-capable, but GPU will be faster. Sandboxes run on CoreWeave.

## Experimentation

Each experiment is a **W&B sandbox sweep**: multiple hyperparameter configurations run in parallel, each for the fixed 2-minute time budget. You control what to explore by editing `SWEEP_CONFIG`, `NUM_AGENTS`, and `RESOURCES` in `sweep_harness.py`, then running:

```bash
uv run sweep_harness.py > sweep.log 2>&1
```

After all agents finish, results are in W&B. Use `uv run query_sweep.py` to pull the latest sweep results.

**What you CAN do:**
- Modify `agent/train.py` — this is the only model file you edit. Everything is fair game: architecture, optimizer, hyperparameters, learning rate schedule, etc.
- Modify `sweep_harness.py` — edit `SWEEP_CONFIG`, `NUM_AGENTS`, and `RESOURCES` before each sweep. `NUM_AGENTS` controls how many sandboxes run in parallel. `RESOURCES` is a single `SandboxResources` applied to all agents. CPU-only sandboxes (`SandboxResources(cpus=4, memory=8)`) are fine and cheap for this problem.

**What you CANNOT do:**
- Modify `agent/prepare.py`. It is read-only. It defines `TIME_BUDGET`, `MAX_SEQ_LEN`, `evaluate_bpb`, and `ensure_cache`.
- Modify the `evaluate_bpb` function. It is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Everything in `agent/train.py` is fair game: depth, width, optimizer, learning rate schedule, positional encoding, attention variant, etc. The only constraints are that the code runs and finishes within the 2-minute budget.

**Simplicity criterion**: All else being equal, simpler is better. Removing something and getting equal or better results is a win.

**The first sweep**: Always establish the baseline first — run with `parameters: {}` and `NUM_AGENTS = 1` to confirm the harness works and record the unmodified val_bpb.

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

After `sweep_harness.py` completes, results are queryable via `uv run query_sweep.py`.

## W&B dashboard

The W&B project is the primary record of research progress. Keep it organized and readable.

### Naming sweeps

Before every sweep, set `name` and `description` in `SWEEP_CONFIG` so the sweep is immediately identifiable in the W&B UI:

```python
SWEEP_CONFIG = {
    "name": "depth-2to8",
    "description": "Does more depth help? Baseline is n_layer=4. Testing 2, 4, 6, 8.",
    "method": "grid",
    "metric": {"name": "final/val_bpb", "goal": "minimize"},
    "parameters": {
        "n_layer": {"values": [2, 4, 6, 8]},
    },
}
```

Use names that describe what is being varied (e.g. `lr-schedule`, `swiglu-vs-gelu`, `width-64to256`). The description should state the hypothesis being tested.

### Naming code artifact versions

Each `sweep_harness.py` run uploads a new version of the `autoresearch-job` artifact. Make the artifact name reflect what changed in that version by editing the `name` argument in `sweep_harness.py` before running:

```python
if create_job(
    job_type="code",
    path="./agent",
    entity=ENTITY,
    project=PROJECT,
    name="autoresearch-job-swiglu",        # ← describe the change
    entrypoint="python train.py",
) is None:
```

You can reuse a name across sweeps that share the same `agent/train.py` — W&B will version it automatically.

### Project workspace dashboard

After the baseline sweep, create a W&B Report that acts as the living research journal. Use the **wandb** skill for the full report API. A starter template:

```python
import os, wandb
from wandb.apis import reports as wr

entity  = os.environ["WANDB_ENTITY"]
project = os.environ.get("WANDB_PROJECT", "autoresearch")

runset = wr.Runset(entity=entity, project=project, name="All runs")

report = wr.Report(
    entity=entity,
    project=project,
    title="Autoresearch: TinyShakespeare",
    description="Character-level Transformer optimization log.",
    width="fixed",
    blocks=[
        wr.H1("Autoresearch: TinyShakespeare"),
        wr.P("Minimizing val_bpb on a 2-minute CPU/GPU training budget."),
        wr.H2("Progress"),
        wr.PanelGrid(
            runsets=[runset],
            panels=[
                wr.LinePlot(title="Val BPB over training steps",
                            x="_step", y=["final/val_bpb"]),
                wr.LinePlot(title="Train loss",
                            x="_step", y=["train/smooth_loss"]),
                wr.ScatterPlot(title="val_bpb by run",
                               x="final/num_params_M", y="final/val_bpb"),
            ],
        ),
    ],
)
report.save()
print(f"Report: {report.url}")
```

Save the report URL (printed by `report.url`) to a local file `report_url.txt`.

**After each sweep**, update the report to add a new narrative section documenting what was tried, what the best result was, and what it means:

```python
import wandb
from wandb.apis import reports as wr

with open("report_url.txt") as f:
    url = f.read().strip()

report = wr.Report.from_url(url)

# Prepend a new findings block after the H2 header
report.blocks.insert(3, wr.P(
    f"[Sweep: depth-2to8] Best val_bpb=1.487 at n_layer=6. "
    f"Shallower (2) underfit; deeper (8) gave no gain. Keeping n_layer=6."
))
report.save()
```

The panels auto-refresh from live W&B data — you only need to update the narrative text.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar25`).

LOOP FOREVER:

1. **Review current state**

   ```bash
   git log --oneline -10
   ```

   To see all runs in the last sweep sorted by val_bpb:

   ```bash
   uv run query_sweep.py
   ```

   To query a specific sweep by ID:

   ```bash
   uv run query_sweep.py entity/project/sweep_id
   ```

   With no argument, queries all finished runs in the project sorted by val_bpb — use this to find the global best. Pass a sweep ID to get the best run from that specific sweep via `sweep.best_run()`.

2. **Form a hypothesis and configure the sweep**

   Decide what to vary and why. Then:

   - Set `name` and `description` in `SWEEP_CONFIG` (required — see above).
   - Edit the `parameters` block.
   - Set `NUM_AGENTS` proportional to the sweep space: match the number of configs for small grids (≤8), use 4–8 agents for larger searches. Don't spin up more agents than there are meaningful configs to run.
   - Use `grid` for exhaustive small searches, `bayes` for large search spaces where you want the controller to guide sampling.

   CPU sandboxes (`SandboxResources(cpus=4, memory=8)`) are cheap. Switch `RESOURCES` to a GPU instance only if the model is too slow on CPU. Consult the **sandbox-sweeps** skill for resource config syntax.

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
   uv run query_sweep.py
   ```

6. **Decide: keep or discard**

   - **Keep** if val_bpb improved (lower) vs the current best, or if equal and simpler.
   - **Discard** if val_bpb is higher (worse).

7. **If keeping: bake the best config into agent/train.py defaults**

   Take the winning config from `uv run query_sweep.py` output and update the corresponding constants at the top of `agent/train.py`. Then commit:

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

9. **Update the W&B Report**

   Add a narrative paragraph to the report summarising what was tried, the outcome, and the next hypothesis. This is the research journal. Keep entries concise: sweep name, best val_bpb, one-sentence interpretation, keep/discard decision.

10. **Repeat**

**Timeout**: If a sweep exceeds 15 minutes wall clock, kill it (`Ctrl-C`), treat as failure, and move on.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human may be asleep and expects you to continue working *indefinitely* until manually stopped. If you run out of ideas, think harder — query W&B for patterns across all runs, try combining previous near-misses, or make more radical architectural changes.

Scale `NUM_AGENTS` to the sweep space — a 4-config grid should run 4 agents, a 2-config grid should run 2. Avoid unnecessary agents, but don't serialize work that can reasonably run in parallel.
