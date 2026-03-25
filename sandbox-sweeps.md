# Sandbox Sweeps Skill

Instead of running a single experiment locally with `uv run train.py`, you can run a **W&B hyperparameter sweep** across many sandbox agents in parallel on CoreWeave GPU hardware. Each agent runs your training script with a different set of hyperparameters sampled from the sweep config.

## When to use this

Use sandbox sweeps when:
- You want to search over multiple hyperparameters at once (e.g. LR × depth × batch size).
- You have a promising direction but are unsure of the right scale.
- You want to run experiments faster by parallelising across several GPUs.

## Prerequisites

The following environment variables must be set:

```bash
WANDB_API_KEY=...        # your W&B API key
WANDB_ENTITY=...         # your W&B username or team
CWSANDBOX_BASE_URL=...   # CoreWeave sandbox endpoint
```

The W&B SDK with sandbox support must be importable. Import from the editable install in this repo's environment:

```python
from wandb.wandb_managed_agent import (
    ManagedAgentSession,
    ManagedAgentSessionConfig,
    CodeArtifactSource,
)
from wandb.sdk.wandb_sweep import sweep as wandb_sweep
import wandb
```

## Step 1 — Define the sweep config

Define the hyperparameter search space. Use `bayes` method for efficient search; use `grid` or `random` for exhaustive or cheap searches.

```python
sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_bpb", "goal": "minimize"},
    "parameters": {
        "lr":    {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-2},
        "depth": {"values": [6, 8, 10, 12]},
    },
}
```

Map hyperparameters to `train.py` by reading them from `wandb.config` at the top of your training run:

```python
# in train.py — read from sweep config when available
import wandb
run = wandb.init()
lr    = wandb.config.get("lr",    LR_DEFAULT)
depth = wandb.config.get("depth", DEPTH_DEFAULT)
```

Log the key metric so the sweep agent can report it:

```python
wandb.log({"val_bpb": val_bpb})
```

## Step 2 — Register a W&B job from your code

A **job artifact** packages your code so the sandbox can download and run it.

```python
import subprocess, sys, os

entity  = os.environ["WANDB_ENTITY"]
project = "autoresearch"

result = subprocess.run(
    [
        sys.executable, "-m", "wandb", "job", "create",
        "code", ".",
        "--entity", entity,
        "--project", project,
        "--name", "autoresearch-job",
        "--entry-point", "train.py",
    ],
    check=True, capture_output=True, text=True,
)
job_artifact_id = f"{entity}/{project}/autoresearch-job:latest"
```

Re-run this step whenever you change `train.py` so the sandbox picks up the latest code.

## Step 3 — Create the sweep

```python
sweep_id = wandb.sweep(sweep_config, entity=entity, project=project)
full_sweep_id = f"{entity}/{project}/{sweep_id}"
print(f"Sweep: https://wandb.ai/{full_sweep_id}")
```

## Step 4 — Launch sandbox agents

Configure and run the sandbox agent pool. Each agent pulls a run from the sweep, trains, logs results, and pulls the next run.

```python
from wandb.wandb_managed_agent import (
    ManagedAgentSession,
    ManagedAgentSessionConfig,
    CodeArtifactSource,
    SandboxResources,
)

cfg = ManagedAgentSessionConfig(
    sweep_id=sweep_id,
    entity=entity,
    project=project,
    source=CodeArtifactSource(job_artifact_id),
    num_agents=4,                          # parallel sandboxes
    container_image="python:3.11-slim",    # base image; wandb + deps installed at runtime
    resources=SandboxResources(
        accelerators="H100:1",             # one H100 per sandbox
        cpus=16,
        memory=64,                         # GiB
    ),
)

with ManagedAgentSession(cfg) as session:
    session.run(attach_logs=True)          # blocks until all agents finish
```

`session.run()` blocks until every sandbox completes or the sweep is exhausted. Log lines stream to stdout docker-compose style:

```
agent-0  | wandb: Run xqz7a started
agent-1  | wandb: Run k3m2p started
agent-0  | val_bpb: 0.9831
```

## Step 5 — Read results

After the session returns, query the best run:

```python
api  = wandb.Api()
sweep = api.sweep(full_sweep_id)
best  = sweep.best_run()
print(f"Best run: {best.id}  val_bpb={best.summary['val_bpb']:.6f}")
print(f"Config:   {dict(best.config)}")
```

Apply the best hyperparameters back to `train.py` as new defaults, then commit.

## Minimal end-to-end example

```python
import os, subprocess, sys
import wandb
from wandb.wandb_managed_agent import (
    ManagedAgentSession, ManagedAgentSessionConfig,
    CodeArtifactSource, SandboxResources,
)

entity  = os.environ["WANDB_ENTITY"]
project = "autoresearch"

# 1. Upload code as a job artifact
subprocess.run([
    sys.executable, "-m", "wandb", "job", "create", "code", ".",
    "--entity", entity, "--project", project,
    "--name", "autoresearch-job", "--entry-point", "train.py",
], check=True)

# 2. Create sweep
sweep_id = wandb.sweep(
    {"method": "bayes",
     "metric": {"name": "val_bpb", "goal": "minimize"},
     "parameters": {"lr": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-2}}},
    entity=entity, project=project,
)

# 3. Run agents
cfg = ManagedAgentSessionConfig(
    sweep_id=sweep_id, entity=entity, project=project,
    source=CodeArtifactSource(f"{entity}/{project}/autoresearch-job:latest"),
    num_agents=4,
    resources=SandboxResources(accelerators="H100:1", cpus=16, memory=64),
)
with ManagedAgentSession(cfg) as session:
    session.run(attach_logs=True)

# 4. Get best result
best = wandb.Api().sweep(f"{entity}/{project}/{sweep_id}").best_run()
print(f"val_bpb={best.summary['val_bpb']:.6f}  config={dict(best.config)}")
```

## CLI equivalent

You can also launch agents from the terminal without writing Python:

```bash
# Case 1 — image already has code + wandb (no artifact needed)
wandb cw-agent entity/project/sweep_id

# Case 2 — upload code first, then run
wandb job create code . --entity acme --project autoresearch --name autoresearch-job --entry-point train.py
wandb cw-agent sweep_id --artifact acme/autoresearch/autoresearch-job:latest -e acme -p autoresearch

# With a resource config file (resources.yaml):
wandb cw-agent sweep_id --artifact acme/autoresearch/autoresearch-job:latest \
    --resource-config resources.yaml
```

`resources.yaml` example:

```yaml
num_agents: 4
container_image: python:3.11-slim
resources:
  accelerators: H100:1
  cpus: 16
  memory: 64
```

## Source modes

| Class | When to use |
|---|---|
| `EnvOnlySource()` | Container image already contains `wandb` and all code |
| `CodeArtifactSource(job_id)` | Generic image; code downloaded via W&B job artifact at runtime |
| `JobArtifactSource(job_id)` | Job artifact specifies its own docker image (`source_type: image`) |

## Integration into the experiment loop

Replace the local `uv run train.py` step with a sandbox sweep when you want parallel search:

```
OLD: edit train.py → git commit → uv run train.py → read val_bpb → keep/discard
NEW: edit train.py → git commit → upload job artifact → create sweep → run sandbox agents → read best val_bpb → apply best config → keep/discard
```

The sandbox loop is slower to start (~1 min overhead) but runs many configurations in parallel, making it worthwhile when exploring a multi-dimensional search space or when local GPU is unavailable.
