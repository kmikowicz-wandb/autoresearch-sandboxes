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
WANDB_API_KEY=...   # your W&B API key (set via wandb login or export)
WANDB_ENTITY=...    # your W&B username or team
```

## Step 1 — Define the sweep config

Define the hyperparameter search space. Use `bayes` for efficient search, `grid` or `random` for exhaustive or cheap searches.

```python
sweep_config = {
    "method": "bayes",
    "metric": {"name": "final/val_bpb", "goal": "minimize"},
    "parameters": {
        "lr":    {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-2},
        "depth": {"values": [6, 8, 10, 12]},
    },
}
```

Read hyperparameters from `wandb.config` in `train.py`:

```python
wandb.init(project=project, config={"lr": LR_DEFAULT, "depth": DEPTH_DEFAULT})
lr    = wandb.config.get("lr",    LR_DEFAULT)
depth = wandb.config.get("depth", DEPTH_DEFAULT)
# ...
wandb.log({"final/val_bpb": val_bpb})
```

## Step 2 — Register a W&B job from your code

A **job artifact** packages your code so the sandbox can download and run it.

```python
from wandb.sdk.launch.create_job import create_job

entity  = os.environ["WANDB_ENTITY"]
project = "autoresearch"

create_job(
    job_type="code",
    path="./agent",          # directory containing train.py
    entity=entity,
    project=project,
    name="autoresearch-job",
    entrypoint="train.py",
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

## Step 4 — Launch sandbox agents with ManagedAgentSession

`ManagedAgentSession` is the recommended way to run a pool of agents. It handles
sandbox adoption on restart, the wait loop, and log streaming — all in one call.

```python
from wandb.wandb_managed_agent import (
    ManagedAgentSession,
    ManagedAgentSessionConfig,
    WBCodeArtifactJobSource,
    SandboxResources,
)

cfg = ManagedAgentSessionConfig(
    sweep_id=sweep_id,
    entity=entity,
    project=project,
    source=WBCodeArtifactJobSource(job_artifact_id),
    num_agents=4,
    container_image="python:3.11-slim",
    resources=SandboxResources(accelerators="H100:1", cpus=16, memory=64),
)

with ManagedAgentSession(cfg) as session:
    session.run(attach_logs=True)   # blocks until all agents finish
```

`attach_logs=True` streams log lines to stdout docker-compose style:

```
agent-0  | wandb: Run xqz7a started
agent-1  | wandb: Run k3m2p started
agent-0  | final/val_bpb: 0.9831
```

## Step 5 — Read results

```python
api   = wandb.Api()
sweep = api.sweep(full_sweep_id)
best  = sweep.best_run()
print(f"Best run: {best.id}  val_bpb={best.summary_metrics.get('final/val_bpb'):.6f}")
print(f"Config:   {dict(best.config)}")
```

Apply the best hyperparameters back to `train.py` as new defaults, then commit.

## Minimal end-to-end example

```python
import os
import wandb
from wandb.sdk.launch.create_job import create_job
from wandb.wandb_managed_agent import (
    ManagedAgentSession, ManagedAgentSessionConfig,
    WBCodeArtifactJobSource, SandboxResources,
)

entity  = os.environ["WANDB_ENTITY"]
project = "autoresearch"

# 1. Upload code as a job artifact
create_job(job_type="code", path="./agent", entity=entity, project=project,
           name="autoresearch-job", entrypoint="train.py")
job_artifact_id = f"{entity}/{project}/autoresearch-job:latest"

# 2. Create sweep
sweep_id = wandb.sweep(
    {"method": "bayes",
     "metric": {"name": "final/val_bpb", "goal": "minimize"},
     "parameters": {"lr": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-2}}},
    entity=entity, project=project,
)
full_sweep_id = f"{entity}/{project}/{sweep_id}"

# 3. Run agents
cfg = ManagedAgentSessionConfig(
    sweep_id=sweep_id, entity=entity, project=project,
    source=WBCodeArtifactJobSource(job_artifact_id),
    num_agents=4,
    resources=SandboxResources(accelerators="H100:1", cpus=16, memory=64),
)
with ManagedAgentSession(cfg) as session:
    session.run(attach_logs=True)

# 4. Get best result
best = wandb.Api().sweep(full_sweep_id).best_run()
print(f"val_bpb={best.summary_metrics.get('final/val_bpb'):.6f}  config={dict(best.config)}")
```

## Source modes

| Class | When to use |
|---|---|
| `UserImageSource()` | Container image already contains `wandb` and all code |
| `WBCodeArtifactJobSource(job_id)` | Generic image; code downloaded via W&B job artifact (code source) at runtime |
| `WBImageJobSource(job_id)` | W&B job artifact specifies its own docker image (`source_type: image`) |

## Advanced: heterogeneous agents

If your agents need different sources, images, or resources from one another,
use `ManagedAgent` and `wandb.sandbox.Session` directly instead of
`ManagedAgentSession`. You are responsible for log draining and the wait loop.
See `sdk/cw-sandbox-sweep-sdk/quickstart.py` for a worked example.

## Integration into the experiment loop

```
OLD: edit train.py → git commit → uv run train.py → read val_bpb → keep/discard
NEW: edit agent/train.py → git commit → uv run sweep_harness.py → read best val_bpb → apply best config → keep/discard
```

The sandbox loop is slower to start (~1 min overhead) but runs many configurations in parallel, making it worthwhile when exploring a multi-dimensional search space or when local GPU is unavailable.
