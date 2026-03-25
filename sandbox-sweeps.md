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

The W&B SDK with sandbox support must be importable:

```python
from wandb.sandbox import Session
from wandb.wandb_managed_agent import (
    ManagedAgent,
    WBCodeArtifactJobSource,
    SandboxResources,
)
from wandb.sdk.launch.create_job import create_job
import wandb
```

## Step 1 — Define the sweep config

Define the hyperparameter search space. Use `bayes` method for efficient search; use `grid` or `random` for exhaustive or cheap searches.

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

Map hyperparameters to `train.py` by reading them from `wandb.config` after `wandb.init()`:

```python
# in train.py — read from sweep config when available
wandb.init(project=project, config={"lr": LR_DEFAULT, "depth": DEPTH_DEFAULT})
lr    = wandb.config.get("lr",    LR_DEFAULT)
depth = wandb.config.get("depth", DEPTH_DEFAULT)
```

Log the key metric so the sweep controller can rank runs:

```python
wandb.log({"final/val_bpb": val_bpb})
```

## Step 2 — Register a W&B job from your code

A **job artifact** packages your code so the sandbox can download and run it. Use the SDK — no subprocess needed:

```python
import os
from wandb.sdk.launch.create_job import create_job

entity  = os.environ["WANDB_ENTITY"]
project = "autoresearch"

artifact = create_job(
    job_type="code",
    path="./agent",          # directory to upload
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

## Step 4 — Launch sandbox agents

Create a `ManagedAgent` and open a `Session`, then create each sandbox line-by-line.
`source.start()` is called after `sandbox.start()` and returns a `RunHandle`.

```python
import concurrent.futures, queue, threading
from wandb.sandbox import Session
from wandb.wandb_managed_agent import ManagedAgent, WBCodeArtifactJobSource, SandboxResources

source  = WBCodeArtifactJobSource(job_artifact_id)
managed = ManagedAgent(sweep_id=sweep_id, entity=entity, project=project, source=source)
image   = managed.resolve_container_image(default="python:3.11-slim")

handles = []
with Session() as session:
    for i in range(num_agents):
        sandbox = session.sandbox(
            container_image=image,
            resources=SandboxResources(accelerators="H100:1", cpus=16, memory=64).to_cwsandbox_dict(),
        )
        managed.consume_sandbox(sandbox)   # applies tags, env vars, startup command
        sandbox.start().result()

        handle = source.start(sandbox, managed._sweep_path)
        handles.append(handle)

    concurrent.futures.wait([h.future for h in handles])

for handle in handles:
    handle.close()
```

### RunHandle

`source.start()` returns a `RunHandle`:

```python
class RunHandle:
    future:    concurrent.futures.Future  # completes when the agent exits
    log_lines: Iterable[str] | None       # stream of log lines from the sandbox
    def close(self) -> None: ...          # stop the log stream
```

### Streaming logs (docker-compose style)

Drain each handle's `log_lines` in a per-sandbox thread into a shared queue:

```python
_STOP = object()

def _printer(q, label_width):
    while True:
        item = q.get()
        if item is _STOP:
            break
        label, line = item
        print(f"{label:<{label_width}}  | {line}", flush=True)

def _drain(log_lines, q, label):
    if log_lines is None:
        return
    for line in log_lines:
        q.put((label, line))

log_queue = queue.Queue()
printer = threading.Thread(target=_printer, args=(log_queue, label_width), daemon=True)
printer.start()

with Session() as session:
    for i in range(num_agents):
        # ... create and start sandbox as above ...
        handle = source.start(sandbox, managed._sweep_path)
        handles.append(handle)
        threading.Thread(target=_drain, args=(handle.log_lines, log_queue, f"agent-{i}"), daemon=True).start()
        # (handle.log_lines may be None if log streaming is unavailable)

    concurrent.futures.wait([h.future for h in handles])

for handle in handles:
    handle.close()
log_queue.put(_STOP)
printer.join()
```

Log lines stream to stdout docker-compose style:

```
agent-0  | wandb: Run xqz7a started
agent-1  | wandb: Run k3m2p started
agent-0  | val_bpb: 0.9831
```

## Step 5 — Read results

After all handles complete, query the best run:

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
import concurrent.futures, os, queue, threading
import wandb
from wandb.sandbox import Session
from wandb.wandb_managed_agent import ManagedAgent, WBCodeArtifactJobSource, SandboxResources
from wandb.sdk.launch.create_job import create_job

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
     "parameters": {"MATRIX_LR": {"distribution": "log_uniform_values", "min": 0.01, "max": 0.08}}},
    entity=entity, project=project,
)
full_sweep_id = f"{entity}/{project}/{sweep_id}"

# 3. Build managed agent
source  = WBCodeArtifactJobSource(job_artifact_id)
managed = ManagedAgent(sweep_id=sweep_id, entity=entity, project=project, source=source)
image   = managed.resolve_container_image(default="python:3.11-slim")

# 4. Launch sandboxes line-by-line, collect RunHandles
handles = []
with Session() as session:
    for i in range(4):
        sandbox = session.sandbox(
            container_image=image,
            resources=SandboxResources(accelerators="H100:1", cpus=16, memory=64).to_cwsandbox_dict(),
        )
        managed.consume_sandbox(sandbox)
        sandbox.start().result()
        handle = source.start(sandbox, managed._sweep_path)
        handles.append(handle)
    concurrent.futures.wait([h.future for h in handles])

for h in handles:
    h.close()

# 5. Get best result
best = wandb.Api().sweep(full_sweep_id).best_run()
print(f"val_bpb={best.summary_metrics.get('final/val_bpb'):.6f}  config={dict(best.config)}")
```

## Source modes

| Class | When to use |
|---|---|
| `UserImageSource()` | Container image already contains `wandb` and all code |
| `WBCodeArtifactJobSource(job_id)` | Generic image; code downloaded via W&B job artifact at runtime |
| `WBImageJobSource(job_id)` | Job artifact specifies its own docker image (`source_type: image`) |

All three implement the same interface: `get_image()`, `apply_command()`, `start()` → `RunHandle`.

## Integration into the experiment loop

```
OLD: edit train.py → git commit → uv run train.py → read val_bpb → keep/discard
NEW: edit agent/train.py → git commit → uv run sweep_harness.py → read best val_bpb → apply best config → keep/discard
```

The sandbox loop is slower to start (~1 min overhead) but runs many configurations in parallel, making it worthwhile when exploring a multi-dimensional search space or when local GPU is unavailable.
