"""
Sweep harness for autoresearch parallel experiments.

Edit SWEEP_CONFIG, CONTAINER_IMAGE, and AGENTS below, then run:
    uv run sweep_harness.py

Each entry in AGENTS defines one sandbox. Give entries different
SandboxResources to build a heterogeneous fleet. Blocks until all
sandboxes finish, then writes last_sweep_result.json.
"""
import concurrent.futures
import json
import os
import queue
import threading

import wandb
from wandb.sandbox import Session
from wandb.wandb_managed_agent import (
    CodeArtifactSource,
    ManagedAgent,
    SandboxResources,
)

ENTITY  = os.environ["WANDB_ENTITY"]
PROJECT = os.environ.get("WANDB_PROJECT", "autoresearch")

CONTAINER_IMAGE = "python:3.11-slim"

# ---------------------------------------------------------------------------
# Edit this section before each sweep
# ---------------------------------------------------------------------------
SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "final/val_bpb", "goal": "minimize"},
    "parameters": {
        # Example — replace with whatever you want to explore:
        "DEPTH":     {"values": [6, 8, 10, 12]},
        "MATRIX_LR": {"distribution": "log_uniform_values", "min": 0.01, "max": 0.08},
    },
}

# One SandboxResources entry per sandbox. Duplicate, remove, or mix
# accelerator types freely — the number of entries controls the fleet size.
AGENTS = [
    SandboxResources(accelerators="H100:1", cpus=16, memory=64),
    SandboxResources(accelerators="H100:1", cpus=16, memory=64),
    SandboxResources(accelerators="H100:1", cpus=16, memory=64),
    SandboxResources(accelerators="H100:1", cpus=16, memory=64),
]
# ---------------------------------------------------------------------------


_STOP = object()


def _printer_thread(log_queue: queue.Queue, label_width: int) -> None:
    """Serialise log lines from all sandboxes to stdout, docker-compose style."""
    while True:
        item = log_queue.get()
        if item is _STOP:
            break
        label, line = item
        print(f"{label:<{label_width}}  | {line}", flush=True)


def main() -> None:
    # 1. Upload current code as a job artifact via the W&B SDK
    print("Uploading code artifact...")
    artifact = wandb.create_job(
        job_type="code",
        path="./agent",
        entity=ENTITY,
        project=PROJECT,
        name="autoresearch-job",
        entrypoint="train.py",
    )
    if artifact is None:
        raise RuntimeError("wandb.create_job returned None — artifact upload failed")
    job_artifact_id = f"{ENTITY}/{PROJECT}/autoresearch-job:latest"
    print(f"Job artifact: {job_artifact_id}")

    # 2. Create the sweep
    sweep_id = wandb.sweep(SWEEP_CONFIG, entity=ENTITY, project=PROJECT)
    full_sweep_id = f"{ENTITY}/{PROJECT}/{sweep_id}"
    print(f"Sweep: https://wandb.ai/{full_sweep_id}")

    # 3. Build source and managed agent (shared across all sandboxes)
    source  = CodeArtifactSource(job_artifact_id)
    managed = ManagedAgent(
        sweep_id=sweep_id,
        entity=ENTITY,
        project=PROJECT,
        source=source,
    )
    image = managed.resolve_container_image(default=CONTAINER_IMAGE)

    # 4. Start log-streaming printer thread
    label_width = len(f"agent-{len(AGENTS) - 1}")
    log_queue: queue.Queue = queue.Queue()
    printer = threading.Thread(
        target=_printer_thread,
        args=(log_queue, label_width),
        daemon=True,
    )
    printer.start()

    # 5. Create and start each sandbox line-by-line, then collect futures
    futures: list[concurrent.futures.Future] = []
    with Session() as session:
        for i, resources in enumerate(AGENTS):
            sandbox = session.sandbox(
                container_image=image,
                resources=resources.to_cwsandbox_dict(),
            )
            managed.consume_sandbox(sandbox)
            sandbox.start().result()
            managed.log_device_resources(sandbox, log_queue, label=f"agent-{i}")
            future, _ = source.attach(sandbox, i, log_queue, managed._sweep_path)
            futures.append(future)

        # 6. Block until every agent finishes
        concurrent.futures.wait(futures)

    log_queue.put(_STOP)
    printer.join()

    # 7. Query best result from the W&B API
    api   = wandb.Api()
    sweep = api.sweep(full_sweep_id)
    best  = sweep.best_run()

    val_bpb      = best.summary_metrics.get("final/val_bpb", float("inf"))
    peak_vram_mb = best.summary_metrics.get("final/peak_vram_mb", 0.0)

    print("\n=== SWEEP COMPLETE ===")
    print(f"Best run:     {best.id}")
    print(f"val_bpb:      {val_bpb:.6f}")
    print(f"peak_vram_mb: {peak_vram_mb:.1f}")
    print(f"Config:       {json.dumps(dict(best.config), indent=2)}")
    print(f"Sweep URL:    https://wandb.ai/{full_sweep_id}")

    result = {
        "sweep_id": full_sweep_id,
        "best_run_id": best.id,
        "val_bpb": val_bpb,
        "peak_vram_mb": peak_vram_mb,
        "config": dict(best.config),
    }
    with open("last_sweep_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Result written to last_sweep_result.json")


if __name__ == "__main__":
    main()
