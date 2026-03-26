"""
Sweep harness for autoresearch parallel experiments.

Edit SWEEP_CONFIG, NUM_AGENTS, and RESOURCES below, then run:
    uv run sweep_harness.py

Blocks until all agents finish. Results are queryable via the W&B API
(use query_sweep.py) — the sweep ID is printed to stdout on start.
"""
import os

import wandb
from wandb.sdk.launch.create_job import create_job
from wandb.wandb_managed_agent import (
    ManagedAgentSession,
    ManagedAgentSessionConfig,
    SandboxResources,
    WBCodeArtifactJobSource,
)

ENTITY  = os.environ["WANDB_ENTITY"]
PROJECT = os.environ.get("WANDB_PROJECT", "autoresearch")

CONTAINER_IMAGE = "python:3.11-slim"

# ---------------------------------------------------------------------------
# Edit this section before each sweep
# ---------------------------------------------------------------------------
SWEEP_CONFIG = {
    "method": "grid",
    "program": "train.py",
    "metric": {"name": "final/val_bpb", "goal": "minimize"},
    "parameters": {
        "n_layer": {"values": [4]},
    },
}

NUM_AGENTS = 1
RESOURCES  = SandboxResources(cpus=4, memory=8)
# ---------------------------------------------------------------------------


def main() -> None:
    # 1. Upload current agent/ code as a job artifact
    print("Uploading code artifact...")
    if create_job(
        job_type="code",
        path="./agent",
        entity=ENTITY,
        project=PROJECT,
        name="autoresearch-job",
        entrypoint="python train.py",
    ) is None:
        raise RuntimeError("create_job returned None — artifact upload failed")
    job_artifact_id = f"{ENTITY}/{PROJECT}/autoresearch-job:latest"
    print(f"Job artifact: {job_artifact_id}")

    # 2. Create the sweep
    sweep_id      = wandb.sweep(SWEEP_CONFIG, entity=ENTITY, project=PROJECT)
    full_sweep_id = f"{ENTITY}/{PROJECT}/{sweep_id}"
    print(f"Sweep: https://wandb.ai/{full_sweep_id}")

    # 3. Run agents (attach_logs=True streams stdout/stderr from all sandboxes)
    cfg = ManagedAgentSessionConfig(
        sweep_id=sweep_id,
        entity=ENTITY,
        project=PROJECT,
        source=WBCodeArtifactJobSource(job_artifact_id),
        num_agents=NUM_AGENTS,
        container_image=CONTAINER_IMAGE,
        resources=RESOURCES,
    )
    with ManagedAgentSession(cfg) as session:
        session.run(attach_logs=True)

    # 4. Report best result
    best    = wandb.Api().sweep(full_sweep_id).best_run()
    val_bpb = best.summary_metrics.get("final/val_bpb", float("inf"))

    print("\n=== SWEEP COMPLETE ===")
    print(f"Best run:  {best.id}")
    print(f"val_bpb:   {val_bpb:.6f}")
    print(f"Sweep URL: https://wandb.ai/{full_sweep_id}")
    print("Run `uv run query_sweep.py` to see full results.")


if __name__ == "__main__":
    main()
