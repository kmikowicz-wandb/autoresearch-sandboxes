"""
Sweep harness for autoresearch parallel experiments.

Edit SWEEP_CONFIG, NUM_AGENTS, and RESOURCES below, then run:
    uv run sweep_harness.py

Blocks until all agents finish, then writes last_sweep_result.json.
"""
import json
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
    "metric": {"name": "final/val_bpb", "goal": "minimize"},
    "program": "train.py",
    "parameters": {
        # Re-tune LR schedule shape with new lr=5e-3 + bf16 defaults
        "min_lr_ratio": {"values": [0.0, 0.05, 0.1, 0.2]},
        "warmup_secs":  {"values": [2.0, 5.0, 10.0]},
    },
}

NUM_AGENTS = 4
RESOURCES  = SandboxResources(cpus=8, memory=16)
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
    print(f"Config:    {json.dumps(dict(best.config), indent=2)}")
    print(f"Sweep URL: https://wandb.ai/{full_sweep_id}")

    result = {
        "sweep_id":    full_sweep_id,
        "best_run_id": best.id,
        "val_bpb":     val_bpb,
        "config":      dict(best.config),
    }
    with open("last_sweep_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Result written to last_sweep_result.json")


if __name__ == "__main__":
    main()
