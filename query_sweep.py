"""
Query W&B run results.

Usage:
    uv run query_sweep.py                            # best runs across the whole project
    uv run query_sweep.py <entity/project/sweep_id>  # best run in one specific sweep
"""
import os
import sys

import wandb


def main():
    api = wandb.Api()

    if len(sys.argv) > 1:
        sweep = api.sweep(sys.argv[1])
        best  = sweep.best_run()
        print(f"Sweep:   {sys.argv[1]}")
        print(f"Best run: {best.id}  val_bpb={best.summary_metrics.get('final/val_bpb', 'N/A'):.6f}")
        print(f"Config:   {dict(best.config)}")
    else:
        entity  = os.environ["WANDB_ENTITY"]
        project = os.environ.get("WANDB_PROJECT", "autoresearch")
        runs    = api.runs(
            f"{entity}/{project}",
            filters={"state": "finished"},
            order="+summary_metrics.final/val_bpb",
        )
        print(f"{'id':<8}  {'val_bpb':>8}  config")
        for r in runs[:20]:
            bpb = r.summary_metrics.get("final/val_bpb", float("inf"))
            print(f"{r.id:<8}  {bpb:>8.6f}  {dict(r.config)}")


if __name__ == "__main__":
    main()
