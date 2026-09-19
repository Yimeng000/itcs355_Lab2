"""Lab 2 — budgeted hyperparameter study.

Run:  python -m src.tune --trials 12 --budget-thb 150

The budget is enforced, not advisory. The study stops when projected spend would exceed
it, and reports what it did not get to. This is the habit the lab is teaching: compute is
a resource you spend deliberately, and a trial that is 0.3% better and four times the cost
is not better.

Every trial logs its estimated cost alongside its metric, so `scripts/compare_runs.py` can
rank by cost per point rather than by metric alone.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import mlflow

from src import config, costs, data, seeds
from src.train import git_commit

from cloudlayer.factory import get_adapter 

# Lab 2 search space: 12 configurations across three hyperparameters.
SEARCH_SPACE: dict[str, list] = {
    "n_estimators": [100, 300],
    "max_depth": [4, 8, 12],
    "min_samples_leaf": [1, 5],
}


def grid(space: dict[str, list]) -> list[dict]:
    keys = list(space)
    return [dict(zip(keys, values)) for values in itertools.product(*(space[k] for k in keys))]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ITCS355 Lab 2 — budgeted study")
    p.add_argument("--trials", type=int, default=12, help="minimum 12 for the lab")
    p.add_argument("--budget-thb", type=float, default=150.0)
    p.add_argument("--instance", default="e2-standard-4", help="key into src/costs.py PRICE_TABLE")
    p.add_argument("--seed", type=int, default=seeds.DEFAULT_SEED)

    p.add_argument("--n-estimators", type=int, default=None)
    p.add_argument("--max-depth", type=int, default=None)
    p.add_argument("--min-samples-leaf", type=int, default=None)

    p.add_argument("--experiment", default="itcs355-lab2")
    p.add_argument("--checkpoint", type=Path, default=Path("reports/tune_checkpoint.json"),
                   help="Resume file. Spot interruption should cost minutes, not the run.")
    return p.parse_args()


def load_checkpoint(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"completed": [], "spent_thb": 0.0}


def save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))


def main() -> None:
    args = parse_args()
    cfg = config.load(strict=False)
    adapter = get_adapter(cfg) 
    seed = seeds.set_all(args.seed)

    df = data.load_raw(cfg.raw_path)
    fingerprint = data.data_fingerprint(cfg.raw_path)
    train_df, val_df, test_df = data.split(df, seed=seed)

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment)

    state = load_checkpoint(args.checkpoint)
    # candidates = grid(SEARCH_SPACE)[: args.trials]
    if (
        args.n_estimators is not None
        and args.max_depth is not None
        and args.min_samples_leaf is not None
    ):
        candidates = [{
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
        }]
    else:
        candidates = grid(SEARCH_SPACE)[: args.trials]

    rate = costs.hourly_rate(cfg.provider, args.instance, spot=True)

    skipped: list[dict] = []
    for i, params in enumerate(candidates):
        # key = json.dumps(params, sort_keys=True)
        key = json.dumps(
            {**params, "seed": seed},
            sort_keys=True
        )
        if key in state["completed"]:
            print(f"trial {i}: already done, skipping (resumed from checkpoint)")
            continue

        if state["spent_thb"] >= args.budget_thb:
            skipped.append(params)
            continue

        started = time.perf_counter()

        job_id = adapter.submit_training(
            image_uri=cfg.training_image_uri,
            args={
                **params,
                "seed": seed,
                "experiment": args.experiment,
                "run-name": f"trial-{i:02d}",
                "metrics-out": f"/app/reports/trial-{i:02d}.json",
            },       
        )

        result = adapter.wait_training(job_id)

        if result["state"] != "PIPELINE_STATE_SUCCEEDED":
            print(f"trial {i}: remote job failed with state={result['state']}")
            continue

        metrics_path = Path(f"reports/trial-{i:02d}.json")
        remote_metrics = f"{cfg.blob_uri.rstrip('/')}/lab2/metrics/trial-{i:02d}.json"
        adapter.download(remote_metrics, str(metrics_path))
        metrics = json.loads(metrics_path.read_text())

        elapsed_h = (time.perf_counter() - started) / 3600.0
        trial_cost = elapsed_h * rate
        state["spent_thb"] += trial_cost

        with mlflow.start_run(run_name=f"trial-{i:02d}"):
            mlflow.log_params({
                **params,
                "seed": seed,
                "instance": args.instance,
                "spot": True,
            })

            mlflow.log_metrics({
                "val_roc_auc": metrics["val_roc_auc"],
                "val_pr_auc": metrics["val_pr_auc"],
                "test_roc_auc": metrics["test_roc_auc"],
                "test_pr_auc": metrics["test_pr_auc"],
                "duration_s": round(elapsed_h * 3600, 3),
                "cost_thb": round(trial_cost, 4),
            })

            mlflow.set_tags({
                "git_commit": git_commit(),
                "data_fingerprint": fingerprint,
                "training_job_id": job_id,
                "lab": "2",
            })

        state["completed"].append(key)
        save_checkpoint(args.checkpoint, state)
        print(f"trial {i}: {params} -> val_roc_auc={metrics['val_roc_auc']:.4f} "
              f"cost={trial_cost:.4f} THB  cumulative={state['spent_thb']:.4f}")

    print(f"\nspent {state['spent_thb']:.4f} of {args.budget_thb} THB")
    if skipped:
        print(f"BUDGET EXHAUSTED — {len(skipped)} configurations not run:")
        for s in skipped:
            print(f"  {s}")
        print("Report this in your README. Which trials you could not afford is a finding, "
              "not an embarrassment.")


if __name__ == "__main__":
    main()
