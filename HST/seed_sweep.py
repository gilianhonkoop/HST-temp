from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import tempfile
from types import SimpleNamespace
from typing import Any

import yaml


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config did not parse as a mapping: {path}")
    return cfg


def _set_nested(cfg: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur = cfg
    for part in parts[:-1]:
        next_value = cur.setdefault(part, {})
        if not isinstance(next_value, dict):
            raise ValueError(f"Cannot set {path}: {part} is not a mapping")
        cur = next_value
    cur[parts[-1]] = value


def _parse_seeds(seed_text: str | None, n_seeds: int, start_seed: int) -> list[int]:
    if seed_text:
        seeds = [int(item.strip()) for item in seed_text.split(",") if item.strip()]
        if not seeds:
            raise ValueError("--seeds was provided but no seeds were parsed")
        return seeds
    if n_seeds < 1:
        raise ValueError("--n-seeds must be at least 1")
    return [start_seed + i for i in range(n_seeds)]


def _numeric_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    out = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and value is not None:
            out[key] = float(value)
    return out


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def _print_summary(summary: dict[str, dict[str, float]], metric_order: list[str]) -> None:
    rows = []
    for metric in metric_order:
        stats = summary.get(metric)
        if stats is None:
            continue
        rows.append((metric, stats["mean"], stats["std"]))

    if not rows:
        print("No numeric metrics found to summarize.")
        return

    metric_width = max(len("metric"), max(len(row[0]) for row in rows))
    print("\nSeed sweep summary")
    print(f"{'metric':<{metric_width}}  {'mean':>12}  {'std':>12}")
    print(f"{'-' * metric_width}  {'-' * 12}  {'-' * 12}")
    for metric, mean, std in rows:
        print(f"{metric:<{metric_width}}  {mean:12.6f}  {std:12.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train one config with multiple seeds and summarize test metrics."
    )
    parser.add_argument("--config", required=True, help="Config path, relative to the current directory or absolute.")
    parser.add_argument("--n-seeds", type=int, default=5, help="Number of consecutive seeds to run.")
    parser.add_argument("--start-seed", type=int, default=42, help="First seed when --seeds is not provided.")
    parser.add_argument("--seeds", default=None, help="Comma-separated explicit seed list, e.g. 1,2,3.")
    parser.add_argument(
        "--run-prefix",
        default=None,
        help="Prefix for generated logging.run_name values. Defaults to the config logging.run_name.",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="Where to write the JSON report. Defaults to <save_dir>/<run_prefix>/seed_sweep_summary.json.",
    )
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Set logging.use_wandb=false for all seed runs.",
    )
    args = parser.parse_args()

    import numpy as np
    import torch

    from train import main as train_main

    config_path = os.path.abspath(args.config)
    base_cfg = _load_yaml(config_path)
    seeds = _parse_seeds(args.seeds, args.n_seeds, args.start_seed)

    logging_cfg = base_cfg.setdefault("logging", {})
    base_cfg.setdefault("train", {})
    save_dir = str(logging_cfg.get("save_dir", "seed_sweep_runs"))
    base_run_name = str(logging_cfg.get("run_name", "seed_sweep"))
    run_prefix = args.run_prefix or f"{base_run_name}"

    report_path = args.report_path
    if report_path is None:
        report_dir = os.path.join(save_dir, run_prefix)
        report_path = os.path.join(report_dir, "seed_sweep_summary.json")
    report_path = os.path.abspath(report_path)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    results: list[dict[str, Any]] = []
    all_metric_values: dict[str, list[float]] = {}

    for index, seed in enumerate(seeds, start=1):
        cfg = yaml.safe_load(yaml.safe_dump(base_cfg, sort_keys=False))
        run_name = f"{run_prefix}_seed_{seed}"
        run_dir = os.path.join(save_dir, run_name)

        _set_nested(cfg, "train.seed", int(seed))
        _set_nested(cfg, "train.run_test", True)
        _set_nested(cfg, "logging.run_name", run_name)
        if args.disable_wandb:
            _set_nested(cfg, "logging.use_wandb", False)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        print("\n======================================================")
        print(f"Seed run {index}/{len(seeds)}")
        print(f"  Seed      : {seed}")
        print(f"  Run name  : {run_name}")
        print(f"  Run dir   : {run_dir}")
        print("======================================================")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.safe_dump(cfg, tmp, sort_keys=False)
            temp_config_path = tmp.name

        try:
            objective, num_params = train_main(SimpleNamespace(config=temp_config_path, resume=False))
        finally:
            os.remove(temp_config_path)

        metrics_path = os.path.join(run_dir, "test_metrics.json")
        if not os.path.exists(metrics_path):
            raise FileNotFoundError(
                f"Expected test metrics were not written for seed {seed}: {metrics_path}"
            )

        with open(metrics_path, "r") as f:
            test_metrics = json.load(f)
        numeric = _numeric_metrics(test_metrics)
        for metric, value in numeric.items():
            all_metric_values.setdefault(metric, []).append(value)

        results.append(
            {
                "seed": seed,
                "run_name": run_name,
                "run_dir": run_dir,
                "metrics_path": metrics_path,
                "objective": float(objective),
                "num_params": int(num_params),
                "metrics": test_metrics,
            }
        )

    summary = {metric: _mean_std(values) for metric, values in sorted(all_metric_values.items())}
    preferred = [
        "test/illicit_f1",
        "test/illicit_pr_auc",
        "test/illicit_roc_auc",
        "test/illicit_precision",
        "test/illicit_recall",
        "test/licit_f1",
        "test_loss",
        "best_threshold",
    ]
    metric_order = preferred + [metric for metric in summary if metric not in preferred]

    report = {
        "config": config_path,
        "seeds": seeds,
        "run_prefix": run_prefix,
        "summary": summary,
        "runs": results,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    _print_summary(summary, metric_order)
    print(f"\nWrote report: {report_path}")


if __name__ == "__main__":
    main()
