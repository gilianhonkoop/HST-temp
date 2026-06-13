import argparse
import copy
import json
import os
from pathlib import Path
from statistics import mean, stdev

import train_config
import config


class FixedTrial:
    def __init__(self, number):
        self.number = number
        self.user_attrs = {}
        self.params = {}

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value


def parse_seeds(seed_text):
    return [int(seed.strip()) for seed in seed_text.split(",") if seed.strip()]


def prepare_run_config(run_config):
    run_config = copy.deepcopy(run_config)

    data_format = run_config.get("data", {}).get("dataset_type") or run_config.get("data", {}).get("data_format")
    if data_format and "data_format" not in run_config["hyperparams_fix"]:
        run_config["hyperparams_fix"]["data_format"] = data_format

    for key in ["aml_split", "aml_train_ratio", "aml_val_ratio"]:
        if key in run_config.get("data", {}) and key not in run_config["hyperparams_fix"]:
            run_config["hyperparams_fix"][key] = run_config["data"][key]

    if "pred_threshold" not in run_config["hyperparams_fix"]:
        threshold = run_config.get("threshold")
        if threshold is None:
            threshold = run_config.get("data", {}).get("threshold")
        run_config["hyperparams_fix"]["pred_threshold"] = 0.5 if threshold is None else threshold

    if (
        "wandb" in run_config
        and "epoch_logging" in run_config["wandb"]
        and "wandb_epoch_logging" not in run_config["hyperparams_fix"]
    ):
        run_config["hyperparams_fix"]["wandb_epoch_logging"] = run_config["wandb"]["epoch_logging"]

    task = run_config["data"]["task"]
    data_dir = run_config["data"].get("data_dir")
    data_root = data_dir if data_dir else task
    embedding_type = run_config["hyperparams_fix"]["embedding_type"]

    run_config["subgraphs_path"] = os.path.join(data_root, "subgraphs.pth")
    run_config["graph_path"] = os.path.join(data_root, "edge_list.txt")
    run_config["shortest_paths_path"] = os.path.join(data_root, "shortest_path_matrix.npy")
    run_config["degree_sequence_path"] = os.path.join(data_root, "degree_sequence.txt")
    run_config["ego_graph_path"] = os.path.join(data_root, "ego_graphs.txt")
    run_config["similarities_path"] = os.path.join(data_root, "similarities/")

    if embedding_type == "gin":
        run_config["embedding_path"] = os.path.join(data_root, "gin_embeddings.pth")
    elif embedding_type == "graphsaint":
        run_config["embedding_path"] = os.path.join(data_root, "graphsaint_gcn_embeddings.pth")
    else:
        raise NotImplementedError(f"Unsupported embedding_type: {embedding_type}")

    global_node_embedding_path = run_config["data"].get("global_node_embedding_path")
    if global_node_embedding_path is not None:
        run_config["hyperparams_fix"]["global_node_embedding_path"] = global_node_embedding_path
        run_config["embedding_path"] = global_node_embedding_path

    if run_config.get("tb", {}).get("local", False):
        run_config["tb"]["dir_full"] = run_config["tb"]["dir"]
    else:
        run_config["tb"]["dir_full"] = os.path.join(config.PROJECT_ROOT, run_config["tb"]["dir"])

    run_config.setdefault("hyperparams_optuna", {})
    return run_config


def numeric_metrics(metrics):
    out = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            out[key] = float(value)
    return out


def summarize(seed_results):
    metric_values = {}
    for result in seed_results:
        for key, value in numeric_metrics(result["test_results"]).items():
            metric_values.setdefault(key, []).append(value)

    summary = {}
    for key, values in sorted(metric_values.items()):
        summary[key] = {
            "mean": mean(values),
            "std": stdev(values) if len(values) > 1 else 0.0,
            "n": len(values),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run one SubGNN config for multiple fixed seeds.")
    parser.add_argument(
        "-config_path",
        type=str,
        default="config_files/aml_HI_Small/aml_config_best_val.json",
        help="Fixed config to run. hyperparams_optuna must be empty.",
    )
    parser.add_argument("-seeds", type=str, default="42,43,44", help="Comma-separated seeds.")
    parser.add_argument("-summary_path", type=str, default=None, help="Where to write aggregate JSON.")
    args = parser.parse_args()

    config_path = Path(args.config_path)
    run_config = prepare_run_config(train_config.read_json(str(config_path)))
    if run_config.get("hyperparams_optuna"):
        raise ValueError(
            "run_fixed_seeds.py expects a single fixed config. Move chosen values into "
            "hyperparams_fix and set hyperparams_optuna to {}."
        )

    seeds = parse_seeds(args.seeds)
    if not seeds:
        raise ValueError("No seeds provided.")

    base_wandb_name = run_config.get("wandb", {}).get("name", "subgnn-fixed")
    base_tb_name = run_config.get("tb", {}).get("name", base_wandb_name)
    run_config["run_test"] = True
    run_config["optuna"]["opt_n_trials"] = 1
    run_config["optuna"]["opt_n_cores"] = 1

    seed_results = []
    for idx, seed in enumerate(seeds):
        seed_config = copy.deepcopy(run_config)
        seed_config["hyperparams_fix"]["seed"] = seed
        seed_config["wandb"]["name"] = f"{base_wandb_name}-seed-{seed}"
        seed_config["wandb"]["log_best_only"] = False
        seed_config["wandb"]["epoch_logging"] = True
        seed_config["hyperparams_fix"]["wandb_epoch_logging"] = True
        seed_config["tb"]["name"] = f"{base_tb_name}-seeds"

        print(f"Running seed {seed} ({idx + 1}/{len(seeds)})", flush=True)
        trial = FixedTrial(number=idx)
        objective_value = train_config.train_model(seed_config, trial=trial)
        test_results = trial.user_attrs.get("test_results")
        if test_results is None:
            raise RuntimeError(f"No test results were produced for seed {seed}.")

        seed_results.append({
            "seed": seed,
            "objective_value": float(objective_value),
            "results_path": trial.user_attrs.get("results_path"),
            "test_results": test_results,
        })

    aggregate = {
        "config_path": str(config_path),
        "seeds": seeds,
        "seed_results": seed_results,
        "test_summary": summarize(seed_results),
    }

    if args.summary_path:
        summary_path = Path(args.summary_path)
    else:
        seed_label = "_".join(str(seed) for seed in seeds)
        summary_path = Path(run_config["tb"]["dir_full"]) / f"{base_tb_name}-seeds" / f"test_summary_seeds_{seed_label}.json"

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(aggregate, indent=2))
    print(f"Wrote summary to {summary_path}", flush=True)

    print("Test mean/std:", flush=True)
    for key, values in aggregate["test_summary"].items():
        print(f"  {key}: mean={values['mean']:.6f}, std={values['std']:.6f}, n={values['n']}", flush=True)


if __name__ == "__main__":
    main()
