import argparse
import copy
import os
import tempfile
from typing import Any

import optuna
import yaml

from datasets import load_aml_dataset
from train import train_main


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def parse_args():
    parser = argparse.ArgumentParser(description="Optuna search for PNA baseline")
    parser.add_argument("--config", type=str, required=True, help="Base training config")
    parser.add_argument("--search-config", type=str, required=True, help="Optuna search config")
    return parser.parse_args()


def suggest_value(trial: optuna.Trial, name: str, spec: dict) -> Any:
    kind = str(spec.get("type", "float")).lower()
    if kind == "float":
        return trial.suggest_float(
            name,
            float(spec["low"]),
            float(spec["high"]),
            log=bool(spec.get("log", False)),
        )
    if kind == "int":
        return trial.suggest_int(
            name,
            int(spec["low"]),
            int(spec["high"]),
            step=int(spec.get("step", 1)),
            log=bool(spec.get("log", False)),
        )
    if kind in {"categorical", "choice"}:
        return trial.suggest_categorical(name, list(spec["choices"]))
    raise ValueError(f"Unsupported search space type for {name}: {kind}")


def set_dotted(cfg: dict, dotted_key: str, value: Any) -> None:
    cur = cfg
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def main():
    args = parse_args()
    base_cfg = load_yaml(args.config)
    search_cfg = load_yaml(args.search_config)
    search_space = search_cfg.get("search_space", {})
    if not search_space:
        raise ValueError("search_config must contain a non-empty search_space mapping")

    study_cfg = search_cfg.get("study", {})
    study = optuna.create_study(
        study_name=str(study_cfg.get("name", "pna_search")),
        storage=study_cfg.get("storage", None),
        direction=str(study_cfg.get("direction", "maximize")),
        load_if_exists=True,
    )
    n_trials = int(study_cfg.get("n_trials", 20))
    n_jobs = int(study_cfg.get("n_jobs", 1))
    objective_metric = str(study_cfg.get("objective_metric", "best_metric"))
    param_penalty = float(study_cfg.get("param_penalty", 0.0))
    preloaded_data = load_aml_dataset(base_cfg.get("dataset", {})) if bool(study_cfg.get("preload_data", True)) else None

    class Args:
        def __init__(self, config_path: str):
            self.config = config_path
            self.resume = False

    def objective(trial: optuna.Trial) -> float:
        cfg = copy.deepcopy(base_cfg)
        sampled = {}
        for key, spec in search_space.items():
            value = suggest_value(trial, key, spec)
            set_dotted(cfg, key, value)
            sampled[key] = value

        run_base = str(cfg.get("logging", {}).get("run_name", "pna_optuna"))
        cfg.setdefault("logging", {})["run_name"] = f"{run_base}_trial_{trial.number}"
        cfg["logging"]["use_wandb"] = False
        cfg.setdefault("train", {})["run_test"] = True

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.safe_dump(cfg, tmp, sort_keys=False)
            tmp_path = tmp.name
        try:
            best_metric, num_params = train_main(Args(tmp_path), preloaded_data=preloaded_data)
        finally:
            os.remove(tmp_path)

        trial.set_user_attr("sampled_params", sampled)
        trial.set_user_attr("num_params", int(num_params))
        trial.set_user_attr("run_name", cfg["logging"]["run_name"])
        trial.set_user_attr(
            "checkpoint_path",
            os.path.join(str(cfg["logging"].get("save_dir", "outputs")), cfg["logging"]["run_name"], "best_checkpoint.pth"),
        )
        if objective_metric == "best_metric":
            value = float(best_metric)
        else:
            value = float(best_metric)
        return value - param_penalty * int(num_params)

    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    print("Best trial:", study.best_trial.number)
    print("Best value:", study.best_value)
    print("Best params:", study.best_trial.params)
    print("Best checkpoint:", study.best_trial.user_attrs.get("checkpoint_path"))


if __name__ == "__main__":
    main()
