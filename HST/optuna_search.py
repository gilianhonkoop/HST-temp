import copy
import os
import re
import tempfile
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import optuna
from optuna.trial import TrialState
from train import evaluate_checkpoint, main as train_main, load_data
import torch

@hydra.main(config_path="configs", config_name="hparams_search.yaml", version_base=None)
def run_optuna_search(cfg: DictConfig) -> float:
    base_use_wandb = bool(OmegaConf.select(cfg, "logging.use_wandb", default=False))

    def _select_hydra(path: str, default=None):
        value = OmegaConf.select(cfg, f"hydra.{path}", default=None)
        if value is not None:
            return value
        try:
            hydra_cfg = OmegaConf.create(OmegaConf.to_container(HydraConfig.get(), resolve=False))
            return OmegaConf.select(hydra_cfg, path, default=default)
        except ValueError:
            return default

    # Correctly extract the search space for Optuna.
    base_params = _select_hydra("sweeper.params", default={})
    if isinstance(base_params, DictConfig):
        base_params = OmegaConf.to_container(base_params, resolve=True)
    if not isinstance(base_params, dict):
        base_params = {}
    if not base_params:
        raise ValueError(
            "No Optuna search parameters were found. Expected hydra.sweeper.params in the Hydra runtime config."
        )

    # Create a plain dict for non-OmegaConf APIs like load_data.
    base_cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(base_cfg_dict, dict):
        raise ValueError("Resolved config is not a dict")
    device = base_cfg_dict.get("train", {}).get("device", "cuda") if torch.cuda.is_available() else "cpu"
    preloaded_data = load_data(base_cfg_dict, device)

    class DummyArgs:
        def __init__(self, config_path: str):
            self.config = config_path
            self.resume = False

    def _parse_atom(token: str):
        token = token.strip()
        if re.fullmatch(r"[-+]?\d+", token):
            return int(token)
        if re.fullmatch(r"[-+]?\d*\.\d+(e[-+]?\d+)?", token, re.IGNORECASE):
            return float(token)
        if re.fullmatch(r"[-+]?\d+e[-+]?\d+", token, re.IGNORECASE):
            return float(token)
        return token

    def _parse_range(spec: str):
        inner = spec[spec.find("(") + 1: spec.rfind(")")]
        parts = [p for p in re.split(r"[,\s]+", inner) if p]
        if len(parts) != 2:
            raise ValueError(f"Invalid range spec: {spec}")
        return float(parts[0]), float(parts[1])

    def _parse_choice(spec: str):
        inner = spec[spec.find("(") + 1: spec.rfind(")")]
        parts = [p for p in re.split(r"[,\s]+", inner) if p]
        return [_parse_atom(p) for p in parts]

    def _suggest_value(trial: optuna.Trial, name: str, spec: str):
        spec = spec.strip()
        if spec.startswith("tag("):
            inner = spec[len("tag("):-1]
            tag, rest = [p.strip() for p in inner.split(",", 1)]
            if tag == "log":
                low, high = _parse_range(rest)
                return trial.suggest_float(name, low, high, log=True)
            return _suggest_value(trial, name, rest)
        if spec.startswith("float("):
            low, high = _parse_range(spec)
            return trial.suggest_float(name, low, high)
        if spec.startswith("int("):
            low, high = _parse_range(spec)
            return trial.suggest_int(name, int(low), int(high))
        if spec.startswith("choice("):
            return trial.suggest_categorical(name, _parse_choice(spec))
        raise ValueError(f"Unsupported sweeper spec: {spec}")

    def _run_trial(trial: optuna.Trial) -> float:
        trial_cfg = copy.deepcopy(cfg)
        params = base_params
        sampled_params: dict[str, object] = {}
        OmegaConf.set_struct(trial_cfg, False)
        for key, spec in params.items():
            key_str = str(key)
            value = _suggest_value(trial, key_str, str(spec))
            OmegaConf.update(trial_cfg, key_str, value, merge=True)
            sampled_params[key_str] = value
        print(f"[Optuna trial {trial.number}] params: {sampled_params if sampled_params else 'No params sampled.'}")
        # print(f"[Optuna trial {trial.number}] final train.pos_weight: {OmegaConf.select(trial_cfg, 'train.pos_weight', default=None)}")

        OmegaConf.set_struct(trial_cfg, False)
        OmegaConf.update(trial_cfg, "train.run_threshold_search", False, merge=True)
        OmegaConf.update(trial_cfg, "train.run_test", False, merge=True)
        OmegaConf.update(trial_cfg, "logging.use_wandb", False, merge=True)
        OmegaConf.set_struct(trial_cfg, True)

        base_run_name = OmegaConf.select(trial_cfg, "logging.run_name", default="optuna")
        OmegaConf.update(trial_cfg, "logging.run_name", f"{base_run_name}_trial_{trial.number}", merge=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            import yaml

            yaml.safe_dump(OmegaConf.to_container(trial_cfg, resolve=True), tmp, sort_keys=False)
            temp_path = tmp.name

        try:
            args = DummyArgs(temp_path)
            best_metric_value, num_params = train_main(args, preloaded_data=preloaded_data)
        finally:
            os.remove(temp_path)

        trial_run_name = OmegaConf.select(trial_cfg, "logging.run_name", default=f"optuna_trial_{trial.number}")
        trial_save_dir = OmegaConf.select(trial_cfg, "logging.save_dir", default=".")
        trial_run_dir = os.path.join(str(trial_save_dir), str(trial_run_name))
        checkpoint_path = os.path.join(trial_run_dir, "best_checkpoint.pth")

        # Penalize the objective with the number of parameters
        param_penalty = OmegaConf.select(trial_cfg, "optim.param_penalty", default=0.0)
        penalized_objective = float(best_metric_value) - (float(param_penalty) * num_params)
        
        trial.set_user_attr("best_metric_value", best_metric_value)
        trial.set_user_attr("num_params", num_params)
        trial.set_user_attr("checkpoint_path", checkpoint_path)
        trial.set_user_attr("run_dir", trial_run_dir)
        trial.set_user_attr("run_name", trial_run_name)

        return penalized_objective

    sweeper_cfg = _select_hydra("sweeper", default={})
    if isinstance(sweeper_cfg, DictConfig):
        sweeper_cfg = OmegaConf.to_container(sweeper_cfg, resolve=True)
    if not isinstance(sweeper_cfg, dict):
        sweeper_cfg = {}
    direction = str(sweeper_cfg.get("direction", "maximize"))
    study_name = str(cfg.get("study_name", "hst_hparam_search"))
    storage = sweeper_cfg.get("storage", None)
    n_trials = int(sweeper_cfg.get("n_trials", 10))
    n_jobs = int(sweeper_cfg.get("n_jobs", 1))

    study = optuna.create_study(direction=direction, study_name=study_name, storage=storage, load_if_exists=True)
    study.optimize(_run_trial, n_trials=n_trials, n_jobs=n_jobs)

    top_k_wandb_runs = int(OmegaConf.select(cfg, "top_k_wandb_runs", default=1))
    completed_trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE and trial.value is not None]
    if not completed_trials:
        raise ValueError("Optuna completed without any successful trials to evaluate from checkpoints.")

    completed_trials_with_values = [(trial, float(trial.value)) for trial in completed_trials if trial.value is not None]

    reverse = direction.lower() == "maximize"
    ranked_trials = sorted(completed_trials_with_values, key=lambda item: item[1], reverse=reverse)
    selected_trials = ranked_trials[:max(top_k_wandb_runs, 0)]

    best_metric_value = None
    for rank, (selected_trial, selected_value) in enumerate(selected_trials, start=1):
        eval_cfg = copy.deepcopy(cfg)
        OmegaConf.set_struct(eval_cfg, False)
        for key in base_params:
            key_str = str(key)
            if key_str in selected_trial.params:
                OmegaConf.update(eval_cfg, key_str, selected_trial.params[key_str], merge=True)

        trial_run_name = selected_trial.user_attrs.get("run_name")
        if trial_run_name is None:
            base_run_name = OmegaConf.select(eval_cfg, "logging.run_name", default="optuna")
            trial_run_name = f"{base_run_name}_trial_{selected_trial.number}"

        checkpoint_path = selected_trial.user_attrs.get("checkpoint_path")
        if checkpoint_path is None:
            save_dir = OmegaConf.select(eval_cfg, "logging.save_dir", default=".")
            checkpoint_path = os.path.join(str(save_dir), str(trial_run_name), "best_checkpoint.pth")

        OmegaConf.update(eval_cfg, "train.checkpoint_path", str(checkpoint_path), merge=True)
        OmegaConf.update(eval_cfg, "train.run_threshold_search", True, merge=True)
        OmegaConf.update(eval_cfg, "train.run_test", True, merge=True)
        OmegaConf.update(eval_cfg, "logging.use_wandb", base_use_wandb, merge=True)

        base_run_name = OmegaConf.select(eval_cfg, "logging.run_name", default="optuna")
        OmegaConf.update(
            eval_cfg,
            "logging.run_name",
            f"{base_run_name}_top_{rank}_trial_{selected_trial.number}",
            merge=True,
        )
        OmegaConf.set_struct(eval_cfg, True)

        print(
            f"[Optuna checkpoint eval {rank}/{len(selected_trials)}] trial={selected_trial.number} "
            f"objective={selected_value:.6f} checkpoint={checkpoint_path} params={selected_trial.params}"
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            import yaml

            yaml.safe_dump(OmegaConf.to_container(eval_cfg, resolve=True), tmp, sort_keys=False)
            temp_path = tmp.name

        try:
            args = DummyArgs(temp_path)
            eval_metric_value, _ = evaluate_checkpoint(
                args,
                preloaded_data=preloaded_data,
                checkpoint_path=str(checkpoint_path),
            )
        finally:
            os.remove(temp_path)

        if rank == 1:
            best_metric_value = float(eval_metric_value)

    if best_metric_value is None:
        best_metric_value = float(study.best_trial.user_attrs.get("best_metric_value", study.best_value))

    return float(best_metric_value)

if __name__ == "__main__":
    run_optuna_search()
