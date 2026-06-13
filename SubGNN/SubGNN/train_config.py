# General
import numpy as np
import random
import argparse
import tqdm
import pickle
import json
import commentjson
import joblib
import os
import sys
import pathlib
import contextlib
from collections import OrderedDict
import random
import string

# Pytorch
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.profiler import AdvancedProfiler

# Optuna
import optuna
from optuna.samplers import TPESampler
from optuna.integration import PyTorchLightningPruningCallback

# Our Methods
import SubGNN as md
sys.path.insert(0, '..') # add config to path
import config
import subgraph_utils
import wandb
from dotenv import load_dotenv


def parse_arguments():
    '''
    Read in the config file specifying all of the parameters
    '''
    parser = argparse.ArgumentParser(description="Learn subgraph embeddings")
    parser.add_argument('-config_path', type=str, default=None, help='Load config file')
    args = parser.parse_args()
    return args

def read_json(fname):
    '''
    Read in the json file specified by 'fname'
    '''
    with open(fname, 'rt') as handle:
        return commentjson.load(handle, object_hook=OrderedDict)

def _json_default(value):
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    return str(value)

def print_config_block(title, payload):
    print(f"\n===== {title} =====", flush=True)
    print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), flush=True)
    print(f"===== END {title} =====\n", flush=True)

def _wandb_incremental_enabled(run_config):
    wandb_config = run_config.get('wandb', {})
    if not wandb_config.get('wandb_logging', False):
        return False
    if 'incremental_logging' in wandb_config:
        return bool(wandb_config['incremental_logging'])
    return not bool(wandb_config.get('log_best_only', False))

def get_optuna_suggest(param_dict, name, trial):
    '''
    Returns a suggested value for the hyperparameter specified by 'name' from the range of values in 'param_dict'

    name: string specifying hyperparameter
    trial: optuna trial
    param_dict: dictionary containing information about the hyperparameter (range of values & type of sampler)
            e.g.{
                    "type" : "suggest_categorical",
                    "args" : [[ 64, 128]]
                }
    '''
    module_name = param_dict['type'] # e.g. suggest_categorical, suggest_float
    args = [name]
    args.extend(param_dict['args']) # resulting list will look something like this ['batch_size', [ 64, 128]]
    if "kwargs" in param_dict:
        kwargs = dict(param_dict["kwargs"])
        return getattr(trial, module_name)(*args, **kwargs) 
    else:
        return getattr(trial, module_name)(*args)

def get_hyperparams_optuna(run_config, trial):
    '''
    Converts the fixed and variable hyperparameters in the run config to a dictionary of the final hyperparameters

    Returns: hyp_fix - dictionary where key is the hyperparameter name (e.g. batch_size) and value is the hyperparameter value
    '''
    #initialize the dict with the fixed hyperparameters
    hyp_fix = dict(run_config["hyperparams_fix"])

    # update the dict with variable value hyperparameters by sampling a hyperparameter value from the range specified in the run_config
    hyp_optuna = {k:get_optuna_suggest(run_config["hyperparams_optuna"][k], k, trial) for k in dict(run_config["hyperparams_optuna"]).keys()}
    hyp_fix.update(hyp_optuna)
    return hyp_fix

def build_model(run_config, trial = None):
    '''
    Creates SubGNN from the hyperparameters specified in the run config
    '''
    # get hyperparameters for the current trial
    hyperparameters = get_hyperparams_optuna(run_config, trial)

    # Set seeds for reproducibility
    torch.manual_seed(hyperparameters['seed'])
    np.random.seed(hyperparameters['seed'])
    torch.cuda.manual_seed(hyperparameters['seed'])
    torch.cuda.manual_seed_all(hyperparameters['seed']) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # initialize SubGNN
    model = md.SubGNN(hyperparameters, run_config["graph_path"], \
        run_config["subgraphs_path"], run_config["embedding_path"], \
        run_config["similarities_path"], run_config["shortest_paths_path"], run_config['degree_sequence_path'], run_config['ego_graph_path'])
    return model, hyperparameters

def build_trainer(run_config, hyperparameters, trial = None):
    '''
    Set up optuna trainer
    '''

    if 'progress_bar_refresh_rate' in hyperparameters:
        p_refresh = hyperparameters['progress_bar_refresh_rate']
    else:
        p_refresh = 5

    # set epochs, gpus, gradient clipping, etc. 
    # if 'no_gpu' in run config, then use CPU
    trainer_kwargs={'max_epochs': hyperparameters['max_epochs'],
                    "gpus": 0 if 'no_gpu' in run_config else 1,
                    "num_sanity_val_steps":0,
                    "progress_bar_refresh_rate":p_refresh,
                    "gradient_clip_val": hyperparameters['grad_clip']
                    }

    # set auto learning rate finder param
    if 'auto_lr_find' in hyperparameters and hyperparameters['auto_lr_find']:
        trainer_kwargs['auto_lr_find'] = hyperparameters['auto_lr_find']

    # Create wandb logger
    lgdir = os.path.join(run_config['tb']['dir_full'], run_config['tb']['name'])
    os.makedirs(lgdir, exist_ok=True)
    version = "version_" + str(random.randint(0, 10000000))
    results_path = os.path.join(lgdir, version)
    os.makedirs(results_path, exist_ok=True)

    if _wandb_incremental_enabled(run_config):
        load_dotenv(config.PROJECT_ROOT / ".env")
        api_key = os.getenv("WANDB_API_KEY")
        if not api_key:
            raise ValueError("WANDB_API_KEY environment variable is not set. Add it to .env or set it in your SLURM script.")
        wandb.login(key=api_key)

        if run_config.get('wandb', {}).get('epoch_logging', True):
            base_name = run_config['wandb'].get('name', None)
            if trial is not None and base_name:
                run_name = f"{base_name}-trial{trial.number}"
            else:
                run_name = base_name
            wandb.init(
                project=run_config['wandb'].get('project', 'ml-detection'),
                name=run_name,
                dir=run_config['tb']['dir_full'],
                reinit=True,
            )
            print("wandb logging at ", results_path)
            trainer_kwargs["logger"] = False
        else:
            logger = WandbLogger(
                project=run_config['wandb'].get('project', 'ml-detection'),
                save_dir=run_config['tb']['dir_full'],
                name=run_config['wandb'].get('name', None)
            )
            print("wandb logging at ", results_path)
            trainer_kwargs["logger"] = logger


    # Save top three model checkpoints
    trainer_kwargs["checkpoint_callback"] = ModelCheckpoint(
        filepath= os.path.join(results_path, "{epoch}-{val_micro_f1:.2f}-{val_acc:.2f}-{val_auroc:.2f}"),
        save_top_k = 3,
        verbose=True,
        monitor=run_config['optuna']['monitor_metric'],
        mode='max'
        )

    # if we use pruning, use the pytorch lightning pruning callback
    if run_config["optuna"]['pruning']:
        trainer_kwargs['early_stop_callback'] = PyTorchLightningPruningCallback(trial, monitor=run_config['optuna']['monitor_metric'])

    if run_config.get('early_stopping', {}).get('enabled', True):
        monitor_metric = run_config['optuna']['monitor_metric']
        mode = 'min' if run_config['optuna']['opt_direction'] == 'minimize' else 'max'
        patience = run_config.get('early_stopping', {}).get('patience', 10)
        min_delta = run_config.get('early_stopping', {}).get('min_delta', 0.0)
        trainer_kwargs['early_stop_callback'] = EarlyStopping(
            monitor=monitor_metric,
            mode=mode,
            patience=patience,
            min_delta=min_delta,
        )

    trainer = pl.Trainer(**trainer_kwargs)
    
    return trainer, trainer_kwargs, results_path

def _concat_valid_outputs(valid_outputs):
    logits = torch.cat([out["logits"] for out in valid_outputs], dim=0)
    labels = torch.cat([out["labels"] for out in valid_outputs], dim=0)
    return logits, labels

def _move_batch_to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: _move_batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [_move_batch_to_device(v, device) for v in batch]
    if isinstance(batch, tuple):
        return tuple(_move_batch_to_device(v, device) for v in batch)
    return batch

def _collect_validation_outputs_for_threshold(model):
    """Collect validation logits/labels without using Trainer.validate (not available in old Lightning)."""
    was_training = model.training
    device = next(model.parameters()).device
    model.eval()

    valid_outputs = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(model.val_dataloader()):
            batch = _move_batch_to_device(batch, device)
            out = model.val_test_step(batch, batch_idx, is_test=False)
            valid_outputs.append({
                "logits": out["val_logits"].detach().cpu(),
                "labels": out["val_labels"].detach().cpu(),
            })

    if was_training:
        model.train()
    return valid_outputs

def _score_threshold(logits, labels, threshold: float, metric_name: str) -> float:
    metric = metric_name.lower()
    if metric in {"minor_f1", "illicit_f1", "val_minor_f1", "val_illicit_f1"}:
        return float(subgraph_utils.calc_minor_f1(logits, labels, threshold=threshold).squeeze().cpu().item())
    if metric in {"macro_f1", "val_macro_f1"}:
        return float(subgraph_utils.calc_f1(logits, labels, avg_type="macro", threshold=threshold).squeeze().cpu().item())
    if metric in {"micro_f1", "val_micro_f1"}:
        return float(subgraph_utils.calc_f1(logits, labels, avg_type="micro", threshold=threshold).squeeze().cpu().item())
    if metric in {"acc", "accuracy", "val_acc"}:
        return float(subgraph_utils.calc_accuracy(logits, labels, threshold=threshold).squeeze().cpu().item())
    raise ValueError(f"Unsupported threshold search metric: {metric_name}")

def _run_threshold_search(logits, labels, n_trials: int, lower: float, upper: float, metric_name: str):
    import optuna

    study = optuna.create_study(direction="maximize")

    def objective(trial):
        threshold = trial.suggest_float("threshold", lower, upper)
        return _score_threshold(logits, labels, threshold, metric_name)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_threshold = float(study.best_params["threshold"])
    best_value = float(study.best_value)
    return best_threshold, best_value

def _progress_output_context(run_config):
    if run_config.get("progress_to_stderr", True):
        return contextlib.redirect_stdout(sys.stderr)
    return contextlib.nullcontext()

def _metrics_to_float_dict(metrics):
    serializable = {}
    for key, value in metrics.items():
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "item"):
            value = value.item()
        serializable[key] = float(value)
    return serializable

def _resolve_best_checkpoint_path(trainer, results_path: str, metric_scores, monitor_metric: str, direction: str) -> str:
    """Resolve the checkpoint selected by the validation monitor across Lightning versions."""
    callback = getattr(trainer, "checkpoint_callback", None)
    if callback is not None:
        best_model_path = getattr(callback, "best_model_path", None)
        if best_model_path and os.path.exists(best_model_path):
            return best_model_path

        best_k_models = getattr(callback, "best_k_models", None)
        if isinstance(best_k_models, dict) and best_k_models:
            try:
                scored_paths = []
                for path, score in best_k_models.items():
                    if path and os.path.exists(path):
                        if hasattr(score, "detach"):
                            score = float(score.detach().cpu().item())
                        else:
                            score = float(score)
                        scored_paths.append((path, score))
                if scored_paths:
                    reverse = direction == "maximize"
                    return sorted(scored_paths, key=lambda x: x[1], reverse=reverse)[0][0]
            except Exception:
                pass

    if metric_scores:
        scores = [float(score[monitor_metric]) for score in metric_scores if monitor_metric in score]
        if scores:
            best_idx = int(np.argmax(scores) if direction == "maximize" else np.argmin(scores))
            for fname in os.listdir(results_path):
                if fname.startswith(f"epoch={best_idx}-") and fname.endswith(".ckpt"):
                    return os.path.join(results_path, fname)

    ckpt_paths = [os.path.join(results_path, fname) for fname in os.listdir(results_path) if fname.endswith(".ckpt")]
    if ckpt_paths:
        return max(ckpt_paths, key=os.path.getmtime)
    return ""

def train_model(run_config, trial = None):
    '''
    Train a single model whose hyperparameters are specified in the run config
    
    Returns the max (or min) metric specified by 'monitor_metric' in the run config
    '''

    # get model and hyperparameter dict
    model, hyperparameters = build_model(run_config, trial)
    trial_label = f"TRIAL {trial.number} FINAL HYPERPARAMETERS" if trial is not None else "FINAL HYPERPARAMETERS"
    print_config_block(trial_label, hyperparameters)
    n_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Learnable parameters: {n_learnable_params}", flush=True)

    # build optuna trainer
    trainer, trainer_kwargs, results_path = build_trainer(run_config, hyperparameters, trial)

    if wandb.run is not None:
        params_payload = {}
        for k, v in hyperparameters.items():
            if isinstance(v, (int, float, bool, str)):
                params_payload[f"params/{k}"] = v
            else:
                params_payload[f"params/{k}"] = json.dumps(v)
        params_payload["params/learnable_parameters"] = n_learnable_params
        if params_payload:
            wandb.log(params_payload, step=0)

    # dump hyperparameters to results dir
    hparam_file = open(os.path.join(results_path, "hyperparams.json"),"w")
    hparam_file.write(json.dumps(hyperparameters, indent=4))
    hparam_file.close()
    
    # dump trainer args to results dir
    tkwarg_file = open(os.path.join(results_path, "trainer_kwargs.json"),"w")
    pop_keys = [key for key in ['logger','profiler','early_stop_callback','checkpoint_callback'] if key in trainer_kwargs.keys()]
    [trainer_kwargs.pop(key) for key in pop_keys]
    tkwarg_file.write(json.dumps(trainer_kwargs, indent=4))
    tkwarg_file.close()

    # train the model
    with _progress_output_context(run_config):
        trainer.fit(model)

    # optionally run test
    test_results = None
    if run_config.get('run_test', False):
        best_model_path = _resolve_best_checkpoint_path(
            trainer,
            results_path,
            getattr(model, "metric_scores", []),
            run_config['optuna']['monitor_metric'],
            run_config['optuna']['opt_direction'],
        )
        try:
            if best_model_path and os.path.exists(best_model_path):
                print(f"Loading best model from {best_model_path}")
                checkpoint = torch.load(best_model_path, map_location=lambda storage, loc: storage)
                if "best_val_threshold" in checkpoint:
                    best_val_threshold = checkpoint["best_val_threshold"]
                    if hasattr(best_val_threshold, "item"):
                        best_val_threshold = best_val_threshold.item()
                    model.best_val_threshold = float(best_val_threshold)
                model.load_state_dict(checkpoint['state_dict'])
            else:
                print("No checkpoint found; testing current in-memory model.")
        except Exception as e:
            print(f"Could not load best checkpoint: {e}")

        threshold_cfg = run_config.get("threshold_search", {})
        if threshold_cfg.get("enabled", False) and best_model_path:
            model._valid_outputs = _collect_validation_outputs_for_threshold(model)
            if model._valid_outputs:
                val_logits, val_labels = _concat_valid_outputs(model._valid_outputs)
                best_threshold, best_value = _run_threshold_search(
                    val_logits,
                    val_labels,
                    n_trials=int(threshold_cfg.get("n_trials", 40)),
                    lower=float(threshold_cfg.get("lower", 0.05)),
                    upper=float(threshold_cfg.get("upper", 0.95)),
                    metric_name=str(threshold_cfg.get("metric", "minor_f1")),
                )
                model.best_val_threshold = best_threshold
                model.threshold_search_results = {
                    "threshold_search_metric": str(threshold_cfg.get("metric", "minor_f1")),
                    "selected_threshold": best_threshold,
                    "best_val_threshold": best_threshold,
                    "best_val_threshold_score": best_value,
                }
                print(
                    f"Selected threshold {best_threshold:.6f} on validation "
                    f"{threshold_cfg.get('metric', 'minor_f1')}={best_value:.6f}"
                )
                if wandb.run is not None:
                    wandb.log({
                        "val/selected_threshold": best_threshold,
                        "val/best_threshold": best_threshold,
                        f"val/{threshold_cfg.get('metric', 'minor_f1')}": best_value,
                        "params/best_val_threshold": best_threshold,
                        "params/best_threshold": best_threshold,
                    })

        with _progress_output_context(run_config):
            test_output = trainer.test(model)
        if isinstance(test_output, list) and len(test_output) > 0 and isinstance(test_output[0], dict):
            if isinstance(test_output[0].get('log'), dict):
                test_results = test_output[0]['log']
            else:
                test_results = test_output[0]
        if getattr(model, "test_results", None):
            test_results = model.test_results
        test_results_serializable = _metrics_to_float_dict(test_results) if test_results is not None else None
        if test_results_serializable is not None:
            print("TEST RESULTS JSON:")
            print(json.dumps(test_results_serializable, indent=2, sort_keys=True), flush=True)
        if trial is not None and test_results is not None:
            trial.set_user_attr("test_results", test_results_serializable)

    if trial is not None:
        trial.set_user_attr("results_path", results_path)

    if (_wandb_incremental_enabled(run_config)
        and run_config.get('wandb', {}).get('epoch_logging', True)
        and wandb.run is not None):
        wandb.finish()
        
    # write results to the results dir
    if results_path is not None:
        hparam_file = open(os.path.join(results_path, "final_metric_scores.json"),"w")
        results_serializable = {k:float(v) for k,v in model.metric_scores[-1].items()}
        hparam_file.write(json.dumps(results_serializable, indent=4))
        hparam_file.close()
        if getattr(model, "metric_scores", None):
            metrics_history = []
            for entry in model.metric_scores:
                metrics_history.append({k: float(v) for k, v in entry.items()})
            with open(os.path.join(results_path, "val_metrics_history.json"), "w") as mh_file:
                mh_file.write(json.dumps(metrics_history, indent=2))
        if getattr(model, "train_metric_scores", None):
            train_history = []
            for entry in model.train_metric_scores:
                train_history.append(_metrics_to_float_dict(entry))
            with open(os.path.join(results_path, "train_metrics_history.json"), "w") as th_file:
                th_file.write(json.dumps(train_history, indent=2))
        if test_results is not None:
            with open(os.path.join(results_path, "test_results.json"), "w") as tr_file:
                tr_file.write(json.dumps(_metrics_to_float_dict(test_results), indent=2))
        if getattr(model, "threshold_search_results", None):
            with open(os.path.join(results_path, "threshold_search_results.json"), "w") as ts_file:
                ts_file.write(json.dumps(model.threshold_search_results, indent=2))
    
    # return the max (or min) metric specified by 'monitor_metric' in the run config
    all_scores = [float(score[run_config['optuna']['monitor_metric']]) for score in model.metric_scores]
    if run_config['optuna']['opt_direction'] == "maximize":
        return(np.max(all_scores))
    else:
        return(np.min(all_scores))

def main():
    '''
    Perform an optuna run according to the hyperparameters and directory locations specified in 'config_path'
    '''
    torch.autograd.set_detect_anomaly(True)
    args = parse_arguments()

    # read in config file
    run_config = read_json(args.config_path)

    data_format = run_config.get('data', {}).get('dataset_type') or run_config.get('data', {}).get('data_format')
    if data_format and 'data_format' not in run_config['hyperparams_fix']:
        run_config['hyperparams_fix']['data_format'] = data_format
    for key in ['aml_split', 'aml_train_ratio', 'aml_val_ratio']:
        if key in run_config.get('data', {}) and key not in run_config['hyperparams_fix']:
            run_config['hyperparams_fix'][key] = run_config['data'][key]
    if 'pred_threshold' not in run_config['hyperparams_fix']:
        threshold = run_config.get('threshold')
        if threshold is None:
            threshold = run_config.get('data', {}).get('threshold')
        if threshold is not None:
            run_config['hyperparams_fix']['pred_threshold'] = threshold
        else:
            run_config['hyperparams_fix']['pred_threshold'] = 0.5
    if 'wandb' in run_config and 'epoch_logging' in run_config['wandb'] and 'wandb_epoch_logging' not in run_config['hyperparams_fix']:
        run_config['hyperparams_fix']['wandb_epoch_logging'] = run_config['wandb']['epoch_logging']

    ## Set paths to data
    task = run_config['data']['task']
    data_dir = run_config['data'].get('data_dir')
    data_root = data_dir if data_dir else task
    embedding_type = run_config['hyperparams_fix']['embedding_type']
    
    # paths to subgraphs, edge list, and shortest paths between all nodes in the graph
    run_config["subgraphs_path"] = os.path.join(data_root, "subgraphs.pth")
    run_config["graph_path"] = os.path.join(data_root, "edge_list.txt")
    run_config['shortest_paths_path'] = os.path.join(data_root, "shortest_path_matrix.npy")
    run_config['degree_sequence_path'] = os.path.join(data_root, "degree_sequence.txt")
    run_config['ego_graph_path'] = os.path.join(data_root, "ego_graphs.txt")

    #directory where similarity calculations will be stored
    run_config["similarities_path"] = os.path.join(data_root, "similarities/")

    # get location of node embeddings
    if embedding_type == 'gin':
        run_config["embedding_path"] = os.path.join(data_root, "gin_embeddings.pth")
    elif embedding_type == 'graphsaint':
        run_config["embedding_path"] = os.path.join(data_root, "graphsaint_gcn_embeddings.pth")
    else:
        raise NotImplementedError

    global_node_embedding_path = run_config['data'].get('global_node_embedding_path')
    if global_node_embedding_path is not None:
        run_config['hyperparams_fix']['global_node_embedding_path'] = global_node_embedding_path
        run_config["embedding_path"] = global_node_embedding_path
    
    # create a tensorboard directory in the folder specified by dir in the PROJECT ROOT folder
    if 'local' in run_config['tb'] and run_config['tb']['local']:
        run_config['tb']['dir_full'] = run_config['tb']['dir']
    else:
        run_config['tb']['dir_full'] = os.path.join(config.PROJECT_ROOT, run_config['tb']['dir'])
    ntrials = run_config['optuna']['opt_n_trials']
    print(f'Running {ntrials} Trials of optuna')
    print_config_block("RESOLVED RUN CONFIG", run_config)

    if run_config['optuna']['pruning']:
        pruner = optuna.pruners.MedianPruner()
    else:
        pruner = None

    # the complete study path is the tensorboard directory + the study name
    run_config['study_path'] = os.path.join(run_config['tb']['dir_full'], run_config['tb']['name'])
    print("Logging to ", run_config['study_path'])
    pathlib.Path(run_config['study_path']).mkdir(parents=True, exist_ok=True)

    # get database file
    db_file = os.path.join(run_config['study_path'], 'optuna_study_sqlite.db')

    # specify sampler
    if run_config['optuna']['sampler'] == "grid" and "grid_search_space" in run_config['optuna']:
        sampler = optuna.samplers.GridSampler(run_config['optuna']['grid_search_space'])
    elif run_config['optuna']['sampler'] == "tpe":
        sampler = optuna.samplers.TPESampler()
    elif run_config['optuna']['sampler'] == "random":
        sampler = optuna.samplers.RandomSampler()
    
    # create an optuna study with the specified sampler, pruner, direction (e.g. maximize)
    # A SQLlite database is used to keep track of results
    # Will load in existing study if one exists
    study = optuna.create_study(direction=run_config['optuna']['opt_direction'],
                                sampler=sampler,
                                pruner=pruner,
                                storage= 'sqlite:///' + db_file,
                                study_name=run_config['study_path'],
                                load_if_exists=True)
    
    study.optimize(lambda trial: train_model(run_config, trial), n_trials=run_config['optuna']['opt_n_trials'], n_jobs =run_config['optuna']['opt_n_cores'])

    if (run_config.get('wandb', {}).get('log_best_only', False)
        and not _wandb_incremental_enabled(run_config)):
        load_dotenv(config.PROJECT_ROOT / ".env")
        api_key = os.getenv("WANDB_API_KEY")
        if not api_key:
            raise ValueError("WANDB_API_KEY environment variable is not set. Add it to .env or set it in your SLURM script.")
        wandb.login(key=api_key)

        def _load_json_if_exists(path):
            if path and os.path.exists(path):
                with open(path, "r") as f:
                    return json.load(f)
            return None

        def _categorize_metric_key(key: str) -> str:
            if key.startswith("train_"):
                return f"train/{key[len('train_'):]}"
            if key.startswith("val_"):
                return f"val/{key[len('val_'):]}"
            if key.startswith("avg_val_"):
                return f"val/avg_{key[len('avg_val_'):]}"
            if key.startswith("test_"):
                return f"test/{key[len('test_'):]}"
            if key.startswith("avg_test_"):
                return f"test/avg_{key[len('avg_test_'):]}"
            return f"params/{key}"

        def _categorize_metrics(metrics_dict):
            if not isinstance(metrics_dict, dict):
                return {}
            categorized = {}
            for k, v in metrics_dict.items():
                categorized[_categorize_metric_key(str(k))] = v
            return categorized

        def _get_trial_test_results(trial):
            test_results = trial.user_attrs.get("test_results")
            results_path = trial.user_attrs.get("results_path")
            if test_results is None and results_path:
                test_results = _load_json_if_exists(os.path.join(results_path, "test_results.json"))
            return test_results

        def _trial_value(trial) -> float:
            if trial.value is None:
                raise ValueError(f"Trial {trial.number} has no objective value.")
            return float(trial.value)

        def _get_completed_trials_sorted():
            complete_trials = [
                trial for trial in study.trials
                if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
            ]
            reverse = run_config['optuna']['opt_direction'] == "maximize"
            return sorted(complete_trials, key=_trial_value, reverse=reverse)

        top_k = int(run_config.get('wandb', {}).get('top_k', 1))
        top_trials = _get_completed_trials_sorted()[:top_k]

        base_name = run_config.get('wandb', {}).get('name', 'subgnn')
        project_name = run_config.get('wandb', {}).get('project', 'ml-detection')
        objective_metric = run_config['optuna']['monitor_metric']

        for rank, trial in enumerate(top_trials, start=1):
            run_name = f"{base_name}-top{rank}-trial{trial.number}"
            run = wandb.init(
                project=project_name,
                name=run_name,
                dir=run_config['tb']['dir_full'],
                reinit=True,
                config={
                    "trial_number": trial.number,
                    "top_rank": rank,
                    "objective_metric": objective_metric,
                    "objective_direction": run_config['optuna']['opt_direction'],
                },
                tags=["top-k", f"top-{rank}", f"trial-{trial.number}"],
            )

            results_path = trial.user_attrs.get("results_path")
            trial_payload = {
                "params/trial_number": trial.number,
                "params/top_rank": rank,
                "params/objective_metric": objective_metric,
                "val/objective_value": _trial_value(trial),
                "params/n_trials_total": len(study.trials),
            }
            if results_path:
                trial_payload["params/results_path"] = results_path

            for param_name, param_value in trial.params.items():
                trial_payload[f"params/{param_name}"] = param_value

            final_val = _load_json_if_exists(os.path.join(results_path, "final_metric_scores.json")) if results_path else None
            if isinstance(final_val, dict):
                trial_payload.update(_categorize_metrics(final_val))

            test_results = _get_trial_test_results(trial)
            if isinstance(test_results, dict):
                trial_payload.update(_categorize_metrics(test_results))

            threshold_results = _load_json_if_exists(os.path.join(results_path, "threshold_search_results.json")) if results_path else None
            if isinstance(threshold_results, dict):
                if "best_val_threshold" in threshold_results:
                    trial_payload["val/selected_threshold"] = threshold_results["best_val_threshold"]
                    trial_payload["val/best_threshold"] = threshold_results["best_val_threshold"]
                    trial_payload["params/best_val_threshold"] = threshold_results["best_val_threshold"]
                    trial_payload["params/best_threshold"] = threshold_results["best_val_threshold"]
                if "best_val_threshold_score" in threshold_results:
                    trial_payload["val/best_threshold_score"] = threshold_results["best_val_threshold_score"]
                if "threshold_search_metric" in threshold_results:
                    trial_payload["params/threshold_search_metric"] = threshold_results["threshold_search_metric"]

            run.summary.update(trial_payload)
            wandb.log(trial_payload, step=0)

            if results_path:
                val_hist = _load_json_if_exists(os.path.join(results_path, "val_metrics_history.json"))
                train_hist = _load_json_if_exists(os.path.join(results_path, "train_metrics_history.json"))
                if isinstance(train_hist, list):
                    for epoch_idx, metrics in enumerate(train_hist):
                        wandb.log(_categorize_metrics(metrics), step=epoch_idx)
                if isinstance(val_hist, list):
                    for epoch_idx, metrics in enumerate(val_hist):
                        wandb.log(_categorize_metrics(metrics), step=epoch_idx)

            wandb.finish()
    
    optuna_results_path = os.path.join(run_config['study_path'], 'optuna_study.pkl')
    print("Saving Study Results to", optuna_results_path)
    joblib.dump(study, optuna_results_path)

    print(study.best_params)
    

if __name__ == "__main__":
    main()
