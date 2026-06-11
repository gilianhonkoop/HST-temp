import os
import time
from typing import List, Optional, Dict, Union, Any
from omegaconf import DictConfig, ListConfig, open_dict

import hydra
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import Logger

from data import SubgraphDataModule
from run_utils import get_logger, log_hyperparameters, finish
from utils import make_deterministic_everything

import torch
import numpy as np
from sklearn.metrics import f1_score

"""Codes are adopted from
    https://github.com/ashleve/lightning-hydra-template/blob/main/src/train.py
    https://github.com/ashleve/lightning-hydra-template/blob/main/run.py"""


log = get_logger(__name__)


def _concat_epoch_outputs(epoch_outputs: List[Dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.cat([out["logits"].detach().cpu() for out in epoch_outputs], dim=0)
    ys = torch.cat([out["ys"].detach().cpu() for out in epoch_outputs], dim=0)
    return logits, ys


def _predict_binary_with_threshold(logits: torch.Tensor, threshold: float) -> torch.Tensor:
    if logits.dim() != 2 or logits.size(-1) != 2:
        raise ValueError("Threshold search currently supports binary two-logit outputs only.")
    probs = torch.softmax(logits, dim=-1)
    return torch.where(
        probs[:, 0] > threshold,
        torch.zeros_like(probs[:, 0], dtype=torch.long),
        torch.ones_like(probs[:, 0], dtype=torch.long),
    )


def _score_threshold(logits: torch.Tensor, ys: torch.Tensor, threshold: float, metric_name: str) -> float:
    preds = _predict_binary_with_threshold(logits, threshold)
    ys_np = ys.cpu().numpy()
    preds_np = preds.cpu().numpy()
    metric_name = metric_name.lower()

    if metric_name in {"illicit_f1", "valid/illicit_f1", "test/illicit_f1", "train/illicit_f1"}:
        return float(f1_score(ys_np, preds_np, pos_label=0, zero_division=0))
    if metric_name in {"licit_f1", "valid/licit_f1", "test/licit_f1", "train/licit_f1"}:
        return float(f1_score(ys_np, preds_np, pos_label=1, zero_division=0))
    if metric_name in {"binary_f1", "valid/binary_f1", "test/binary_f1", "train/binary_f1"}:
        return float(f1_score(ys_np, preds_np, pos_label=1, zero_division=0))

    raise ValueError(f"Unsupported threshold search metric: {metric_name}")


def _run_threshold_search(
    trainer: Trainer,
    model: LightningModule,
    datamodule: LightningDataModule,
    best_ckpt_path: str,
    metric_name: str,
    n_trials: int,
    lower: float,
    upper: float,
) -> tuple[LightningModule, float, float]:
    if not best_ckpt_path:
        raise ValueError("Best checkpoint path is required for threshold search.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log.info("Loading checkpoint for threshold search: %s", best_ckpt_path)
    best_model = model.__class__.load_from_checkpoint(
        best_ckpt_path,
        given_datamodule=datamodule,
        map_location=device,
    )
    best_model.to(device)
    best_model.eval()

    outputs = []
    started = time.time()
    val_loader = datamodule.val_dataloader()
    log.info("Collecting validation logits for threshold search.")
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            batch = batch.to(device) if hasattr(batch, "to") else batch
            step_out = best_model.step(batch=batch, batch_idx=batch_idx)
            outputs.append(best_model._to_epoch_record(step_out))
            log.info(
                "Collected validation logits batch %d (%d examples so far).",
                batch_idx,
                sum(int(out["ys"].numel()) for out in outputs),
            )

    logits, ys = _concat_epoch_outputs(outputs)
    log.info(
        "Collected %d validation predictions in %.1fs. Scanning %d thresholds.",
        int(ys.numel()),
        time.time() - started,
        n_trials,
    )

    thresholds = np.linspace(lower, upper, max(1, int(n_trials)))
    scored = [
        (float(threshold), _score_threshold(logits, ys, float(threshold), metric_name))
        for threshold in thresholds
    ]
    best_threshold, best_value = max(scored, key=lambda item: item[1])
    best_model.hparams.train_val_threshold = best_threshold
    best_model.hparams.test_threshold = best_threshold
    best_model.cpu()
    return best_model, best_threshold, best_value


def train(config: DictConfig, seed_forced: Optional[int] = None) -> Union[Dict[str, Any], float, None]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.
    Args:
        config (DictConfig): Configuration composed by Hydra.
        seed_forced: This value will be replaced for config.seed for multiruns.
    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
        Optional[Dict[str, float]]: Metric score for averaging scores.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        if seed_forced is not None:
            config.seed = seed_forced
        seed_everything(config.seed, workers=True)
        make_deterministic_everything(config.seed)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model, given_datamodule=datamodule)
    # model.hparams.train_val_threshold = 0.85
    # model.hparams.test_threshold = 0.85

    log.info(model)

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                # Lazy load metrics and set the first as monitor.
                if "monitor" in cb_conf and cb_conf.monitor is None:
                    cb_conf.monitor = f"valid/{config.model.metrics[0]}"
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger: List[Logger] = []
    if "logger" in config:
        def _normalize_wandb_logger_conf(lg_conf: DictConfig) -> DictConfig:
            """Flatten optional nested `wandb` overrides into logger kwargs.

            This supports CLI overrides like:
            - logger.wandb.entity=...
            - logger.wandb.project=...
            - logger.wandb.name=...

            while keeping WandbLogger init kwargs valid.
            """
            if "_target_" in lg_conf and str(lg_conf._target_).endswith("WandbLogger"):
                nested = lg_conf.get("wandb")
                if isinstance(nested, DictConfig):
                    with open_dict(lg_conf):
                        for k, v in nested.items():
                            # Only override when value is explicitly provided.
                            if v is not None and not (isinstance(v, str) and v == ""):
                                lg_conf[k] = v
                        # Remove unsupported kwarg before instantiation.
                        if "wandb" in lg_conf:
                            del lg_conf["wandb"]
            return lg_conf

        logger_conf = config.logger
        if isinstance(logger_conf, DictConfig):
            # Single logger config: has keys like _target_, project, entity, ...
            if "_target_" in logger_conf:
                logger_conf = _normalize_wandb_logger_conf(logger_conf)
                log.info(f"Instantiating logger <{logger_conf._target_}>")
                logger.append(hydra.utils.instantiate(logger_conf))
            # Grouped logger configs: logger: {wandb: {...}, csv: {...}}
            else:
                for _, lg_conf in logger_conf.items():
                    if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
                        lg_conf = _normalize_wandb_logger_conf(lg_conf)
                        log.info(f"Instantiating logger <{lg_conf._target_}>")
                        logger.append(hydra.utils.instantiate(lg_conf))
        elif isinstance(logger_conf, ListConfig):
            for lg_conf in logger_conf:
                if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
                    lg_conf = _normalize_wandb_logger_conf(lg_conf)
                    log.info(f"Instantiating logger <{lg_conf._target_}>")
                    logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )
    
    # Log wandb run ID to make it easy to find offline runs later
    for lg in logger:
        if hasattr(lg, "experiment") and hasattr(lg, "version"):
            try:
                _ = lg.experiment # force initialization
                log.info(f"Wandb run ID: {lg.version}")
            except Exception:
                pass

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Train the model
    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    best_model = model
    best_ckpt_path = trainer.checkpoint_callback.best_model_path if trainer.checkpoint_callback is not None else ""

    threshold_cfg = config.get("threshold_search")
    if threshold_cfg and threshold_cfg.get("enabled", False):
        log.info("Running Optuna threshold search on validation predictions from the best checkpoint.")
        best_model, best_threshold, best_threshold_metric = _run_threshold_search(
            trainer=trainer,
            model=model,
            datamodule=datamodule,
            best_ckpt_path=best_ckpt_path,
            metric_name=threshold_cfg.get("metric", "illicit_f1"),
            n_trials=int(threshold_cfg.get("n_trials", 40)),
            lower=float(threshold_cfg.get("lower", 0.05)),
            upper=float(threshold_cfg.get("upper", 0.95)),
        )
        log.info(
            "Best validation threshold %.4f with %s=%.6f",
            best_threshold,
            threshold_cfg.get("metric", "illicit_f1"),
            best_threshold_metric,
        )
        for lg in logger:
            try:
                lg.log_metrics({
                    "val/threshold": best_threshold,
                    f"val/{threshold_cfg.get('metric', 'illicit_f1')}": best_threshold_metric,
                })
            except Exception:
                pass
    elif best_ckpt_path:
        best_model = model.__class__.load_from_checkpoint(best_ckpt_path, given_datamodule=datamodule)

    # Evaluate model on test set, using the best model achieved during training
    if config.get("test_after_training") and not config.trainer.get("fast_dev_run"):
        # is_hpo_run = bool(config.get("objective"))
        # run_test_during_hpo = bool(config.get("test_during_hpo", False))
        # if is_hpo_run and not run_test_during_hpo:
        #     log.info("Skipping testing during hyperparameter search. Set test_during_hpo=true to enable it.")
        # else:
        #     log.info("Starting testing!")
        #     trainer.test(model=model, datamodule=datamodule, ckpt_path="best")

        log.info("Starting testing!")
        trainer.test(model=best_model, datamodule=datamodule)

    # Make sure everything closed properly
    log.info("Finalizing!")
    finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")
    if config.get("remove_best_model_ckpt") and trainer.checkpoint_callback.best_model_path:
        os.remove(trainer.checkpoint_callback.best_model_path)
        log.info(f"Removed: {trainer.checkpoint_callback.best_model_path}")

    if config.get("objective"):
        objective_key = config.objective
        value = None
        if trainer.checkpoint_callback is not None \
                and trainer.checkpoint_callback.monitor == objective_key \
                and trainer.checkpoint_callback.best_model_score is not None:
            value = trainer.checkpoint_callback.best_model_score
        if value is None:
            value = trainer.callback_metrics.get(objective_key)
        if value is None and trainer.checkpoint_callback is not None:
            value = trainer.checkpoint_callback.best_model_score
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        log.info(f"HPARAMS_SEARCH OBJECTIVE: {objective_key}, VALUE: {value}")
        if value is None:
            raise ValueError(f"Objective '{objective_key}' not found in callback metrics.")
        return float(value)

    return {
        trainer.checkpoint_callback.monitor: trainer.checkpoint_callback.best_model_score.cpu(),
        **trainer.callback_metrics,
    }


@hydra.main(config_path="../configs/", config_name="config.yaml")
def main(config: DictConfig) -> Union[Dict[str, Any], float, None]:

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    import run_utils
    import utils
    import torch

    # A100 (Tensor Cores): trade a bit of FP32 precision for speed.
    # Safe to call once at startup.
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # A couple of optional utilities:
    # - disabling python warnings
    # - easier access to debug mode
    # - forcing debug friendly configuration
    # You can safely get rid of this line if you don't want those
    run_utils.extras(config)

    # Pretty print config using Rich library
    if config.get("print_config"):
        run_utils.print_config(config, resolve=False)

    # Train model, if num_averaging is given, run multiple `train`.
    if config.get("objective"):
        # For Optuna, we run the training once and return the objective
        optimized_metric = train(config)
        if not isinstance(optimized_metric, (int, float)):
            raise TypeError(f"Expected scalar optimized metric, got: {type(optimized_metric).__name__}")
        log.info(f"Returning optimized_metric from main: {optimized_metric}")
        return float(optimized_metric)
    else:
        # For normal runs, we can average over multiple seeds
        num_averaging: int = config.get("num_averaging", default_value=1)
        if num_averaging is None:
            num_averaging = 1
        seed = config.seed
        seed_name_template = None
        if isinstance(config.get("logger"), DictConfig):
            if isinstance(config.logger.get("wandb"), DictConfig):
                maybe_name = config.logger.wandb.get("name")
                if isinstance(maybe_name, str) and ("{seed}" in maybe_name or "__seed__" in maybe_name):
                    seed_name_template = ("nested_wandb", maybe_name)
            else:
                maybe_name = config.logger.get("name")
                if isinstance(maybe_name, str) and ("{seed}" in maybe_name or "__seed__" in maybe_name):
                    seed_name_template = ("flat_logger", maybe_name)

        trained_metrics = []
        for run_no in range(num_averaging):
            log.info(f"Running experiment {run_no + 1} out of {num_averaging}")
            seed_forced = (seed + run_no) if seed is not None else seed
            if seed_name_template is not None and seed_forced is not None:
                name_kind, name_template = seed_name_template
                formatted_name = name_template.replace("__seed__", str(seed_forced)).format(
                    seed=seed_forced,
                    run_no=run_no,
                )
                if name_kind == "nested_wandb" and isinstance(config.logger.get("wandb"), DictConfig):
                    with open_dict(config.logger.wandb):
                        config.logger.wandb.name = formatted_name
                else:
                    with open_dict(config.logger):
                        config.logger.name = formatted_name
                log.info(f"Using seed-specific logger name: {formatted_name}")
            trained_metrics.append(train(config, seed_forced=seed_forced))
        trained_metrics = utils.ld_to_dl(trained_metrics)

        # Log the summary
        log.info("--- Summary ({} runs) ---".format(num_averaging))
        for m, vs in trained_metrics.items():
            if m.startswith("test"):
                log.info("{}: {:.5f} +- {:.5f}".format(m, *utils.mean_std(vs)))

        optimized_metric = config.get("optimized_metric")
        if optimized_metric:
            om, _ = utils.mean_std(trained_metrics[optimized_metric])
            log.info("Return {}: {:.5f}".format(optimized_metric, om))
            return om

if __name__ == "__main__":
    import sys
    main()  # type: ignore[call-arg]
