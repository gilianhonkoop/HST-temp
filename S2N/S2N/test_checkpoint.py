import json
import os
import time
from pathlib import Path
from typing import List

import hydra
import torch
from omegaconf import DictConfig, open_dict
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import Logger

from run_main import _run_threshold_search
from run_utils import finish, get_logger, log_hyperparameters
from utils import make_deterministic_everything


log = get_logger(__name__)


def _normalize_wandb_logger_conf(lg_conf: DictConfig) -> DictConfig:
    if "_target_" in lg_conf and str(lg_conf._target_).endswith("WandbLogger"):
        nested = lg_conf.get("wandb")
        if isinstance(nested, DictConfig):
            with open_dict(lg_conf):
                for k, v in nested.items():
                    if v is not None and not (isinstance(v, str) and v == ""):
                        lg_conf[k] = v
                if "wandb" in lg_conf:
                    del lg_conf["wandb"]
    return lg_conf


def _instantiate_callbacks(config: DictConfig) -> List[Callback]:
    callbacks: List[Callback] = []
    if "callbacks" not in config:
        return callbacks
    for _, cb_conf in config.callbacks.items():
        if "_target_" not in cb_conf:
            continue
        if "monitor" in cb_conf and cb_conf.monitor is None:
            cb_conf.monitor = f"valid/{config.model.metrics[0]}"
        log.info(f"Instantiating callback <{cb_conf._target_}>")
        callbacks.append(hydra.utils.instantiate(cb_conf))
    return callbacks


def _instantiate_loggers(config: DictConfig) -> List[Logger]:
    logger: List[Logger] = []
    if "logger" not in config:
        return logger

    logger_conf = config.logger
    if isinstance(logger_conf, DictConfig) and "_target_" in logger_conf:
        logger_conf = _normalize_wandb_logger_conf(logger_conf)
        log.info(f"Instantiating logger <{logger_conf._target_}>")
        logger.append(hydra.utils.instantiate(logger_conf))
    elif isinstance(logger_conf, DictConfig):
        for _, lg_conf in logger_conf.items():
            if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
                lg_conf = _normalize_wandb_logger_conf(lg_conf)
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))
    return logger


def _metrics_to_jsonable(metrics):
    result = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        result[key] = float(value) if isinstance(value, (int, float)) else value
    return result


@hydra.main(config_path="../configs/", config_name="config.yaml")
def main(config: DictConfig) -> None:
    checkpoint_path = config.get("checkpoint_path")
    if not checkpoint_path:
        raise ValueError("Pass +checkpoint_path=/path/to/checkpoint.ckpt")
    checkpoint_path = os.path.abspath(str(checkpoint_path))
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    if "seed" in config:
        seed_everything(config.seed, workers=True)
        make_deterministic_everything(config.seed)

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    logger = _instantiate_loggers(config)
    for lg in logger:
        try:
            _ = lg.experiment
            lg.log_metrics({"checkpoint_test/status": 0.0}, step=0)
        except Exception:
            pass

    started = time.time()
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    log.info("Datamodule setup finished in %.1fs.", time.time() - started)

    started = time.time()
    log.info(f"Instantiating model shell <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model, given_datamodule=datamodule)
    log.info("Model shell instantiated in %.1fs.", time.time() - started)

    callbacks = _instantiate_callbacks(config)

    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    log.info("Logging hyperparameters!")
    log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    threshold_cfg = config.get("threshold_search")
    if threshold_cfg and threshold_cfg.get("enabled", False):
        started = time.time()
        log.info("Running threshold search from checkpoint validation predictions.")
        best_model, best_threshold, best_threshold_metric = _run_threshold_search(
            trainer=trainer,
            model=model,
            datamodule=datamodule,
            best_ckpt_path=checkpoint_path,
            metric_name=threshold_cfg.get("metric", "illicit_f1"),
            n_trials=int(threshold_cfg.get("n_trials", 40)),
            lower=float(threshold_cfg.get("lower", 0.05)),
            upper=float(threshold_cfg.get("upper", 0.95)),
        )
        log.info(
            "Best validation threshold %.4f with %s=%.6f (%.1fs)",
            best_threshold,
            threshold_cfg.get("metric", "illicit_f1"),
            best_threshold_metric,
            time.time() - started,
        )
        for lg in logger:
            try:
                lg.log_metrics({
                    "val/threshold": best_threshold,
                    f"val/{threshold_cfg.get('metric', 'illicit_f1')}": best_threshold_metric,
                })
            except Exception:
                pass
    else:
        log.info("Loading checkpoint without threshold search.")
        best_model = model.__class__.load_from_checkpoint(checkpoint_path, given_datamodule=datamodule)

    started = time.time()
    log.info("Starting checkpoint testing!")
    test_metrics = trainer.test(model=best_model, datamodule=datamodule)
    log.info("Checkpoint testing finished in %.1fs.", time.time() - started)
    metrics = _metrics_to_jsonable(test_metrics[0] if test_metrics else {})

    output_path = Path(str(config.get("metrics_output", "checkpoint_test_metrics.json"))).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    log.info(f"Saved test metrics to {output_path}")
    print(json.dumps(metrics, indent=2, sort_keys=True))

    finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )


if __name__ == "__main__":
    main()
