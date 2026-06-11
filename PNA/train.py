import argparse
import copy
import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import yaml
from dotenv import load_dotenv
from torch_geometric.loader import DataLoader

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

from datasets import LoadedData, load_aml_dataset
from model import PNAGraphClassifier
from utils import (
    as_float,
    as_int,
    binary_metrics,
    class_pos_weight,
    count_trainable_params,
    set_seed,
    setup_logging,
)


load_dotenv()
setup_logging()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def save_config(cfg: dict, path: str) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Train PNA baseline")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    return parser.parse_args()


def build_loaders(data: LoadedData, cfg: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_cfg = cfg.get("train", {})
    batch_size = as_int(train_cfg.get("batch_size", 128), "train.batch_size")
    eval_batch_size = as_int(train_cfg.get("eval_batch_size", batch_size), "train.eval_batch_size")
    num_workers = as_int(train_cfg.get("num_workers", 0), "train.num_workers")
    pin_memory = bool(train_cfg.get("pin_memory", False))
    persistent_workers = bool(train_cfg.get("persistent_workers", False)) and num_workers > 0

    return (
        DataLoader(
            data.train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        ),
        DataLoader(
            data.val,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        ),
        DataLoader(
            data.test,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        ),
    )


def build_model(data: LoadedData, cfg: dict, device: torch.device) -> PNAGraphClassifier:
    model_cfg = cfg.get("model", {})
    model = PNAGraphClassifier(
        in_channels=data.num_node_features,
        hidden_channels=as_int(model_cfg.get("hidden_dim", 128), "model.hidden_dim"),
        num_layers=as_int(model_cfg.get("num_layers", 4), "model.num_layers"),
        deg=data.deg,
        edge_dim=data.num_edge_features if bool(cfg.get("dataset", {}).get("use_edge_features", True)) else 0,
        dropout=as_float(model_cfg.get("dropout", 0.2), "model.dropout"),
        towers=as_int(model_cfg.get("towers", 4), "model.towers"),
        pre_layers=as_int(model_cfg.get("pre_layers", 1), "model.pre_layers"),
        post_layers=as_int(model_cfg.get("post_layers", 1), "model.post_layers"),
        aggregators=list(model_cfg.get("aggregators", ["mean", "min", "max", "std"])),
        scalers=list(model_cfg.get("scalers", ["identity", "amplification", "attenuation"])),
        readout=str(model_cfg.get("readout", "mean_sum_max")),
    )
    return model.to(device)


def build_optimizer(model: nn.Module, cfg: dict):
    optim_cfg = cfg.get("optim", {})
    lr = as_float(optim_cfg.get("base_lr", 1e-3), "optim.base_lr")
    weight_decay = as_float(optim_cfg.get("weight_decay", 1e-4), "optim.weight_decay")
    optimizer_name = str(optim_cfg.get("optimizer", "adamw")).lower()
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_scheduler(optimizer, cfg: dict):
    optim_cfg = cfg.get("optim", {})
    scheduler_cfg = optim_cfg.get("lr_scheduler", {})
    if not bool(scheduler_cfg.get("enabled", False)):
        return None
    scheduler_type = str(scheduler_cfg.get("type", "plateau")).lower()
    if scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(scheduler_cfg.get("mode", "max")),
            factor=as_float(scheduler_cfg.get("factor", 0.7), "optim.lr_scheduler.factor"),
            patience=as_int(scheduler_cfg.get("patience", 5), "optim.lr_scheduler.patience"),
            min_lr=as_float(scheduler_cfg.get("min_lr", 1e-6), "optim.lr_scheduler.min_lr"),
        )
    if scheduler_type == "cosine":
        epochs = as_int(optim_cfg.get("max_epoch", 100), "optim.max_epoch")
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs),
            eta_min=as_float(scheduler_cfg.get("min_lr", 1e-6), "optim.lr_scheduler.min_lr"),
        )
    raise ValueError(f"Unsupported lr scheduler: {scheduler_type}")


def flatten_metrics(prefix: str, metrics: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}/{key}": float(value) for key, value in metrics.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
) -> Tuple[float, Dict[str, float], torch.Tensor, torch.Tensor]:
    model.eval()
    losses = []
    logits_all = []
    targets_all = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        targets = batch.y.float().view(-1)
        loss = criterion(logits, targets)
        losses.append(float(loss.item()) * int(targets.numel()))
        logits_all.append(logits.detach().cpu())
        targets_all.append(targets.detach().cpu())

    logits_cat = torch.cat(logits_all, dim=0)
    targets_cat = torch.cat(targets_all, dim=0)
    denom = max(1, int(targets_cat.numel()))
    metrics = binary_metrics(logits_cat, targets_cat, threshold=threshold)
    return sum(losses) / denom, metrics, logits_cat, targets_cat


def find_best_threshold(
    logits: torch.Tensor,
    targets: torch.Tensor,
    cfg: dict,
) -> Tuple[float, float, Dict[str, float]]:
    search_cfg = cfg.get("threshold_search", {})
    lower = as_float(search_cfg.get("lower", 0.05), "threshold_search.lower")
    upper = as_float(search_cfg.get("upper", 0.95), "threshold_search.upper")
    n_trials = as_int(search_cfg.get("n_trials", 40), "threshold_search.n_trials")
    metric_name = str(search_cfg.get("metric", "illicit_f1"))
    if metric_name.startswith("val/"):
        metric_name = metric_name.split("/", 1)[1]

    best_threshold = lower
    best_score = -float("inf")
    best_metrics = {}
    for threshold in torch.linspace(lower, upper, max(1, n_trials)).tolist():
        metrics = binary_metrics(logits, targets, threshold=float(threshold))
        score = float(metrics[metric_name])
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_score, best_metrics


def metric_value(metrics: Dict[str, float], metric_name: str) -> float:
    if metric_name not in metrics:
        raise KeyError(f"Metric {metric_name!r} not found. Available: {sorted(metrics)}")
    return float(metrics[metric_name])


def init_wandb(cfg: dict, run_name: str):
    logging_cfg = cfg.get("logging", {})
    if not bool(logging_cfg.get("use_wandb", False)):
        return None
    if wandb is None:
        raise ImportError("wandb is not installed, but logging.use_wandb=true")
    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        raise ValueError("WANDB_API_KEY is not set, but logging.use_wandb=true")
    wandb.login(key=api_key)
    return wandb.init(
        entity=logging_cfg.get("wandb_entity", "hierarchical-subgraph-transformer"),
        project=logging_cfg.get("wandb_project", "PNA"),
        name=run_name,
        config=cfg,
    )


def _checkpoint_payload(model, optimizer, scheduler, epoch, best_metric, patience_counter, cfg, num_params):
    return {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_metric": float(best_metric),
        "patience_counter": int(patience_counter),
        "config": cfg,
        "num_params": int(num_params),
    }


def train_main(args=None, cfg_override: dict | None = None, preloaded_data: LoadedData | None = None):
    if args is None:
        args = parse_args()
    cfg = cfg_override if cfg_override is not None else load_config(args.config)

    seed = as_int(cfg.get("train", {}).get("seed", 42), "train.seed")
    set_seed(seed)
    device = torch.device(cfg.get("train", {}).get("device", "cuda") if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    data = preloaded_data if preloaded_data is not None else load_aml_dataset(cfg.get("dataset", {}))
    train_loader, val_loader, test_loader = build_loaders(data, cfg)
    model = build_model(data, cfg, device)
    num_params = count_trainable_params(model)
    logging.info("Trainable parameters: %d", num_params)

    train_targets = torch.cat([d.y.float().view(-1) for d in data.train]).to(device)
    computed_pos_weight = class_pos_weight(train_targets, device=device)
    pos_weight_cfg = cfg.get("train", {}).get("pos_weight", None)
    pos_weight = torch.tensor(
        [float(pos_weight_cfg)] if pos_weight_cfg is not None else [float(computed_pos_weight.item())],
        device=device,
    )
    logging.info("Using BCEWithLogitsLoss pos_weight=%.6f", float(pos_weight.item()))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    logging_cfg = cfg.get("logging", {})
    run_name = str(logging_cfg.get("run_name", "pna_run"))
    save_dir = str(logging_cfg.get("save_dir", "outputs"))
    run_dir = os.path.join(save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    save_config(cfg, os.path.join(run_dir, "config.yaml"))

    monitor_metric = str(logging_cfg.get("monitor_metric", "val/illicit_pr_auc"))
    monitor_mode = str(logging_cfg.get("monitor_mode", "max")).lower()
    logging_interval = as_int(logging_cfg.get("log_interval", 1), "logging.log_interval")
    best_metric = -float("inf") if monitor_mode == "max" else float("inf")
    patience_counter = 0
    start_epoch = 1
    best_path = os.path.join(run_dir, "best_checkpoint.pth")
    last_path = os.path.join(run_dir, "last_checkpoint.pth")

    if getattr(args, "resume", False):
        if not os.path.exists(last_path):
            raise FileNotFoundError(f"--resume requested but checkpoint not found: {last_path}")
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_metric = float(ckpt.get("best_metric", best_metric))
        patience_counter = int(ckpt.get("patience_counter", 0))
        start_epoch = int(ckpt["epoch"]) + 1
        logging.info("Resumed from %s at epoch %d", last_path, start_epoch)

    run = init_wandb(cfg, run_name)
    max_epoch = as_int(cfg.get("optim", {}).get("max_epoch", cfg.get("train", {}).get("epochs", 100)), "optim.max_epoch")
    min_epoch = as_int(cfg.get("optim", {}).get("min_epoch", 0), "optim.min_epoch")
    patience = as_int(cfg.get("optim", {}).get("patience", 20), "optim.patience")
    eval_period = as_int(cfg.get("train", {}).get("eval_period", 1), "train.eval_period")
    threshold = as_float(cfg.get("train", {}).get("training_threshold", 0.5), "train.training_threshold")
    clip_grad = bool(cfg.get("optim", {}).get("clip_grad_norm", True))
    max_grad_norm = as_float(cfg.get("optim", {}).get("max_grad_norm", 5.0), "optim.max_grad_norm")

    for epoch in range(start_epoch, max_epoch + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            targets = batch.y.float().view(-1)
            loss = criterion(logits, targets)
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            running_loss += float(loss.item()) * int(targets.numel())
            seen += int(targets.numel())

        train_loss = running_loss / max(1, seen)
        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()

        should_eval = epoch % eval_period == 0 or epoch == max_epoch
        if not should_eval:
            continue

        train_eval_loss, train_metrics, _, _ = evaluate(model, train_loader, criterion, device, threshold)
        val_loss, val_metrics, _, _ = evaluate(model, val_loader, criterion, device, threshold)
        flat_metrics = {
            "epoch": float(epoch),
            "train/loss": train_eval_loss,
            "val/loss": val_loss,
            **flatten_metrics("train", train_metrics),
            **flatten_metrics("val", val_metrics),
        }
        monitor = metric_value(flat_metrics, monitor_metric)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(monitor)

        improved = monitor > best_metric if monitor_mode == "max" else monitor < best_metric
        if improved:
            best_metric = monitor
            patience_counter = 0
            torch.save(
                _checkpoint_payload(model, optimizer, scheduler, epoch, best_metric, patience_counter, cfg, num_params),
                best_path,
            )
        else:
            patience_counter += 1

        torch.save(
            _checkpoint_payload(model, optimizer, scheduler, epoch, best_metric, patience_counter, cfg, num_params),
            last_path,
        )

        if epoch % logging_interval == 0 or epoch == max_epoch:
            logging.info(
                "Epoch %d/%d | train_loss=%.4f val_loss=%.4f val_illicit_f1=%.4f "
                "val_illicit_pr_auc=%.4f monitor(%s)=%.4f",
                epoch,
                max_epoch,
                train_loss,
                val_loss,
                flat_metrics["val/illicit_f1"],
                flat_metrics["val/illicit_pr_auc"],
                monitor_metric,
                monitor,
            )
            
        if run is not None:
            run.log(flat_metrics)

        if patience_counter >= patience and epoch >= min_epoch:
            logging.info("Early stopping at epoch %d after %d stale evals", epoch, patience)
            break

    eval_metrics = evaluate_checkpoint_from_config(cfg, data, checkpoint_path=best_path)
    metrics_path = os.path.join(run_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(eval_metrics, f, indent=2)
    threshold_metric = str(cfg.get("threshold_search", {}).get("metric", "illicit_f1"))
    if not threshold_metric.startswith("val/"):
        threshold_metric = f"val/{threshold_metric}"
    logging.info(
        "Best threshold on validation (%s) = %.4f (score=%s)",
        threshold_metric,
        eval_metrics["best_threshold"],
        "n/a" if eval_metrics["threshold_search_score"] is None else f"{eval_metrics['threshold_search_score']:.4f}",
    )
    logging.info(
        "Test metrics | Loss: %.4f, Illicit F1: %.4f, Illicit PR-AUC: %.4f, "
        "Illicit ROC-AUC: %.4f, Licit F1: %.4f",
        eval_metrics["test/loss"],
        eval_metrics["test/illicit_f1"],
        eval_metrics["test/illicit_pr_auc"],
        eval_metrics["test/illicit_roc_auc"],
        eval_metrics["test/licit_f1"],
    )
    logging.info("Saved final metrics to %s", metrics_path)
    if run is not None:
        run.log({k: v for k, v in eval_metrics.items() if isinstance(v, (int, float))})
        run.finish()
    return float(best_metric), num_params


def evaluate_checkpoint_from_config(
    cfg: dict,
    data: LoadedData | None = None,
    checkpoint_path: str | None = None,
) -> Dict[str, float]:
    device = torch.device(cfg.get("train", {}).get("device", "cuda") if torch.cuda.is_available() else "cpu")
    data = data if data is not None else load_aml_dataset(cfg.get("dataset", {}))
    _, val_loader, test_loader = build_loaders(data, cfg)
    model = build_model(data, cfg, device)

    logging_cfg = cfg.get("logging", {})
    run_name = str(logging_cfg.get("run_name", "pna_run"))
    run_dir = os.path.join(str(logging_cfg.get("save_dir", "outputs")), run_name)
    checkpoint_path = checkpoint_path or str(cfg.get("train", {}).get("checkpoint_path", ""))
    if not checkpoint_path:
        checkpoint_path = os.path.join(run_dir, "best_checkpoint.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    train_targets = torch.cat([d.y.float().view(-1) for d in data.train]).to(device)
    computed_pos_weight = class_pos_weight(train_targets, device=device)
    pos_weight_cfg = cfg.get("train", {}).get("pos_weight", None)
    pos_weight = torch.tensor(
        [float(pos_weight_cfg)] if pos_weight_cfg is not None else [float(computed_pos_weight.item())],
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    threshold = as_float(cfg.get("train", {}).get("training_threshold", 0.5), "train.training_threshold")
    val_loss, val_metrics, val_logits, val_targets = evaluate(model, val_loader, criterion, device, threshold)
    best_threshold = threshold
    threshold_score = None
    if bool(cfg.get("threshold_search", {}).get("enabled", False)):
        best_threshold, threshold_score, val_metrics = find_best_threshold(val_logits, val_targets, cfg)

    test_loss, test_metrics, _, _ = evaluate(model, test_loader, criterion, device, best_threshold)
    return {
        "checkpoint_path": checkpoint_path,
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "checkpoint_best_metric": float(ckpt.get("best_metric", float("nan"))),
        "best_threshold": float(best_threshold),
        "threshold_search_score": float(threshold_score) if threshold_score is not None else None,
        "val/loss": float(val_loss),
        "test/loss": float(test_loss),
        **flatten_metrics("val", val_metrics),
        **flatten_metrics("test", test_metrics),
        "num_params": int(ckpt.get("num_params", count_trainable_params(model))),
    }


if __name__ == "__main__":
    train_main()
