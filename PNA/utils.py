import logging
import random
import sys
from typing import Dict

import numpy as np
import torch

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except ImportError:  # pragma: no cover
    average_precision_score = None
    roc_auc_score = None


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def as_float(value, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Config '{name}' must be float-compatible, got {value!r}") from exc


def as_int(value, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Config '{name}' must be int-compatible, got {value!r}") from exc


def count_trainable_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def class_pos_weight(targets: torch.Tensor, device: torch.device) -> torch.Tensor:
    targets = targets.float().view(-1)
    pos = float((targets > 0.5).sum().item())
    neg = float((targets <= 0.5).sum().item())
    if pos <= 0.0:
        logging.warning("No positive labels in train split; using pos_weight=1.0")
        return torch.tensor([1.0], device=device)
    return torch.tensor([neg / pos], device=device)


def binary_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    logits = logits.detach().view(-1).cpu()
    targets = targets.detach().view(-1).float().cpu()
    probs = torch.sigmoid(logits)
    preds = probs >= float(threshold)
    true = targets >= 0.5

    def _prf(pred_mask: torch.Tensor, true_mask: torch.Tensor) -> Dict[str, float]:
        tp = float((pred_mask & true_mask).sum().item())
        fp = float((pred_mask & ~true_mask).sum().item())
        fn = float((~pred_mask & true_mask).sum().item())
        eps = 1e-12
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    illicit = _prf(preds, true)
    licit = _prf(~preds, ~true)

    illicit_roc_auc = 0.0
    illicit_pr_auc = 0.0
    licit_roc_auc = 0.0
    licit_pr_auc = 0.0
    y_np = true.numpy().astype(np.int64)
    p_np = probs.numpy()
    if roc_auc_score is not None and len(np.unique(y_np)) > 1:
        illicit_roc_auc = float(roc_auc_score(y_np, p_np))
        illicit_pr_auc = float(average_precision_score(y_np, p_np))
        licit_roc_auc = float(roc_auc_score(1 - y_np, 1.0 - p_np))
        licit_pr_auc = float(average_precision_score(1 - y_np, 1.0 - p_np))

    return {
        "illicit_precision": float(illicit["precision"]),
        "illicit_recall": float(illicit["recall"]),
        "illicit_f1": float(illicit["f1"]),
        "illicit_roc_auc": illicit_roc_auc,
        "illicit_pr_auc": illicit_pr_auc,
        "licit_precision": float(licit["precision"]),
        "licit_recall": float(licit["recall"]),
        "licit_f1": float(licit["f1"]),
        "licit_roc_auc": licit_roc_auc,
        "licit_pr_auc": licit_pr_auc,
    }
