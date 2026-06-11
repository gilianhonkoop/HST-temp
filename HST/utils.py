import logging
import torch
import sys
import numpy as np
try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except ImportError:
    pass

def setup_logging():
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )

def _as_float(v, name: str) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        raise ValueError(f"Config '{name}' must be a float-compatible value, got: {v!r}")

def _as_int(v, name: str) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        raise ValueError(f"Config '{name}' must be an int-compatible value, got: {v!r}")

def _compute_class_weight_from_targets(targets: torch.Tensor, device: str) -> torch.Tensor:
    """
    pos_weight for BCEWithLogitsLoss: num_negative / num_positive
    """
    targets = targets.float()
    pos = float((targets > 0.5).sum().item())
    neg = float((targets <= 0.5).sum().item())
    if pos <= 0.0:
        logging.warning("No positive labels found in train split; using pos_weight=1.0")
        return torch.tensor([1.0], device=device)
    return torch.tensor([neg / pos], device=device)

def _binary_metrics_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold)
    t = (targets.float() >= 0.5)

    tp = (preds & t).sum().item()
    fp = (preds & ~t).sum().item()
    fn = (~preds & t).sum().item()

    eps = 1e-12
    recall = tp / (tp + fn + eps)
    precision = tp / (tp + fp + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)

    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
    }

def _binary_metrics_from_logits_minor(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict:
    """
    Computes precision/recall/F1 for the minority class only.
    Minority class is determined from `targets` within the provided slice.
    If both classes are equally frequent, class 1 is used.
    """
    probs = torch.sigmoid(logits)
    preds_pos = (probs >= threshold)
    t_pos = (targets.float() >= 0.5)

    num_pos = int(t_pos.sum().item())
    num_neg = int((~t_pos).sum().item())

    minority_is_pos = num_pos <= num_neg

    if minority_is_pos:
        preds_minor = preds_pos
        t_minor = t_pos
        probs_minor = probs
    else:
        preds_minor = ~preds_pos
        t_minor = ~t_pos
        probs_minor = 1.0 - probs

    tp = (preds_minor & t_minor).sum().item()
    fp = (preds_minor & ~t_minor).sum().item()
    fn = (~preds_minor & t_minor).sum().item()

    eps = 1e-12
    recall = tp / (tp + fn + eps)
    precision = tp / (tp + fp + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    
    roc_auc, pr_auc = 0.0, 0.0
    try:
        if 'roc_auc_score' in globals():
            t_np = t_minor.cpu().numpy()
            p_np = probs_minor.detach().cpu().numpy()
            
            # AUC requires at least one positive and one negative sample
            if len(np.unique(t_np)) > 1:
                roc_auc = float(roc_auc_score(t_np, p_np))
                pr_auc = float(average_precision_score(t_np, p_np))
    except Exception as e:
        pass

    return {
        'minority_class': 1 if minority_is_pos else 0,
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
    }

def _binary_metrics_by_class(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    illicit_label: int = 1,
) -> dict:
    """
    Computes precision/recall/F1 and PR/ROC-AUC for each class:
    class 1 = illicit, class 0 = licit.
    """
    probs_pos = torch.sigmoid(logits)
    t_pos = (targets.float() >= 0.5)

    if illicit_label == 1:
        probs_illicit = probs_pos
        t_illicit = t_pos
    else:
        probs_illicit = 1.0 - probs_pos
        t_illicit = ~t_pos

    preds_illicit = (probs_illicit >= threshold)
    probs_licit = 1.0 - probs_illicit
    t_licit = ~t_illicit
    preds_licit = ~preds_illicit

    def _class_metrics(preds: torch.Tensor, t: torch.Tensor) -> dict:
        tp = (preds & t).sum().item()
        fp = (preds & ~t).sum().item()
        fn = (~preds & t).sum().item()

        eps = 1e-12
        recall = tp / (tp + fn + eps)
        precision = tp / (tp + fp + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
        }

    illicit = _class_metrics(preds_illicit, t_illicit)
    licit = _class_metrics(preds_licit, t_licit)

    illicit_roc_auc, illicit_pr_auc = 0.0, 0.0
    licit_roc_auc, licit_pr_auc = 0.0, 0.0
    try:
        if 'roc_auc_score' in globals():
            t_illicit_np = t_illicit.detach().cpu().numpy()
            probs_illicit_np = probs_illicit.detach().cpu().numpy()
            if len(np.unique(t_illicit_np)) > 1:
                illicit_roc_auc = float(roc_auc_score(t_illicit_np, probs_illicit_np))
                illicit_pr_auc = float(average_precision_score(t_illicit_np, probs_illicit_np))

            t_licit_np = t_licit.detach().cpu().numpy()
            probs_licit_np = probs_licit.detach().cpu().numpy()
            if len(np.unique(t_licit_np)) > 1:
                licit_roc_auc = float(roc_auc_score(t_licit_np, probs_licit_np))
                licit_pr_auc = float(average_precision_score(t_licit_np, probs_licit_np))
    except Exception:
        pass

    return {
        'illicit': {
            **illicit,
            'roc_auc': illicit_roc_auc,
            'pr_auc': illicit_pr_auc,
        },
        'licit': {
            **licit,
            'roc_auc': licit_roc_auc,
            'pr_auc': licit_pr_auc,
        },
    }
