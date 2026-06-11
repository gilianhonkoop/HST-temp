from typing import List, Dict, Any

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
)
import torch
import numpy as np


def _pred_labels(logits, is_multi_labels=False, threshold=0.5):
    if is_multi_labels:
        return (logits > 0.0).int()

    if logits.dim() == 1 or logits.size(-1) == 1:
        return (logits > 0.0).long().view(-1)

    if logits.size(-1) == 2:
        probs = torch.softmax(logits, dim=-1)
        # Threshold the probability of class 0 and emit the actual predicted class id.
        return torch.where(
            probs[:, 0] > threshold,
            torch.zeros_like(probs[:, 0], dtype=torch.long),
            torch.ones_like(probs[:, 0], dtype=torch.long),
        )

    return torch.argmax(logits, dim=-1)


def _f1_score(logits, ys, avg_type="micro", is_multi_labels=False, num_classes=None):
    preds = (logits > 0.0) if is_multi_labels else torch.argmax(logits, dim=-1)
    if avg_type == "per_class":
        labels = list(range(num_classes)) if num_classes is not None else None
        return f1_score(ys.cpu().detach(), preds.cpu().detach(), average=None, labels=labels, zero_division=0)
    return f1_score(ys.cpu().detach(), preds.cpu().detach(), average=avg_type, zero_division=0)


def _accuracy_score(logits, ys, is_multi_labels=False):
    preds = (logits > 0.0) if is_multi_labels else torch.argmax(logits, dim=-1)
    return accuracy_score(ys.cpu().detach(), preds.cpu().detach())


def _roc_auc_score(logits, ys, is_multi_labels=False, num_classes=None):
    ys_np = ys.cpu().detach().numpy()
    if is_multi_labels:
        probs = torch.sigmoid(logits).cpu().detach().numpy()
        if num_classes is not None:
            # Compute per-label scores, handling cases where a class is not in the batch
            scores = []
            for i in range(num_classes):
                if len(np.unique(ys_np[:, i])) > 1:
                    scores.append(roc_auc_score(ys_np[:, i], probs[:, i]))
                else:
                    scores.append(float('nan')) # or 0.5, depends on desired behavior
            return scores
        return roc_auc_score(ys_np, probs, average="macro")
    else:
        probs = torch.softmax(logits, dim=-1).cpu().detach().numpy()
        if num_classes is not None and num_classes > 1:
            y_true_binarized = torch.nn.functional.one_hot(ys.long(), num_classes=num_classes).cpu().detach().numpy()
            return roc_auc_score(y_true_binarized, probs, average=None, multi_class="ovr")
        # Fallback for binary case without num_classes, returns single score for class 1
        return roc_auc_score(ys_np, probs[:, 1])


def _pr_auc_score(logits, ys, is_multi_labels=False, num_classes=None):
    ys_np = ys.cpu().detach().numpy()
    if is_multi_labels:
        probs = torch.sigmoid(logits).cpu().detach().numpy()
        if num_classes is not None:
            scores = []
            for i in range(num_classes):
                if np.sum(ys_np[:, i]) > 0:
                    scores.append(average_precision_score(ys_np[:, i], probs[:, i]))
                else:
                    scores.append(float('nan'))
            return scores
        return average_precision_score(ys_np, probs, average="macro")
    else:
        probs = torch.softmax(logits, dim=-1).cpu().detach().numpy()
        if num_classes is not None and num_classes > 1:
            scores = []
            for i in range(num_classes):
                y_true_class = (ys_np == i).astype(int)
                if np.sum(y_true_class) > 0:
                    scores.append(average_precision_score(y_true_class, probs[:, i]))
                else:
                    scores.append(float('nan'))
            return scores
        # Fallback for binary case without num_classes, returns single score for class 1
        return average_precision_score(ys_np, probs[:, 1])


class Evaluator:

    def __init__(self, metrics: List[str], is_multi_labels, num_classes=None):
        self.metrics = metrics
        self.is_multi_labels = is_multi_labels
        self.num_classes = num_classes

    def __call__(self, logits, ys, threshold=0.5) -> Dict[str, Any]:
        evaluated = {}
        preds = _pred_labels(logits, self.is_multi_labels, threshold=threshold)
        if "micro_f1" in self.metrics:
            evaluated["micro_f1"] = f1_score(ys.cpu(), preds.cpu(), average="micro", zero_division=0)
        if "macro_f1" in self.metrics:
            evaluated["macro_f1"] = f1_score(ys.cpu(), preds.cpu(), average="macro", zero_division=0)
        if "binary_f1" in self.metrics:
            evaluated["binary_f1"] = f1_score(ys.cpu(), preds.cpu(), average="binary", pos_label=1, zero_division=0)
        if "accuracy" in self.metrics:
            evaluated["accuracy"] = accuracy_score(ys.cpu(), preds.cpu())
        if "roc_auc" in self.metrics:
            evaluated["roc_auc"] = _roc_auc_score(logits, ys, self.is_multi_labels)
        if "pr_auc" in self.metrics:
            evaluated["pr_auc"] = _pr_auc_score(logits, ys, self.is_multi_labels)
        if self.num_classes is not None and not self.is_multi_labels:
            labels = list(range(self.num_classes))
            f1_per_class = f1_score(
                ys.cpu(), preds.cpu(), average=None, labels=labels, zero_division=0,
            )
            precision_per_class = precision_score(
                ys.cpu(), preds.cpu(), average=None, labels=labels, zero_division=0,
            )
            recall_per_class = recall_score(
                ys.cpu(), preds.cpu(), average=None, labels=labels, zero_division=0,
            )
            roc_aucs = _roc_auc_score(logits, ys, self.is_multi_labels, self.num_classes)
            pr_aucs = _pr_auc_score(logits, ys, self.is_multi_labels, self.num_classes)
            for i in range(self.num_classes):
                evaluated[f"f1_class_{i}"] = f1_per_class[i]
                evaluated[f"precision_class_{i}"] = precision_per_class[i]
                evaluated[f"recall_class_{i}"] = recall_per_class[i]
                evaluated[f"roc_auc_class_{i}"] = roc_aucs[i]
                evaluated[f"pr_auc_class_{i}"] = pr_aucs[i]
        return evaluated
