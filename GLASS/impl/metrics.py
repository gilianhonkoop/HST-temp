from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_fscore_support, roc_auc_score)
import numpy as np


def _binary_targets(pred, label, threshold=0.5):
    pred = np.asarray(pred)
    label = np.asarray(label)

    if pred.ndim > 1 and pred.shape[1] == 1:
        pred = pred.reshape(-1)
    elif pred.ndim > 1 and pred.shape[1] == 2:
        pred = pred[:, 1]

    label = label.reshape(-1)
    if label.dtype.kind in {"f", "c"}:
        y_true = (label > 0.5).astype(np.int64)
    else:
        y_true = label.astype(np.int64)

    prob = 1.0 / (1.0 + np.exp(-pred))
    y_pred = (prob > threshold).astype(np.int64)
    return y_true, y_pred, prob


def binaryf1(pred, label):
    '''
    pred, label are numpy array
    can process multi-label target
    '''
    pred_i = (pred > 0).astype(np.int64)
    label_i = label.reshape(pred.shape[0], -1)
    return f1_score(label_i, pred_i, average="micro")


def microf1(pred, label):
    '''
    multi-class micro-f1
    '''
    pred_i = np.argmax(pred, axis=1)
    return f1_score(label, pred_i, average="micro")


def auroc(pred, label):
    '''
    calculate auroc
    '''
    return roc_auc_score(label, pred)


def minor_f1(pred, label, threshold=0.5):
    y_true, y_pred, _ = _binary_targets(pred, label, threshold=threshold)
    if y_true.size == 0:
        return 0.0
    counts = np.bincount(y_true, minlength=2)
    minor_label = int(np.argmin(counts))
    try:
        scores = f1_score(y_true, y_pred, average=None, labels=[0, 1])
        return float(scores[minor_label])
    except ValueError:
        return 0.0


def major_f1(pred, label, threshold=0.5):
    y_true, y_pred, _ = _binary_targets(pred, label, threshold=threshold)
    if y_true.size == 0:
        return 0.0
    counts = np.bincount(y_true, minlength=2)
    major_label = int(np.argmax(counts))
    try:
        scores = f1_score(y_true, y_pred, average=None, labels=[0, 1])
        return float(scores[major_label])
    except ValueError:
        return 0.0


def prauc(pred, label):
    y_true, _, prob = _binary_targets(pred, label)
    try:
        return average_precision_score(y_true, prob)
    except ValueError:
        return float("nan")


def aml_metrics(pred, label, threshold=0.5):
    y_true, y_pred, prob = _binary_targets(pred, label, threshold=threshold)
    metrics = {
        "minor_f1": minor_f1(pred, label, threshold=threshold),
        "major_f1": major_f1(pred, label, threshold=threshold),
        "rocauc": float("nan"),
        "prauc": float("nan"),
        "minor_rocauc": float("nan"),
        "major_rocauc": float("nan"),
        "minor_prauc": float("nan"),
        "major_prauc": float("nan"),
        "minor_precision": 0.0,
        "major_precision": 0.0,
        "minor_recall": 0.0,
        "major_recall": 0.0,
    }
    
    if y_true.size > 0:
        counts = np.bincount(y_true, minlength=2)
        minor_label = int(np.argmin(counts))
        major_label = int(np.argmax(counts))

        precision, recall, _, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[0, 1],
            zero_division=0,
        )
        metrics["minor_precision"] = float(precision[minor_label])
        metrics["major_precision"] = float(precision[major_label])
        metrics["minor_recall"] = float(recall[minor_label])
        metrics["major_recall"] = float(recall[major_label])

        if minor_label == 0:
            y_minor = 1 - y_true
            prob_minor = 1.0 - prob
        else:
            y_minor = y_true
            prob_minor = prob

        if major_label == 0:
            y_major = 1 - y_true
            prob_major = 1.0 - prob
        else:
            y_major = y_true
            prob_major = prob

        try:
            metrics["minor_rocauc"] = float(roc_auc_score(y_minor, prob_minor))
        except ValueError:
            pass
        try:
            metrics["minor_prauc"] = float(
                average_precision_score(y_minor, prob_minor))
        except ValueError:
            pass
        try:
            metrics["major_rocauc"] = float(roc_auc_score(y_major, prob_major))
        except ValueError:
            pass
        try:
            metrics["major_prauc"] = float(
                average_precision_score(y_major, prob_major))
        except ValueError:
            pass

        metrics["rocauc"] = metrics["minor_rocauc"]
        metrics["prauc"] = metrics["minor_prauc"]
            
    return metrics
