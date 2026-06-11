import argparse
import json
import logging
import os
from typing import Tuple

import numpy as np
import torch

from train import load_config, build_aggregators
from datasets.utils import init_dataset
from summarize import SubgraphSummarizer
from models import HierarchicalSubgraphTransformer
from configs.config import ModelConfig, PearlConfig, FraudGTConfig, PNAConfig
from utils import _binary_metrics_by_class, setup_logging


setup_logging()


def _degree_histogram(edge_index: torch.Tensor | None, num_nodes: int, device: torch.device) -> torch.Tensor:
    if edge_index is None or edge_index.numel() == 0 or num_nodes <= 0:
        return torch.ones(1, dtype=torch.long, device=device)
    dst = edge_index[1].detach().to(device=device, dtype=torch.long)
    node_degrees = torch.bincount(dst, minlength=int(num_nodes))
    hist = torch.bincount(node_degrees.cpu(), minlength=int(node_degrees.max().item()) + 1)
    return hist.clamp_min(1).to(device)


def _resolve_ckpt_path(run_dir: str, ckpt_path: str | None, ckpt_name: str | None) -> str:
    if ckpt_path:
        return ckpt_path
    if ckpt_name:
        return os.path.join(run_dir, ckpt_name)
    return os.path.join(run_dir, "best_checkpoint.pth")


def _resolve_threshold_search(cfg: dict, args: argparse.Namespace) -> Tuple[list[float], str]:
    threshold_search_cfg = cfg.get("threshold_search", {})
    lower = float(args.lower if args.lower is not None else threshold_search_cfg.get("lower", 0.1))
    upper = float(args.upper if args.upper is not None else threshold_search_cfg.get("upper", 0.9))
    n_trials = int(args.n_trials if args.n_trials is not None else threshold_search_cfg.get("n_trials", 30))
    metric = args.threshold_metric if args.threshold_metric else threshold_search_cfg.get("metric", "val/illicit_f1")

    if "/" not in str(metric):
        metric = f"val/{metric}"

    if n_trials < 2:
        thresholds = [float(lower)]
    else:
        thresholds = np.linspace(lower, upper, n_trials).tolist()
    return thresholds, metric


def _flatten_metrics(prefix: str, cls_metrics: dict) -> dict:
    return {
        f"{prefix}/illicit_precision": cls_metrics["illicit"]["precision"],
        f"{prefix}/illicit_recall": cls_metrics["illicit"]["recall"],
        f"{prefix}/illicit_f1": cls_metrics["illicit"]["f1"],
        f"{prefix}/illicit_pr_auc": cls_metrics["illicit"]["pr_auc"],
        f"{prefix}/illicit_roc_auc": cls_metrics["illicit"]["roc_auc"],
        f"{prefix}/licit_precision": cls_metrics["licit"]["precision"],
        f"{prefix}/licit_recall": cls_metrics["licit"]["recall"],
        f"{prefix}/licit_f1": cls_metrics["licit"]["f1"],
        f"{prefix}/licit_pr_auc": cls_metrics["licit"]["pr_auc"],
        f"{prefix}/licit_roc_auc": cls_metrics["licit"]["roc_auc"],
    }


def _get_metric_value(metrics: dict, key: str) -> float:
    if key not in metrics:
        available = ", ".join(sorted(metrics.keys()))
        raise KeyError(f"threshold_metric '{key}' not found. Available: {available}")
    return float(metrics[key])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-training threshold search and test evaluation")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory containing config.yaml")
    parser.add_argument("--config", type=str, default=None, help="Override config path (default: run-dir/config.yaml)")
    parser.add_argument("--ckpt-path", type=str, default=None, help="Path to checkpoint (.pth)")
    parser.add_argument("--ckpt-name", type=str, default=None, help="Checkpoint filename inside run-dir")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda (default from config)")
    parser.add_argument("--threshold-metric", type=str, default=None, help="Metric to maximize, e.g. val/illicit_f1")
    parser.add_argument("--lower", type=float, default=None, help="Lower bound for threshold search")
    parser.add_argument("--upper", type=float, default=None, help="Upper bound for threshold search")
    parser.add_argument("--n-trials", type=int, default=None, help="Number of thresholds to try")
    parser.add_argument("--output-json", type=str, default=None, help="Output json path (default: run-dir/test_metrics.json)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = args.config or os.path.join(args.run_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = load_config(config_path)
    if cfg is None:
        raise ValueError(f"Config is empty or invalid YAML: {config_path}")
    thresholds, threshold_metric = _resolve_threshold_search(cfg, args)

    device = args.device or (cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    if device not in {"cpu", "cuda"}:
        raise ValueError("Device must be 'cpu' or 'cuda'")

    illicit_label = int(cfg.get("dataset", {}).get("illicit_label", 1))

    logging.info(f"Using device: {device}")
    logging.info(f"Threshold metric: {threshold_metric}")

    ckpt_path = _resolve_ckpt_path(args.run_dir, args.ckpt_path, args.ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    logging.info("Loading dataset...")
    ds_cfg = cfg["dataset"]
    ds = init_dataset(
        ds_cfg["data_path"],
        ds_cfg["dataset_name"],
        ds_cfg["dataset_type"],
        ds_cfg["embedding_type"],
    )
    global_data, subgraphs_list = ds.load()

    train_sub, val_sub, test_sub = ds.split(
        subgraphs_list,
        ds_cfg.get("train_ratio", 0.7),
        ds_cfg.get("val_ratio", 0.15),
        ds_cfg.get("split_mode", "temporal"),
    )

    train_cumulative = train_sub
    val_cumulative = train_sub + val_sub
    test_cumulative = train_sub + val_sub + test_sub

    train_mask = torch.arange(len(train_sub), dtype=torch.long, device=device)
    val_mask = torch.arange(len(train_sub), len(train_sub) + len(val_sub), dtype=torch.long, device=device)
    test_mask = torch.arange(len(train_sub) + len(val_sub), len(test_cumulative), dtype=torch.long, device=device)

    global_data = global_data.to(device)
    train_cumulative = [sg.to(device) for sg in train_cumulative]
    val_cumulative = [sg.to(device) for sg in val_cumulative]
    test_cumulative = [sg.to(device) for sg in test_cumulative]

    node_agg, edge_agg = build_aggregators(cfg)
    node_agg, edge_agg = node_agg.to(device), edge_agg.to(device)

    def filter_global_edges(g_data, mask):
        if mask is None:
            return g_data
        filtered = g_data.clone()
        filtered.edge_index = g_data.edge_index[:, mask]
        if g_data.edge_attr is not None:
            filtered.edge_attr = g_data.edge_attr[mask]
        return filtered

    train_global = filter_global_edges(global_data, getattr(global_data, "edge_train_mask", None))
    val_global = filter_global_edges(global_data, getattr(global_data, "edge_val_mask", None))
    test_global = filter_global_edges(global_data, getattr(global_data, "edge_test_mask", None))

    train_summarizer = SubgraphSummarizer(train_global, train_cumulative, node_aggregator=node_agg, edge_aggregator=edge_agg)
    val_summarizer = SubgraphSummarizer(val_global, val_cumulative, node_aggregator=node_agg, edge_aggregator=edge_agg)
    test_summarizer = SubgraphSummarizer(test_global, test_cumulative, node_aggregator=node_agg, edge_aggregator=edge_agg)

    m_cfg = cfg["model"]
    pearl_cfg = dict(m_cfg["pearl"])
    pearl_cfg.pop("random_seed", None)
    model_cfg = ModelConfig(
        hidden_dim=m_cfg["hidden_dim"],
        pearl=PearlConfig(**pearl_cfg),
        fraudgt=FraudGTConfig(**m_cfg["fraudgt"]),
        pna=PNAConfig(**dict(m_cfg.get("pna", {}))),
        backbone=str(m_cfg.get("backbone", "fraudgt")).lower(),
        dropout=float(m_cfg.get("dropout", 0.0)),
    )
    pna_deg = None
    if str(m_cfg.get("backbone", "fraudgt")).lower() == "pna":
        with torch.no_grad():
            train_super_graph = train_summarizer.summarize()
        pna_deg = _degree_histogram(train_super_graph.edge_index, int(train_super_graph.num_nodes), device)
    seed = int(cfg.get("train", {}).get("seed", 42))
    model = HierarchicalSubgraphTransformer(model_cfg, seed=seed, pna_deg=pna_deg).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if "node_agg_state_dict" in ckpt:
        node_agg.load_state_dict(ckpt["node_agg_state_dict"])
    if "edge_agg_state_dict" in ckpt:
        edge_agg.load_state_dict(ckpt["edge_agg_state_dict"])

    model.eval()
    node_agg.eval()
    edge_agg.eval()

    with torch.no_grad():
        val_super_graph = val_summarizer.summarize()
        val_out = model(val_super_graph.x, val_super_graph.edge_index, val_super_graph.edge_attr)
        val_target = torch.as_tensor(val_super_graph.y, device=device).float()

        best_threshold = thresholds[0] if thresholds else 0.5
        best_score = -float("inf")

        for thr in thresholds:
            val_cls_metrics = _binary_metrics_by_class(
                val_out[val_mask],
                val_target[val_mask],
                thr,
                illicit_label=illicit_label,
            )
            flat_val_metrics = _flatten_metrics("val", val_cls_metrics)
            score = _get_metric_value(flat_val_metrics, threshold_metric)
            if score > best_score:
                best_score = score
                best_threshold = thr

        logging.info(
            f"Best threshold on validation ({threshold_metric}) = {best_threshold:.4f} (score={best_score:.6f})"
        )

        test_super_graph = test_summarizer.summarize()
        test_out = model(test_super_graph.x, test_super_graph.edge_index, test_super_graph.edge_attr)
        test_target = torch.as_tensor(test_super_graph.y, device=device).float()

        test_cls_metrics = _binary_metrics_by_class(
            test_out[test_mask],
            test_target[test_mask],
            best_threshold,
            illicit_label=illicit_label,
        )
        flat_test_metrics = _flatten_metrics("test", test_cls_metrics)

    logging.info(
        f"Test metrics | Illicit F1: {flat_test_metrics['test/illicit_f1']:.4f}, "
        f"Illicit PR-AUC: {flat_test_metrics['test/illicit_pr_auc']:.4f}, "
        f"Illicit ROC-AUC: {flat_test_metrics['test/illicit_roc_auc']:.4f}, "
        f"Licit F1: {flat_test_metrics['test/licit_f1']:.4f}"
    )

    out_path = args.output_json or os.path.join(args.run_dir, "test_metrics.json")
    payload = {
        "best_threshold": float(best_threshold),
        "best_threshold_metric": threshold_metric,
        "best_threshold_score": float(best_score),
        **flat_test_metrics,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    logging.info(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
