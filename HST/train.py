import argparse
import json
import logging
import math
import os
import random
import yaml
import torch
import torch.nn as nn
from torch.nn.parameter import UninitializedParameter
import wandb
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
import time

from datasets.utils import init_dataset
import aggregators
from summarize import SubgraphSummarizer, _mode_to_class_prefix
from models import HierarchicalSubgraphTransformer
from utils import _as_float, _as_int, _compute_class_weight_from_targets, setup_logging, _binary_metrics_by_class
from loaders import create_loaders
from configs.config import ModelConfig, PearlConfig, FraudGTConfig, PNAConfig

load_dotenv()
setup_logging()

torch.backends.cuda.matmul.allow_tf32 = True  # Default False in PyTorch 1.12+
torch.backends.cudnn.allow_tf32 = True  # Default True


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _configure_reproducibility(cfg: dict, seed: int) -> None:
    _seed_everything(seed)
    repro_cfg = cfg.get('train', {}).get('reproducibility', {})
    if not bool(repro_cfg.get('deterministic', False)):
        return

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    logging.info("Deterministic cuDNN mode enabled.")

    if bool(repro_cfg.get('use_deterministic_algorithms', False)):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
        logging.info("Strict deterministic PyTorch algorithms enabled.")


class ExponentialMovingAverage:
    def __init__(self, modules: dict[str, nn.Module], decay: float = 0.999):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must be between 0 and 1, got {decay}")
        self.modules = modules
        self.decay = float(decay)
        self.shadow = {name: self._clone_initialized_state(module) for name, module in modules.items()}
        self.num_updates = 0

    @staticmethod
    def _clone_value(value: torch.Tensor) -> torch.Tensor | None:
        if isinstance(value, UninitializedParameter):
            return None
        try:
            return value.detach().clone()
        except ValueError:
            return None

    @classmethod
    def _clone_initialized_state(cls, module: nn.Module) -> dict[str, torch.Tensor]:
        cloned = {}
        for key, value in module.state_dict().items():
            cloned_value = cls._clone_value(value)
            if cloned_value is not None:
                cloned[key] = cloned_value
        return cloned

    @torch.no_grad()
    def update(self) -> None:
        self.num_updates += 1
        for name, module in self.modules.items():
            module_state = module.state_dict()
            shadow_state = self.shadow[name]
            for key, value in module_state.items():
                cloned_value = self._clone_value(value)
                if cloned_value is None:
                    continue
                if not torch.is_floating_point(value):
                    shadow_state[key] = cloned_value
                    continue
                if key not in shadow_state:
                    shadow_state[key] = cloned_value
                    continue
                shadow_state[key].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self) -> dict[str, dict[str, torch.Tensor]]:
        backup = {name: self._clone_initialized_state(module) for name, module in self.modules.items()}
        for name, module in self.modules.items():
            module_state = module.state_dict()
            ema_state = {
                key: self._clone_value(value)
                for key, value in module_state.items()
            }
            ema_state = {
                key: value
                for key, value in ema_state.items()
                if value is not None
            }
            ema_state.update(self.shadow[name])
            module.load_state_dict(ema_state, strict=False)
        return backup

    @torch.no_grad()
    def restore(self, backup: dict[str, dict[str, torch.Tensor]]) -> None:
        for name, module in self.modules.items():
            module.load_state_dict(backup[name], strict=False)

    def state_dict(self) -> dict:
        return {
            'decay': self.decay,
            'num_updates': self.num_updates,
            'shadow': self.shadow,
        }

    def load_state_dict(self, state: dict) -> None:
        self.decay = float(state.get('decay', self.decay))
        self.num_updates = int(state.get('num_updates', 0))
        self.shadow = state['shadow']


class SuperGraphFeatureNormalizer:
    def __init__(
        self,
        x_mean: torch.Tensor | None = None,
        x_std: torch.Tensor | None = None,
        edge_mean: torch.Tensor | None = None,
        edge_std: torch.Tensor | None = None,
        clip: float | None = None,
        eps: float = 1.0e-6,
    ):
        self.x_mean = x_mean
        self.x_std = x_std
        self.edge_mean = edge_mean
        self.edge_std = edge_std
        self.clip = clip
        self.eps = eps

    @staticmethod
    def _fit_tensor(tensor: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
        clean = torch.nan_to_num(tensor.float(), nan=0.0, posinf=0.0, neginf=0.0)
        mean = clean.mean(dim=0, keepdim=True)
        std = clean.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
        return mean, std

    @classmethod
    def fit(cls, graph, cfg: dict) -> "SuperGraphFeatureNormalizer":
        norm_cfg = cfg.get('supergraph_normalization', cfg.get('normalization', {}))
        if not bool(norm_cfg.get('enabled', False)):
            return cls()

        eps = float(norm_cfg.get('eps', 1.0e-6))
        clip_value = norm_cfg.get('clip', 10.0)
        clip = None if clip_value is None else float(clip_value)
        normalize_x = bool(norm_cfg.get('x', norm_cfg.get('node_features', True)))
        normalize_edge_attr = bool(norm_cfg.get('edge_attr', norm_cfg.get('edge_features', True)))

        x_mean = x_std = edge_mean = edge_std = None
        if normalize_x and getattr(graph, 'x', None) is not None and graph.x.numel() > 0:
            x_mean, x_std = cls._fit_tensor(graph.x, eps)
        if (
            normalize_edge_attr
            and getattr(graph, 'edge_attr', None) is not None
            and graph.edge_attr.numel() > 0
        ):
            edge_mean, edge_std = cls._fit_tensor(graph.edge_attr, eps)

        logging.info(
            "Supergraph feature normalization | "
            f"enabled=True, x={x_mean is not None}, edge_attr={edge_mean is not None}, "
            f"clip={clip if clip is not None else 'none'}"
        )
        return cls(x_mean=x_mean, x_std=x_std, edge_mean=edge_mean, edge_std=edge_std, clip=clip, eps=eps)

    def enabled(self) -> bool:
        return self.x_mean is not None or self.edge_mean is not None

    def _transform_tensor(
        self,
        tensor: torch.Tensor,
        mean: torch.Tensor | None,
        std: torch.Tensor | None,
    ) -> torch.Tensor:
        out = torch.nan_to_num(tensor.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if mean is None or std is None:
            return out
        out = (out - mean.to(out.device)) / std.to(out.device)
        if self.clip is not None and self.clip > 0:
            out = out.clamp(min=-self.clip, max=self.clip)
        return out

    def transform(self, graph):
        if not self.enabled():
            return graph
        out = graph.clone()
        if getattr(out, 'x', None) is not None and self.x_mean is not None:
            out.x = self._transform_tensor(out.x, self.x_mean, self.x_std)
        if getattr(out, 'edge_attr', None) is not None and self.edge_mean is not None:
            out.edge_attr = self._transform_tensor(out.edge_attr, self.edge_mean, self.edge_std)
        return out

    def summarize(self, summarizer: "SubgraphSummarizer"):
        return self.transform(summarizer.summarize())


def _fit_supergraph_normalizer(cfg: dict, train_super_graph) -> SuperGraphFeatureNormalizer:
    return SuperGraphFeatureNormalizer.fit(train_super_graph, cfg)


def _summarize_normalized(summarizer: "SubgraphSummarizer", normalizer: SuperGraphFeatureNormalizer):
    return normalizer.summarize(summarizer)


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def build_aggregators(cfg):
    import inspect
    
    node_mode_raw = str(cfg['aggregators']['node_pooling_mode'])
    edge_mode_raw = str(cfg['aggregators']['edge_pooling_mode'])
    node_mode = _mode_to_class_prefix(node_mode_raw)
    edge_mode = _mode_to_class_prefix(edge_mode_raw)
    
    node_class = getattr(aggregators, f"{node_mode}NodeAggregator")
    edge_class = getattr(aggregators, f"{edge_mode}EdgeAggregator")
    
    # Prioritize a dedicated aggregator hidden_dim if specified
    agg_hidden_dim = _as_int(cfg.get('aggregators', {}).get('hidden_dim', 0), 'aggregators.hidden_dim')
    if agg_hidden_dim <= 0:
        agg_hidden_dim = _as_int(cfg['model'].get('hidden_dim', 64), 'model.hidden_dim')

    heads = _as_int(
        cfg.get('aggregators', {}).get('fraudgt', {}).get(
            'heads',
            cfg['model'].get('fraudgt', {}).get(
                'aggregator_heads',
                cfg['model'].get('fraudgt', {}).get('heads', 4),
            ),
        ),
        'aggregators.fraudgt.heads',
    )
    
    def _build(cls):
        sig = inspect.signature(cls.__init__)
        kwargs = {}
        specific_cfg = cfg.get('aggregators', {}).get(node_mode_raw.replace("-", "_"), {})
        if not isinstance(specific_cfg, dict):
            specific_cfg = {}
        if 'hidden_dim' in sig.parameters:
            kwargs['hidden_dim'] = agg_hidden_dim
        if 'heads' in sig.parameters:
            kwargs['heads'] = heads
        for name in sig.parameters:
            if name in {'self', 'hidden_dim', 'heads'}:
                continue
            if name in specific_cfg:
                kwargs[name] = specific_cfg[name]
        return cls(**kwargs)

    node_agg = _build(node_class)
    edge_agg = _build(edge_class)

    # Freeze clearly non-learnable choices to enable caching downstream
    non_learnable_nodes = {"MeanNodeAggregator", "MaxNodeAggregator", "StatsNodeAggregator"}
    if node_class.__name__ in non_learnable_nodes:
        for p in node_agg.parameters():
            p.requires_grad = False

    # Edge Stats/Mean aggregators are either parameterless or should be frozen
    non_learnable_edges = {"MeanEdgeAggregator", "StatsEdgeAggregator"}
    if edge_class.__name__ in non_learnable_edges:
        for p in edge_agg.parameters():
            p.requires_grad = False

    return node_agg, edge_agg

def load_data(cfg, device):
    # 1. Load Data
    logging.info("Loading dataset...")
    ds_cfg = cfg['dataset']
    ds = init_dataset(
        ds_cfg['data_path'], ds_cfg['dataset_name'],
        ds_cfg['dataset_type'], ds_cfg['embedding_type']
    )
    global_data, subgraphs_list = ds.load()

    # Get splits
    train_sub, val_sub, test_sub = ds.split(
        subgraphs_list,
        ds_cfg.get('train_ratio', 0.7),
        ds_cfg.get('val_ratio', 0.15),
        ds_cfg.get('split_mode', 'temporal'),
    )

    # Cumulative subgraph lists for Super-Graph Inductive protocol
    train_cumulative = train_sub
    val_cumulative = train_sub + val_sub
    test_cumulative = train_sub + val_sub + test_sub

    # Masks are now relative to the cumulative lists
    train_mask = torch.arange(len(train_sub), dtype=torch.long, device=device)
    val_mask = torch.arange(len(train_sub), len(train_sub) + len(val_sub), dtype=torch.long, device=device)
    test_mask = torch.arange(len(train_sub) + len(val_sub), len(test_cumulative), dtype=torch.long, device=device)

    global_data = global_data.to(device)
    # Ensure all subgraphs in splits are on device
    train_cumulative = [sg.to(device) for sg in train_cumulative]
    val_cumulative = [sg.to(device) for sg in val_cumulative]
    test_cumulative = [sg.to(device) for sg in test_cumulative]
    return global_data, train_cumulative, val_cumulative, test_cumulative, train_mask, val_mask, test_mask

def parse_args():
    parser = argparse.ArgumentParser(description="Train Hierarchical Subgraph Transformer")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config yaml")
    parser.add_argument("--resume", action="store_true", help="Resume from save_dir/run_name/last_checkpoint.pth")
    return parser.parse_args()

def _flatten_metrics(prefix: str, cls_metrics: dict) -> dict:
    return {
        f"{prefix}/illicit_precision": cls_metrics['illicit']['precision'],
        f"{prefix}/illicit_recall": cls_metrics['illicit']['recall'],
        f"{prefix}/illicit_f1": cls_metrics['illicit']['f1'],
        f"{prefix}/illicit_pr_auc": cls_metrics['illicit']['pr_auc'],
        f"{prefix}/illicit_roc_auc": cls_metrics['illicit']['roc_auc'],
        f"{prefix}/licit_precision": cls_metrics['licit']['precision'],
        f"{prefix}/licit_recall": cls_metrics['licit']['recall'],
        f"{prefix}/licit_f1": cls_metrics['licit']['f1'],
        f"{prefix}/licit_pr_auc": cls_metrics['licit']['pr_auc'],
        f"{prefix}/licit_roc_auc": cls_metrics['licit']['roc_auc'],
    }

def _get_metric_value(metrics: dict, key: str) -> float:
    if key not in metrics:
        available = ", ".join(sorted(metrics.keys()))
        raise KeyError(f"metric '{key}' not found. Available: {available}")
    return float(metrics[key])

def _resolve_thresholds(cfg: dict) -> tuple[list[float], str]:
    threshold_search_cfg = cfg.get('threshold_search', {})
    threshold_search_enabled = bool(threshold_search_cfg.get('enabled', False))
    if threshold_search_enabled:
        lower = float(threshold_search_cfg.get('lower', 0.1))
        upper = float(threshold_search_cfg.get('upper', 0.9))
        n_trials = int(threshold_search_cfg.get('n_trials', 30))
        thresholds = np.linspace(lower, upper, n_trials).tolist()
        threshold_search_metric = threshold_search_cfg.get('metric', 'illicit_f1')
    else:
        thresholds = cfg['train'].get('thresholds', [cfg['train'].get('threshold', 0.2)])
        if not isinstance(thresholds, list):
            thresholds = [thresholds]
        thresholds = [float(t) for t in thresholds]
        threshold_search_metric = cfg['train'].get('threshold_search_metric', 'val/illicit_f1')

    if '/' not in str(threshold_search_metric):
        threshold_search_metric = f"val/{threshold_search_metric}"

    return thresholds, str(threshold_search_metric)

def _filter_global_edges(g_data, mask):
    if mask is None:
        return g_data
    filtered = g_data.clone()
    filtered.edge_index = g_data.edge_index[:, mask]
    if g_data.edge_attr is not None:
        filtered.edge_attr = g_data.edge_attr[mask]
    return filtered

def _resolve_eval_node_batch_size(cfg: dict) -> int | None:
    value = cfg.get('train', {}).get('eval_node_batch_size', None)
    if value is None:
        value = cfg.get('aggregators', {}).get('eval_node_batch_size', None)
    if value is None:
        return None
    value = int(value)
    return value if value > 0 else None

def _build_pearl_config(cfg: dict) -> PearlConfig:
    pearl_cfg = dict(cfg['model']['pearl'])
    pearl_cfg.pop('random_seed', None)
    return PearlConfig(**pearl_cfg)

def _build_pna_config(cfg: dict) -> PNAConfig:
    pna_cfg = dict(cfg.get('model', {}).get('pna', {}))
    return PNAConfig(**pna_cfg)

def _build_degree_histogram(edge_index: torch.Tensor | None, num_nodes: int, device: str | torch.device) -> torch.Tensor:
    device = torch.device(device)
    if edge_index is None or edge_index.numel() == 0 or num_nodes <= 0:
        return torch.ones(1, dtype=torch.long, device=device)
    dst = edge_index[1].detach().to(device=device, dtype=torch.long)
    node_degrees = torch.bincount(dst, minlength=int(num_nodes))
    hist = torch.bincount(node_degrees.cpu(), minlength=int(node_degrees.max().item()) + 1)
    return hist.clamp_min(1).to(device)

def _build_model(cfg: dict, device: str, pna_deg: torch.Tensor | None = None):
    m_cfg = cfg['model']
    fraudgt_cfg = dict(m_cfg.get('fraudgt', {}))
    fraudgt_cfg.pop('aggregator_heads', None)
    hidden_dim = _as_int(m_cfg.get('hidden_dim', 128), 'model.hidden_dim')
    model_cfg = ModelConfig(
        hidden_dim=hidden_dim,
        pearl=_build_pearl_config(cfg),
        fraudgt=FraudGTConfig(**fraudgt_cfg),
        pna=_build_pna_config(cfg),
        backbone=str(m_cfg.get('backbone', 'fraudgt')).lower(),
        dropout=float(m_cfg.get('dropout', 0.0))
    )
    seed = int(cfg.get('train', {}).get('seed', 42))
    return HierarchicalSubgraphTransformer(model_cfg, seed=seed, pna_deg=pna_deg).to(device)

def _count_trainable_params(modules: list[nn.Module]) -> int:
    num_params = 0
    for module in modules:
        for p in module.parameters():
            if not p.requires_grad:
                continue
            try:
                num_params += p.numel()
            except ValueError:
                continue
    return num_params

def _build_lr_scheduler(optimizer, optim_cfg: dict, total_epochs: int, steps_per_epoch: int):
    scheduler_cfg = optim_cfg.get('lr_scheduler', optim_cfg.get('scheduler', {}))
    if not bool(scheduler_cfg.get('enabled', False)):
        return None

    scheduler_type = str(scheduler_cfg.get('type', 'warmup_cosine')).lower()
    if scheduler_type in {'plateau', 'reduce_on_plateau', 'reduce_lr_on_plateau'}:
        mode = str(scheduler_cfg.get('mode', 'max')).lower()
        factor = float(scheduler_cfg.get('factor', 0.7))
        patience = int(scheduler_cfg.get('patience', 6))
        min_lr = float(scheduler_cfg.get('min_lr', 1.0e-6))
        threshold = float(scheduler_cfg.get('threshold', 1.0e-4))
        threshold_mode = str(scheduler_cfg.get('threshold_mode', 'rel')).lower()
        cooldown = int(scheduler_cfg.get('cooldown', 0))
        eps = float(scheduler_cfg.get('eps', 1.0e-8))
        logging.info(
            "Using LR scheduler | "
            f"type=plateau, mode={mode}, factor={factor:g}, patience={patience}, min_lr={min_lr:g}"
        )
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode=threshold_mode,
            cooldown=cooldown,
            min_lr=min_lr,
            eps=eps,
        )

    if scheduler_type not in {'warmup_cosine', 'cosine'}:
        raise ValueError(f"Unsupported optim.lr_scheduler.type: {scheduler_type}")

    steps_per_epoch = max(1, int(steps_per_epoch))
    total_steps = max(1, int(total_epochs) * steps_per_epoch)
    warmup_epochs = float(scheduler_cfg.get('warmup_epochs', 0 if scheduler_type == 'cosine' else 5))
    warmup_steps = max(0, int(round(warmup_epochs * steps_per_epoch)))
    min_lr_factor = float(scheduler_cfg.get('min_lr_factor', 0.1))
    min_lr_factor = min(max(min_lr_factor, 0.0), 1.0)

    def lr_lambda(step: int) -> float:
        step = max(0, int(step))
        if warmup_steps > 0 and step < warmup_steps:
            return max(1.0 / warmup_steps, float(step + 1) / warmup_steps)
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, float(step - warmup_steps) / decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_factor + (1.0 - min_lr_factor) * cosine

    logging.info(
        "Using LR scheduler | "
        f"type={scheduler_type}, warmup_epochs={warmup_epochs:g}, "
        f"min_lr_factor={min_lr_factor:g}, steps_per_epoch={steps_per_epoch}"
    )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _is_plateau_scheduler(lr_scheduler) -> bool:
    return isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

def evaluate_checkpoint(args, preloaded_data=None, checkpoint_path: str | None = None):
    cfg = load_config(args.config)

    seed = int(cfg['train'].get('seed', 42))
    _configure_reproducibility(cfg, seed)

    training_threshold = float(cfg['train'].get('training_threshold', 0.5))
    illicit_label = int(cfg.get('dataset', {}).get('illicit_label', 0))
    thresholds, threshold_search_metric = _resolve_thresholds(cfg)
    run_threshold_search = bool(cfg.get('train', {}).get('run_threshold_search', True))
    run_test = bool(cfg.get('train', {}).get('run_test', True))

    device = cfg['train']['device'] if torch.cuda.is_available() else "cpu"
    assert device in ["cpu", "cuda"], "Device must be 'cpu' or 'cuda'"
    logging.info(f"Using device: {device}")

    if preloaded_data is None:
        global_data, train_cumulative, val_cumulative, test_cumulative, train_mask, val_mask, test_mask = load_data(cfg, device)
    else:
        global_data, train_cumulative, val_cumulative, test_cumulative, train_mask, val_mask, test_mask = preloaded_data

    logging.info("Initializing aggregators...")
    node_agg, edge_agg = build_aggregators(cfg)
    node_agg, edge_agg = node_agg.to(device), edge_agg.to(device)
    eval_node_batch_size = _resolve_eval_node_batch_size(cfg)

    train_global = _filter_global_edges(global_data, getattr(global_data, 'edge_train_mask', None))
    val_global = _filter_global_edges(global_data, getattr(global_data, 'edge_val_mask', None))
    test_global = _filter_global_edges(global_data, getattr(global_data, 'edge_test_mask', None))
    train_summarizer = None
    if str(cfg.get('model', {}).get('backbone', 'fraudgt')).lower() == 'pna':
        train_summarizer = SubgraphSummarizer(
            train_global,
            train_cumulative,
            node_aggregator=node_agg,
            edge_aggregator=edge_agg,
            node_batch_size=eval_node_batch_size,
        )
    val_summarizer = SubgraphSummarizer(
        val_global,
        val_cumulative,
        node_aggregator=node_agg,
        edge_aggregator=edge_agg,
        node_batch_size=eval_node_batch_size,
    )
    test_summarizer = SubgraphSummarizer(
        test_global,
        test_cumulative,
        node_aggregator=node_agg,
        edge_aggregator=edge_agg,
        node_batch_size=eval_node_batch_size,
    )

    logging.info("Initializing model...")
    pna_deg = None
    supergraph_normalizer = SuperGraphFeatureNormalizer()
    if train_summarizer is not None:
        with torch.no_grad():
            train_super_graph = train_summarizer.summarize()
        pna_deg = _build_degree_histogram(
            train_super_graph.edge_index,
            int(train_super_graph.num_nodes),
            device,
        )
        supergraph_normalizer = _fit_supergraph_normalizer(cfg, train_super_graph)
    model = _build_model(cfg, device, pna_deg=pna_deg)

    train_targets_for_weight = torch.stack([
        torch.as_tensor(y, device=device).float().view(-1)[0]
        for y in (sg.y for sg in train_cumulative)
    ])
    calulated_pos_weight = _compute_class_weight_from_targets(train_targets_for_weight, device=device)
    logging.info(f'pos weight computed from train split: {calulated_pos_weight.item():.6f}')

    cfg_pos_weight = cfg.get('train', {}).get('pos_weight', None)
    logging.info(f"train.pos_weight from config: {cfg_pos_weight}")

    pos_weight = torch.as_tensor(cfg['train'].get('pos_weight', calulated_pos_weight.item()), device=device).float()
    logging.info(f"Final pos_weight for BCEWithLogitsLoss: {pos_weight.item():.6f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    save_dir = cfg['logging']['save_dir']
    run_name = cfg['logging']['run_name']
    run_dir = os.path.join(save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    resolved_cfg_path = os.path.join(run_dir, 'config.yaml')
    with open(resolved_cfg_path, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    logging.info(f"Saved evaluation config to {resolved_cfg_path}")

    if checkpoint_path is None:
        checkpoint_path = cfg.get('train', {}).get('checkpoint_path', None)
    if checkpoint_path is None:
        checkpoint_path = os.path.join(run_dir, 'best_checkpoint.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found for evaluation: {checkpoint_path}")

    logging.info(f"Loading checkpoint for evaluation: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    node_agg.load_state_dict(ckpt['node_agg_state_dict'])
    edge_agg.load_state_dict(ckpt['edge_agg_state_dict'])

    use_wandb = bool(cfg['logging'].get('use_wandb', False))
    run = None
    if use_wandb:
        api_key = os.getenv("WANDB_API_KEY")
        if not api_key:
            raise ValueError("WANDB_API_KEY environment variable is not set. Please export it in your SLURM script, or set use_wandb: False in your config.")
        wandb.login(key=api_key)
        run = wandb.init(
            entity="hierarchical-subgraph-transformer",
            project=cfg['logging']['wandb_project'],
            config=cfg,
            name=run_name,
        )

    model.eval()
    node_agg.eval()
    edge_agg.eval()

    monitor_metric = cfg.get('logging', {}).get('monitor_metric', 'val/illicit_pr_auc')
    checkpoint_best_metric = ckpt.get('best_metric', None)
    best_threshold = float(training_threshold)
    best_thr_value = None
    flat_val_metrics: dict = {}
    flat_test_metrics: dict = {}
    test_loss = None

    with torch.no_grad():
        val_super_graph = _summarize_normalized(val_summarizer, supergraph_normalizer)
        val_out = model(val_super_graph.x, val_super_graph.edge_index, val_super_graph.edge_attr)
        val_target = torch.as_tensor(val_super_graph.y, device=device).float()

        if run_threshold_search:
            best_threshold = thresholds[0] if len(thresholds) > 0 else float(training_threshold)
            best_thr_value = -float('inf')

            for thr in thresholds:
                val_cls_metrics = _binary_metrics_by_class(
                    val_out[val_mask],
                    val_target[val_mask],
                    thr,
                    illicit_label=illicit_label,
                )
                cur_val_metrics = _flatten_metrics('val', val_cls_metrics)
                metric_value = _get_metric_value(cur_val_metrics, threshold_search_metric)
                if metric_value > best_thr_value:
                    best_thr_value = metric_value
                    best_threshold = float(thr)
                    flat_val_metrics = cur_val_metrics

            logging.info(
                f"Best threshold on validation ({threshold_search_metric}) = {best_threshold:.4f} "
                f"(score={best_thr_value:.4f})"
            )
        else:
            val_cls_metrics = _binary_metrics_by_class(
                val_out[val_mask],
                val_target[val_mask],
                best_threshold,
                illicit_label=illicit_label,
            )
            flat_val_metrics = _flatten_metrics('val', val_cls_metrics)
            logging.info(f"Using configured threshold for evaluation: {best_threshold:.4f}")

        if run_test:
            super_graph = _summarize_normalized(test_summarizer, supergraph_normalizer)
            out = model(super_graph.x, super_graph.edge_index, super_graph.edge_attr)
            target = torch.as_tensor(super_graph.y, device=device).float()

            test_loss = criterion(out[test_mask], target[test_mask]).item()
            test_cls_metrics = _binary_metrics_by_class(
                out[test_mask],
                target[test_mask],
                best_threshold,
                illicit_label=illicit_label,
            )
            flat_test_metrics = _flatten_metrics('test', test_cls_metrics)

    if run_test:
        logging.info(
            f"Test metrics | Loss: {test_loss:.4f}, "
            f"Illicit F1: {flat_test_metrics['test/illicit_f1']:.4f}, "
            f"Illicit PR-AUC: {flat_test_metrics['test/illicit_pr_auc']:.4f}, "
            f"Illicit ROC-AUC: {flat_test_metrics['test/illicit_roc_auc']:.4f}, "
            f"Licit F1: {flat_test_metrics['test/licit_f1']:.4f}"
        )

    num_params = _count_trainable_params([model, node_agg, edge_agg])
    eval_metrics = {
        'checkpoint_path': checkpoint_path,
        'checkpoint_epoch': int(ckpt.get('epoch', -1)),
        'checkpoint_best_metric': float(checkpoint_best_metric) if checkpoint_best_metric is not None else None,
        'best_threshold': float(best_threshold),
        'threshold_search_metric': threshold_search_metric,
        'threshold_search_score': float(best_thr_value) if best_thr_value is not None else None,
        'test_loss': float(test_loss) if test_loss is not None else None,
        'num_params': int(num_params),
        **flat_val_metrics,
        **flat_test_metrics,
    }

    metrics_path = os.path.join(run_dir, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(eval_metrics, f, indent=2)
    logging.info(f"Saved checkpoint evaluation metrics to {metrics_path}")

    if run is not None:
        wandb_metrics = {
            key: value for key, value in eval_metrics.items()
            if isinstance(value, (int, float)) and value is not None
        }
        run.log(wandb_metrics)
        run.finish()

    if checkpoint_best_metric is not None:
        metric_value = float(checkpoint_best_metric)
    elif monitor_metric in flat_val_metrics:
        metric_value = float(flat_val_metrics[monitor_metric])
    else:
        metric_value = float('nan')

    print("\n")
    return metric_value, num_params

def main(args, preloaded_data=None):
    cfg = load_config(args.config)

    seed = int(cfg['train'].get('seed', 42))
    _configure_reproducibility(cfg, seed)
    
    training_threshold = float(cfg['train'].get('training_threshold', 0.5))
    illicit_label = int(cfg.get('dataset', {}).get('illicit_label', 0))
    summarize_once_per_epoch = bool(cfg['train'].get('summarize_once_per_epoch', False))

    threshold_search_cfg = cfg.get('threshold_search', {})
    threshold_search_enabled = bool(threshold_search_cfg.get('enabled', False))
    run_threshold_search = bool(cfg.get('train', {}).get('run_threshold_search', True))
    run_test = bool(cfg.get('train', {}).get('run_test', True))
    if threshold_search_enabled:
        lower = float(threshold_search_cfg.get('lower', 0.1))
        upper = float(threshold_search_cfg.get('upper', 0.9))
        n_trials = int(threshold_search_cfg.get('n_trials', 30))
        thresholds = np.linspace(lower, upper, n_trials).tolist()
        threshold_search_metric = threshold_search_cfg.get('metric', 'illicit_f1')
    else:
        thresholds = cfg['train'].get('thresholds', [cfg['train'].get('threshold', 0.2)])
        if not isinstance(thresholds, list):
            thresholds = [thresholds]
        thresholds = [float(t) for t in thresholds]
        threshold_search_metric = cfg['train'].get('threshold_search_metric', 'val/illicit_f1')

    if '/' not in str(threshold_search_metric):
        threshold_search_metric = f"val/{threshold_search_metric}"
    
    device = cfg['train']['device'] if torch.cuda.is_available() else "cpu"
    assert device in ["cpu", "cuda"], "Device must be 'cpu' or 'cuda'"
    logging.info(f"Using device: {device}")

    if preloaded_data is None:
        global_data, train_cumulative, val_cumulative, test_cumulative, train_mask, val_mask, test_mask = load_data(cfg, device)
    else:
        global_data, train_cumulative, val_cumulative, test_cumulative, train_mask, val_mask, test_mask = preloaded_data

    # 2. Setup Aggregators & Summarizers (Temporal variants)
    logging.info("Initializing aggregators...")
    node_agg, edge_agg = build_aggregators(cfg)
    node_agg, edge_agg = node_agg.to(device), edge_agg.to(device)
    aggregators_trainable = any(p.requires_grad for p in node_agg.parameters()) or any(
        p.requires_grad for p in edge_agg.parameters()
    )
    eval_node_batch_size = _resolve_eval_node_batch_size(cfg)

    # this logs time distribution of the first summarization, which can help identify bottlenecks
    # node_agg.profile = True
    
    def filter_global_edges(g_data, mask):
        if mask is None:
            return g_data
        filtered = g_data.clone()
        filtered.edge_index = g_data.edge_index[:, mask]
        if g_data.edge_attr is not None:
            filtered.edge_attr = g_data.edge_attr[mask]
        return filtered

    train_global = filter_global_edges(global_data, getattr(global_data, 'edge_train_mask', None))
    val_global = filter_global_edges(global_data, getattr(global_data, 'edge_val_mask', None))
    test_global = filter_global_edges(global_data, getattr(global_data, 'edge_test_mask', None))
    
    # uses the same node_agg and edge_agg, so test / val summarizers will share weights with 
    # train summarizer (but operate on different global graphs)
    train_summarizer = SubgraphSummarizer(train_global, train_cumulative, node_aggregator=node_agg, 
                                          edge_aggregator=edge_agg, verbose=False)
    val_summarizer = SubgraphSummarizer(val_global, val_cumulative, node_aggregator=node_agg, 
                                        edge_aggregator=edge_agg, node_batch_size=eval_node_batch_size)
    test_summarizer = SubgraphSummarizer(test_global, test_cumulative, node_aggregator=node_agg, 
                                         edge_aggregator=edge_agg, node_batch_size=eval_node_batch_size)

    # 3. Setup Model
    logging.info("Initializing model...")
    initial_super_graph_for_pna = None
    pna_deg = None
    if str(cfg.get('model', {}).get('backbone', 'fraudgt')).lower() == 'pna':
        logging.info("Building train super-graph degree histogram for PNA layer-2 backbone...")
        with torch.no_grad():
            initial_super_graph_for_pna = train_summarizer.summarize()
        pna_deg = _build_degree_histogram(
            initial_super_graph_for_pna.edge_index,
            int(initial_super_graph_for_pna.num_nodes),
            device,
        )
        logging.info("PNA layer-2 degree histogram length: %d", int(pna_deg.numel()))

    with torch.no_grad():
        initial_super_graph_for_norm = (
            initial_super_graph_for_pna
            if initial_super_graph_for_pna is not None
            else train_summarizer.summarize()
        )
    supergraph_normalizer = _fit_supergraph_normalizer(cfg, initial_super_graph_for_norm)

    model = _build_model(cfg, device, pna_deg=pna_deg)
    
    optim_cfg = cfg.get('optim', {})
    opt_type = optim_cfg.get('optimizer', 'adam').lower()
    lr = _as_float(optim_cfg.get('base_lr', 0.001), 'optim.base_lr')
    aggregator_lr = _as_float(optim_cfg.get('aggregator_lr', lr), 'optim.aggregator_lr')
    weight_decay = _as_float(optim_cfg.get('weight_decay', 1e-4), 'optim.weight_decay')

    # Keep aggregator params in a separate group so we can tune their LR independently.
    model_params = [p for p in model.parameters() if p.requires_grad]
    node_agg_params = [p for p in node_agg.parameters() if p.requires_grad]
    edge_agg_params = [p for p in edge_agg.parameters() if p.requires_grad]
    aggregator_params = node_agg_params + edge_agg_params
    learnable_params = model_params + aggregator_params

    param_groups = []
    if model_params:
        param_groups.append({"params": model_params, "lr": lr})
    if aggregator_params:
        param_groups.append({"params": aggregator_params, "lr": aggregator_lr})

    if opt_type == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(param_groups, weight_decay=weight_decay)

    ema_cfg = cfg.get('train', {}).get('ema', {})
    ema_enabled = bool(ema_cfg.get('enabled', False))
    ema_decay = float(ema_cfg.get('decay', 0.999))
    ema_start_epoch = _as_int(ema_cfg.get('start_epoch', 1), 'train.ema.start_epoch')
    ema_update_every = max(1, _as_int(ema_cfg.get('update_every', 1), 'train.ema.update_every'))
    ema = None
    if ema_enabled:
        ema = ExponentialMovingAverage(
            {'model': model, 'node_agg': node_agg, 'edge_agg': edge_agg},
            decay=ema_decay,
        )
        logging.info(
            "Using EMA weights | "
            f"decay={ema_decay:g}, start_epoch={ema_start_epoch}, update_every={ema_update_every}"
        )

    def _safe_num_params(params: list[torch.nn.Parameter]) -> int:
        total = 0
        for p in params:
            try:
                total += p.numel()
            except ValueError:
                continue
        return total

    logging.info(
        "Optimizer LRs | "
        f"model: {lr:.6g}, aggregators: {aggregator_lr:.6g}, "
        f"model params: {_safe_num_params(model_params)}, "
        f"aggregator params: {_safe_num_params(aggregator_params)}"
    )

    # Dynamic class weighting from train split labels
    train_targets_for_weight = torch.stack([
        torch.as_tensor(y, device=device).float().view(-1)[0]
        for y in (sg.y for sg in train_cumulative)
    ])
    calulated_pos_weight = _compute_class_weight_from_targets(train_targets_for_weight, device=device)
    logging.info(f'pos weight computed from train split: {calulated_pos_weight.item():.6f}')

    cfg_pos_weight = cfg.get('train', {}).get('pos_weight', None)
    logging.info(f"train.pos_weight from config: {cfg_pos_weight}")

    pos_weight = torch.as_tensor(cfg['train'].get('pos_weight', calulated_pos_weight.item()), device=device).float()
    logging.info(f"Final pos_weight for BCEWithLogitsLoss: {pos_weight.item():.6f}")

    # Good criterion for binary classification with class imbalance
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    save_dir = cfg['logging']['save_dir']
    run_name = cfg['logging']['run_name']
    run_dir = os.path.join(save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Persist resolved training config for reproducibility
    resolved_cfg_path = os.path.join(run_dir, 'config.yaml')
    with open(resolved_cfg_path, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    logging.info(f"Saved run config to {resolved_cfg_path}")
    
    epochs = _as_int(optim_cfg.get('max_epoch', cfg['train'].get('epochs', 100)), 'optim.max_epoch')
    min_epoch = _as_int(optim_cfg.get('min_epoch', 0), 'optim.min_epoch')
    ckpt_period = _as_int(cfg['logging'].get('ckpt_period', 0), 'logging.ckpt_period')
    patience = _as_int(optim_cfg.get('patience', 10), 'optim.patience')
    
    monitor_metric = cfg.get('logging', {}).get('monitor_metric', 'val/illicit_pr_auc')
    monitor_mode = cfg.get('logging', {}).get('monitor_mode', None)
    if monitor_mode is None:
        monitor_mode = 'min' if 'loss' in monitor_metric.lower() else 'max'
    monitor_mode = monitor_mode.lower()
    if monitor_mode not in {'min', 'max'}:
        raise ValueError(f"logging.monitor_mode must be 'min' or 'max', got: {monitor_mode}")

    best_metric = float('inf') if monitor_mode == 'min' else -float('inf')
    patience_counter = 0
    start_epoch = 1
    last_epoch = 0
    global_step = 0
    resume_ckpt = None
    
    USE_WANDB = cfg['logging'].get('use_wandb', False)
    USE_CHECKPOINTS = cfg['logging'].get('use_checkpoints', False)

    if USE_WANDB:
        api_key = os.getenv("WANDB_API_KEY")
        if not api_key:
            raise ValueError("WANDB_API_KEY environment variable is not set. Please export it in your SLURM script, or set use_wandb: False in your config.")
        wandb.login(key=api_key)

        run = wandb.init(
            entity="hierarchical-subgraph-transformer",
            project=cfg['logging']['wandb_project'], 
            config=cfg,
            name=run_name,
        )

    if args.resume:
        resume_path = os.path.join(run_dir, 'last_checkpoint.pth')
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"--resume was set but checkpoint not found: {resume_path}")

        logging.info(f"Resuming from checkpoint: {resume_path}")
        resume_ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(resume_ckpt['model_state_dict'])
        node_agg.load_state_dict(resume_ckpt['node_agg_state_dict'])
        edge_agg.load_state_dict(resume_ckpt['edge_agg_state_dict'])
        optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
        if ema is not None and resume_ckpt.get('ema_state_dict') is not None:
            ema.load_state_dict(resume_ckpt['ema_state_dict'])
        start_epoch = int(resume_ckpt.get('epoch', 0)) + 1
        best_metric = float(resume_ckpt.get('best_metric', best_metric))
        patience_counter = int(resume_ckpt.get('patience_counter', 0))
        global_step = int(resume_ckpt.get('global_step', 0))
        logging.info(f"Resume state: start_epoch={start_epoch}, best_metric={best_metric:.6f}, patience_counter={patience_counter}")
    
    use_amp = bool(cfg['train'].get('use_amp', False)) and device == "cuda"
    amp = getattr(torch, "amp", None)
    
    if amp is not None and hasattr(amp, "GradScaler"):
        scaler = amp.GradScaler(device, enabled=use_amp)
        autocast_ctx = lambda: amp.autocast(device, enabled=use_amp)
    elif hasattr(torch, "GradScaler"):
        scaler = torch.GradScaler(device, enabled=use_amp)
        autocast_ctx = lambda: torch.autocast(device, enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        autocast_ctx = lambda: torch.cuda.amp.autocast(enabled=use_amp)

    logging.info("Building dataloaders...")

    # Pre-build Dummy Dataloaders for structural sampling
    with torch.no_grad():
        initial_super_graph = initial_super_graph_for_pna if initial_super_graph_for_pna is not None else train_summarizer.summarize()
        dummy_graph = initial_super_graph.clone()
        dummy_graph.x = torch.empty(0)
        dummy_graph.edge_attr = None
        dummy_graph = dummy_graph.detach().cpu()
        
    train_loader, _, _ = create_loaders(dummy_graph, train_mask.cpu(), val_mask.cpu(), test_mask.cpu(), cfg, 'cpu')
    balanced_sampling_cfg = cfg.get('train', {}).get('balanced_sampling', {})
    resample_balanced_each_epoch = bool(
        cfg['train'].get('mini_batch', False)
        and balanced_sampling_cfg.get('enabled', False)
        and balanced_sampling_cfg.get('resample_each_epoch', False)
    )
    if resample_balanced_each_epoch:
        logging.info("Balanced train sampling will be resampled each epoch using train.seed + epoch_index.")

    if cfg['train'].get('num_threads', -1) >= 0:
        torch.set_num_threads(cfg['train'].get('num_threads', 1))
    iter_per_epoch = cfg['train'].get('iter_per_epoch', None)
    eval_period = cfg['train'].get('eval_period', 1)
    grad_accum_steps = max(1, _as_int(optim_cfg.get('grad_accum_steps', 1), 'optim.grad_accum_steps'))
    if iter_per_epoch:
        train_batches_per_epoch = min(int(iter_per_epoch), len(train_loader))
    else:
        train_batches_per_epoch = len(train_loader)
    steps_per_epoch = max(1, math.ceil(train_batches_per_epoch / grad_accum_steps))
    lr_scheduler = _build_lr_scheduler(optimizer, optim_cfg, epochs, steps_per_epoch)
    if grad_accum_steps > 1:
        logging.info(
            "Using gradient accumulation | "
            f"grad_accum_steps={grad_accum_steps}, train_batches_per_epoch={train_batches_per_epoch}, "
            f"optimizer_steps_per_epoch={steps_per_epoch}"
        )
    if (
        lr_scheduler is not None
        and resume_ckpt is not None
        and resume_ckpt.get('lr_scheduler_state_dict') is not None
    ):
        lr_scheduler.load_state_dict(resume_ckpt['lr_scheduler_state_dict'])

    logging.info("Starting training...")

    def _flatten_metrics(prefix: str, cls_metrics: dict) -> dict:
        return {
            f"{prefix}/illicit_precision": cls_metrics['illicit']['precision'],
            f"{prefix}/illicit_recall": cls_metrics['illicit']['recall'],
            f"{prefix}/illicit_f1": cls_metrics['illicit']['f1'],
            f"{prefix}/illicit_pr_auc": cls_metrics['illicit']['pr_auc'],
            f"{prefix}/illicit_roc_auc": cls_metrics['illicit']['roc_auc'],
            f"{prefix}/licit_precision": cls_metrics['licit']['precision'],
            f"{prefix}/licit_recall": cls_metrics['licit']['recall'],
            f"{prefix}/licit_f1": cls_metrics['licit']['f1'],
            f"{prefix}/licit_pr_auc": cls_metrics['licit']['pr_auc'],
            f"{prefix}/licit_roc_auc": cls_metrics['licit']['roc_auc'],
        }

    def _get_metric_value(metrics: dict, key: str) -> float:
        if key not in metrics:
            available = ", ".join(sorted(metrics.keys()))
            raise KeyError(f"monitor_metric '{key}' not found. Available: {available}")
        return float(metrics[key])

    for epoch in range(start_epoch, epochs + 1):
        if resample_balanced_each_epoch:
            train_loader, _, _ = create_loaders(
                dummy_graph,
                train_mask.cpu(),
                val_mask.cpu(),
                test_mask.cpu(),
                cfg,
                'cpu',
                balanced_seed_offset=epoch - 1,
            )

        model.train()
        node_agg.train()
        edge_agg.train()
        
        total_loss = 0

        # Compute summarization once per epoch if enabled (or when cache is enabled)
        if summarize_once_per_epoch or getattr(train_summarizer, 'cache_enabled', False):
            start = time.time()
            super_graph = _summarize_normalized(train_summarizer, supergraph_normalizer)
            end = time.time()
            # logging.info(f"Summarization time for epoch {epoch}: {end - start:.2f}s")

            if summarize_once_per_epoch and aggregators_trainable:
                # Detach to avoid reusing the same autograd graph across batches.
                if super_graph.x is not None:
                    super_graph.x = super_graph.x.detach()
                if getattr(super_graph, 'edge_attr', None) is not None:
                    super_graph.edge_attr = super_graph.edge_attr.detach()  # type: ignore[assignment]

        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}"), start=1):

            summarize_start = time.time()
            # Recompute only when not using per-epoch summarize and cache is disabled
            if (not summarize_once_per_epoch) and (not getattr(train_summarizer, 'cache_enabled', False)):
                super_graph = _summarize_normalized(train_summarizer, supergraph_normalizer)
            summarize_end = time.time()
            
            forward_start = time.time()
            # Unpack depending on batch layout (LinkNeighborLoader vs Full Batch)
            if isinstance(batch, tuple):
                cur_graph, b_mask = batch
                b_out = model(cur_graph.x, cur_graph.edge_index, cur_graph.edge_attr)
                target = torch.as_tensor(cur_graph.y, device=device).float() # type: ignore
                loss = criterion(b_out[b_mask], target[b_mask])
            else:
                cur_graph = batch.to(device)
                
                # Fetch Real GPU features with Gradients dynamically
                cur_graph.x = super_graph.x[cur_graph.n_id] # type: ignore
                if super_graph.edge_attr is not None and hasattr(cur_graph, 'e_id'):
                    cur_graph.edge_attr = super_graph.edge_attr[cur_graph.e_id] # type: ignore
                else:
                    cur_graph.edge_attr = None
                    
                b_out = model(cur_graph.x, cur_graph.edge_index, cur_graph.edge_attr)
                target = torch.as_tensor(cur_graph.y, device=device).float() # type: ignore
                # Use batch output directly. Masks vary depending on setup. Assuming full evaluation natively if disjoint
                loss = criterion(b_out[:cur_graph.batch_size], target[:cur_graph.batch_size])

            forward_end = time.time()
            # logging.info(f"Total Time: {forward_end - summarize_start:.2f}s | Summarize Time: {summarize_end - summarize_start:.2f}s | Forward Time: {forward_end - forward_start:.2f}s")

            with autocast_ctx():
                pass # Already forwarded

            loss_for_backward = loss / grad_accum_steps
            scaler.scale(loss_for_backward).backward()
            total_loss += loss.item()

            end_of_epoch_batches = bool(iter_per_epoch and i >= iter_per_epoch)
            should_step = (i % grad_accum_steps == 0) or end_of_epoch_batches

            if should_step:
                if optim_cfg.get('clip_grad_norm', False):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        learnable_params,
                        _as_float(optim_cfg.get('max_grad_norm', 1.0), 'optim.max_grad_norm'),
                    )

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if lr_scheduler is not None and not _is_plateau_scheduler(lr_scheduler):
                    lr_scheduler.step()
                if ema is not None and epoch >= ema_start_epoch and global_step % ema_update_every == 0:
                    ema.update()

            if iter_per_epoch and i >= iter_per_epoch:
                # logging.info(f"Reached iter_per_epoch={iter_per_epoch} at iteration {i}. Ending epoch early.")
                break

        last_epoch = epoch
        avg_train_loss = total_loss / max(1, i)
        
        if epoch % eval_period == 0:
            model.eval()
            node_agg.eval()
            edge_agg.eval()
            ema_backup = ema.apply_to() if ema is not None and ema.num_updates > 0 else None
            with torch.no_grad():
                val_super_graph = _summarize_normalized(val_summarizer, supergraph_normalizer)
                
                # Simple full batch metrics fallback for now (can use val_loader for mini-batch evaluation if needed)
                val_out = model(val_super_graph.x, val_super_graph.edge_index, val_super_graph.edge_attr)
                val_target = torch.as_tensor(val_super_graph.y, device=device).float()

                val_loss = criterion(val_out[val_mask], val_target[val_mask]).item()
                # For simplicity using the full super_graph for metrics
                train_eval_super_graph = (
                    _summarize_normalized(train_summarizer, supergraph_normalizer)
                    if ema_backup is not None
                    else super_graph
                )
                train_out = model(
                    train_eval_super_graph.x,
                    train_eval_super_graph.edge_index,
                    train_eval_super_graph.edge_attr,
                )
                train_target = torch.as_tensor(train_eval_super_graph.y, device=device).float()
                
                train_cls_metrics = _binary_metrics_by_class(
                    train_out[train_mask],
                    train_target[train_mask],
                    training_threshold,
                    illicit_label=illicit_label,
                )
                val_cls_metrics = _binary_metrics_by_class(
                    val_out[val_mask],
                    val_target[val_mask],
                    training_threshold,
                    illicit_label=illicit_label,
                )

                flat_train_metrics = _flatten_metrics('train', train_cls_metrics)
                flat_val_metrics = _flatten_metrics('val', val_cls_metrics)
                metrics = {
                    **flat_train_metrics,
                    **flat_val_metrics,
                    'train/loss': float(avg_train_loss),
                    'val/loss': float(val_loss),
                }

                monitor_value = _get_metric_value(metrics, monitor_metric)
                if lr_scheduler is not None and _is_plateau_scheduler(lr_scheduler):
                    lr_scheduler.step(monitor_value)
            if ema is not None and ema_backup is not None:
                ema.restore(ema_backup)
            model.train()
            node_agg.train()
            edge_agg.train()

        if epoch % eval_period == 0:
            if epoch % cfg['logging'].get('log_interval', 10) == 0:
                logging.info(
                    f"Epoch [{epoch}/{epochs}], "
                    f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                    f"Train licit F1: {flat_train_metrics['train/licit_f1']:.4f}, Val Licit F1: {flat_val_metrics['val/licit_f1']:.4f}, "
                    f"Train Illicit F1: {flat_train_metrics['train/illicit_f1']:.4f}, Val Illicit F1: {flat_val_metrics['val/illicit_f1']:.4f}, "
                    f"Val Illicit PR-AUC: {flat_val_metrics['val/illicit_pr_auc']:.4f}, Val Illicit ROC-AUC: {flat_val_metrics['val/illicit_roc_auc']:.4f}, "
                    f"Monitor({monitor_metric}): {monitor_value:.4f}"
                )

            if USE_WANDB:
                run.log({
                    **flat_train_metrics,
                    **flat_val_metrics,
                    'train/loss': avg_train_loss,
                    'val/loss': val_loss,
                    'val/monitor_value': monitor_value,
                    'optim/lr_model': optimizer.param_groups[0]['lr'],
                    'optim/lr_aggregators': optimizer.param_groups[-1]['lr'],
                    'epoch': epoch
                })

            # Early Stopping Logic (based on monitor metric)
            is_improved = (monitor_value < best_metric) if monitor_mode == 'min' else (monitor_value > best_metric)
            if is_improved:
                best_metric = monitor_value
                patience_counter = 0
                # Save best full checkpoint
                best_path = os.path.join(run_dir, 'best_checkpoint.pth')
                ema_checkpoint_backup = ema.apply_to() if ema is not None and ema.num_updates > 0 else None
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'node_agg_state_dict': node_agg.state_dict(),
                    'edge_agg_state_dict': edge_agg.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler is not None else None,
                    'ema_state_dict': ema.state_dict() if ema is not None else None,
                    'used_ema_weights': bool(ema_checkpoint_backup is not None),
                    'global_step': global_step,
                    'loss': loss.item(),
                    'best_metric': best_metric,
                    'monitor_metric': monitor_metric,
                    'monitor_mode': monitor_mode,
                    'patience_counter': patience_counter,
                }, best_path)
                if ema is not None and ema_checkpoint_backup is not None:
                    ema.restore(ema_checkpoint_backup)
            else:
                patience_counter += 1
                if patience_counter >= patience and epoch >= min_epoch:
                    logging.info(f"Early stopping triggered at epoch {epoch} (No improvement for {patience} epochs).")
                    break
            
        # Optional: Save intermediate checkpoint
        if USE_CHECKPOINTS and ckpt_period > 0 and epoch % ckpt_period == 0:
            ckpt_path = os.path.join(run_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'node_agg_state_dict': node_agg.state_dict(),
                'edge_agg_state_dict': edge_agg.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler is not None else None,
                'ema_state_dict': ema.state_dict() if ema is not None else None,
                'used_ema_weights': False,
                'global_step': global_step,
                'loss': loss.item(),
                'best_metric': best_metric,
                'monitor_metric': monitor_metric,
                'monitor_mode': monitor_mode,
                'patience_counter': patience_counter,
            }, ckpt_path)
            logging.info(f"Saved intermediate checkpoint: {ckpt_path}")

        # Always update last checkpoint for reliable resume
        last_path = os.path.join(run_dir, 'last_checkpoint.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'node_agg_state_dict': node_agg.state_dict(),
            'edge_agg_state_dict': edge_agg.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler is not None else None,
            'ema_state_dict': ema.state_dict() if ema is not None else None,
            'used_ema_weights': False,
            'global_step': global_step,
            'loss': loss.item(),
            'best_metric': best_metric,
            'monitor_metric': monitor_metric,
            'monitor_mode': monitor_mode,
            'patience_counter': patience_counter,
        }, last_path)

    if run_threshold_search or run_test:
        # Evaluate best checkpoint on validation (and optionally test)
        best_path = os.path.join(run_dir, 'best_checkpoint.pth')
        if os.path.exists(best_path):
            logging.info(f"Loading best checkpoint for post-training evaluation: {best_path}")
            best_ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(best_ckpt['model_state_dict'])
            node_agg.load_state_dict(best_ckpt['node_agg_state_dict'])
            edge_agg.load_state_dict(best_ckpt['edge_agg_state_dict'])

        model.eval()
        node_agg.eval()
        edge_agg.eval()
        with torch.no_grad():
            # Threshold search on validation set (optional)
            val_super_graph = _summarize_normalized(val_summarizer, supergraph_normalizer)
            val_out = model(val_super_graph.x, val_super_graph.edge_index, val_super_graph.edge_attr)
            val_target = torch.as_tensor(val_super_graph.y, device=device).float()

            if run_threshold_search:
                best_threshold = thresholds[0] if len(thresholds) > 0 else 0.5
                best_thr_value = -float('inf')

                for thr in thresholds:
                    val_cls_metrics = _binary_metrics_by_class(
                        val_out[val_mask],
                        val_target[val_mask],
                        thr,
                        illicit_label=illicit_label,
                    )
                    flat_val_metrics = _flatten_metrics('val', val_cls_metrics)
                    metric_value = _get_metric_value(flat_val_metrics, threshold_search_metric)
                    if metric_value > best_thr_value:
                        best_thr_value = metric_value
                        best_threshold = thr

                logging.info(
                    f"Best threshold on validation ({threshold_search_metric}) = {best_threshold:.4f} "
                    f"(score={best_thr_value:.4f})"
                )
            else:
                best_threshold = float(training_threshold)

            if run_test:
                super_graph = _summarize_normalized(test_summarizer, supergraph_normalizer)
                out = model(super_graph.x, super_graph.edge_index, super_graph.edge_attr)
                target = torch.as_tensor(super_graph.y, device=device).float()

                test_loss = criterion(out[test_mask], target[test_mask]).item()
                test_cls_metrics = _binary_metrics_by_class(
                    out[test_mask],
                    target[test_mask],
                    best_threshold,
                    illicit_label=illicit_label,
                )
                flat_test_metrics = _flatten_metrics('test', test_cls_metrics)

        if run_test:
            logging.info(
                f"Test metrics | Loss: {test_loss:.4f}, "
                f"Illicit F1: {flat_test_metrics['test/illicit_f1']:.4f}, "
                f"Illicit PR-AUC: {flat_test_metrics['test/illicit_pr_auc']:.4f}, "
                f"Illicit ROC-AUC: {flat_test_metrics['test/illicit_roc_auc']:.4f}, "
                f"Licit F1: {flat_test_metrics['test/licit_f1']:.4f}"
            )

            test_metrics_path = os.path.join(run_dir, 'test_metrics.json')
            all_test_metrics = {
                'test_loss': test_loss,
                'best_threshold': best_threshold,
                **flat_test_metrics,
            }
            with open(test_metrics_path, 'w') as f:
                json.dump(all_test_metrics, f, indent=2)
            logging.info(f"Saved test metrics to {test_metrics_path}")

            if USE_WANDB:
                run.log({
                    **flat_test_metrics,
                    'test/loss': test_loss,
                    'test/best_threshold': best_threshold,
                })
                run.finish()

    if USE_WANDB and not run_test:
        run.finish()

    # Saving Final Checkpoint
    checkpoint_path = os.path.join(run_dir, 'final_checkpoint.pth')
    torch.save({
        'epoch': last_epoch,
        'model_state_dict': model.state_dict(),
        'node_agg_state_dict': node_agg.state_dict(),
        'edge_agg_state_dict': edge_agg.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler is not None else None,
        'ema_state_dict': ema.state_dict() if ema is not None else None,
        'used_ema_weights': False,
        'global_step': global_step,
        'loss': loss.item(),
        'best_metric': best_metric,
        'monitor_metric': monitor_metric,
        'monitor_mode': monitor_mode,
        'patience_counter': patience_counter,
    }, checkpoint_path)
    logging.info(f"Training complete. Run dir: {run_dir}")
    logging.info(f"Final checkpoint saved to {checkpoint_path}")

    # Count parameters at the end of training
    # uninit_names: list[str] = []
    # for name, p in list(model.named_parameters()) + list(node_agg.named_parameters()) + list(edge_agg.named_parameters()):
    #     if isinstance(p, UninitializedParameter):
    #         uninit_names.append(name)

    # if len(uninit_names) > 0:
    #     logging.warning(f"Uninitialized parameters detected: {uninit_names}")

    num_params = 0
    for p in learnable_params:
        if not p.requires_grad:
            continue
        try:
            num_params += p.numel()
        except ValueError:
            continue

    logging.info(f"Total trainable parameters (model + aggregators): {num_params}")

    print("\n")

    return float(best_metric), num_params

if __name__ == "__main__":
    args = parse_args()
    main(args, preloaded_data=None)
