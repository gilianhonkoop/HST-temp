import logging
import math
import random

import numpy as np
import torch
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader, ImbalancedSampler


def _sample_nodes(nodes: torch.Tensor, count: int, generator: torch.Generator) -> torch.Tensor:
    if count <= 0:
        return nodes.new_empty((0,), dtype=torch.long)
    if nodes.numel() == 0:
        return nodes.new_empty((0,), dtype=torch.long)
    if count <= nodes.numel():
        perm = torch.randperm(nodes.numel(), generator=generator)
        return nodes[perm[:count]]

    extra = count - nodes.numel()
    extra_idx = torch.randint(nodes.numel(), (extra,), generator=generator)
    return torch.cat([nodes, nodes[extra_idx]], dim=0)


def _balanced_train_input_nodes(super_graph, train_mask, cfg, seed_offset=0):
    sampling_cfg = cfg.get('train', {}).get('balanced_sampling', {})
    if not sampling_cfg or not sampling_cfg.get('enabled', False):
        return train_mask

    if not hasattr(super_graph, 'y') or super_graph.y is None:
        logging.warning("Balanced sampling requested, but super_graph.y is missing; using regular train_mask.")
        return train_mask

    illicit_label = int(cfg.get('dataset', {}).get('illicit_label', 0))
    illicit_fraction = float(sampling_cfg.get('illicit_fraction', 0.25))
    if not 0.0 < illicit_fraction < 1.0:
        raise ValueError(f"train.balanced_sampling.illicit_fraction must be in (0, 1), got {illicit_fraction}")

    labels = torch.as_tensor(super_graph.y).view(-1).cpu()
    train_mask = train_mask.cpu().long()
    train_labels = labels[train_mask]

    illicit_nodes = train_mask[train_labels == illicit_label]
    licit_nodes = train_mask[train_labels != illicit_label]
    if illicit_nodes.numel() == 0 or licit_nodes.numel() == 0:
        logging.warning(
            "Balanced sampling requested, but one class is missing in train_mask "
            f"(illicit={illicit_nodes.numel()}, licit={licit_nodes.numel()}); using regular train_mask."
        )
        return train_mask

    epoch_size_cfg = sampling_cfg.get('epoch_size', None)
    if epoch_size_cfg is None or str(epoch_size_cfg).lower() == 'auto':
        # Preserve roughly one pass over the larger class while oversampling the
        # smaller class to the requested fraction.
        epoch_size = max(
            train_mask.numel(),
            math.ceil(illicit_nodes.numel() / illicit_fraction),
            math.ceil(licit_nodes.numel() / (1.0 - illicit_fraction)),
        )
    elif str(epoch_size_cfg).lower() in {'train', 'train_size', 'match_train_size'}:
        epoch_size = train_mask.numel()
    else:
        epoch_size = int(epoch_size_cfg)

    if epoch_size < 2:
        raise ValueError(f"train.balanced_sampling.epoch_size must be at least 2, got {epoch_size}")

    illicit_count = int(round(epoch_size * illicit_fraction))
    illicit_count = min(max(illicit_count, 1), epoch_size - 1)
    licit_count = epoch_size - illicit_count

    seed = int(cfg.get('train', {}).get('seed', 42)) + int(seed_offset)
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed)

    balanced_nodes = torch.cat(
        [
            _sample_nodes(illicit_nodes, illicit_count, generator),
            _sample_nodes(licit_nodes, licit_count, generator),
        ],
        dim=0,
    )
    perm = torch.randperm(balanced_nodes.numel(), generator=generator)
    balanced_nodes = balanced_nodes[perm]

    logging.info(
        "Balanced train sampling enabled | "
        f"illicit_label={illicit_label}, target_illicit_fraction={illicit_fraction:.3f}, "
        f"train_illicit={illicit_nodes.numel()}, train_licit={licit_nodes.numel()}, "
        f"epoch_size={balanced_nodes.numel()}, sampled_illicit={illicit_count}, sampled_licit={licit_count}, "
        f"sample_seed={seed}"
    )
    return balanced_nodes

def create_loaders(super_graph, train_mask, val_mask, test_mask, cfg, device, balanced_seed_offset=0):
    use_mini_batch = cfg['train'].get('mini_batch', False)
    num_workers = cfg['train'].get('num_workers', 0)
    
    if not use_mini_batch:
        # Full batch fallback
        loader_train = [(super_graph, train_mask)]
        loader_val = [(super_graph, val_mask)]
        loader_test = [(super_graph, test_mask)]
        return loader_train, loader_val, loader_test

    batch_size = cfg['train'].get('batch_size', 2048)
    neighbor_sizes = cfg['train'].get('neighbor_sizes', [15, 10])
    persistent_workers = cfg['train'].get('persistent_workers', False)
    pin_memory = cfg['train'].get('pin_memory', False)
    disjoint = cfg['train'].get('disjoint', False)
    seed = int(cfg.get('train', {}).get('seed', 42)) + int(balanced_seed_offset)
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed)

    def _seed_worker(worker_id: int) -> None:
        worker_seed = (seed + worker_id) % (2**32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    loader_kwargs = {
        "num_workers": num_workers,
        "persistent_workers": persistent_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": _seed_worker,
        "generator": generator,
    }
    
    # Safely check if 'time' exists and is actually populated
    time_attr = "time" if hasattr(super_graph, "time") and super_graph.time is not None else None
    train_input_nodes = _balanced_train_input_nodes(
        super_graph,
        train_mask,
        cfg,
        seed_offset=balanced_seed_offset,
    )
    
    loader_train = NeighborLoader(
        super_graph,
        num_neighbors=neighbor_sizes,
        batch_size=batch_size,
        input_nodes=train_input_nodes,
        disjoint=disjoint,
        temporal_strategy='uniform' if time_attr else 'last',
        time_attr=time_attr,
        shuffle=True,
        **loader_kwargs,
    )
    
    loader_val = NeighborLoader(
        super_graph,
        num_neighbors=neighbor_sizes,
        batch_size=batch_size,
        input_nodes=val_mask,
        disjoint=disjoint,
        temporal_strategy='uniform' if time_attr else 'last',
        time_attr=time_attr,
        shuffle=False,
        **loader_kwargs,
    )
    
    loader_test = NeighborLoader(
        super_graph,
        num_neighbors=neighbor_sizes,
        batch_size=batch_size,
        input_nodes=test_mask,
        disjoint=disjoint,
        temporal_strategy='uniform' if time_attr else 'last',
        time_attr=time_attr,
        shuffle=False,
        **loader_kwargs,
    )
    
    return loader_train, loader_val, loader_test
